"""
simulator.py
------------
Multi-core CPU Thermal Simulator for Thermal-Aware Scheduling Project
CSD 204 — OS Project

This module simulates the thermal behaviour of a multi-core CPU.
Each core tracks its temperature, workload, and throttling state.
The simulator advances time tick by tick, applying heating and cooling
equations at each step.

Thermal model is inspired by:
  Dong et al., "Research on Multi-Core Thermal-Aware Task Scheduling
  Method Based on Reinforcement Learning" (MPTO / TGD model)
"""

import random


# ─────────────────────────────────────────────
#  Constants  (edit these to tune the simulation)
# ─────────────────────────────────────────────

NUM_CORES        = 4       # number of CPU cores
AMBIENT_TEMP     = 40.0    # °C — idle baseline / room temperature
THROTTLE_TEMP    = 95.0    # °C — core throttles above this
MAX_TEMP         = 100.0   # °C — used for TGD normalisation (T_max in paper)
COOLING_RATE     = 2.0     # °C lost per tick when a core is idle
TGD_RHO          = 0.5     # gradient decay factor ρ (Dong et al. Eq. 3)


# ─────────────────────────────────────────────
#  Task  — a simple data container
# ─────────────────────────────────────────────

class Task:
    """
    Represents a single unit of work to be scheduled onto a core.

    Attributes
    ----------
    task_id     : int   — unique identifier
    burst_time  : int   — number of ticks needed to complete
    thermal_load: float — °C added to core temperature per tick while running
    ticks_done  : int   — how many ticks have been executed so far (runtime state)
    """

    def __init__(self, task_id: int, burst_time: int, thermal_load: float):
        self.task_id      = task_id
        self.burst_time   = burst_time
        self.thermal_load = thermal_load
        self.ticks_done   = 0          # updated by the simulator each tick

    def is_complete(self) -> bool:
        """Returns True when the task has finished executing."""
        return self.ticks_done >= self.burst_time

    def __repr__(self):
        return (f"Task(id={self.task_id}, burst={self.burst_time}, "
                f"load={self.thermal_load:.1f}°C/tick, done={self.ticks_done})")


# ─────────────────────────────────────────────
#  Core  — models one physical CPU core
# ─────────────────────────────────────────────

class Core:
    """
    Models one CPU core with thermal dynamics.

    Thermal model (simplified TGD from Dong et al.):
      - When busy  : temp += task.thermal_load  each tick
      - When idle  : temp -= COOLING_RATE        each tick  (floored at AMBIENT_TEMP)
      - TGD correction is applied to give a gradient-aware thermal state
        used by the RL agent as its state input.

    Attributes
    ----------
    core_id          : int
    temperature      : float   — current raw temperature (°C)
    prev_temperature : float   — temperature one tick ago (for gradient calc)
    prev_prev_temp   : float   — temperature two ticks ago (for TGD k factor)
    current_task     : Task | None
    is_throttling    : bool    — True when temp > THROTTLE_TEMP
    throttle_count   : int     — total ticks spent throttling (metric)
    cumulative_energy: float   — thermal_load × ticks_run (proxy for energy, Qiao et al.)
    """

    def __init__(self, core_id: int):
        self.core_id           = core_id
        self.temperature       = AMBIENT_TEMP
        self.prev_temperature  = AMBIENT_TEMP
        self.prev_prev_temp    = AMBIENT_TEMP
        self.current_task      = None
        self.is_throttling     = False
        self.throttle_count    = 0
        self.cumulative_energy = 0.0   # used by the EFS-inspired scheduler

    # ── thermal state ──────────────────────────────────────

    def tgd_corrected_temp(self) -> float:
        """
        Returns the TGD-corrected temperature (T_corrected from Dong et al. Eq. 2).

        T_corrected = T(t) + k × [T(t) − T(t−Δt)]
        k           = ρ × D0 / D1    (Eq. 3)
        D0          = T(t)   − T(t−Δt)
        D1          = T(t−1) − T(t−2Δt)

        If D1 is zero (flat history) k defaults to 0.
        Normalised to [0, 1] by dividing by MAX_TEMP.
        """
        D0 = self.temperature      - self.prev_temperature
        D1 = self.prev_temperature - self.prev_prev_temp

        if D1 == 0:
            k = 0.0
        else:
            k = TGD_RHO * (D0 / D1)

        T_corrected = self.temperature + k * D0
        return T_corrected / MAX_TEMP   # normalised, as in Dong et al. Eq. 1

    def normalised_temp(self) -> float:
        """Raw temperature normalised to [0, 1]. Used as part of RL state."""
        return self.temperature / MAX_TEMP

    # ── tick logic ─────────────────────────────────────────

    def tick(self):
        """
        Advance this core by one time unit.

        - If a task is assigned : heat the core, advance task progress.
        - If idle               : cool the core toward AMBIENT_TEMP.
        - Check for throttling  : if temp > THROTTLE_TEMP, record it.
        - If throttling         : halve effective thermal_load (simulates
                                  clock speed reduction, as described in
                                  the preliminary report).
        """
        # shift temperature history for TGD gradient calculation
        self.prev_prev_temp = self.prev_temperature
        self.prev_temperature = self.temperature

        if self.current_task is not None:
            effective_load = self.current_task.thermal_load

            # throttling: clock drops ~50% → thermal load halved
            if self.temperature >= THROTTLE_TEMP:
                effective_load *= 0.5
                self.is_throttling = True
                self.throttle_count += 1
            else:
                self.is_throttling = False

            # heat the core
            self.temperature += effective_load

            # update energy proxy (Qiao et al. — thermal_load × execution time)
            self.cumulative_energy += effective_load

            # advance task
            self.current_task.ticks_done += 1

            # release task when complete
            if self.current_task.is_complete():
                self.current_task = None

        else:
            # idle — cool toward ambient
            self.is_throttling = False
            self.temperature = max(
                AMBIENT_TEMP,
                self.temperature - COOLING_RATE
            )

    # ── task assignment ─────────────────────────────────────

    def assign_task(self, task: Task):
        """
        Assign a task to this core.
        Only allowed when the core is idle (no current task).
        """
        if self.current_task is not None:
            raise ValueError(
                f"Core {self.core_id} is busy — cannot assign task {task.task_id}."
            )
        self.current_task = task

    def is_idle(self) -> bool:
        """Returns True when the core has no running task."""
        return self.current_task is None

    def __repr__(self):
        status = f"task={self.current_task.task_id}" if self.current_task else "idle"
        return (f"Core(id={self.core_id}, temp={self.temperature:.1f}°C, "
                f"tgd={self.tgd_corrected_temp():.3f}, {status}, "
                f"throttle={self.throttle_count})")


# ─────────────────────────────────────────────
#  CPUSimulator  — manages all cores together
# ─────────────────────────────────────────────

class CPUSimulator:
    """
    Manages N CPU cores and advances simulation time.

    This is the environment that all schedulers (Round Robin, SJF,
    EFS-inspired, and the RL agent) operate on.

    Usage
    -----
        sim = CPUSimulator()
        sim.assign_task(core_id=0, task=some_task)
        sim.tick()
        state = sim.get_state()
    """

    def __init__(self, num_cores: int = NUM_CORES):
        self.num_cores   = num_cores
        self.cores       = [Core(i) for i in range(num_cores)]
        self.current_tick = 0
        self.total_throttle_events = 0

    # ── simulation control ──────────────────────────────────

    def tick(self):
        """
        Advance the entire system by one time step.
        Every core runs its tick() — heating or cooling accordingly.
        """
        self.current_tick += 1
        for core in self.cores:
            was_throttling = core.is_throttling
            core.tick()
            # count a new throttle event (rising edge only)
            if core.is_throttling and not was_throttling:
                self.total_throttle_events += 1

    # ── task interface (used by all schedulers) ─────────────

    def assign_task(self, core_id: int, task: Task):
        """Public interface for schedulers to assign a task to a core."""
        self.cores[core_id].assign_task(task)

    def get_idle_cores(self) -> list:
        """Returns list of core_ids that are currently idle."""
        return [c.core_id for c in self.cores if c.is_idle()]

    def get_temperatures(self) -> list:
        """Returns list of current temperatures for all cores."""
        return [c.temperature for c in self.cores]

    def get_coolest_idle_core(self) -> int | None:
        """
        Returns the core_id of the coolest idle core.
        Used by thermal-aware schedulers and the RL agent.
        Returns None if all cores are busy.
        """
        idle = [c for c in self.cores if c.is_idle()]
        if not idle:
            return None
        return min(idle, key=lambda c: c.temperature).core_id

    # ── RL state vector ─────────────────────────────────────

    def get_state(self) -> list:
        """
        Returns the RL state vector used by the DQN agent.

        State (per core): [tgd_corrected_temp, normalised_raw_temp,
                           is_idle, cumulative_energy_normalised]

        Total length: num_cores × 4

        Inspired by Dong et al. state modelling (Eq. 18–19):
          S(t) = {s1, s2, ..., sn}
          s_i  = (T_hat_i, P_i)
        """
        state = []
        for c in self.cores:
            state.append(c.tgd_corrected_temp())
            state.append(c.normalised_temp())
            state.append(1.0 if c.is_idle() else 0.0)
            # normalise energy proxy to roughly [0,1]
            state.append(min(c.cumulative_energy / 500.0, 1.0))
        return state

    # ── metrics ─────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """
        Returns a summary of simulation metrics for comparison analysis.

        Metrics match those used in Dong et al. (Table V) and the
        preliminary report: avg temp, peak temp, throttle count.
        """
        temps = self.get_temperatures()
        return {
            "tick"                  : self.current_tick,
            "avg_temperature"       : sum(temps) / len(temps),
            "peak_temperature"      : max(temps),
            "total_throttle_events" : self.total_throttle_events,
            "per_core_throttle"     : [c.throttle_count for c in self.cores],
            "per_core_energy"       : [c.cumulative_energy for c in self.cores],
        }

    def reset(self):
        """
        Reset simulator to initial state.
        Called before each new experiment run.
        """
        self.cores       = [Core(i) for i in range(self.num_cores)]
        self.current_tick = 0
        self.total_throttle_events = 0

    def __repr__(self):
        lines = [f"CPUSimulator | tick={self.current_tick} | "
                 f"throttles={self.total_throttle_events}"]
        for c in self.cores:
            lines.append(f"  {c}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Task Generator  — creates synthetic workloads
# ─────────────────────────────────────────────

def generate_tasks(
    num_tasks     : int   = 20,
    min_burst     : int   = 1,
    max_burst     : int   = 10,
    min_load      : float = 1.0,
    max_load      : float = 8.0,
    seed          : int   = 42
) -> list:
    """
    Generate a list of synthetic tasks for scheduling experiments.

    Parameters
    ----------
    num_tasks  : total number of tasks to generate
    min_burst  : minimum burst time (ticks)
    max_burst  : maximum burst time (ticks)
    min_load   : minimum thermal load per tick (°C)
    max_load   : maximum thermal load per tick (°C)
    seed       : random seed for reproducibility across scheduler comparisons

    Returns
    -------
    list of Task objects
    """
    random.seed(seed)
    tasks = []
    for i in range(num_tasks):
        burst   = random.randint(min_burst, max_burst)
        load    = round(random.uniform(min_load, max_load), 2)
        tasks.append(Task(task_id=i, burst_time=burst, thermal_load=load))
    return tasks


# ─────────────────────────────────────────────
#  Quick smoke test  — run this file directly
#  to verify the simulator works before building
#  schedulers on top of it.
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Simulator Smoke Test ===\n")

    sim   = CPUSimulator(num_cores=4)
    tasks = generate_tasks(num_tasks=8)

    print("Generated tasks:")
    for t in tasks:
        print(f"  {t}")

    print("\nAssigning first 4 tasks to cores 0–3...")
    for i in range(4):
        sim.assign_task(core_id=i, task=tasks[i])

    print("\nRunning 10 ticks:\n")
    for tick in range(10):
        sim.tick()
        temps = sim.get_temperatures()
        print(f"Tick {tick+1:2d} | Temps: {[f'{t:.1f}' for t in temps]} "
              f"| Throttles: {sim.total_throttle_events}")

    print("\nFinal state:")
    print(sim)

    print("\nMetrics:")
    for k, v in sim.get_metrics().items():
        print(f"  {k}: {v}")

    print("\nRL State vector:")
    print(f"  {sim.get_state()}")