"""
schedulers.py
-------------
Baseline scheduling algorithms for thermal-aware CPU scheduling project.
CSD 204 — OS Project

All schedulers use CPUSimulator from simulator.py as the shared environment.
This ensures fair comparison — same hardware model, different scheduling logic.

Schedulers implemented:
  1. Round Robin (RR)        — thermal-blind, cycles through cores
  2. Shortest Job First (SJF)— thermal-blind, picks shortest task
  3. EFS-inspired            — energy-aware, picks core with lowest
                               cumulative thermal energy (concept from
                               Qiao et al., simplified for simulation)

Note: The RL agent (your novel approach) lives in rl_agent.py, not here.
"""

from simulator import CPUSimulator, generate_tasks, Task


# ─────────────────────────────────────────────
#  1. Round Robin Scheduler
# ─────────────────────────────────────────────

def run_round_robin(tasks: list, n_cores: int = 4) -> dict:
    """
    Round Robin scheduler — assigns tasks to cores in a fixed cycle.
    Completely ignores temperature. Thermal-blind baseline.

    Logic
    -----
    - Maintain a pointer that cycles 0 → 1 → 2 → 3 → 0 → ...
    - Each tick: if pointer core is idle AND tasks remain, assign next task
    - Advance pointer regardless of whether assignment happened

    Parameters
    ----------
    tasks   : list of Task objects (from generate_tasks())
    n_cores : number of CPU cores (default 4)

    Returns
    -------
    dict of metrics (avg temp, peak temp, throttle events, ticks taken)
    """
    sim     = CPUSimulator(num_cores=n_cores)
    pool    = list(tasks)          # copy so original list is not modified
    pointer = 0                    # RR pointer — cycles through core IDs

    while pool or any(not c.is_idle() for c in sim.cores):
        # try to assign a task to the current pointer core
        if sim.cores[pointer].is_idle() and pool:
            task = pool.pop(0)
            sim.assign_task(pointer, task)

        # always advance pointer (this is what makes it Round Robin)
        pointer = (pointer + 1) % n_cores

        # tick the simulator — all cores heat or cool
        sim.tick()

    metrics = sim.get_metrics()
    metrics["scheduler"] = "Round Robin"
    return metrics


# ─────────────────────────────────────────────
#  2. Shortest Job First (SJF)
# ─────────────────────────────────────────────

def run_sjf(tasks: list, n_cores: int = 4) -> dict:
    """
    Shortest Job First scheduler — always picks the task with the
    smallest burst_time from the queue. Thermal-blind baseline.

    Logic
    -----
    - Each tick: sort remaining tasks by burst_time (ascending)
    - For every idle core, assign the shortest available task
    - Does not consider temperature at all

    Parameters
    ----------
    tasks   : list of Task objects (from generate_tasks())
    n_cores : number of CPU cores (default 4)

    Returns
    -------
    dict of metrics
    """
    sim  = CPUSimulator(num_cores=n_cores)
    pool = list(tasks)

    while pool or any(not c.is_idle() for c in sim.cores):
        # sort pool by burst_time — shortest first
        pool.sort(key=lambda t: t.burst_time)

        # assign to all idle cores
        idle_cores = sim.get_idle_cores()
        for core_id in idle_cores:
            if pool:
                task = pool.pop(0)     # shortest task
                sim.assign_task(core_id, task)

        sim.tick()

    metrics = sim.get_metrics()
    metrics["scheduler"] = "SJF"
    return metrics


# ─────────────────────────────────────────────
#  3. EFS-Inspired Scheduler (Qiao et al.)
# ─────────────────────────────────────────────

def run_efs(tasks: list, n_cores: int = 4) -> dict:
    """
    Energy-Fair Scheduling inspired by Qiao et al. (Wattmeter paper).

    Concept from paper
    ------------------
    Qiao et al. argue that CFS (Linux default) equalises CPU TIME
    across processes but ignores energy consumption. Their EFS
    equalises ENERGY instead — the process with the lowest energy
    share gets scheduled next.

    Our adaptation for simulation
    ------------------------------
    We cannot use RAPL hardware counters (requires real Linux kernel).
    Instead, we use cumulative_energy = sum of (thermal_load × ticks_run)
    as an energy proxy per core. The core with the lowest cumulative
    energy gets the next task — mimicking the energy-fair idea.

    This is NOT thermal-aware (does not look at current temperature).
    It is energy-fair — it tries to balance total work done across cores.

    Parameters
    ----------
    tasks   : list of Task objects (from generate_tasks())
    n_cores : number of CPU cores (default 4)

    Returns
    -------
    dict of metrics
    """
    sim  = CPUSimulator(num_cores=n_cores)
    pool = list(tasks)

    while pool or any(not c.is_idle() for c in sim.cores):
        idle_cores = sim.get_idle_cores()

        if idle_cores and pool:
            # sort idle cores by cumulative_energy — lowest energy core first
            # this is the EFS concept: give work to the core that has done least
            idle_cores_sorted = sorted(
                idle_cores,
                key=lambda cid: sim.cores[cid].cumulative_energy
            )
            for core_id in idle_cores_sorted:
                if pool:
                    task = pool.pop(0)
                    sim.assign_task(core_id, task)

        sim.tick()

    metrics = sim.get_metrics()
    metrics["scheduler"] = "EFS-Inspired (Qiao et al.)"
    return metrics


# ─────────────────────────────────────────────
#  Run all three and print comparison table
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import copy

    # generate ONE task set — same seed for all schedulers (fair comparison)
    original_tasks = generate_tasks(num_tasks=20, seed=42)

    # deep copy for each scheduler so they all start with identical tasks
    tasks_rr  = copy.deepcopy(original_tasks)
    tasks_sjf = copy.deepcopy(original_tasks)
    tasks_efs = copy.deepcopy(original_tasks)

    # run all three
    results = [
        run_round_robin(tasks_rr),
        run_sjf(tasks_sjf),
        run_efs(tasks_efs),
    ]

    # print comparison table
    print("\n" + "=" * 65)
    print(f"{'Scheduler':<30} {'Avg Temp':>9} {'Peak Temp':>10} {'Throttles':>10} {'Ticks':>7}")
    print("=" * 65)
    for r in results:
        print(
            f"{r['scheduler']:<30} "
            f"{r['avg_temperature']:>9.2f} "
            f"{r['peak_temperature']:>10.2f} "
            f"{r['total_throttle_events']:>10} "
            f"{r['tick']:>7}"
        )
    print("=" * 65)
    print("\nNote: RL agent results will be added here after training.")
