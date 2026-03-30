"""
gym_env.py — CSD 204 OS Project
Author: P

Wraps CPUSimulator as a Gymnasium environment so Stable Baselines3 can train a DQN
agent on it.

Design decisions
----------------
* State  : CPUSimulator.get_state() — 16 floats (4 cores × 4 values each).
           Layout per core: [tgd_corrected_temp, normalised_temp, is_idle, energy_norm]
           We normalise into [-1, 1] for stable neural network training.
* Action : integer 0–3 → which core to assign the next pending task to.
           If the chosen core is busy or queue is empty, step is a no-op.
* Reward : Thermal-Inertia formulation (novel contribution).
           Penalises dT/dt (rate of temperature rise) rather than threshold crossing.
           PROACTIVE: agent learns to spread load before cores get hot.
* Done   : episode ends when task queue is empty AND all cores are idle.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── single source of truth ──────────────────────────────────────────────────
from simulator import CPUSimulator, generate_tasks

# ── constants ────────────────────────────────────────────────────────────────
NUM_CORES     = 4
STATE_DIM     = 16      # 4 cores × 4 values — must match get_state() length
AMBIENT_TEMP  = 40.0
THROTTLE_TEMP = 95.0


class ThermalCPUEnv(gym.Env):
    """
    Gymnasium environment wrapping CPUSimulator.

    Observation space : Box(16,) — normalised floats in [-1, 1].
    Action space      : Discrete(4) — target core index (0–3).
    """

    metadata = {"render_modes": ["human"]}

    # ── Reward hyper-parameters (Dong et al. weighted-reward philosophy) ─────
    W_COMPLETION = 2.0   # reward per task completed this tick
    W_DTDT       = 0.05  # penalty per (°C/tick) of rising temperature
    W_THROTTLE   = 5.0   # penalty per new throttle event

    def __init__(self, seed: int = 42, render_mode=None):
        super().__init__()
        self.seed_val    = seed
        self.render_mode = render_mode

        self.action_space      = spaces.Discrete(NUM_CORES)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )

        self._sim: CPUSimulator | None = None
        self._task_queue: list         = []
        self._prev_temps               = np.zeros(NUM_CORES, dtype=np.float32)

    # ────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # CPUSimulator.__init__(num_cores) — tasks are fed in via assign_task()
        self._sim        = CPUSimulator(num_cores=NUM_CORES)
        self._task_queue = generate_tasks(seed=self.seed_val)
        self._prev_temps = np.array(
            [c.temperature for c in self._sim.cores], dtype=np.float32
        )

        return self._get_obs(), {}

    def step(self, action: int):
        assert self._sim is not None, "Call reset() before step()."

        # ── 1. Assign next queued task to chosen core (if idle + queue exists) 
        assigned = False
        if self._sim.cores[action].is_idle() and self._task_queue:
            task = self._task_queue.pop(0)
            self._sim.assign_task(core_id=action, task=task)
            assigned = True

        # ── 2. Snapshot counts before tick ───────────────────────────────────
        tasks_before    = sum(1 for c in self._sim.cores if c.current_task is not None)
        throttle_before = self._sim.total_throttle_events

        # ── 3. Tick — simulator.tick() returns None; we derive metrics ────────
        self._sim.tick()

        tasks_after     = sum(1 for c in self._sim.cores if c.current_task is not None)
        completions     = max(tasks_before - tasks_after, 0)
        throttle_events = self._sim.total_throttle_events - throttle_before

        # ── 4. Thermal-Inertia reward ─────────────────────────────────────────
        reward = self._compute_reward(completions, throttle_events)

        # ── 5. Terminal: queue empty AND all cores idle ───────────────────────
        terminated = (
            len(self._task_queue) == 0
            and all(c.is_idle() for c in self._sim.cores)
        )

        obs  = self._get_obs()
        info = {
            "assigned"       : assigned,
            "completions"    : completions,
            "throttle_events": throttle_events,
            "queue_remaining": len(self._task_queue),
            "core_temps"     : [c.temperature for c in self._sim.cores],
        }

        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human" and self._sim is not None:
            temps = [f"{c.temperature:.1f}°C" for c in self._sim.cores]
            print(f"[ThermalCPUEnv] tick={self._sim.current_tick}  "
                  f"queue={len(self._task_queue)}  temps={temps}")

    def close(self):
        self._sim = None

    # ────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        raw = np.array(self._sim.get_state(), dtype=np.float32)
        return self._normalise_state(raw)

    def _normalise_state(self, raw: np.ndarray) -> np.ndarray:
        """
        get_state() layout (per core, 4 values each, 16 total):
          [4i+0] tgd_corrected_temp  — already normalised to [0,1] by simulator
          [4i+1] normalised_temp     — already normalised to [0,1] by simulator
          [4i+2] is_idle             — 0.0 or 1.0
          [4i+3] cumulative_energy   — normalised to [0,1] by simulator (capped at 500)

        All values already in [0,1] from simulator.py — just map to [-1, 1].
        """
        return np.clip(raw * 2.0 - 1.0, -1.0, 1.0)

    def _compute_reward(self, completions: int, throttle_events: int) -> float:
        """
        Thermal-Inertia reward — novel contribution.

        r = W_COMPLETION × completions
          − W_DTDT       × Σ max(dT/dt, 0) per core
          − W_THROTTLE   × throttle_events

        Penalises RISING temperature, not absolute temperature.
        Agent learns proactive load spreading, not reactive scrambling.
        """
        curr_temps      = np.array([c.temperature for c in self._sim.cores], dtype=np.float32)
        heating_penalty = float(np.sum(np.maximum(curr_temps - self._prev_temps, 0.0)))
        self._prev_temps = curr_temps.copy()

        return (
              self.W_COMPLETION * completions
            - self.W_DTDT       * heating_penalty
            - self.W_THROTTLE   * throttle_events
        )


# ────────────────────────────────────────────────────────────────────────────
# Smoke test — python gym_env.py
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    print("=== ThermalCPUEnv smoke test ===")
    env = ThermalCPUEnv(seed=42, render_mode="human")

    print("Running Gymnasium env_checker …")
    check_env(env)
    print("check_env passed ✓\n")

    obs, _ = env.reset()
    print(f"Obs shape : {obs.shape}")
    print(f"Obs sample: {obs}\n")

    total_reward, done, steps = 0.0, False, 0
    while not done:
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        total_reward += rew
        done          = term or trunc
        steps        += 1
        if steps % 50 == 0:
            env.render()

    print(f"\nEpisode finished in {steps} steps")
    print(f"Total reward     : {total_reward:.2f}")
    print(f"Final core temps : {info['core_temps']}")
    env.close()
    print("Done ✓")