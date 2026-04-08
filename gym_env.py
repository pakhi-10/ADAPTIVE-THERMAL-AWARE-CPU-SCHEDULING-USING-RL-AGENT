"""
gym_env.py — CSD 204 OS Project
Author: P

Wraps CPUSimulator as a Gymnasium environment for DQN training.

Option 2 change: W_DTDT and W_COMPLETION are now constructor parameters
instead of hardcoded class variables. This lets rl_agent.py create multiple
environments with different weight combinations for Pareto sweep training.
Everything else is identical to the previous version.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulator import CPUSimulator, generate_tasks

NUM_CORES     = 4
STATE_DIM     = 16
AMBIENT_TEMP  = 40.0
THROTTLE_TEMP = 95.0


class ThermalCPUEnv(gym.Env):
    """
    Gymnasium environment wrapping CPUSimulator.

    Observation space : Box(16,) — normalised floats in [-1, 1].
    Action space      : Discrete(4) — target core index (0–3).

    Parameters
    ----------
    w_dtdt       : penalty weight for rising temperature (dT/dt)
    w_completion : reward weight per task completed
    w_throttle   : penalty per new throttle event
    seed         : task generation seed (must be 42 for fair comparison)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        w_dtdt      : float = 0.05,
        w_completion: float = 2.0,
        w_throttle  : float = 5.0,
        seed        : int   = 42,
        render_mode         = None,
    ):
        super().__init__()
        self.W_DTDT       = w_dtdt
        self.W_COMPLETION = w_completion
        self.W_THROTTLE   = w_throttle
        self.seed_val     = seed
        self.render_mode  = render_mode

        self.action_space      = spaces.Discrete(NUM_CORES)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )

        self._sim: CPUSimulator | None = None
        self._task_queue: list         = []
        self._prev_temps               = np.zeros(NUM_CORES, dtype=np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._sim        = CPUSimulator(num_cores=NUM_CORES)
        self._task_queue = generate_tasks(seed=self.seed_val)
        self._prev_temps = np.array(
            [c.temperature for c in self._sim.cores], dtype=np.float32
        )
        return self._get_obs(), {}

    def step(self, action: int):
        assert self._sim is not None, "Call reset() before step()."

        assigned = False
        if self._sim.cores[action].is_idle() and self._task_queue:
            task = self._task_queue.pop(0)
            self._sim.assign_task(core_id=action, task=task)
            assigned = True

        tasks_before    = sum(1 for c in self._sim.cores if c.current_task is not None)
        throttle_before = self._sim.total_throttle_events

        self._sim.tick()

        tasks_after     = sum(1 for c in self._sim.cores if c.current_task is not None)
        completions     = max(tasks_before - tasks_after, 0)
        throttle_events = self._sim.total_throttle_events - throttle_before

        reward = self._compute_reward(completions, throttle_events)

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
            "ticks"          : self._sim.current_tick,
        }

        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human" and self._sim is not None:
            temps = [f"{c.temperature:.1f}°C" for c in self._sim.cores]
            print(f"[ThermalCPUEnv] tick={self._sim.current_tick}  "
                  f"queue={len(self._task_queue)}  temps={temps}")

    def close(self):
        self._sim = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        raw = np.array(self._sim.get_state(), dtype=np.float32)
        return np.clip(raw * 2.0 - 1.0, -1.0, 1.0)

    def _compute_reward(self, completions: int, throttle_events: int) -> float:
        """
        Thermal-Inertia reward.
        Penalises rising temperature (dT/dt), not absolute temperature.
        Weights are constructor parameters — varied across agents for Pareto sweep.
        """
        curr_temps      = np.array([c.temperature for c in self._sim.cores], dtype=np.float32)
        heating_penalty = float(np.sum(np.maximum(curr_temps - self._prev_temps, 0.0)))
        self._prev_temps = curr_temps.copy()

        return (
              self.W_COMPLETION * completions
            - self.W_DTDT       * heating_penalty
            - self.W_THROTTLE   * throttle_events
        )


# ── Smoke test ────────────────────────────────────────────────────────────────

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