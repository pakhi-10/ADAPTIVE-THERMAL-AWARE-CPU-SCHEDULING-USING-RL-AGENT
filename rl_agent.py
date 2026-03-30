"""
rl_agent.py — CSD 204 OS Project
Author: P

Trains a DQN agent on ThermalCPUEnv using Stable Baselines3.
Saves the trained model as dqn_thermal.zip for use in main.py.

Usage
-----
    python rl_agent.py            # full training (~100k steps, ~5 mins)
    python rl_agent.py --quick    # short test run (10k steps, confirms no crash)
"""

import sys
import os
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

from gym_env import ThermalCPUEnv

# ── paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "dqn_thermal"          # SB3 saves as dqn_thermal.zip
LOG_DIR    = "./logs/"

# ── training config ───────────────────────────────────────────────────────────
FULL_TIMESTEPS  = 100_000
QUICK_TIMESTEPS = 10_000


# ────────────────────────────────────────────────────────────────────────────
# Progress callback — prints a line every 10k steps so you can see it running
# ────────────────────────────────────────────────────────────────────────────

class ProgressCallback(BaseCallback):
    def __init__(self, print_every: int = 10_000):
        super().__init__()
        self.print_every = print_every

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_every == 0:
            print(f"  Step {self.num_timesteps:>8,} | "
                  f"episodes={self.model._episode_num}")
        return True   # return False to stop training early


# ────────────────────────────────────────────────────────────────────────────
# Train
# ────────────────────────────────────────────────────────────────────────────

def train(timesteps: int = FULL_TIMESTEPS):
    print(f"\n=== DQN Training — {timesteps:,} steps ===\n")

    env = ThermalCPUEnv(seed=42)

    # sanity check before we hand it to SB3
    print("Checking environment …")
    check_env(env)
    print("check_env passed ✓\n")

    # ── DQN hyperparameters ───────────────────────────────────────────────────
    # Chosen to match a small discrete-action problem:
    # - learning_rate   : 1e-3  (standard starting point)
    # - buffer_size     : 50k   (enough replay diversity without huge RAM)
    # - learning_starts : 1k    (collect some experience before first update)
    # - batch_size      : 64    (stable gradient estimates)
    # - target_update_interval: 500 (frequent enough for short episodes)
    # - exploration_fraction  : 0.3  (explore for first 30% of training)
    # - exploration_final_eps : 0.05 (5% random actions at end)
    # - policy "MlpPolicy" : simple feed-forward network, fits our 16-dim state
    model = DQN(
        policy                 = "MlpPolicy",
        env                    = env,
        learning_rate          = 1e-3,
        buffer_size            = 50_000,
        learning_starts        = 1_000,
        batch_size             = 64,
        gamma                  = 0.99,
        target_update_interval = 500,
        exploration_fraction   = 0.3,
        exploration_final_eps  = 0.05,
        verbose                = 0,          # we use our own callback
        tensorboard_log        = None,
        seed                   = 42,
    )

    print("Training started …")
    model.learn(
        total_timesteps = timesteps,
        callback        = ProgressCallback(print_every=max(timesteps // 10, 1000)),
        progress_bar    = False,
    )

    model.save(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}.zip ✓")
    env.close()
    return model


# ────────────────────────────────────────────────────────────────────────────
# Evaluate — runs one episode with the trained model, returns metrics dict
# This is called by main.py to get DQN results for comparison
# ────────────────────────────────────────────────────────────────────────────

def evaluate(model_path: str = MODEL_PATH, seed: int = 42) -> dict:
    """
    Load a saved DQN model and run one deterministic episode.

    Returns a metrics dict with the same keys as the other schedulers:
        avg_temp, peak_temp, throttle_events, ticks
    So main.py can treat all four schedulers identically.
    """
    env   = ThermalCPUEnv(seed=seed)
    model = DQN.load(model_path, env=env)

    obs, _     = env.reset()
    done       = False
    temp_log   = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        temp_log.append(info["core_temps"])
        done = terminated or truncated

    env.close()

    # flatten all per-tick core temperatures
    all_temps = [t for tick in temp_log for t in tick]

    return {
        "avg_temp"        : float(np.mean(all_temps)),
        "peak_temp"       : float(np.max(all_temps)),
        "throttle_events" : info.get("throttle_events", 0),  # last tick value
        "ticks"           : len(temp_log),
    }


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    timesteps = QUICK_TIMESTEPS if quick else FULL_TIMESTEPS

    model = train(timesteps=timesteps)

    print("\n=== Quick evaluation of trained model ===")
    metrics = evaluate()
    print(f"  avg_temp        : {metrics['avg_temp']:.2f} °C")
    print(f"  peak_temp       : {metrics['peak_temp']:.2f} °C")
    print(f"  throttle_events : {metrics['throttle_events']}")
    print(f"  ticks           : {metrics['ticks']}")
    print("\nAll done ✓")
