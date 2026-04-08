"""
rl_agent.py — CSD 204 OS Project
Author: P

Trains multiple DQN agents across different reward weight combinations.
Each agent represents a different thermal vs completion trade-off.
Results are used to plot the Pareto frontier in main.py.

Usage
-----
    python rl_agent.py            # full training (~5–10 mins)
    python rl_agent.py --quick    # quick test (confirms no crash, ~1 min)
"""

import sys
import os
import json
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from gym_env import ThermalCPUEnv

# ── output folder for saved models ───────────────────────────────────────────
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Pareto sweep: (w_dtdt, w_completion) pairs ───────────────────────────────
# w_dtdt       = how much to penalise rising temperature
# w_completion = how much to reward finishing tasks
# Low w_dtdt  → agent is aggressive, fast but hot
# High w_dtdt → agent is cautious, cool but slower
WEIGHT_CONFIGS = [
    {"w_dtdt": 0.01, "w_completion": 3.0,  "label": "aggressive"},
    {"w_dtdt": 0.03, "w_completion": 2.5,  "label": "fast"},
    {"w_dtdt": 0.05, "w_completion": 2.0,  "label": "balanced"},
    {"w_dtdt": 0.10, "w_completion": 1.5,  "label": "cautious"},
    {"w_dtdt": 0.20, "w_completion": 1.0,  "label": "cool"},
    {"w_dtdt": 0.40, "w_completion": 0.5,  "label": "very_cool"},
    {"w_dtdt": 0.60, "w_completion": 0.2,  "label": "thermal_safe"},
]

FULL_TIMESTEPS  = 100_000
QUICK_TIMESTEPS = 10_000


# ── Train one agent ───────────────────────────────────────────────────────────

def train_one(w_dtdt: float, w_completion: float, label: str,
              timesteps: int = FULL_TIMESTEPS) -> str:
    """
    Train a single DQN agent with given reward weights.
    Saves model to models/dqn_{label}.zip
    Returns the saved model path.
    """
    print(f"\n  Training '{label}'  (w_dtdt={w_dtdt}, w_completion={w_completion}) …")

    env = ThermalCPUEnv(w_dtdt=w_dtdt, w_completion=w_completion, seed=42)

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
        verbose                = 0,
        tensorboard_log        = None,
        seed                   = 42,
    )

    model.learn(total_timesteps=timesteps, progress_bar=False)

    path = os.path.join(MODELS_DIR, f"dqn_{label}")
    model.save(path)
    print(f"  Saved → {path}.zip ✓")
    env.close()
    return path


# ── Evaluate one saved model ──────────────────────────────────────────────────

def evaluate(model_path: str, label: str, seed: int = 42) -> dict:
    """
    Run one deterministic episode with a saved model.
    Returns metrics dict compatible with schedulers.py output.
    """
    env   = ThermalCPUEnv(seed=seed)
    model = DQN.load(model_path, env=env)

    obs, _   = env.reset()
    done     = False
    temp_log = []
    total_throttle = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        temp_log.append(info["core_temps"])
        total_throttle += info["throttle_events"]
        done = terminated or truncated

    env.close()

    all_temps = [t for tick in temp_log for t in tick]

    return {
        "label"           : label,
        "avg_temp"        : float(np.mean(all_temps)),
        "peak_temp"       : float(np.max(all_temps)),
        "throttle_events" : total_throttle,
        "ticks"           : len(temp_log),
    }


# ── Train all + evaluate all ──────────────────────────────────────────────────

def train_all(timesteps: int = FULL_TIMESTEPS) -> list:
    """
    Train one DQN agent per weight config in WEIGHT_CONFIGS.
    Returns list of metrics dicts for all agents.
    Called by main.py.
    """
    print(f"\n=== Pareto Sweep Training — {len(WEIGHT_CONFIGS)} agents × "
          f"{timesteps:,} steps each ===")

    # sanity check env once before training loop
    check_env(ThermalCPUEnv(seed=42))
    print("check_env passed ✓")

    results = []
    for cfg in WEIGHT_CONFIGS:
        path    = train_one(
            w_dtdt      = cfg["w_dtdt"],
            w_completion= cfg["w_completion"],
            label       = cfg["label"],
            timesteps   = timesteps,
        )
        metrics = evaluate(path, label=cfg["label"])
        metrics["w_dtdt"]       = cfg["w_dtdt"]
        metrics["w_completion"] = cfg["w_completion"]
        results.append(metrics)
        print(f"  → avg_temp={metrics['avg_temp']:.1f}°C  "
              f"ticks={metrics['ticks']}  "
              f"throttles={metrics['throttle_events']}")

    # save results to JSON so main.py can load without retraining
    results_path = os.path.join(MODELS_DIR, "pareto_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved → {results_path}")

    return results


def load_results() -> list:
    """Load previously saved Pareto results (skip retraining)."""
    path = os.path.join(MODELS_DIR, "pareto_results.json")
    with open(path) as f:
        return json.load(f)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    quick     = "--quick" in sys.argv
    timesteps = QUICK_TIMESTEPS if quick else FULL_TIMESTEPS

    results = train_all(timesteps=timesteps)

    print("\n=== Final Pareto Results ===")
    print(f"{'Label':<15} {'w_dtdt':>8} {'w_comp':>8} "
          f"{'avg_temp':>10} {'ticks':>8} {'throttles':>10}")
    print("-" * 65)
    for r in results:
        print(f"{r['label']:<15} {r['w_dtdt']:>8.2f} {r['w_completion']:>8.2f} "
              f"{r['avg_temp']:>9.1f}°C {r['ticks']:>8} {r['throttle_events']:>10}")

    print("\nDone ✓  Run main.py to generate plots.")