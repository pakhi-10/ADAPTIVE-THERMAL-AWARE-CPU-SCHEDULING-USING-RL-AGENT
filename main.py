"""
main.py — CSD 204 OS Project
Author: P

Entry point for the Thermal-Aware CPU Scheduling project.

What it does
------------
  1. Parses CLI flags (--quick, --skip-train, --seed, --tasks).
  2. Optionally trains the DQN agent (or loads a saved model).
  3. Runs all four schedulers on an identical task set:
       • Round Robin  (thermal-blind baseline)
       • SJF          (thermal-blind baseline)
       • EFS-Inspired (energy-fair, Qiao et al.)
       • DQN Agent    (novel thermal-inertia RL approach)
  4. Prints a formatted comparison table.
  5. Declares a winner on each metric.

Usage
-----
    python main.py                   # full training + all schedulers
    python main.py --quick           # 10k-step training (fast smoke test)
    python main.py --skip-train      # load existing dqn_thermal.zip, skip training
    python main.py --tasks 30        # use 30 tasks instead of default 20
    python main.py --seed 7          # change random seed for task generation
    python main.py --quick --tasks 10 --seed 99   # combine flags freely
"""

import argparse
import copy
import os
import sys

# ── project modules ───────────────────────────────────────────────────────────
from simulator  import generate_tasks
from schedulers import run_round_robin, run_sjf, run_efs
from rl_agent   import train, evaluate, MODEL_PATH, FULL_TIMESTEPS, QUICK_TIMESTEPS


# ─────────────────────────────────────────────────────────────────────────────
#  CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thermal-Aware CPU Scheduling — CSD 204 OS Project"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 10k training steps instead of 100k (fast smoke test).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training entirely and load an existing dqn_thermal.zip.",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=20,
        metavar="N",
        help="Number of tasks to schedule (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="S",
        help="Random seed for task generation (default: 42).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  Metric normalisation
#  Baseline schedulers and rl_agent.evaluate() use slightly different key names.
#  This function returns a unified dict so the comparison table is simple.
# ─────────────────────────────────────────────────────────────────────────────

def normalise_metrics(raw: dict) -> dict:
    """
    Returns a unified metrics dict with four keys:
        avg_temp  | peak_temp  | throttles  | ticks
    Works for both baseline-scheduler output and rl_agent.evaluate() output.
    """
    return {
        "avg_temp"  : raw.get("avg_temperature",        raw.get("avg_temp",   float("nan"))),
        "peak_temp" : raw.get("peak_temperature",       raw.get("peak_temp",  float("nan"))),
        "throttles" : raw.get("total_throttle_events",  raw.get("throttle_events", 0)),
        "ticks"     : raw.get("tick",                   raw.get("ticks",      0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Print helpers
# ─────────────────────────────────────────────────────────────────────────────

LINE  = "=" * 72
DLINE = "─" * 72

def section(title: str):
    print(f"\n{LINE}")
    print(f"  {title}")
    print(LINE)


def print_comparison_table(results: list[dict]):
    """
    Prints a formatted comparison table and highlights the best value
    for each metric (lower is better for all four metrics).
    """
    section("RESULTS — Scheduler Comparison")

    header = (
        f"  {'Scheduler':<30} {'Avg Temp':>9} {'Peak Temp':>10} "
        f"{'Throttles':>10} {'Ticks':>7}"
    )
    print(header)
    print(f"  {DLINE}")

    # find best (minimum) value per metric for highlighting
    metrics_keys = ["avg_temp", "peak_temp", "throttles", "ticks"]
    best = {k: min(r["metrics"][k] for r in results) for k in metrics_keys}

    for r in results:
        m    = r["metrics"]
        name = r["name"]

        def fmt(key: str, fmt_str: str) -> str:
            val  = m[key]
            mark = " ★" if val == best[key] else "  "
            return format(val, fmt_str) + mark

        print(
            f"  {name:<30} "
            f"{fmt('avg_temp',  '.2f'):>11} "
            f"{fmt('peak_temp', '.2f'):>12} "
            f"{fmt('throttles', 'd'):>12} "
            f"{fmt('ticks',     'd'):>9}"
        )

    print(f"  {DLINE}")
    print("  ★ = best value for that metric (lower is better for all)\n")


def print_winner_summary(results: list[dict]):
    """Prints one-line winner per metric."""
    section("WINNER PER METRIC")
    metrics_keys = ["avg_temp", "peak_temp", "throttles", "ticks"]
    labels = {
        "avg_temp"  : "Lowest average temperature",
        "peak_temp" : "Lowest peak temperature   ",
        "throttles" : "Fewest throttle events    ",
        "ticks"     : "Completed in fewest ticks ",
    }
    for key in metrics_keys:
        winner  = min(results, key=lambda r: r["metrics"][key])
        val     = winner["metrics"][key]
        display = f"{val:.2f}" if isinstance(val, float) else str(val)
        print(f"  {labels[key]} → {winner['name']}  ({display})")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n{'=' * 72}")
    print(f"  CSD 204 — Thermal-Aware CPU Scheduling  |  OS Project")
    print(f"{'=' * 72}")
    print(f"  Tasks  : {args.tasks}")
    print(f"  Seed   : {args.seed}")
    print(f"  Mode   : {'quick (10k steps)' if args.quick else 'full (100k steps)'}")
    print(f"  Train  : {'NO — loading saved model' if args.skip_train else 'YES'}")

    # ── Step 1 : Generate tasks (one set, shared across ALL schedulers) ───────
    section("STEP 1 — Generating Task Set")
    original_tasks = generate_tasks(num_tasks=args.tasks, seed=args.seed)
    print(f"  Generated {len(original_tasks)} tasks  (seed={args.seed})")
    print(f"  Sample: {original_tasks[0]}  …  {original_tasks[-1]}")

    # ── Step 2 : Train / load DQN agent ──────────────────────────────────────
    section("STEP 2 — DQN Agent")

    if args.skip_train:
        model_file = MODEL_PATH + ".zip"
        if not os.path.exists(model_file):
            print(f"  ERROR: --skip-train was set but '{model_file}' was not found.")
            print(f"         Run without --skip-train first to train and save the model.")
            sys.exit(1)
        print(f"  Skipping training — using saved model: {model_file}")
    else:
        timesteps = QUICK_TIMESTEPS if args.quick else FULL_TIMESTEPS
        print(f"  Training DQN for {timesteps:,} steps …")
        train(timesteps=timesteps)

    # ── Step 3 : Run baseline schedulers ──────────────────────────────────────
    section("STEP 3 — Running Baseline Schedulers")

    # deep-copy for each scheduler — identical starting conditions
    tasks_rr  = copy.deepcopy(original_tasks)
    tasks_sjf = copy.deepcopy(original_tasks)
    tasks_efs = copy.deepcopy(original_tasks)

    print("  Running Round Robin …", end="  ", flush=True)
    rr_raw  = run_round_robin(tasks_rr)
    print("done ✓")

    print("  Running SJF         …", end="  ", flush=True)
    sjf_raw = run_sjf(tasks_sjf)
    print("done ✓")

    print("  Running EFS         …", end="  ", flush=True)
    efs_raw = run_efs(tasks_efs)
    print("done ✓")

    # ── Step 4 : Evaluate DQN agent ───────────────────────────────────────────
    section("STEP 4 — Evaluating DQN Agent")
    print("  Running one deterministic episode with trained model …", end="  ", flush=True)
    dqn_raw = evaluate(model_path=MODEL_PATH, seed=args.seed)
    print("done ✓")

    # ── Step 5 : Build results list and print table ───────────────────────────
    results = [
        {"name": "Round Robin",              "metrics": normalise_metrics(rr_raw)},
        {"name": "SJF",                      "metrics": normalise_metrics(sjf_raw)},
        {"name": "EFS-Inspired (Qiao et al.)","metrics": normalise_metrics(efs_raw)},
        {"name": "DQN — Thermal Inertia (ours)", "metrics": normalise_metrics(dqn_raw)},
    ]

    print_comparison_table(results)
    print_winner_summary(results)

    # ── Optional: per-core detail for baseline schedulers ─────────────────────
    section("PER-CORE DETAIL  (baseline schedulers)")
    for raw, label in [(rr_raw, "Round Robin"), (sjf_raw, "SJF"), (efs_raw, "EFS-Inspired")]:
        print(f"\n  {label}")
        if "per_core_throttle" in raw:
            print(f"    Throttle counts : {raw['per_core_throttle']}")
        if "per_core_energy" in raw:
            energy = [f"{e:.1f}" for e in raw["per_core_energy"]]
            print(f"    Energy proxy    : {energy}")

    print(f"\n{'=' * 72}")
    print("  All experiments complete ✓")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
