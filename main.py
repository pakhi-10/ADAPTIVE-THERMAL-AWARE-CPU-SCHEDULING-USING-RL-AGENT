"""
main.py — CSD 204 OS Project
Authors: P + S

Runs all schedulers on the same task set (seed=42) and produces:
  1. Bar chart  — avg_temp, peak_temp, throttle_events, ticks
                  comparing RR, SJF, EFS, and the balanced DQN agent
  2. Pareto plot — avg_temp vs ticks for all 7 DQN agents + 3 baselines
                   This is our novel contribution (Option 2)

Usage
-----
    python main.py              # uses pre-trained models (must run rl_agent.py first)
    python main.py --retrain    # retrains all DQN agents before plotting
    python main.py --quick      # retrain with fewer steps (for testing)
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── project imports ───────────────────────────────────────────────────────────
from simulator   import generate_tasks
from schedulers  import run_round_robin, run_sjf, run_efs
from rl_agent    import train_all, load_results, MODELS_DIR
from gym_env     import ThermalCPUEnv
from stable_baselines3 import DQN

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Step 1: Run baseline schedulers ──────────────────────────────────────────

def normalise_keys(m: dict) -> dict:
    """
    Map schedulers.py key names to the common keys used throughout main.py.
    schedulers.py uses: avg_temperature, peak_temperature,
                        total_throttle_events, tick
    common keys used:   avg_temp, peak_temp, throttle_events, ticks
    """
    return {
        "avg_temp"        : m.get("avg_temperature",       m.get("avg_temp", 0)),
        "peak_temp"       : m.get("peak_temperature",      m.get("peak_temp", 0)),
        "throttle_events" : m.get("total_throttle_events", m.get("throttle_events", 0)),
        "ticks"           : m.get("tick",                  m.get("ticks", 0)),
    }


def run_baselines() -> dict:
    """
    Run RR, SJF, EFS on generate_tasks(seed=42).
    Returns dict of {scheduler_name: metrics_dict}.
    """
    tasks = generate_tasks(seed=42)

    print("Running baseline schedulers …")
    rr  = normalise_keys(run_round_robin(tasks));  print("  Round Robin  ✓")
    sjf = normalise_keys(run_sjf(tasks));          print("  SJF          ✓")
    efs = normalise_keys(run_efs(tasks));          print("  EFS          ✓")

    return {
        "Round Robin" : rr,
        "SJF"         : sjf,
        "EFS"         : efs,
    }


# ── Step 2: Get DQN Pareto results ───────────────────────────────────────────

def get_dqn_results(retrain: bool = False, quick: bool = False) -> list:
    results_path = os.path.join(MODELS_DIR, "pareto_results.json")

    if retrain or not os.path.exists(results_path):
        from rl_agent import FULL_TIMESTEPS, QUICK_TIMESTEPS
        timesteps = QUICK_TIMESTEPS if quick else FULL_TIMESTEPS
        return train_all(timesteps=timesteps)
    else:
        print("Loading pre-trained DQN results …")
        results = load_results()
        print(f"  Loaded {len(results)} agent results ✓")
        return results


# ── Plot 1: Bar chart comparison ─────────────────────────────────────────────

def plot_bar_comparison(baselines: dict, dqn_results: list):
    """
    Bar chart comparing RR, SJF, EFS, and balanced DQN on 4 metrics.
    Uses the 'balanced' DQN agent (w_dtdt=0.05) as the representative DQN result.
    """
    # find balanced agent
    balanced = next((r for r in dqn_results if r["label"] == "balanced"), dqn_results[0])

    schedulers = list(baselines.keys()) + ["DQN (balanced)"]
    all_metrics = list(baselines.values()) + [balanced]

    metrics_to_plot = [
        ("avg_temp",        "Average Temperature (°C)",  "tomato"),
        ("peak_temp",       "Peak Temperature (°C)",     "orangered"),
        ("throttle_events", "Throttle Events",           "steelblue"),
        ("ticks",           "Ticks to Complete",         "mediumseagreen"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        "Scheduler Comparison — CSD 204 OS Project\n"
        "Adaptive Thermal-Aware CPU Scheduling using Reinforcement Learning",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ax, (key, ylabel, color) in zip(axes, metrics_to_plot):
        values = [m[key] for m in all_metrics]
        bars   = ax.bar(schedulers, values, color=color, alpha=0.85, edgecolor="black", linewidth=0.7)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(range(len(schedulers)))
        ax.set_xticklabels(schedulers, rotation=20, ha="right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        # value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8
            )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "bar_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nBar chart saved → {path}")


# ── Plot 2: Pareto frontier ───────────────────────────────────────────────────

def plot_pareto(baselines: dict, dqn_results: list):
    """
    Pareto frontier plot: avg_temp (X) vs ticks (Y).

    DQN agents form a curve — the Pareto frontier.
    Baseline schedulers appear as single points.
    Points toward the bottom-left are better (cooler AND faster).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # ── plot DQN Pareto frontier ──────────────────────────────────────────────
    dqn_temps = [r["avg_temp"] for r in dqn_results]
    dqn_ticks = [r["ticks"]    for r in dqn_results]
    dqn_labels= [r["label"]    for r in dqn_results]

    # sort by avg_temp so the line connects points in order
    sorted_points = sorted(zip(dqn_temps, dqn_ticks, dqn_labels))
    sx, sy, sl    = zip(*sorted_points)

    ax.plot(sx, sy, "o-", color="royalblue", linewidth=2,
            markersize=8, markerfacecolor="white", markeredgewidth=2,
            label="DQN agents (Pareto frontier)", zorder=3)

    # label each DQN point
    for x, y, lbl in zip(sx, sy, sl):
        ax.annotate(
            lbl, (x, y),
            textcoords="offset points", xytext=(6, 6),
            fontsize=8, color="royalblue"
        )

    # ── shade the Pareto frontier region ─────────────────────────────────────
    ax.fill_between(sx, sy, max(sy) * 1.1,
                    alpha=0.07, color="royalblue", label="_nolegend_")

    # ── plot baseline schedulers ──────────────────────────────────────────────
    colors  = {"Round Robin": "tomato", "SJF": "darkorange", "EFS": "mediumseagreen"}
    markers = {"Round Robin": "s",      "SJF": "^",          "EFS": "D"}

    for name, metrics in baselines.items():
        ax.scatter(
            metrics["avg_temp"], metrics["ticks"],
            s=120, color=colors[name], marker=markers[name],
            zorder=4, label=name, edgecolors="black", linewidth=0.8
        )
        ax.annotate(
            name, (metrics["avg_temp"], metrics["ticks"]),
            textcoords="offset points", xytext=(6, -12),
            fontsize=9, color=colors[name], fontweight="bold"
        )

    # ── labels and formatting ─────────────────────────────────────────────────
    ax.set_xlabel("Average Temperature (°C)", fontsize=12)
    ax.set_ylabel("Ticks to Complete", fontsize=12)
    ax.set_title(
        "Pareto Frontier: Thermal Safety vs Completion Speed\n"
        "Novel Contribution — Multi-Objective DQN Scheduling Analysis",
        fontsize=13, fontweight="bold"
    )

    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)

    # annotation explaining the plot
    ax.text(
        0.98, 0.02,
        "← Bottom-left is better\n(cooler AND faster)",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9,
        color="gray", style="italic"
    )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "pareto_frontier.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Pareto plot saved  → {path}")


# ── Print results table ───────────────────────────────────────────────────────

def print_table(baselines: dict, dqn_results: list):
    print("\n" + "=" * 70)
    print("RESULTS TABLE (Dong et al. Table V format)")
    print("=" * 70)
    print(f"{'Scheduler':<20} {'avg_temp':>10} {'peak_temp':>10} "
          f"{'throttles':>10} {'ticks':>8}")
    print("-" * 70)

    for name, m in baselines.items():
        print(f"{name:<20} {m['avg_temp']:>9.1f}°C {m['peak_temp']:>9.1f}°C "
              f"{m['throttle_events']:>10} {m['ticks']:>8}")

    print("-" * 70)
    for r in dqn_results:
        label = f"DQN ({r['label']})"
        print(f"{label:<20} {r['avg_temp']:>9.1f}°C {r['peak_temp']:>9.1f}°C "
              f"{r['throttle_events']:>10} {r['ticks']:>8}")

    print("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retrain = "--retrain" in sys.argv
    quick   = "--quick"   in sys.argv

    print("=" * 50)
    print("CSD 204 — Thermal-Aware Scheduling Comparison")
    print("=" * 50)

    baselines   = run_baselines()
    dqn_results = get_dqn_results(retrain=retrain, quick=quick)

    print_table(baselines, dqn_results)
    plot_bar_comparison(baselines, dqn_results)
    plot_pareto(baselines, dqn_results)

    print("\nAll plots saved to /plots/")
    print("Done ✓")