# 07_compare_runs.py
# Loads all saved probe runs from data/results/runs/ and generates
# comparison plots so you can see how hyperparameter changes affect performance.
#
# Plots saved to data/results/comparison/:
#   1. layer_accuracy_comparison.png  — all runs on one layer-accuracy chart
#   2. best_accuracy_bar.png          — bar chart of each run's best layer accuracy
#   3. delta_heatmap.png              — heatmap of accuracy differences vs. baseline run
#   4. per_tone_f1_comparison.png     — per-tone F1 for the best layer of each run
#
# Usage: python 07_compare_runs.py
#        python 07_compare_runs.py --baseline "20240101_120000_baseline"  (optional)

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from config import RESULTS_DIR, TONE_NAMES


COMPARISON_DIR = os.path.join(RESULTS_DIR, "comparison")
os.makedirs(COMPARISON_DIR, exist_ok=True)

TONE_KEYS = [TONE_NAMES[i] for i in range(len(TONE_NAMES))]


# ── Load runs ─────────────────────────────────────────────────────────────────

def load_all_runs(runs_dir: str) -> list[dict]:
    """
    Load all probe_results.json files from data/results/runs/ subdirectories.
    Returns a list of run dicts sorted by timestamp (oldest first).
    """
    if not os.path.exists(runs_dir):
        raise FileNotFoundError(
            f"No runs directory found at {runs_dir}. Run 04_probe.py at least once."
        )

    runs = []
    for entry in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, entry, "probe_results.json")
        if not os.path.exists(run_path):
            continue
        with open(run_path) as f:
            data = json.load(f)
        data["_run_dir"]  = entry
        data["_run_name"] = data.get("run_label", entry)
        runs.append(data)

    if not runs:
        raise ValueError("No completed runs found. Run 04_probe.py first.")

    print(f"Loaded {len(runs)} run(s):")
    for r in runs:
        best = max(r["layer_results"], key=lambda x: x["accuracy"])
        print(f"  [{r['_run_dir']}]  best={best['accuracy']:.4f}  "
              f"model={r.get('whisper_model','?')}  "
              f"probe={r['probe_config'].get('probe_type','?')}  "
              f"C={r['probe_config'].get('C','?')}")
    return runs


# ── Plot 1: Layer accuracy comparison ─────────────────────────────────────────

def plot_layer_accuracy(runs: list[dict]):
    """One line per run showing probe accuracy at each encoder layer."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = cm.tab10(np.linspace(0, 1, len(runs)))

    for run, color in zip(runs, colors):
        layers = [r["layer"] for r in run["layer_results"]]
        accs   = [r["accuracy"] for r in run["layer_results"]]
        cfg    = run["probe_config"]
        label  = (f"{run['_run_name']}  "
                  f"({cfg.get('probe_type','?')}, C={cfg.get('C','?')}, "
                  f"seed={cfg.get('random_seed','?')})")
        ax.plot(layers, accs, marker="o", linewidth=2, color=color, label=label)

    # Majority baseline from first run
    majority = runs[0]["majority_baseline"]
    ax.axhline(majority, linestyle=":", color="gray", linewidth=1, label="Majority baseline")

    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Layer-wise Probe Accuracy — All Runs")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = os.path.join(COMPARISON_DIR, "layer_accuracy_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── Plot 2: Best accuracy bar chart ───────────────────────────────────────────

def plot_best_accuracy_bar(runs: list[dict]):
    """Bar chart: each run's best layer accuracy, annotated with which layer."""
    labels     = []
    best_accs  = []
    best_layers= []
    majority   = runs[0]["majority_baseline"]

    for run in runs:
        best = max(run["layer_results"], key=lambda r: r["accuracy"])
        cfg  = run["probe_config"]
        labels.append(
            f"{run['_run_name']}\n"
            f"({cfg.get('probe_type','?')}, C={cfg.get('C','?')})"
        )
        best_accs.append(best["accuracy"])
        best_layers.append(best["layer"])

    x      = np.arange(len(labels))
    colors = cm.tab10(np.linspace(0, 1, len(runs)))

    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 2), 5))
    bars = ax.bar(x, best_accs, color=colors, edgecolor="white", width=0.6)

    # Annotate with best layer number
    for bar, layer, acc in zip(bars, best_layers, best_accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"L{layer}\n{acc:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.axhline(majority, linestyle=":", color="gray", linewidth=1.5, label="Majority baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Best Probe Accuracy")
    ax.set_title("Best Layer Accuracy per Run")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(COMPARISON_DIR, "best_accuracy_bar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── Plot 3: Delta heatmap vs. baseline ────────────────────────────────────────

def plot_delta_heatmap(runs: list[dict], baseline_dir: str | None):
    """
    Heatmap showing accuracy delta of each run vs. the baseline run,
    at every layer. Red = improvement, Blue = regression.
    """
    if len(runs) < 2:
        print("  Skipping delta heatmap — need at least 2 runs.")
        return

    # Pick baseline: user-specified or first run
    baseline = None
    if baseline_dir:
        for r in runs:
            if r["_run_dir"] == baseline_dir:
                baseline = r
                break
    if baseline is None:
        baseline = runs[0]
        print(f"  Using '{baseline['_run_dir']}' as baseline.")

    baseline_accs = {r["layer"]: r["accuracy"] for r in baseline["layer_results"]}
    n_layers = len(baseline["layer_results"])

    other_runs = [r for r in runs if r["_run_dir"] != baseline["_run_dir"]]
    if not other_runs:
        print("  Skipping delta heatmap — all runs are the baseline.")
        return

    # Build delta matrix: rows = other runs, cols = layers
    delta_matrix = []
    row_labels   = []
    for run in other_runs:
        row = []
        for layer_idx in range(n_layers):
            run_acc  = next((r["accuracy"] for r in run["layer_results"]
                             if r["layer"] == layer_idx), 0.0)
            baseline_acc = baseline_accs.get(layer_idx, 0.0)
            row.append(round(run_acc - baseline_acc, 4))
        delta_matrix.append(row)
        cfg = run["probe_config"]
        row_labels.append(
            f"{run['_run_name']} ({cfg.get('probe_type','?')}, C={cfg.get('C','?')})"
        )

    delta_matrix = np.array(delta_matrix)
    vmax = max(abs(delta_matrix.max()), abs(delta_matrix.min()), 0.01)

    fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.6), max(3, len(other_runs) * 0.8)))
    sns.heatmap(
        delta_matrix,
        annot=True, fmt=".3f",
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        xticklabels=[f"L{i}" for i in range(n_layers)],
        yticklabels=row_labels,
        ax=ax, linewidths=0.3
    )
    ax.set_xlabel("Encoder Layer")
    ax.set_title(f"Accuracy Delta vs. Baseline ({baseline['_run_name']})\n"
                 f"Red = better than baseline, Blue = worse")
    fig.tight_layout()

    out = os.path.join(COMPARISON_DIR, "delta_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── Plot 4: Per-tone F1 comparison ────────────────────────────────────────────

def plot_per_tone_f1(runs: list[dict]):
    """
    Grouped bar chart comparing per-tone F1 scores at each run's best layer.
    Each group of bars = one tone; each bar = one run.
    """
    n_runs  = len(runs)
    n_tones = len(TONE_KEYS)
    x       = np.arange(n_tones)
    width   = 0.8 / n_runs
    colors  = cm.tab10(np.linspace(0, 1, n_runs))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (run, color) in enumerate(zip(runs, colors)):
        best = max(run["layer_results"], key=lambda r: r["accuracy"])
        f1_scores = []
        for tone in TONE_KEYS:
            pc = best.get("per_class", {})
            f1 = pc.get(tone, {}).get("f1-score", 0.0)
            f1_scores.append(f1)

        cfg   = run["probe_config"]
        label = (f"{run['_run_name']} "
                 f"({cfg.get('probe_type','?')}, C={cfg.get('C','?')}) "
                 f"[L{best['layer']}]")
        offset = (i - n_runs / 2 + 0.5) * width
        ax.bar(x + offset, f1_scores, width=width, color=color,
               label=label, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(TONE_KEYS, fontsize=10)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-tone F1 at Best Layer — All Runs")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(COMPARISON_DIR, "per_tone_f1_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(runs: list[dict]):
    print("\n-- Summary Table ----------------------------------------")
    print(f"  {'Run':<30} {'BestAcc':>8} {'BestL':>6} {'Probe':<8} {'C':>6} {'Seed':>6}")
    print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*8} {'-'*6} {'-'*6}")
    for run in runs:
        best = max(run["layer_results"], key=lambda r: r["accuracy"])
        cfg  = run["probe_config"]
        print(f"  {run['_run_name']:<30} {best['accuracy']:>8.4f} {best['layer']:>6} "
              f"{cfg.get('probe_type','?'):<8} {cfg.get('C',0):>6} {cfg.get('random_seed',0):>6}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare probe runs.")
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Run directory name to use as baseline for delta heatmap "
             "(e.g. '20240101_120000_baseline'). Defaults to earliest run."
    )
    args = parser.parse_args()

    runs_dir = os.path.join(RESULTS_DIR, "runs")
    runs     = load_all_runs(runs_dir)

    if len(runs) == 1:
        print("\nOnly one run found — comparison plots need 2+ runs.")
        print("Change PROBE_CONFIG settings in config.py and run 04_probe.py again.")
        print("Generating single-run summary instead...\n")

    print_summary_table(runs)

    print("Generating comparison plots...")
    plot_layer_accuracy(runs)
    plot_best_accuracy_bar(runs)
    plot_delta_heatmap(runs, args.baseline)
    plot_per_tone_f1(runs)

    print(f"\nAll plots saved -> {COMPARISON_DIR}/")


if __name__ == "__main__":
    main()
