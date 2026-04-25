# 06_visualize.py
# Generates three visualizations saved to data/results/:
#   1. layer_accuracy.png     — Line plot of probe accuracy per Whisper layer
#   2. umap_best_layer.png    — UMAP projection of best layer colored by tone
#   3. confusion_matrix.png   — Confusion matrix heatmap for best layer
#
# Usage: python 06_visualize.py

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from config import HIDDEN_STATES_DIR, RESULTS_DIR, TONE_NAMES

# Color palette — one color per tone
TONE_COLORS = {
    0: "#4C72B0",   # ngang  — blue
    1: "#55A868",   # huyen  — green
    2: "#C44E52",   # sac    — red
    3: "#8172B2",   # hoi    — purple
    4: "#CCB974",   # nga    — gold
    5: "#64B5CD",   # nang   — teal
}


def load_hidden_states(hidden_states_dir: str):
    meta_path = os.path.join(hidden_states_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    X_list, y_list = [], []
    for idx_str, tone_id in sorted(meta.items(), key=lambda x: int(x[0])):
        npy_path = os.path.join(hidden_states_dir, f"{idx_str}.npy")
        if not os.path.exists(npy_path):
            continue
        X_list.append(np.load(npy_path))
        y_list.append(tone_id)

    return np.stack(X_list), np.array(y_list)


def plot_layer_accuracy(probe_results: dict, baseline_acc: float | None):
    """Line plot: probe accuracy per layer vs. acoustic baseline."""
    layers     = [r["layer"] for r in probe_results["layer_results"]]
    accuracies = [r["accuracy"] for r in probe_results["layer_results"]]
    best_layer = max(probe_results["layer_results"], key=lambda r: r["accuracy"])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, accuracies, marker="o", linewidth=2, color="#4C72B0", label="Whisper probe")
    ax.axhline(probe_results["majority_baseline"], linestyle=":", color="gray", label="Majority baseline")

    if baseline_acc is not None:
        ax.axhline(baseline_acc, linestyle="--", color="#C44E52", label="Acoustic baseline (F0+MFCC)")

    # Annotate best layer
    ax.annotate(
        f"Best: layer {best_layer['layer']}\n({best_layer['accuracy']:.3f})",
        xy=(best_layer["layer"], best_layer["accuracy"]),
        xytext=(best_layer["layer"] + 0.5, best_layer["accuracy"] - 0.05),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9
    )

    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Whisper Encoder: Tone Probe Accuracy per Layer")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "layer_accuracy.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def plot_umap(X: np.ndarray, y: np.ndarray, best_layer: int):
    """UMAP projection of the best layer's representations, colored by tone."""
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed — skipping UMAP plot. pip install umap-learn")
        return

    X_layer = X[:, best_layer, :]
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_layer)

    print(f"  Running UMAP on layer {best_layer} ({X_scaled.shape})…")
    reducer  = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedded = reducer.fit_transform(X_scaled)   # (n_samples, 2)

    fig, ax = plt.subplots(figsize=(7, 6))
    for tone_id, color in TONE_COLORS.items():
        mask = y == tone_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=color, label=TONE_NAMES[tone_id], alpha=0.7, s=20, edgecolors="none"
        )

    ax.set_title(f"UMAP — Whisper Layer {best_layer} Representations by Tone")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(title="Tone", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "umap_best_layer.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_confusion_matrix(cm_data: list[list[int]], best_layer: int):
    """Heatmap of the confusion matrix for the best layer."""
    cm     = np.array(cm_data)
    labels = [TONE_NAMES[i] for i in range(len(TONE_NAMES))]

    # Normalize by row (true label)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        vmin=0, vmax=1
    )
    ax.set_xlabel("Predicted Tone")
    ax.set_ylabel("True Tone")
    ax.set_title(f"Confusion Matrix (normalized) — Layer {best_layer}")
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


def main():
    # ── Load probe results ────────────────────────────────────────────────────
    probe_path = os.path.join(RESULTS_DIR, "probe_results.json")
    if not os.path.exists(probe_path):
        raise FileNotFoundError("probe_results.json not found. Run 04_probe.py first.")

    with open(probe_path, encoding="utf-8") as f:
        probe_results = json.load(f)

    # Optional: acoustic baseline for comparison line
    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    baseline_acc  = None
    if os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            baseline_acc = json.load(f)["acoustic_baseline_accuracy"]

    best_layer_info = max(probe_results["layer_results"], key=lambda r: r["accuracy"])
    best_layer      = best_layer_info["layer"]
    print(f"Best layer: {best_layer}  (accuracy={best_layer_info['accuracy']:.4f})")

    # ── Plot 1: Layer accuracy ────────────────────────────────────────────────
    print("\nPlotting layer accuracy…")
    plot_layer_accuracy(probe_results, baseline_acc)

    # ── Plot 2: UMAP ──────────────────────────────────────────────────────────
    print("\nLoading hidden states for UMAP…")
    X, y = load_hidden_states(HIDDEN_STATES_DIR)
    plot_umap(X, y, best_layer)

    # ── Plot 3: Confusion matrix ───────────────────────────────────────────────
    print("\nPlotting confusion matrix…")
    plot_confusion_matrix(best_layer_info["confusion_matrix"], best_layer)

    print(f"\n✅ All plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
