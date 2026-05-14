# 09_layer_analysis.py
# Deep analysis of hidden states across all encoder layers using the best
# settings: PhoWhisper-small, balanced class weights, 2000 samples.
#
# Generates the following plots in data/results/layer_analysis/:
#   1. layer_accuracy_detailed.png  — accuracy + macro F1 per layer
#   2. per_tone_f1_per_layer.png    — F1 heatmap: tones × layers
#   3. umap_all_layers.png          — UMAP grid showing all 13 layers
#   4. confusion_evolution.png      — confusion matrices for early/mid/late layers
#   5. layer_separation.png         — silhouette score per layer (cluster quality)
#
# Usage: python 09_layer_analysis.py

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    silhouette_score
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config import (
    HIDDEN_STATES_DIR, RESULTS_DIR, TONE_NAMES, WHISPER_MODEL,
    PROBE_CONFIG
)

# ── Output directory ──────────────────────────────────────────────────────────
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "layer_analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Tone colors ───────────────────────────────────────────────────────────────
TONE_COLORS = {
    0: "#4C72B0",   # ngang — blue
    1: "#55A868",   # huyen — green
    2: "#C44E52",   # sac   — red
    3: "#8172B2",   # hoi   — purple
    4: "#CCB974",   # nga   — gold
    5: "#64B5CD",   # nang  — teal
}
TONE_KEYS = [TONE_NAMES[i] for i in range(len(TONE_NAMES))]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_hidden_states() -> tuple[np.ndarray, np.ndarray]:
    """Load all saved hidden states. Returns X (n, layers, dim) and y (n,)."""
    meta_path = os.path.join(HIDDEN_STATES_DIR, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("meta.json not found. Run 03_extract_hidden_states.py first.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    X_list, y_list = [], []
    for idx_str, tone_id in sorted(meta.items(), key=lambda x: int(x[0])):
        npy_path = os.path.join(HIDDEN_STATES_DIR, f"{idx_str}.npy")
        if not os.path.exists(npy_path):
            continue
        X_list.append(np.load(npy_path))
        y_list.append(tone_id)
    return np.stack(X_list), np.array(y_list)


# ── Probe per layer ───────────────────────────────────────────────────────────

def probe_all_layers(X: np.ndarray, y: np.ndarray) -> list[dict]:
    """
    Train a balanced logistic regression probe on each layer.
    Returns list of result dicts with accuracy, macro_f1, per_tone_f1,
    confusion matrix, and silhouette score.
    """
    n_layers = X.shape[1]
    results  = []

    print("Probing all layers...")
    for layer_idx in tqdm(range(n_layers), desc="Layers"):
        X_layer = X[:, layer_idx, :]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_layer, y,
            test_size=PROBE_CONFIG["test_size"],
            random_state=PROBE_CONFIG["random_seed"],
            stratify=y
        )

        # Scale
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Probe — balanced to handle class imbalance
        clf = LogisticRegression(
            C=PROBE_CONFIG["C"],
            max_iter=PROBE_CONFIG["max_iter"],
            solver=PROBE_CONFIG["solver"],
            class_weight="balanced",
            random_state=PROBE_CONFIG["random_seed"],
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        cm  = confusion_matrix(y_test, preds)

        present_labels = sorted(list(set(y_test.tolist())))
        present_keys   = [TONE_NAMES[i] for i in present_labels]
        report = classification_report(
            y_test, preds,
            labels=present_labels,
            target_names=present_keys,
            output_dict=True,
            zero_division=0
        )

        macro_f1     = report["macro avg"]["f1-score"]
        per_tone_f1  = {
            tone: report.get(tone, {}).get("f1-score", 0.0)
            for tone in TONE_KEYS
        }

        # Silhouette score — measures cluster separation in representation space
        # Use scaled test features; subsample if large
        try:
            sil_X = X_test
            sil_y = y_test
            # Need at least 2 classes with >1 sample
            unique, counts = np.unique(sil_y, return_counts=True)
            valid = unique[counts > 1]
            if len(valid) >= 2:
                mask = np.isin(sil_y, valid)
                sil_score = silhouette_score(
                    sil_X[mask], sil_y[mask],
                    sample_size=min(500, mask.sum()),
                    random_state=42
                )
            else:
                sil_score = 0.0
        except Exception:
            sil_score = 0.0

        results.append({
            "layer"       : layer_idx,
            "accuracy"    : round(float(acc), 4),
            "macro_f1"    : round(float(macro_f1), 4),
            "per_tone_f1" : {k: round(v, 4) for k, v in per_tone_f1.items()},
            "confusion_matrix": cm.tolist(),
            "silhouette"  : round(float(sil_score), 4),
        })

    return results


# ── Plot 1: Accuracy + Macro F1 per layer ────────────────────────────────────

def plot_accuracy_and_f1(results: list[dict]):
    layers   = [r["layer"] for r in results]
    accs     = [r["accuracy"] for r in results]
    f1s      = [r["macro_f1"] for r in results]
    best_acc = max(results, key=lambda r: r["accuracy"])
    best_f1  = max(results, key=lambda r: r["macro_f1"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, accs, marker="o", linewidth=2,
            color="#4C72B0", label="Accuracy")
    ax.plot(layers, f1s,  marker="s", linewidth=2,
            color="#C44E52", label="Macro F1", linestyle="--")

    # Annotate peaks
    ax.annotate(
        f"Best acc: L{best_acc['layer']} ({best_acc['accuracy']:.3f})",
        xy=(best_acc["layer"], best_acc["accuracy"]),
        xytext=(best_acc["layer"] + 0.5, best_acc["accuracy"] + 0.03),
        arrowprops=dict(arrowstyle="->", color="#4C72B0"),
        color="#4C72B0", fontsize=9
    )
    ax.annotate(
        f"Best F1: L{best_f1['layer']} ({best_f1['macro_f1']:.3f})",
        xy=(best_f1["layer"], best_f1["macro_f1"]),
        xytext=(best_f1["layer"] - 3, best_f1["macro_f1"] + 0.03),
        arrowprops=dict(arrowstyle="->", color="#C44E52"),
        color="#C44E52", fontsize=9
    )

    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("Score")
    ax.set_title(f"Probe Accuracy and Macro F1 per Layer\n{WHISPER_MODEL} | balanced | 2000 samples")
    ax.set_ylim(0, 0.8)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = os.path.join(ANALYSIS_DIR, "layer_accuracy_detailed.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 2: Per-tone F1 heatmap (tones × layers) ─────────────────────────────

def plot_per_tone_f1_heatmap(results: list[dict]):
    n_layers = len(results)
    matrix   = np.zeros((len(TONE_KEYS), n_layers))

    for r in results:
        for j, tone in enumerate(TONE_KEYS):
            matrix[j, r["layer"]] = r["per_tone_f1"].get(tone, 0.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        matrix,
        annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=[f"L{i}" for i in range(n_layers)],
        yticklabels=TONE_KEYS,
        ax=ax, vmin=0, vmax=0.8,
        linewidths=0.3
    )
    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("Tone")
    ax.set_title(f"Per-tone F1 Score Across Layers\n{WHISPER_MODEL} | balanced | 2000 samples")
    fig.tight_layout()

    out = os.path.join(ANALYSIS_DIR, "per_tone_f1_per_layer.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 3: UMAP grid (all layers) ───────────────────────────────────────────

def plot_umap_grid(X: np.ndarray, y: np.ndarray, n_layers: int):
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed — skipping UMAP grid.")
        return

    # Arrange layers in a grid
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.5, n_rows * 3.2))
    axes = axes.flatten()

    print(f"  Running UMAP for all {n_layers} layers (this may take a few minutes)...")
    for layer_idx in tqdm(range(n_layers), desc="UMAP layers"):
        ax = axes[layer_idx]
        X_layer = X[:, layer_idx, :]
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X_layer)

        reducer  = umap.UMAP(
            n_components=2, random_state=42,
            n_neighbors=15, min_dist=0.1, n_jobs=1
        )
        embedded = reducer.fit_transform(X_scaled)

        for tone_id, color in TONE_COLORS.items():
            mask = y == tone_id
            if mask.sum() == 0:
                continue
            ax.scatter(
                embedded[mask, 0], embedded[mask, 1],
                c=color, alpha=0.6, s=8, edgecolors="none",
                label=TONE_NAMES[tone_id]
            )

        ax.set_title(f"Layer {layer_idx}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    # Add shared legend
    handles = [
        plt.scatter([], [], c=color, s=30, label=TONE_NAMES[tid])
        for tid, color in TONE_COLORS.items()
        if tid in TONE_NAMES
    ]
    fig.legend(handles=handles, title="Tone",
               loc="lower right", fontsize=8)
    fig.suptitle(
        f"UMAP Representations Across All Layers\n{WHISPER_MODEL} | 2000 samples",
        fontsize=11
    )
    fig.tight_layout()

    out = os.path.join(ANALYSIS_DIR, "umap_all_layers.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 4: Confusion matrix evolution ───────────────────────────────────────

def plot_confusion_evolution(results: list[dict]):
    """Show confusion matrices at early, middle, and best layer."""
    n_layers = len(results)
    best_layer = max(results, key=lambda r: r["accuracy"])["layer"]

    # Pick representative layers
    early  = 0
    mid    = n_layers // 2
    late   = best_layer
    chosen = sorted(set([early, mid, late]))
    titles = {early: f"Early (Layer {early})",
               mid  : f"Middle (Layer {mid})",
               late : f"Best (Layer {late})"}

    fig, axes = plt.subplots(1, len(chosen), figsize=(len(chosen) * 5, 5))
    if len(chosen) == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, chosen):
        r      = results[layer_idx]
        cm     = np.array(r["confusion_matrix"])

        # Pad confusion matrix to full 6×6 if some classes were absent
        full_cm = np.zeros((len(TONE_KEYS), len(TONE_KEYS)))
        present = sorted(set(
            [i for row in r["confusion_matrix"] for i in range(len(row))]
        ))
        for i, pi in enumerate(range(len(cm))):
            for j, pj in enumerate(range(len(cm[0]))):
                if i < len(TONE_KEYS) and j < len(TONE_KEYS):
                    full_cm[i, j] = cm[i, j]

        cm_norm = full_cm / (full_cm.sum(axis=1, keepdims=True) + 1e-9)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=TONE_KEYS, yticklabels=TONE_KEYS,
            ax=ax, vmin=0, vmax=1, linewidths=0.3
        )
        ax.set_title(
            f"{titles[layer_idx]}\nAcc={r['accuracy']:.3f}  F1={r['macro_f1']:.3f}",
            fontsize=10
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle(
        f"Confusion Matrix Evolution Across Layers\n{WHISPER_MODEL}",
        fontsize=11
    )
    fig.tight_layout()

    out = os.path.join(ANALYSIS_DIR, "confusion_evolution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Plot 5: Silhouette score per layer ────────────────────────────────────────

def plot_silhouette(results: list[dict]):
    """
    Silhouette score measures how well-separated tone clusters are in
    representation space. Higher = more separable = better tone encoding.
    Range: -1 (worst) to +1 (best).
    """
    layers = [r["layer"] for r in results]
    scores = [r["silhouette"] for r in results]
    best   = max(results, key=lambda r: r["silhouette"])

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(layers, scores,
                  color=["#C44E52" if s == max(scores) else "#4C72B0"
                         for s in scores],
                  edgecolor="white")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(
        f"Tone Cluster Separation per Layer (Silhouette Score)\n"
        f"{WHISPER_MODEL} | Higher = more separable tones"
    )
    ax.annotate(
        f"Best: L{best['layer']} ({best['silhouette']:.3f})",
        xy=(best["layer"], best["silhouette"]),
        xytext=(best["layer"] + 0.5, best["silhouette"] + 0.01),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(ANALYSIS_DIR, "layer_separation.png")
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved → {out}")


# ── Save results ──────────────────────────────────────────────────────────────

def save_analysis(results: list[dict]):
    out = os.path.join(ANALYSIS_DIR, "layer_analysis_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "whisper_model": WHISPER_MODEL,
            "probe_config" : PROBE_CONFIG,
            "layer_results": results
        }, f, indent=2)
    print(f"  Results saved → {out}")


# ── Print summary ─────────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    best_acc = max(results, key=lambda r: r["accuracy"])
    best_f1  = max(results, key=lambda r: r["macro_f1"])
    best_sil = max(results, key=lambda r: r["silhouette"])

    print("\n── Layer Analysis Summary ───────────────────────────────")
    print(f"  {'Layer':<8} {'Accuracy':>9} {'Macro F1':>9} {'Silhouette':>11}")
    print(f"  {'─'*8} {'─'*9} {'─'*9} {'─'*11}")
    for r in results:
        marker = " ← best acc" if r["layer"] == best_acc["layer"] else ""
        print(f"  {r['layer']:<8} {r['accuracy']:>9.4f} "
              f"{r['macro_f1']:>9.4f} {r['silhouette']:>11.4f}{marker}")

    print(f"\n  Best accuracy  : Layer {best_acc['layer']} "
          f"({best_acc['accuracy']:.4f})")
    print(f"  Best macro F1  : Layer {best_f1['layer']} "
          f"({best_f1['macro_f1']:.4f})")
    print(f"  Best separation: Layer {best_sil['layer']} "
          f"({best_sil['silhouette']:.4f})")

    print(f"\n── Per-tone F1 at Best Layer (Layer {best_acc['layer']}) ──")
    for tone, f1 in best_acc["per_tone_f1"].items():
        bar = "█" * int(f1 * 30)
        print(f"  {tone:<8} {f1:.4f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Model : {WHISPER_MODEL}")
    print(f"Config: balanced=True, samples=2000\n")

    # Load data
    print("Loading hidden states...")
    X, y = load_hidden_states()
    n_samples, n_layers, hidden_dim = X.shape
    print(f"  Samples: {n_samples} | Layers: {n_layers} | Hidden dim: {hidden_dim}")

    # Probe all layers
    results = probe_all_layers(X, y)

    # Print summary
    print_summary(results)

    # Save results
    print("\nSaving results...")
    save_analysis(results)

    # Generate plots
    print("\nGenerating plots...")
    print("\n1. Accuracy + Macro F1 curve...")
    plot_accuracy_and_f1(results)

    print("\n2. Per-tone F1 heatmap...")
    plot_per_tone_f1_heatmap(results)

    print("\n3. UMAP grid across all layers...")
    plot_umap_grid(X, y, n_layers)

    print("\n4. Confusion matrix evolution...")
    plot_confusion_evolution(results)

    print("\n5. Silhouette score per layer...")
    plot_silhouette(results)

    print(f"\n✅ All outputs saved to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
