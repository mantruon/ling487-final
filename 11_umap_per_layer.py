# 11_umap_per_layer.py
# Generates a separate UMAP image for each encoder layer.
# Each image is saved as umap_layer_XX.png in data/results/umap_layers/.
#
# Also generates a summary strip image showing all layers side by side
# for quick visual comparison.
#
# Usage:
#   python 11_umap_per_layer.py                  # all layers
#   python 11_umap_per_layer.py --layers 0 6 9   # specific layers only
#   python 11_umap_per_layer.py --force           # regenerate existing images
#   python 11_umap_per_layer.py --no-strip        # skip the summary strip

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config import HIDDEN_STATES_DIR, RESULTS_DIR, TONE_NAMES, WHISPER_MODEL

# ── Output directory ──────────────────────────────────────────────────────────
UMAP_DIR = os.path.join(RESULTS_DIR, "umap_layers")
os.makedirs(UMAP_DIR, exist_ok=True)

# ── Tone colors ───────────────────────────────────────────────────────────────
TONE_COLORS = {
    0: "#4C72B0",   # ngang — blue
    1: "#55A868",   # huyen — green
    2: "#C44E52",   # sac   — red
    3: "#8172B2",   # hoi   — purple
    4: "#CCB974",   # nga   — gold
    5: "#64B5CD",   # nang  — teal
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_hidden_states() -> tuple[np.ndarray, np.ndarray]:
    """Load all saved hidden states. Returns X (n, layers, dim) and y (n,)."""
    meta_path = os.path.join(HIDDEN_STATES_DIR, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            "meta.json not found. Run 03_extract_hidden_states.py first."
        )
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


def get_best_layer() -> int:
    """Read best layer from saved probe results."""
    for path in [
        os.path.join(RESULTS_DIR, "layer_analysis", "layer_analysis_results.json"),
        os.path.join(RESULTS_DIR, "probe_results.json"),
    ]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            best = max(data["layer_results"], key=lambda r: r["accuracy"])
            return best["layer"]
    return 9   # default


# ── UMAP per layer ────────────────────────────────────────────────────────────

def compute_umap(X_layer: np.ndarray) -> np.ndarray:
    """Scale and run 2D UMAP on one layer's representations."""
    import umap
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_layer)
    reducer  = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1,
        n_jobs=1
    )
    return reducer.fit_transform(X_scaled)


def plot_single_layer(embedded: np.ndarray, y: np.ndarray,
                      layer_idx: int, best_layer: int,
                      acc: float | None = None) -> plt.Figure:
    """
    Create a single UMAP figure for one layer.
    Marks the best layer with a gold border.
    Returns the figure without saving.
    """
    is_best   = layer_idx == best_layer
    border_color = "#FFD700" if is_best else "white"
    border_width = 3 if is_best else 1

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot each tone class
    legend_handles = []
    for tone_id, color in TONE_COLORS.items():
        mask = y == tone_id
        if mask.sum() == 0:
            continue
        tone_name = TONE_NAMES.get(tone_id, str(tone_id))
        ax.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=color, alpha=0.7, s=18,
            edgecolors="none", label=tone_name
        )
        legend_handles.append(
            mpatches.Patch(color=color, label=tone_name)
        )

    # Title — highlight best layer
    title = f"Layer {layer_idx}"
    if is_best:
        title += "  ★ Best Layer"
    if acc is not None:
        title += f"  (acc={acc:.3f})"

    ax.set_title(title, fontsize=11,
                 color="#B8860B" if is_best else "black",
                 fontweight="bold" if is_best else "normal")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.legend(handles=legend_handles, title="Tone",
              fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)

    # Gold border for best layer
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_width)

    # Model name watermark
    fig.text(0.01, 0.01, WHISPER_MODEL,
             fontsize=7, color="gray", alpha=0.6)

    fig.tight_layout()
    return fig


def save_layer_umap(X: np.ndarray, y: np.ndarray,
                    layer_idx: int, best_layer: int,
                    acc: float | None, force: bool) -> np.ndarray | None:
    """
    Compute and save UMAP for one layer.
    Returns the embedded coordinates for use in the strip plot.
    Skips if file exists and force=False.
    """
    out_path = os.path.join(UMAP_DIR, f"umap_layer_{layer_idx:02d}.png")

    if os.path.exists(out_path) and not force:
        print(f"  Layer {layer_idx:02d}: skipping (already exists) — use --force to regenerate")
        # Still need to return embedded for strip — recompute
        return compute_umap(X[:, layer_idx, :])

    print(f"  Layer {layer_idx:02d}: running UMAP...")
    embedded = compute_umap(X[:, layer_idx, :])
    fig = plot_single_layer(embedded, y, layer_idx, best_layer, acc)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {out_path}")
    return embedded


# ── Summary strip ─────────────────────────────────────────────────────────────

def plot_summary_strip(all_embedded: dict[int, np.ndarray],
                       y: np.ndarray, best_layer: int,
                       layer_accs: dict[int, float]):
    """
    Single wide image showing all layers in a grid.
    Saved as umap_all_layers_strip.png.
    """
    layers  = sorted(all_embedded.keys())
    n       = len(layers)
    n_cols  = 4
    n_rows  = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 3.2)
    )
    axes = axes.flatten()

    for i, layer_idx in enumerate(layers):
        ax       = axes[i]
        embedded = all_embedded[layer_idx]
        is_best  = layer_idx == best_layer

        for tone_id, color in TONE_COLORS.items():
            mask = y == tone_id
            if mask.sum() == 0:
                continue
            ax.scatter(
                embedded[mask, 0], embedded[mask, 1],
                c=color, alpha=0.65, s=8, edgecolors="none"
            )

        acc   = layer_accs.get(layer_idx)
        title = f"L{layer_idx}"
        if is_best:
            title += " ★"
        if acc is not None:
            title += f"  {acc:.3f}"

        ax.set_title(title, fontsize=9,
                     color="#B8860B" if is_best else "black",
                     fontweight="bold" if is_best else "normal")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor("#FFD700" if is_best else "#dddddd")
            spine.set_linewidth(2 if is_best else 0.5)

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    # Shared legend
    handles = [
        mpatches.Patch(color=color, label=TONE_NAMES.get(tid, str(tid)))
        for tid, color in TONE_COLORS.items()
        if np.any(y == tid)
    ]
    fig.legend(
        handles=handles, title="Tone",
        loc="lower right", fontsize=8, ncol=2
    )

    fig.suptitle(
        f"UMAP Representations Across All Layers\n"
        f"{WHISPER_MODEL}  |  ★ = best layer",
        fontsize=11
    )
    fig.tight_layout()

    out = os.path.join(UMAP_DIR, "umap_all_layers_strip.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Strip saved → {out}")


# ── Load layer accuracies ─────────────────────────────────────────────────────

def load_layer_accuracies() -> dict[int, float]:
    """Load per-layer accuracies from saved probe results if available."""
    for path in [
        os.path.join(RESULTS_DIR, "layer_analysis", "layer_analysis_results.json"),
        os.path.join(RESULTS_DIR, "probe_results.json"),
    ]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {r["layer"]: r["accuracy"] for r in data["layer_results"]}
    return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate individual UMAP images per encoder layer."
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Specific layer indices to generate (e.g. --layers 0 6 9). "
             "Default: all layers."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate images even if they already exist."
    )
    parser.add_argument(
        "--no-strip", action="store_true",
        help="Skip generating the summary strip image."
    )
    args = parser.parse_args()

    # ── Check umap-learn ──────────────────────────────────────────────────────
    try:
        import umap
    except ImportError:
        print("umap-learn not installed. Run: uv pip install umap-learn")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading hidden states...")
    X, y = load_hidden_states()
    n_samples, n_layers, hidden_dim = X.shape
    print(f"  Samples: {n_samples} | Layers: {n_layers} | Hidden dim: {hidden_dim}")

    best_layer   = get_best_layer()
    layer_accs   = load_layer_accuracies()
    print(f"  Best layer: {best_layer}  "
          f"(acc={layer_accs.get(best_layer, '?')})")

    # ── Determine which layers to process ─────────────────────────────────────
    layers_to_run = args.layers if args.layers else list(range(n_layers))
    print(f"\nGenerating UMAP for {len(layers_to_run)} layer(s): {layers_to_run}")
    print(f"Output directory: {UMAP_DIR}\n")

    # ── Generate per-layer images ─────────────────────────────────────────────
    all_embedded = {}

    for layer_idx in tqdm(layers_to_run, desc="UMAP layers"):
        acc      = layer_accs.get(layer_idx)
        embedded = save_layer_umap(
            X, y, layer_idx, best_layer, acc, args.force
        )
        if embedded is not None:
            all_embedded[layer_idx] = embedded

    # ── Summary strip ─────────────────────────────────────────────────────────
    if not args.no_strip and len(all_embedded) > 1:
        strip_path = os.path.join(UMAP_DIR, "umap_all_layers_strip.png")
        if os.path.exists(strip_path) and not args.force:
            print(f"\n  Strip already exists — skipping "
                  f"(use --force to regenerate): {strip_path}")
        else:
            print("\nGenerating summary strip...")
            plot_summary_strip(all_embedded, y, best_layer, layer_accs)

    # ── Summary ───────────────────────────────────────────────────────────────
    generated = [
        f for f in os.listdir(UMAP_DIR) if f.startswith("umap_layer_")
    ]
    print(f"\n✅ Done.")
    print(f"   {len(generated)} layer image(s) in {UMAP_DIR}/")
    print(f"   Best layer (L{best_layer}) is marked with ★ and a gold border.")


if __name__ == "__main__":
    main()
