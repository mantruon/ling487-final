# 04_probe.py
# Trains a logistic regression probe on each Whisper encoder layer.
# For each layer: splits data into train/test, fits probe, reports accuracy.
# Saves results to data/results/probe_results.json and prints a layer table.
#
# Usage: python 04_probe.py

import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config import HIDDEN_STATES_DIR, RESULTS_DIR, TONE_NAMES


def load_data(hidden_states_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all saved hidden states and labels.
    Returns:
        X: (n_samples, n_layers, hidden_dim)
        y: (n_samples,)
    """
    meta_path = os.path.join(hidden_states_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("meta.json not found. Run 03_extract_hidden_states.py first.")

    with open(meta_path) as f:
        meta = json.load(f)   # {str(index): tone_id}

    X_list, y_list = [], []
    for idx_str, tone_id in sorted(meta.items(), key=lambda x: int(x[0])):
        npy_path = os.path.join(hidden_states_dir, f"{idx_str}.npy")
        if not os.path.exists(npy_path):
            continue
        hidden = np.load(npy_path)   # (n_layers, hidden_dim)
        X_list.append(hidden)
        y_list.append(tone_id)

    X = np.stack(X_list)          # (n_samples, n_layers, hidden_dim)
    y = np.array(y_list)          # (n_samples,)
    return X, y


def probe_layer(X_layer: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    """
    Train a logistic regression probe on a single layer's representations.
    Returns accuracy and confusion matrix.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_layer, y, test_size=test_size, random_state=42, stratify=y
    )

    # Standardize features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm  = confusion_matrix(y_test, preds)
    return acc, cm, clf


def main():
    print("Loading hidden states…")
    X, y = load_data(HIDDEN_STATES_DIR)
    n_samples, n_layers, hidden_dim = X.shape
    print(f"  Samples: {n_samples}  |  Layers: {n_layers}  |  Hidden dim: {hidden_dim}")

    # Baseline: random chance
    from collections import Counter
    counts   = Counter(y.tolist())
    majority = max(counts.values()) / n_samples
    print(f"  Majority-class baseline: {majority:.3f}  ({n_samples} samples, {len(counts)} classes)")

    # ── Probe each layer ──────────────────────────────────────────────────────
    results = []
    print("\nProbing layers…")

    for layer_idx in tqdm(range(n_layers), desc="Layers"):
        X_layer = X[:, layer_idx, :]   # (n_samples, hidden_dim)
        acc, cm, _ = probe_layer(X_layer, y)

        results.append({
            "layer"   : layer_idx,
            "accuracy": round(acc, 4),
            "confusion_matrix": cm.tolist()
        })

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n── Layer-wise Probe Accuracy ─────────────────────")
    print(f"  {'Layer':<8} {'Accuracy':>10}  {'Bar'}")
    print(f"  {'─'*8} {'─'*10}  {'─'*30}")
    for r in results:
        bar = "█" * int(r["accuracy"] * 40)
        print(f"  {r['layer']:<8} {r['accuracy']:>10.4f}  {bar}")

    best = max(results, key=lambda r: r["accuracy"])
    print(f"\n  ✅ Best layer: {best['layer']}  (accuracy={best['accuracy']:.4f})")
    print(f"     Majority baseline: {majority:.4f}")

    # ── Confusion matrix for best layer ──────────────────────────────────────
    print(f"\n── Confusion Matrix (Layer {best['layer']}) ──────────")
    cm = np.array(best["confusion_matrix"])
    tone_keys = [TONE_NAMES[i] for i in range(len(TONE_NAMES))]
    header = "       " + "  ".join(f"{t[:5]:>5}" for t in tone_keys)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>5}" for v in row)
        print(f"  {tone_keys[i][:5]:>5}  {row_str}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "probe_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_samples"        : n_samples,
            "n_layers"         : n_layers,
            "majority_baseline": round(majority, 4),
            "layer_results"    : results
        }, f, indent=2)

    print(f"\n  Results saved → {out_path}")
    print("  Proceed to 05_baseline.py or 06_visualize.py")


if __name__ == "__main__":
    main()
