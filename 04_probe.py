# 04_probe.py
# Trains a probe classifier on each Whisper encoder layer.
# Reads all hyperparameters from PROBE_CONFIG in config.py.
#
# Each run saves a timestamped report to:
#   data/results/runs/<timestamp>_<label>/
#     ├── probe_results.json   ← layer accuracies + confusion matrices
#     ├── config_snapshot.json ← exact settings used
#     └── report.txt           ← human-readable summary
#
# Also overwrites data/results/probe_results.json (used by 06_visualize.py).
#
# Usage: python 04_probe.py

import json
import os
import datetime
import platform
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config import HIDDEN_STATES_DIR, RESULTS_DIR, TONE_NAMES, PROBE_CONFIG, WHISPER_MODEL


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(hidden_states_dir: str) -> tuple[np.ndarray, np.ndarray]:
    meta_path = os.path.join(hidden_states_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("meta.json not found. Run 03_extract_hidden_states.py first.")
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


# ── Probe classifiers ─────────────────────────────────────────────────────────

def make_probe(cfg: dict):
    if cfg["probe_type"] == "linear":
        return LogisticRegression(
            C=cfg["C"], max_iter=cfg["max_iter"], solver=cfg["solver"], 
            class_weight="balanced",random_state=cfg["random_seed"],
        )          
    elif cfg["probe_type"] == "mlp":
        hidden = tuple([cfg["mlp_hidden"]] * cfg["mlp_layers"])
        return MLPClassifier(
            hidden_layer_sizes=hidden, max_iter=cfg["max_iter"],
            random_state=cfg["random_seed"],
        )
    else:
        raise ValueError(f"Unknown probe_type: {cfg['probe_type']}. Use 'linear' or 'mlp'.")


def probe_layer(X_layer: np.ndarray, y: np.ndarray, cfg: dict) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X_layer, y, test_size=cfg["test_size"],
        random_state=cfg["random_seed"], stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    clf = make_probe(cfg)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm  = confusion_matrix(y_test, preds)
    # Use only the tone labels actually present in the test set
    present_labels = sorted(list(set(y_test.tolist())))
    tone_keys = [TONE_NAMES[i] for i in present_labels]
    per_class = classification_report(
        y_test, preds,
        labels=present_labels,
        target_names=tone_keys,
        output_dict=True,
        zero_division=0
    )
    return {
        "accuracy"        : round(float(acc), 4),
        "confusion_matrix": cm.tolist(),
        "per_class"       : per_class,
    }


# ── Run directory ─────────────────────────────────────────────────────────────

def make_run_dir() -> str:
    runs_dir  = os.path.join(RESULTS_DIR, "runs")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    label     = PROBE_CONFIG.get("run_label", "run").replace(" ", "_")
    run_dir   = os.path.join(runs_dir, f"{timestamp}_{label}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ── Report ────────────────────────────────────────────────────────────────────

def save_report(run_dir: str, results: list, n_samples: int,
                n_layers: int, majority: float):
    cfg       = PROBE_CONFIG
    best      = max(results, key=lambda r: r["accuracy"])
    tone_keys = [TONE_NAMES[i] for i in range(len(TONE_NAMES))]

    lines = [
        "=" * 60,
        "  WHISPER TONE PROBE -- RUN REPORT",
        "=" * 60,
        f"  Run label    : {cfg['run_label']}",
        f"  Notes        : {cfg['notes'] or '(none)'}",
        f"  Timestamp    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Platform     : {platform.system()}",
        f"  Whisper model: {WHISPER_MODEL}",
        "",
        "-- Hyperparameters ------------------------------------------",
        f"  probe_type   : {cfg['probe_type']}",
        f"  C (reg)      : {cfg['C']}",
        f"  solver       : {cfg['solver']}",
        f"  max_iter     : {cfg['max_iter']}",
        f"  test_size    : {cfg['test_size']}",
        f"  random_seed  : {cfg['random_seed']}",
        f"  pooling      : {cfg['pooling']}",
    ]
    if cfg["probe_type"] == "mlp":
        lines += [f"  mlp_hidden   : {cfg['mlp_hidden']}",
                  f"  mlp_layers   : {cfg['mlp_layers']}"]
    lines += [
        "",
        "-- Dataset --------------------------------------------------",
        f"  Samples      : {n_samples}",
        f"  Layers probed: {n_layers}",
        f"  Majority base: {majority:.4f}",
        "",
        "-- Layer-wise Accuracy --------------------------------------",
        f"  {'Layer':<8} {'Accuracy':>10}  Bar",
        f"  {'':->8} {'':->10}  {'':->30}",
    ]
    for r in results:
        bar = "X" * int(r["accuracy"] * 40)
        lines.append(f"  {r['layer']:<8} {r['accuracy']:>10.4f}  {bar}")
    lines += [
        "",
        f"  Best layer   : {best['layer']}  (accuracy={best['accuracy']:.4f})",
        f"  Delta vs maj : {best['accuracy'] - majority:+.4f}",
        "",
        f"-- Per-class F1 (Best Layer {best['layer']}) -------------------------",
    ]
    for tone in tone_keys:
        if tone in best["per_class"]:
            f1 = best["per_class"][tone]["f1-score"]
            lines.append(f"  {tone:<8} F1={f1:.4f}")
    lines += [
        "",
        f"-- Confusion Matrix (Layer {best['layer']}) --------------------------",
        "       " + "  ".join(f"{t[:5]:>5}" for t in tone_keys),
    ]
    cm = np.array(best["confusion_matrix"])
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>5}" for v in row)
        lines.append(f"  {tone_keys[i][:5]:>5}  {row_str}")
    lines += ["", "=" * 60]

    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open(os.path.join(run_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = PROBE_CONFIG
    print("Loading hidden states...")
    X, y = load_data(HIDDEN_STATES_DIR)
    n_samples, n_layers, hidden_dim = X.shape
    print(f"  Samples: {n_samples}  |  Layers: {n_layers}  |  Hidden dim: {hidden_dim}")
    print(f"  Probe: {cfg['probe_type']}  C={cfg['C']}  seed={cfg['random_seed']}")

    counts   = Counter(y.tolist())
    majority = max(counts.values()) / n_samples
    print(f"  Majority baseline: {majority:.4f}")

    results = []
    print("\nProbing layers...")
    for layer_idx in tqdm(range(n_layers), desc="Layers"):
        X_layer = X[:, layer_idx, :]
        layer_result = probe_layer(X_layer, y, cfg)
        layer_result["layer"] = layer_idx
        results.append(layer_result)

    run_dir = make_run_dir()

    full_results = {
        "run_label"        : cfg["run_label"],
        "notes"            : cfg["notes"],
        "whisper_model"    : WHISPER_MODEL,
        "probe_config"     : cfg,
        "n_samples"        : n_samples,
        "n_layers"         : n_layers,
        "majority_baseline": round(majority, 4),
        "layer_results"    : results,
    }

    with open(os.path.join(run_dir, "probe_results.json"), "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    with open(os.path.join(run_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump({"whisper_model": WHISPER_MODEL, **cfg}, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "probe_results.json"), "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    save_report(run_dir, results, n_samples, n_layers, majority)

    print(f"\n  Run saved    -> {run_dir}")
    print(f"  Main results -> {os.path.join(RESULTS_DIR, 'probe_results.json')}")
    print("  Run 06_visualize.py or 07_compare_runs.py to plot.")


if __name__ == "__main__":
    main()