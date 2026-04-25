# 05_baseline.py
# Acoustic baseline: classify tones using hand-crafted features (F0 + MFCCs)
# extracted with librosa. Compare this accuracy to the Whisper probe results.
#
# Features per clip:
#   - F0 mean, std, min, max, range  (5)
#   - MFCCs 1–13: mean + std         (26)
#   Total: 31 features
#
# Usage: python 05_baseline.py

import json
import os
import numpy as np
import librosa
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config import (
    DATASET_NAME, DATASET_LANG, DATASET_SPLIT, MAX_SAMPLES,
    SAMPLE_RATE, LABELED_PATH, RESULTS_DIR, TONE_NAMES
)


def extract_acoustic_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract F0 statistics and MFCC statistics from a waveform.
    Returns a 1D feature vector of length 31.
    """
    # ── F0 (pitch) ────────────────────────────────────────────────────────────
    # librosa.yin is fast and works well for speech
    f0 = librosa.yin(audio, fmin=75, fmax=400, sr=sr)
    # Replace unvoiced frames (≈0 Hz) with NaN for stats
    f0_voiced = f0[f0 > 50]
    if len(f0_voiced) == 0:
        f0_voiced = np.array([0.0])

    f0_feats = np.array([
        np.mean(f0_voiced),
        np.std(f0_voiced),
        np.min(f0_voiced),
        np.max(f0_voiced),
        np.max(f0_voiced) - np.min(f0_voiced),   # range (contour span)
    ])

    # ── MFCCs ─────────────────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)  # (13, frames)
    mfcc_mean = np.mean(mfccs, axis=1)   # (13,)
    mfcc_std  = np.std(mfccs, axis=1)    # (13,)
    mfcc_feats = np.concatenate([mfcc_mean, mfcc_std])        # (26,)

    return np.concatenate([f0_feats, mfcc_feats])   # (31,)


def main():
    # ── Load labels ───────────────────────────────────────────────────────────
    if not os.path.exists(LABELED_PATH):
        raise FileNotFoundError(f"{LABELED_PATH} not found. Run 02_label_tones.py first.")

    with open(LABELED_PATH) as f:
        records = json.load(f)
    index_to_tone = {r["index"]: r["tone_id"] for r in records}

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading dataset…")
    args = [DATASET_NAME] + ([DATASET_LANG] if DATASET_LANG else [])
    ds = load_dataset(*args, split=DATASET_SPLIT)
    if MAX_SAMPLES is not None:
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

    # ── Extract features ──────────────────────────────────────────────────────
    X_list, y_list = [], []

    for i, example in enumerate(tqdm(ds, desc="Extracting acoustic features")):
        if i not in index_to_tone:
            continue

        audio = example["audio"]["array"].astype(np.float32)
        sr    = example["audio"]["sampling_rate"]

        # Resample to 16 kHz if needed
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        feats = extract_acoustic_features(audio, sr)
        X_list.append(feats)
        y_list.append(index_to_tone[i])

    X = np.stack(X_list)
    y = np.array(y_list)
    print(f"\nFeature matrix: {X.shape}  (samples × features)")

    # ── Classify ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs", random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\n── Acoustic Baseline Results ─────────────────")
    print(f"  Accuracy: {acc:.4f}")

    # Per-class report
    target_names = [TONE_NAMES[i] for i in range(len(TONE_NAMES))]
    print("\n" + classification_report(y_test, preds, target_names=target_names, zero_division=0))

    # ── Compare to Whisper probe ───────────────────────────────────────────────
    probe_path = os.path.join(RESULTS_DIR, "probe_results.json")
    if os.path.exists(probe_path):
        with open(probe_path) as f:
            probe_data = json.load(f)
        best_whisper = max(probe_data["layer_results"], key=lambda r: r["accuracy"])
        print(f"── Comparison ────────────────────────────────")
        print(f"  Acoustic baseline   : {acc:.4f}")
        print(f"  Whisper best layer  : {best_whisper['accuracy']:.4f}  (layer {best_whisper['layer']})")
        print(f"  Majority baseline   : {probe_data['majority_baseline']:.4f}")
        delta = best_whisper["accuracy"] - acc
        print(f"  Δ (Whisper - acoustic): {delta:+.4f}")
    else:
        print("(Run 04_probe.py to compare against Whisper layer probes.)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump({"acoustic_baseline_accuracy": round(acc, 4)}, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
