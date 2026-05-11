# 08_infer.py
# Run your own audio through the Whisper encoder and predict its Vietnamese tone.
# Extracts hidden states, applies the trained probe, and shows confidence scores
# and a layer-wise prediction plot.
#
# Usage:
#   python 08_infer.py --audio path/to/audio.wav
#   python 08_infer.py --audio path/to/audio.wav --layer 12
#   python 08_infer.py --audio path/to/audio.wav --record  (record from mic)
#
# Supported formats: .wav, .mp3, .flac, .m4a, .ogg

import argparse
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import WhisperModel, WhisperProcessor
from config import (
    WHISPER_MODEL, SAMPLE_RATE, DEVICE, RESULTS_DIR,
    TONE_NAMES, TONE_LABELS
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load any audio file and resample to 16kHz mono."""
    import librosa
    print(f"  Loading audio: {path}")
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / SAMPLE_RATE
    print(f"  Duration: {duration:.2f}s  |  Sample rate: {SAMPLE_RATE} Hz")
    return audio


def record_audio(duration: int = 5) -> np.ndarray:
    """Record audio from the microphone."""
    try:
        import sounddevice as sd
    except ImportError:
        print("  sounddevice not installed. Run: uv pip install sounddevice")
        sys.exit(1)

    print(f"  Recording {duration} seconds... speak now!")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    print("  Recording complete.")
    return audio.squeeze()


# ── Hidden state extraction ───────────────────────────────────────────────────

def extract_hidden_states(audio: np.ndarray, processor, model) -> np.ndarray:
    """
    Run audio through Whisper encoder.
    Returns mean-pooled hidden states: (n_layers, hidden_dim)
    """
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.no_grad():
        outputs = model.encoder(
            inputs,
            output_hidden_states=True,
            return_dict=True
        )

    pooled = []
    for layer in outputs.hidden_states:
        pooled.append(layer.squeeze(0).mean(dim=0).cpu().numpy())
    return np.stack(pooled)   # (n_layers, hidden_dim)


# ── Probe loading and inference ───────────────────────────────────────────────

def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load saved hidden states and labels for fitting probes."""
    from config import HIDDEN_STATES_DIR
    meta_path = os.path.join(HIDDEN_STATES_DIR, "meta.json")
    if not os.path.exists(meta_path):
        print("  ERROR: No training data found. Run 03_extract_hidden_states.py first.")
        sys.exit(1)

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


def fit_probe(X_train: np.ndarray, y_train: np.ndarray, layer: int):
    """Fit a logistic regression probe on one layer's training data."""
    X_layer = X_train[:, layer, :]
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_layer)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(X_scaled, y_train)
    return clf, scaler


def predict_tone(hidden: np.ndarray, clf, scaler, layer: int) -> dict:
    """
    Predict tone for a single audio clip at a given layer.
    Returns predicted tone name, confidence scores for all tones.
    """
    X = hidden[layer].reshape(1, -1)
    X_scaled = scaler.transform(X)

    proba       = clf.predict_proba(X_scaled)[0]
    pred_idx    = np.argmax(proba)
    pred_label  = clf.classes_[pred_idx]
    pred_tone   = TONE_NAMES.get(pred_label, str(pred_label))

    # Build confidence dict for all tones
    confidences = {}
    for cls, prob in zip(clf.classes_, proba):
        tone_name = TONE_NAMES.get(cls, str(cls))
        confidences[tone_name] = round(float(prob), 4)

    return {
        "predicted_tone": pred_tone,
        "confidence"    : round(float(proba[pred_idx]), 4),
        "all_confidences": confidences
    }


# ── Visualization ─────────────────────────────────────────────────────────────

TONE_COLORS = {
    "ngang": "#4C72B0",
    "huyen": "#55A868",
    "sac"  : "#C44E52",
    "hoi"  : "#8172B2",
    "nga"  : "#CCB974",
    "nang" : "#64B5CD",
}


def plot_results(hidden: np.ndarray, X_train: np.ndarray, y_train: np.ndarray,
                 best_layer: int, audio_path: str):
    """
    Generate two plots:
      1. Layer-wise confidence for each tone
      2. Bar chart of tone confidences at the best layer
    """
    n_layers   = hidden.shape[0]
    tone_names = [TONE_NAMES[i] for i in range(len(TONE_NAMES))
                  if i in TONE_NAMES]

    # Collect confidence per layer per tone
    layer_confidences = {t: [] for t in tone_names}

    for layer_idx in range(n_layers):
        clf, scaler = fit_probe(X_train, y_train, layer_idx)
        result = predict_tone(hidden, clf, scaler, layer_idx)
        for tone in tone_names:
            conf = result["all_confidences"].get(tone, 0.0)
            layer_confidences[tone].append(conf)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Layer-wise confidence curves
    for tone, confs in layer_confidences.items():
        color = TONE_COLORS.get(tone, "gray")
        ax1.plot(range(n_layers), confs, marker="o", linewidth=2,
                 color=color, label=tone)

    ax1.axvline(best_layer, linestyle="--", color="black", linewidth=1,
                label=f"Best layer ({best_layer})")
    ax1.set_xlabel("Encoder Layer")
    ax1.set_ylabel("Probe Confidence")
    ax1.set_title("Tone Confidence Across Layers")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Plot 2: Confidence bar chart at best layer
    clf, scaler = fit_probe(X_train, y_train, best_layer)
    result      = predict_tone(hidden, clf, scaler, best_layer)
    confs       = result["all_confidences"]
    tones       = list(confs.keys())
    values      = list(confs.values())
    colors      = [TONE_COLORS.get(t, "gray") for t in tones]

    bars = ax2.bar(tones, values, color=colors, edgecolor="white")
    ax2.set_ylabel("Confidence")
    ax2.set_title(f"Tone Confidence at Layer {best_layer}\n"
                  f"Prediction: {result['predicted_tone'].upper()} "
                  f"({result['confidence']:.1%})")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Vietnamese Tone Analysis\n{os.path.basename(audio_path)}",
                 fontsize=12)
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "infer_result.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved -> {out}")
    return layer_confidences


def get_best_layer() -> int:
    """Read best layer from saved probe results."""
    probe_path = os.path.join(RESULTS_DIR, "probe_results.json")
    if not os.path.exists(probe_path):
        print("  No probe_results.json found, defaulting to layer 12.")
        return 12
    with open(probe_path, encoding="utf-8") as f:
        data = json.load(f)
    best = max(data["layer_results"], key=lambda r: r["accuracy"])
    return best["layer"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict Vietnamese tone from your own audio."
    )
    parser.add_argument("--audio",  type=str, default=None,
                        help="Path to audio file (.wav, .mp3, .flac, .m4a)")
    parser.add_argument("--record", action="store_true",
                        help="Record from microphone instead of loading a file")
    parser.add_argument("--duration", type=int, default=5,
                        help="Recording duration in seconds (default: 5)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Encoder layer to use for prediction (default: best layer)")
    args = parser.parse_args()

    if not args.audio and not args.record:
        parser.print_help()
        print("\nExample: python 08_infer.py --audio my_audio.wav")
        sys.exit(0)

    # ── Load audio ────────────────────────────────────────────────────────────
    print("\n── Loading Audio ────────────────────────────────")
    if args.record:
        audio = record_audio(args.duration)
        audio_path = "microphone_recording"
    else:
        if not os.path.exists(args.audio):
            print(f"  ERROR: File not found: {args.audio}")
            sys.exit(1)
        audio = load_audio(args.audio)
        audio_path = args.audio

    # ── Load Whisper ──────────────────────────────────────────────────────────
    print(f"\n── Loading Whisper ({WHISPER_MODEL}) on {DEVICE} ──")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model     = WhisperModel.from_pretrained(WHISPER_MODEL).to(DEVICE)
    model.eval()

    # ── Extract hidden states ─────────────────────────────────────────────────
    print("\n── Extracting Hidden States ─────────────────────")
    hidden = extract_hidden_states(audio, processor, model)
    print(f"  Hidden states shape: {hidden.shape}  (layers x hidden_dim)")

    # ── Load training data ────────────────────────────────────────────────────
    print("\n── Loading Training Data ────────────────────────")
    X_train, y_train = load_training_data()
    print(f"  Training samples: {len(y_train)}")

    # ── Determine best layer ──────────────────────────────────────────────────
    best_layer = args.layer if args.layer is not None else get_best_layer()
    print(f"  Using layer: {best_layer}")

    # ── Predict ───────────────────────────────────────────────────────────────
    print("\n── Prediction ───────────────────────────────────")
    clf, scaler = fit_probe(X_train, y_train, best_layer)
    result      = predict_tone(hidden, clf, scaler, best_layer)

    print(f"\n  Predicted tone : {result['predicted_tone'].upper()}")
    print(f"  Confidence     : {result['confidence']:.1%}")
    print(f"\n  All tone confidences (Layer {best_layer}):")
    for tone, conf in sorted(result["all_confidences"].items(),
                             key=lambda x: x[1], reverse=True):
        bar   = "█" * int(conf * 30)
        print(f"    {tone:<8} {conf:.3f}  {bar}")

    # ── Tone description ──────────────────────────────────────────────────────
    TONE_DESCRIPTIONS = {
        "ngang": "Level tone — mid flat pitch, modal phonation",
        "huyen": "Grave tone — low falling pitch, breathy phonation",
        "sac"  : "Acute tone — high rising pitch, modal phonation",
        "hoi"  : "Hook tone — dipping contour, creaky phonation",
        "nga"  : "Tilde tone — rising-glottalized contour, creaky",
        "nang" : "Dot tone — low falling, creaky/stopped phonation",
    }
    desc = TONE_DESCRIPTIONS.get(result["predicted_tone"], "")
    if desc:
        print(f"\n  Description: {desc}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n── Generating Plots ─────────────────────────────")
    plot_results(hidden, X_train, y_train, best_layer, audio_path)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
