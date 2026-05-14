# 03_extract_hidden_states.py
# Runs each audio clip through the Whisper encoder and saves the
# mean-pooled hidden state for every layer to disk as .npy files.
#
# Mean pooling: average across the time dimension so each clip becomes
# a single vector per layer → (num_layers+1, hidden_dim)
#
# Output: data/hidden_states/{index}.npy  — shape (num_layers+1, hidden_dim)
#         data/hidden_states/meta.json    — maps index → tone_id
#
# Usage: python 03_extract_hidden_states.py

import json
import os
import numpy as np
import torch
from datasets import load_dataset, Audio
from transformers import WhisperModel, WhisperProcessor
from tqdm import tqdm
from config import (
    DATASET_NAME, DATASET_LANG, DATASET_SPLIT, MAX_SAMPLES,
    WHISPER_MODEL, SAMPLE_RATE, LABELED_PATH, HIDDEN_STATES_DIR, DEVICE
)


def load_labeled_indices(labeled_path: str) -> dict[int, int]:
    """Return {index: tone_id} from the labeled dataset JSON."""
    with open(labeled_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["index"]: r["tone_id"] for r in records}


def mean_pool(hidden_states: tuple) -> np.ndarray:
    """
    hidden_states: tuple of (1, seq_len, hidden_dim) tensors, one per layer.
    Returns: np.ndarray of shape (num_layers, hidden_dim)
    """
    pooled = []
    for layer in hidden_states:
        # layer shape: (batch=1, seq_len, hidden_dim)
        pooled.append(layer.squeeze(0).mean(dim=0).cpu().numpy())
    return np.stack(pooled)   # (num_layers, hidden_dim)


def main():
    # ── Load labels ───────────────────────────────────────────────────────────
    if not os.path.exists(LABELED_PATH):
        raise FileNotFoundError(f"{LABELED_PATH} not found. Run 02_label_tones.py first.")

    index_to_tone = load_labeled_indices(LABELED_PATH)
    print(f"Loaded {len(index_to_tone)} labeled indices.")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print(f"Loading dataset...")
    args = [DATASET_NAME] + ([DATASET_LANG] if DATASET_LANG else [])
    ds = load_dataset(*args, split=DATASET_SPLIT)
    if MAX_SAMPLES is not None:
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    # Cast after selecting to avoid decoding all audio upfront
    ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))

    # ── Load Whisper ──────────────────────────────────────────────────────────
    print(f"Loading {WHISPER_MODEL} on {DEVICE}…")
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model     = WhisperModel.from_pretrained(WHISPER_MODEL).to(DEVICE)
    model.eval()

    num_layers = model.config.encoder_layers
    print(f"Encoder has {num_layers} layers (+1 embedding layer = {num_layers+1} total).")

    # ── Extract hidden states ─────────────────────────────────────────────────
    meta = {}   # index → tone_id, saved at the end

    for i, example in enumerate(tqdm(ds, desc="Extracting hidden states")):
        out_path = os.path.join(HIDDEN_STATES_DIR, f"{i}.npy")

        # Skip if already extracted (allows resuming)
        if os.path.exists(out_path):
            meta[i] = index_to_tone[i]
            continue

        # Resample if needed
        audio_array   = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]

        if sampling_rate != SAMPLE_RATE:
            import torchaudio
            waveform  = torch.tensor(audio_array).unsqueeze(0).float()
            resampler = torchaudio.transforms.Resample(sampling_rate, SAMPLE_RATE)
            audio_array = resampler(waveform).squeeze(0).numpy()

        # Preprocess → log-mel spectrogram
        inputs = processor(
            audio_array,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).input_features.to(DEVICE)

        # Forward pass through encoder
        with torch.no_grad():
            outputs = model.encoder(
                inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Mean-pool all layers → (num_layers+1, hidden_dim)
        pooled = mean_pool(outputs.hidden_states)   # includes embedding layer
        np.save(out_path, pooled)
        meta[i] = index_to_tone[i]

    # Save meta
    meta_path = os.path.join(HIDDEN_STATES_DIR, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print(f"\n✅ Saved hidden states for {len(meta)} clips → {HIDDEN_STATES_DIR}/")
    print(f"   Each file shape: ({num_layers+1}, {model.config.d_model})")
    print(f"   Meta saved → {meta_path}")


if __name__ == "__main__":
    main()
