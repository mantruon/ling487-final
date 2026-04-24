# config.py
# Shared constants and paths for the entire project.
# Edit these values to change model size, dataset split, etc.
# Automatically detects platform: Mac (MPS), Windows/Linux (CUDA or CPU).

import os
import sys
import platform
import torch

# ── Platform detection ────────────────────────────────────────────────────────
SYSTEM = platform.system()   # "Darwin" = Mac, "Windows" = Windows, "Linux" = Linux

def get_device() -> str:
    """
    Picks the best available compute device:
      - Mac Apple Silicon → 'mps'   (Metal Performance Shaders, M1/M2/M3/M4)
      - Windows/Linux GPU → 'cuda'
      - Fallback          → 'cpu'
    """
    if SYSTEM == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
print(f"[config] Platform: {SYSTEM} | Device: {DEVICE} | Python: {sys.version.split()[0]}")

# ── Model ─────────────────────────────────────────────────────────────────────
WHISPER_MODEL  = "openai/whisper-small"   # options: tiny, base, small, medium, large-v3
SAMPLE_RATE    = 16_000                   # Whisper expects 16 kHz audio

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_NAME   = "google/fleurs"
DATASET_LANG   = "vi_vn"
DATASET_SPLIT  = "train"                  # "train" | "validation" | "test"
MAX_SAMPLES    = 500                      # Set to None to use the full split

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR            = "data"
LABELED_PATH        = os.path.join(DATA_DIR, "labeled_dataset.json")
HIDDEN_STATES_DIR   = os.path.join(DATA_DIR, "hidden_states")
RESULTS_DIR         = os.path.join(DATA_DIR, "results")

for _dir in [DATA_DIR, HIDDEN_STATES_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Vietnamese tone map ───────────────────────────────────────────────────────
# Maps Unicode combining diacritics / precomposed vowel marks → tone label.
# Northern Vietnamese 6-tone system.
TONE_LABELS = {
    "ngang": 0,   # level (no mark)
    "huyen": 1,   # grave  ◌̀
    "sac"  : 2,   # acute  ◌́
    "hoi"  : 3,   # hook above ◌̉
    "nga"  : 4,   # tilde  ◌̃
    "nang" : 5,   # dot below ◌̣
}
TONE_NAMES = {v: k for k, v in TONE_LABELS.items()}   # reverse map

# Diacritic characters that signal each tone (used in 02_label_tones.py)
TONE_DIACRITICS = {
    "\u0300": "huyen",   # combining grave accent
    "\u0301": "sac",     # combining acute accent
    "\u0309": "hoi",     # combining hook above
    "\u0303": "nga",     # combining tilde
    "\u0323": "nang",    # combining dot below
    # Also handle precomposed characters via unicodedata normalization
}
