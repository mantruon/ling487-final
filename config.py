# config.py
# Shared constants and paths for the entire project.
# Edit these values to change model size, dataset split, etc.
# Automatically detects platform: Mac (MPS), Windows/Linux (CUDA or CPU).

HF_HUB_DISABLE_SYMLINKS_WARNING=1 # gets rid of Hugging Face cache warning on Windows

import os
import sys
import platform
import torch

# Suppress Windows symlinks warning from Hugging Face cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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
DATASET_NAME   = "doof-ferb/LSVSC"
DATASET_LANG   = None
DATASET_SPLIT  = "train"                  # "train" | "validation" | "test"
MAX_SAMPLES    = 500                      # Set to None to use the full split

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR            = "data"
LABELED_PATH        = os.path.join(DATA_DIR, "labeled_dataset.json")
HIDDEN_STATES_DIR   = os.path.join(DATA_DIR, "hidden_states")
RESULTS_DIR         = os.path.join(DATA_DIR, "results")

for _dir in [DATA_DIR, HIDDEN_STATES_DIR, RESULTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Probe hyperparameters (tune these to experiment) ─────────────────────────
#
# Each time you change these and re-run 04_probe.py, a new timestamped report
# is saved to data/results/runs/ so you can compare runs side by side.
#
PROBE_CONFIG = {
    # ── Probe classifier ──────────────────────────────────────────────────────
    "C"           : 1.0,       # Regularization strength. Lower = stronger regularization
                               # (simpler probe, less overfitting). Try: 0.01, 0.1, 1.0, 10.0

    "max_iter"    : 1000,      # Max iterations for logistic regression solver.
                               # Increase if you see convergence warnings.

    "solver"      : "lbfgs",   # Optimization algorithm. "lbfgs" is best for small-medium
                               # data. Alternatives: "saga" (faster on large data),
                               # "newton-cg", "sag"

    "probe_type"  : "linear",  # "linear" = logistic regression (standard, interpretable)
                               # "mlp"    = small neural probe (more expressive, less
                               #            interpretable — use to check if linear is enough)

    # ── MLP probe settings (only used when probe_type = "mlp") ───────────────
    "mlp_hidden"  : 128,       # Hidden layer size for MLP probe. Try: 64, 128, 256
    "mlp_layers"  : 1,         # Number of hidden layers. Try: 1, 2

    # ── Data split ────────────────────────────────────────────────────────────
    "test_size"   : 0.2,       # Fraction of data held out for testing.
                               # Try: 0.1 (more training data), 0.3 (stricter test)

    "random_seed" : 42,        # Controls train/test split and classifier init.
                               # Change to check result stability across splits.

    # ── Pooling strategy ──────────────────────────────────────────────────────
    "pooling"     : "mean",    # How to collapse the time dimension of hidden states.
                               # "mean" = average all frames (default)
                               # "max"  = max over frames (captures peaks)
                               # "first" = first frame only (like [CLS] token)

    # ── Run metadata (shown in report, helps you remember what you tested) ────
    "run_label"   : "baseline",  # Short name for this run. Change for each experiment,
                                 # e.g. "C=0.1", "mlp_probe", "max_pooling", "seed=7"
    "notes"       : "",          # Optional free-text notes about this run.
}

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