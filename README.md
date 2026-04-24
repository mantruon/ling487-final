# Probing Whisper for Vietnamese Tone Representations

## Project Structure

```
whisper-probe/
├── README.md
├── .gitignore                 # Excludes data/, caches, and OS files from GitHub
├── requirements.txt           # Python dependencies (torch installed separately)
├── setup.py                   # Cross-platform installer (Mac + Windows)
├── config.py                  # Shared constants, paths, and device detection
├── 01_download_data.py        # Download and cache FLEURS Vietnamese dataset
├── 02_label_tones.py          # Parse diacritics → tone labels, save labeled dataset
├── 03_extract_hidden_states.py # Run Whisper encoder, save hidden states per layer
├── 04_probe.py                # Train logistic regression probes per layer
├── 05_baseline.py             # Acoustic baseline (F0 + MFCCs via librosa)
├── 06_visualize.py            # Layer-wise accuracy plots + UMAP/t-SNE
└── data/                      # ⚠️ Auto-created locally, NOT pushed to GitHub
    ├── labeled_dataset.json
    ├── hidden_states/
    └── results/
```

## Setup

### 1. Clone the repo
```bash
git clone git@github.com:mantruon/ling487-final.git
cd ling487-final
```

### 2. Create a virtual environment (recommended)

**Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Run setup (auto-detects Mac MPS vs. Windows CUDA vs. CPU)
```bash
python setup.py
```

This will:
- Install PyTorch with MPS support on Apple Silicon Macs
- Install PyTorch with CUDA on Windows machines with an NVIDIA GPU
- Fall back to CPU-only PyTorch otherwise
- Install all remaining dependencies from requirements.txt

## Run Order

```bash
python 01_download_data.py
python 02_label_tones.py
python 03_extract_hidden_states.py
python 04_probe.py
python 05_baseline.py
python 06_visualize.py
```

## What Gets Pushed to GitHub

Only source code is tracked. The `data/` folder (hidden states, results, plots)
is excluded via `.gitignore` because hidden state `.npy` files can be several GB.
Regenerate locally by running the scripts in order.

## Notes
- Whisper `small` is used by default. Change in `config.py`.
- FLEURS Vietnamese (`vi_vn`) is the data source.
- Hidden states are saved to disk so you don't re-run the encoder each time.
- The tone labeler works on Northern Vietnamese diacritics (6-tone system).
- Start with `MAX_SAMPLES = 200` in `config.py` to test the pipeline quickly.
