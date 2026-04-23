# Probing Whisper for Vietnamese Tone Representations

## Project Structure

```
whisper-probe/
├── README.md
├── requirements.txt
├── config.py                  # Shared constants and paths
├── 01_download_data.py        # Download and cache FLEURS Vietnamese dataset
├── 02_label_tones.py          # Parse diacritics → tone labels, save labeled dataset
├── 03_extract_hidden_states.py # Run Whisper encoder, save hidden states per layer
├── 04_probe.py                # Train logistic regression probes per layer
├── 05_baseline.py             # Acoustic baseline (F0 + MFCCs via librosa)
├── 06_visualize.py            # Layer-wise accuracy plots + UMAP/t-SNE
└── data/                      # Auto-created by scripts
    ├── labeled_dataset.json
    ├── hidden_states/
    └── results/
```

## Setup

```bash
pip install -r requirements.txt
```

## Run Order

```bash
python 01_download_data.py
python 02_label_tones.py
python 03_extract_hidden_states.py
python 04_probe.py
python 05_baseline.py
python 06_visualize.py
```

## Notes
- Whisper `small` is used by default. Change in `config.py`.
- FLEURS Vietnamese (`vi_vn`) is the data source.
- Hidden states are saved to disk so you don't re-run the encoder each time.
- The tone labeler works on Northern Vietnamese diacritics.
