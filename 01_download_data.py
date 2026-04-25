# 01_download_data.py
# Downloads and caches the FLEURS Vietnamese dataset from Hugging Face.
# Run this first to verify your internet connection and HF setup.
#
# Usage: python 01_download_data.py

from datasets import load_dataset
from config import DATASET_NAME, DATASET_LANG, DATASET_SPLIT, MAX_SAMPLES

def main():
    print(f"Loading dataset: {DATASET_NAME} | lang: {DATASET_LANG} | split: {DATASET_SPLIT}")

    args = [DATASET_NAME] + ([DATASET_LANG] if DATASET_LANG else [])
    ds = load_dataset(*args, split=DATASET_SPLIT)

    if MAX_SAMPLES is not None:
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

    print(f"\n✅ Loaded {len(ds)} samples.")
    print(f"   Columns: {ds.column_names}")

    # Preview a single example
    # Common Voice uses 'sentence', FLEURS uses 'transcription'
    example = ds[0]
    text_col = "sentence" if "sentence" in example else "transcription"
    print(f"\n── Example ──────────────────────────────────")
    print(f"  text          : {example[text_col]}")
    print(f"  columns       : {ds.column_names}")
    print(f"  audio keys    : {list(example['audio'].keys())}")
    print(f"  sample_rate   : {example['audio']['sampling_rate']} Hz")
    print(f"  audio length  : {len(example['audio']['array'])} samples "
          f"({len(example['audio']['array']) / example['audio']['sampling_rate']:.2f}s)")

    # Save to HF cache (automatic) — just confirming here
    print("\nDataset is cached by Hugging Face. Proceed to 02_label_tones.py")

if __name__ == "__main__":
    main()
