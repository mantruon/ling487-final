# 02_label_tones.py
# Parses Vietnamese diacritics from FLEURS transcriptions and assigns
# one tone label (0–5) per audio clip. Saves labeled_dataset.json.
#
# Strategy: decompose each character with unicodedata NFD normalization,
# then check for combining diacritic marks that encode tone.
# If a transcription contains multiple syllables, the FIRST toned syllable
# is used as the clip's label (you can refine this later).
#
# Usage: python 02_label_tones.py

import json
import unicodedata
from datasets import load_dataset, Audio
from tqdm import tqdm
from config import (
    DATASET_NAME, DATASET_LANG, DATASET_SPLIT, MAX_SAMPLES,
    TONE_DIACRITICS, TONE_LABELS, LABELED_PATH
)


def extract_tone(text: str) -> str:
    """
    Return the tone name of the first toned syllable found in `text`.
    Falls back to 'ngang' (level, no mark) if none found.
    """
    # Normalize to NFD so combining diacritics are separate code points
    nfd = unicodedata.normalize("NFD", text)
    for char in nfd:
        if char in TONE_DIACRITICS:
            return TONE_DIACRITICS[char]
    return "ngang"   # default: level tone (no diacritic)


def label_dataset(ds) -> list[dict]:
    """
    Iterate over dataset, extract tone label, return list of records.
    Each record stores the index, transcription, tone name, and tone id.
    """
    records = []
    tone_counts = {name: 0 for name in TONE_LABELS}

    for i, example in enumerate(tqdm(ds, desc="Labeling tones")):
        transcription = example.get("transcription") or example.get("sentence", "")
        tone_name     = extract_tone(transcription)
        tone_id       = TONE_LABELS[tone_name]

        records.append({
            "index"        : i,
            "transcription": transcription,
            "tone_name"    : tone_name,
            "tone_id"      : tone_id,
        })
        tone_counts[tone_name] += 1

    return records, tone_counts


def main():
    print(f"Loading {DATASET_NAME} ({DATASET_LANG}, {DATASET_SPLIT})…")
    args = [DATASET_NAME] + ([DATASET_LANG] if DATASET_LANG else [])
    ds = load_dataset(*args, split=DATASET_SPLIT)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000, decode=True))

    if MAX_SAMPLES is not None:
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

    records, tone_counts = label_dataset(ds)

    # Save to JSON
    with open(LABELED_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(records)} labeled samples → {LABELED_PATH}")
    print("\n── Tone distribution ────────────────────────")
    for tone, count in tone_counts.items():
        bar = "█" * (count // 2)
        print(f"  {tone:<8} ({TONE_LABELS[tone]}): {count:>4}  {bar}")

    # Quick sanity check
    print("\n── Sample labels ────────────────────────────")
    for r in records[:5]:
        print(f"  [{r['tone_id']}] {r['tone_name']:<8}  '{r['transcription'][:40]}'")

if __name__ == "__main__":
    main()