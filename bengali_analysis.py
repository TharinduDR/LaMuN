"""
Build the Bengali (ben) hard test set for LaMuN.

Runs sagorsarker/mbert-bengali-ner on each test caption and flags rows with
>= 4 named entities (PER / ORG / LOC) as the hard subset.

Outputs:
  - data/ben/hard_test_set.json    full hard rows with entity spans
  - bengali_hard_summary.json      counts + ratio (one-language version of the CSV)
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

DATASET_NAME = "tharindu/LaMuN"
LANG = "ben"
NER_MODEL = "sagorsarker/mbert-bengali-ner"
HARD_THRESHOLD = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--threshold", type=int, default=HARD_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    lang_dir = output_dir / "data" / LANG
    lang_dir.mkdir(parents=True, exist_ok=True)

    # Load NER pipeline
    print(f"Loading NER model: {NER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Running on {'CUDA' if device == 0 else 'CPU'}")

    nlp = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # modern replacement for grouped_entities=True
        device=device,
        batch_size=args.batch_size,
    )

    # Load Bengali test set
    print(f"\nLoading {DATASET_NAME} / {LANG} test split...")
    ds = load_dataset(DATASET_NAME, LANG, split="test")
    print(f"Test size: {len(ds)}")

    if "caption" not in ds.column_names:
        raise RuntimeError(f"'caption' column missing — found {ds.column_names}")

    captions = [(ex.get("caption") or "").strip() for ex in ds]

    # Run NER (pipeline handles batching internally)
    print(f"\nRunning NER on {len(captions)} captions...")
    all_entities = []
    for i in tqdm(range(0, len(captions), args.batch_size), desc="NER batches"):
        batch = captions[i : i + args.batch_size]
        # Pipeline returns [] for empty strings; guard anyway
        batch_safe = [c if c else " " for c in batch]
        try:
            results = nlp(batch_safe)
        except Exception as e:
            print(f"  Batch {i} failed ({e}); falling back to per-example")
            results = []
            for c in batch_safe:
                try:
                    results.append(nlp(c))
                except Exception as inner:
                    print(f"    skipped one: {inner}")
                    results.append([])
        # pipeline returns a list-of-lists for a list input
        if batch_safe and isinstance(results, list) and results and not isinstance(results[0], list):
            # Single-example shape — wrap
            results = [results]
        all_entities.extend(results)

    assert len(all_entities) == len(captions), \
        f"length mismatch: {len(all_entities)} vs {len(captions)}"

    # Build hard set
    hard_rows = []
    entity_counts = []
    for idx, (caption, ents) in enumerate(zip(captions, all_entities)):
        n = len(ents)
        entity_counts.append(n)
        if n >= args.threshold:
            hard_rows.append({
                "index": idx,
                "caption": caption,
                "num_entities": n,
                "entities": [
                    {
                        "text": e["word"],
                        "label": e["entity_group"],
                        "score": float(e["score"]),
                    }
                    for e in ents
                ],
            })

    total = len(captions)
    hard_n = len(hard_rows)
    ratio = hard_n / total if total else 0.0
    avg_ents = sum(entity_counts) / len(entity_counts) if entity_counts else 0.0

    summary = {
        "language": LANG,
        "ner_model": NER_MODEL,
        "test_size": total,
        "hard_test_size": hard_n,
        "hard_ratio": round(ratio, 4),
        "avg_entities_per_caption": round(avg_ents, 3),
        "threshold": args.threshold,
    }

    # Write outputs
    hard_path = lang_dir / "hard_test_set.json"
    with hard_path.open("w", encoding="utf-8") as f:
        json.dump(hard_rows, f, ensure_ascii=False, indent=2)

    summary_path = output_dir / "bengali_hard_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Bengali hard test set summary")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {hard_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()