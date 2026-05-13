"""
LaMuN dataset analysis:
  1) Unique news_source values per language.
  2) For each language test set, run GLiNER on the human caption and count how
     many examples qualify as the "hard" subset (>= 4 named entities).

Outputs:
  - news_sources_per_language.json   (full list per language)
  - news_sources_summary.csv         (counts per language)
  - hard_test_set_counts.csv         (hard-set size + ratio per language)
  - data/{lang}/hard_test_set.json   (per-language hard subset, optional)
"""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from gliner import GLiNER
from tqdm import tqdm

# ---------- Config ----------

DATASET_NAME = "tharindu/LaMuN"

LANGUAGES = [
    "sqi", "amh", "ara", "hye", "ben", "bos", "bul", "mya", "ckb", "cmn",
    "hrv", "prs", "eng", "fra", "kat", "deu", "guj", "hat", "hau", "hin",
    "ibo", "ind", "jpn", "khm", "kin", "kor", "kmr", "kir", "lao", "lin",
    "mkd", "mar", "nep", "pcm", "nde", "orm", "pus", "fas", "pol", "pan",
    "ron", "rus", "gla", "srp", "sna", "sin", "som", "swa", "tel", "tha",
    "bod", "tir", "tur", "ukr", "urd", "uzb", "vie", "cym", "yor",
]

NER_LABELS = ["person", "organization", "location", "date", "event"]
HARD_THRESHOLD = 4
NER_MODEL = "urchade/gliner_multi-v2.1"


# ---------- Task 1: unique news sources ----------

def collect_news_sources(languages, output_dir: Path, splits=("train", "test")):
    """For each language, list unique values of the `news_source` column.

    Looks across the requested splits (defaults to train + test) so the count
    reflects the full picture, not just one split.
    """
    per_language = {}
    summary_rows = []

    for lang in languages:
        try:
            ds = load_dataset(DATASET_NAME, lang)
        except Exception as e:
            print(f"[news_sources] Skipping {lang}: {e}")
            continue

        sources = Counter()
        for split in splits:
            if split not in ds:
                continue
            split_ds = ds[split]
            if "news_source" not in split_ds.column_names:
                print(f"[news_sources] {lang}: no 'news_source' column in {split}")
                continue
            for src in split_ds["news_source"]:
                if src is None:
                    continue
                sources[str(src).strip()] += 1

        per_language[lang] = {
            "num_unique": len(sources),
            "sources": dict(sources.most_common()),
        }
        summary_rows.append({
            "language": lang,
            "num_unique_sources": len(sources),
            "total_rows_with_source": sum(sources.values()),
        })
        print(f"[news_sources] {lang}: {len(sources)} unique sources")

    # Write JSON (full breakdown)
    out_json = output_dir / "news_sources_per_language.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(per_language, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_json}")

    # Write CSV (summary)
    out_csv = output_dir / "news_sources_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["language", "num_unique_sources", "total_rows_with_source"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {out_csv}")

    return per_language


# ---------- Task 2: hard test set via GLiNER ----------

def build_hard_test_sets(
    languages,
    output_dir: Path,
    save_per_language: bool = True,
    threshold: int = HARD_THRESHOLD,
):
    """Run GLiNER on each test set's `caption` column and flag rows with
    >= `threshold` named entities as the hard subset."""
    print(f"\nLoading GLiNER model: {NER_MODEL}")
    model = GLiNER.from_pretrained(NER_MODEL)

    # Use GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("GLiNER running on CUDA")
        else:
            print("GLiNER running on CPU (slow)")
    except Exception:
        pass

    summary_rows = []

    for lang in languages:
        try:
            ds = load_dataset(DATASET_NAME, lang, split="test")
        except Exception as e:
            print(f"[hard_set] Skipping {lang}: {e}")
            continue

        if "caption" not in ds.column_names:
            print(f"[hard_set] {lang}: no 'caption' column, skipping")
            continue

        hard_rows = []
        entity_counts = []

        for idx, example in enumerate(tqdm(ds, desc=f"NER {lang}")):
            caption = example.get("caption") or ""
            if not caption.strip():
                entity_counts.append(0)
                continue

            try:
                entities = model.predict_entities(caption, NER_LABELS)
            except Exception as e:
                print(f"  [{lang} idx={idx}] NER failed: {e}")
                entity_counts.append(0)
                continue

            n_ents = len(entities)
            entity_counts.append(n_ents)

            if n_ents >= threshold:
                hard_rows.append({
                    "index": idx,
                    "caption": caption,
                    "num_entities": n_ents,
                    "entities": [
                        {"text": e["text"], "label": e["label"]} for e in entities
                    ],
                })

        total = len(ds)
        hard_n = len(hard_rows)
        ratio = (hard_n / total) if total else 0.0
        avg_ents = (sum(entity_counts) / len(entity_counts)) if entity_counts else 0.0

        summary_rows.append({
            "language": lang,
            "test_size": total,
            "hard_test_size": hard_n,
            "hard_ratio": round(ratio, 4),
            "avg_entities_per_caption": round(avg_ents, 3),
        })
        print(
            f"[hard_set] {lang}: {hard_n}/{total} hard "
            f"({ratio:.1%}), avg ents={avg_ents:.2f}"
        )

        if save_per_language and hard_rows:
            lang_dir = output_dir / "data" / lang
            lang_dir.mkdir(parents=True, exist_ok=True)
            with (lang_dir / "hard_test_set.json").open("w", encoding="utf-8") as f:
                json.dump(hard_rows, f, ensure_ascii=False, indent=2)

    # Write CSV summary
    out_csv = output_dir / "hard_test_set_counts.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "language",
                "test_size",
                "hard_test_size",
                "hard_ratio",
                "avg_entities_per_caption",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote {out_csv}")

    return summary_rows


# ---------- Entry point ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["sources", "hard", "all"],
        default="all",
        help="Which task to run.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=LANGUAGES,
        help="Subset of language codes to process (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=HARD_THRESHOLD,
        help="Minimum named entities for 'hard' classification (default: 4).",
    )
    parser.add_argument(
        "--no-per-language-files",
        action="store_true",
        help="Skip writing data/{lang}/hard_test_set.json files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task in ("sources", "all"):
        print("=" * 70)
        print("TASK 1: Unique news sources per language")
        print("=" * 70)
        collect_news_sources(args.languages, output_dir)

    if args.task in ("hard", "all"):
        print("\n" + "=" * 70)
        print(f"TASK 2: Hard test set (>= {args.threshold} entities) per language")
        print("=" * 70)
        build_hard_test_sets(
            args.languages,
            output_dir,
            save_per_language=not args.no_per_language_files,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
