"""
LaMuN dataset analysis:
  1) Unique news_source values per language.
  2) For each language test set, run Davlan/xlm-roberta-base-ner-hrl on the
     human caption and count how many examples qualify as the "hard" subset
     (>= 4 named entities of type PER / ORG / LOC).

Outputs:
  - news_sources_per_language.json   (full list per language)
  - news_sources_summary.csv         (counts per language)
  - hard_test_set_counts.csv         (hard-set size + ratio per language)
  - data/{lang}/hard_test_set.json   (per-language hard subset, optional)
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

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

# Davlan/xlm-roberta-base-ner-hrl emits PER / ORG / LOC.
HARD_THRESHOLD = 4
NER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"


# ---------- Task 1: unique news sources ----------

def collect_news_sources(languages, output_dir: Path, splits=("train", "test")):
    """For each language, list unique values of the `news_source` column."""
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

    out_json = output_dir / "news_sources_per_language.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(per_language, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_json}")

    out_csv = output_dir / "news_sources_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["language", "num_unique_sources", "total_rows_with_source"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {out_csv}")

    return per_language


# ---------- Task 2: hard test set via XLM-R NER ----------

def build_hard_test_sets(
    languages,
    output_dir: Path,
    save_per_language: bool = True,
    threshold: int = HARD_THRESHOLD,
    batch_size: int = 16,
):
    """Run NER on each test set's `caption` column and flag rows with
    >= `threshold` named entities as the hard subset."""
    print(f"\nLoading NER model: {NER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Running on {'CUDA' if device == 0 else 'CPU'}")

    nlp = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",  # merges B-/I- spans into single entities
        device=device,
        batch_size=batch_size,
    )

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

        captions = [(ex.get("caption") or "").strip() for ex in ds]
        captions_safe = [c if c else " " for c in captions]

        all_entities = []
        for start in tqdm(
            range(0, len(captions_safe), batch_size),
            desc=f"NER {lang}",
        ):
            batch = captions_safe[start : start + batch_size]
            try:
                results = nlp(batch)
            except Exception as e:
                print(f"  [{lang} batch {start}] failed ({e}); per-example fallback")
                results = []
                for c in batch:
                    try:
                        results.append(nlp(c))
                    except Exception as inner:
                        print(f"    skipped one: {inner}")
                        results.append([])
            # When a list is passed, pipeline returns a list of lists.
            # When a single string is passed (fallback), it returns one list.
            if results and not isinstance(results[0], list):
                results = [results]
            all_entities.extend(results)

        assert len(all_entities) == len(captions), \
            f"{lang}: length mismatch {len(all_entities)} vs {len(captions)}"

        hard_rows = []
        entity_counts = []
        for idx, (caption, ents) in enumerate(zip(captions, all_entities)):
            n_ents = len(ents)
            entity_counts.append(n_ents)
            if n_ents >= threshold:
                hard_rows.append({
                    "index": idx,
                    "caption": caption,
                    "num_entities": n_ents,
                    "entities": [
                        {
                            "text": e["word"],
                            "label": e["entity_group"],
                            "score": float(e["score"]),
                        }
                        for e in ents
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
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the NER pipeline.",
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
        print(f"NER model: {NER_MODEL}")
        print("=" * 70)
        build_hard_test_sets(
            args.languages,
            output_dir,
            save_per_language=not args.no_per_language_files,
            threshold=args.threshold,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()