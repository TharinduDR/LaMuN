"""
Build the Hindi (hin) hard test set for LaMuN.

Uses MichaelHuang/muril_base_cased_hindi_ner. The model emits 11 entity types,
but per the user's request we ONLY count PERSON and LOCATION (mirroring the
PER/LOC scope of the Bengali model — ORG is dropped here because the Bengali
model has it and this one does too, but the user asked for PER/LOC only).

A caption is "hard" if it contains >= 4 named entities (PERSON or LOCATION),
where each entity is a single merged BIO span (B-X followed by zero or more
I-X tokens collapses to one entity).

Outputs:
  - data/hin/hard_test_set.json
  - hindi_hard_summary.json
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

DATASET_NAME = "tharindu/LaMuN"
LANG = "hin"
NER_MODEL = "MichaelHuang/muril_base_cased_hindi_ner"
TOKENIZER_NAME = "google/muril-base-cased"
HARD_THRESHOLD = 4
MAX_LEN = 128

# Full label set the model emits
LABELS_DICT = {
    0: "B-FESTIVAL", 1: "B-GAME", 2: "B-LANGUAGE", 3: "B-LITERATURE",
    4: "B-LOCATION", 5: "B-MISC", 6: "B-NUMEX", 7: "B-ORGANIZATION",
    8: "B-PERSON", 9: "B-RELIGION", 10: "B-TIMEX",
    11: "I-FESTIVAL", 12: "I-GAME", 13: "I-LANGUAGE", 14: "I-LITERATURE",
    15: "I-LOCATION", 16: "I-MISC", 17: "I-NUMEX", 18: "I-ORGANIZATION",
    19: "I-PERSON", 20: "I-RELIGION", 21: "I-TIMEX", 22: "O",
}

# We only count these entity types (PERSON + LOCATION, matching PER/LOC scope)
COUNTED_TYPES = {"PERSON", "LOCATION", "ORGANIZATION"}


def merge_bio_spans(tokens, labels):
    """Collapse BIO tag sequences into merged entity spans.

    Returns a list of dicts: {"text": str, "label": str}
    Only entity types in COUNTED_TYPES are kept.

    Handles WordPiece subwords: tokens starting with '##' are stitched onto
    the previous token without a space. Special tokens ([CLS], [SEP], [PAD])
    are filtered out by the caller.
    """
    spans = []
    cur_label = None      # entity type currently being built, or None
    cur_tokens = []       # token pieces collected for the current span

    def flush():
        nonlocal cur_label, cur_tokens
        if cur_label is not None and cur_tokens and cur_label in COUNTED_TYPES:
            text = ""
            for t in cur_tokens:
                if t.startswith("##"):
                    text += t[2:]
                else:
                    text += (" " if text else "") + t
            spans.append({"text": text, "label": cur_label})
        cur_label = None
        cur_tokens = []

    for tok, lab in zip(tokens, labels):
        if lab == "O":
            flush()
            continue
        # lab is "B-X" or "I-X"
        prefix, _, etype = lab.partition("-")
        if prefix == "B":
            flush()
            cur_label = etype
            cur_tokens = [tok]
        elif prefix == "I":
            if cur_label == etype:
                cur_tokens.append(tok)
            else:
                # I- tag without matching B- — treat as a new span (lenient).
                flush()
                cur_label = etype
                cur_tokens = [tok]
        else:
            flush()

    flush()
    return spans


def predict_entities(captions, model, tokenizer, device, batch_size=16):
    """Run NER on a list of captions; return list of entity-span lists."""
    SPECIALS = set(tokenizer.all_special_tokens)
    all_spans = []

    for start in tqdm(range(0, len(captions), batch_size), desc="NER batches"):
        batch = captions[start : start + batch_size]
        batch_safe = [c if c.strip() else " " for c in batch]

        enc = tokenizer(
            batch_safe,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids = torch.argmax(logits, dim=2).cpu().tolist()
        input_ids = enc["input_ids"].cpu().tolist()
        attention = enc["attention_mask"].cpu().tolist()

        for ids, preds, mask in zip(input_ids, pred_ids, attention):
            tokens = tokenizer.convert_ids_to_tokens(ids)
            # Filter out special tokens and padding
            filt_tokens, filt_labels = [], []
            for tok, p, m in zip(tokens, preds, mask):
                if m == 0:
                    continue
                if tok in SPECIALS:
                    continue
                filt_tokens.append(tok)
                filt_labels.append(LABELS_DICT[p])
            spans = merge_bio_spans(filt_tokens, filt_labels)
            all_spans.append(spans)

    return all_spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--threshold", type=int, default=HARD_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    lang_dir = output_dir / "data" / LANG
    lang_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Loading NER model: {NER_MODEL}")
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Running on {device}")

    print(f"\nLoading {DATASET_NAME} / {LANG} test split...")
    ds = load_dataset(DATASET_NAME, LANG, split="test")
    print(f"Test size: {len(ds)}")

    if "caption" not in ds.column_names:
        raise RuntimeError(f"'caption' column missing — found {ds.column_names}")

    captions = [(ex.get("caption") or "").strip() for ex in ds]

    print(f"\nRunning NER on {len(captions)} captions "
          f"(counting only {sorted(COUNTED_TYPES)})...")
    all_entities = predict_entities(
        captions, model, tokenizer, device, batch_size=args.batch_size
    )
    assert len(all_entities) == len(captions)

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
                "entities": ents,
            })

    total = len(captions)
    hard_n = len(hard_rows)
    ratio = hard_n / total if total else 0.0
    avg_ents = sum(entity_counts) / len(entity_counts) if entity_counts else 0.0

    summary = {
        "language": LANG,
        "ner_model": NER_MODEL,
        "counted_entity_types": sorted(COUNTED_TYPES),
        "test_size": total,
        "hard_test_size": hard_n,
        "hard_ratio": round(ratio, 4),
        "avg_entities_per_caption": round(avg_ents, 3),
        "threshold": args.threshold,
    }

    hard_path = lang_dir / "hard_test_set.json"
    with hard_path.open("w", encoding="utf-8") as f:
        json.dump(hard_rows, f, ensure_ascii=False, indent=2)

    summary_path = output_dir / "hindi_hard_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Hindi hard test set summary")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {hard_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()