"""
Build the Nepali (nep) hard test set for LaMuN.

Uses a Nepali token-classification model with the label schema:
  O, B-Location, I-Location, B-Person, I-Person, B-Organization,
  I-Organization, B-Event, I-Event, B-Date, I-Date

Per request, we count only PERSON and LOCATION entities (matching the Hindi
scope). A caption is "hard" if it contains >= 4 such merged BIO spans.

Set NER_MODEL below to the actual HF repo before running.

Outputs:
  - data/nep/hard_test_set.json
  - nepali_hard_summary.json
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

DATASET_NAME = "tharindu/LaMuN"
LANG = "nep"

# TODO: set this to the actual model repo.
NER_MODEL = "Saugatkafley/mbert-Nepali-NER"  # placeholder — replace with your model id
TOKENIZER_NAME = NER_MODEL          # change if tokenizer differs from model

HARD_THRESHOLD = 4
MAX_LEN = 128

# Label schema exactly as given.
LABELS_DICT = {
    0: "O",
    1: "B-Location",
    2: "I-Organization",
    3: "B-Person",
    4: "B-Event",
    5: "B-Organization",
    6: "I-Person",
    7: "B-Date",
    8: "I-Date",
    9: "I-Event",
    10: "I-Location",
}

# Only count these entity types. Note the model uses Title-case
# ("Person", "Location") rather than uppercase ("PERSON", "LOCATION").
COUNTED_TYPES = {"Person", "Location", "Organization"}


def merge_bio_spans(tokens, labels):
    """Collapse BIO sequences into merged spans.

    Returns a list of {"text": str, "label": str}. Only entity types in
    COUNTED_TYPES are kept. Handles WordPiece (##) and SentencePiece (▁)
    subword conventions.
    """
    spans = []
    cur_label = None
    cur_tokens = []

    def stitch(toks):
        text = ""
        for t in toks:
            if t.startswith("##"):
                text += t[2:]
            elif t.startswith("▁"):
                piece = t[1:]
                text += (" " if text else "") + piece
            else:
                text += (" " if text else "") + t
        return text.strip()

    def flush():
        nonlocal cur_label, cur_tokens
        if cur_label is not None and cur_tokens and cur_label in COUNTED_TYPES:
            spans.append({"text": stitch(cur_tokens), "label": cur_label})
        cur_label = None
        cur_tokens = []

    for tok, lab in zip(tokens, labels):
        if lab == "O":
            flush()
            continue
        prefix, _, etype = lab.partition("-")
        if prefix == "B":
            flush()
            cur_label = etype
            cur_tokens = [tok]
        elif prefix == "I":
            if cur_label == etype:
                cur_tokens.append(tok)
            else:
                # Lenient: I- without matching B- starts a new span.
                flush()
                cur_label = etype
                cur_tokens = [tok]
        else:
            flush()

    flush()
    return spans


def predict_entities(captions, model, tokenizer, device, batch_size=16):
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
    parser.add_argument(
        "--model",
        default=NER_MODEL,
        help="Override the NER model repo id from the command line.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Override the tokenizer repo id (defaults to --model).",
    )
    args = parser.parse_args()

    model_id = args.model
    tokenizer_id = args.tokenizer or model_id

    output_dir = Path(args.output_dir)
    lang_dir = output_dir / "data" / LANG
    lang_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    print(f"Loading NER model: {model_id}")
    model = AutoModelForTokenClassification.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Running on {device}")

    # Sanity-check: model label count must match our schema
    n_labels = model.config.num_labels
    if n_labels != len(LABELS_DICT):
        print(
            f"WARNING: model has {n_labels} labels but LABELS_DICT has "
            f"{len(LABELS_DICT)}. Check the schema before trusting results."
        )

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
        "ner_model": model_id,
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

    summary_path = output_dir / "nepali_hard_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Nepali hard test set summary")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {hard_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()