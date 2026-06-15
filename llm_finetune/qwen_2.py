"""
QLoRA instruction fine-tuning for Qwen2.5-VL-7B-Instruct on LaMuN, followed
by automatic evaluation on the test set with BLEU-4, chrF, and CIDEr.

Uses the same prompt format as the standalone eval script so train and
inference are aligned: only the language and the number of training
examples need to change.

Usage:
  python finetune_qwen_qlora.py                                # eng, 1000 train, full test
  python finetune_qwen_qlora.py --language sin --num-samples 1000
  python finetune_qwen_qlora.py --language ben --epochs 3 --eval-samples 200
  python finetune_qwen_qlora.py --skip-train --adapter-path ./.../final-adapter
  python finetune_qwen_qlora.py --skip-eval         # train only

Hardware:
  ~16-20 GB VRAM at batch_size=1. Tested target: A5000 (24 GB) or T4 (16 GB).

Outputs (under <output-dir>):
  - final-adapter/            LoRA weights + processor
  - run_config.json           training args
  - eval_results.json         BLEU/chrF/CIDEr + meta
  - eval_predictions.json     per-example predictions + references
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

# Metrics
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider

# Optional CJK tokenisation
try:
    import jieba
    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    CJK_TOKENIZATION_AVAILABLE = False


# ---------- Config: shared with the eval script ----------

MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
DATASET_NAME = "tharindu/LaMuN"

LANGUAGE_NAMES = {
    "ara": "Arabic",
    "eng": "English",
    "cmn": "Chinese",
    "hin": "Hindi",
    "ind": "Indonesian",
    "ben": "Bengali",
    "sin": "Sinhala",
    "nep": "Nepalese",
    "swa": "Swahili",
}

CJK_LANGUAGES = {"cmn"}


def build_prompt(news_content: str, language_code: str) -> str:
    """Identical prompt template to the standalone eval script."""
    language = LANGUAGE_NAMES[language_code]
    return f"""You are writing a caption for a newspaper image.

Given the image and this news article excerpt:
{news_content[:1200]}

Task: Write a concise, informative caption for this image in {language}.

Guidelines:
- Write in {language} language only
- Keep it brief (10-12 words)
- Identify and include: people's names, locations, and organizations visible in the image
- Connect what you see in the image to the news context
- Use journalistic style (factual, clear, objective)
- Focus on the main subject of the image

Caption in {language}:"""


# ---------- Training-time data ----------

def make_train_messages(example, language_code: str):
    """Chat with assistant turn included — the model learns to produce it."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": build_prompt(example["content"], language_code)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["caption"]}],
        },
    ]


class QwenVLDataCollator:
    """Tokenises a batch and masks everything before the assistant turn so
    loss is computed only on the caption."""

    def __init__(self, processor, language_code: str):
        self.processor = processor
        self.language_code = language_code
        self.assistant_marker_ids = processor.tokenizer(
            "<|im_start|>assistant\n", add_special_tokens=False
        )["input_ids"]

    def __call__(self, examples):
        batch_texts, batch_messages = [], []
        for ex in examples:
            messages = make_train_messages(ex, self.language_code)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            batch_texts.append(text)
            batch_messages.append(messages)

        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = self.processor(
            text=batch_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        marker = self.assistant_marker_ids
        m_len = len(marker)
        for i in range(labels.size(0)):
            ids = inputs["input_ids"][i].tolist()
            split_at = None
            for j in range(len(ids) - m_len + 1):
                if ids[j : j + m_len] == marker:
                    split_at = j + m_len
                    break
            if split_at is None:
                labels[i, :] = -100  # safer than training on the prompt
            else:
                labels[i, :split_at] = -100

        inputs["labels"] = labels
        return inputs


# ---------- Model + LoRA ----------

def load_quantised_base():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"Loading {MODEL_ID} in 4-bit...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    return model


def attach_lora(model, r=16, alpha=32, dropout=0.05):
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------- Inference + metrics ----------

def tokenize_for_metrics(text, language_code):
    """Jieba word-segment for Chinese; pass-through otherwise."""
    if language_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
        try:
            return " ".join(jieba.cut(text))
        except Exception:
            return " ".join(list(text))
    return text


def generate_caption(model, processor, image, news_content, language_code,
                     max_new_tokens=100):
    prompt = build_prompt(news_content, language_code)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                   do_sample=False)
    trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, generated)]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0].strip()


def evaluate(model, processor, language_code, num_samples=None):
    print(f"\n{'=' * 70}")
    print(f"Evaluating {LANGUAGE_NAMES[language_code]} ({language_code})")
    print(f"{'=' * 70}")

    ds = load_dataset(DATASET_NAME, language_code, split="test")
    if num_samples is not None and num_samples < len(ds):
        ds = ds.select(range(num_samples))
    print(f"Test examples: {len(ds)}")

    model.eval()
    predictions, references = [], []

    for i, ex in enumerate(tqdm(ds, desc=f"Gen {language_code}")):
        try:
            pred = generate_caption(model, processor, ex["image"],
                                    ex["content"], language_code)
        except Exception as e:
            print(f"  [idx={i}] generation failed: {e}")
            pred = ""
        predictions.append(pred)
        references.append([ex["caption"]])

        if i < 3:
            print(f"\nSample {i + 1}")
            print(f"  Generated: {pred}")
            print(f"  Reference: {ex['caption']}")

    # CJK tokenisation for BLEU + CIDEr
    if language_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
        tok_preds = [tokenize_for_metrics(p, language_code) for p in predictions]
        tok_refs = [[tokenize_for_metrics(r[0], language_code)] for r in references]
    else:
        tok_preds = predictions
        tok_refs = references

    print("\nComputing metrics...")
    bleu = BLEU(max_ngram_order=4)
    bleu_score = bleu.corpus_score(tok_preds, [[r[0] for r in tok_refs]])

    chrf = CHRF()
    chrf_score = chrf.corpus_score(predictions, [[r[0] for r in references]])

    cider_scorer = Cider()
    preds_dict = {i: [p] for i, p in enumerate(tok_preds)}
    refs_dict = {i: r for i, r in enumerate(tok_refs)}
    cider_score, _ = cider_scorer.compute_score(refs_dict, preds_dict)

    results = {
        "language": language_code,
        "language_name": LANGUAGE_NAMES[language_code],
        "num_samples": len(predictions),
        "bleu4": float(bleu_score.score),
        "chrf": float(chrf_score.score),
        "cider": float(cider_score * 100),
    }

    print(f"\n{LANGUAGE_NAMES[language_code]} results:")
    print(f"  BLEU-4: {results['bleu4']:.2f}")
    print(f"  chrF:   {results['chrf']:.2f}")
    print(f"  CIDEr:  {results['cider']:.2f}")

    return results, predictions, references


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="eng",
                        choices=list(LANGUAGE_NAMES.keys()))
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Training examples.")
    parser.add_argument("--eval-samples", type=int, default=None,
                        help="Test examples (None = full test set).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and only evaluate a saved adapter.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Train without running the final evaluation.")
    parser.add_argument("--adapter-path", default=None,
                        help="Adapter to load when --skip-train is set.")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"./qwen-vl-qlora-{args.language}-{args.num_samples}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    adapter_path = args.adapter_path or str(out_dir / "final-adapter")

    # -------- Training --------
    if not args.skip_train:
        print(f"\nLoading {DATASET_NAME} / {args.language} train split...")
        train_ds = load_dataset(DATASET_NAME, args.language, split="train")
        print(f"Full train size: {len(train_ds)}")

        if args.num_samples and args.num_samples < len(train_ds):
            train_ds = train_ds.shuffle(seed=args.seed).select(range(args.num_samples))
        train_ds = train_ds.filter(
            lambda ex: bool(ex.get("caption", "").strip()) and ex.get("image") is not None
        )
        print(f"Training on {len(train_ds)} examples")

        model = load_quantised_base()
        model = attach_lora(model, r=args.lora_r, alpha=args.lora_alpha)
        model.gradient_checkpointing_enable()

        collator = QwenVLDataCollator(processor, args.language)

        training_args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine",
            max_grad_norm=args.max_grad_norm,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            report_to="none",
            seed=args.seed,
            dataloader_num_workers=2,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=collator,
        )
        print("\nStarting training...")
        trainer.train()

        trainer.model.save_pretrained(adapter_path)
        processor.save_pretrained(adapter_path)
        print(f"\nSaved LoRA adapter to {adapter_path}")

        # Free training-time memory before eval. We need use_cache=True for
        # generation, and want to drop the optimiser/scheduler state.
        del trainer, model
        torch.cuda.empty_cache()

    # -------- Evaluation --------
    if args.skip_eval:
        print("\n--skip-eval set; skipping evaluation.")
        return

    print(f"\nLoading base model + adapter from {adapter_path} for inference...")
    base = load_quantised_base()
    eval_model = PeftModel.from_pretrained(base, adapter_path)
    eval_model.config.use_cache = True
    eval_model.eval()

    results, predictions, references = evaluate(
        eval_model, processor, args.language, num_samples=args.eval_samples
    )

    # Save eval outputs
    with (out_dir / "eval_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with (out_dir / "eval_predictions.json").open("w", encoding="utf-8") as f:
        json.dump({
            "language": args.language,
            "adapter_path": adapter_path,
            "predictions": predictions,
            "references": references,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_dir / 'eval_results.json'}")
    print(f"Wrote {out_dir / 'eval_predictions.json'}")


if __name__ == "__main__":
    main()