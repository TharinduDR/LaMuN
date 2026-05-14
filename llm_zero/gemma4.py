"""
LaMuN multilingual captioning evaluation with google/gemma-4-E4B-it.

Gemma 4 E4B is ~4B effective parameters with native multimodal (text + image)
support. Fits comfortably on a single A5000 (24 GB) in bf16.

Same prompt template and 9-language scope (eng, ara, cmn, hin, ind, ben,
nep, sin, swa) as the Qwen2.5-VL-7B, Qwen3-VL-8B, Aya-Vision-8B, and
Janus-Pro-7B baselines, so this row is directly comparable.

NOTE: Gemma 4 is a new architecture. You need a recent transformers:
    pip install -U transformers accelerate

If you hit "unknown model type 'gemma4'" errors, install transformers from
main:
    pip install git+https://github.com/huggingface/transformers
"""

import json

import pandas as pd
import torch
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    import jieba
    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jieba not installed. Install with: pip install jieba")
    CJK_TOKENIZATION_AVAILABLE = False


# ---------- Config ----------

MODEL_PATH = "google/gemma-4-E4B-it"

LANGUAGES = ["ara", "eng", "cmn", "hin", "ind", "ben", "nep", "sin", "swa"]

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


# ---------- Model load ----------

print(f"Loading {MODEL_PATH}...")
# AutoModelForImageTextToText resolves the correct Gemma 4 VL class from the
# config's `architectures` field. Avoids hard-coding the class name in case
# the exact symbol differs from `Gemma4ForConditionalGeneration`.
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)


# ---------- Metrics ----------

chrf_metric = CHRF()
cider_scorer = Cider()


def tokenize_text(text, lang_code):
    """Jieba segmentation for Chinese; pass-through otherwise."""
    if lang_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
        try:
            return " ".join(jieba.cut(text))
        except Exception:
            return " ".join(list(text))
    return text


# ---------- Caption generation ----------

def build_prompt(news_content: str, language_code: str) -> str:
    """Same prompt template as the other baselines."""
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


def generate_caption_gemma(image, news_content, language_code,
                            max_new_tokens=100):
    """Generate a caption with Gemma 4 E4B."""
    prompt = build_prompt(news_content, language_code)

    # Gemma 4 chat format: same {role, content: [...]} structure as Qwen,
    # processor handles image placeholder injection.
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        # Match the model's bf16 dtype for image features
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()

    except Exception as e:
        print(f"Error generating caption: {e}")
        import traceback
        traceback.print_exc()
        return ""


# ---------- Per-language evaluation ----------

def evaluate_language(lang_code, dataset_name="tharindu/LaMuN",
                      num_samples=None):
    print(f"\n{'=' * 80}")
    print(f"Evaluating {LANGUAGE_NAMES[lang_code]} ({lang_code})")
    print(f"{'=' * 80}")

    dataset = load_dataset(dataset_name, lang_code)
    test_data = dataset["test"]

    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))

    print(f"Processing {len(test_data)} examples...")

    predictions, references = [], []

    for i, example in enumerate(tqdm(test_data, desc=f"Generating {lang_code}")):
        try:
            pred = generate_caption_gemma(
                example["image"], example["content"], lang_code
            )
            predictions.append(pred)
            references.append([example["caption"]])

            if i < 3:
                print(f"\nSample {i + 1}:")
                print(f"Generated: {pred}")
                print(f"Reference: {example['caption']}")
                print("-" * 80)

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            references.append([example["caption"]])

    # CJK tokenisation
    if lang_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
        print(f"Tokenizing texts for {lang_code} (CJK)...")
        tok_preds = [tokenize_text(p, lang_code) for p in predictions]
        tok_refs = [[tokenize_text(r[0], lang_code)] for r in references]
    else:
        tok_preds = predictions
        tok_refs = references

    print(f"\nCalculating BLEU-4 for {lang_code}...")
    bleu_metric = BLEU(max_ngram_order=4)
    bleu_score = bleu_metric.corpus_score(
        tok_preds, [[r[0] for r in tok_refs]]
    )

    print(f"Calculating chrF for {lang_code}...")
    chrf_score = chrf_metric.corpus_score(
        predictions, [[r[0] for r in references]]
    )

    print(f"Calculating CIDEr for {lang_code}...")
    preds_dict = {i: [p] for i, p in enumerate(tok_preds)}
    refs_dict = {i: r for i, r in enumerate(tok_refs)}
    cider_score, _ = cider_scorer.compute_score(refs_dict, preds_dict)

    results = {
        "language": lang_code,
        "language_name": LANGUAGE_NAMES[lang_code],
        "num_samples": len(predictions),
        "bleu4": float(bleu_score.score),
        "chrf": float(chrf_score.score),
        "cider": float(cider_score * 100),
    }

    print(f"\nResults for {LANGUAGE_NAMES[lang_code]}:")
    print(f"  BLEU-4: {results['bleu4']:.2f}")
    print(f"  chrF:   {results['chrf']:.2f}")
    print(f"  CIDEr:  {results['cider']:.2f}")

    return results, predictions, references


# ---------- Main ----------

if __name__ == "__main__":
    all_results = []
    all_predictions, all_references = {}, {}

    for lang in LANGUAGES:
        try:
            results, preds, refs = evaluate_language(
                lang,
                dataset_name="tharindu/LaMuN",
                num_samples=None,  # set 10-100 for quick smoke test
            )
            all_results.append(results)
            all_predictions[lang] = preds
            all_references[lang] = refs
        except Exception as e:
            print(f"Error evaluating {lang}: {e}")
            import traceback
            traceback.print_exc()
            continue

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY - GEMMA-4-E4B-IT")
    print("=" * 80)
    print(results_df.to_string(index=False))

    results_df.to_csv("gemma4_e4b_evaluation_results.csv", index=False)
    print("\n✓ Results saved to gemma4_e4b_evaluation_results.csv")

    with open("gemma4_e4b_predictions.json", "w", encoding="utf-8") as f:
        json.dump(
            {"predictions": all_predictions, "references": all_references},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("✓ Predictions saved to gemma4_e4b_predictions.json")

    print("\nAverage Scores Across All Languages:")
    print(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}")
    print(f"  Average chrF:   {results_df['chrf'].mean():.2f}")
    print(f"  Average CIDEr:  {results_df['cider'].mean():.2f}")