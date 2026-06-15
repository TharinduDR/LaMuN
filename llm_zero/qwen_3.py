"""
LaMuN multilingual captioning evaluation with Qwen/Qwen3-VL-8B-Instruct.

Same prompt template and 9-language scope (eng, ara, cmn, hin, ind, ben,
nep, sin, swa) as the Qwen2.5-VL-7B, Aya-Vision-8B, and Janus-Pro-7B
baselines, so the rows are directly comparable.

Qwen3-VL uses Qwen3VLForConditionalGeneration and supports the chat
template directly via processor.apply_chat_template(..., tokenize=True,
return_dict=True), removing the need for qwen_vl_utils.process_vision_info.
"""

import json

import pandas as pd
import torch
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    import jieba
    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jieba not installed. Install with: pip install jieba")
    CJK_TOKENIZATION_AVAILABLE = False


# ---------- Config ----------

MODEL_PATH = "Qwen/Qwen3-VL-235B-A22B-Instruct"

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
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
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


def generate_caption_qwen3(image, news_content, language_code,
                           max_new_tokens=100):
    """Generate a caption with Qwen3-VL-8B-Instruct."""
    prompt = build_prompt(news_content, language_code)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    try:
        # Qwen3-VL: apply_chat_template tokenises and handles images directly.
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
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
            pred = generate_caption_qwen3(
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
                num_samples=None,  # set 10-100 for a quick smoke test
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
    print("FINAL RESULTS SUMMARY - QWEN3-VL-8B-INSTRUCT")
    print("=" * 80)
    print(results_df.to_string(index=False))

    results_df.to_csv("qwen3_vl_evaluation_results.csv", index=False)
    print("\n✓ Results saved to qwen3_vl_evaluation_results.csv")

    with open("qwen3_vl_predictions.json", "w", encoding="utf-8") as f:
        json.dump(
            {"predictions": all_predictions, "references": all_references},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("✓ Predictions saved to qwen3_vl_predictions.json")

    print("\nAverage Scores Across All Languages:")
    print(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}")
    print(f"  Average chrF:   {results_df['chrf'].mean():.2f}")
    print(f"  Average CIDEr:  {results_df['cider'].mean():.2f}")