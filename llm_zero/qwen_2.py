import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider

# For Japanese/Chinese tokenization
try:
    import jieba
    import MeCab

    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jieba or MeCab not installed. Install with:")
    print("  pip install jieba mecab-python3 unidic-lite")
    CJK_TOKENIZATION_AVAILABLE = False

# Initialize model and processor
print("Loading Qwen2.5-VL-72B-Instruct...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

# Optional: Enable flash attention for better performance
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# Initialize metrics
chrf_metric = CHRF()
cider_scorer = Cider()

# Language codes
languages = ["ara", "eng", "cmn", "hin", "ind", "ben", "nep", "sin", "swa"]

language_names = {
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

# CJK languages that need special tokenization
CJK_LANGUAGES = ["cmn"]


def tokenize_text(text, lang_code):
    """Tokenize text appropriately based on language"""
    if not CJK_TOKENIZATION_AVAILABLE or lang_code not in CJK_LANGUAGES:
        return text

    if lang_code in ["cmn"]:
        # Chinese/Cantonese - use jieba
        try:
            return " ".join(jieba.cut(text))
        except Exception as e:
            print(f"Jieba tokenization failed: {e}, using character-level")
            return " ".join(list(text))

    return text


def generate_caption_qwen(image, news_content, language):
    """Generate caption using Qwen2.5-VL-7B"""

    # Create prompt
    prompt = f"""You are writing a caption for a newspaper image.

Given the image and this news article excerpt:
{news_content[:1200]}

Task: Write a concise, informative caption for this image in {language_names[language]}.

Guidelines:
- Write in {language_names[language]} language only
- Keep it brief (10-12 words)
- Identify and include: people's names, locations, and organizations visible in the image
- Connect what you see in the image to the news context
- Use journalistic style (factual, clear, objective)
- Focus on the main subject of the image

Caption in {language_names[language]}:"""

    # Format message for Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    try:
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        caption = output_text[0].strip()
        return caption

    except Exception as e:
        print(f"Error generating caption: {e}")
        import traceback
        traceback.print_exc()
        return ""


def evaluate_language(lang_code, dataset_name="tharindu/LaMuN", num_samples=None):
    """Evaluate Qwen2.5-VL model on a specific language with BLEU-4, CIDEr, and chrF"""

    print(f"\n{'=' * 80}")
    print(f"Evaluating {language_names[lang_code]} ({lang_code})")
    print(f"{'=' * 80}")

    # Load test set
    dataset = load_dataset(dataset_name, lang_code)
    test_data = dataset['test']

    if num_samples:
        test_data = test_data.select(range(min(num_samples, len(test_data))))

    print(f"Processing {len(test_data)} examples...")

    predictions = []
    references = []

    # Generate captions
    for i, example in enumerate(tqdm(test_data, desc=f"Generating {lang_code}")):
        try:
            generated_caption = generate_caption_qwen(
                example['image'],
                example['content'],
                lang_code
            )

            predictions.append(generated_caption)
            references.append([example['caption']])

            # Print sample outputs
            if i < 3:
                print(f"\nSample {i + 1}:")
                print(f"Generated: {generated_caption}")
                print(f"Reference: {example['caption']}")
                print("-" * 80)

        except Exception as e:
            print(f"Error on example {i}: {e}")
            predictions.append("")
            references.append([example['caption']])

    # Tokenize for CJK languages if needed
    if lang_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
        print(f"Tokenizing texts for {lang_code} (CJK language)...")
        tokenized_predictions = [tokenize_text(pred, lang_code) for pred in predictions]
        tokenized_references = [[tokenize_text(ref[0], lang_code)] for ref in references]
    else:
        tokenized_predictions = predictions
        tokenized_references = references

    # Calculate BLEU-4 with tokenized texts
    print(f"\nCalculating BLEU-4 for {lang_code}...")
    if lang_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
        print(f"  Using proper tokenization for BLEU-4")

    bleu_metric = BLEU(max_ngram_order=4)
    tokenized_references_transposed = [[ref[0] for ref in tokenized_references]]
    bleu_score = bleu_metric.corpus_score(tokenized_predictions, tokenized_references_transposed)

    # Calculate chrF
    print(f"Calculating chrF for {lang_code}...")
    references_transposed = [[ref[0] for ref in references]]
    chrf_score = chrf_metric.corpus_score(predictions, references_transposed)

    # Calculate CIDEr with tokenized texts
    print(f"Calculating CIDEr for {lang_code}...")
    if lang_code in CJK_LANGUAGES and CJK_TOKENIZATION_AVAILABLE:
        print(f"  Using proper tokenization for CIDEr")

    predictions_dict = {i: [pred] for i, pred in enumerate(tokenized_predictions)}
    references_dict = {i: refs for i, refs in enumerate(tokenized_references)}
    cider_score, _ = cider_scorer.compute_score(references_dict, predictions_dict)

    results = {
        "language": lang_code,
        "language_name": language_names[lang_code],
        "num_samples": len(predictions),
        "bleu4": bleu_score.score,
        "chrf": chrf_score.score,
        "cider": cider_score * 100
    }

    print(f"\nResults for {language_names[lang_code]}:")
    print(f"  BLEU-4: {results['bleu4']:.2f}")
    print(f"  chrF:   {results['chrf']:.2f}")
    print(f"  CIDEr:  {results['cider']:.2f}")

    return results, predictions, references


# Evaluate all languages
all_results = []
all_predictions = {}
all_references = {}

for lang in languages:
    try:
        results, preds, refs = evaluate_language(
            lang,
            dataset_name="tharindu/LaMuN",
            num_samples=None  # Set to 10-100 for quick testing
        )
        all_results.append(results)
        all_predictions[lang] = preds
        all_references[lang] = refs

    except Exception as e:
        print(f"Error evaluating {lang}: {e}")
        import traceback

        traceback.print_exc()
        continue

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Print summary
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY - QWEN2.5-VL-72B-INSTRUCT")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("qwen_vl_evaluation_results.csv", index=False)
print("\nResults saved to qwen_vl_evaluation_results.csv")

# Save detailed predictions
with open("qwen_vl_predictions.json", "w", encoding="utf-8") as f:
    json.dump({
        "predictions": all_predictions,
        "references": all_references
    }, f, ensure_ascii=False, indent=2)
print("Predictions saved to qwen_vl_predictions.json")

# Calculate averages
print("\nAverage Scores Across All Languages:")
print(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}")
print(f"  Average chrF:   {results_df['chrf'].mean():.2f}")
print(f"  Average CIDEr:  {results_df['cider'].mean():.2f}")
