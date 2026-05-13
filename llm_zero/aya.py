import torch
from transformers import pipeline
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import json
from sacrebleu.metrics import BLEU, CHRF
from pycocoevalcap.cider.cider import Cider
from io import BytesIO
import base64

# For Japanese/Chinese tokenization
try:
    import jieba
    import MeCab

    CJK_TOKENIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jieba or MeCab not installed. Install with:")
    print("  pip install jieba mecab-python3 unidic-lite")
    CJK_TOKENIZATION_AVAILABLE = False

# Initialize pipeline
print("Loading Aya Vision 8B...")
pipe = pipeline(
    model="CohereLabs/aya-vision-8b",
    task="image-text-to-text",
    device_map="auto"
)

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
        # For non-CJK languages, return as is
        return text

    if lang_code in ["cmn"]:
        # Chinese/Cantonese - use jieba
        try:
            return " ".join(jieba.cut(text))
        except Exception as e:
            print(f"Jieba tokenization failed: {e}, using character-level")
            return " ".join(list(text))

    return text


def pil_image_to_url(image):
    """Convert PIL Image to data URL for Aya Vision"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    buffered.seek(0)
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_base64}"


def generate_caption_aya(image, news_content, language):
    """Generate caption using Aya Vision 8B"""

    # Convert image to URL format
    image_url = pil_image_to_url(image)

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

    # Format message with the aya-vision chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    try:
        outputs = pipe(text=messages, max_new_tokens=100, return_full_text=False)
        caption = outputs[0]['generated_text'].strip()
        return caption
    except Exception as e:
        print(f"Error generating caption: {e}")
        return ""


def evaluate_language(lang_code, dataset_name="tharindu/LaMuN", num_samples=None):
    """Evaluate Aya Vision model on a specific language with BLEU-4, CIDEr, and chrF"""

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
            generated_caption = generate_caption_aya(
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

    # Calculate chrF (doesn't need tokenization - works on characters)
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
print("FINAL RESULTS SUMMARY - AYA VISION 8B")
print("=" * 80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv("aya_vision_evaluation_results.csv", index=False)
print("\n✓ Results saved to aya_vision_evaluation_results.csv")

# Save detailed predictions
with open("aya_vision_predictions.json", "w", encoding="utf-8") as f:
    json.dump({
        "predictions": all_predictions,
        "references": all_references
    }, f, ensure_ascii=False, indent=2)
print("✓ Predictions saved to aya_vision_predictions.json")

# Calculate averages
print("\nAverage Scores Across All Languages:")
print(f"  Average BLEU-4: {results_df['bleu4'].mean():.2f}")
print(f"  Average chrF:   {results_df['chrf'].mean():.2f}")
print(f"  Average CIDEr:  {results_df['cider'].mean():.2f}")
