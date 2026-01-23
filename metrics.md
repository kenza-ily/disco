# Evaluation Metrics

## Overview

This project uses task-specific evaluation metrics for OCR parsing and Visual Question Answering (VQA).

## VQA Metrics (DocVQA, InfographicVQA)

All VQA tasks compute **6 metrics** per prediction:

- **ANLS** (Average Normalized Levenshtein Similarity)
  - Standard VQA metric from DocVQA/InfographicVQA papers
  - Range: [0.0, 1.0], higher is better
  - Threshold: 0.5 (scores below threshold → 0)
  - Allows partial credit for near-matches

- **Exact Match**
  - Binary: 1.0 if prediction matches any ground truth exactly, 0.0 otherwise
  - Case-insensitive, whitespace-normalized

- **Substring Match** (Bidirectional)
  - 1.0 if prediction ⊆ ground truth OR ground truth ⊆ prediction
  - 0.0 otherwise
  - Useful for answers embedded in longer text

- **Prediction in Ground Truth**
  - 1.0 if entire prediction appears in any ground truth
  - 0.0 otherwise
  - Detects exact substring extractions

- **Ground Truth in Prediction**
  - 1.0 if any ground truth appears in prediction
  - 0.0 otherwise
  - Detects correct answers with extra text

- **Embedding Similarity**
  - Maximum cosine similarity between prediction and ground truth embeddings
  - Range: [0.0, 1.0], higher is better
  - Uses Azure OpenAI `text-embedding-3-small` (3072 dimensions)
  - Captures semantic equivalence (synonyms, paraphrases)
  - Returns 0.0 if embeddings unavailable

## OCR Parsing Metrics (IAM, ICDAR, PubLayNet, VOC2007)

All parsing tasks compute **4 metrics** per prediction:

- **CER** (Character Error Rate)
  - Character-level edit distance
  - Range: [0.0, ∞), lower is better
  - Fine-grained accuracy for OCR evaluation

- **WER** (Word Error Rate)
  - Word-level edit distance
  - Range: [0.0, ∞), lower is better
  - More forgiving than CER for minor errors

- **ANLS**
  - Same as VQA ANLS
  - Applied to OCR text comparison

- **Exact Match**
  - Same as VQA Exact Match
  - Applied to OCR text comparison

## Multi-Answer Handling

- VQA datasets have multiple valid answers per question
- All metrics check against **all** ground truths and return **best score**
- Example: If ground_truths = ["Paris", "paris, france"], prediction "Paris" matches first → score = 1.0

## Implementation Details

**Dependencies:**
- `editdistance` - For CER/WER/ANLS computation
- `scipy` - For cosine similarity
- Azure OpenAI SDK - For embeddings (optional)

**Error Handling:**
- Missing scipy → embedding_similarity returns 0.0
- Empty inputs → all metrics return 0.0
- API errors → logged with warning, returns empty embeddings

**Text Normalization:**
- Lowercase conversion
- Whitespace trimming
- Applied before all comparisons

## Cost Considerations

- **String metrics** (ANLS, EM, substring matching): <1ms per sample, free
- **Embedding generation**: ~500-1000ms per sample, ~$0.00002 per 1K tokens
- Total overhead: ~1 second per sample for embedding computation

## CSV Export

All benchmark results include these columns:
- VQA: `anls_score`, `exact_match`, `substring_match`, `prediction_in_ground_truth`, `ground_truth_in_prediction`, `embedding_similarity`
- OCR: `cer`, `wer`, `anls`, `exact_match`

## Use Cases

| Metric | Best For |
|--------|----------|
| ANLS | Standard VQA evaluation, partial credit |
| Exact Match | Strict accuracy requirements |
| Substring Match | Flexible answer formats |
| Prediction in GT | Detecting exact extractions |
| GT in Prediction | Detecting over-generation |
| Embedding Similarity | Semantic equivalence, paraphrases |
| CER | Fine-grained OCR accuracy |
| WER | Document-level OCR quality |

## Example Comparisons

**VQA Example:**
```
Ground truth: "Paris"
Prediction: "The answer is Paris"

- exact_match: 0.0
- prediction_in_ground_truth: 0.0
- ground_truth_in_prediction: 1.0
- substring_match: 1.0
- anls: ~0.35
- embedding_similarity: ~0.85
```

**Semantic Similarity Example:**
```
Ground truth: "1952"
Prediction: "nineteen fifty-two"

- exact_match: 0.0
- anls: ~0.2
- embedding_similarity: ~0.85
```

## Notes

- All functions handle `List[str]` ground truths correctly
- Embeddings are stored in CSV for analysis but not required
- Metrics are backward compatible - existing code continues to work
- New metrics are additive, not replacing existing ones
