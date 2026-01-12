# Evaluation Metrics

## Metrics 

### For All Tasks
- **CER** (Character Error Rate) - Character-level edit distance
- **WER** (Word Error Rate) - Word-level edit distance  
- **ANLS** (Average Normalized Levenshtein Similarity) - Standard VQA metric
- **EM** (Exact Match) - Binary exact match score

### Additional Metrics for VQA Tasks (DocVQA, InfographicVQA)
- **Cosine Similarity** - Semantic similarity using embeddings
- **Substring Match** - Fuzzy matching where prediction is substring of ground truth or vice versa

## Rationale for Each Metric

### Character Error Rate (CER)
**Use Case:** Fine-grained OCR evaluation, especially for handwriting
**Why:** Character-level accuracy is crucial for tasks like form filling, where even small errors matter
**Datasets:** IAM Mini, ICDAR Mini, PubLayNet, VOC2007

### Word Error Rate (WER)
**Use Case:** Higher-level text recognition accuracy
**Why:** More forgiving than CER for minor character mistakes; reflects human reading patterns
**Datasets:** All parsing tasks

### ANLS (Average Normalized Levenshtein Similarity)
**Use Case:** Standard VQA metric from DocVQA/InfographicVQA papers
**Why:** Allows partial credit for near-matches; threshold prevents spurious matches
**Datasets:** All VQA tasks, can be used for parsing too

### Exact Match (EM)
**Use Case:** Strict accuracy measurement
**Why:** Important for applications requiring perfect answers
**Datasets:** All tasks

### Substring Match
**Use Case:** VQA with varied ground truth formats
**Why:** Catches cases where answer is embedded in longer text or vice versa
**Example:** "50%" matches "approximately 50 percent"
**Datasets:** DocVQA, InfographicVQA

### Cosine Similarity
**Use Case:** Semantic equivalence detection
**Why:** Catches synonyms and paraphrases that other metrics miss
**Example:** "car" vs "automobile", "10%" vs "ten percent"
**Datasets:** DocVQA, InfographicVQA
**Implementation:** Uses embedding_integration.py with text-embedding-3-small

## Dependencies

Required packages (added to existing requirements):
```
editdistance  # For CER/WER computation
scipy         # For cosine similarity
```

Embeddings (for cosine similarity):
- Requires: `llms/embeddings.py` module
- Uses: Azure OpenAI text-embedding-3-small
- Optional: Cosine similarity returns 0.0 if embeddings unavailable

## Visualization Enhancements

All notebooks now include:

1. **Comprehensive Metrics Dashboard**: 2×3 subplot grid showing all metrics
2. **Color-coded Performance**: 
   - Green = High performance
   - Red = Low performance
   - Inverted for error rates (CER, WER)
3. **Value Labels**: Metric values displayed on bars for easy reading
4. **Directional Indicators**: Labels indicate "Higher is Better" or "Lower is Better"

## Backward Compatibility

- All existing metrics (ANLS, EM) continue to work
- Existing analysis code remains functional
- New metrics are additive, not replacing existing ones
- CSV outputs maintain existing columns; new metrics can be added

## Future Enhancements

Potential additions:
1. **F1 Score** for multi-class classification tasks
2. **BLEU Score** for longer text generation
3. **BERTScore** for semantic similarity with contextualized embeddings
4. **Custom Thresholds** for ANLS per dataset
5. **Confidence Intervals** for all metrics


## Notes

- **Cosine Similarity**: May be 0.0 if embeddings not computed or API unavailable
- **Multiple Ground Truths**: All metrics properly handle lists; max score across all ground truths is returned
- **Normalization**: Text is lowercased and stripped before comparison
- **Error Handling**: All functions gracefully handle missing/empty values
