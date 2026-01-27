# Results Post-Processing Guide

This directory contains consolidation scripts that process raw benchmark results into summary tables and visualizations.

## Overview

After running a benchmark (e.g., `benchmark_voc2007.py`), results are stored in individual model folders:

```
results/VOC2007/
├── azure_intelligence/VOC2007/azure_intelligence/phase_1_results.csv
├── mistral_document_ai/VOC2007/mistral_document_ai/phase_1_results.csv
├── gpt-5-mini/VOC2007/gpt-5-mini/phase_2_results.csv
└── gpt-5-nano/VOC2007/gpt-5-nano/phase_2_results.csv
```

The consolidation script transforms this into analysis-ready format:

```
results_postprocessing/VOC2007/
├── phase_1_consolidated.csv       # All models' results side-by-side
├── phase_1_summary.csv            # Inference time, error rates, etc.
├── phase_2_consolidated.csv
├── phase_2_summary.csv
├── all_phases_summary.csv         # Combined summary
└── consolidate_results.py         # Consolidation script
```

## Available Consolidation Scripts

### VOC2007 (Chinese Medical Lab Reports)
```bash
cd ocr_vs_vlm/results_postprocessing/VOC2007
uv run python consolidate_results.py
```

**Outputs:**
- `phase_1_consolidated.csv` - Azure Intelligence & Mistral Document AI
- `phase_2_consolidated.csv` - GPT-5-mini & GPT-5-nano (generic prompting)
- `phase_3a_consolidated.csv` - GPT-5-mini & GPT-5-nano (with domain context)
- `phase_4_consolidated.csv` - GPT-5-mini & GPT-5-nano (detailed prompting)
- `all_phases_summary.csv` - Summary table with inference times and error rates

### PubLayNet (Academic Paper Layout)
```bash
cd ocr_vs_vlm/results_postprocessing/publaynet_full
uv run python consolidate_results.py
```

**Outputs:**
- `P-A_consolidated.csv` - OCR baseline results
- `P-B_consolidated.csv` - VLM direct analysis results
- `P-C_consolidated.csv` - VLM+OCR pipeline results

### DocVQA Mini (QA on Documents)
```bash
cd ocr_vs_vlm/results_postprocessing/DocVQA_mini
uv run python consolidate_results.py
```

## Consolidated CSV Structure

### Phase Results CSV
Each `phase_X_consolidated.csv` contains:

**Common Columns (shared across all models):**
- `sample_id` - Unique sample identifier
- `image_path` - Path to the input image
- `dataset` - Dataset name
- `language` - Language tag
- `ground_truth` - Ground truth text/annotation

**Model-Specific Columns (prefixed with model name):**
- `{model}_prediction` - Model's output/prediction
- `{model}_prompt` - The prompt sent to the model
- `{model}_inference_time_ms` - Time taken (milliseconds)
- `{model}_tokens_used` - Tokens consumed (for VLMs)
- `{model}_error` - Error message if failed
- `{model}_timestamp` - When the inference ran

**Example for VOC2007 Phase 1:**
```
sample_id | image_path | dataset | ground_truth | azure_intelligence_prediction | ... | mistral_document_ai_prediction | ...
```

## Summary CSV Structure

Each `phase_X_summary.csv` contains:

- `model` - Model name
- `phase` - Phase identifier
- `total_samples` - Number of samples processed
- `avg_inference_time_ms` - Average inference time
- `median_inference_time_ms` - Median inference time
- `min_inference_time_ms` - Fastest inference
- `max_inference_time_ms` - Slowest inference
- `error_count` - Number of failed API calls
- `error_rate` - Percentage of failures
- `avg_tokens` - Average tokens used (VLMs)
- `total_tokens` - Total tokens consumed
- `predictions_count` - Number of successful predictions
- `prediction_rate` - % of samples with valid predictions

## Key Statistics Computed

### Inference Performance
- **Latency:** Min/max/mean/median times per model
- **Throughput:** Samples per hour (derived from mean time)
- **Error Rate:** % of failed API calls
- **Coverage:** % of samples with successful predictions

### Model Outputs
- **Consistency:** Do all models process all samples?
- **Token Usage:** How many tokens does each model consume?
- **Failure Modes:** What types of errors occur?

## Working with Consolidated Data

### In Python (Pandas)
```python
import pandas as pd

# Load consolidated results
df = pd.read_csv('phase_1_consolidated.csv')

# Compare models on same samples
comparison = df[['sample_id', 'azure_intelligence_prediction', 'gpt-5-mini_prediction']]

# Calculate accuracy (requires reference implementation)
from editdistance import eval as levenshtein
cer = levenshtein(gt, pred) / len(gt)
```

### In Analysis Notebooks
The evaluation notebooks load and process consolidated CSVs:

```python
# Load consolidation output
RESULTS_DIR = Path("../results_postprocessing/VOC2007")
phase_1 = pd.read_csv(RESULTS_DIR / "phase_1_consolidated.csv")
summary = pd.read_csv(RESULTS_DIR / "all_phases_summary.csv")

# Metrics are then calculated (CER, WER, ANLS, etc.)
```

## Troubleshooting

### "No result files found!"
- Check that benchmark has been run: `results/[DATASET]/[MODEL_NAME]/...`
- Verify file structure matches expected layout
- Run `find results/ -name "*.csv"` to verify files exist

### "WARNING: No common samples found"
- Different models may have processed different samples
- Consolidation will use union of all sample_ids
- Missing model results will appear as NaN

### Multiline Cell Handling
Some predictions contain newlines or special characters. The consolidation script:
- Uses pandas CSV reader (robust multiline handling)
- Preserves all characters in predictions
- Handles Unicode Chinese characters correctly

## Advanced Usage

### Filter by Performance
```python
# Find samples where models disagree
import editdistance
comparison = df[['sample_id', 'model1_prediction', 'model2_prediction']].copy()
comparison['difference'] = comparison.apply(
    lambda row: editdistance.eval(row['model1_prediction'], row['model2_prediction']),
    axis=1
)
disagreements = comparison[comparison['difference'] > 0]
```

### Export for Publication
```python
# Create summary table for paper
summary_table = summary[['model', 'phase', 'avg_inference_time_ms', 
                          'predictions_count', 'prediction_rate']].copy()
summary_table.to_csv('summary_for_paper.csv', index=False)
```

### Merge Multiple Datasets
```python
# Compare VOC2007 and PubLayNet performance
voc = pd.read_csv('VOC2007/all_phases_summary.csv')
pub = pd.read_csv('publaynet_full/all_phases_summary.csv')
voc['dataset'] = 'VOC2007'
pub['dataset'] = 'PubLayNet'
combined = pd.concat([voc, pub])
```

## Performance Baselines

### VOC2007 (238 samples, Chinese Medical Text)
- **Fastest:** Azure Intelligence OCR (~3.8s/sample)
- **Slowest:** GPT-5-nano with detailed prompts (~33s/sample)
- **Most Complete:** GPT-5-mini (100% predictions across all phases)
- **Least Complete:** GPT-5-nano Phase 2 (26% prediction rate)

### PubLayNet (500 samples, Document Layout)
- **Phase P-A (OCR):** ~3-4s per sample
- **Phase P-B (VLM Direct):** ~15-23s per sample  
- **Phase P-C (VLM+OCR):** ~15-20s per sample

## See Also
- [Analysis Notebooks README](../results_analysis/README.md)
- Individual dataset benchmark scripts (`benchmark_[dataset].py`)
