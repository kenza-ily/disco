# OCR vs VLM Benchmark Analysis Notebooks

This directory contains Jupyter notebooks for analyzing and evaluating OCR and Vision Language Model (VLM) performance across multiple datasets.

## Available Benchmarks

### 1. **VOC2007: Chinese Medical Lab Reports** (`voc2007_eval.ipynb`)
**Dataset:** 238 Simplified Chinese medical laboratory report images

**Phases:**
- **Phase 1:** OCR Baseline (Azure Intelligence, Mistral Document AI)
- **Phase 2:** VLM Generic Prompting (GPT-5-mini, GPT-5-nano)
- **Phase 3a:** VLM with Language + Document Type Context
- **Phase 4:** VLM with Detailed Medical Field Instructions

**Evaluation Metrics:**
- Character Error Rate (CER) - For Chinese character-level accuracy
- Word Error Rate (WER) - For semantic unit accuracy
- Medical field extraction accuracy (reports, names, results, etc.)
- Inference time comparison

**Key Insights:**
- Pure OCR (Phase 1) takes ~3.8-4.0 seconds per image
- VLM inference takes ~25-33 seconds per image
- Detailed medical prompting (Phase 4) shows best results for GPT-5-mini
- GPT-5-nano has consistency issues (incomplete responses in phases 2-3)

---

### 2. **DocVQA Mini: Question Answering on Documents** (`docvqa_qa_eval.ipynb`)
**Dataset:** 500 document image samples with question-answer pairs

**Phases (QA Evaluation):**
- **QA1a/1b/1c:** OCR Pipeline with varying prompt styles
  - QA1a: Simple prompts
  - QA1b: Detailed prompts
  - QA1c: Chain-of-Thought reasoning
  
- **QA2a/2b/2c:** VLM Parse Pipeline
  - VLM first extracts text, then answers questions
  - Variations in prompt complexity
  
- **QA3a/3b:** Direct VQA (End-to-end VLM)
  - VLM answers directly from image without intermediate text extraction
  - Simple vs Chain-of-Thought prompts

**Evaluation Metrics:**
- ANLS (Average Normalized Levenshtein Similarity) - Text matching similarity
- Exact Match Rate - Percentage of perfect answers
- Error rate and failure analysis

**Approaches Compared:**
- OCR Pipeline: Extract text first, then answer questions
- VLM Parse: Use VLM for extraction, then answer
- Direct VQA: End-to-end VLM question answering

---

### 3. **IAM Mini: Handwriting Recognition** (`iammini_eval.ipynb`)
**Dataset:** IAM handwritten text dataset (mini version)

**Focus:**
- Handwriting recognition accuracy
- Comparison of OCR models on cursive text
- Impact of model architecture on handwriting

---

### 4. **ICDAR Mini: Document Layout Analysis** (`icdar_mini_eval.ipynb`)
**Dataset:** ICDAR dataset (mini version)

**Focus:**
- Document layout understanding
- Text region detection
- Layout structure preservation

---

### 5. **PubLayNet: Document Layout Understanding** (`publaynet_eval.ipynb`)
**Dataset:** 500 academic paper document images

**Phases:**
- **P-A:** OCR Baseline (Azure Intelligence, Mistral Document AI)
- **P-B:** VLM Direct Analysis (GPT-5-mini, GPT-5-nano)
- **P-C:** VLM+OCR Pipeline (Combined approach)

**Focus:**
- Document structure understanding
- Layout preservation
- Title, abstract, and section extraction accuracy

---

## Workflow for Analysis

### Step 1: Run Benchmark
```bash
# Example for VOC2007
uv run python -m ocr_vs_vlm.benchmark_voc2007 --models gpt-5-mini gpt-5-nano --phases 2 3 4
```

### Step 2: Consolidate Results
```bash
# Results are consolidated by the consolidation script
cd ocr_vs_vlm/results_postprocessing/VOC2007
uv run python consolidate_results.py
```

This creates:
- `phase_X_consolidated.csv` - Combined results from all models for phase X
- `phase_X_summary.csv` - Summary statistics (inference time, error rates, etc.)
- `all_phases_summary.csv` - Combined summary across all phases

### Step 3: Open Analysis Notebook
```bash
# Open in Jupyter or VS Code
jupyter notebook ocr_vs_vlm/results_analysis/voc2007_eval.ipynb
```

The notebook loads pre-consolidated results and generates:
- Summary statistics tables
- Model performance comparisons
- CER/WER distributions
- Visualization plots
- Field extraction analysis
- Error pattern analysis

## Key Evaluation Metrics

### Text Similarity (For OCR/VLM)
- **CER (Character Error Rate):** Edit distance / ground truth length
  - Ideal for Chinese where characters are basic units
  - Lower is better
  
- **WER (Word Error Rate):** Token-level edit distance
  - For languages with clear word boundaries
  - Lower is better

### QA Metrics (For DocVQA)
- **ANLS (Average Normalized Levenshtein Similarity):** 
  - Measures how similar predicted answer is to ground truth
  - Range: 0-1 (higher is better)
  
- **Exact Match (EM):**
  - Percentage of perfectly correct answers
  - Binary metric: 0 or 1

### Performance Metrics
- **Inference Time:** Wall-clock time per sample (ms)
- **Error Rate:** Percentage of failed API calls
- **Prediction Coverage:** Percentage of samples with predictions
- **Token Usage:** For VLMs, tokens consumed per request

## Data Organization

```
ocr_vs_vlm/
├── results/
│   ├── VOC2007/
│   │   ├── azure_intelligence/
│   │   ├── mistral_document_ai/
│   │   ├── gpt-5-mini/
│   │   └── gpt-5-nano/
│   ├── DocVQA_mini/
│   │   └── [phase_results]/
│   └── publaynet_full/
│       └── [phase_results]/
│
├── results_postprocessing/
│   ├── VOC2007/
│   │   ├── consolidate_results.py
│   │   ├── phase_1_consolidated.csv
│   │   ├── phase_2_consolidated.csv
│   │   ├── phase_3a_consolidated.csv
│   │   ├── phase_4_consolidated.csv
│   │   └── all_phases_summary.csv
│   ├── DocVQA_mini/
│   ├── publaynet_full/
│   └── [other_datasets]/
│
└── results_analysis/
    ├── voc2007_eval.ipynb
    ├── docvqa_qa_eval.ipynb
    ├── publaynet_eval.ipynb
    └── [other_eval_notebooks]/
```

## Running Analysis Notebooks

### In Jupyter
```bash
cd ocr_vs_vlm/results_analysis
jupyter notebook voc2007_eval.ipynb
```

### In VS Code
1. Open the notebook file
2. Select Python interpreter (should have required packages)
3. Run cells sequentially

## Custom Evaluation

To evaluate on your own dataset:

1. **Create a benchmark script** following the structure of `benchmark_voc2007.py`
2. **Run the benchmark** to generate results
3. **Create consolidation script** similar to VOC2007
4. **Create evaluation notebook** using templates from this directory

## Dependencies

The notebooks require:
- `pandas` - Data manipulation
- `numpy` - Numerical operations  
- `matplotlib` - Visualization
- `seaborn` - Statistical plots
- `editdistance` - Text similarity metrics (for CER/WER)

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn editdistance
```

## Notes

- **Chinese Text:** Notebooks include special handling for Chinese characters (CER is per-character)
- **Medical Domain:** VOC2007 includes analysis of medical field extraction
- **Inference Times:** OCR typically faster (~3-4s) than VLM (~25-33s)
- **Error Handling:** Failed API calls tracked separately; analysis focuses on successful runs
- **Multiline Outputs:** CSV consolidation handles multiline cells for full predictions

## Contact & Issues

For questions about specific notebooks or benchmarks, refer to the comments within each notebook.
