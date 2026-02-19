# Results Pipeline Documentation

## Directory Structure

- **0_info/** - Pipeline scripts and documentation (you are here)
- **1_raw/** - Raw benchmark outputs organized by dataset and phase
- **2_clean/** - Consolidated results (one CSV per phase with all models)
- **3_embeddings/** - Pre-computed embeddings for semantic analysis
- **4_postprocessing/** - Aggregated statistics and analysis (future)

## Running the Pipeline

### Step 1: Run Benchmarks
```bash
uv run python scripts/run_benchmark.py --dataset docvqa --phases QA1a --sample-limit 100
```

Results saved to: `results/1_raw/DocVQA_mini/QA1a/{model}_results_{timestamp}.csv`

### Step 2: Clean and Consolidate
```bash
cd results/2_clean
python clean_files.py --incremental
```

This validates and consolidates raw results into: `results/2_clean/DocVQA_mini_QA1a.csv`

### Step 3: Generate Embeddings (Optional)
```bash
cd results/2_clean
python create_embeddings_manifest.py
```

### Step 4: Analyze Results
```bash
cd results/3_notebooks  # (To be implemented)
jupyter notebook analysis.ipynb
```

## File Naming Conventions

**Raw results (1_raw/):**
- `{model}_results_{YYYYMMDD_HHMMSS}.csv`
- Organized in: `{dataset}/{phase}/`

**Clean results (2_clean/):**
- `{dataset}_{phase}.csv`
- Contains all models for that phase

## Cleaning Strategy

The `clean_files.py` script:
1. Discovers all CSV files in 1_raw/
2. Validates row counts, error patterns, empty predictions
3. Selects best file per model (most rows, newest timestamp)
4. Consolidates into single CSV per experiment
5. Moves invalid/duplicate files to zzz_ignore/

## Phase Naming Conventions

### QA Benchmarks (DocVQA, InfographicVQA, DUDE, ChartQA)
- **QA1a**: OCR extraction → simple QA prompt
- **QA1b**: OCR extraction → detailed QA prompt
- **QA1c**: OCR extraction → chain-of-thought QA
- **QA2a**: Direct VLM with simple prompt
- **QA2b**: Direct VLM with detailed prompt
- **QA3a**: Hybrid (OCR + VLM reasoning)
- **QA4a**: Multi-page with retrieval (VisRBench only)

### Parsing Benchmarks (PubLayNet, RX-PAD, IAM)
- **P-A**: Pure OCR baseline
- **P-B**: Direct VLM extraction
- **P-C**: Hybrid (OCR + VLM refinement)
- **1, 2, 3**: IAM uses integer phases (equivalent to P-A, P-B, P-C)
