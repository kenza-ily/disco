# OCR vs VLM Results Pipeline

This folder contains the complete results pipeline for the OCR vs VLM research project, from raw experimental outputs to analysis notebooks and visualizations.

## Quick Start

### Run Full Pipeline
```bash
# From ocr_vs_vlm/results/
python 0_info/run_pipeline.py --full
```

### Add New Experiment Results
1. Place raw CSVs in `1_raw/[dataset]/[phase]/`
2. Update `0_info/clean_experiments.json` with new experiment metadata
3. Run consolidation:
```bash
cd 2_clean
python clean_files.py --incremental
```
4. Open and run relevant notebook in `3_notebooks/`

## Folder Structure

```
results/
├── 0_info/                       # Pipeline documentation and metadata
│   ├── README.md                 # This file
│   ├── clean_experiments.json    # Registry of production experiments
│   ├── visualization_config.json # Color schemes and plot settings
│   ├── pipeline_status.json      # Pipeline run timestamps
│   └── moved_files_log.md        # Archive tracking
│
├── 1_raw/                        # Raw experiment outputs (one CSV per model per phase)
│   ├── DocVQA_mini/              # QA dataset (8 phases: QA1a-QA3b)
│   ├── InfographicVQA_mini/      # QA dataset (11 phases)
│   ├── IAM_mini/                 # Handwriting (3 phases: P-A, P-B, P-C)
│   ├── ICDAR_mini/               # Multilingual OCR (10 languages)
│   ├── publaynet_full/           # Layout detection (3 phases)
│   ├── VOC2007/                  # Chinese medical (238 samples)
│   ├── chartqapro_mini/          # Chart QA (experimental)
│   ├── dude_mini/                # Document understanding (experimental)
│   ├── RX-PAD/                   # Medical prescriptions (experimental)
│   └── zzz_ignore/               # Archived test runs and failed experiments
│
├── 2_clean/                      # Consolidated results (one CSV per phase with all models)
│   ├── DocVQA_mini/              # QA1a.csv, QA1b.csv, ..., QA3b.csv
│   ├── InfographicVQA_mini/      # QA1a.csv, ...
│   ├── IAM_mini/                 # P-A.csv, P-B.csv, P-C.csv
│   ├── ICDAR_mini/               # 20251223.csv
│   ├── publaynet_full/           # P-A.csv, P-B.csv, P-C.csv
│   ├── VOC2007/                  # 20251220_baseline.csv, 20251220_prompted.csv
│   └── clean_files.py            # Validation and consolidation script
│
├── 3_notebooks/                  # Analysis notebooks for visualization
│   ├── shared/                   # Shared utilities
│   │   ├── colors.py             # Brand color palettes
│   │   ├── data_loaders.py       # Standardized data loading
│   │   └── plot_utils.py         # Common plotting functions
│   ├── figures/                  # Generated plots (by dataset)
│   ├── 00_master_evaluation.ipynb       # Cross-dataset dashboard
│   ├── 01_docvqa_analysis.ipynb         # DocVQA deep dive
│   ├── 02_infographicvqa_analysis.ipynb # InfographicVQA analysis
│   ├── 03_iam_handwriting.ipynb         # IAM detailed analysis
│   ├── 04_icdar_multilingual.ipynb      # Language-specific performance
│   ├── 05_voc2007_chinese.ipynb         # Medical Chinese analysis
│   └── zzz_archive/              # Deprecated/broken notebooks
│
└── 4_postprocessing/             # Aggregated statistics and summaries
    ├── publaynet_full/
    ├── IAM_mini/
    ├── ICDAR_mini/
    └── VOC2007/
```

## Metadata Files

### clean_experiments.json

Defines production-ready experiments. This is the single source of truth for which raw files should be processed.

**Structure:**
```json
{
  "datasets": {
    "DatasetName": {
      "description": "Dataset description",
      "type": "qa" or "parsing",
      "sample_count": 500,
      "phases": {
        "phase_name": {
          "description": "Phase description",
          "raw_files": ["1_raw/DatasetName/phase/*.csv"],
          "clean_file": "2_clean/DatasetName/phase.csv",
          "models": ["model1", "model2"],
          "sample_count": 500,
          "status": "production",
          "metrics": ["ANLS", "CER", "WER"]
        }
      }
    }
  }
}
```

**Adding a new experiment:**
1. Add entry to `clean_experiments.json`
2. Run `python 2_clean/clean_files.py --incremental`

### visualization_config.json

Centralized color schemes, plot settings, and export configurations.

**Usage in notebooks:**
```python
import json
with open('../0_info/visualization_config.json') as f:
    config = json.load(f)

# Get color for a specific model
model_color = config['colors']['models']['gpt-5-mini']

# Get plot defaults
figsize = config['plot_defaults']['figsize']
```

**Key sections:**
- `colors.providers`: Brand colors for Azure, Mistral, OpenAI, Claude
- `colors.models`: Specific colors for each model
- `colors.datasets`: Colors for each dataset
- `colors.metrics`: Colors for different metrics
- `plot_defaults`: Figure size, font sizes, style
- `heatmap_defaults`: Heatmap-specific settings
- `export_formats`: Output formats for figures and tables

### pipeline_status.json

Tracks pipeline execution status and last update times.

**Updated automatically by:**
- `run_pipeline.py` - Master orchestrator
- `clean_files.py` - When consolidation completes
- Individual scripts as they run

## Notebooks

### 00_master_evaluation.ipynb

**Cross-dataset dashboard** showing:
- CER/WER/ANLS distributions by dataset
- Model performance heatmaps across all datasets
- Phase comparison charts (P-A vs P-B vs P-C, QA1 vs QA2 vs QA3)
- Correlation matrices between metrics
- Cost-performance analysis
- Executive summary with key findings

**Best for:** High-level overview, comparing models across experiments

### Dataset-Specific Notebooks (01-05)

Deep dives into individual datasets:

- **01_docvqa_analysis.ipynb**: DocVQA with 8 pipeline variants (QA1a-QA3b)
  - Impact of prompt engineering (generic vs task-aware)
  - Chain-of-thought vs simple prompting
  - Few-shot learning effects
  - OCR + LLM vs direct VLM comparison

- **02_infographicvqa_analysis.ipynb**: Infographic QA with pre-extracted OCR
  - Visual reasoning on complex infographics
  - Performance by question type
  - OCR quality impact on QA performance

- **03_iam_handwriting.ipynb**: Handwriting recognition analysis
  - OCR vs VLM on handwritten text
  - Text length effects on accuracy
  - Model comparison for cursive text

- **04_icdar_multilingual.ipynb**: Language-specific performance
  - Performance across 10 languages
  - Script-specific challenges (Latin, Chinese, Arabic, etc.)
  - Model language support comparison

- **05_voc2007_chinese.ipynb**: Medical Chinese document analysis
  - Medical terminology handling
  - Prompting impact on Chinese text extraction
  - Baseline vs prompted model comparison

**Notebook structure (all follow this template):**
1. **Header**: Dataset description, sample count, metrics
2. **Data Loading**: Use shared `data_loaders.py`
3. **Model Comparison**: Heatmaps, bar charts
4. **Phase Analysis**: Performance by phase with statistical tests
5. **Error Analysis**: Best/worst examples
6. **Conclusions**: Key findings, recommendations
7. **Exports**: Save figures to `3_notebooks/figures/[dataset]/`

### Shared Utilities (3_notebooks/shared/)

**colors.py:**
- Brand color palettes for providers and models
- Helper functions for consistent color usage
- Originally from `results_final/shared/colors.py`

**data_loaders.py:**
```python
def load_clean_results(dataset: str, phase: str) -> pd.DataFrame:
    """Load consolidated CSV from 2_clean/ folder"""

def load_metadata() -> dict:
    """Load clean_experiments.json"""

def load_execution_summary(dataset: str) -> dict:
    """Load execution_summary.json from raw results"""
```

**plot_utils.py:**
```python
def create_model_comparison_heatmap(df, metric, ...):
    """Standardized heatmap with colors.py"""

def create_phase_barplot(df, metric, ...):
    """Standardized bar chart"""

def create_error_analysis_table(df, n_examples=5):
    """Show best/worst predictions"""
```

## Pipeline Scripts

### clean_files.py (in 2_clean/)

**Purpose:** Validates and consolidates raw result files into clean CSVs.

**Usage:**
```bash
# Validate all files, move invalid to zzz_ignore/
python clean_files.py --validate-only

# Full consolidation (regenerate all 2_clean/ files)
python clean_files.py

# Incremental (only process new models)
python clean_files.py --incremental

# Dry run (preview changes without writing)
python clean_files.py --dry-run

# Load configuration from JSON
python clean_files.py --config ../0_info/clean_experiments.json

# Update metadata after validation
python clean_files.py --update-metadata
```

**Validation checks:**
- CSV has required columns (sample_id, prediction, ground_truth, metrics)
- All samples have values
- Error rate < 10%
- Sample count matches expected count
- No duplicate sample IDs

**Output:**
- Consolidated CSV files in `2_clean/[dataset]/[phase].csv`
- One row per sample, columns for each model's predictions and metrics
- Validation report showing files processed, errors found

### run_pipeline.py (in 0_info/)

**Purpose:** Master orchestrator for end-to-end pipeline execution.

**Usage:**
```bash
# Full pipeline: validate + consolidate + postprocess
python 0_info/run_pipeline.py --full

# Only validation
python 0_info/run_pipeline.py --validate

# Only consolidation
python 0_info/run_pipeline.py --consolidate

# Only postprocessing
python 0_info/run_pipeline.py --postprocess

# Specific dataset
python 0_info/run_pipeline.py --dataset DocVQA_mini --full
```

**Pipeline stages:**
1. **Validation**: Check all raw files for errors
2. **Consolidation**: Generate clean CSV files
3. **Postprocessing**: Aggregate statistics
4. **Status update**: Update `pipeline_status.json`

### run_notebooks.py (in 3_notebooks/)

**Purpose:** Execute notebooks programmatically for automated reporting.

**Usage:**
```bash
# Run all notebooks
python 3_notebooks/run_notebooks.py --all

# Run specific notebook
python 3_notebooks/run_notebooks.py --notebook 00_master_evaluation.ipynb

# Run with papermill for parameterized execution
python 3_notebooks/run_notebooks.py --notebook 01_docvqa_analysis.ipynb --params dataset=DocVQA_mini
```

## Adding New Results

### Step-by-Step Workflow

1. **Run experiment** - Generate raw CSV files with your model
   ```bash
   uv run python -m ocr_vs_vlm.benchmark_docvqa --sample-limit 500 --models your_model --phases QA1a
   ```

2. **Place files** - Results automatically saved to `1_raw/[dataset]/[phase]/`
   ```
   1_raw/DocVQA_mini/QA1a/your_model_results_20260127_120000.csv
   ```

3. **Update metadata** - Add entry to `clean_experiments.json`
   ```json
   "QA1a": {
     "models": ["azure_intelligence", "mistral_document_ai", "your_model"],
     ...
   }
   ```

4. **Validate** - Check for errors
   ```bash
   cd 2_clean
   python clean_files.py --validate-only
   ```

5. **Consolidate** - Generate clean CSV with new model column
   ```bash
   python clean_files.py --incremental
   ```

6. **Analyze** - Open relevant notebook and refresh data
   ```bash
   jupyter notebook ../3_notebooks/01_docvqa_analysis.ipynb
   ```

### File Naming Convention

Raw files must follow one of these patterns:
- `{model}_results_{timestamp}.csv` (standard)
- `{phase}_{model}_results_{timestamp}.csv` (with phase prefix)

Where:
- `model`: Lowercase with underscores (e.g., `gpt_5_mini`, `azure_intelligence`)
- `timestamp`: Optional, format `YYYYMMDD_HHMMSS`

**Examples:**
```
azure_intelligence_results_20260114_181918.csv
gpt-5-mini_results_20260127_120000.csv
QA1a_claude_sonnet_results.csv
```

### Required CSV Columns

All raw CSV files must include:
- `sample_id`: Unique identifier for each sample
- `prediction`: Model's predicted output
- `ground_truth`: Correct answer(s) (can be list for multiple valid answers)
- Metric columns (depends on task type):
  - **Parsing tasks**: `CER`, `WER`, `cosine_similarity`
  - **QA tasks**: `ANLS`, `EM`, `substring_match`

**Optional columns:**
- `prompt`: The prompt used
- `image_path`: Path to input image
- `execution_time`: Time taken for prediction
- `tokens_used`: Token count (for cost analysis)
- `error`: Error message if prediction failed

## Archive Folder (zzz_ignore/)

Contains test runs and failed experiments. **DO NOT DELETE** - these may be useful for debugging or historical reference.

### Contents

- `zzz/` - Original archive from `results/raw/zzz/`
- `publaynet_test/` - Test runs with only 5 samples
- `publaynet_incomplete/` - Incomplete publaynet runs
- `empty or bad/` - Files that failed validation
- Other test runs and incomplete experiments

### Restoring Files

To restore a file from archive:

1. Move file back to appropriate `1_raw/` location
   ```bash
   mv 1_raw/zzz_ignore/publaynet_test/P-A/azure_intelligence_results.csv 1_raw/publaynet_full/P-A/
   ```

2. Update `clean_experiments.json` to include it
   ```json
   "P-A": {
     "raw_files": ["1_raw/publaynet_full/P-A/*.csv"],
     "models": ["azure_intelligence", "mistral_document_ai"],
     ...
   }
   ```

3. Run incremental consolidation
   ```bash
   cd 2_clean
   python clean_files.py --incremental
   ```

## Troubleshooting

### "No files found for dataset X"

**Cause:** Files are not in the expected location or don't match naming convention.

**Solutions:**
1. Check file naming matches convention: `{model}_results_{timestamp}.csv`
2. Verify files are in correct `1_raw/[dataset]/[phase]/` subdirectory
3. Check `clean_experiments.json` has correct paths in `raw_files`
4. Run with debug mode: `python clean_files.py --validate-only --verbose`

### "Notebook can't load data"

**Cause:** Clean CSV files are missing or notebooks have incorrect paths.

**Solutions:**
1. Verify `2_clean/[dataset]/[phase].csv` files exist
2. Run consolidation: `python 2_clean/clean_files.py`
3. Check notebook is using correct relative paths:
   ```python
   df = pd.read_csv('../2_clean/DocVQA_mini/QA1a.csv')
   ```
4. Verify you're running notebook from `3_notebooks/` directory

### "Invalid CSV format"

**Cause:** CSV is missing required columns or has malformed data.

**Solutions:**
1. Run validation to see specific errors:
   ```bash
   python clean_files.py --validate-only
   ```
2. Check CSV has header row and required columns
3. Verify no missing values in critical columns (sample_id, prediction, ground_truth)
4. Check for encoding issues (use UTF-8)
5. Look at validation report for specific line numbers with errors

### "Metadata file not found"

**Cause:** JSON files in `0_info/` are missing or corrupted.

**Solutions:**
1. Check files exist:
   ```bash
   ls -la 0_info/
   ```
2. Validate JSON syntax:
   ```bash
   python -m json.tool 0_info/clean_experiments.json
   ```
3. Restore from git if corrupted:
   ```bash
   git checkout 0_info/clean_experiments.json
   ```

### "Permission denied" errors

**Cause:** File permissions are incorrect.

**Solutions:**
```bash
# Fix file permissions
chmod 644 1_raw/**/*.csv
chmod 755 2_clean/clean_files.py
chmod 755 0_info/run_pipeline.py
```

## Best Practices

### When Adding New Models

1. **Test on small sample first**: Use `--sample-limit 10` to verify pipeline works
2. **Check validation**: Run `--validate-only` before consolidation
3. **Update colors**: Add model color to `visualization_config.json`
4. **Document in notebooks**: Add notes about model in relevant analysis notebooks

### When Adding New Datasets

1. **Create metadata first**: Add full entry to `clean_experiments.json`
2. **Follow naming conventions**: Use consistent phase names (P-A/P-B/P-C or QA1a/QA1b/etc.)
3. **Copy notebook template**: Use existing notebook as template for new dataset analysis
4. **Add dataset color**: Define color in `visualization_config.json`

### Reproducibility

1. **Track versions**: Update `last_updated` in JSON files when making changes
2. **Document experiments**: Use `execution_summary.json` in each dataset folder
3. **Save random seeds**: Include seed values in experiment metadata
4. **Version control notebooks**: Commit notebooks with outputs cleared

### Performance Optimization

1. **Use incremental consolidation**: Only regenerate files that changed
2. **Archive old experiments**: Move to `zzz_ignore/` to reduce clutter
3. **Compress large files**: Use gzip for very large CSV files
4. **Use sampling for development**: Test notebooks on subset of data first

## Extensibility

### Adding a New Phase Type

Currently supports:
- **P-A/P-B/P-C**: Parsing tasks (OCR baseline, VLM generic, VLM task-aware)
- **QA1-QA3**: QA tasks with prompt variants

To add new phase type:
1. Define in `clean_experiments.json` with clear description
2. Add color to `visualization_config.json` under `colors.phases`
3. Update `clean_files.py` if special handling needed
4. Document in this README

### Adding a New Metric

To add custom evaluation metric:
1. Implement in `ocr_vs_vlm/evaluation_metrics.py`
2. Add to benchmark scripts to compute during evaluation
3. Add color to `visualization_config.json` under `colors.metrics`
4. Update `plot_utils.py` with visualization functions
5. Document metric in relevant notebooks

### Creating Custom Analysis

1. Copy template notebook from `3_notebooks/`
2. Import shared utilities:
   ```python
   from shared.data_loaders import load_clean_results, load_metadata
   from shared.plot_utils import create_model_comparison_heatmap
   from shared.colors import get_model_color
   ```
3. Load data using standardized functions
4. Create visualizations using plot utils
5. Save figures to `figures/` subdirectory
6. Export results to CSV/Excel for reporting

## Maintenance

### Weekly Tasks
- [ ] Run validation on all datasets: `python clean_files.py --validate-only`
- [ ] Check for new raw files and consolidate: `python clean_files.py --incremental`
- [ ] Update `pipeline_status.json` with latest run info

### Monthly Tasks
- [ ] Review archive folder size and compress if needed
- [ ] Update visualization config with any new models/datasets
- [ ] Re-run all notebooks to ensure they still work
- [ ] Check for broken links in metadata files

### Before Publishing Results
- [ ] Run full validation across all datasets
- [ ] Execute all notebooks from scratch
- [ ] Verify all figures are generated correctly
- [ ] Update metadata files with latest timestamps
- [ ] Review and update this README with any changes

## Version History

- **v1.0.0** (2026-01-27): Initial restructuring
  - Created numbered folder structure (0-4)
  - Established metadata registry system
  - Migrated working notebooks
  - Created comprehensive documentation
  - Set up archive system

## Support

For issues or questions:
1. Check this README first
2. Review troubleshooting section
3. Check `moved_files_log.md` for historical context
4. Review git history for recent changes
5. Contact research team leads

## Related Documentation

- **Main project README**: `/Users/kenzabenkirane/Documents/GitHub/research-playground/README.md`
- **Claude instructions**: `/Users/kenzabenkirane/Documents/GitHub/research-playground/.claude/CLAUDE.md`
- **Benchmark documentation**: `ocr_vs_vlm/README.md`
- **Dataset loaders**: `ocr_vs_vlm/dataset_loaders.py` and `dataset_loaders_qa.py`
- **Model API**: `ocr_vs_vlm/unified_model_api.py`
