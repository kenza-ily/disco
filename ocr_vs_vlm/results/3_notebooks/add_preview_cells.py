#!/usr/bin/env python3
"""
Add data preview cells to all analysis notebooks.

This script adds a data preview cell after the data loading section
showing ground truth vs predictions for 10 random samples.
"""

import json
from pathlib import Path
from typing import Dict, List


# Notebook configurations
NOTEBOOK_CONFIGS = {
    "00_master_evaluation.ipynb": {
        "type": "multi-dataset",
        "insert_after_text": "Loading QA Task Summaries:",
        "preview_datasets": [
            {"name": "DocVQA_mini", "phase": "QA1a", "type": "qa"},
            {"name": "IAM_mini", "phase": "phase_1", "type": "parsing"}
        ]
    },
    "01_docvqa_analysis.ipynb": {
        "type": "qa",
        "dataset": "DocVQA_mini",
        "phase": "QA1a",
        "insert_after_text": "# Load data"
    },
    "02_infographicvqa_analysis.ipynb": {
        "type": "qa",
        "dataset": "InfographicVQA_mini",
        "phase": "QA1a",
        "insert_after_text": "# Load data"
    },
    "03_iam_handwriting.ipynb": {
        "type": "parsing",
        "dataset": "IAM_mini",
        "phase": "phase_1",
        "insert_after_text": "# Load data"
    },
    "04_icdar_multilingual.ipynb": {
        "type": "parsing",
        "dataset": "ICDAR_mini",
        "phase": "phase_1",
        "insert_after_text": "# Load data"
    },
    "05_voc2007_chinese.ipynb": {
        "type": "parsing",
        "dataset": "VOC2007",
        "phase": "phase_1",
        "insert_after_text": "# Load data"
    }
}


def create_qa_preview_cell(dataset: str, phase: str) -> Dict:
    """Create a data preview cell for QA datasets."""
    code = f'''# ============================================================================
# DATA PREVIEW: Ground Truth vs Predictions (10 Random Samples)
# ============================================================================

# Load one phase to show examples
phase_to_preview = '{phase}'
dataset_name = '{dataset}'

preview_file = f"../2_clean/{{dataset_name}}/{{phase_to_preview}}.csv"

if Path(preview_file).exists():
    df_preview = pd.read_csv(preview_file)

    # Get 10 random samples
    random_samples = df_preview.sample(n=min(10, len(df_preview)), random_state=42)

    # Extract columns for preview
    columns_to_show = ['sample_id', 'question', 'ground_truths']

    # Add prediction columns (find all columns starting with 'prediction_')
    pred_cols = [col for col in df_preview.columns if col.startswith('prediction_')]
    columns_to_show.extend(pred_cols[:2])  # Show first 2 models

    # Create display dataframe
    display_df = random_samples[columns_to_show].copy()

    # Truncate long strings for readability
    for col in display_df.columns:
        if display_df[col].dtype == 'object':
            display_df[col] = display_df[col].apply(
                lambda x: str(x)[:100] + '...' if pd.notna(x) and len(str(x)) > 100 else x
            )

    print(f"\\n{{'='*100}}")
    print(f"DATA PREVIEW: {{dataset_name}} - {{phase_to_preview}}")
    print(f"Showing 10 random samples with ground truth and first 2 model predictions")
    print(f"{{'='*100}}\\n")

    display(display_df)

    print(f"\\nTotal samples in {{phase_to_preview}}: {{len(df_preview)}}")
    print(f"Available models: {{', '.join([col.replace('prediction_', '') for col in pred_cols])}}")
else:
    print(f"Preview file not found: {{preview_file}}")'''

    # Split by newline and add newline character to each line (except last)
    lines = code.split('\n')
    source = [line + '\n' for line in lines[:-1]] + [lines[-1]]

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }


def create_parsing_preview_cell(dataset: str, phase: str) -> Dict:
    """Create a data preview cell for parsing datasets."""
    code = f'''# ============================================================================
# DATA PREVIEW: Ground Truth vs Predictions (10 Random Samples)
# ============================================================================

# Load one phase to show examples
phase_to_preview = '{phase}'
dataset_name = '{dataset}'

preview_file = f"../2_clean/{{dataset_name}}/{{phase_to_preview}}.csv"

if Path(preview_file).exists():
    df_preview = pd.read_csv(preview_file)

    # Get 10 random samples
    random_samples = df_preview.sample(n=min(10, len(df_preview)), random_state=42)

    # Extract columns for preview
    columns_to_show = ['sample_id', 'ground_truth']

    # Add prediction columns (find all columns starting with 'prediction_')
    pred_cols = [col for col in df_preview.columns if col.startswith('prediction_')]
    columns_to_show.extend(pred_cols[:3])  # Show first 3 models

    # Create display dataframe
    display_df = random_samples[columns_to_show].copy()

    # Truncate long strings for readability
    for col in display_df.columns:
        if display_df[col].dtype == 'object':
            display_df[col] = display_df[col].apply(
                lambda x: str(x)[:80] + '...' if pd.notna(x) and len(str(x)) > 80 else x
            )

    print(f"\\n{{'='*100}}")
    print(f"DATA PREVIEW: {{dataset_name}} - {{phase_to_preview}}")
    print(f"Showing 10 random samples with ground truth and first 3 model predictions")
    print(f"{{'='*100}}\\n")

    display(display_df)

    print(f"\\nTotal samples in {{phase_to_preview}}: {{len(df_preview)}}")
    print(f"Available models: {{', '.join([col.replace('prediction_', '') for col in pred_cols])}}")

    # Show metric columns for these samples
    metric_cols = [col for col in df_preview.columns if any(
        metric in col for metric in ['CER', 'WER', 'ANLS', 'EM', '_cer_', '_wer_', '_anls_']
    )]

    if metric_cols:
        print(f"\\nMetric Preview (same 10 samples):")
        metric_display = random_samples[['sample_id'] + metric_cols[:6]].copy()
        display(metric_display)
else:
    print(f"Preview file not found: {{preview_file}}")'''

    # Split by newline and add newline character to each line (except last)
    lines = code.split('\n')
    source = [line + '\n' for line in lines[:-1]] + [lines[-1]]

    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }


def create_markdown_header() -> Dict:
    """Create markdown header for preview section."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 📊 Data Preview\n", "\n", "Quick look at 10 random samples showing ground truth vs model predictions."]
    }


def find_insertion_point(cells: List[Dict], search_text: str) -> int:
    """Find the cell index to insert after based on search text."""
    for idx, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                source_text = ''.join(source)
            else:
                source_text = source

            if search_text in source_text:
                return idx + 1

    # Default: insert after 3rd cell (usually after imports)
    return min(3, len(cells))


def add_preview_to_notebook(notebook_path: Path, config: Dict) -> bool:
    """Add preview cell(s) to a notebook."""
    try:
        # Load notebook
        with open(notebook_path, 'r') as f:
            nb = json.load(f)

        cells = nb.get('cells', [])

        # Check if preview already exists
        for cell in cells:
            source = cell.get('source', [])
            if isinstance(source, list):
                source_text = ''.join(source)
            else:
                source_text = source

            if 'DATA PREVIEW: Ground Truth vs Predictions' in source_text:
                print(f"  ⚠️  Preview already exists in {notebook_path.name}")
                return False

        # Find insertion point
        insert_after = config.get('insert_after_text', '# Load data')
        insert_idx = find_insertion_point(cells, insert_after)

        # Create preview cells
        new_cells = [create_markdown_header()]

        if config['type'] == 'multi-dataset':
            # Add preview for each dataset
            for ds_config in config['preview_datasets']:
                if ds_config['type'] == 'qa':
                    new_cells.append(create_qa_preview_cell(ds_config['name'], ds_config['phase']))
                else:
                    new_cells.append(create_parsing_preview_cell(ds_config['name'], ds_config['phase']))
        elif config['type'] == 'qa':
            new_cells.append(create_qa_preview_cell(config['dataset'], config['phase']))
        else:  # parsing
            new_cells.append(create_parsing_preview_cell(config['dataset'], config['phase']))

        # Insert cells
        cells[insert_idx:insert_idx] = new_cells
        nb['cells'] = cells

        # Save notebook
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)

        print(f"  ✅ Added preview cell(s) to {notebook_path.name}")
        return True

    except Exception as e:
        print(f"  ❌ Error processing {notebook_path.name}: {e}")
        return False


def main():
    """Main function to add preview cells to all notebooks."""
    notebooks_dir = Path(__file__).parent

    print("\n" + "="*80)
    print("ADDING DATA PREVIEW CELLS TO NOTEBOOKS")
    print("="*80 + "\n")

    success_count = 0
    skip_count = 0
    error_count = 0

    for notebook_name, config in NOTEBOOK_CONFIGS.items():
        notebook_path = notebooks_dir / notebook_name

        if not notebook_path.exists():
            print(f"  ⚠️  Notebook not found: {notebook_name}")
            skip_count += 1
            continue

        result = add_preview_to_notebook(notebook_path, config)
        if result:
            success_count += 1
        elif result is False:
            skip_count += 1
        else:
            error_count += 1

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")
    print()


if __name__ == "__main__":
    main()
