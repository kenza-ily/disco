"""
VOC2007 Medical Lab Reports Results Consolidation Script

Consolidates benchmark results from multiple models into comparison tables.
- Evaluates columns common to all models
- Renames columns with {model_name}_{column_name} prefix
- Outputs consolidated CSVs for each phase

Dataset: Chinese Medical Lab Reports (Simplified Chinese)
Phases:
- Phase 1: OCR baseline (azure_intelligence, mistral_document_ai)
- Phase 2: VLM baseline with generic Chinese-aware prompts
- Phase 3a: VLM with intermediate context (language + document type)
- Phase 4: VLM with detailed context-aware prompts for medical lab reports
"""

import csv
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional
import pandas as pd

# Column names based on benchmark.py structure
COLUMN_NAMES = [
    'sample_id', 'image_path', 'dataset', 'model', 'phase', 'language', 
    'ground_truth', 'prediction', 'prompt', 'inference_time_ms', 
    'tokens_used', 'error', 'timestamp'
]

# Columns that should be kept as common (not prefixed with model name)
COMMON_COLUMNS = ['sample_id', 'image_path', 'dataset', 'language', 'ground_truth']

# Columns to prefix with model name
MODEL_SPECIFIC_COLUMNS = ['prediction', 'prompt', 'inference_time_ms', 'tokens_used', 'error', 'timestamp']


def load_csv_with_multiline(file_path: Path) -> List[Dict]:
    """
    Load a CSV file that may have multiline cells.
    Uses pandas for robust CSV parsing.
    """
    try:
        df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES, 
                         dtype=str, na_values=[''], keep_default_na=False)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def find_results_files(results_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Find all result files organized by phase and model.
    
    VOC2007 structure:
    results/VOC2007/{model_name}/VOC2007/{model_name}/phase_X_results.csv
    
    Returns:
        Dict[phase_name, Dict[model_name, file_path]]
    """
    results = defaultdict(dict)
    
    # Iterate through model folders directly under VOC2007
    for model_folder in results_dir.iterdir():
        if not model_folder.is_dir():
            continue
        
        model_name = model_folder.name
        
        # Navigate to the nested structure: VOC2007/{model_name}/
        nested_path = model_folder / "VOC2007" / model_name
        
        if not nested_path.exists():
            continue
        
        for csv_file in nested_path.glob("phase_*_results.csv"):
            # Extract phase name (e.g., "phase_1", "phase_2", "phase_3a", "phase_4")
            phase_name = csv_file.stem.replace("_results", "")
            results[phase_name][model_name] = csv_file
    
    return dict(results)


def consolidate_phase_results(phase_files: Dict[str, Path], output_file: Path) -> Optional[pd.DataFrame]:
    """
    Consolidate results from multiple models for a single phase.
    
    Args:
        phase_files: Dict mapping model_name -> csv_file_path
        output_file: Path to write consolidated CSV
        
    Returns:
        Consolidated DataFrame or None if failed
    """
    if not phase_files:
        print("No files to consolidate")
        return None
    
    print(f"\nConsolidating {len(phase_files)} models: {list(phase_files.keys())}")
    
    # Load all model results
    model_dfs = {}
    for model_name, file_path in phase_files.items():
        print(f"  Loading {model_name} from {file_path.name}...")
        records = load_csv_with_multiline(file_path)
        if records:
            df = pd.DataFrame(records)
            df = df.set_index('sample_id')
            model_dfs[model_name] = df
            print(f"    Loaded {len(df)} samples")
        else:
            print(f"    WARNING: No records loaded")
    
    if not model_dfs:
        print("No data loaded from any model")
        return None
    
    # Find common sample_ids across all models
    common_sample_ids = None
    for model_name, df in model_dfs.items():
        if common_sample_ids is None:
            common_sample_ids = set(df.index)
        else:
            common_sample_ids &= set(df.index)
    
    print(f"\n  Found {len(common_sample_ids)} samples common to all models")
    
    if not common_sample_ids:
        print("  WARNING: No common samples found. Using all samples with NaN for missing.")
        # Get union of all sample_ids
        all_sample_ids = set()
        for df in model_dfs.values():
            all_sample_ids |= set(df.index)
        common_sample_ids = all_sample_ids
    
    # Build consolidated DataFrame
    # Start with common columns from first model
    first_model = list(model_dfs.keys())[0]
    first_df = model_dfs[first_model]
    
    # Create base DataFrame with common columns
    consolidated = first_df.loc[first_df.index.isin(common_sample_ids), 
                                 [c for c in COMMON_COLUMNS if c in first_df.columns and c != 'sample_id']].copy()
    
    # Add model-specific columns with prefixes
    for model_name, df in model_dfs.items():
        model_data = df.loc[df.index.isin(common_sample_ids)]
        
        for col in MODEL_SPECIFIC_COLUMNS:
            if col in model_data.columns:
                new_col_name = f"{model_name}_{col}"
                consolidated[new_col_name] = model_data[col]
    
    # Reset index to make sample_id a column again
    consolidated = consolidated.reset_index()
    
    # Sort by sample_id
    consolidated = consolidated.sort_values('sample_id')
    
    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    consolidated.to_csv(output_file, index=False)
    print(f"\n  Saved consolidated results to: {output_file}")
    print(f"  Shape: {consolidated.shape}")
    
    return consolidated


def generate_summary_stats(consolidated_df: pd.DataFrame, phase_name: str) -> pd.DataFrame:
    """Generate summary statistics for each model in the consolidated data."""
    stats = []
    
    # Find all model names by looking at columns
    models = set()
    for col in consolidated_df.columns:
        for suffix in MODEL_SPECIFIC_COLUMNS:
            if col.endswith(f"_{suffix}"):
                model = col.replace(f"_{suffix}", "")
                models.add(model)
    
    for model in sorted(models):
        model_stats = {
            'model': model,
            'phase': phase_name,
            'total_samples': len(consolidated_df),
        }
        
        # Inference time stats
        time_col = f"{model}_inference_time_ms"
        if time_col in consolidated_df.columns:
            times = pd.to_numeric(consolidated_df[time_col], errors='coerce')
            model_stats['avg_inference_time_ms'] = times.mean()
            model_stats['median_inference_time_ms'] = times.median()
            model_stats['min_inference_time_ms'] = times.min()
            model_stats['max_inference_time_ms'] = times.max()
        
        # Error rate
        error_col = f"{model}_error"
        if error_col in consolidated_df.columns:
            errors = consolidated_df[error_col].fillna('').astype(str)
            error_count = (errors != '').sum()
            model_stats['error_count'] = error_count
            model_stats['error_rate'] = error_count / len(consolidated_df) * 100 if len(consolidated_df) > 0 else 0
        
        # Token usage
        tokens_col = f"{model}_tokens_used"
        if tokens_col in consolidated_df.columns:
            tokens = pd.to_numeric(consolidated_df[tokens_col], errors='coerce')
            model_stats['avg_tokens'] = tokens.mean()
            model_stats['total_tokens'] = tokens.sum()
        
        # Prediction availability
        pred_col = f"{model}_prediction"
        if pred_col in consolidated_df.columns:
            predictions = consolidated_df[pred_col].fillna('')
            non_empty = (predictions != '').sum()
            model_stats['predictions_count'] = non_empty
            model_stats['prediction_rate'] = non_empty / len(consolidated_df) * 100 if len(consolidated_df) > 0 else 0
        
        stats.append(model_stats)
    
    return pd.DataFrame(stats)


def main():
    # Paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent.parent / "results" / "VOC2007"
    output_dir = script_dir
    
    print("=" * 60)
    print("VOC2007 Medical Lab Reports Results Consolidation")
    print("=" * 60)
    print(f"\nResults directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all result files
    phase_files = find_results_files(results_dir)
    
    if not phase_files:
        print("\nNo result files found!")
        return
    
    print(f"\nFound phases: {list(phase_files.keys())}")
    
    all_summaries = []
    
    # Process each phase
    for phase_name in sorted(phase_files.keys()):
        print(f"\n{'=' * 60}")
        print(f"Processing {phase_name}")
        print("=" * 60)
        
        output_file = output_dir / f"{phase_name}_consolidated.csv"
        
        consolidated_df = consolidate_phase_results(
            phase_files[phase_name], 
            output_file
        )
        
        if consolidated_df is not None:
            # Generate and save summary stats
            summary_df = generate_summary_stats(consolidated_df, phase_name)
            summary_file = output_dir / f"{phase_name}_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"\n  Summary saved to: {summary_file}")
            print(summary_df.to_string(index=False))
            
            all_summaries.append(summary_df)
    
    # Combine all summaries
    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_file = output_dir / "all_phases_summary.csv"
        combined_summary.to_csv(combined_file, index=False)
        print(f"\n\nCombined summary saved to: {combined_file}")
        print("\n" + "=" * 60)
        print("ALL PHASES SUMMARY")
        print("=" * 60)
        print(combined_summary.to_string(index=False))
    
    print("\n\nDone!")


if __name__ == "__main__":
    main()
