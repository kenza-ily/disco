"""
Data loading utilities for OCR vs VLM evaluation.

Loads cleaned results from results_clean/ directory with consistent interfaces.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import warnings


# =============================================================================
# CONSTANTS
# =============================================================================

RESULTS_CLEAN_DIR = Path(__file__).parent.parent.parent / 'results_clean'

# Dataset configurations
DATASETS = {
    'DocVQA_mini': {
        'task': 'qa',
        'samples': 500,
        'phases': ['QA-OCR_LLM_simple', 'QA-OCR_LLM_detailed', 'QA-OCR_LLM_cot', 'QA-VLM_LLM_simple', 'QA-VLM_LLM_detailed', 'QA-VLM_LLM_cot', 'QA-VLM_direct_simple', 'QA-VLM_direct_detailed'],
        'metrics': ['anls_score', 'exact_match'],
    },
    'InfographicVQA_mini': {
        'task': 'qa',
        'samples': 500,
        'phases': ['QA-OCR_LLM_simple', 'QA-OCR_LLM_detailed', 'QA-OCR_LLM_cot', 'QA-VLM_LLM_simple', 'QA-VLM_LLM_detailed', 'QA-VLM_LLM_cot', 'QA-VLM_direct_simple', 'QA-VLM_direct_detailed', 'QA-EXT_OCR_LLM_simple', 'QA-EXT_OCR_LLM_detailed', 'QA-EXT_OCR_LLM_cot'],
        'metrics': ['anls_score', 'exact_match'],
    },
    'IAM_mini': {
        'task': 'parsing',
        'samples': 500,
        'phases': ['P-OCR', 'P-VLM_simple', 'P-VLM_taskaware'],
        'metrics': ['cer', 'wer', 'anls'],
    },
    'ICDAR_mini': {
        'task': 'parsing',
        'samples': 500,
        'phases': ['P-OCR', 'P-VLM_simple', 'P-VLM_taskaware'],
        'metrics': ['cer', 'wer', 'anls'],
    },
    'VOC2007': {
        'task': 'parsing',
        'samples': 238,
        'phases': ['P-OCR', 'P-VLM_simple', 'P-VLM_taskaware', 'P-VLM_domainspecific'],
        'metrics': ['cer', 'wer', 'anls'],
    },
    'publaynet': {
        'task': 'layout',
        'samples': 500,
        'phases': ['PL-OCR', 'PL-VLM', 'PL-OCR_x_VLM'],
        'metrics': ['iou', 'f1'],
    },
}

# Phase to approach mapping (NEW names → approach)
PHASE_TO_APPROACH = {
    # QA phases
    'QA-OCR_LLM_simple': 'ocr_pipeline', 'QA-OCR_LLM_detailed': 'ocr_pipeline', 'QA-OCR_LLM_cot': 'ocr_pipeline',
    'QA-VLM_LLM_simple': 'vlm_pipeline', 'QA-VLM_LLM_detailed': 'vlm_pipeline', 'QA-VLM_LLM_cot': 'vlm_pipeline',
    'QA-VLM_direct_simple': 'direct_vqa', 'QA-VLM_direct_detailed': 'direct_vqa',
    'QA-EXT_OCR_LLM_simple': 'preextracted', 'QA-EXT_OCR_LLM_detailed': 'preextracted', 'QA-EXT_OCR_LLM_cot': 'preextracted',
    # Parsing phases
    'P-OCR': 'ocr_baseline', 'P-VLM_simple': 'vlm_generic', 
    'P-VLM_taskaware': 'vlm_task_aware', 'P-VLM_domainspecific': 'vlm_domain_specific',
    'phase_3a': 'vlm_task_aware',  # Keep for backward compatibility
    # Layout phases
    'PL-OCR': 'ocr_layout', 'PL-VLM': 'vlm_direct', 'PL-OCR_x_VLM': 'vlm_hybrid',
}

# CRITICAL: Mapping from OLD file names (in results_clean) to NEW display names (in notebooks)
# This allows notebooks to use new descriptive names while reading old-named files
# NOTE: Some datasets already use new names (P-A, P-B, P-C) - these pass through unchanged
OLD_TO_NEW_PHASE = {
    # QA phases: OLD → NEW
    'QA1a': 'QA-OCR_LLM_simple',
    'QA1b': 'QA-OCR_LLM_detailed',
    'QA1c': 'QA-OCR_LLM_cot',
    'QA2a': 'QA-VLM_LLM_simple',
    'QA2b': 'QA-VLM_LLM_detailed',
    'QA2c': 'QA-VLM_LLM_cot',
    'QA3a': 'QA-VLM_direct_simple',
    'QA3b': 'QA-VLM_direct_detailed',
    'QA4a': 'QA-EXT_OCR_LLM_simple',
    'QA4b': 'QA-EXT_OCR_LLM_detailed',
    'QA4c': 'QA-EXT_OCR_LLM_cot',
    # Combined QA (all phases in one file)
    'QA': 'QA-combined',
    # Parsing phases (variant 1: P1, P2, P3 format) → NEW
    'P1': 'P-OCR',
    'P2': 'P-VLM_simple',
    'P3': 'P-VLM_taskaware',
    'P4': 'P-VLM_domainspecific',
    # Parsing phases (variant 2: phase_1, phase_2, phase_3 format) → NEW
    'phase_1': 'P-OCR',
    'phase_2': 'P-VLM_simple',
    'phase_3': 'P-VLM_taskaware',
    'phase_3a': 'P-VLM_taskaware',
    'phase_4': 'P-VLM_domainspecific',
    # Layout phases (variant 1: already using NEW names)
    'P-A': 'PL-OCR',
    'P-B': 'PL-VLM',
    'P-C': 'PL-OCR_x_VLM',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def get_available_experiments(base_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Discover available experiments by scanning results_clean directory.
    
    Args:
        base_dir: Base directory for results (default: results_clean/)
    
    Returns:
        Dict mapping dataset names to lists of available phase files
    """
    base_dir = base_dir or RESULTS_CLEAN_DIR
    
    experiments = {}
    
    for dataset_dir in base_dir.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('_'):
            csv_files = list(dataset_dir.glob('*.csv'))
            # Filter out concatenated/combined files
            csv_files = [f for f in csv_files if ' ' not in f.stem and f.stem != 'QA']
            if csv_files:
                experiments[dataset_dir.name] = [f.stem for f in csv_files]
    
    return experiments


def load_experiment_data(
    dataset: str,
    phase: str,
    base_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load data for a specific experiment (dataset + phase).
    
    IMPORTANT: Phase parameter can be either OLD name (e.g., 'QA1a') or NEW name (e.g., 'QA-OCR_LLM_simple').
    The function automatically handles the mapping and returns data with NEW phase names.
    
    Args:
        dataset: Dataset name (e.g., 'DocVQA_mini')
        phase: Phase name - either OLD (QA1a, P1, etc.) or NEW (QA-OCR_LLM_simple, P-OCR, etc.)
        base_dir: Base directory for results
    
    Returns:
        DataFrame with experiment results, with 'phase' column set to NEW name
    """
    base_dir = base_dir or RESULTS_CLEAN_DIR
    
    # Determine which file to load (use OLD name if provided with NEW name)
    file_phase = phase
    display_phase = phase
    
    # If NEW name provided, find OLD name to load file
    if phase in OLD_TO_NEW_PHASE.values():
        # This is a NEW name, find the OLD one
        for old_name, new_name in OLD_TO_NEW_PHASE.items():
            if new_name == phase:
                file_phase = old_name
                display_phase = phase
                break
    else:
        # This is an OLD name, convert to NEW for display
        display_phase = OLD_TO_NEW_PHASE.get(phase, phase)
        file_phase = phase
    
    file_path = base_dir / dataset / f'{file_phase}.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {file_path}\n"
                              f"  (looked for file with phase '{file_phase}' in {base_dir / dataset})")
    
    df = pd.read_csv(file_path)
    df['dataset'] = dataset
    df['phase'] = display_phase  # Use NEW name for display
    df['approach'] = PHASE_TO_APPROACH.get(display_phase, 'unknown')
    
    return df


def load_dataset_data(
    dataset: str,
    phases: Optional[List[str]] = None,
    base_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load all experiment data for a dataset.
    
    Args:
        dataset: Dataset name
        phases: List of phases to load (default: all available)
        base_dir: Base directory for results
    
    Returns:
        Combined DataFrame with all phases
    """
    base_dir = base_dir or RESULTS_CLEAN_DIR
    
    dataset_dir = base_dir / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Get available phases, filtering out concatenated/combined files
    available_phases = [f.stem for f in dataset_dir.glob('*.csv') if ' ' not in f.stem and f.stem != 'QA']
    
    if phases is None:
        phases = available_phases
    else:
        phases = [p for p in phases if p in available_phases]
    
    if not phases:
        raise ValueError(f"No valid phases found for dataset {dataset}")
    
    dfs = []
    for phase in phases:
        try:
            df = load_experiment_data(dataset, phase, base_dir)
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Failed to load {dataset}/{phase}: {e}")
    
    if not dfs:
        raise ValueError(f"No data loaded for dataset {dataset}")
    
    return pd.concat(dfs, ignore_index=True)


def load_all_data(
    datasets: Optional[List[str]] = None,
    task_type: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load data from multiple datasets.
    
    Args:
        datasets: List of dataset names (default: all)
        task_type: Filter by task type ('qa', 'parsing', 'layout')
        base_dir: Base directory for results
    
    Returns:
        Combined DataFrame with all datasets
    """
    base_dir = base_dir or RESULTS_CLEAN_DIR
    
    if datasets is None:
        datasets = list(DATASETS.keys())
    
    if task_type:
        datasets = [d for d in datasets if DATASETS.get(d, {}).get('task') == task_type]
    
    dfs = []
    for dataset in datasets:
        try:
            df = load_dataset_data(dataset, base_dir=base_dir)
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Failed to load {dataset}: {e}")
    
    if not dfs:
        raise ValueError("No data loaded")
    
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# MODEL EXTRACTION
# =============================================================================

def extract_models_from_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract model names from DataFrame columns.
    
    Columns follow pattern: prediction_{model}, anls_score_{model}, etc.
    
    Args:
        df: DataFrame with model-specific columns
    
    Returns:
        List of unique model names
    """
    models = set()
    
    prefixes = ['prediction_', 'anls_score_', 'exact_match_', 'cer_', 'wer_', 
                'inference_time_ms_', 'iou_', 'f1_']
    
    for col in df.columns:
        for prefix in prefixes:
            if col.startswith(prefix):
                model = col[len(prefix):]
                if model and not model.startswith('_'):
                    models.add(model)
                break
    
    return sorted(list(models))


def extract_metric_scores(
    df: pd.DataFrame,
    metric: str = 'anls_score'
) -> Dict[str, np.ndarray]:
    """
    Extract metric scores for each model from DataFrame.
    
    Args:
        df: DataFrame with metric columns
        metric: Metric prefix (e.g., 'anls_score', 'cer', 'exact_match')
    
    Returns:
        Dict mapping model names to score arrays
    """
    models = extract_models_from_columns(df)
    scores = {}
    
    for model in models:
        col = f'{metric}_{model}'
        if col in df.columns:
            scores[model] = df[col].values
    
    return scores


# =============================================================================
# AGGREGATION
# =============================================================================

def compute_summary_statistics(
    df: pd.DataFrame,
    metric: str = 'anls_score',
    group_by: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute summary statistics (mean, std, CI) for a metric.
    
    Args:
        df: DataFrame with metric columns
        metric: Metric prefix
        group_by: Columns to group by (e.g., ['dataset', 'phase'])
    
    Returns:
        Summary DataFrame
    """
    from .stats_utils import bootstrap_ci
    
    models = extract_models_from_columns(df)
    
    if group_by:
        groups = df.groupby(group_by)
    else:
        groups = [(None, df)]
    
    rows = []
    for group_key, group_df in groups:
        for model in models:
            col = f'{metric}_{model}'
            if col not in group_df.columns:
                continue
            
            values = group_df[col].dropna().values
            if len(values) == 0:
                continue
            
            mean, ci_lo, ci_hi = bootstrap_ci(values, 'mean')
            
            row = {
                'model': model,
                'metric': metric,
                'mean': mean,
                'std': np.std(values),
                'ci_lower': ci_lo,
                'ci_upper': ci_hi,
                'n': len(values),
            }
            
            if group_by:
                if isinstance(group_key, tuple):
                    for col_name, val in zip(group_by, group_key):
                        row[col_name] = val
                else:
                    row[group_by[0]] = group_key
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def compute_approach_comparison(
    df: pd.DataFrame,
    metric: str = 'anls_score'
) -> pd.DataFrame:
    """
    Aggregate results by approach (OCR pipeline, VLM pipeline, Direct VQA).
    
    Args:
        df: DataFrame with 'approach' column and metric columns
        metric: Metric to aggregate
    
    Returns:
        Summary DataFrame by approach
    """
    if 'approach' not in df.columns:
        raise ValueError("DataFrame must have 'approach' column")
    
    return compute_summary_statistics(df, metric, group_by=['approach'])


# =============================================================================
# COST ANALYSIS
# =============================================================================

def load_pricing() -> Dict[str, Dict[str, float]]:
    """
    Load model pricing information from prices.json.
    
    Returns:
        Dict mapping model names to pricing info
    """
    prices_path = Path(__file__).parent.parent.parent.parent / 'llms' / 'prices.json'
    
    if prices_path.exists():
        with open(prices_path) as f:
            return json.load(f)
    
    # Fallback pricing (per 1M tokens or per 1K pages)
    return {
        'azure_intelligence': {'type': 'page', 'cost_per_1k': 1.50},
        'mistral_document_ai': {'type': 'page', 'cost_per_1k': 3.00},
        'gpt-5-mini': {'type': 'token', 'input_per_1m': 0.25, 'output_per_1m': 2.00},
        'gpt-5-nano': {'type': 'token', 'input_per_1m': 0.05, 'output_per_1m': 0.40},
        'claude_sonnet': {'type': 'token', 'input_per_1m': 3.00, 'output_per_1m': 15.00},
        'claude_haiku': {'type': 'token', 'input_per_1m': 0.25, 'output_per_1m': 1.25},
    }


def compute_cost_performance(
    summary_df: pd.DataFrame,
    metric: str = 'mean',
    pricing: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Add cost estimates to summary statistics.
    
    Args:
        summary_df: Summary DataFrame from compute_summary_statistics
        metric: Column to use as performance metric
        pricing: Pricing dict (default: load from file)
    
    Returns:
        DataFrame with 'estimated_cost' column added
    """
    pricing = pricing or load_pricing()
    
    df = summary_df.copy()
    
    # Estimate cost per sample (rough approximation)
    costs = []
    for _, row in df.iterrows():
        model = row['model']
        if model in pricing:
            info = pricing[model]
            if info.get('type') == 'page':
                # Assume 1 page per sample
                cost = info['cost_per_1k'] / 1000
            else:
                # Assume ~1000 input tokens, ~100 output tokens per sample
                cost = (1000 * info.get('input_per_1m', 0) / 1_000_000 +
                       100 * info.get('output_per_1m', 0) / 1_000_000)
        else:
            cost = np.nan
        costs.append(cost)
    
    df['cost_per_sample'] = costs
    
    return df
