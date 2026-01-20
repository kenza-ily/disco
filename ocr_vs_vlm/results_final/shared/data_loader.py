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
        'phases': ['QA1a', 'QA1b', 'QA1c', 'QA2a', 'QA2b', 'QA2c', 'QA3a', 'QA3b'],
        'metrics': ['anls_score', 'exact_match'],
    },
    'InfographicVQA_mini': {
        'task': 'qa',
        'samples': 500,
        'phases': ['QA1a', 'QA1b', 'QA1c', 'QA2a', 'QA2b', 'QA2c', 'QA3a', 'QA3b', 'QA4a', 'QA4b', 'QA4c'],
        'metrics': ['anls_score', 'exact_match'],
    },
    'IAM_mini': {
        'task': 'parsing',
        'samples': 500,
        'phases': ['phase_1', 'phase_2', 'phase_3'],
        'metrics': ['cer', 'wer', 'anls'],
    },
    'ICDAR_mini': {
        'task': 'parsing',
        'samples': 500,
        'phases': ['phase_1', 'phase_2', 'phase_3'],
        'metrics': ['cer', 'wer', 'anls'],
    },
    'VOC2007': {
        'task': 'parsing',
        'samples': 238,
        'phases': ['phase_1', 'phase_2', 'phase_3', 'phase_4'],
        'metrics': ['cer', 'wer', 'anls'],
    },
    'publaynet': {
        'task': 'layout',
        'samples': 500,
        'phases': ['P-A', 'P-B', 'P-C'],
        'metrics': ['iou', 'f1'],
    },
}

# Phase to approach mapping
PHASE_TO_APPROACH = {
    # QA phases
    'QA1a': 'ocr_pipeline', 'QA1b': 'ocr_pipeline', 'QA1c': 'ocr_pipeline',
    'QA2a': 'vlm_pipeline', 'QA2b': 'vlm_pipeline', 'QA2c': 'vlm_pipeline',
    'QA3a': 'direct_vqa', 'QA3b': 'direct_vqa',
    'QA4a': 'preextracted', 'QA4b': 'preextracted', 'QA4c': 'preextracted',
    # Parsing phases
    'phase_1': 'ocr_baseline', 'phase_2': 'vlm_generic', 
    'phase_3': 'vlm_task_aware', 'phase_4': 'vlm_domain_specific',
    'phase_3a': 'vlm_task_aware',
    # Layout phases
    'P-A': 'ocr_layout', 'P-B': 'vlm_direct', 'P-C': 'vlm_hybrid',
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
    
    Args:
        dataset: Dataset name (e.g., 'DocVQA_mini')
        phase: Phase name (e.g., 'QA1a', 'phase_1')
        base_dir: Base directory for results
    
    Returns:
        DataFrame with experiment results
    """
    base_dir = base_dir or RESULTS_CLEAN_DIR
    
    file_path = base_dir / dataset / f'{phase}.csv'
    
    if not file_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['dataset'] = dataset
    df['phase'] = phase
    df['approach'] = PHASE_TO_APPROACH.get(phase, 'unknown')
    
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
    
    available_phases = [f.stem for f in dataset_dir.glob('*.csv')]
    
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
