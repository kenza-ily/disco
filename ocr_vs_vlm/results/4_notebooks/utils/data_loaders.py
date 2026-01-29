"""
Data loading utilities for results analysis notebooks.

This module provides standardized functions to load data from the
clean results folder and metadata files.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union


# Base paths relative to notebooks folder
RESULTS_BASE = Path(__file__).parent.parent.parent
RAW_PATH = RESULTS_BASE / "1_raw"
CLEAN_PATH = RESULTS_BASE / "2_clean"
INFO_PATH = RESULTS_BASE / "0_info"
POSTPROCESSING_PATH = RESULTS_BASE / "4_postprocessing"


def load_clean_results(dataset: str, phase: str) -> pd.DataFrame:
    """
    Load consolidated CSV from 2_clean/ folder.

    Args:
        dataset: Dataset name (e.g., 'DocVQA_mini', 'IAM_mini')
        phase: Phase name (e.g., 'QA1a', 'P-A')

    Returns:
        DataFrame with columns for each model's predictions and metrics

    Example:
        df = load_clean_results('DocVQA_mini', 'QA1a')
        print(df.columns)  # sample_id, ground_truth, azure_intelligence_prediction, ...
    """
    file_path = CLEAN_PATH / dataset / f"{phase}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Clean results file not found: {file_path}\n"
            f"Run: cd 2_clean && python clean_files.py --incremental"
        )

    df = pd.read_csv(file_path)
    return df


def load_metadata() -> Dict:
    """
    Load clean_experiments.json metadata file.

    Returns:
        Dictionary with dataset and phase information

    Example:
        metadata = load_metadata()
        datasets = metadata['datasets']
        models = metadata['datasets']['DocVQA_mini']['phases']['QA1a']['models']
    """
    metadata_path = INFO_PATH / "clean_experiments.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r') as f:
        return json.load(f)


def load_visualization_config() -> Dict:
    """
    Load visualization_config.json with color schemes and plot settings.

    Returns:
        Dictionary with colors, plot defaults, and export settings

    Example:
        config = load_visualization_config()
        model_color = config['colors']['models']['gpt-5-mini']
        figsize = config['plot_defaults']['figsize']
    """
    config_path = INFO_PATH / "visualization_config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Visualization config not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def load_execution_summary(dataset: str) -> Dict:
    """
    Load execution_summary.json from raw results folder.

    Args:
        dataset: Dataset name (e.g., 'DocVQA_mini')

    Returns:
        Dictionary with execution statistics (timing, token usage, etc.)

    Example:
        summary = load_execution_summary('DocVQA_mini')
        print(summary['total_samples'])
        print(summary['phases'])
    """
    summary_path = RAW_PATH / dataset / "execution_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Execution summary not found: {summary_path}")

    with open(summary_path, 'r') as f:
        return json.load(f)


def load_all_phases(dataset: str, phases: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load and concatenate multiple phases for a dataset.

    Args:
        dataset: Dataset name
        phases: List of phase names, or None to load all phases

    Returns:
        DataFrame with phase column added

    Example:
        df = load_all_phases('DocVQA_mini', ['QA1a', 'QA1b', 'QA1c'])
        df.groupby('phase').mean()
    """
    metadata = load_metadata()

    if dataset not in metadata['datasets']:
        raise ValueError(f"Dataset {dataset} not found in metadata")

    dataset_info = metadata['datasets'][dataset]

    if phases is None:
        phases = list(dataset_info['phases'].keys())

    dfs = []
    for phase in phases:
        try:
            df = load_clean_results(dataset, phase)
            df['phase'] = phase
            dfs.append(df)
        except FileNotFoundError as e:
            print(f"Warning: Skipping phase {phase}: {e}")

    if not dfs:
        raise ValueError(f"No data found for dataset {dataset}")

    return pd.concat(dfs, ignore_index=True)


def get_dataset_models(dataset: str, phase: str) -> List[str]:
    """
    Get list of models for a specific dataset and phase.

    Args:
        dataset: Dataset name
        phase: Phase name

    Returns:
        List of model names

    Example:
        models = get_dataset_models('DocVQA_mini', 'QA1a')
        # ['azure_intelligence', 'mistral_document_ai']
    """
    metadata = load_metadata()

    try:
        return metadata['datasets'][dataset]['phases'][phase]['models']
    except KeyError:
        raise ValueError(f"Dataset {dataset} or phase {phase} not found in metadata")


def get_dataset_metrics(dataset: str, phase: str) -> List[str]:
    """
    Get list of metrics for a specific dataset and phase.

    Args:
        dataset: Dataset name
        phase: Phase name

    Returns:
        List of metric names

    Example:
        metrics = get_dataset_metrics('DocVQA_mini', 'QA1a')
        # ['ANLS', 'EM', 'substring_match']
    """
    metadata = load_metadata()

    try:
        return metadata['datasets'][dataset]['phases'][phase]['metrics']
    except KeyError:
        raise ValueError(f"Dataset {dataset} or phase {phase} not found in metadata")


def get_model_color(model: str) -> str:
    """
    Get color for a specific model from visualization config.

    Args:
        model: Model name

    Returns:
        Hex color code

    Example:
        color = get_model_color('gpt-5-mini')
        # '#10A37F'
    """
    config = load_visualization_config()

    return config['colors']['models'].get(model, '#808080')  # Gray default


def get_dataset_color(dataset: str) -> str:
    """
    Get color for a specific dataset from visualization config.

    Args:
        dataset: Dataset name

    Returns:
        Hex color code
    """
    config = load_visualization_config()

    return config['colors']['datasets'].get(dataset, '#808080')


def get_metric_color(metric: str) -> str:
    """
    Get color for a specific metric from visualization config.

    Args:
        metric: Metric name

    Returns:
        Hex color code
    """
    config = load_visualization_config()

    return config['colors']['metrics'].get(metric, '#808080')


def list_available_datasets() -> List[str]:
    """
    Get list of all available datasets.

    Returns:
        List of dataset names
    """
    metadata = load_metadata()
    return list(metadata['datasets'].keys())


def list_available_phases(dataset: str) -> List[str]:
    """
    Get list of all available phases for a dataset.

    Args:
        dataset: Dataset name

    Returns:
        List of phase names
    """
    metadata = load_metadata()

    if dataset not in metadata['datasets']:
        raise ValueError(f"Dataset {dataset} not found")

    return list(metadata['datasets'][dataset]['phases'].keys())


# Backwards compatibility with existing notebooks
# (they might use data_loader.py from shared/)
try:
    from .data_loader import *
except ImportError:
    pass
