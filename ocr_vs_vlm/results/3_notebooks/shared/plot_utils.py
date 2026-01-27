"""
Plotting utilities for results analysis notebooks.

This module provides standardized plotting functions for consistent
visualization across all analysis notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from .data_loaders import load_visualization_config, get_model_color, get_metric_color


def setup_plot_style():
    """
    Set up matplotlib/seaborn style from visualization config.
    Call this at the beginning of each notebook.
    """
    config = load_visualization_config()
    defaults = config['plot_defaults']

    # Set seaborn style
    sns.set_style('whitegrid')

    # Set matplotlib parameters
    plt.rcParams['figure.figsize'] = defaults['figsize']
    plt.rcParams['font.size'] = defaults['font_size']
    plt.rcParams['axes.labelsize'] = defaults['label_font_size']
    plt.rcParams['axes.titlesize'] = defaults['title_font_size']
    plt.rcParams['legend.fontsize'] = defaults['legend_font_size']
    plt.rcParams['figure.dpi'] = defaults['dpi']


def create_model_comparison_heatmap(
    df: pd.DataFrame,
    metric: str,
    models: List[str],
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap comparing model performance on a metric.

    Args:
        df: DataFrame with model columns (e.g., model_CER, model_WER)
        metric: Metric name (e.g., 'CER', 'ANLS')
        models: List of model names
        title: Plot title (auto-generated if None)
        figsize: Figure size tuple (width, height)
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure object

    Example:
        fig = create_model_comparison_heatmap(
            df, 'ANLS',
            ['azure_intelligence', 'gpt-5-mini', 'claude_sonnet']
        )
    """
    config = load_visualization_config()
    heatmap_defaults = config['heatmap_defaults']

    if figsize is None:
        figsize = tuple(config['plot_defaults']['figsize'])

    # Extract metric values for each model
    metric_cols = [f"{model}_{metric}" for model in models if f"{model}_{metric}" in df.columns]

    if not metric_cols:
        raise ValueError(f"No columns found for metric {metric} and models {models}")

    # Create pivot table (samples x models)
    data = df[metric_cols].copy()
    data.columns = [col.replace(f"_{metric}", "") for col in data.columns]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        data.T,  # Transpose so models are rows
        cmap=heatmap_defaults['cmap'],
        annot=False,  # Too many samples to annotate
        fmt=heatmap_defaults.get('fmt', '.3f'),
        linewidths=heatmap_defaults.get('linewidths', 0.5),
        cbar_kws=heatmap_defaults.get('cbar_kws', {}),
        ax=ax
    )

    if title is None:
        title = f"Model Comparison: {metric}"
    ax.set_title(title, fontsize=config['plot_defaults']['title_font_size'])
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Model')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config['plot_defaults']['save_dpi'], bbox_inches='tight')

    return fig


def create_model_performance_barplot(
    df: pd.DataFrame,
    metric: str,
    models: List[str],
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a bar plot showing average model performance.

    Args:
        df: DataFrame with model columns
        metric: Metric name
        models: List of model names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure object
    """
    config = load_visualization_config()
    bar_defaults = config['barplot_defaults']

    if figsize is None:
        figsize = tuple(config['plot_defaults']['figsize'])

    # Calculate mean and std for each model
    means = []
    stds = []
    model_names = []

    for model in models:
        col = f"{model}_{metric}"
        if col in df.columns:
            means.append(df[col].mean())
            stds.append(df[col].std())
            model_names.append(model)

    if not means:
        raise ValueError(f"No data found for metric {metric} and models {models}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get colors for each model
    colors = [get_model_color(model) for model in model_names]

    # Create bar plot
    x = np.arange(len(model_names))
    bars = ax.bar(
        x, means,
        yerr=stds,
        color=colors,
        alpha=bar_defaults.get('alpha', 0.8),
        edgecolor=bar_defaults.get('edgecolor', 'black'),
        linewidth=bar_defaults.get('linewidth', 0.7),
        capsize=5
    )

    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel(metric)

    if title is None:
        title = f"Model Performance: {metric}"
    ax.set_title(title, fontsize=config['plot_defaults']['title_font_size'])

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{mean:.3f}',
            ha='center', va='bottom',
            fontsize=config['plot_defaults']['font_size'] - 1
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config['plot_defaults']['save_dpi'], bbox_inches='tight')

    return fig


def create_phase_comparison_plot(
    df: pd.DataFrame,
    metric: str,
    model: str,
    phases: List[str],
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a plot comparing performance across phases for one model.

    Args:
        df: DataFrame with phase column and metric columns
        metric: Metric name
        model: Model name
        phases: List of phase names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure object
    """
    config = load_visualization_config()

    if figsize is None:
        figsize = tuple(config['plot_defaults']['figsize'])

    fig, ax = plt.subplots(figsize=figsize)

    # Extract data for each phase
    means = []
    stds = []

    for phase in phases:
        phase_df = df[df['phase'] == phase]
        col = f"{model}_{metric}"

        if col in phase_df.columns:
            means.append(phase_df[col].mean())
            stds.append(phase_df[col].std())
        else:
            means.append(0)
            stds.append(0)

    # Create bar plot
    x = np.arange(len(phases))
    color = get_model_color(model)

    bars = ax.bar(x, means, yerr=stds, color=color, alpha=0.8, capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(phases, rotation=45, ha='right')
    ax.set_ylabel(metric)

    if title is None:
        title = f"{model}: Performance Across Phases ({metric})"
    ax.set_title(title, fontsize=config['plot_defaults']['title_font_size'])

    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{mean:.3f}',
            ha='center', va='bottom'
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config['plot_defaults']['save_dpi'], bbox_inches='tight')

    return fig


def create_error_analysis_table(
    df: pd.DataFrame,
    model: str,
    metric: str,
    n_best: int = 5,
    n_worst: int = 5
) -> pd.DataFrame:
    """
    Create a table showing best and worst predictions for a model.

    Args:
        df: DataFrame with predictions and ground truth
        model: Model name
        metric: Metric name
        n_best: Number of best examples to show
        n_worst: Number of worst examples to show

    Returns:
        DataFrame with best and worst examples

    Example:
        error_table = create_error_analysis_table(df, 'gpt-5-mini', 'ANLS')
        display(error_table)
    """
    pred_col = f"{model}_prediction"
    metric_col = f"{model}_{metric}"

    if pred_col not in df.columns or metric_col not in df.columns:
        raise ValueError(f"Columns not found for model {model} and metric {metric}")

    # Sort by metric
    sorted_df = df.sort_values(by=metric_col)

    # Get best and worst
    worst = sorted_df.head(n_worst)[['sample_id', 'ground_truth', pred_col, metric_col]]
    best = sorted_df.tail(n_best)[['sample_id', 'ground_truth', pred_col, metric_col]]

    # Add category column
    worst['category'] = 'Worst'
    best['category'] = 'Best'

    # Combine and reset index
    result = pd.concat([worst, best]).reset_index(drop=True)

    # Reorder columns
    result = result[['category', 'sample_id', 'ground_truth', pred_col, metric_col]]
    result.columns = ['Category', 'Sample ID', 'Ground Truth', 'Prediction', metric]

    return result


def create_correlation_matrix(
    df: pd.DataFrame,
    metrics: List[str],
    model: str,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a correlation matrix heatmap for multiple metrics.

    Args:
        df: DataFrame with metric columns
        metrics: List of metric names
        model: Model name
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure object
    """
    config = load_visualization_config()

    # Extract metric columns
    metric_cols = [f"{model}_{metric}" for metric in metrics if f"{model}_{metric}" in df.columns]

    if not metric_cols:
        raise ValueError(f"No metric columns found for model {model}")

    # Calculate correlation
    corr = df[metric_cols].corr()
    corr.columns = [col.replace(f"{model}_", "") for col in corr.columns]
    corr.index = [idx.replace(f"{model}_", "") for idx in corr.index]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    if title is None:
        title = f"Metric Correlation: {model}"
    ax.set_title(title, fontsize=config['plot_defaults']['title_font_size'])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config['plot_defaults']['save_dpi'], bbox_inches='tight')

    return fig


def save_figure(fig: plt.Figure, path: str, formats: Optional[List[str]] = None):
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure
        path: Base path (without extension)
        formats: List of formats (e.g., ['png', 'svg']), uses config default if None
    """
    config = load_visualization_config()

    if formats is None:
        formats = config['export_formats']['figures']

    for fmt in formats:
        output_path = f"{path}.{fmt}"
        fig.savefig(
            output_path,
            dpi=config['plot_defaults']['save_dpi'],
            bbox_inches='tight'
        )
        print(f"Saved: {output_path}")


# Backwards compatibility - import viz_utils if available
try:
    from .viz_utils import *
except ImportError:
    pass
