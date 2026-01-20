"""
Shared utilities for OCR vs VLM evaluation notebooks.

Provides:
- colors: Brand color palettes for consistent visualizations
- stats_utils: Statistical tests (bootstrap CI, Wilcoxon, effect sizes)
- viz_utils: Plotly templates and interactive widgets
- data_loader: Unified data loading from results_clean/
"""

from .colors import COLORS, MODEL_COLORS, DATASET_COLORS, get_model_color, get_dataset_color
from .stats_utils import (
    bootstrap_ci, wilcoxon_test, mcnemar_test, cohens_d,
    compute_significance_matrix, format_pvalue
)
from .viz_utils import (
    create_metric_boxplot, create_heatmap, create_radar_chart,
    create_pareto_chart, setup_plotly_template
)
from .data_loader import load_experiment_data, load_dataset_data, get_available_experiments

__all__ = [
    # Colors
    'COLORS', 'MODEL_COLORS', 'DATASET_COLORS', 'get_model_color', 'get_dataset_color',
    # Stats
    'bootstrap_ci', 'wilcoxon_test', 'mcnemar_test', 'cohens_d',
    'compute_significance_matrix', 'format_pvalue',
    # Viz
    'create_metric_boxplot', 'create_heatmap', 'create_radar_chart',
    'create_pareto_chart', 'setup_plotly_template',
    # Data
    'load_experiment_data', 'load_dataset_data', 'get_available_experiments',
]
