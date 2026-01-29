"""
Visualization utilities for OCR vs VLM evaluation using Plotly.

Provides interactive charts with consistent styling:
- Metric boxplots with significance markers
- Heatmaps for pairwise comparisons
- Radar charts for multi-metric comparison
- Pareto charts for cost-performance analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .colors import MODEL_COLORS, DATASET_COLORS, get_model_color, get_dataset_color, APPROACH_COLORS


# =============================================================================
# PLOTLY TEMPLATE SETUP
# =============================================================================

def setup_plotly_template():
    """
    Configure default Plotly template for consistent styling.
    Call this at the start of each notebook.
    """
    import plotly.io as pio
    
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#2B2B2B"),
        title=dict(font=dict(size=16, color="#1F1F1F")),
        paper_bgcolor='white',
        plot_bgcolor='#FAFAFA',
        xaxis=dict(
            gridcolor='#E5E7EB',
            linecolor='#D1D5DB',
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            gridcolor='#E5E7EB',
            linecolor='#D1D5DB',
            tickfont=dict(size=11)
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E5E7EB',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor='white',
            font_size=12,
            font_family="Inter, Arial, sans-serif"
        )
    )
    
    pio.templates['ocr_vlm'] = template
    pio.templates.default = 'ocr_vlm'
    
    return template


# =============================================================================
# BOXPLOTS
# =============================================================================

def create_metric_boxplot(
    data: Dict[str, np.ndarray],
    metric_name: str = 'ANLS',
    title: Optional[str] = None,
    significance_pairs: Optional[Dict[Tuple[str, str], float]] = None,
    color_by: str = 'model',
    show_points: bool = True,
    height: int = 500
) -> go.Figure:
    """
    Create an interactive boxplot comparing models/experiments.
    
    Args:
        data: Dict mapping names to score arrays
        metric_name: Name of metric for y-axis label
        title: Chart title
        significance_pairs: Dict of (model_a, model_b) -> p_value for annotation
        color_by: 'model' or 'dataset' for color assignment
        show_points: Whether to show individual data points
        height: Figure height in pixels
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    names = list(data.keys())
    
    for name in names:
        values = data[name]
        values = values[~np.isnan(values)]
        
        if color_by == 'model':
            color = get_model_color(name)
        else:
            color = get_dataset_color(name)
        
        fig.add_trace(go.Box(
            y=values,
            name=name,
            marker_color=color,
            boxpoints='outliers' if not show_points else 'all',
            jitter=0.3 if show_points else 0,
            pointpos=-1.5 if show_points else 0,
            hovertemplate=f"<b>{name}</b><br>{metric_name}: %{{y:.4f}}<extra></extra>"
        ))
    
    # Add significance annotations
    if significance_pairs:
        y_max = max(np.nanmax(v) for v in data.values())
        y_step = (y_max - min(np.nanmin(v) for v in data.values())) * 0.08
        
        for i, ((a, b), pval) in enumerate(significance_pairs.items()):
            if pval < 0.05:  # Only annotate significant differences
                idx_a = names.index(a)
                idx_b = names.index(b)
                y_line = y_max + y_step * (i + 1)
                
                # Add bracket
                fig.add_shape(
                    type="line",
                    x0=idx_a, x1=idx_b,
                    y0=y_line, y1=y_line,
                    line=dict(color="#6B7280", width=1.5)
                )
                
                # Add p-value annotation
                stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*'
                fig.add_annotation(
                    x=(idx_a + idx_b) / 2,
                    y=y_line + y_step * 0.3,
                    text=stars,
                    showarrow=False,
                    font=dict(size=14, color="#374151")
                )
    
    fig.update_layout(
        title=title or f'{metric_name} Distribution by Model',
        yaxis_title=metric_name,
        xaxis_title='',
        height=height,
        showlegend=False
    )
    
    return fig


# =============================================================================
# HEATMAPS
# =============================================================================

def create_heatmap(
    matrix: pd.DataFrame,
    title: str = 'Comparison Matrix',
    annotation_format: str = '.3f',
    colorscale: str = 'RdYlGn',
    reverse_colorscale: bool = False,
    height: int = 500
) -> go.Figure:
    """
    Create an annotated heatmap for pairwise comparisons.
    
    Args:
        matrix: DataFrame with row/column labels and numeric values
        title: Chart title
        annotation_format: Format string for cell annotations
        colorscale: Plotly colorscale name
        reverse_colorscale: Whether to reverse the colorscale
        height: Figure height
    
    Returns:
        Plotly Figure object
    """
    if reverse_colorscale:
        colorscale = colorscale + '_r'
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate='%{y} vs %{x}<br>Value: %{z:.4f}<extra></extra>'
    ))
    
    # Add annotations
    for i, row in enumerate(matrix.index):
        for j, col in enumerate(matrix.columns):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                fig.add_annotation(
                    x=col, y=row,
                    text=f'{val:{annotation_format}}',
                    showarrow=False,
                    font=dict(size=10, color='white' if abs(val) > 0.5 else 'black')
                )
    
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def create_significance_heatmap(
    p_values: Dict[Tuple[str, str], float],
    models: List[str],
    title: str = 'Pairwise Significance (Wilcoxon)',
    alpha: float = 0.05
) -> go.Figure:
    """
    Create a heatmap showing pairwise statistical significance.
    
    Args:
        p_values: Dict mapping (model_a, model_b) to p-value
        models: List of model names (for ordering)
        title: Chart title
        alpha: Significance threshold
    
    Returns:
        Plotly Figure object
    """
    n = len(models)
    matrix = np.full((n, n), np.nan)
    
    for (a, b), p in p_values.items():
        i, j = models.index(a), models.index(b)
        matrix[i, j] = -np.log10(p) if p > 0 else 10
        matrix[j, i] = matrix[i, j]
    
    df = pd.DataFrame(matrix, index=models, columns=models)
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale='Viridis',
        colorbar=dict(title='-log10(p)'),
        hovertemplate='%{y} vs %{x}<br>-log10(p): %{z:.2f}<extra></extra>'
    ))
    
    # Add significance markers
    threshold_log = -np.log10(alpha)
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(matrix[i, j]):
                if matrix[i, j] > threshold_log:
                    fig.add_annotation(
                        x=models[j], y=models[i],
                        text='*',
                        showarrow=False,
                        font=dict(size=16, color='white')
                    )
    
    fig.update_layout(
        title=title,
        height=500,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    return fig


# =============================================================================
# RADAR CHARTS
# =============================================================================

def create_radar_chart(
    data: Dict[str, Dict[str, float]],
    metrics: List[str],
    title: str = 'Multi-Metric Comparison',
    fill: bool = True,
    height: int = 500
) -> go.Figure:
    """
    Create a radar chart comparing multiple metrics across models.
    
    Args:
        data: Dict mapping model names to {metric: value} dicts
        metrics: List of metric names (in order for radar)
        title: Chart title
        fill: Whether to fill the radar areas
        height: Figure height
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    for model, values in data.items():
        r = [values.get(m, 0) for m in metrics]
        r.append(r[0])  # Close the polygon
        theta = metrics + [metrics[0]]
        
        color = get_model_color(model)
        
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill='toself' if fill else None,
            fillcolor=color + '40' if fill else None,  # 25% opacity
            line=dict(color=color, width=2),
            name=model,
            hovertemplate='<b>%{theta}</b>: %{r:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        title=title,
        height=height,
        showlegend=True
    )
    
    return fig


# =============================================================================
# PARETO CHARTS
# =============================================================================

def create_pareto_chart(
    data: pd.DataFrame,
    x_col: str = 'cost',
    y_col: str = 'performance',
    label_col: str = 'model',
    color_col: Optional[str] = None,
    title: str = 'Cost vs Performance',
    x_label: str = 'Cost ($)',
    y_label: str = 'Performance',
    show_pareto_frontier: bool = True,
    height: int = 500
) -> go.Figure:
    """
    Create a Pareto chart for cost-performance analysis.
    
    Args:
        data: DataFrame with cost, performance, and label columns
        x_col: Column name for x-axis (cost)
        y_col: Column name for y-axis (performance)
        label_col: Column name for point labels
        color_col: Column name for coloring (optional)
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        show_pareto_frontier: Whether to highlight Pareto-optimal points
        height: Figure height
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Identify Pareto frontier (minimize cost, maximize performance)
    if show_pareto_frontier:
        pareto_mask = []
        for i, row in data.iterrows():
            is_pareto = True
            for j, other in data.iterrows():
                if i != j:
                    if other[x_col] <= row[x_col] and other[y_col] >= row[y_col]:
                        if other[x_col] < row[x_col] or other[y_col] > row[y_col]:
                            is_pareto = False
                            break
            pareto_mask.append(is_pareto)
        data = data.copy()
        data['is_pareto'] = pareto_mask
    
    # Plot points
    for idx, row in data.iterrows():
        model = row[label_col]
        color = get_model_color(model) if color_col is None else row.get(color_col, '#6B7280')
        
        marker_symbol = 'star' if show_pareto_frontier and row.get('is_pareto', False) else 'circle'
        marker_size = 15 if marker_symbol == 'star' else 10
        
        fig.add_trace(go.Scatter(
            x=[row[x_col]],
            y=[row[y_col]],
            mode='markers+text',
            marker=dict(
                color=color,
                size=marker_size,
                symbol=marker_symbol,
                line=dict(width=1, color='white')
            ),
            text=[model],
            textposition='top center',
            textfont=dict(size=10),
            name=model,
            hovertemplate=f"<b>{model}</b><br>{x_label}: %{{x:.4f}}<br>{y_label}: %{{y:.4f}}<extra></extra>"
        ))
    
    # Draw Pareto frontier line
    if show_pareto_frontier:
        pareto_points = data[data['is_pareto']].sort_values(x_col)
        if len(pareto_points) > 1:
            fig.add_trace(go.Scatter(
                x=pareto_points[x_col],
                y=pareto_points[y_col],
                mode='lines',
                line=dict(color='#9CA3AF', width=2, dash='dash'),
                name='Pareto Frontier',
                showlegend=True,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        showlegend=True
    )
    
    return fig


# =============================================================================
# BAR CHARTS
# =============================================================================

def create_grouped_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str = 'Comparison',
    y_label: str = 'Score',
    error_col: Optional[str] = None,
    height: int = 500
) -> go.Figure:
    """
    Create a grouped bar chart for comparing metrics across groups.
    
    Args:
        data: DataFrame with x, y, and group columns
        x_col: Column for x-axis categories
        y_col: Column for bar heights
        group_col: Column for grouping (different colors)
        title: Chart title
        y_label: Y-axis label
        error_col: Column for error bars (optional)
        height: Figure height
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    groups = data[group_col].unique()
    
    for group in groups:
        group_data = data[data[group_col] == group]
        color = get_model_color(str(group))
        
        error_y = None
        if error_col and error_col in group_data.columns:
            error_y = dict(type='data', array=group_data[error_col].values, visible=True)
        
        fig.add_trace(go.Bar(
            x=group_data[x_col],
            y=group_data[y_col],
            name=str(group),
            marker_color=color,
            error_y=error_y,
            hovertemplate=f"<b>{group}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='',
        yaxis_title=y_label,
        barmode='group',
        height=height
    )
    
    return fig


# =============================================================================
# LINE CHARTS
# =============================================================================

def create_line_chart_with_ci(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    group_col: str,
    title: str = 'Trend with Confidence Intervals',
    y_label: str = 'Score',
    height: int = 500
) -> go.Figure:
    """
    Create a line chart with confidence interval bands.
    
    Args:
        data: DataFrame with x, y, CI bounds, and group columns
        x_col: Column for x-axis
        y_col: Column for line values
        ci_lower_col: Column for CI lower bound
        ci_upper_col: Column for CI upper bound
        group_col: Column for different lines
        title: Chart title
        y_label: Y-axis label
        height: Figure height
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    groups = data[group_col].unique()
    
    for group in groups:
        group_data = data[data[group_col] == group].sort_values(x_col)
        color = get_model_color(str(group))
        
        # CI band
        fig.add_trace(go.Scatter(
            x=list(group_data[x_col]) + list(group_data[x_col][::-1]),
            y=list(group_data[ci_upper_col]) + list(group_data[ci_lower_col][::-1]),
            fill='toself',
            fillcolor=color + '30',  # 19% opacity
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name=f'{group} CI',
            hoverinfo='skip'
        ))
        
        # Main line
        fig.add_trace(go.Scatter(
            x=group_data[x_col],
            y=group_data[y_col],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=8),
            name=str(group),
            hovertemplate=f"<b>{group}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_label,
        height=height
    )
    
    return fig


# =============================================================================
# INTERACTIVE WIDGETS
# =============================================================================

def create_model_selector(models: List[str], default: Optional[List[str]] = None):
    """
    Create an ipywidgets multi-select for models.
    
    Args:
        models: List of model names
        default: Default selected models
    
    Returns:
        ipywidgets SelectMultiple widget
    """
    import ipywidgets as widgets
    
    return widgets.SelectMultiple(
        options=models,
        value=default or models[:3],
        description='Models:',
        disabled=False,
        layout=widgets.Layout(width='300px', height='150px')
    )


def create_dataset_dropdown(datasets: List[str], default: Optional[str] = None):
    """
    Create an ipywidgets dropdown for dataset selection.
    
    Args:
        datasets: List of dataset names
        default: Default selected dataset
    
    Returns:
        ipywidgets Dropdown widget
    """
    import ipywidgets as widgets
    
    return widgets.Dropdown(
        options=datasets,
        value=default or datasets[0],
        description='Dataset:',
        disabled=False,
        layout=widgets.Layout(width='300px')
    )


def create_metric_radio(metrics: List[str], default: Optional[str] = None):
    """
    Create an ipywidgets radio button for metric selection.
    
    Args:
        metrics: List of metric names
        default: Default selected metric
    
    Returns:
        ipywidgets RadioButtons widget
    """
    import ipywidgets as widgets
    
    return widgets.RadioButtons(
        options=metrics,
        value=default or metrics[0],
        description='Metric:',
        disabled=False
    )
