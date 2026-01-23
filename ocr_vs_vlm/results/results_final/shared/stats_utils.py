"""
Statistical utilities for OCR vs VLM evaluation.

Provides:
- Bootstrap confidence intervals
- Wilcoxon signed-rank test for paired comparisons
- McNemar's test for binary metrics (Exact Match)
- Cohen's d effect size
- Multiple comparison corrections (Bonferroni)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from scipy import stats
import warnings


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = 'mean',
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: 1D array of values
        statistic: 'mean', 'median', or 'std'
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(seed)
    data = np.asarray(data)
    data = data[~np.isnan(data)]  # Remove NaN
    
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    
    # Point estimate
    if statistic == 'mean':
        point_est = np.mean(data)
        stat_func = np.mean
    elif statistic == 'median':
        point_est = np.median(data)
        stat_func = np.median
    elif statistic == 'std':
        point_est = np.std(data)
        stat_func = np.std
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Bootstrap
    bootstrap_stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Confidence interval (percentile method)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return point_est, ci_lower, ci_upper


def wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired samples.
    
    Non-parametric test for comparing two related samples.
    Use when comparing same samples across two models/experiments.
    
    Args:
        x: First sample (e.g., model A scores)
        y: Second sample (e.g., model B scores)
        alternative: 'two-sided', 'greater', or 'less'
    
    Returns:
        Tuple of (statistic, p_value)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove pairs with NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    
    if len(x) < 10:
        warnings.warn("Small sample size (<10) may yield unreliable results")
    
    if len(x) == 0:
        return np.nan, np.nan
    
    # Check if samples are identical
    if np.allclose(x, y):
        return 0.0, 1.0
    
    try:
        stat, pval = stats.wilcoxon(x, y, alternative=alternative)
        return float(stat), float(pval)
    except ValueError as e:
        warnings.warn(f"Wilcoxon test failed: {e}")
        return np.nan, np.nan


def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray
) -> Tuple[float, float]:
    """
    McNemar's test for comparing binary outcomes (e.g., Exact Match).
    
    Tests whether two models have significantly different accuracy
    on the same samples.
    
    Args:
        correct_a: Binary array (1=correct, 0=incorrect) for model A
        correct_b: Binary array for model B
    
    Returns:
        Tuple of (statistic, p_value)
    """
    correct_a = np.asarray(correct_a).astype(bool)
    correct_b = np.asarray(correct_b).astype(bool)
    
    # Build contingency table
    # b = A correct, B wrong
    # c = A wrong, B correct
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)
    
    if b + c == 0:
        return 0.0, 1.0
    
    # McNemar's test with continuity correction
    if b + c < 25:
        # Use exact binomial test for small samples
        pval = stats.binom_test(b, b + c, 0.5)
        stat = b
    else:
        # Chi-squared approximation with continuity correction
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        pval = 1 - stats.chi2.cdf(stat, df=1)
    
    return float(stat), float(pval)


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cohen's d effect size for comparing two groups.
    
    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    
    Args:
        x: First group values
        y: Second group values
    
    Returns:
        Cohen's d value (positive = x > y)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove NaN
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    if len(x) == 0 or len(y) == 0:
        return np.nan
    
    # Pooled standard deviation
    nx, ny = len(x), len(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(x) - np.mean(y)) / pooled_std


def effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Significance level
    
    Returns:
        List of booleans indicating significance after correction
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests
    return [p < adjusted_alpha for p in p_values]


def compute_significance_matrix(
    data: Dict[str, np.ndarray],
    test: str = 'wilcoxon',
    alpha: float = 0.05
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], bool]]:
    """
    Compute pairwise significance tests between all models.
    
    Args:
        data: Dict mapping model names to score arrays
        test: 'wilcoxon' or 'mcnemar'
        alpha: Significance level
    
    Returns:
        Tuple of (p_values dict, significance dict)
    """
    models = list(data.keys())
    p_values = {}
    
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i >= j:
                continue
            
            if test == 'wilcoxon':
                _, pval = wilcoxon_test(data[model_a], data[model_b])
            elif test == 'mcnemar':
                _, pval = mcnemar_test(data[model_a], data[model_b])
            else:
                raise ValueError(f"Unknown test: {test}")
            
            p_values[(model_a, model_b)] = pval
    
    # Apply Bonferroni correction
    all_pvals = list(p_values.values())
    significant = bonferroni_correction(all_pvals, alpha)
    
    significance = {}
    for (pair, pval), sig in zip(p_values.items(), significant):
        significance[pair] = sig
    
    return p_values, significance


def format_pvalue(p: float, threshold: float = 0.001) -> str:
    """
    Format p-value for display.
    
    Args:
        p: P-value
        threshold: Below this, display as '< threshold'
    
    Returns:
        Formatted string
    """
    if np.isnan(p):
        return "N/A"
    if p < threshold:
        return f"< {threshold}"
    elif p < 0.01:
        return f"{p:.3f}"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"


def compute_ci_table(
    data: Dict[str, np.ndarray],
    metric_name: str = 'Score',
    confidence: float = 0.95
) -> 'pd.DataFrame':
    """
    Create a summary table with means and confidence intervals.
    
    Args:
        data: Dict mapping model names to score arrays
        metric_name: Name of the metric for column labeling
        confidence: Confidence level
    
    Returns:
        DataFrame with columns: Model, Mean, CI_Lower, CI_Upper, N
    """
    import pandas as pd
    
    rows = []
    for model, scores in data.items():
        mean, ci_lo, ci_hi = bootstrap_ci(scores, 'mean', confidence=confidence)
        n = len(scores[~np.isnan(scores)])
        rows.append({
            'Model': model,
            f'{metric_name}_Mean': mean,
            f'{metric_name}_CI_Lower': ci_lo,
            f'{metric_name}_CI_Upper': ci_hi,
            'N': n
        })
    
    return pd.DataFrame(rows)
