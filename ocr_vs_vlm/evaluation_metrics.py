"""
Comprehensive evaluation metrics for OCR and VQA tasks.

Provides metrics including:
- CER (Character Error Rate)
- WER (Word Error Rate)
- EM (Exact Match)
- ANLS (Average Normalized Levenshtein Similarity)
- Cosine Similarity (using embeddings)
- Substring matching for VQA
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_cer(prediction: str, reference: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = (S + D + I) / N
    where:
        S = substitutions
        D = deletions
        I = insertions
        N = number of characters in reference
    
    Args:
        prediction: Predicted text
        reference: Ground truth text
        
    Returns:
        CER value (0.0 = perfect, higher = worse)
    """
    try:
        import editdistance
    except ImportError:
        logger.warning("editdistance not installed, returning 0.0 for CER")
        return 0.0
    
    if not reference:
        return 1.0 if prediction else 0.0
    
    pred = str(prediction).strip()
    ref = str(reference).strip()
    
    edit_dist = editdistance.eval(pred, ref)
    cer = edit_dist / len(ref) if ref else 0.0
    
    return cer


def calculate_wer(prediction: str, reference: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER = (S + D + I) / N
    where:
        S = word substitutions
        D = word deletions
        I = word insertions
        N = number of words in reference
    
    Args:
        prediction: Predicted text
        reference: Ground truth text
        
    Returns:
        WER value (0.0 = perfect, higher = worse)
    """
    try:
        import editdistance
    except ImportError:
        logger.warning("editdistance not installed, returning 0.0 for WER")
        return 0.0
    
    if not reference:
        return 1.0 if prediction else 0.0
    
    pred_words = str(prediction).strip().split()
    ref_words = str(reference).strip().split()
    
    if not ref_words:
        return 1.0 if pred_words else 0.0
    
    edit_dist = editdistance.eval(pred_words, ref_words)
    wer = edit_dist / len(ref_words)
    
    return wer


def compute_anls(prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    """
    Compute Average Normalized Levenshtein Similarity (ANLS).
    
    This is the standard DocVQA/InfographicVQA evaluation metric.
    
    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        threshold: Minimum similarity threshold (default 0.5)
        
    Returns:
        ANLS score (0.0 to 1.0, higher is better)
    """
    if not prediction or not ground_truths:
        return 0.0
    
    try:
        import editdistance
    except ImportError:
        logger.warning("editdistance not installed, using basic comparison")
        pred_lower = prediction.lower().strip()
        return 1.0 if any(pred_lower == gt.lower().strip() for gt in ground_truths) else 0.0
    
    pred = prediction.lower().strip()
    
    max_score = 0.0
    for gt in ground_truths:
        gt = gt.lower().strip()
        if not gt:
            continue
        
        # Compute normalized Levenshtein distance
        edit_dist = editdistance.eval(pred, gt)
        max_len = max(len(pred), len(gt))
        
        if max_len == 0:
            nls = 1.0
        else:
            nls = 1.0 - (edit_dist / max_len)
        
        # Apply threshold
        if nls < threshold:
            nls = 0.0
        
        max_score = max(max_score, nls)
    
    return max_score


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """
    Check if prediction exactly matches any ground truth.
    
    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if not prediction or not ground_truths:
        return 0.0
    
    pred = prediction.lower().strip()
    return 1.0 if any(pred == gt.lower().strip() for gt in ground_truths) else 0.0


def compute_substring_match(prediction: str, ground_truths: List[str]) -> float:
    """
    Check if prediction is substring of any ground truth or vice versa.
    
    Useful for VQA tasks where ground truths may have variations.
    
    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        
    Returns:
        1.0 if substring match found, 0.0 otherwise
    """
    if not prediction or not ground_truths:
        return 0.0
    
    pred = prediction.lower().strip()
    
    for gt in ground_truths:
        gt = gt.lower().strip()
        # Check if prediction in ground truth or ground truth in prediction
        if pred in gt or gt in pred:
            return 1.0
    
    return 0.0


def compute_cosine_similarity(
    prediction: str,
    ground_truths: List[str],
    embeddings_dir: Optional[Path] = None
) -> float:
    """
    Compute cosine similarity between prediction and ground truths using embeddings.
    
    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        embeddings_dir: Path to embeddings directory (optional)
        
    Returns:
        Maximum cosine similarity score (0.0 to 1.0, higher is better)
    """
    if not prediction or not ground_truths:
        return 0.0
    
    try:
        from llms.embeddings import EmbeddingCalculator
        from scipy.spatial.distance import cosine
    except ImportError:
        logger.warning("Required libraries not available for cosine similarity")
        return 0.0
    
    try:
        # Initialize embedding calculator
        calc = EmbeddingCalculator()
        
        # Get embedding for prediction
        pred_result = calc.get_embedding(prediction)
        pred_embedding = pred_result.embedding
        
        # Compute similarity with each ground truth
        max_similarity = 0.0
        for gt in ground_truths:
            gt_result = calc.get_embedding(gt)
            gt_embedding = gt_result.embedding
            
            # Cosine similarity = 1 - cosine distance
            similarity = 1 - cosine(pred_embedding, gt_embedding)
            max_similarity = max(max_similarity, similarity)
        
        return float(max_similarity)
    
    except Exception as e:
        logger.warning(f"Error computing cosine similarity: {e}")
        return 0.0


def compute_all_vqa_metrics(
    prediction: str,
    ground_truths: List[str],
    embeddings_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    Compute all VQA evaluation metrics at once.
    
    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        embeddings_dir: Path to embeddings directory (optional)
        
    Returns:
        Dictionary with all metric scores
    """
    return {
        'anls': compute_anls(prediction, ground_truths),
        'exact_match': compute_exact_match(prediction, ground_truths),
        'substring_match': compute_substring_match(prediction, ground_truths),
        'cosine_similarity': compute_cosine_similarity(prediction, ground_truths, embeddings_dir),
    }


def compute_all_ocr_metrics(
    prediction: str,
    reference: str
) -> Dict[str, float]:
    """
    Compute all OCR evaluation metrics at once.
    
    Args:
        prediction: Predicted text
        reference: Ground truth text
        
    Returns:
        Dictionary with all metric scores
    """
    return {
        'cer': calculate_cer(prediction, reference),
        'wer': calculate_wer(prediction, reference),
        'anls': compute_anls(prediction, [reference]),
        'exact_match': compute_exact_match(prediction, [reference]),
    }


def compute_metrics_for_multiple_references(
    prediction: str,
    references: List[str]
) -> Dict[str, float]:
    """
    Compute OCR metrics when multiple reference texts are available.
    Takes the best score across all references.
    
    Args:
        prediction: Predicted text
        references: List of ground truth texts
        
    Returns:
        Dictionary with best metric scores across all references
    """
    if not references:
        return {
            'cer': 1.0,
            'wer': 1.0,
            'anls': 0.0,
            'exact_match': 0.0,
        }
    
    # Compute metrics for each reference
    all_metrics = [compute_all_ocr_metrics(prediction, ref) for ref in references]
    
    # Return best scores (minimum for error rates, maximum for similarity)
    return {
        'cer': min(m['cer'] for m in all_metrics),
        'wer': min(m['wer'] for m in all_metrics),
        'anls': max(m['anls'] for m in all_metrics),
        'exact_match': max(m['exact_match'] for m in all_metrics),
    }


def aggregate_metrics(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple samples.
    
    Args:
        metric_dicts: List of metric dictionaries
        
    Returns:
        Dictionary with aggregated metrics (mean values)
    """
    if not metric_dicts:
        return {}
    
    # Get all metric names
    metric_names = set()
    for d in metric_dicts:
        metric_names.update(d.keys())
    
    # Compute mean for each metric
    aggregated = {}
    for metric in metric_names:
        values = [d.get(metric, 0.0) for d in metric_dicts if metric in d]
        if values:
            aggregated[metric] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
    
    return aggregated
