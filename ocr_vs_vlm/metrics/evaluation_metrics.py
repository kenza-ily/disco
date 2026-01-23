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


def compute_prediction_in_ground_truth(prediction: str, ground_truths: List[str]) -> float:
    """
    Check if the entire prediction string appears in any ground truth.
    
    Measures how much of the prediction is covered by ground truth text.
    
    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        
    Returns:
        1.0 if prediction is substring of any ground truth, 0.0 otherwise
    """
    if not prediction or not ground_truths:
        return 0.0
    
    pred = prediction.lower().strip()
    
    for gt in ground_truths:
        gt = gt.lower().strip()
        if pred in gt:
            return 1.0
    
    return 0.0


def compute_ground_truth_in_prediction(prediction: str, ground_truths: List[str]) -> float:
    """
    Check if any ground truth string appears in the prediction.
    
    Measures how much of the ground truth is covered by the prediction.
    
    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        
    Returns:
        1.0 if any ground truth is substring of prediction, 0.0 otherwise
    """
    if not prediction or not ground_truths:
        return 0.0
    
    pred = prediction.lower().strip()
    
    for gt in ground_truths:
        gt = gt.lower().strip()
        if gt in pred:
            return 1.0
    
    return 0.0


def compute_embedding_similarity(
    prediction: str,
    ground_truths: List[str]
) -> Tuple[List[float], List[float]]:
    """
    Compute embedding vectors for prediction and ground truths.
    
    Uses the embedding_integration module to generate embeddings.
    
    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        
    Returns:
        Tuple of (prediction_embedding, ground_truth_embeddings)
        Each embedding is a list of floats
    """
    try:
        from llms.embeddings import EmbeddingCalculator
    except ImportError:
        logger.warning("embedding modules not available, returning empty embeddings")
        return [], []
    
    try:
        calc = EmbeddingCalculator()
        
        # Get embedding for prediction
        pred_result = calc.embed_text(prediction) if prediction else None
        pred_embedding = pred_result.embedding if pred_result else []
        
        # Get embeddings for ground truths
        gt_embeddings = []
        for gt in ground_truths:
            if gt:
                gt_result = calc.embed_text(gt)
                gt_embeddings.append(gt_result.embedding)
        
        return pred_embedding, gt_embeddings
    
    except Exception as e:
        logger.warning(f"Error computing embeddings: {e}")
        return [], []


def compute_max_embedding_similarity(
    prediction: str,
    ground_truths: List[str]
) -> float:
    """
    Compute maximum cosine similarity between prediction and ground truths using embeddings.
    
    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        
    Returns:
        Maximum cosine similarity score (0.0 to 1.0, higher is better)
    """
    pred_emb, gt_embs = compute_embedding_similarity(prediction, ground_truths)
    
    if not pred_emb or not gt_embs:
        return 0.0
    
    try:
        from scipy.spatial.distance import cosine
    except ImportError:
        logger.warning("scipy not available, returning 0.0 for embedding similarity")
        return 0.0
    
    try:
        max_similarity = 0.0
        for gt_emb in gt_embs:
            similarity = 1 - cosine(pred_emb, gt_emb)
            max_similarity = max(max_similarity, similarity)
        
        return float(max_similarity)
    
    except Exception as e:
        logger.warning(f"Error computing cosine similarity: {e}")
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
        embeddings_dir: Path to embeddings directory (optional, kept for compatibility)
        
    Returns:
        Maximum cosine similarity score (0.0 to 1.0, higher is better)
    """
    return compute_max_embedding_similarity(prediction, ground_truths)


def compute_all_vqa_metrics(
    prediction: str,
    ground_truths: List[str],
    embeddings_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    Compute all VQA evaluation metrics at once.
    
    Metrics include:
    - anls: Average Normalized Levenshtein Similarity
    - exact_match: Exact string match
    - substring_match: Bidirectional substring containment
    - prediction_in_ground_truth: Prediction is substring of ground truth
    - ground_truth_in_prediction: Ground truth is substring of prediction
    - embedding_similarity: Max cosine similarity of embeddings
    
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
        'prediction_in_ground_truth': compute_prediction_in_ground_truth(prediction, ground_truths),
        'ground_truth_in_prediction': compute_ground_truth_in_prediction(prediction, ground_truths),
        'embedding_similarity': compute_max_embedding_similarity(prediction, ground_truths),
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
