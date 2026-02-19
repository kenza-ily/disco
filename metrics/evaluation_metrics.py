"""
Comprehensive evaluation metrics for OCR and VQA tasks.

Provides metrics including:
- CER (Character Error Rate)
- WER (Word Error Rate)
- EM (Exact Match)
- ANLS (Average Normalized Levenshtein Similarity)
- Cosine Similarity (using embeddings)
- Substring matching for VQA
- Ground truth parsing with support for triple-encoded JSON
- Multiple-choice answer normalization
"""

import logging
import json
import ast
import re
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


def compute_ground_truth_in_prediction(prediction: str, ground_truths: List[str], is_unanswerable: bool = False) -> float:
    """
    Check if any ground truth string appears in the prediction.
    
    Measures how much of the ground truth is covered by the prediction.
    
    Special case: If the question is marked as unanswerable, check if the prediction
    contains keywords indicating inability to answer: "can't", "cannot", "can not", or "not"
    
    Args:
        prediction: Predicted answer
        ground_truths: List of acceptable ground truth answers
        is_unanswerable: Whether the question is marked as unanswerable
        
    Returns:
        1.0 if any ground truth is substring of prediction (or if unanswerable and 
        prediction contains negation keywords), 0.0 otherwise
    """
    if not prediction:
        return 0.0
    
    pred = prediction.lower().strip()
    
    # Special handling for unanswerable questions (check before requiring ground truths)
    if is_unanswerable:
        # Check if prediction contains keywords indicating inability to answer
        # Keywords: "can't", "cannot", "can not", or "not" (as a word, not substring)
        unanswerable_keywords = ["can't", "cannot", "can not"]
        for keyword in unanswerable_keywords:
            if keyword in pred:
                return 1.0
        
        # Special check for "not" - should match as whole word or at start
        if "not " in pred or pred.startswith("not"):
            return 1.0
    
    # Standard ground truth matching (requires ground truths)
    if not ground_truths:
        return 0.0
    
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
        from models.embeddings import EmbeddingCalculator
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


# ============================================================================
# DATA PREPROCESSING AND NORMALIZATION FUNCTIONS
# ============================================================================

def parse_ground_truths(gt_string) -> List[str]:
    """
    Parse ground_truths from various formats to a list of strings.
    
    Handles multiple encoding formats:
    - Triple-encoded JSON: "[\"['value']\"]" → ["value"]
    - Double-encoded JSON: "["value"]" → ["value"]
    - Python literal syntax: "['value1', 'value2']" → ["value1", "value2"]
    - Single strings: "value" → ["value"]
    
    Args:
        gt_string: Ground truth in any supported format (str, list, or encoded JSON)
        
    Returns:
        List of parsed ground truth strings
    """
    # Handle None/NaN
    try:
        import pandas as pd
        if pd.isna(gt_string):
            return []
    except (ImportError, TypeError):
        if gt_string is None:
            return []
    
    # Already a list
    if isinstance(gt_string, list):
        return [str(x).strip() for x in gt_string if x]
    
    gt_str = str(gt_string).strip()
    if not gt_str:
        return []
    
    # Try JSON parsing first
    try:
        first_parse = json.loads(gt_str)
        
        # If it parsed to a list, process each item
        if isinstance(first_parse, list):
            result = []
            for item in first_parse:
                if isinstance(item, str):
                    item_stripped = item.strip()
                    
                    # Try JSON parsing
                    try:
                        inner_parse = json.loads(item_stripped)
                        if isinstance(inner_parse, list):
                            result.extend([str(x).strip() for x in inner_parse if x])
                        else:
                            result.append(str(inner_parse).strip())
                    except (json.JSONDecodeError, ValueError):
                        # Try Python literal eval for single-quote syntax
                        try:
                            inner_parse = ast.literal_eval(item_stripped)
                            if isinstance(inner_parse, list):
                                result.extend([str(x).strip() for x in inner_parse if x])
                            else:
                                result.append(str(inner_parse).strip())
                        except (ValueError, SyntaxError):
                            result.append(item_stripped)
                else:
                    result.append(str(item).strip())
            return result
        else:
            return [str(first_parse).strip()]
    
    except (json.JSONDecodeError, ValueError):
        # Top level is not JSON, try ast.literal_eval
        try:
            parsed = ast.literal_eval(gt_str)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
            return [str(parsed).strip()]
        except (ValueError, SyntaxError):
            # Return as-is
            return [gt_str]


def is_unanswerable(ground_truths: List[str]) -> bool:
    """
    Check if the ground truths indicate an unanswerable question.
    
    Args:
        ground_truths: List of ground truth answers
        
    Returns:
        True if any ground truth indicates unanswerable, False otherwise
    """
    unanswerable_keywords = [
        'unanswerable',
        'cannot',
        'not answerable',
        'no answer',
        'cannot be determined',
        'insufficient information',
        'unknown'
    ]
    
    if not ground_truths:
        return False
    
    combined = ' '.join(ground_truths).lower().strip()
    
    for keyword in unanswerable_keywords:
        if keyword in combined:
            return True
    
    return False


def extract_multiple_choice_answer(prediction: str) -> Optional[str]:
    """
    Extract the main answer letter/choice from a verbose prediction.
    
    For multiple-choice questions, models often return verbose explanations.
    This extracts just the answer choice (A, B, C, D, etc.)
    
    Examples:
        "b) 24%" → "B"
        "Answer: A" → "A"
        "The answer is C" → "C"
        "29.73%" → None (not a multiple choice)
    
    Args:
        prediction: Predicted answer (potentially verbose)
        
    Returns:
        Extracted choice letter (uppercase), or None if not a multiple-choice answer
    """
    if not prediction:
        return None
    
    pred = str(prediction).strip()
    
    # Pattern 1: Starts with letter + ) or .
    # e.g., "b) 24%", "A. answer text"
    match = re.match(r'^([a-zA-Z])[).\s]', pred)
    if match:
        return match.group(1).upper()
    
    # Pattern 2: "Answer: X" or "answer is X"
    match = re.search(r'(?:answer|choice|option)[\s:]+([a-zA-Z])', pred, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: Starts with lowercase and followed by ) or . with space
    # e.g., "c) some answer"
    match = re.match(r'^([a-zA-Z])\s*\)', pred)
    if match:
        return match.group(1).upper()
    
    return None


def normalize_prediction_for_comparison(prediction: str, ground_truths: List[str]) -> str:
    """
    Normalize a prediction for better comparison with ground truths.
    
    Applies various normalization strategies:
    - For multiple-choice: extract just the letter if GT is single letter
    - Strip brackets/quotes from parsed answers
    - Normalize whitespace
    
    Args:
        prediction: Raw prediction
        ground_truths: List of ground truths for context
        
    Returns:
        Normalized prediction
    """
    if not prediction:
        return ""
    
    pred = str(prediction).strip()
    
    # Check if ground truths are single letters (multiple choice)
    if ground_truths:
        gt_letters = [g.upper().strip() for g in ground_truths if len(g.strip()) == 1 and g.strip().isalpha()]
        
        if gt_letters:  # We're in multiple-choice territory
            extracted = extract_multiple_choice_answer(pred)
            if extracted:
                return extracted
    
    # Strip common bracket patterns left from parsing
    # e.g., "['answer']" → "answer"
    pred = re.sub(r"^[\[\(\"']+|[\]\)\"']+$", "", pred)
    
    # Normalize whitespace
    pred = ' '.join(pred.split())
    
    return pred


def preprocess_qa_sample(
    prediction: str,
    ground_truths_raw,
    normalize: bool = True,
    check_unanswerable: bool = True
) -> Tuple[str, List[str], bool]:
    """
    Comprehensive preprocessing for a single QA sample.
    
    Applies:
    1. Parse ground truths from encoded formats
    2. Check if question is unanswerable
    3. Normalize prediction for comparison
    
    Args:
        prediction: Raw prediction
        ground_truths_raw: Raw ground truths (any format)
        normalize: Whether to normalize prediction
        check_unanswerable: Whether to flag unanswerable questions
        
    Returns:
        Tuple of (normalized_prediction, parsed_gts, is_unanswerable)
    """
    # Parse ground truths
    parsed_gts = parse_ground_truths(ground_truths_raw)
    
    # Check for unanswerable
    is_unanswerable_q = check_unanswerable and is_unanswerable(parsed_gts)
    
    # Normalize prediction
    normalized_pred = normalize_prediction_for_comparison(prediction, parsed_gts) if normalize else str(prediction).strip()
    
    return normalized_pred, parsed_gts, is_unanswerable_q
