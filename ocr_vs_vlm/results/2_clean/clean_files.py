#!/usr/bin/env python3
"""
Results Cleaning and Consolidation Script

This script:
1. Validates result CSV files (checks row counts, errors, empty predictions)
2. Selects the best run per model per experiment
3. Moves invalid/incomplete files to zzz/ folder
4. Consolidates results into one CSV per dataset+experiment with model columns
5. Supports incremental mode to add new models without full rebuild

Usage:
    # Full rebuild
    python -m ocr_vs_vlm.results_clean.clean_files

    # Incremental mode (only process new models)
    python -m ocr_vs_vlm.results_clean.clean_files --incremental

    # Dry run (show what would be done)
    python -m ocr_vs_vlm.results_clean.clean_files --dry-run

    # Validate only (no consolidation)
    python -m ocr_vs_vlm.results_clean.clean_files --validate-only
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tiktoken

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Expected row counts per dataset
EXPECTED_ROWS = {
    "DocVQA_mini": 500,
    "InfographicVQA_mini": 500,
    "IAM_mini": 500,
    "ICDAR_mini": 500,
    "publaynet": 500,
    "publaynet_full": 500,
    "VOC2007": 238,
}

# Dataset categorization
QA_DATASETS = ["DocVQA_mini", "InfographicVQA_mini"]
PARSING_DATASETS = ["IAM_mini", "ICDAR_mini", "publaynet", "publaynet_full", "VOC2007"]

# Error patterns to detect in predictions
ERROR_PATTERNS = [
    "error",
    "sso token",
    "connection reset",
    "timeout",
    "rate limit",
    "api error",
    "failed",
    "exception",
]

# Paths
RESULTS_DIR = Path(__file__).parent.parent / "1_raw"
RESULTS_CLEAN_DIR = Path(__file__).parent
ZZZ_DIR = Path(__file__).parent.parent / "1_raw" / "zzz_ignore"

# Price data (loaded from prices.json)
PRICES_FILE = Path(__file__).parent.parent.parent / "llms" / "prices.json"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FileValidation:
    """Validation result for a single file."""
    file_path: Path
    dataset: str
    experiment: str
    model: str
    row_count: int
    expected_rows: int
    has_header: bool
    empty_predictions: int
    error_predictions: int
    is_valid: bool
    invalid_reason: Optional[str] = None
    timestamp: Optional[str] = None  # From filename if present


@dataclass
class ValidFile:
    """Information about a validated file selected for consolidation."""
    file_path: Path
    dataset: str
    experiment: str
    model: str
    row_count: int
    timestamp: Optional[str] = None


@dataclass
class ConsolidationResult:
    """Result of consolidating files for a dataset+experiment."""
    dataset: str
    experiment: str
    output_file: Path
    models_included: List[str]
    total_rows: int
    columns: List[str]


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def discover_result_files(results_dir: Path) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    """
    Discover all result CSV files organized by dataset/experiment/model.

    Returns:
        {dataset: {experiment: {model: [file_paths]}}}
    """
    files: Dict[str, Dict[str, Dict[str, List[Path]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for dataset_dir in results_dir.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name in ["zzz_ignore", "results_clean"]:
            continue

        dataset_name = dataset_dir.name
        
        # Handle QA datasets (DocVQA_mini, InfographicVQA_mini)
        if dataset_name in QA_DATASETS:
            for phase_dir in dataset_dir.iterdir():
                if not phase_dir.is_dir() or phase_dir.name.startswith("benchmark"):
                    continue
                experiment = phase_dir.name
                # CSV files are directly in the phase folder
                for csv_file in phase_dir.glob("*.csv"):
                    model = _extract_model_from_filename(csv_file.name)
                    if model:
                        files[dataset_name][experiment][model].append(csv_file)

        # Handle IAM_mini (organized by model folder)
        elif dataset_name == "IAM_mini":
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name.startswith(("benchmark", "execution")):
                    continue
                model = model_dir.name
                for csv_file in model_dir.glob("*.csv"):
                    # Extract phase from filename (e.g., phase_1_results.csv -> phase_1)
                    phase_match = re.match(r"(phase_\d+[a-z]?)_", csv_file.name)
                    if phase_match:
                        experiment = phase_match.group(1)
                        files[dataset_name][experiment][model].append(csv_file)

        # Handle ICDAR_mini (has date subfolder)
        elif dataset_name == "ICDAR_mini":
            for date_dir in dataset_dir.iterdir():
                if not date_dir.is_dir() or date_dir.name == "execution_summary.json":
                    continue
                for model_dir in date_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    model = model_dir.name
                    for csv_file in model_dir.glob("*.csv"):
                        phase_match = re.match(r"(phase_\d+[a-z]?)_", csv_file.name)
                        if phase_match:
                            experiment = phase_match.group(1)
                            files[dataset_name][experiment][model].append(csv_file)
                        else:
                            # Default phase for files without phase prefix
                            files[dataset_name]["phase_1"][model].append(csv_file)

        # Handle PubLayNet (P-A, P-B, P-C phases)
        elif dataset_name.startswith("publaynet"):
            for phase_dir in dataset_dir.iterdir():
                if not phase_dir.is_dir() or not phase_dir.name.startswith("P-"):
                    continue
                experiment = phase_dir.name
                for model_dir in phase_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    model = model_dir.name
                    for csv_file in model_dir.glob("*.csv"):
                        files[dataset_name][experiment][model].append(csv_file)

        # Handle VOC2007 (nested model/VOC2007/model structure)
        elif dataset_name == "VOC2007":
            for model_dir in dataset_dir.iterdir():
                if not model_dir.is_dir() or model_dir.name.startswith("execution"):
                    continue
                model = model_dir.name
                # Handle nested structure VOC2007/model/VOC2007/model/
                inner_path = model_dir / "VOC2007" / model
                if inner_path.exists():
                    search_dir = inner_path
                else:
                    search_dir = model_dir
                for csv_file in search_dir.glob("*.csv"):
                    phase_match = re.match(r"(phase_\d+[a-z]?)_", csv_file.name)
                    if phase_match:
                        experiment = phase_match.group(1)
                        files[dataset_name][experiment][model].append(csv_file)
                    else:
                        files[dataset_name]["phase_1"][model].append(csv_file)

    return files


def _extract_model_from_filename(filename: str) -> Optional[str]:
    """Extract model name from result filename."""
    # Pattern: {model}_results.csv or {model}_results_{timestamp}.csv
    match = re.match(r"(.+?)_results(?:_\d{8}_\d{6})?\.csv", filename)
    if match:
        return match.group(1)
    # Pattern: P-{phase}_{model}_results*.csv
    match = re.match(r"P-[A-C]_(.+?)_results", filename)
    if match:
        return match.group(1)
    return None


def _extract_timestamp_from_filename(filename: str) -> Optional[str]:
    """Extract timestamp from filename if present."""
    match = re.search(r"_(\d{8}_\d{6})\.csv", filename)
    if match:
        return match.group(1)
    return None


# ============================================================================
# FILE VALIDATION
# ============================================================================

def validate_file(file_path: Path, dataset: str, experiment: str, model: str) -> FileValidation:
    """Validate a single result file."""
    expected_rows = EXPECTED_ROWS.get(dataset, 500)
    timestamp = _extract_timestamp_from_filename(file_path.name)

    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            # Detect if file has header
            sample = f.read(4096)
            f.seek(0)
            
            # Check for common header indicators
            has_header = any(h in sample.lower() for h in ['sample_id', 'image_path', 'prediction', 'ground_truth'])
            
            reader = csv.reader(f)
            rows = list(reader)
            
            if has_header and rows:
                header = rows[0]
                data_rows = rows[1:]
            else:
                header = None
                data_rows = rows

            row_count = len(data_rows)
            
            # Find prediction column index
            pred_col_idx = None
            if header:
                for idx, col in enumerate(header):
                    if 'prediction' in col.lower():
                        pred_col_idx = idx
                        break
            else:
                # Assume prediction is column 4 or 7 based on dataset type
                pred_col_idx = 4 if dataset in QA_DATASETS else 7

            # Count empty and error predictions
            empty_predictions = 0
            error_predictions = 0
            
            if pred_col_idx is not None:
                for row in data_rows:
                    if len(row) > pred_col_idx:
                        pred = row[pred_col_idx].strip().lower() if row[pred_col_idx] else ""
                        if not pred:
                            empty_predictions += 1
                        elif any(err in pred for err in ERROR_PATTERNS):
                            error_predictions += 1

            # Determine validity
            is_valid = True
            invalid_reason = None

            # Check row count (allow 90% threshold for minor issues)
            if row_count < expected_rows * 0.9:
                is_valid = False
                invalid_reason = f"Insufficient rows: {row_count}/{expected_rows}"
            # Check error rate (more than 10% errors is problematic)
            elif error_predictions > row_count * 0.1:
                is_valid = False
                invalid_reason = f"High error rate: {error_predictions}/{row_count}"
            # Check empty predictions (more than 20% empty)
            elif empty_predictions > row_count * 0.2:
                is_valid = False
                invalid_reason = f"Too many empty predictions: {empty_predictions}/{row_count}"

            return FileValidation(
                file_path=file_path,
                dataset=dataset,
                experiment=experiment,
                model=model,
                row_count=row_count,
                expected_rows=expected_rows,
                has_header=has_header,
                empty_predictions=empty_predictions,
                error_predictions=error_predictions,
                is_valid=is_valid,
                invalid_reason=invalid_reason,
                timestamp=timestamp,
            )

    except Exception as e:
        return FileValidation(
            file_path=file_path,
            dataset=dataset,
            experiment=experiment,
            model=model,
            row_count=0,
            expected_rows=expected_rows,
            has_header=False,
            empty_predictions=0,
            error_predictions=0,
            is_valid=False,
            invalid_reason=f"Error reading file: {str(e)}",
            timestamp=timestamp,
        )


def select_best_file(validations: List[FileValidation]) -> Optional[FileValidation]:
    """
    Select the best file from multiple validations for the same model/experiment.

    Priority:
    1. Valid files (all rows, no errors)
    2. Highest row count
    3. Most recent timestamp (if tied)
    """
    # Filter to valid files first
    valid_files = [v for v in validations if v.is_valid]

    if not valid_files:
        # If no valid files, pick the best incomplete one
        valid_files = sorted(
            validations,
            key=lambda v: (v.row_count, v.timestamp or ""),
            reverse=True
        )
        if valid_files:
            return valid_files[0]
        return None

    # Sort by row count (desc), then timestamp (desc)
    valid_files.sort(
        key=lambda v: (v.row_count, v.timestamp or ""),
        reverse=True
    )

    return valid_files[0]


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def move_to_zzz(file_path: Path, reason: str, dry_run: bool = False) -> None:
    """Move an invalid file to the zzz folder, preserving relative structure."""
    rel_path = file_path.relative_to(RESULTS_DIR)
    dest_path = ZZZ_DIR / rel_path

    if dry_run:
        print(f"  [DRY RUN] Would move: {rel_path}")
        print(f"            Reason: {reason}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If destination exists, add timestamp suffix
    if dest_path.exists():
        stem = dest_path.stem
        suffix = dest_path.suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_path = dest_path.parent / f"{stem}_{timestamp}{suffix}"

    shutil.move(str(file_path), str(dest_path))
    print(f"  Moved: {rel_path} -> zzz/{rel_path}")
    print(f"  Reason: {reason}")


# ============================================================================
# CONSOLIDATION
# ============================================================================

def load_embeddings_for_phase(phase_dir: Path, phase: str) -> Dict:
    """
    Load embeddings from JSON file for a given phase, or return empty cache.

    Args:
        phase_dir: Path to phase directory in 1_raw (e.g., 1_raw/DocVQA_mini/QA1a)
        phase: Phase name (e.g., "QA1a")

    Returns:
        Dict with structure:
        {
            'ground_truths': {str(gt_list): [[emb1], [emb2], ...]},
            'predictions': {sample_id: {model_key: [emb]}}
        }
        Returns empty dict if no embeddings file found (embeddings will be generated on-the-fly)
    """
    # Find embeddings file (may have timestamp)
    embeddings_files = list(phase_dir.glob("embeddings_*.json"))

    if not embeddings_files:
        logger.info(f"No embeddings file found for phase {phase}, will generate embeddings on-the-fly")
        return {'ground_truths': {}, 'predictions': {}}

    # Take most recent if multiple
    emb_file = max(embeddings_files, key=lambda f: f.stat().st_mtime)

    try:
        with open(emb_file, 'r') as f:
            emb_data = json.load(f)

        logger.info(f"Loaded embeddings from {emb_file.name}")
        return emb_data

    except Exception as e:
        logger.warning(f"Failed to load embeddings from {emb_file}: {e}, will generate on-the-fly")
        return {'ground_truths': {}, 'predictions': {}}


def compute_cosine_similarity(pred_embedding: List[float], gt_embeddings: List[List[float]]) -> float:
    """
    Compute max cosine similarity between prediction and ground truth embeddings.

    Args:
        pred_embedding: Single embedding vector
        gt_embeddings: List of ground truth embedding vectors

    Returns:
        Max cosine similarity score (0.0-1.0), or 0.0 if computation fails
    """
    if not pred_embedding or not gt_embeddings:
        return 0.0

    try:
        from scipy.spatial.distance import cosine

        max_sim = 0.0
        for gt_emb in gt_embeddings:
            if len(pred_embedding) != len(gt_emb):
                continue
            sim = 1.0 - cosine(pred_embedding, gt_emb)
            max_sim = max(max_sim, sim)

        return float(max_sim)

    except Exception:
        return 0.0


def compute_missing_metrics(
    row: Dict,
    embeddings_data: Optional[Dict],
    actual_model: str,
    generate_embeddings: bool = True,
    embedding_calculator=None
) -> Dict[str, float]:
    """
    Compute the 4 missing metrics for a single row.

    Args:
        row: CSV row dict with prediction and ground_truths
        embeddings_data: Loaded embeddings dict (or None)
        actual_model: Model name for embedding lookup
        generate_embeddings: If True, generate embeddings if not found in cache

    Returns:
        Dict with keys: substring_match, prediction_in_ground_truth,
                       ground_truth_in_prediction, embedding_similarity
    """
    # Import metric functions from metrics module
    import sys
    from pathlib import Path

    # Add parent directory to path to import from ocr_vs_vlm.metrics
    metrics_path = Path(__file__).parent.parent.parent
    if str(metrics_path) not in sys.path:
        sys.path.insert(0, str(metrics_path))

    from ocr_vs_vlm.metrics.evaluation_metrics import (
        compute_substring_match,
        compute_prediction_in_ground_truth,
        compute_ground_truth_in_prediction
    )

    prediction = row.get('prediction', '')
    ground_truths_str = row.get('ground_truths', '[]')

    # Parse ground truths JSON
    try:
        import json as json_module
        ground_truths = json_module.loads(ground_truths_str)
        if not isinstance(ground_truths, list):
            ground_truths = [str(ground_truths)]
    except:
        ground_truths = []

    # Compute string-based metrics
    substring_match = compute_substring_match(prediction, ground_truths)
    pred_in_gt = compute_prediction_in_ground_truth(prediction, ground_truths)
    gt_in_pred = compute_ground_truth_in_prediction(prediction, ground_truths)

    # Compute embedding similarity
    embedding_similarity = 0.0

    if embeddings_data:
        # Try to get embeddings from cache
        sample_id = row.get('sample_id', '')

        # Get prediction embedding
        pred_embeddings = embeddings_data.get('predictions', {}).get(sample_id, {})
        pred_emb = pred_embeddings.get(actual_model)

        # Get ground truth embeddings
        gt_key = ground_truths_str  # Use exact JSON string as key
        gt_embs = embeddings_data.get('ground_truths', {}).get(gt_key)

        if pred_emb and gt_embs:
            embedding_similarity = compute_cosine_similarity(pred_emb, gt_embs)
        elif generate_embeddings and prediction and ground_truths and embedding_calculator:
            # Generate embeddings on-the-fly if not in cache
            try:
                # Generate prediction embedding
                pred_result = embedding_calculator.embed_text(prediction)
                pred_emb = pred_result.embedding if pred_result else []

                # Generate ground truth embeddings
                gt_embs = []
                for gt in ground_truths:
                    if gt:
                        gt_result = embedding_calculator.embed_text(gt)
                        gt_embs.append(gt_result.embedding)

                # Compute similarity
                if pred_emb and gt_embs:
                    embedding_similarity = compute_cosine_similarity(pred_emb, gt_embs)

                    # Store in cache for reuse
                    if sample_id not in embeddings_data.get('predictions', {}):
                        if 'predictions' not in embeddings_data:
                            embeddings_data['predictions'] = {}
                        embeddings_data['predictions'][sample_id] = {}
                    embeddings_data['predictions'][sample_id][actual_model] = pred_emb

                    if gt_key not in embeddings_data.get('ground_truths', {}):
                        if 'ground_truths' not in embeddings_data:
                            embeddings_data['ground_truths'] = {}
                        embeddings_data['ground_truths'][gt_key] = gt_embs

            except Exception as e:
                logger.warning(f"Failed to generate embeddings for sample {sample_id}: {e}")
    elif generate_embeddings and prediction and ground_truths and embedding_calculator:
        # No embeddings data at all, generate from scratch
        try:
            pred_result = embedding_calculator.embed_text(prediction)
            pred_emb = pred_result.embedding if pred_result else []

            gt_embs = []
            for gt in ground_truths:
                if gt:
                    gt_result = embedding_calculator.embed_text(gt)
                    gt_embs.append(gt_result.embedding)

            if pred_emb and gt_embs:
                embedding_similarity = compute_cosine_similarity(pred_emb, gt_embs)

        except Exception as e:
            logger.warning(f"Failed to generate embeddings: {e}")

    return {
        'substring_match': substring_match,
        'prediction_in_ground_truth': pred_in_gt,
        'ground_truth_in_prediction': gt_in_pred,
        'embedding_similarity': embedding_similarity
    }


def consolidate_qa_files(
    valid_files: Dict[str, ValidFile],
    dataset: str,
    experiment: str,
    output_dir: Path,
    dry_run: bool = False,
    incremental: bool = False,
) -> Optional[ConsolidationResult]:
    """
    Consolidate QA result files into a single CSV with model columns.

    Output format:
        sample_id, image_path, question, ground_truths,
        prediction_{model1}, anls_score_{model1}, inference_time_ms_{model1},
        prediction_{model2}, anls_score_{model2}, inference_time_ms_{model2},
        ...
    """
    if not valid_files:
        return None

    output_file = output_dir / dataset / f"{experiment}.csv"
    
    if dry_run:
        print(f"  [DRY RUN] Would create: {output_file}")
        print(f"            Models: {list(valid_files.keys())}")
        return None

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load embeddings for this dataset+phase
    embeddings_cache: Dict[str, Optional[Dict]] = {}  # {phase: embeddings_data}

    # Initialize embedding calculator once for reuse
    embedding_calculator = None
    try:
        from llms.embeddings import EmbeddingCalculator
        embedding_calculator = EmbeddingCalculator()
        logger.info(f"Initialized EmbeddingCalculator for computing missing metrics")
    except Exception as e:
        logger.warning(f"Could not initialize EmbeddingCalculator: {e}. Embedding similarity will be 0.0")

    # Load existing data if incremental mode
    existing_data: Dict[str, Dict[str, Any]] = {}
    existing_models: List[str] = []

    if incremental and output_file.exists():
        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_columns = reader.fieldnames or []
            # Extract existing model names from column headers
            for col in existing_columns:
                if col.startswith("prediction_"):
                    model = col.replace("prediction_", "")
                    if model not in existing_models:
                        existing_models.append(model)
            for row in reader:
                existing_data[row['sample_id']] = row

    # Collect all data by sample_id
    all_data: Dict[str, Dict[str, Any]] = existing_data.copy()
    all_models = set(existing_models)

    for model, vf in valid_files.items():
        # Read first row to determine actual model name from parsing_model and qa_model columns
        actual_model = model  # fallback to filename-based model

        with open(vf.file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            first_row = next(reader, None)

            if first_row:
                parsing_model = first_row.get('parsing_model', '')
                qa_model = first_row.get('qa_model', '')

                # Construct model name based on available information
                if parsing_model and qa_model:
                    # Both models present: use format "parsing__qa"
                    actual_model = f"{parsing_model}__{qa_model}"
                elif parsing_model:
                    # Only parsing model: use it directly
                    actual_model = parsing_model
                elif qa_model:
                    # Only QA model (shouldn't happen in QA datasets, but handle it)
                    actual_model = qa_model
                # else: keep filename-based model name

            # Reset file pointer
            f.seek(0)
            next(reader)  # Skip header again

        all_models.add(actual_model)

        with open(vf.file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = row.get('sample_id', '')
                if not sample_id:
                    continue

                if sample_id not in all_data:
                    all_data[sample_id] = {
                        'sample_id': sample_id,
                        'image_path': row.get('image_path', ''),
                        'question': row.get('question', ''),
                        'ground_truths': row.get('ground_truths', ''),
                    }

                # Add model-specific columns using actual_model name
                all_data[sample_id][f'prediction_{actual_model}'] = row.get('prediction', '')
                all_data[sample_id][f'anls_score_{actual_model}'] = row.get('anls_score', '')
                all_data[sample_id][f'exact_match_{actual_model}'] = row.get('exact_match', '')

                # Load embeddings for this phase if not already loaded
                if experiment not in embeddings_cache:
                    phase_dir = vf.file_path.parent  # Path to phase dir in 1_raw
                    embeddings_cache[experiment] = load_embeddings_for_phase(phase_dir, experiment)

                # Compute missing metrics on-the-fly
                missing_metrics = compute_missing_metrics(
                    row,
                    embeddings_cache.get(experiment),
                    actual_model,
                    generate_embeddings=True,
                    embedding_calculator=embedding_calculator
                )

                all_data[sample_id][f'embedding_similarity_{actual_model}'] = missing_metrics['embedding_similarity']
                all_data[sample_id][f'substring_match_{actual_model}'] = missing_metrics['substring_match']
                all_data[sample_id][f'prediction_in_ground_truth_{actual_model}'] = missing_metrics['prediction_in_ground_truth']
                all_data[sample_id][f'ground_truth_in_prediction_{actual_model}'] = missing_metrics['ground_truth_in_prediction']
                all_data[sample_id][f'inference_time_ms_{actual_model}'] = row.get('inference_time_ms', '')
                all_data[sample_id][f'extracted_text_{actual_model}'] = row.get('extracted_text', '')
                all_data[sample_id][f'error_{actual_model}'] = row.get('error', '')

    # Build column list
    base_cols = ['sample_id', 'image_path', 'question', 'ground_truths']
    model_cols = []
    sorted_models = sorted(all_models)
    for model in sorted_models:
        model_cols.extend([
            f'prediction_{model}',
            f'anls_score_{model}',
            f'exact_match_{model}',
            f'embedding_similarity_{model}',
            f'substring_match_{model}',
            f'prediction_in_ground_truth_{model}',
            f'ground_truth_in_prediction_{model}',
            f'inference_time_ms_{model}',
            f'extracted_text_{model}',
            f'error_{model}',
        ])
    
    columns = base_cols + model_cols

    # Write consolidated file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for sample_id in sorted(all_data.keys()):
            writer.writerow(all_data[sample_id])

    return ConsolidationResult(
        dataset=dataset,
        experiment=experiment,
        output_file=output_file,
        models_included=sorted_models,
        total_rows=len(all_data),
        columns=columns,
    )


def consolidate_parsing_files(
    valid_files: Dict[str, ValidFile],
    dataset: str,
    experiment: str,
    output_dir: Path,
    dry_run: bool = False,
    incremental: bool = False,
) -> Optional[ConsolidationResult]:
    """
    Consolidate parsing result files into a single CSV with model columns.

    Handles different schemas for different datasets:
    - IAM_mini: sample_id, ground_truth, prediction_{model}, ...
    - ICDAR_mini: sample_id, language, ground_truth, prediction_{model}, ...
    - PubLayNet: sample_id, ground_truth_boxes, predicted_boxes_{model}, ...
    - VOC2007: sample_id, language, ground_truth, prediction_{model}, ...
    """
    if not valid_files:
        return None

    output_file = output_dir / dataset / f"{experiment}.csv"

    if dry_run:
        print(f"  [DRY RUN] Would create: {output_file}")
        print(f"            Models: {list(valid_files.keys())}")
        return None

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine schema based on dataset
    if dataset.startswith("publaynet"):
        return _consolidate_publaynet(valid_files, output_file, incremental)
    else:
        return _consolidate_text_parsing(valid_files, dataset, experiment, output_file, incremental)


def _consolidate_text_parsing(
    valid_files: Dict[str, ValidFile],
    dataset: str,
    experiment: str,
    output_file: Path,
    incremental: bool = False,
) -> ConsolidationResult:
    """Consolidate text-based parsing results (IAM, ICDAR, VOC2007)."""
    
    # Load existing data if incremental
    existing_data: Dict[str, Dict[str, Any]] = {}
    existing_models: List[str] = []
    
    if incremental and output_file.exists():
        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_columns = reader.fieldnames or []
            for col in existing_columns:
                if col.startswith("prediction_"):
                    model = col.replace("prediction_", "")
                    if model not in existing_models:
                        existing_models.append(model)
            for row in reader:
                sample_id = row.get('sample_id', '')
                if sample_id:
                    existing_data[sample_id] = row

    all_data: Dict[str, Dict[str, Any]] = existing_data.copy()
    all_models = set(existing_models)

    for model, vf in valid_files.items():
        all_models.add(model)

        with open(vf.file_path, 'r', newline='', encoding='utf-8') as f:
            # Try to read as CSV with header
            sample = f.read(2048)
            f.seek(0)
            has_header = 'sample_id' in sample.lower() or 'image_path' in sample.lower()
            
            if has_header:
                reader = csv.DictReader(f)
                for row in reader:
                    sample_id = row.get('sample_id', '') or row.get(list(row.keys())[0], '')
                    if not sample_id:
                        continue

                    if sample_id not in all_data:
                        all_data[sample_id] = {
                            'sample_id': sample_id,
                            'image_path': row.get('image_path', row.get(list(row.keys())[1], '')),
                            'ground_truth': row.get('ground_truth', ''),
                            'language': row.get('language', ''),
                            'dataset': row.get('dataset', dataset),
                        }

                    all_data[sample_id][f'prediction_{model}'] = row.get('prediction', '')
                    all_data[sample_id][f'inference_time_ms_{model}'] = row.get('inference_time_ms', row.get('prediction_inference_time_ms', ''))
                    all_data[sample_id][f'tokens_used_{model}'] = row.get('tokens_used', '')
                    all_data[sample_id][f'error_{model}'] = row.get('error', row.get('prediction_error', ''))
            else:
                # Handle headerless CSV (positional columns)
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 5:
                        continue
                    # Assume: sample_id, image_path, ..., ground_truth, prediction, ...
                    sample_id = row[0]
                    if sample_id not in all_data:
                        all_data[sample_id] = {
                            'sample_id': sample_id,
                            'image_path': row[1] if len(row) > 1 else '',
                            'ground_truth': row[6] if len(row) > 6 else '',
                            'language': row[5] if len(row) > 5 else '',
                            'dataset': dataset,
                        }
                    # Prediction is typically column 7
                    all_data[sample_id][f'prediction_{model}'] = row[7] if len(row) > 7 else ''
                    all_data[sample_id][f'inference_time_ms_{model}'] = row[9] if len(row) > 9 else ''

    # Build columns
    base_cols = ['sample_id', 'image_path', 'ground_truth', 'language', 'dataset']
    model_cols = []
    sorted_models = sorted(all_models)
    for model in sorted_models:
        model_cols.extend([
            f'prediction_{model}',
            f'inference_time_ms_{model}',
            f'tokens_used_{model}',
            f'error_{model}',
        ])

    columns = base_cols + model_cols

    # Write
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for sample_id in sorted(all_data.keys()):
            writer.writerow(all_data[sample_id])

    return ConsolidationResult(
        dataset=dataset,
        experiment=experiment,
        output_file=output_file,
        models_included=sorted_models,
        total_rows=len(all_data),
        columns=columns,
    )


def _consolidate_publaynet(
    valid_files: Dict[str, ValidFile],
    output_file: Path,
    incremental: bool = False,
) -> ConsolidationResult:
    """Consolidate PubLayNet layout detection results."""
    
    existing_data: Dict[str, Dict[str, Any]] = {}
    existing_models: List[str] = []
    
    if incremental and output_file.exists():
        with open(output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_columns = reader.fieldnames or []
            for col in existing_columns:
                if col.startswith("predicted_boxes_"):
                    model = col.replace("predicted_boxes_", "")
                    if model not in existing_models:
                        existing_models.append(model)
            for row in reader:
                sample_id = row.get('sample_id', '')
                if sample_id:
                    existing_data[sample_id] = row

    all_data: Dict[str, Dict[str, Any]] = existing_data.copy()
    all_models = set(existing_models)
    vf = None  # Track last valid file for metadata

    for model, vf in valid_files.items():
        all_models.add(model)

        with open(vf.file_path, 'r', newline='', encoding='utf-8') as f:
            # Check if file has header
            sample = f.read(1024)
            f.seek(0)
            has_header = 'sample_id' in sample.lower() or 'image_path' in sample.lower()
            
            if has_header:
                reader = csv.DictReader(f)
                for row in reader:
                    sample_id = row.get('sample_id', '')
                    if not sample_id:
                        continue

                    if sample_id not in all_data:
                        all_data[sample_id] = {
                            'sample_id': sample_id,
                            'image_path': row.get('image_path', ''),
                            'ground_truth_boxes': row.get('ground_truth_boxes', ''),
                            'phase': row.get('phase', ''),
                        }

                    all_data[sample_id][f'predicted_boxes_{model}'] = row.get('predicted_boxes', '')
                    all_data[sample_id][f'inference_time_ms_{model}'] = row.get('inference_time_ms', '')
                    all_data[sample_id][f'error_{model}'] = row.get('error', '')
            else:
                # Handle headerless CSV
                # Format: sample_id, image_id, model, phase, ground_truth_boxes, predicted_boxes, inference_time_ms, error, timestamp
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 6:
                        continue
                    sample_id = row[0]
                    if not sample_id:
                        continue

                    if sample_id not in all_data:
                        all_data[sample_id] = {
                            'sample_id': sample_id,
                            'image_path': row[1] if len(row) > 1 else '',
                            'ground_truth_boxes': row[4] if len(row) > 4 else '',
                            'phase': row[3] if len(row) > 3 else '',
                        }

                    all_data[sample_id][f'predicted_boxes_{model}'] = row[5] if len(row) > 5 else ''
                    all_data[sample_id][f'inference_time_ms_{model}'] = row[6] if len(row) > 6 else ''
                    all_data[sample_id][f'error_{model}'] = row[7] if len(row) > 7 else ''

    # Build columns
    base_cols = ['sample_id', 'image_path', 'ground_truth_boxes', 'phase']
    model_cols = []
    sorted_models = sorted(all_models)
    for model in sorted_models:
        model_cols.extend([
            f'predicted_boxes_{model}',
            f'inference_time_ms_{model}',
            f'error_{model}',
        ])

    columns = base_cols + model_cols
    dataset = "publaynet"
    experiment = vf.file_path.parent.parent.name if vf else "unknown"

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for sample_id in sorted(all_data.keys()):
            writer.writerow(all_data[sample_id])

    return ConsolidationResult(
        dataset=dataset,
        experiment=experiment,
        output_file=output_file,
        models_included=sorted_models,
        total_rows=len(all_data),
        columns=columns,
    )


# ============================================================================
# VALID FILES TRACKING
# ============================================================================

VALID_FILES: Dict[str, Dict[str, Dict[str, ValidFile]]] = {}
"""
Selected valid files organized by dataset/experiment/model.
Structure: {dataset: {experiment: {model: ValidFile}}}
"""


def build_valid_files_dict(
    all_files: Dict[str, Dict[str, Dict[str, List[Path]]]],
    dry_run: bool = False,
) -> Dict[str, Dict[str, Dict[str, ValidFile]]]:
    """
    Validate all discovered files and select the best one per model/experiment.
    
    Returns the VALID_FILES dict and moves invalid files to zzz/.
    """
    valid_files: Dict[str, Dict[str, Dict[str, ValidFile]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    print("\n" + "=" * 70)
    print("VALIDATING RESULT FILES")
    print("=" * 70)

    for dataset, experiments in all_files.items():
        print(f"\n📁 {dataset}")
        
        for experiment, models in experiments.items():
            print(f"  📂 {experiment}")
            
            for model, file_paths in models.items():
                # Validate all files for this model/experiment
                validations = [
                    validate_file(fp, dataset, experiment, model)
                    for fp in file_paths
                ]

                # Select best file
                best = select_best_file(validations)

                if best and best.is_valid:
                    print(f"    ✅ {model}: {best.row_count} rows [{best.file_path.name}]")
                    valid_files[dataset][experiment][model] = ValidFile(
                        file_path=best.file_path,
                        dataset=dataset,
                        experiment=experiment,
                        model=model,
                        row_count=best.row_count,
                        timestamp=best.timestamp,
                    )
                elif best:
                    print(f"    ⚠️  {model}: {best.row_count}/{best.expected_rows} rows (using incomplete)")
                    valid_files[dataset][experiment][model] = ValidFile(
                        file_path=best.file_path,
                        dataset=dataset,
                        experiment=experiment,
                        model=model,
                        row_count=best.row_count,
                        timestamp=best.timestamp,
                    )
                else:
                    print(f"    ❌ {model}: No valid files found")

                # Move non-selected files to zzz
                for v in validations:
                    if best is None or v.file_path != best.file_path:
                        if not v.is_valid:
                            move_to_zzz(v.file_path, v.invalid_reason or "Not selected", dry_run)
                        elif len(file_paths) > 1:
                            move_to_zzz(v.file_path, "Duplicate - newer/better file selected", dry_run)

    return valid_files


def save_valid_files_list(valid_files: Dict, output_path: Path) -> None:
    """Save the list of valid files to a JSON file."""
    output = {
        "generated_at": datetime.now().isoformat(),
        "valid_files": {},
    }

    for dataset, experiments in valid_files.items():
        output["valid_files"][dataset] = {}
        for experiment, models in experiments.items():
            output["valid_files"][dataset][experiment] = {}
            for model, vf in models.items():
                output["valid_files"][dataset][experiment][model] = {
                    "file_path": str(vf.file_path),
                    "row_count": vf.row_count,
                    "timestamp": vf.timestamp,
                }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved valid files list to: {output_path}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Validate and consolidate benchmark result files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only process new models, preserving existing consolidated files.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate files, don't consolidate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Process only a specific dataset (e.g., DocVQA_mini, InfographicVQA_mini)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("OCR vs VLM Benchmark Results Cleaner")
    print("=" * 70)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Output directory: {RESULTS_CLEAN_DIR}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'INCREMENTAL' if args.incremental else 'FULL REBUILD'}")
    if args.dataset:
        print(f"Dataset filter: {args.dataset}")

    # Step 1: Discover all result files
    print("\n🔍 Discovering result files...")
    all_files = discover_result_files(RESULTS_DIR)

    # Apply dataset filter if specified
    if args.dataset:
        if args.dataset in all_files:
            all_files = {args.dataset: all_files[args.dataset]}
        else:
            print(f"❌ Dataset '{args.dataset}' not found. Available datasets: {', '.join(all_files.keys())}")
            return

    total_files = sum(
        len(files)
        for experiments in all_files.values()
        for models in experiments.values()
        for files in models.values()
    )
    print(f"   Found {total_files} result files across {len(all_files)} datasets")

    # Step 2: Validate and select best files
    valid_files = build_valid_files_dict(all_files, args.dry_run)

    # Save valid files list
    valid_files_path = RESULTS_CLEAN_DIR / "valid_files.json"
    if not args.dry_run:
        save_valid_files_list(valid_files, valid_files_path)

    if args.validate_only:
        print("\n✅ Validation complete (--validate-only mode)")
        return

    # Step 3: Consolidate files
    print("\n" + "=" * 70)
    print("CONSOLIDATING RESULT FILES")
    print("=" * 70)

    consolidation_results: List[ConsolidationResult] = []

    for dataset, experiments in valid_files.items():
        print(f"\n📊 {dataset}")
        
        for experiment, models in experiments.items():
            if dataset in QA_DATASETS:
                result = consolidate_qa_files(
                    models, dataset, experiment, RESULTS_CLEAN_DIR,
                    args.dry_run, args.incremental
                )
            else:
                result = consolidate_parsing_files(
                    models, dataset, experiment, RESULTS_CLEAN_DIR,
                    args.dry_run, args.incremental
                )

            if result:
                consolidation_results.append(result)
                print(f"  ✅ {experiment}: {result.total_rows} rows, {len(result.models_included)} models")
                print(f"     Models: {', '.join(result.models_included)}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Datasets processed: {len(valid_files)}")
    print(f"Files consolidated: {len(consolidation_results)}")

    if not args.dry_run:
        print(f"\nOutput files saved to: {RESULTS_CLEAN_DIR}")
        print("\nNext steps:")
        print("  1. Review consolidated CSVs in results_clean/{dataset}/")
        print("  2. Run experiments.md and models.md generation")


if __name__ == "__main__":
    main()
