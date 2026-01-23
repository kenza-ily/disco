#!/usr/bin/env python3
"""
Compute embeddings for QA results in results_clean directory.

This script:
1. Reads existing CSV files from results_clean
2. Computes embeddings for predictions and ground truths (once per unique GT)
3. Computes new metrics (substring matches, embedding similarity)
4. Saves updated CSV with new metric columns
5. Saves embeddings separately in a _embeddings.json file
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr_vs_vlm.metrics.evaluation_metrics import (
    compute_substring_match,
    compute_prediction_in_ground_truth,
    compute_ground_truth_in_prediction,
    compute_embedding_similarity,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def compute_ground_truth_embeddings(ground_truths_list: List[str]) -> Dict[str, List[List[float]]]:
    """
    Compute embeddings for unique ground truths.

    Args:
        ground_truths_list: List of ground truth strings (may contain duplicates)

    Returns:
        Dictionary mapping ground truth JSON string to list of embedding vectors (one per answer)
    """
    unique_gts = list(set(ground_truths_list))
    gt_embeddings = {}

    logger.info(f"Computing embeddings for {len(unique_gts)} unique ground truths...")

    from llms.embeddings import EmbeddingCalculator
    calc = EmbeddingCalculator()

    for gt in tqdm(unique_gts, desc="GT embeddings"):
        if not gt:
            continue
        try:
            # Parse JSON if needed
            if gt.startswith('['):
                answers = json.loads(gt)
            else:
                answers = [gt]

            # Compute embedding for each answer
            answer_embeddings = []
            for answer in answers:
                if answer:
                    result = calc.embed_text(answer)
                    answer_embeddings.append(result.embedding)

            gt_embeddings[gt] = answer_embeddings

        except Exception as e:
            logger.warning(f"Failed to compute embedding for GT '{gt[:50]}...': {e}")
            gt_embeddings[gt] = []

    return gt_embeddings


def process_qa_csv(csv_path: Path, output_dir: Path):
    """
    Process a single QA results CSV file.

    Args:
        csv_path: Path to input CSV
        output_dir: Directory to save output files
    """
    logger.info(f"\nProcessing: {csv_path.name}")

    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"  Loaded {len(df)} rows")

    # Identify model columns (predictions end with model name)
    model_cols = [col for col in df.columns if col.startswith('prediction_') and col != 'prediction']
    models = [col.replace('prediction_', '') for col in model_cols]

    logger.info(f"  Found {len(models)} models: {models}")

    # Compute ground truth embeddings once
    gt_embeddings_cache = compute_ground_truth_embeddings(df['ground_truths'].tolist())

    # Storage for all embeddings
    all_embeddings = {
        'ground_truths': {},
        'predictions': {}
    }

    # Process each model
    for model in tqdm(models, desc="Models"):
        pred_col = f'prediction_{model}'

        if pred_col not in df.columns:
            continue

        # Add new metric columns
        substring_col = f'substring_match_{model}'
        pred_in_gt_col = f'prediction_in_ground_truth_{model}'
        gt_in_pred_col = f'ground_truth_in_prediction_{model}'
        emb_sim_col = f'embedding_similarity_{model}'

        substring_scores = []
        pred_in_gt_scores = []
        gt_in_pred_scores = []
        emb_sim_scores = []

        model_embeddings = {}

        for idx, row in df.iterrows():
            prediction = str(row[pred_col]) if pd.notna(row[pred_col]) else ""
            ground_truths_str = str(row['ground_truths']) if pd.notna(row['ground_truths']) else "[]"

            # Parse ground truths
            try:
                if ground_truths_str.startswith('['):
                    ground_truths = json.loads(ground_truths_str)
                else:
                    ground_truths = [ground_truths_str]
            except:
                ground_truths = [ground_truths_str]

            # Compute string-based metrics
            substring_scores.append(compute_substring_match(prediction, ground_truths))
            pred_in_gt_scores.append(compute_prediction_in_ground_truth(prediction, ground_truths))
            gt_in_pred_scores.append(compute_ground_truth_in_prediction(prediction, ground_truths))

            # Compute embedding similarity
            sample_id = row['sample_id']

            try:
                # Get prediction embedding (compute once per prediction)
                pred_key = f"{sample_id}_{model}"
                if prediction and pred_key not in model_embeddings:
                    from llms.embeddings import EmbeddingCalculator
                    calc = EmbeddingCalculator()
                    result = calc.embed_text(prediction)
                    model_embeddings[pred_key] = result.embedding

                pred_emb = model_embeddings.get(pred_key, [])

                # Get GT embeddings from cache (already computed)
                gt_embs = gt_embeddings_cache.get(ground_truths_str, [])

                # Compute cosine similarity (max over all GT answers)
                if pred_emb and gt_embs:
                    from scipy.spatial.distance import cosine
                    max_sim = 0.0
                    for gt_emb in gt_embs:
                        if gt_emb and len(gt_emb) == len(pred_emb):
                            sim = 1 - cosine(pred_emb, gt_emb)
                            max_sim = max(max_sim, sim)
                    emb_sim_scores.append(float(max_sim))
                else:
                    emb_sim_scores.append(0.0)

            except Exception as e:
                logger.debug(f"Embedding error for {sample_id}: {e}")
                emb_sim_scores.append(0.0)

        # Add columns to dataframe
        df[substring_col] = substring_scores
        df[pred_in_gt_col] = pred_in_gt_scores
        df[gt_in_pred_col] = gt_in_pred_scores
        df[emb_sim_col] = emb_sim_scores

        # Store embeddings
        all_embeddings['predictions'][model] = model_embeddings

    # Add ground truth embeddings to storage
    all_embeddings['ground_truths'] = gt_embeddings_cache

    # Save updated CSV
    output_csv = output_dir / csv_path.name
    df.to_csv(output_csv, index=False)
    logger.info(f"  ✓ Saved updated CSV: {output_csv}")

    # Save embeddings separately
    embeddings_file = output_dir / f"{csv_path.stem}_embeddings.json"
    with open(embeddings_file, 'w') as f:
        json.dump(all_embeddings, f, indent=2)
    logger.info(f"  ✓ Saved embeddings: {embeddings_file}")

    # Print summary stats
    logger.info(f"  Summary:")
    logger.info(f"    - Rows: {len(df)}")
    logger.info(f"    - Models: {len(models)}")
    logger.info(f"    - Unique GTs: {len(gt_embeddings_cache)}")
    logger.info(f"    - New columns added: 4 per model ({len(models) * 4} total)")


def main():
    """Process all QA CSV files in results_clean."""
    import argparse

    parser = argparse.ArgumentParser(description='Compute embeddings and new metrics for QA results')
    parser.add_argument('--test', action='store_true', help='Test mode: process only first file from each dataset')
    parser.add_argument('--dataset', type=str, help='Process specific dataset only (DocVQA_mini or InfographicVQA_mini)')
    parser.add_argument('--file', type=str, help='Process specific CSV file')
    args = parser.parse_args()

    results_clean_dir = Path(__file__).parent / "ocr_vs_vlm" / "results_clean"

    # Handle specific file
    if args.file:
        csv_file = Path(args.file)
        if not csv_file.exists():
            logger.error(f"File not found: {csv_file}")
            return
        output_dir = csv_file.parent
        logger.info(f"Processing single file: {csv_file}")
        process_qa_csv(csv_file, output_dir)
        return

    # Find QA datasets
    qa_datasets = ["DocVQA_mini", "InfographicVQA_mini"]
    if args.dataset:
        if args.dataset in qa_datasets:
            qa_datasets = [args.dataset]
        else:
            logger.error(f"Unknown dataset: {args.dataset}. Must be one of {qa_datasets}")
            return

    for dataset in qa_datasets:
        dataset_dir = results_clean_dir / dataset

        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"Processing dataset: {dataset}")
        logger.info(f"{'='*70}")

        # Find all CSV files (exclude _embeddings files and combined files)
        csv_files = [f for f in dataset_dir.glob("QA*.csv")
                     if not f.stem.endswith('_embeddings')
                     and len(f.stem) <= 4]  # QA1a, QA2b, etc. are 4 chars or less
        csv_files = sorted(csv_files)
        logger.info(f"Found {len(csv_files)} CSV files")

        # Test mode: process only first file
        if args.test and csv_files:
            csv_files = [csv_files[0]]
            logger.info(f"TEST MODE: Processing only {csv_files[0].name}")

        # Create output directory (overwrite in place)
        output_dir = dataset_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        for csv_file in csv_files:
            try:
                process_qa_csv(csv_file, output_dir)
            except Exception as e:
                logger.error(f"Failed to process {csv_file.name}: {e}")
                import traceback
                traceback.print_exc()

    logger.info(f"\n{'='*70}")
    logger.info("✓ All files processed successfully!")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
