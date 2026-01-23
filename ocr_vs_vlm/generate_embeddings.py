#!/usr/bin/env python3
"""
Embedding generation script for QA datasets.

Usage:
    # Generate embeddings for DocVQA_mini questions
    python -m ocr_vs_vlm.generate_embeddings --dataset docvqa
    
    # Generate embeddings for InfographicVQA_mini questions
    python -m ocr_vs_vlm.generate_embeddings --dataset infographicvqa
    
    # Custom output directory
    python -m ocr_vs_vlm.generate_embeddings --dataset docvqa --output /path/to/output
    
    # Resume from checkpoint
    python -m ocr_vs_vlm.generate_embeddings --dataset docvqa --resume
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llms.embeddings import (
    EmbeddingConfig,
    EmbeddingPipeline,
    create_embeddings_for_dataset,
)
from ocr_vs_vlm.datasets.dataset_loaders_qa import DocVQAMiniDataset, InfographicVQAMiniDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def get_texts_from_dataset(dataset_name: str, dataset_root: Path) -> tuple[List[str], str]:
    """
    Load texts from a QA dataset.
    
    Args:
        dataset_name: Name of dataset ('docvqa' or 'infographicvqa')
        dataset_root: Root path to datasets_subsets
        
    Returns:
        Tuple of (texts list, dataset display name)
    """
    if dataset_name.lower() == 'docvqa':
        dataset = DocVQAMiniDataset(str(dataset_root / 'datasets_subsets'))
        texts = [sample.question for sample in dataset.samples]
        return texts, 'DocVQA_mini'
    
    elif dataset_name.lower() == 'infographicvqa':
        dataset = InfographicVQAMiniDataset(str(dataset_root / 'datasets_subsets'))
        texts = [sample.question for sample in dataset.samples]
        return texts, 'InfographicVQA_mini'
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for QA dataset questions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        '--dataset',
        choices=['docvqa', 'infographicvqa'],
        default='docvqa',
        help='Dataset to generate embeddings for',
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Custom output directory (default: datasets/[dataset]_embeddings)',
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for API requests (default: 100)',
    )
    
    parser.add_argument(
        '--model',
        default='text-embedding-3-large',
        help='Embedding model to use (default: text-embedding-3-large)',
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint if available',
    )
    
    parser.add_argument(
        '--cache-dir',
        type=Path,
        help='Directory for caching embeddings',
    )
    
    args = parser.parse_args()
    
    # Determine paths
    project_root = Path(__file__).parent.parent
    dataset_root = project_root / 'datasets' / 'datasets_subsets'
    
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        sys.exit(1)
    
    # Get texts from dataset
    try:
        texts, dataset_display_name = get_texts_from_dataset(args.dataset, project_root)
        logger.info(f"Loaded {len(texts)} texts from {dataset_display_name}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / 'datasets' / f"{dataset_display_name}_embeddings"
    
    # Create embedding configuration
    config = EmbeddingConfig(
        model=args.model,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )
    
    # Run embedding pipeline
    logger.info(f"Starting embedding pipeline")
    logger.info(f"  Dataset: {dataset_display_name}")
    logger.info(f"  Texts: {len(texts)}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Batch size: {args.batch_size}")
    
    try:
        pipeline = EmbeddingPipeline(output_dir, config)
        results = pipeline.embed_texts(texts, resume=args.resume)
        
        logger.info(f"✓ Embedding pipeline completed successfully")
        logger.info(f"  Embeddings saved to: {output_dir}")
        logger.info(f"  Summary:")
        for key, value in results['summary'].items():
            logger.info(f"    {key}: {value}")
        
    except Exception as e:
        logger.error(f"✗ Embedding pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
