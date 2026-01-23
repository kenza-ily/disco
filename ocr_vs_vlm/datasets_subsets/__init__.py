"""
Dataset subsets for benchmarking.

Contains pre-sampled datasets with balanced language representation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from ..datasets.dataset_loaders import create_icdar_mini, save_icdar_mini_index, Sample

# DUDE sample loader exports
from .dude_sample_loader import (
    DUDESampleLoader,
    DUDEQAPair,
    DUDEDocument,
    DUDEUnifiedItem,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DUDESampleLoader",
    "DUDEQAPair",
    "DUDEDocument",
    "DUDEUnifiedItem",
    "generate_icdar_mini",
    "load_icdar_mini_index",
]


def generate_icdar_mini(icdar_root: str, samples_per_language: int = 50) -> Dict[str, List[Sample]]:
    """
    Generate ICDAR mini dataset with 50 samples per language.
    
    Args:
        icdar_root: Root path to ICDAR dataset
        samples_per_language: Samples to select per language
    
    Returns:
        Dict mapping language -> List[Sample]
    """
    logger.info(f"Generating ICDAR_mini with {samples_per_language} samples per language...")
    mini_dataset = create_icdar_mini(icdar_root, samples_per_language)
    
    # Save index
    subsets_dir = Path(__file__).parent
    save_icdar_mini_index(mini_dataset, str(subsets_dir))
    
    logger.info(f"ICDAR_mini dataset ready with {len(mini_dataset)} languages")
    return mini_dataset


def load_icdar_mini_index() -> Dict:
    """
    Load ICDAR mini index from saved files.
    
    Returns:
        Dict with ICDAR_mini metadata
    """
    subsets_dir = Path(__file__).parent
    index_file = subsets_dir / "icdar_mini_index.json"
    
    if not index_file.exists():
        logger.warning(f"ICDAR_mini index not found at {index_file}. Run generate_icdar_mini() first.")
        return {}
    
    with open(index_file, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    """Generate ICDAR_mini from command line."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    icdar_root = "/Users/kenzabenkirane/Documents/GitHub/research-playground/datasets/parsing/ICDAR"
    
    try:
        mini = generate_icdar_mini(icdar_root, samples_per_language=50)
        index = load_icdar_mini_index()
        
        print(f"\nICDAR_mini Generated Successfully:")
        print(f"  Languages: {index['total_languages']}")
        print(f"  Total Samples: {index['total_samples']}")
        print(f"  Languages: {list(index['languages'].keys())}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
