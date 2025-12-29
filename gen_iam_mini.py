#!/usr/bin/env python3
"""
Create IAM_mini subset with 500 randomly selected samples.
Stores absolute paths to images for reliable location.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_iam_mini(
    iam_root: str = "datasets/parsing/IAM",
    output_dir: str = "ocr_vs_vlm/datasets_subsets/iam_mini",
    num_samples: int = 500,
    seed: int = 42
):
    """
    Create IAM_mini dataset with randomly selected samples.
    Uses absolute paths for images.
    
    Args:
        iam_root: Root path to IAM dataset
        output_dir: Output directory for IAM_mini
        num_samples: Number of samples to select
        seed: Random seed for reproducibility
    """
    
    iam_root = Path(iam_root).resolve()
    output_dir = Path(output_dir)
    data_dir = iam_root / "data"
    
    if not data_dir.exists():
        raise FileNotFoundError(f"IAM data directory not found: {data_dir}")
    
    # Collect all images
    logger.info("Collecting all IAM images...")
    all_images = list(sorted(data_dir.glob("*/*.png")))
    logger.info(f"Found {len(all_images)} total images")
    
    # Randomly select samples
    random.seed(seed)
    selected_images = random.sample(all_images, min(num_samples, len(all_images)))
    logger.info(f"Selected {len(selected_images)} random samples")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create samples list
    samples = []
    for idx, image_path in enumerate(sorted(selected_images)):
        sample_id = f"iam_{image_path.parent.name}_{image_path.stem}"
        
        # Get image size
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                image_size = img.size
        except Exception as e:
            logger.warning(f"Could not get size for {image_path}: {e}")
            image_size = [0, 0]
        
        sample = {
            "sample_id": sample_id,
            "image_path": str(image_path),  # Use absolute path for reliable location
            "ground_truth": "",  # Empty for IAM (no ground truth provided)
            "metadata": {
                "dataset": "IAM",
                "writer_id": image_path.stem.split('_')[0] if '_' in image_path.stem else image_path.stem.split('-')[0],
                "image_size": image_size,
            }
        }
        samples.append(sample)
    
    # Create single JSON file for all IAM_mini samples
    iam_mini_data = {
        "dataset": "IAM_mini",
        "num_samples": len(samples),
        "seed": seed,
        "samples": samples
    }
    
    output_file = output_dir / "iam_mini.json"
    with open(output_file, 'w') as f:
        json.dump(iam_mini_data, f, indent=2)
    
    logger.info(f"✓ Created {output_file}")
    
    # Create index file
    index_data = {
        "dataset": "IAM_mini",
        "version": "1.0",
        "num_samples": len(samples),
        "files": {
            "iam_mini.json": {
                "num_samples": len(samples),
                "description": "All 500 randomly selected IAM samples"
            }
        }
    }
    
    index_file = output_dir / "iam_mini_index.json"
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    logger.info(f"✓ Created {index_file}")
    
    logger.info(f"\nIAM_mini created at {output_dir}")
    logger.info(f"  Total samples: {len(samples)}")
    logger.info(f"  Image paths: Absolute paths stored in JSON")
    logger.info(f"  JSON files: {output_dir}/iam_mini.json, {output_dir}/iam_mini_index.json")
    
    return output_dir


if __name__ == '__main__':
    create_iam_mini()
