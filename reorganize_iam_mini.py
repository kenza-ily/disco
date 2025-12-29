#!/usr/bin/env python3
"""
Reorganize IAM_mini: Create dedicated folders with cropped printed/handwritten images.

Structure:
iam_mini/
├── iam_000_a01-000u/
│   ├── printed.png
│   └── handwritten.png
├── iam_001_a01-000v/
│   ├── printed.png
│   └── handwritten.png
├── ... (500 folders total)
└── iam_mini_index.json
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reorganize_iam_mini():
    """Reorganize IAM_mini with per-image folders containing cropped images."""
    
    # Paths
    script_dir = Path(__file__).parent
    subsets_dir = script_dir / "ocr_vs_vlm" / "datasets_subsets"
    iam_mini_dir = subsets_dir / "iam_mini"
    old_json = iam_mini_dir / "iam_mini.json"
    
    # Load current structure
    logger.info(f"Loading {old_json}...")
    with open(old_json) as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    logger.info(f"Found {len(samples)} samples")
    
    # Import Line 2 detection
    try:
        from ocr_vs_vlm.line2_detection import find_line2, crop_handwritten_only, crop_printed_only
    except ImportError:
        logger.error("Could not import line2_detection module. Make sure it's in ocr_vs_vlm/")
        return False
    
    # Process each sample
    logger.info("Creating per-image folders with cropped images...")
    new_samples = []
    
    for i, sample in enumerate(tqdm(samples, desc="Processing images")):
        sample_id = sample['sample_id']
        image_path = sample['image_path']
        
        # Create folder for this image
        img_folder = iam_mini_dir / sample_id
        img_folder.mkdir(parents=True, exist_ok=True)
        
        try:
            # Detect Line 2
            line2 = find_line2(image_path)
            
            # Crop and save
            printed = crop_printed_only(image_path, line2)
            handwritten = crop_handwritten_only(image_path, line2)
            
            printed_path = img_folder / "printed.png"
            handwritten_path = img_folder / "handwritten.png"
            
            printed.save(printed_path)
            handwritten.save(handwritten_path)
            
            # Update sample with new paths
            new_sample = {
                'sample_id': sample_id,
                'folder': sample_id,
                'printed_path': str(printed_path),
                'handwritten_path': str(handwritten_path),
                'metadata': {
                    'dataset': 'IAM_mini',
                    'line2': int(line2),
                    'original_image': image_path,
                }
            }
            
            # Add any existing metadata
            if 'metadata' in sample:
                new_sample['metadata'].update(sample['metadata'])
            
            new_samples.append(new_sample)
            
        except Exception as e:
            logger.warning(f"Failed to process {sample_id}: {e}")
            # Still add sample with error flag
            new_sample = {
                'sample_id': sample_id,
                'folder': sample_id,
                'error': str(e),
                'metadata': sample.get('metadata', {})
            }
            new_samples.append(new_sample)
    
    # Save new index
    logger.info("Saving reorganized index...")
    new_data = {
        'dataset': 'IAM_mini',
        'version': '2.0',
        'total_samples': len(new_samples),
        'samples': new_samples
    }
    
    with open(iam_mini_dir / "iam_mini_index.json", 'w') as f:
        json.dump(new_data, f, indent=2)
    
    # Remove old JSON if it exists
    if old_json.exists() and old_json != iam_mini_dir / "iam_mini_index.json":
        old_json.unlink()
        logger.info(f"Removed old {old_json.name}")
    
    logger.info(f"\n✓ IAM_mini reorganized successfully!")
    logger.info(f"  - {len(new_samples)} samples with folders")
    logger.info(f"  - Each folder contains: printed.png, handwritten.png")
    logger.info(f"  - Location: {iam_mini_dir}")
    
    return True


if __name__ == "__main__":
    success = reorganize_iam_mini()
    exit(0 if success else 1)
