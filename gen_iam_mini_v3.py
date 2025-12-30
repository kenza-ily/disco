#!/usr/bin/env python3
"""Generate IAM_mini dataset with 500 samples using 3-separator detection."""

import json
import random
import glob
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from ocr_vs_vlm.datasets_subsets.iam_mini.line2_detection import (
    find_separators, crop_printed_only, crop_handwritten_only, crop_original
)

# Settings
NUM_SAMPLES = 500
SEED = 42
MIN_CONFIDENCE = 0.8  # Only use images with clear separator detection
OUTPUT_DIR = Path('ocr_vs_vlm/datasets_subsets/iam_mini')

print(f'Creating IAM_mini dataset with {NUM_SAMPLES} samples...')
print(f'Output directory: {OUTPUT_DIR}')

# Get all IAM images
all_images = sorted(glob.glob('datasets/parsing/IAM/data/*/*.png'))
print(f'Found {len(all_images)} total IAM images')

# Pre-filter to images with high-confidence separator detection
print(f'Filtering to images with confidence >= {MIN_CONFIDENCE}...')
high_conf_images = []
for path in tqdm(all_images, desc='Filtering'):
    result = find_separators(path, debug=False)
    if result.confidence >= MIN_CONFIDENCE and result.method == 'density':
        high_conf_images.append(path)

print(f'Found {len(high_conf_images)} images with clear separators')

# Random selection from high-confidence images
random.seed(SEED)
selected = random.sample(high_conf_images, NUM_SAMPLES)
print(f'Selected {len(selected)} random samples (seed={SEED})')

# Process each image
samples = []
success_count = 0
fallback_count = 0

for i, path in enumerate(tqdm(selected, desc='Processing')):
    name = Path(path).stem
    folder_name = Path(path).parent.name
    sample_id = f'iam_{folder_name}_{name}'
    
    # Create folder for this sample
    sample_dir = OUTPUT_DIR / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Detect separators
        result = find_separators(path, debug=False)
        
        # Crop and save images
        original_img = crop_original(path, result.sep1, result.sep3)
        printed_img = crop_printed_only(path, result.sep2, result.sep1)
        handwritten_img = crop_handwritten_only(path, result.sep2, result.sep3)
        
        original_img.save(sample_dir / 'original.png')
        printed_img.save(sample_dir / 'printed.png')
        handwritten_img.save(sample_dir / 'handwritten.png')
        
        # Record sample info
        sample_info = {
            'sample_id': sample_id,
            'folder': sample_id,
            'original_path': str(sample_dir / 'original.png'),
            'printed_path': str(sample_dir / 'printed.png'),
            'handwritten_path': str(sample_dir / 'handwritten.png'),
            'metadata': {
                'source_image': path,
                'separator_1': result.sep1,
                'separator_2': result.sep2,
                'separator_3': result.sep3,
                'confidence': result.confidence,
                'method': result.method,
                'original_size': list(original_img.size),
                'printed_size': list(printed_img.size),
                'handwritten_size': list(handwritten_img.size),
            }
        }
        samples.append(sample_info)
        
        if result.confidence >= 0.5:
            success_count += 1
        else:
            fallback_count += 1
            
    except Exception as e:
        print(f'Error processing {sample_id}: {e}')
        samples.append({
            'sample_id': sample_id,
            'folder': sample_id,
            'error': str(e)
        })

# Save index
index_data = {
    'dataset': 'IAM_mini',
    'version': '3.0',
    'description': 'IAM handwriting dataset mini subset with 3-separator detection',
    'total_samples': len(samples),
    'successful_detections': success_count,
    'fallback_detections': fallback_count,
    'seed': SEED,
    'columns': ['original', 'printed', 'handwritten'],
    'samples': samples
}

index_path = OUTPUT_DIR / 'iam_mini_index.json'
with open(index_path, 'w') as f:
    json.dump(index_data, f, indent=2)

print(f'\n✓ IAM_mini dataset created successfully!')
print(f'  Total samples: {len(samples)}')
print(f'  Successful detections: {success_count}')
print(f'  Fallback detections: {fallback_count}')
print(f'  Index file: {index_path}')
