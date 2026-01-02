#!/usr/bin/env python3
"""Generate DocVQA_mini dataset with 500 random samples."""

import json
import random
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import sys

# Add the ocr_vs_vlm module to path
sys.path.insert(0, str(Path(__file__).parent / 'ocr_vs_vlm'))

from dataset_loaders import DatasetRegistry

# Settings
NUM_SAMPLES = 500
SEED = 42
OUTPUT_DIR = Path('ocr_vs_vlm/datasets_subsets/docvqa_mini')

print(f'Creating DocVQA_mini dataset with {NUM_SAMPLES} samples...')
print(f'Output directory: {OUTPUT_DIR}')

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load full DocVQA validation set in QA mode
docvqa_root = 'datasets/QA/DocVQA_hf'
print(f'Loading DocVQA from: {docvqa_root}')

try:
    docvqa = DatasetRegistry.get_dataset('DocVQA', docvqa_root, mode='qa', split='validation')
    print(f'Loaded {len(docvqa)} QA samples from DocVQA validation set')
except Exception as e:
    print(f'Error loading DocVQA: {e}')
    sys.exit(1)

# Random selection
random.seed(SEED)
if len(docvqa.samples) > NUM_SAMPLES:
    selected_samples = random.sample(docvqa.samples, NUM_SAMPLES)
else:
    selected_samples = docvqa.samples

print(f'Selected {len(selected_samples)} random samples (seed={SEED})')

# Create images directory
images_dir = OUTPUT_DIR / 'images'
images_dir.mkdir(exist_ok=True)

# Process samples
mini_samples = []
for i, sample in enumerate(tqdm(selected_samples, desc='Processing samples')):
    # Copy image to mini dataset
    original_image_path = Path(sample.image_path)
    if original_image_path.exists():
        # Create new filename
        new_filename = f'docvqa_{i:04d}.png'
        new_image_path = images_dir / new_filename

        # Copy the image
        try:
            with Image.open(original_image_path) as img:
                img.save(new_image_path)
        except Exception as e:
            print(f'Error copying image {original_image_path}: {e}')
            continue

        # Create mini sample record
        mini_sample = {
            'sample_id': f'docvqa_mini_{i:04d}',
            'image_path': str(new_image_path.relative_to(OUTPUT_DIR)),
            'ground_truth': sample.ground_truth,
            'question': sample.question,
            'answers': sample.answers,
            'question_type': sample.question_type,
            'metadata': {
                **sample.metadata,
                'original_sample_id': sample.sample_id,
                'mini_dataset': 'DocVQA_mini',
            }
        }
        mini_samples.append(mini_sample)
    else:
        print(f'Warning: Image not found: {original_image_path}')

# Save index
index_data = {
    'dataset': 'DocVQA_mini',
    'version': '1.0',
    'description': 'DocVQA mini subset with 500 random QA samples from validation set',
    'total_samples': len(mini_samples),
    'seed': SEED,
    'source_split': 'validation',
    'source_mode': 'qa',
    'samples': mini_samples
}

index_path = OUTPUT_DIR / 'docvqa_mini_index.json'
with open(index_path, 'w') as f:
    json.dump(index_data, f, indent=2)

print(f'\n✓ DocVQA_mini dataset created successfully!')
print(f'  Total samples: {len(mini_samples)}')
print(f'  Images directory: {images_dir}')
print(f'  Index file: {index_path}')