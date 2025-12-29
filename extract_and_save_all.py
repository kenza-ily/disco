"""
Extract and save printed/handwritten sections for all test images.
"""

import numpy as np
from PIL import Image
from pathlib import Path


def find_line2_and_extract(image_path: str, output_dir: Path, line1_end: int = 255, line3_start: int = 2796):
    """
    Find Line 2 and extract both sections, saving them to output_dir.
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"⚠️  File not found: {image_path}")
        return None
    
    # Load image
    image = Image.open(image_path).convert('L')
    img_array = np.array(image)
    
    print(f"\nProcessing: {image_path.name}")
    
    # Calculate darkness and find clusters
    row_darkness = 1 - (img_array.mean(axis=1) / 255.0)
    threshold = 0.12
    dark_rows = np.where(row_darkness > threshold)[0]
    
    dark_clusters = []
    if len(dark_rows) > 0:
        current_cluster = [dark_rows[0]]
        for row in dark_rows[1:]:
            if row - current_cluster[-1] <= 3:
                current_cluster.append(row)
            else:
                dark_clusters.append(current_cluster)
                current_cluster = [row]
        dark_clusters.append(current_cluster)
    
    # Find gaps
    gaps = []
    for i in range(len(dark_clusters) - 1):
        gap_size = dark_clusters[i+1][0] - dark_clusters[i][-1]
        gaps.append({
            'gap_size': gap_size,
            'gap_after_row': dark_clusters[i][-1],
        })
    
    # Find Line 2 (largest gap before row 1000)
    early_gaps = [g for g in gaps if g['gap_after_row'] < 1000]
    if not early_gaps:
        print("Could not find Line 2!")
        return None
    
    line2_gap = max(early_gaps, key=lambda x: x['gap_size'])
    line2_row = line2_gap['gap_after_row']
    
    print(f"  Line 2 at row: {line2_row} (gap: {line2_gap['gap_size']} rows)")
    
    # Extract sections
    margin = 5
    printed_start = line1_end + margin
    printed_end = line2_row
    handwritten_start = line2_row + margin
    handwritten_end = line3_start
    
    printed_section = img_array[printed_start:printed_end, :]
    handwritten_section = img_array[handwritten_start:handwritten_end, :]
    
    # Convert to images
    printed_img = Image.fromarray(printed_section.astype(np.uint8))
    handwritten_img = Image.fromarray(handwritten_section.astype(np.uint8))
    
    # Save with descriptive names
    base_name = image_path.stem  # filename without extension
    printed_path = output_dir / f"{base_name}_printed.png"
    handwritten_path = output_dir / f"{base_name}_handwritten.png"
    
    printed_img.save(printed_path)
    handwritten_img.save(handwritten_path)
    
    print(f"  Saved: {printed_path.name} ({printed_img.size[1]} px)")
    print(f"  Saved: {handwritten_path.name} ({handwritten_img.size[1]} px)")
    
    return {
        'image': image_path.name,
        'line2': line2_row,
        'printed_file': printed_path.name,
        'handwritten_file': handwritten_path.name,
        'printed_height': printed_img.size[1],
        'handwritten_height': handwritten_img.size[1],
    }


if __name__ == '__main__':
    output_dir = Path("/Users/kenzabenkirane/Documents/GitHub/research-playground/ocr_vs_vlm/crop_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = [
        "/Users/kenzabenkirane/Documents/GitHub/research-playground/datasets/parsing/IAM/data/622/p03-072.png",
        "/Users/kenzabenkirane/Documents/GitHub/research-playground/datasets/parsing/IAM/data/432/k03-138.png",
        "/Users/kenzabenkirane/Documents/GitHub/research-playground/datasets/parsing/IAM/data/422/k02-053.png",
    ]
    
    print("="*70)
    print("Extracting printed and handwritten sections from all test images")
    print("="*70)
    
    results = []
    for image_path in images:
        result = find_line2_and_extract(image_path, output_dir)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    for r in results:
        print(f"\n{r['image']}:")
        print(f"  Line 2: row {r['line2']}")
        print(f"  Printed: {r['printed_file']} ({r['printed_height']} px)")
        print(f"  Handwritten: {r['handwritten_file']} ({r['handwritten_height']} px)")
    
    print(f"\n✓ All extractions saved to: {output_dir}/")
