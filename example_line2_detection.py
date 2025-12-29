#!/usr/bin/env python3
"""
Example: Using IAM Dataset with Automatic Handwritten-Only Cropping

Demonstrates how to load IAM images with automatic separation of printed
reference text from handwritten content for fair VLM evaluation.

The IAM dataset has a known fairness issue: each image contains BOTH
- Printed reference text (the ground truth)
- Handwritten text (what the human wrote)

VLMs could "cheat" by reading the printed text. This example shows how to
automatically extract ONLY the handwritten portion using Line 2 detection.
"""

from pathlib import Path
from ocr_vs_vlm.dataset_loaders import IAMMiniDataset
from ocr_vs_vlm.line2_detection import find_line2, crop_handwritten_only

def example_load_dataset():
    """Load IAM dataset with automatic cropping."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Load Dataset with Handwritten-Only Cropping")
    print("="*70)
    
    # Load without cropping (default)
    print("\n1. Loading IAM_mini WITHOUT cropping:")
    dataset_full = IAMMiniDataset(
        dataset_root=Path("datasets"),
        crop_handwritten_only=False
    )
    print(f"   Loaded {len(dataset_full)} samples")
    print(f"   Sample[0]: {dataset_full[0].image_path}")
    
    # Load with cropping
    print("\n2. Loading IAM_mini WITH handwritten-only cropping:")
    dataset_hw_only = IAMMiniDataset(
        dataset_root=Path("datasets"),
        crop_handwritten_only=True
    )
    print(f"   Loaded {len(dataset_hw_only)} samples")
    sample = dataset_hw_only[0]
    print(f"   Sample[0] metadata: {sample.metadata}")
    if 'line2' in sample.metadata:
        print(f"   Line 2 detected at row: {sample.metadata['line2']}")
        print(f"   Crop valid: {sample.metadata.get('crop_valid', 'unknown')}")


def example_manual_detection():
    """Manually detect and crop a single image."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Manual Line 2 Detection and Cropping")
    print("="*70)
    
    test_image = "datasets/parsing/IAM/data/622/p03-072.png"
    
    if not Path(test_image).exists():
        print(f"⚠ Test image not found: {test_image}")
        return
    
    print(f"\nProcessing: {test_image}")
    
    # Detect Line 2
    line2 = find_line2(test_image)
    print(f"  Line 2 detected at row: {line2}")
    
    # Extract handwritten portion
    hw_img = crop_handwritten_only(test_image, line2)
    print(f"  Handwritten portion size: {hw_img.size}")
    print(f"  (Original height: 3542px, handwritten height: {hw_img.height}px)")
    
    # Save example
    output_path = Path("ocr_vs_vlm/crop_tests/example_handwritten.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hw_img.save(output_path)
    print(f"  Saved to: {output_path}")


def example_batch_processing():
    """Process multiple images and compare sizes."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Processing and Statistics")
    print("="*70)
    
    test_images = [
        "datasets/parsing/IAM/data/622/p03-072.png",
        "datasets/parsing/IAM/data/432/k03-138.png",
        "datasets/parsing/IAM/data/422/k02-053.png",
    ]
    
    print("\nImage Analysis (Line 2 Detection):")
    print("="*70)
    print(f"{'Image':<20} {'Line2':>8} {'Printed':>12} {'Handwritten':>12}")
    print(f"{'':<20} {'(row)':>8} {'(px height)':>12} {'(px height)':>12}")
    print("-"*70)
    
    from PIL import Image as PILImage
    from ocr_vs_vlm.line2_detection import DEFAULT_LINE1_END, DEFAULT_LINE3_START
    
    total_printed = 0
    total_handwritten = 0
    count = 0
    
    for img_path in test_images:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"⚠ Not found: {img_path}")
            continue
        
        line2 = find_line2(str(img_path))
        printed_height = line2 - DEFAULT_LINE1_END
        handwritten_height = DEFAULT_LINE3_START - line2
        
        print(f"{img_path.name:<20} {line2:>8d} {printed_height:>12d} {handwritten_height:>12d}")
        
        total_printed += printed_height
        total_handwritten += handwritten_height
        count += 1
    
    if count > 0:
        print("-"*70)
        avg_printed = total_printed / count
        avg_handwritten = total_handwritten / count
        print(f"{'AVERAGE':<20} {'':<8} {avg_printed:>12.0f} {avg_handwritten:>12.0f}")
        print("="*70)


if __name__ == "__main__":
    print("\n" + "🔍 IAM Dataset Line 2 Detection Examples".center(70))
    print("="*70)
    
    try:
        example_load_dataset()
    except Exception as e:
        print(f"⚠ Example 1 failed: {e}")
    
    try:
        example_manual_detection()
    except Exception as e:
        print(f"⚠ Example 2 failed: {e}")
    
    try:
        example_batch_processing()
    except Exception as e:
        print(f"⚠ Example 3 failed: {e}")
    
    print("\n" + "="*70)
    print("✓ Examples complete!")
    print("="*70)
