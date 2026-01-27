#!/usr/bin/env python3
"""Download only missing VisR-Bench mini images."""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import concurrent.futures

mini_dir = Path("/Users/kenzabenkirane/Documents/GitHub/research-playground/ocr_vs_vlm/datasets_subsets/visr_bench_mini")
cache_base = Path.home() / ".cache/huggingface/hub/datasets--puar-playground--VisR-Bench/snapshots"

snapshot_dir = list(cache_base.glob("*"))[0]
print(f"Using cache: {snapshot_dir}\n")

# Find missing images
missing_images = []

for ctype in ["figure", "table", "text", "multilingual"]:
    qa_file = mini_dir / f"{ctype}_QA_mini.json"
    if not qa_file.exists():
        continue
    
    with open(qa_file) as f:
        docs = json.load(f)
    
    hf_folder = "Multimodal" if ctype != "multilingual" else "Multilingual"
    
    for doc in docs:
        doc_id = doc["file_name"]
        for img_name in doc.get("all_page_images", []):
            cache_path = snapshot_dir / hf_folder / doc_id / img_name
            if not cache_path.exists():
                missing_images.append((doc_id, img_name, hf_folder))

print(f"📥 Downloading {len(missing_images)} missing images...\n")

def download_image(doc_id, img_name, hf_folder):
    """Download a single missing image."""
    try:
        filepath = hf_hub_download(
            repo_id="puar-playground/VisR-Bench",
            repo_type="dataset",
            filename=f"{hf_folder}/{doc_id}/{img_name}",
            cache_dir=str(cache_base.parent),
            force_download=False
        )
        return True
    except Exception as e:
        return False

downloaded = 0
errors = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(download_image, doc_id, img_name, hf_folder)
        for doc_id, img_name, hf_folder in missing_images
    ]
    
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        try:
            if future.result():
                downloaded += 1
            else:
                errors += 1
        except:
            errors += 1
        
        if (i + 1) % 50 == 0 or (i + 1) == len(futures):
            print(f"  ✓ {i + 1}/{len(missing_images)} | Downloaded: {downloaded} | Errors: {errors}")

print(f"\n✅ Complete!")
print(f"   Downloaded: {downloaded} images")
print(f"   Errors: {errors}")
