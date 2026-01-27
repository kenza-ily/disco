#!/usr/bin/env python3
"""Check and verify image availability for benchmark."""
import json
from pathlib import Path
import subprocess

mini_dir = Path("/Users/kenzabenkirane/Documents/GitHub/research-playground/ocr_vs_vlm/datasets_subsets/visr_bench_mini")
hf_repo = Path.home() / "Documents" / "GitHub" / "VisR-Bench-Repo"

print("🔍 Checking image availability...\n")

# Check HF repo
print(f"HuggingFace repo: {hf_repo}")
if hf_repo.exists():
    print(f"  ✓ Repo exists")
    # Check Multimodal
    multimodal_path = hf_repo / "Multimodal" / "0159"
    if multimodal_path.exists():
        pngs = list(multimodal_path.glob("*.png"))
        if pngs:
            print(f"  ✓ Multimodal images found: {len(pngs)} PNGs in doc 0159")
            print(f"    First few: {[p.name for p in pngs[:3]]}")
        else:
            print(f"  ❌ Multimodal 0159 exists but NO PNG files")
            # List what's in the directory
            contents = list(multimodal_path.iterdir())
            if contents:
                print(f"    Contents: {[c.name for c in contents[:5]]}")
            else:
                print(f"    Directory is EMPTY")
    else:
        print(f"  ❌ Multimodal/0159 directory not found")

# Check symlink
symlink_path = mini_dir / "documents" / "text" / "0159" / "images"
print(f"\nLocal symlink: {symlink_path}")
if symlink_path.is_symlink():
    target = symlink_path.resolve()
    print(f"  ✓ Symlink exists → {target}")
    pngs = list(symlink_path.glob("*.png"))
    if pngs:
        print(f"  ✓ Can access {len(pngs)} images through symlink")
    else:
        print(f"  ❌ Symlink target exists but has NO PNG files")
else:
    print(f"  ❌ Not a symlink or doesn't exist")

# Try git lfs status
print("\n📦 Git LFS status:")
try:
    result = subprocess.run(
        ["git", "lfs", "status"],
        cwd=hf_repo,
        capture_output=True,
        timeout=10,
        text=True
    )
    if "On branch" in result.stdout or not result.stdout.strip():
        print("  ✓ No pending LFS files (all downloaded or tracking)")
    else:
        # Show first few lines
        lines = result.stdout.strip().split('\n')[:5]
        print("  " + "\n  ".join(lines))
except Exception as e:
    print(f"  ⚠️  Could not check: {e}")
