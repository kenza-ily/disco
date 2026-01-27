# File Cleanup Recommendations

## Files Analysis

### ✅ KEEP - Potentially Useful

#### 1. **check_images.py** - KEEP
**Purpose**: Diagnostic tool to verify image availability
**Use case**: Troubleshoot image symlinks and git-lfs status
**Recommendation**: Keep for debugging image issues

#### 2. **download_missing_images.py** - KEEP
**Purpose**: Downloads only missing images from HuggingFace
**Use case**: Recover specific missing images without re-downloading everything
**Recommendation**: Keep as utility for incremental downloads

#### 3. **upload_parquet_to_hf.py** - ARCHIVE (duplicate)
**Purpose**: Upload Parquet to HuggingFace
**Issue**: We already have `ocr_vs_vlm/datasets_subsets/visr_bench_mini/upload_to_hf.py`
**Recommendation**: Archive - better version exists in dataset directory

#### 4. **upload_to_hf.py** - ARCHIVE (duplicate)
**Purpose**: Upload dataset with images to HuggingFace
**Issue**: Duplicate of upload_parquet_to_hf.py
**Recommendation**: Archive - superseded by dataset-local script

### ❌ ARCHIVE - Not Needed

#### 5. **check_chartqapro_status.py** - ARCHIVE
**Purpose**: Check ChartQAPro benchmark results
**Issue**: Unrelated to VisR-Bench, specific to different dataset
**Recommendation**: Archive unless actively working on ChartQAPro

#### 6. **download_mini_images.py** - ARCHIVE
**Purpose**: One-time bulk download of mini images
**Issue**: Setup already complete, not needed for daily use
**Recommendation**: Archive - can recover from zzz_ignore if needed

#### 7. **move_images_to_mini.py** - ARCHIVE
**Purpose**: Move images from cache to dataset directory
**Issue**: One-time reorganization script
**Recommendation**: Archive - setup complete

## Recommended Actions

### Keep These Files (2 files)
```bash
# Diagnostic utilities
check_images.py              # Image troubleshooting
download_missing_images.py   # Incremental image downloads
```

### Archive These Files (5 files)
```bash
mv check_chartqapro_status.py zzz_ignore/visr_bench_old/scripts/
mv download_mini_images.py zzz_ignore/visr_bench_old/scripts/
mv move_images_to_mini.py zzz_ignore/visr_bench_old/scripts/
mv upload_parquet_to_hf.py zzz_ignore/visr_bench_old/scripts/
mv upload_to_hf.py zzz_ignore/visr_bench_old/scripts/
```

### Final Root Directory

After cleanup, you should have:
```
research-playground/
├── check_images.py              # Utility: check image availability
├── download_missing_images.py   # Utility: download missing images
├── ocr_vs_vlm/
│   └── datasets_subsets/
│       └── visr_bench_mini/     # Main dataset location
├── results/                     # Benchmark results
└── zzz_ignore/                  # Archived files
```

## Summary

- **Keep**: 2 utility scripts for image management
- **Archive**: 5 scripts (setup/duplicates)
- **Already archived**: 10 files from previous cleanup
