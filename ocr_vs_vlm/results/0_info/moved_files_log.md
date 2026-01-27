# Moved Files Log

This document tracks files that have been moved to the `1_raw/zzz_ignore/` archive during the results folder restructuring.

## Date: 2026-01-27

### Archive Operations

1. **Original zzz/ folder contents**
   - Source: `results/raw/zzz/`
   - Destination: `results/1_raw/zzz_ignore/`
   - Contents:
     - `DocVQA_mini/` - Test runs
     - `IAM/` - Old IAM experiments
     - `IAM_mini/` - Test runs
     - `InfographicVQA_mini/` - Test runs
     - `VOC2007/` - Test runs
     - `empty or bad/` - Failed validation files
     - `gpt-5-mini/` - Test runs
     - `gpt-5-nano/` - Test runs
     - `publaynet/` - Incomplete runs

2. **publaynet_test/**
   - Source: `results/raw/publaynet_test/`
   - Destination: `results/1_raw/zzz_ignore/publaynet_test/`
   - Reason: Only 5 samples (test run, not production)

3. **publaynet (incomplete)**
   - Source: `results/raw/publaynet/`
   - Destination: `results/1_raw/zzz_ignore/publaynet_incomplete/`
   - Reason: Incomplete model coverage, timestamped runs

## Restoration Instructions

To restore any archived file:

```bash
# Example: Restore a specific result file
mv 1_raw/zzz_ignore/[dataset]/[file] 1_raw/[dataset]/

# Update metadata
# Edit 0_info/clean_experiments.json to add the restored experiment

# Regenerate clean files
cd 2_clean
python clean_files.py --incremental
```

## Archive Size

Approximate size of archived data: ~2 GB

## Notes

- No files were deleted, only moved for organization
- All archived files can be restored if needed
- Archive contains valuable debugging and historical information
