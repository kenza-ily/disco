#!/usr/bin/env python3
"""
Run QA2a benchmark for gpt-5-nano on the 64 missing samples only.

The QA2a gpt-5-nano run was incomplete (only 430/494 samples).
This script runs only the missing sample IDs 430-493.

Usage:
    uv run python ocr_vs_vlm/results/1_raw/run_qa2a_nano_missing.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.benchmark_chartqapro import (
    ChartQAProBenchmark,
    ChartQAProbenchmarkConfig
)

def main():
    """Run QA2a benchmark for missing gpt-5-nano samples."""

    print("="*70)
    print("QA2a gpt-5-nano - Missing Samples Only")
    print("="*70)
    print()
    print("Target: 64 missing samples (IDs 430-493)")
    print("Phase: QA2a (VLM extraction → VLM QA, simple prompts)")
    print("Model: gpt-5-nano")
    print()

    # Load missing sample IDs
    missing_ids_file = Path('zzz_ignore/chartqapro_deduplication/missing_sample_ids/QA2a_gpt-5-nano_incomplete_run_missing.txt')

    if not missing_ids_file.exists():
        print(f"ERROR: Missing sample IDs file not found: {missing_ids_file}")
        print("Run the deduplication analysis first to generate this file.")
        sys.exit(1)

    with open(missing_ids_file) as f:
        missing_sample_ids = set(line.strip() for line in f if line.strip())

    print(f"Loaded {len(missing_sample_ids)} missing sample IDs from: {missing_ids_file.name}")
    print()

    # Configure benchmark for QA2a with gpt-5-nano
    config = ChartQAProbenchmarkConfig(
        phases=["QA2a"],
        vlm_models=["gpt-5-nano"],
        sample_limit=None,  # Load all samples, we'll filter in custom logic
        ocr_models=["azure_intelligence"],  # Not used in QA2a but required
    )

    # Create benchmark instance
    benchmark = ChartQAProBenchmark(config)

    # Monkey-patch the _load_dataset method to filter by missing sample IDs
    original_load_dataset = benchmark._load_dataset

    def filtered_load_dataset():
        """Load only the missing samples."""
        all_samples = original_load_dataset()
        filtered = [s for s in all_samples if s.sample_id in missing_sample_ids]

        print(f"Filtered dataset: {len(filtered)}/{len(all_samples)} samples")
        print(f"Expected: {len(missing_sample_ids)} samples")

        if len(filtered) != len(missing_sample_ids):
            print(f"WARNING: Mismatch between expected and filtered samples!")

        return filtered

    benchmark._load_dataset = filtered_load_dataset

    # Run benchmark
    print("\nStarting benchmark...")
    print("-"*70)
    summary = benchmark.run()

    print("\n" + "="*70)
    print("QA2a gpt-5-nano MISSING SAMPLES COMPLETED")
    print("="*70)
    print(f"\nResults saved to: {config.results_dir}")
    print()
    print("Next steps:")
    print("1. Verify results were appended correctly")
    print("2. Check if file now has 494 rows (430 + 64)")
    print("3. Run deduplication if needed (though shouldn't be)")
    print("4. Re-generate missing predictions report")

if __name__ == '__main__':
    main()
