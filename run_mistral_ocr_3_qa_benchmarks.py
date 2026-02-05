#!/usr/bin/env python3
"""
Run Mistral OCR 3 on all QA benchmarks (DocVQA, InfographicVQA, DUDE).

Usage:
    uv run python run_mistral_ocr_3_qa_benchmarks.py --test  # 10 samples each
    uv run python run_mistral_ocr_3_qa_benchmarks.py         # Full run
"""

import subprocess
import sys
from datetime import datetime


def run_benchmark(dataset: str, test_mode: bool = False) -> bool:
    """Run a single benchmark."""
    module_map = {
        "DocVQA": "ocr_vs_vlm.benchmarks.dataset_specific.benchmark_docvqa",
        "InfographicVQA": "ocr_vs_vlm.benchmarks.dataset_specific.benchmark_infographicvqa",
        "DUDE": "ocr_vs_vlm.benchmarks.dataset_specific.benchmark_dude",
    }

    cmd = [
        "uv", "run", "python", "-m", module_map[dataset],
        "--phases", "QA1a", "QA1b", "QA1c",
        "--ocr-models", "mistral_ocr_3"
    ]

    if test_mode:
        cmd.extend(["--sample-limit", "10"])

    print(f"\n{'='*70}")
    print(f"Running {dataset} {'TEST' if test_mode else 'FULL'} - {datetime.now()}")
    print(f"{'='*70}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"ERROR: {dataset} failed with code {result.returncode}")
        return False

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test mode (10 samples)")
    args = parser.parse_args()

    datasets = ["DocVQA", "InfographicVQA", "DUDE"]

    start_time = datetime.now()
    print(f"Starting Mistral OCR 3 QA Benchmarks - {start_time}")

    results = {}
    for dataset in datasets:
        success = run_benchmark(dataset, test_mode=args.test)
        results[dataset] = "SUCCESS" if success else "FAILED"

    end_time = datetime.now()
    elapsed = end_time - start_time

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for dataset, status in results.items():
        print(f"{dataset}: {status}")
    print(f"\nTotal time: {elapsed}")


if __name__ == "__main__":
    main()
