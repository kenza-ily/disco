#!/usr/bin/env python3
"""
Unified benchmark runner for all datasets.

This is a convenience wrapper around individual benchmark scripts.
You can still run benchmarks directly:
    python -m benchmarks.dataset_specific.benchmark_docvqa --phases QA1a --sample-limit 50

Usage:
    uv run python scripts/run_benchmark.py --dataset docvqa --phases QA1a QA3a --sample-limit 50
    uv run python scripts/run_benchmark.py --dataset publaynet --phases P-A P-B --models gpt-5-mini
    uv run python scripts/run_benchmark.py --dataset iam --phases 1 2 --models claude_sonnet
"""

import sys
import subprocess
from pathlib import Path

# Map dataset names to benchmark modules
BENCHMARK_MODULES = {
    "chartqapro": "benchmarks.benchmark_chartqapro",
    "docvqa": "benchmarks.dataset_specific.benchmark_docvqa",
    "infographicvqa": "benchmarks.dataset_specific.benchmark_infographicvqa",
    "dude": "benchmarks.dataset_specific.benchmark_dude",
    "visrbench": "benchmarks.dataset_specific.benchmark_visrbench",
    "publaynet": "benchmarks.dataset_specific.benchmark_publaynet",
    "iam": "benchmarks.dataset_specific.benchmark_iammini",
    "rxpad": "benchmarks.dataset_specific.benchmark_rxpad",
    "voc2007": "benchmarks.dataset_specific.benchmark_voc2007",
}

def main():
    if len(sys.argv) < 3 or sys.argv[1] != "--dataset":
        print("Usage: python scripts/run_benchmark.py --dataset <name> [benchmark args...]")
        print(f"Available datasets: {', '.join(BENCHMARK_MODULES.keys())}")
        sys.exit(1)

    dataset = sys.argv[2]
    if dataset not in BENCHMARK_MODULES:
        print(f"Error: Unknown dataset '{dataset}'")
        print(f"Available: {', '.join(BENCHMARK_MODULES.keys())}")
        sys.exit(1)

    # Pass remaining args to the benchmark module
    benchmark_args = sys.argv[3:]
    module = BENCHMARK_MODULES[dataset]

    # Run benchmark as subprocess
    cmd = ["python", "-m", module] + benchmark_args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
