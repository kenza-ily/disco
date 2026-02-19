"""
Benchmark suite for OCR vs VLM comparison.

Modules:
- benchmark: Base BenchmarkRunner framework
- benchmark_chartqapro: Chart QA benchmark
- dataset_specific/: Dataset-specific benchmark implementations
"""

from benchmarks.benchmark import BenchmarkRunner, BenchmarkConfig

__all__ = ["BenchmarkRunner", "BenchmarkConfig"]
