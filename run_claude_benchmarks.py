#!/usr/bin/env python3
"""
Claude Sonnet Benchmark Suite Runner
Runs all benchmarks with Claude Sonnet across all datasets
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Configuration
BENCHMARKS = [
    ("publaynet", "PubLayNet"),
    ("docvqa", "DocVQA"),
    ("infographicvqa", "InfographicVQA"),
    ("voc2007", "VOC2007"),
    ("iammini", "IAM Mini"),
]

PHASES = ["P-B"]  # VLM direct phase
DEFAULT_SAMPLES = 10


def run_benchmark(module: str, display_name: str, samples: int, phases: List[str]) -> Tuple[bool, str]:
    """Run a single benchmark and return success status and results path."""
    try:
        print(f"\n{'─' * 60}")
        print(f"📊 Running: {display_name} ({module})")
        print(f"{'─' * 60}")
        
        cmd = [
            "uv", "run", "python", "-m",
            f"ocr_vs_vlm.benchmark_{module}",
            "--sample-limit", str(samples),
            "--models", "claude_sonnet",
            "--phases", " ".join(phases),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per benchmark
        )
        
        # Check for success in output
        if result.returncode == 0 and "completed" in result.stdout.lower():
            print(f"✅ {display_name} completed successfully")
            
            # Try to extract results file path from stdout
            for line in result.stdout.split('\n'):
                if 'results_file' in line or 'results/' in line:
                    return True, line.strip()
            
            return True, ""
        else:
            print(f"❌ {display_name} failed")
            print(result.stderr[-500:] if result.stderr else "No error output")
            return False, ""
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  {display_name} timeout (1 hour exceeded)")
        return False, ""
    except Exception as e:
        print(f"❌ {display_name} error: {e}")
        return False, ""


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Claude Sonnet benchmarks")
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Number of samples per benchmark (default: {DEFAULT_SAMPLES})"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=PHASES,
        help="Phases to run (default: P-B)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=[b[0] for b in BENCHMARKS],
        help="Specific benchmarks to run"
    )
    parser.add_argument(
        "--skip-logs",
        action="store_true",
        help="Don't save logs to file"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 60)
    print("🚀 CLAUDE SONNET BENCHMARK SUITE")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Samples: {args.samples}")
    print(f"  Phases: {', '.join(args.phases)}")
    print(f"  Benchmarks: {len(args.benchmarks)}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Setup logging
    log_dir = Path("logs/claude_benchmarks")
    if not args.skip_logs:
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    results: Dict[str, Tuple[bool, str]] = {}
    start_time = time.time()
    
    for module, display_name in BENCHMARKS:
        if module not in args.benchmarks:
            continue
        
        success, results_path = run_benchmark(module, display_name, args.samples, args.phases)
        results[module] = (success, results_path)
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 BENCHMARK SUMMARY")
    print("=" * 60 + "\n")
    
    success_count = sum(1 for success, _ in results.values() if success)
    total_count = len(results)
    
    for module, display_name in BENCHMARKS:
        if module not in results:
            continue
        
        success, results_path = results[module]
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status:12} - {display_name}")
        if results_path:
            print(f"                  {results_path}")
    
    print(f"\n{'─' * 60}")
    print(f"Results: {success_count}/{total_count} benchmarks successful")
    print(f"Total Time: {int(elapsed // 60)}m {int(elapsed % 60)}s")
    print(f"Results saved to: ocr_vs_vlm/results/")
    
    if not args.skip_logs:
        print(f"Logs saved to: {log_dir}")
    
    print("=" * 60 + "\n")
    
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    exit(main())
