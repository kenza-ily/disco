#!/usr/bin/env python3
"""
Complete Claude Sonnet Benchmark Suite
Validates, runs, and analyzes all benchmarks with Claude Sonnet

Usage:
  python run_all_claude_benchmarks.py                    # 10-sample validation
  python run_all_claude_benchmarks.py --full             # Full 500-sample run
  python run_all_claude_benchmarks.py --samples 50       # Custom sample size
  python run_all_claude_benchmarks.py --validate-only    # Validate setup only
  python run_all_claude_benchmarks.py --model claude_sonnet  # Use Claude Sonnet

Results are saved with timestamps in the filename to preserve previous runs.
Each run creates new CSV/JSON files in ocr_vs_vlm/results/<dataset>/<model>/
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys
import argparse
import re
from tqdm import tqdm


class ClaudeBenchmarkSuite:
    """Main benchmark suite runner."""
    
    # Benchmarks with their modules, display names, and phases
    # Phases map to: OCR pipeline (QA1/P-A), VLM parse (QA2/P-B), Direct VQA (QA3/P-C)
    BENCHMARKS = [
        ("publaynet", "PubLayNet", ["P-A", "P-B", "P-C"]),
        ("docvqa", "DocVQA", ["QA1a", "QA1b", "QA1c", "QA2a", "QA2b", "QA2c", "QA3a", "QA3b"]),
        ("infographicvqa", "InfographicVQA", ["QA1a", "QA1b", "QA1c", "QA2a", "QA2b", "QA2c", "QA3a", "QA3b", "QA4a", "QA4b", "QA4c"]),
        ("voc2007", "VOC2007", ["2", "3", "4"]),
        ("iammini", "IAM Mini", ["2", "3"]),
    ]
    
    # Available models
    VLM_MODELS = ["gpt-5-mini", "gpt-5-nano", "claude_sonnet"]
    OCR_MODELS = ["azure_intelligence", "mistral_document_ai"]
    
    def __init__(self, samples: Optional[int] = None, skip_logs: bool = False, 
                 vlm_models: Optional[List[str]] = None, 
                 ocr_models: Optional[List[str]] = None):
        self.samples = samples
        self.skip_logs = skip_logs
        self.vlm_models = vlm_models or ["claude_sonnet"]
        self.ocr_models = ocr_models or ["azure_intelligence", "mistral_document_ai"]
        self.results: Dict[str, Tuple[bool, str]] = {}
        self.log_dir = Path("logs/claude_benchmarks")
        
        if not self.skip_logs:
            self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_setup(self) -> bool:
        """Validate that AWS credentials and dependencies are available."""
        print("\n" + "=" * 60)
        print("🔍 VALIDATING SETUP")
        print("=" * 60)
        
        checks = [
            ("AWS Credentials", self._check_aws),
            ("Python Environment", self._check_python),
            ("Required Packages", self._check_packages),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            result = check_func()
            status = "✅" if result else "❌"
            print(f"{status} {check_name}")
            if not result:
                all_passed = False
        
        return all_passed
    
    def _check_aws(self) -> bool:
        """Check AWS credentials are configured."""
        try:
            result = subprocess.run(
                ["aws", "sts", "get-caller-identity"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_python(self) -> bool:
        """Check Python environment."""
        try:
            result = subprocess.run(
                ["python", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_packages(self) -> bool:
        """Check required packages."""
        try:
            import boto3
            import anthropic
            return True
        except ImportError:
            return False
    
    def run_single_benchmark(self, module: str, display_name: str, phases: List[str]) -> Tuple[bool, str]:
        """Run a single benchmark with configured models."""
        try:
            print(f"\n{'─' * 60}")
            print(f"📊 Running: {display_name}")
            sample_str = "all" if self.samples is None else str(self.samples)
            print(f"   Samples: {sample_str} | Phases: {', '.join(phases)}")
            print(f"   VLM Models: {', '.join(self.vlm_models)}")
            print(f"   OCR Models: {', '.join(self.ocr_models)}")
            print(f"{'─' * 60}")
            
            cmd = [
                "uv", "run", "python", "-m",
                f"ocr_vs_vlm.benchmark_{module}",
            ]
            
            # Add sample limit only if not None (None means all)
            if self.samples is not None:
                cmd.extend(["--sample-limit", str(self.samples)])
            
            # Add phases - pass them individually
            cmd.extend(["--phases"] + phases)
            
            # Add model arguments based on benchmark type
            if module in ["docvqa", "infographicvqa"]:
                # QA benchmarks have --vlm-models and --ocr-models
                cmd.extend(["--vlm-models"] + self.vlm_models)
                cmd.extend(["--ocr-models"] + self.ocr_models)
            elif module == "publaynet":
                # PubLayNet has --models for VLM and --ocr-models for OCR
                cmd.extend(["--models"] + self.vlm_models)
                cmd.extend(["--ocr-models"] + self.ocr_models)
            elif module == "voc2007":
                # VOC2007 uses --models for VLM models only
                cmd.extend(["--models"] + self.vlm_models)
            elif module == "iammini":
                # IAM mini uses --models for VLM models
                cmd.extend(["--models"] + self.vlm_models)
            
            start = time.time()
            
            # Run with progress tracking
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            result_paths = []
            csv_files = []
            
            for line in process.stdout:
                # Keep line as-is to preserve progress bar formatting
                stripped_line = line.rstrip('\n')
                
                # Show progress bars (they have % and |)
                if "%" in stripped_line and "|" in stripped_line:
                    print(f"   {stripped_line}")
                # Show key info lines
                elif any(x in stripped_line for x in ["[INFO]", "[WARNING]", "[ERROR]", "Processing", "Checkpoint", "ANLS", "results_file", "Saved", "Results saved"]):
                    print(f"   {stripped_line}")
                
                # Capture result file paths - CSV files (both absolute and relative)
                if ".csv" in stripped_line and "results" in stripped_line:
                    # Try absolute path first
                    match = re.search(r'(/[^\s"\']*ocr_vs_vlm/results/[^\s"\']+\.csv)', stripped_line)
                    if not match:
                        # Try relative path
                        match = re.search(r'(ocr_vs_vlm/results/[^\s"\']+\.csv)', stripped_line)
                    
                    if match:
                        full_path = match.group(1)
                        # Convert to relative path if it's absolute
                        try:
                            if full_path.startswith('/'):
                                relative_path = Path(full_path).relative_to(Path.cwd())
                                path_str = str(relative_path)
                            else:
                                path_str = full_path
                        except (ValueError, TypeError):
                            path_str = full_path
                        
                        if path_str not in csv_files:
                            csv_files.append(path_str)
            
            process.wait()
            elapsed = time.time() - start
            
            if process.returncode == 0:
                print(f"\n✅ {display_name} completed in {int(elapsed // 60)}m {int(elapsed % 60)}s")
                
                # Display CSV files found
                if csv_files:
                    print(f"\n   📊 CSV Result Files ({len(csv_files)}):")
                    for csv_path in sorted(set(csv_files)):
                        print(f"      ✓ {csv_path}")
                
                # Also find any results directory for this benchmark
                self._find_and_display_results(module, display_name)
                
                return True, f"Completed in {int(elapsed // 60)}m {int(elapsed % 60)}s"
            else:
                print(f"❌ {display_name} failed")
                return False, f"Failed after {int(elapsed // 60)}m {int(elapsed % 60)}s"
        
        except subprocess.TimeoutExpired:
            print(f"⏱️  {display_name} timeout (1 hour exceeded)")
            return False, "Timeout"
        except Exception as e:
            print(f"❌ {display_name} error: {e}")
            return False, str(e)
    
    def _find_and_display_results(self, module: str, display_name: str):
        """Find and display all result files for a benchmark."""
        results_base = Path("ocr_vs_vlm/results")
        
        if not results_base.exists():
            return
        
        # Map module to directory patterns
        patterns = {
            "publaynet": ["publaynet*"],
            "docvqa": ["DocVQA*"],
            "infographicvqa": ["InfographicVQA*"],
            "voc2007": ["VOC2007*"],
            "iammini": ["IAM*"],
        }
        
        search_patterns = patterns.get(module, [f"{display_name}*"])
        
        all_csv_files = []
        for pattern in search_patterns:
            try:
                matching_dirs = sorted(results_base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
                if matching_dirs:
                    latest_dir = matching_dirs[0]
                    csv_files = sorted(latest_dir.rglob("*.csv"))
                    all_csv_files.extend(csv_files)
            except Exception as e:
                continue
        
        # Display unique CSV files with safe path handling
        if all_csv_files:
            unique_files = sorted(set(all_csv_files))
            if not csv_files:  # Only show if not already displayed above
                print(f"\n   📁 Results Directory: ocr_vs_vlm/results")
            for csv_file in unique_files:
                try:
                    relative = csv_file.relative_to(Path.cwd())
                    print(f"      📄 {relative}")
                except (ValueError, TypeError):
                    # Fallback: just print the file name
                    print(f"      📄 {csv_file.name}")
    
    def run_all(self) -> int:
        """Run all benchmarks."""
        print("\n" + "=" * 60)
        print("🚀 CLAUDE SONNET BENCHMARK SUITE")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Samples: {self.samples if self.samples else 'All'}")
        print(f"  VLM Models: {', '.join(self.vlm_models)}")
        print(f"  OCR Models: {', '.join(self.ocr_models)}")
        print(f"  Benchmarks: {len(self.BENCHMARKS)}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Validate setup
        if not self.validate_setup():
            print("\n❌ Setup validation failed!")
            print("Please ensure:")
            print("  • AWS credentials are configured: aws configure")
            print("  • Required packages installed: pip install boto3 anthropic")
            return 1
        
        # Run benchmarks
        start_time = time.time()
        
        for module, display_name, phases in self.BENCHMARKS:
            success, info = self.run_single_benchmark(module, display_name, phases)
            self.results[module] = (success, info)
        
        elapsed = time.time() - start_time
        
        # Print summary
        self._print_summary(elapsed)
        
        # Return success if all completed
        success_count = sum(1 for success, _ in self.results.values() if success)
        return 0 if success_count == len(self.results) else 1
    
    def _print_summary(self, elapsed: float):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("📋 BENCHMARK SUMMARY")
        print("=" * 60 + "\n")
        
        success_count = 0
        for module, display_name, _ in self.BENCHMARKS:
            if module not in self.results:
                continue
            
            success, info = self.results[module]
            if success:
                success_count += 1
                status = "✅ SUCCESS"
            else:
                status = "❌ FAILED"
            
            print(f"  {status:12} - {display_name:20} ({info})")
        
        total = len(self.results)
        print(f"\n{'─' * 60}")
        print(f"Results: {success_count}/{total} benchmarks successful")
        print(f"Total Time: {int(elapsed // 60)}m {int(elapsed % 60)}s")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results: ocr_vs_vlm/results/")
        
        if not self.skip_logs:
            print(f"Logs: {self.log_dir}")
        
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Claude Sonnet Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # 10-sample validation with Claude
  %(prog)s --full                       # Full 500-sample run
  %(prog)s --samples 5                  # 5-sample quick test
  %(prog)s --validate-only              # Validate setup only
  %(prog)s --vlm-models claude_sonnet   # Use only Claude Sonnet
  %(prog)s --vlm-models gpt-5-mini gpt-5-nano claude_sonnet  # All VLM models
        """
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples per benchmark (default: 10)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark (all samples)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate setup, don't run benchmarks"
    )
    parser.add_argument(
        "--skip-logs",
        action="store_true",
        help="Don't save logs to file"
    )
    parser.add_argument(
        "--vlm-models",
        nargs='+',
        default=["claude_sonnet"],
        help="VLM models to use (default: claude_sonnet). Options: gpt-5-mini, gpt-5-nano, claude_sonnet"
    )
    parser.add_argument(
        "--ocr-models",
        nargs='+',
        default=["azure_intelligence", "mistral_document_ai"],
        help="OCR models to use (default: azure_intelligence, mistral_document_ai)"
    )
    
    args = parser.parse_args()
    
    # Determine sample count
    if args.full:
        samples = None  # None means all samples
    else:
        samples = args.samples
    
    # Create and run suite
    suite = ClaudeBenchmarkSuite(
        samples=samples, 
        skip_logs=args.skip_logs,
        vlm_models=args.vlm_models,
        ocr_models=args.ocr_models
    )
    
    if args.validate_only:
        return 0 if suite.validate_setup() else 1
    
    return suite.run_all()


if __name__ == "__main__":
    sys.exit(main())
