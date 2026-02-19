#!/usr/bin/env python3
"""
Script to run missing benchmark experiments identified in MISSING_DATA_REPORT.md

This script orchestrates existing benchmark modules via subprocess to fill gaps in the
experimental results. It ensures all results are saved with unique timestamps to prevent
overwrites.

Usage:
    python benchmark_missing_data.py                    # Run all missing experiments
    python benchmark_missing_data.py --dry-run          # Show commands without executing
    python benchmark_missing_data.py --datasets publaynet_full  # Run specific dataset
    python benchmark_missing_data.py --verbose          # Show detailed output
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import time


@dataclass
class Experiment:
    """Represents a single benchmark experiment to run."""
    dataset: str              # e.g., "InfographicVQA_mini", "publaynet_full"
    phase: str                # e.g., "QA1b", "P-B"
    models: List[str]         # e.g., ["gpt-5-mini", "claude_sonnet"]
    benchmark_module: str     # e.g., "benchmark_infographicvqa"
    sample_count: int = 500   # Production standard

    def __str__(self) -> str:
        models_str = ", ".join(self.models)
        return f"{self.dataset}/{self.phase} ({models_str})"


@dataclass
class ExperimentResult:
    """Results from running a single experiment."""
    experiment: Experiment
    success: bool
    duration_seconds: float
    output_files: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    exit_code: int = 0

    def to_dict(self):
        """Convert to dict for JSON serialization."""
        return {
            "dataset": self.experiment.dataset,
            "phase": self.experiment.phase,
            "models": self.experiment.models,
            "sample_count": self.experiment.sample_count,
            "success": self.success,
            "duration_seconds": round(self.duration_seconds, 2),
            "output_files": self.output_files,
            "error_message": self.error_message,
            "exit_code": self.exit_code,
        }


@dataclass
class ExecutionSummary:
    """Summary of all experiments run."""
    start_time: datetime
    end_time: Optional[datetime] = None
    experiments: List[ExperimentResult] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for exp in self.experiments if exp.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for exp in self.experiments if not exp.success)

    @property
    def total_duration(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self):
        """Convert to dict for JSON serialization."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": round(self.total_duration, 2),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_experiments": len(self.experiments),
            "experiments": [exp.to_dict() for exp in self.experiments],
        }


class MissingDataBenchmarkRunner:
    """Orchestrates running missing benchmark experiments."""

    def __init__(self, verbose: bool = False, timeout: int = 3600):
        self.verbose = verbose
        self.timeout = timeout
        # Script is now in ocr_vs_vlm/benchmarks/, so go up 2 levels to project root
        self.project_root = Path(__file__).parent.parent.parent.resolve()
        self.results_dir = self.project_root / "ocr_vs_vlm" / "results" / "1_raw"
        self.valid_files_path = self.project_root / "ocr_vs_vlm" / "results" / "2_clean" / "valid_files.json"

    def get_missing_experiments(self) -> List[Experiment]:
        """Returns list of all missing experiments to run."""
        experiments = []

        # InfographicVQA_mini - QA1b (VLM generic)
        experiments.append(Experiment(
            dataset="InfographicVQA_mini",
            phase="QA1b",
            models=["gpt-5-mini", "gpt-5-nano", "claude_sonnet"],
            benchmark_module="benchmark_infographicvqa",
            sample_count=500
        ))

        # InfographicVQA_mini - QA1c (VLM task-aware)
        experiments.append(Experiment(
            dataset="InfographicVQA_mini",
            phase="QA1c",
            models=["gpt-5-mini", "gpt-5-nano", "claude_sonnet"],
            benchmark_module="benchmark_infographicvqa",
            sample_count=500
        ))

        # publaynet_full - P-B (VLM generic)
        experiments.append(Experiment(
            dataset="publaynet_full",
            phase="P-B",
            models=["claude_sonnet"],
            benchmark_module="benchmark_publaynet",
            sample_count=500
        ))

        # publaynet_full - P-C (VLM task-aware)
        experiments.append(Experiment(
            dataset="publaynet_full",
            phase="P-C",
            models=["claude_sonnet"],
            benchmark_module="benchmark_publaynet",
            sample_count=500
        ))

        return experiments

    def build_command(self, experiment: Experiment) -> List[str]:
        """Build subprocess command for an experiment."""
        base_cmd = ["uv", "run", "python", "-m", f"ocr_vs_vlm.benchmarks.{experiment.benchmark_module}"]

        # Compute results directory (absolute path to 1_raw)
        results_path = str(self.results_dir / experiment.dataset)

        # Add common args
        cmd = base_cmd + [
            "--phases", experiment.phase,
            "--sample-limit", str(experiment.sample_count),
            "--results-dir", results_path,
        ]

        # Add model argument (different flag for QA vs parsing benchmarks)
        if "qa" in experiment.benchmark_module.lower() or experiment.dataset.startswith("InfographicVQA"):
            cmd.extend(["--vlm-models"] + experiment.models)
        else:
            cmd.extend(["--models"] + experiment.models)

        return cmd

    def validate_environment(self) -> bool:
        """Validate that environment is ready to run benchmarks."""
        print("🔍 Validating environment...")

        # Check benchmark modules exist
        benchmark_modules = [
            "ocr_vs_vlm.benchmarks.benchmark_infographicvqa",
            "ocr_vs_vlm.benchmarks.benchmark_publaynet",
        ]

        for module_name in benchmark_modules:
            module_path = self.project_root / module_name.replace(".", "/")
            if not module_path.with_suffix(".py").exists():
                print(f"❌ Benchmark module not found: {module_path}.py")
                return False

        # Check results directory exists
        if not self.results_dir.exists():
            print(f"❌ Results directory not found: {self.results_dir}")
            return False

        # Check AWS credentials (for Claude Bedrock)
        try:
            result = subprocess.run(
                ["aws", "sts", "get-caller-identity"],
                capture_output=True,
                timeout=10
            )
            if result.returncode != 0:
                print("⚠️  AWS credentials not configured (needed for Claude models)")
                print("   Run: aws-login")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠️  AWS CLI not available or not responding")
            return False

        print("✅ Environment validation passed")
        return True

    def run_experiment(self, experiment: Experiment, dry_run: bool = False) -> ExperimentResult:
        """Run a single experiment via subprocess."""
        cmd = self.build_command(experiment)

        print(f"\n{'='*80}")
        print(f"📊 Running: {experiment}")
        print(f"{'='*80}")
        print(f"Command: {' '.join(cmd)}")

        if dry_run:
            print("🔍 DRY RUN - Command not executed")
            return ExperimentResult(
                experiment=experiment,
                success=True,
                duration_seconds=0.0,
                output_files=[],
            )

        start_time = time.time()

        try:
            # Run subprocess with streaming output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=self.project_root,
            )

            output_lines = []

            # Stream output line by line
            if process.stdout:
                for line in process.stdout:
                    line = line.rstrip()
                    output_lines.append(line)
                    if self.verbose:
                        print(line)
                    elif any(keyword in line.lower() for keyword in ["error", "warning", "✅", "❌", "saved"]):
                        print(line)

            # Wait for completion with timeout
            try:
                process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                duration = time.time() - start_time
                return ExperimentResult(
                    experiment=experiment,
                    success=False,
                    duration_seconds=duration,
                    error_message=f"Timeout after {self.timeout} seconds",
                    exit_code=-1,
                )

            duration = time.time() - start_time
            success = process.returncode == 0

            # Parse output files from logs
            output_files = []
            for line in output_lines:
                if "saved" in line.lower() and ".csv" in line.lower():
                    # Extract file path from line
                    parts = line.split()
                    for part in parts:
                        if ".csv" in part:
                            output_files.append(part)

            result = ExperimentResult(
                experiment=experiment,
                success=success,
                duration_seconds=duration,
                output_files=output_files,
                exit_code=process.returncode,
                error_message=None if success else "Experiment failed (see logs)",
            )

            if success:
                print(f"✅ Completed in {duration:.1f}s")
            else:
                print(f"❌ Failed with exit code {process.returncode}")

            return result

        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Exception: {e}")
            return ExperimentResult(
                experiment=experiment,
                success=False,
                duration_seconds=duration,
                error_message=str(e),
                exit_code=-1,
            )

    def save_summary(self, summary: ExecutionSummary) -> Path:
        """Save execution summary to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.results_dir / f"missing_data_execution_summary_{timestamp}.json"

        with open(summary_file, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)

        print(f"\n📄 Execution summary saved to: {summary_file}")
        return summary_file

    def print_summary(self, summary: ExecutionSummary):
        """Print execution summary to console."""
        print(f"\n{'='*80}")
        print("📊 EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total experiments: {len(summary.experiments)}")
        print(f"✅ Successful: {summary.success_count}")
        print(f"❌ Failed: {summary.failure_count}")
        print(f"⏱️  Total duration: {summary.total_duration:.1f}s ({summary.total_duration/60:.1f} min)")

        if summary.experiments:
            print(f"\n{'Dataset':<25} {'Phase':<10} {'Status':<10} {'Duration':<10}")
            print("-" * 80)
            for result in summary.experiments:
                status = "✅ Success" if result.success else "❌ Failed"
                duration = f"{result.duration_seconds:.1f}s"
                print(f"{result.experiment.dataset:<25} {result.experiment.phase:<10} {status:<10} {duration:<10}")

        # Show failed experiments with details
        failed = [exp for exp in summary.experiments if not exp.success]
        if failed:
            print(f"\n{'='*80}")
            print("❌ FAILED EXPERIMENTS")
            print(f"{'='*80}")
            for result in failed:
                print(f"\n{result.experiment}")
                print(f"  Error: {result.error_message}")
                print(f"  Exit code: {result.exit_code}")

    def update_valid_files_json(self, summary: ExecutionSummary) -> bool:
        """Update valid_files.json with newly generated results."""
        try:
            print(f"\n{'='*80}")
            print("📝 UPDATING valid_files.json")
            print(f"{'='*80}")

            # Load existing valid_files.json
            if not self.valid_files_path.exists():
                print(f"❌ valid_files.json not found at: {self.valid_files_path}")
                return False

            with open(self.valid_files_path, 'r') as f:
                valid_files = json.load(f)

            # Track changes
            added_count = 0

            # Process successful experiments
            for result in summary.experiments:
                if not result.success:
                    continue

                exp = result.experiment
                dataset = exp.dataset
                phase = exp.phase

                # Ensure dataset exists in valid_files
                if dataset not in valid_files['valid_files']:
                    valid_files['valid_files'][dataset] = {}
                    print(f"  ➕ Added new dataset: {dataset}")

                # Ensure phase exists
                if phase not in valid_files['valid_files'][dataset]:
                    valid_files['valid_files'][dataset][phase] = {}
                    print(f"  ➕ Added new phase: {dataset}/{phase}")

                # Find generated CSV files for each model
                for model in exp.models:
                    # Construct expected file path based on dataset type
                    if dataset == "InfographicVQA_mini":
                        # Flat structure with timestamp
                        phase_dir = self.results_dir / dataset / phase
                        csv_files = list(phase_dir.glob(f"{model}_results_*.csv"))

                        if csv_files:
                            # Get most recent file
                            csv_file = sorted(csv_files, key=lambda p: p.stat().st_mtime)[-1]

                            # Extract timestamp from filename
                            filename = csv_file.name
                            timestamp = filename.split('_results_')[1].replace('.csv', '') if '_results_' in filename else None

                            # Count rows
                            try:
                                import csv as csv_module
                                with open(csv_file, 'r') as csvf:
                                    row_count = sum(1 for _ in csv_module.reader(csvf)) - 1  # Subtract header
                            except:
                                row_count = exp.sample_count  # Default to expected count

                            # Add entry
                            valid_files['valid_files'][dataset][phase][model] = {
                                "file_path": str(csv_file),
                                "row_count": row_count,
                                "timestamp": timestamp
                            }
                            added_count += 1
                            print(f"  ✅ Added: {dataset}/{phase}/{model} ({timestamp})")
                        else:
                            print(f"  ⚠️  File not found: {dataset}/{phase}/{model}")

                    elif dataset == "publaynet_full":
                        # Nested model directory, no timestamp
                        csv_file = self.results_dir / dataset / phase / model / f"{phase}_{model}_results.csv"

                        if csv_file.exists():
                            # Count rows
                            try:
                                import csv as csv_module
                                with open(csv_file, 'r') as csvf:
                                    row_count = sum(1 for _ in csv_module.reader(csvf)) - 1
                            except:
                                row_count = exp.sample_count

                            # Add entry
                            valid_files['valid_files'][dataset][phase][model] = {
                                "file_path": str(csv_file),
                                "row_count": row_count,
                                "timestamp": None
                            }
                            added_count += 1
                            print(f"  ✅ Added: {dataset}/{phase}/{model}")
                        else:
                            print(f"  ⚠️  File not found: {csv_file}")

            if added_count > 0:
                # Update generated_at timestamp
                valid_files['generated_at'] = datetime.now().isoformat()

                # Save updated valid_files.json
                with open(self.valid_files_path, 'w') as f:
                    json.dump(valid_files, f, indent=2)

                print(f"\n✅ Updated valid_files.json with {added_count} new entries")
                print(f"   Saved to: {self.valid_files_path}")
                return True
            else:
                print("\n⚠️  No new entries to add to valid_files.json")
                return False

        except Exception as e:
            print(f"\n❌ Failed to update valid_files.json: {e}")
            return False

    def run_consolidation(self) -> bool:
        """Run the clean_files.py script to consolidate results."""
        try:
            print(f"\n{'='*80}")
            print("🔄 RUNNING CONSOLIDATION")
            print(f"{'='*80}")

            clean_script = self.project_root / "ocr_vs_vlm" / "results" / "2_clean" / "clean_files.py"

            if not clean_script.exists():
                print(f"❌ clean_files.py not found at: {clean_script}")
                return False

            # Run clean_files.py --incremental
            cmd = ["python", str(clean_script), "--incremental"]
            print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=clean_script.parent,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode == 0:
                print("✅ Consolidation completed successfully")
                if result.stdout:
                    print("\nOutput:")
                    print(result.stdout)
                return True
            else:
                print(f"❌ Consolidation failed with exit code {result.returncode}")
                if result.stderr:
                    print("\nError:")
                    print(result.stderr)
                return False

        except subprocess.TimeoutExpired:
            print("❌ Consolidation timed out after 5 minutes")
            return False
        except Exception as e:
            print(f"❌ Failed to run consolidation: {e}")
            return False

    def run_all(self, dry_run: bool = False, skip_validation: bool = False,
                dataset_filter: Optional[List[str]] = None) -> ExecutionSummary:
        """Run all missing experiments."""
        if not skip_validation and not dry_run:
            if not self.validate_environment():
                print("\n❌ Environment validation failed. Fix issues and try again.")
                print("   Use --skip-validation to bypass checks (not recommended)")
                sys.exit(1)

        experiments = self.get_missing_experiments()

        # Filter by dataset if requested
        if dataset_filter:
            experiments = [exp for exp in experiments if exp.dataset in dataset_filter]
            print(f"\n🔍 Filtered to {len(experiments)} experiments for datasets: {', '.join(dataset_filter)}")

        if not experiments:
            print("❌ No experiments to run (check --datasets filter)")
            sys.exit(1)

        print(f"\n{'='*80}")
        print(f"🚀 STARTING BENCHMARK EXECUTION")
        print(f"{'='*80}")
        print(f"Total experiments: {len(experiments)}")
        print(f"Timeout per experiment: {self.timeout}s ({self.timeout/60:.1f} min)")
        print(f"Estimated total time: {len(experiments) * self.timeout / 3600:.1f} hours (if all timeout)")

        if dry_run:
            print("\n🔍 DRY RUN MODE - Commands will be shown but not executed")

        summary = ExecutionSummary(start_time=datetime.now())

        for i, experiment in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}] Starting: {experiment}")
            result = self.run_experiment(experiment, dry_run=dry_run)
            summary.experiments.append(result)

            # Show progress
            if not dry_run:
                elapsed = (datetime.now() - summary.start_time).total_seconds()
                avg_time = elapsed / i
                remaining = avg_time * (len(experiments) - i)
                print(f"⏱️  Progress: {i}/{len(experiments)} | "
                      f"Elapsed: {elapsed/60:.1f}min | "
                      f"Est. remaining: {remaining/60:.1f}min")

        summary.end_time = datetime.now()

        # Print summary
        self.print_summary(summary)

        # Save summary to file
        if not dry_run:
            self.save_summary(summary)

            # Update valid_files.json and run consolidation if any experiments succeeded
            if summary.success_count > 0:
                print("\n" + "="*80)
                print("🔧 POST-PROCESSING")
                print("="*80)

                # Update valid_files.json
                if self.update_valid_files_json(summary):
                    # Run consolidation script
                    self.run_consolidation()
                else:
                    print("⚠️  Skipping consolidation due to valid_files.json update failure")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run missing benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_missing_data.py                           # Run all experiments
  python benchmark_missing_data.py --dry-run                 # Show commands only
  python benchmark_missing_data.py --datasets publaynet_full # Run specific dataset
  python benchmark_missing_data.py --verbose --timeout 7200  # Detailed output, 2hr timeout
        """
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing them"
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["InfographicVQA_mini", "publaynet_full"],
        help="Run only specified datasets"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per experiment in seconds (default: 3600 = 1 hour)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed benchmark output"
    )

    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip environment validation checks"
    )

    args = parser.parse_args()

    runner = MissingDataBenchmarkRunner(
        verbose=args.verbose,
        timeout=args.timeout
    )

    summary = runner.run_all(
        dry_run=args.dry_run,
        skip_validation=args.skip_validation,
        dataset_filter=args.datasets
    )

    # Exit with error code if any experiments failed
    if summary.failure_count > 0 and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
