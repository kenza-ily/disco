#!/usr/bin/env python3
"""
Master Pipeline Orchestrator

Runs the complete results processing pipeline:
1. Validation - Check all raw files for errors
2. Consolidation - Generate clean CSV files
3. Postprocessing - Aggregate statistics
4. Status update - Update pipeline_status.json

Usage:
    # Full pipeline
    python 0_info/run_pipeline.py --full

    # Individual stages
    python 0_info/run_pipeline.py --validate
    python 0_info/run_pipeline.py --consolidate
    python 0_info/run_pipeline.py --postprocess

    # Specific dataset
    python 0_info/run_pipeline.py --dataset DocVQA_mini --full
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# Paths
RESULTS_BASE = Path(__file__).parent.parent
INFO_DIR = RESULTS_BASE / "0_info"
CLEAN_DIR = RESULTS_BASE / "2_clean"
NOTEBOOKS_DIR = RESULTS_BASE / "3_notebooks"

PIPELINE_STATUS_FILE = INFO_DIR / "pipeline_status.json"
CLEAN_EXPERIMENTS_FILE = INFO_DIR / "clean_experiments.json"


def load_pipeline_status() -> Dict:
    """Load pipeline_status.json."""
    if not PIPELINE_STATUS_FILE.exists():
        return {
            "version": "1.0.0",
            "last_updated": None,
            "pipeline_runs": {},
            "datasets_status": {}
        }

    with open(PIPELINE_STATUS_FILE, 'r') as f:
        return json.load(f)


def save_pipeline_status(status: Dict):
    """Save pipeline_status.json."""
    status['last_updated'] = datetime.now().isoformat()

    with open(PIPELINE_STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)

    print(f"Updated: {PIPELINE_STATUS_FILE}")


def load_clean_experiments() -> Dict:
    """Load clean_experiments.json."""
    if not CLEAN_EXPERIMENTS_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {CLEAN_EXPERIMENTS_FILE}")

    with open(CLEAN_EXPERIMENTS_FILE, 'r') as f:
        return json.load(f)


def run_validation(dataset: Optional[str] = None) -> bool:
    """
    Run validation stage.

    Args:
        dataset: Specific dataset to validate, or None for all

    Returns:
        True if validation passed, False otherwise
    """
    print("\n" + "="*80)
    print("STAGE 1: VALIDATION")
    print("="*80 + "\n")

    cmd = [sys.executable, str(CLEAN_DIR / "clean_files.py"), "--validate-only"]

    if dataset:
        cmd.extend(["--dataset", dataset])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Update status
        status = load_pipeline_status()
        status['pipeline_runs']['last_validation'] = datetime.now().isoformat()
        save_pipeline_status(status)

        return True

    except subprocess.CalledProcessError as e:
        print(f"Validation failed: {e}")
        print(e.stdout)
        print(e.stderr)
        return False


def run_consolidation(dataset: Optional[str] = None, incremental: bool = False) -> bool:
    """
    Run consolidation stage.

    Args:
        dataset: Specific dataset to consolidate, or None for all
        incremental: Only process new models

    Returns:
        True if consolidation succeeded, False otherwise
    """
    print("\n" + "="*80)
    print("STAGE 2: CONSOLIDATION")
    print("="*80 + "\n")

    cmd = [sys.executable, str(CLEAN_DIR / "clean_files.py")]

    if incremental:
        cmd.append("--incremental")

    if dataset:
        cmd.extend(["--dataset", dataset])

    # Add config flag to use metadata
    cmd.extend(["--config", str(CLEAN_EXPERIMENTS_FILE)])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Update status
        status = load_pipeline_status()
        status['pipeline_runs']['last_consolidation'] = datetime.now().isoformat()

        if dataset:
            if 'datasets_status' not in status:
                status['datasets_status'] = {}
            if dataset not in status['datasets_status']:
                status['datasets_status'][dataset] = {}
            status['datasets_status'][dataset]['consolidation_status'] = 'completed'

        save_pipeline_status(status)

        return True

    except subprocess.CalledProcessError as e:
        print(f"Consolidation failed: {e}")
        print(e.stdout)
        print(e.stderr)
        return False


def run_postprocessing(dataset: Optional[str] = None) -> bool:
    """
    Run postprocessing stage (aggregate statistics).

    Args:
        dataset: Specific dataset to process, or None for all

    Returns:
        True if postprocessing succeeded, False otherwise
    """
    print("\n" + "="*80)
    print("STAGE 3: POSTPROCESSING")
    print("="*80 + "\n")

    # TODO: Implement postprocessing script
    # For now, just update status
    print("Postprocessing stage not yet implemented")
    print("(Manual aggregation still required)")

    status = load_pipeline_status()
    status['pipeline_runs']['last_postprocessing'] = datetime.now().isoformat()
    save_pipeline_status(status)

    return True


def run_full_pipeline(dataset: Optional[str] = None, incremental: bool = False) -> bool:
    """
    Run complete pipeline.

    Args:
        dataset: Specific dataset to process, or None for all
        incremental: Only process new models in consolidation

    Returns:
        True if all stages succeeded, False otherwise
    """
    print("\n" + "="*80)
    print("RUNNING FULL PIPELINE")
    print("="*80)

    # Stage 1: Validation
    if not run_validation(dataset):
        print("\nERROR: Validation failed. Stopping pipeline.")
        return False

    # Stage 2: Consolidation
    if not run_consolidation(dataset, incremental):
        print("\nERROR: Consolidation failed. Stopping pipeline.")
        return False

    # Stage 3: Postprocessing
    if not run_postprocessing(dataset):
        print("\nWARNING: Postprocessing had issues, but continuing.")

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")

    return True


def list_datasets() -> List[str]:
    """List all available datasets from metadata."""
    metadata = load_clean_experiments()
    return list(metadata['datasets'].keys())


def show_status():
    """Display current pipeline status."""
    print("\n" + "="*80)
    print("PIPELINE STATUS")
    print("="*80 + "\n")

    status = load_pipeline_status()

    print(f"Version: {status.get('version', 'unknown')}")
    print(f"Last Updated: {status.get('last_updated', 'never')}")
    print()

    print("Last Pipeline Runs:")
    for stage, timestamp in status.get('pipeline_runs', {}).items():
        print(f"  {stage}: {timestamp or 'never'}")
    print()

    print("Dataset Status:")
    datasets = status.get('datasets_status', {})
    if not datasets:
        print("  No datasets processed yet")
    else:
        for dataset, info in datasets.items():
            print(f"\n  {dataset}:")
            print(f"    Last updated: {info.get('last_updated', 'unknown')}")
            print(f"    Validation: {info.get('validation_status', 'unknown')}")
            print(f"    Consolidation: {info.get('consolidation_status', 'unknown')}")
            print(f"    Notebook: {info.get('notebook_status', 'unknown')}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline orchestrator for OCR vs VLM results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py --full

  # Run only validation
  python run_pipeline.py --validate

  # Run consolidation (incremental mode)
  python run_pipeline.py --consolidate --incremental

  # Process specific dataset
  python run_pipeline.py --dataset DocVQA_mini --full

  # Show current status
  python run_pipeline.py --status
        """
    )

    parser.add_argument(
        '--full', action='store_true',
        help='Run full pipeline (validate + consolidate + postprocess)'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Run validation stage only'
    )
    parser.add_argument(
        '--consolidate', action='store_true',
        help='Run consolidation stage only'
    )
    parser.add_argument(
        '--postprocess', action='store_true',
        help='Run postprocessing stage only'
    )
    parser.add_argument(
        '--incremental', action='store_true',
        help='Use incremental mode for consolidation (only new models)'
    )
    parser.add_argument(
        '--dataset', type=str,
        help='Process specific dataset only'
    )
    parser.add_argument(
        '--status', action='store_true',
        help='Show current pipeline status'
    )
    parser.add_argument(
        '--list-datasets', action='store_true',
        help='List all available datasets'
    )

    args = parser.parse_args()

    # Show status
    if args.status:
        show_status()
        return 0

    # List datasets
    if args.list_datasets:
        print("\nAvailable datasets:")
        for dataset in list_datasets():
            print(f"  - {dataset}")
        print()
        return 0

    # Run pipeline stages
    success = True

    if args.full:
        success = run_full_pipeline(args.dataset, args.incremental)

    elif args.validate:
        success = run_validation(args.dataset)

    elif args.consolidate:
        success = run_consolidation(args.dataset, args.incremental)

    elif args.postprocess:
        success = run_postprocessing(args.dataset)

    else:
        parser.print_help()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
