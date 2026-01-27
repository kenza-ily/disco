#!/usr/bin/env python3
"""
Notebook Execution Script

Executes analysis notebooks programmatically for automated reporting.

Usage:
    # Run all notebooks
    python run_notebooks.py --all

    # Run specific notebook
    python run_notebooks.py --notebook 00_master_evaluation.ipynb

    # Run with papermill (parameterized execution)
    python run_notebooks.py --notebook 01_docvqa_analysis.ipynb --use-papermill
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# Paths
NOTEBOOKS_DIR = Path(__file__).parent
RESULTS_BASE = NOTEBOOKS_DIR.parent
INFO_DIR = RESULTS_BASE / "0_info"

# Notebook execution order (for --all)
NOTEBOOK_ORDER = [
    "00_master_evaluation.ipynb",
    "01_docvqa_analysis.ipynb",
    "02_infographicvqa_analysis.ipynb",
    "03_iam_handwriting.ipynb",
    "04_icdar_multilingual.ipynb",
    "05_voc2007_chinese.ipynb",
]


def check_jupyter_installed() -> bool:
    """Check if Jupyter is installed."""
    try:
        subprocess.run(
            ["jupyter", "--version"],
            check=True,
            capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_papermill_installed() -> bool:
    """Check if papermill is installed."""
    try:
        subprocess.run(
            ["papermill", "--version"],
            check=True,
            capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def execute_notebook_jupyter(notebook_path: Path) -> bool:
    """
    Execute notebook using jupyter nbconvert.

    Args:
        notebook_path: Path to notebook file

    Returns:
        True if execution succeeded, False otherwise
    """
    print(f"\nExecuting: {notebook_path.name}")
    print("-" * 80)

    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        str(notebook_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=NOTEBOOKS_DIR
        )
        print(f"✓ Successfully executed: {notebook_path.name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Execution failed: {notebook_path.name}")
        print(e.stdout)
        print(e.stderr)
        return False


def execute_notebook_papermill(
    notebook_path: Path,
    params: Optional[Dict] = None
) -> bool:
    """
    Execute notebook using papermill (for parameterization).

    Args:
        notebook_path: Path to notebook file
        params: Optional parameters to pass to notebook

    Returns:
        True if execution succeeded, False otherwise
    """
    print(f"\nExecuting with papermill: {notebook_path.name}")
    print("-" * 80)

    output_path = notebook_path.parent / f"{notebook_path.stem}_executed.ipynb"

    cmd = [
        "papermill",
        str(notebook_path),
        str(output_path)
    ]

    # Add parameters
    if params:
        for key, value in params.items():
            cmd.extend(["-p", key, str(value)])

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=NOTEBOOKS_DIR
        )
        print(f"✓ Successfully executed: {notebook_path.name}")
        print(f"  Output: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Execution failed: {notebook_path.name}")
        print(e.stdout)
        print(e.stderr)
        return False


def run_all_notebooks(use_papermill: bool = False) -> bool:
    """
    Execute all notebooks in order.

    Args:
        use_papermill: Use papermill instead of jupyter nbconvert

    Returns:
        True if all executions succeeded, False otherwise
    """
    print("\n" + "="*80)
    print("EXECUTING ALL NOTEBOOKS")
    print("="*80)

    if use_papermill and not check_papermill_installed():
        print("ERROR: papermill is not installed")
        print("Install with: pip install papermill")
        return False

    if not use_papermill and not check_jupyter_installed():
        print("ERROR: jupyter is not installed")
        print("Install with: pip install jupyter")
        return False

    success_count = 0
    fail_count = 0

    for notebook_name in NOTEBOOK_ORDER:
        notebook_path = NOTEBOOKS_DIR / notebook_name

        if not notebook_path.exists():
            print(f"\nWARNING: Notebook not found: {notebook_name}")
            continue

        if use_papermill:
            success = execute_notebook_papermill(notebook_path)
        else:
            success = execute_notebook_jupyter(notebook_path)

        if success:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Succeeded: {success_count}")
    print(f"Failed: {fail_count}")
    print()

    return fail_count == 0


def run_single_notebook(
    notebook_name: str,
    use_papermill: bool = False,
    params: Optional[Dict] = None
) -> bool:
    """
    Execute a single notebook.

    Args:
        notebook_name: Name of notebook file
        use_papermill: Use papermill instead of jupyter nbconvert
        params: Optional parameters for papermill

    Returns:
        True if execution succeeded, False otherwise
    """
    notebook_path = NOTEBOOKS_DIR / notebook_name

    if not notebook_path.exists():
        print(f"ERROR: Notebook not found: {notebook_path}")
        return False

    if use_papermill:
        if not check_papermill_installed():
            print("ERROR: papermill is not installed")
            print("Install with: pip install papermill")
            return False
        return execute_notebook_papermill(notebook_path, params)
    else:
        if not check_jupyter_installed():
            print("ERROR: jupyter is not installed")
            print("Install with: pip install jupyter")
            return False
        return execute_notebook_jupyter(notebook_path)


def list_notebooks() -> List[str]:
    """List all available notebooks."""
    notebooks = []
    for nb_path in NOTEBOOKS_DIR.glob("*.ipynb"):
        if not nb_path.name.startswith("."):
            notebooks.append(nb_path.name)
    return sorted(notebooks)


def main():
    parser = argparse.ArgumentParser(
        description="Execute analysis notebooks programmatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all notebooks
  python run_notebooks.py --all

  # Run specific notebook
  python run_notebooks.py --notebook 00_master_evaluation.ipynb

  # Run with papermill
  python run_notebooks.py --notebook 01_docvqa_analysis.ipynb --use-papermill

  # List available notebooks
  python run_notebooks.py --list
        """
    )

    parser.add_argument(
        '--all', action='store_true',
        help='Execute all notebooks in order'
    )
    parser.add_argument(
        '--notebook', type=str,
        help='Execute specific notebook'
    )
    parser.add_argument(
        '--use-papermill', action='store_true',
        help='Use papermill for execution (allows parameterization)'
    )
    parser.add_argument(
        '--params', type=str,
        help='Parameters for papermill as JSON string'
    )
    parser.add_argument(
        '--list', action='store_true',
        help='List all available notebooks'
    )

    args = parser.parse_args()

    # List notebooks
    if args.list:
        print("\nAvailable notebooks:")
        for nb in list_notebooks():
            print(f"  - {nb}")
        print()
        return 0

    # Parse parameters if provided
    params = None
    if args.params:
        try:
            params = json.loads(args.params)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in --params: {e}")
            return 1

    # Execute notebooks
    success = True

    if args.all:
        success = run_all_notebooks(args.use_papermill)

    elif args.notebook:
        success = run_single_notebook(args.notebook, args.use_papermill, params)

    else:
        parser.print_help()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
