#!/usr/bin/env python3
"""
Comprehensive notebook verification script.

Tests:
1. All notebooks can be loaded (valid JSON)
2. No pip install cells remain
3. All data paths resolve correctly
4. All required libraries can be imported
5. Preview cells exist and are properly formatted
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def test_notebook_structure(nb_path: Path) -> Tuple[bool, List[str]]:
    """Test notebook JSON structure and contents."""
    issues = []

    try:
        with open(nb_path, 'r') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return False, [f"File not found"]

    # Check for pip install cells
    pip_cells = []
    for idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                source_text = ''.join(source)
            else:
                source_text = source

            if '%pip install' in source_text or 'pip install' in source_text:
                pip_cells.append(idx)

    if pip_cells:
        issues.append(f"Found pip install in cells: {pip_cells}")

    # Check for preview cells
    has_preview = False
    for cell in nb.get('cells', []):
        source = cell.get('source', [])
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source

        if 'DATA PREVIEW' in source_text and 'random_samples' in source_text:
            has_preview = True
            break

    if not has_preview:
        issues.append("No data preview cell found")

    return len(issues) == 0, issues


def test_data_paths(nb_path: Path) -> Tuple[bool, List[str]]:
    """Test that data paths in notebook resolve correctly."""
    issues = []

    with open(nb_path, 'r') as f:
        nb = json.load(f)

    # Extract paths from cells
    found_paths = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                source_text = ''.join(source)
            else:
                source_text = source

            # Look for path patterns
            if '2_clean' in source_text:
                # Extract the relative path depth
                if '../../../2_clean' in source_text:
                    found_paths.append('../../../2_clean')
                elif '../../2_clean' in source_text:
                    found_paths.append('../../2_clean')
                elif '../2_clean' in source_text:
                    found_paths.append('../2_clean')

    if not found_paths:
        issues.append("No data paths found in notebook")
        return False, issues

    # Verify paths resolve from notebook location
    nb_dir = nb_path.parent
    for rel_path in set(found_paths):
        full_path = (nb_dir / rel_path).resolve()
        if not full_path.exists():
            issues.append(f"Path doesn't exist: {rel_path} -> {full_path}")

    return len(issues) == 0, issues


def test_imports() -> Tuple[bool, List[str]]:
    """Test that all required libraries can be imported."""
    issues = []
    required = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'sklearn',
        'editdistance'
    ]

    for lib in required:
        try:
            __import__(lib)
        except ImportError:
            issues.append(f"Cannot import {lib}")

    return len(issues) == 0, issues


def main():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("NOTEBOOK VERIFICATION")
    print("="*80 + "\n")

    # Test imports first
    print("1. Testing Required Libraries...")
    imports_ok, import_issues = test_imports()
    if imports_ok:
        print("   ✅ All libraries available")
    else:
        print("   ❌ Missing libraries:")
        for issue in import_issues:
            print(f"      - {issue}")
        print("\n   Run: uv sync")
        return 1

    # Find all notebooks
    notebooks = {
        'overview/00_master_evaluation.ipynb': 'Overview',
        'by_task/qa/01_docvqa_analysis.ipynb': 'DocVQA',
        'by_task/qa/02_infographicvqa_analysis.ipynb': 'InfographicVQA',
        'by_task/parsing/03_iam_handwriting.ipynb': 'IAM',
        'by_task/parsing/04_icdar_multilingual.ipynb': 'ICDAR',
        'by_task/parsing/05_voc2007_chinese.ipynb': 'VOC2007',
    }

    print("\n2. Testing Notebook Structure...")
    all_ok = True
    for nb_path, name in notebooks.items():
        path = Path(nb_path)
        if not path.exists():
            print(f"   ❌ {name}: File not found")
            all_ok = False
            continue

        structure_ok, structure_issues = test_notebook_structure(path)

        if structure_ok:
            print(f"   ✅ {name}")
        else:
            print(f"   ❌ {name}:")
            for issue in structure_issues:
                print(f"      - {issue}")
            all_ok = False

    print("\n3. Testing Data Paths...")
    for nb_path, name in notebooks.items():
        path = Path(nb_path)
        if not path.exists():
            continue

        paths_ok, path_issues = test_data_paths(path)

        if paths_ok:
            print(f"   ✅ {name}")
        else:
            print(f"   ❌ {name}:")
            for issue in path_issues:
                print(f"      - {issue}")
            all_ok = False

    # Summary
    print("\n" + "="*80)
    if all_ok:
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nNotebooks are ready to use!")
        print("Launch with: jupyter lab")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*80)
        print("\nFix issues above before running notebooks.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
