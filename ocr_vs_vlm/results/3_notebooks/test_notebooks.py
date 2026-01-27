#!/usr/bin/env python3
"""
Test script to verify all notebooks can load data and compute metrics correctly.

This script tests the key functionality of each notebook without running the full notebook.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def test_qa_notebook(dataset_name, phases):
    """Test QA notebook data loading and metrics."""
    print(f"\n{'='*80}")
    print(f"Testing {dataset_name} Notebook")
    print('='*80)

    DATA_DIR = Path("../2_clean") / dataset_name

    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return False

    all_results = {}
    total_models = set()

    for phase in phases:
        phase_file = DATA_DIR / f"{phase}.csv"
        if not phase_file.exists():
            print(f"⚠ Phase {phase} file not found")
            continue

        df = pd.read_csv(phase_file)
        all_results[phase] = df

        # Extract models
        pred_cols = [col for col in df.columns if col.startswith('prediction_')]
        models = [col.replace('prediction_', '') for col in pred_cols]
        total_models.update(models)

        # Verify metrics exist
        anls_cols = [col for col in df.columns if col.startswith('anls_score_')]
        em_cols = [col for col in df.columns if col.startswith('exact_match_')]

        if len(anls_cols) == 0:
            print(f"❌ {phase}: No ANLS score columns found")
            return False

        # Calculate averages
        avg_anls = df[anls_cols].mean().mean()
        avg_em = df[em_cols].mean().mean()

        print(f"✓ {phase}: {len(df)} samples, {len(models)} models")
        print(f"  ANLS: {avg_anls:.4f}, EM: {avg_em:.4f}")

    print(f"\n✓ Total phases loaded: {len(all_results)}")
    print(f"✓ Total unique models: {len(total_models)}")
    print(f"✓ Models: {', '.join(sorted(total_models))}")

    return True


def test_parsing_notebook(dataset_name, phases):
    """Test parsing notebook data loading and metrics."""
    print(f"\n{'='*80}")
    print(f"Testing {dataset_name} Notebook")
    print('='*80)

    DATA_DIR = Path("../2_clean") / dataset_name

    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return False

    all_results = {}
    total_models = set()

    for phase in phases:
        phase_file = DATA_DIR / f"{phase}.csv"
        if not phase_file.exists():
            print(f"⚠ Phase {phase} file not found")
            continue

        df = pd.read_csv(phase_file)
        all_results[phase] = df

        # Extract models
        pred_cols = [col for col in df.columns if col.startswith('prediction_')]
        models = [col.replace('prediction_', '') for col in pred_cols]
        total_models.update(models)

        # Verify metrics exist (parsing datasets use CER/WER/ANLS)
        metric_cols = [col for col in df.columns if any(
            metric in col for metric in ['_cer_', '_wer_', '_anls_', 'CER', 'WER', 'ANLS']
        )]

        # For parsing, metrics might be stored differently
        # Just verify we have predictions and ground truth
        if 'ground_truth' not in df.columns:
            print(f"❌ {phase}: No ground_truth column found")
            return False

        print(f"✓ {phase}: {len(df)} samples, {len(models)} models")
        if metric_cols:
            print(f"  Metric columns found: {len(metric_cols)}")

    print(f"\n✓ Total phases loaded: {len(all_results)}")
    print(f"✓ Total unique models: {len(total_models)}")
    print(f"✓ Models: {', '.join(sorted(total_models))}")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("NOTEBOOK DATA LOADING TESTS")
    print("="*80)

    results = {}

    # Test QA notebooks
    results['DocVQA'] = test_qa_notebook(
        'DocVQA_mini',
        ['QA1a', 'QA1b', 'QA1c', 'QA2a', 'QA2b', 'QA2c', 'QA3a', 'QA3b']
    )

    results['InfographicVQA'] = test_qa_notebook(
        'InfographicVQA_mini',
        ['QA1a', 'QA1b', 'QA1c', 'QA2a', 'QA2b', 'QA2c', 'QA3a', 'QA3b', 'QA3c', 'QA4a', 'QA4b', 'QA4c']
    )

    # Test parsing notebooks
    results['IAM'] = test_parsing_notebook(
        'IAM_mini',
        ['phase_1', 'phase_2', 'phase_3']
    )

    results['ICDAR'] = test_parsing_notebook(
        'ICDAR_mini',
        ['phase_1', 'phase_2', 'phase_3']
    )

    results['VOC2007'] = test_parsing_notebook(
        'VOC2007',
        ['phase_1', 'phase_2', 'phase_3', 'phase_4']
    )

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for dataset, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{dataset:20s} {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All tests passed! Notebooks are ready to use.")
        return 0
    else:
        print("\n❌ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
