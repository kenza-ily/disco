#!/usr/bin/env python3
"""
Analyze missing data across all benchmark experiments.
Generates tables showing which models were run for each experiment.
"""

import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

def scan_raw_files(dataset_path: Path) -> dict:
    """Scan raw files and extract model/phase information."""
    results = defaultdict(set)

    for csv_file in dataset_path.rglob("*.csv"):
        # Get relative path from dataset root
        rel_path = csv_file.relative_to(dataset_path)
        parts = rel_path.parts
        filename = csv_file.stem

        # Pattern 1: model/phase_N_results.csv (e.g., azure_intelligence/phase_1_results.csv)
        # Check this FIRST before generic len(parts)==2 check
        if len(parts) == 2 and 'phase_' in filename:
            import re
            model = parts[0]
            # Map phase numbers to phase codes
            phase_map = {'phase_1': 'P-A', 'phase_2': 'P-B', 'phase_3': 'P-C', 'phase_4': 'P-D'}
            # Extract phase number
            phase_match = re.search(r'phase_(\d+)', filename)
            if phase_match:
                phase_num = phase_match.group(1)
                phase = phase_map.get(f'phase_{phase_num}', f'phase_{phase_num}')
                results[phase].add(model)

        # Pattern 2: phase/model_results.csv (e.g., QA1a/azure_intelligence_results.csv)
        # Pattern 3: phase/model/phase_model_results.csv (e.g., P-A/azure_intelligence/P-A_azure_intelligence_results.csv)
        elif len(parts) == 2 or (len(parts) == 3 and parts[0].startswith(('QA', 'P-', '202'))):
            phase = parts[0]
            if len(parts) == 3:
                model = parts[1]
            else:
                # Extract model from filename, remove _results and timestamp
                model = filename.replace('_results', '').split('_20')[0]
            results[phase].add(model)

    return dict(results)

def load_expected_experiments():
    """Load expected experiments from clean_experiments.json."""
    config_path = Path("ocr_vs_vlm/results/0_info/clean_experiments.json")
    with open(config_path) as f:
        return json.load(f)

def analyze_dataset(dataset_name: str, expected_config: dict, raw_path: Path):
    """Analyze a single dataset for missing data."""
    phases = expected_config.get("phases", {})
    if not phases:
        return None

    # Scan actual files
    actual_files = scan_raw_files(raw_path)

    # Build comparison table
    results = []
    for phase_name, phase_config in phases.items():
        expected_models = set(phase_config.get("models", []))
        actual_models = actual_files.get(phase_name, set())

        # Handle composite model names (e.g., "azure_intelligence__gpt-5-mini")
        # For these, check if individual model components exist
        for model in expected_models:
            if "__" in model:
                # Composite model (parsing__qa)
                parts = model.split("__")
                status = "✓" if any(parts[0] in str(m) for m in actual_models) else "✗"
            else:
                # Single model
                status = "✓" if model in actual_models else "✗"

            results.append({
                "Phase": phase_name,
                "Model": model,
                "Status": status,
                "Expected_Count": phase_config.get("sample_count", "N/A")
            })

    return results

def main():
    config = load_expected_experiments()
    raw_base = Path("ocr_vs_vlm/results/1_raw")

    print("=" * 80)
    print("BENCHMARK DATA COMPLETENESS REPORT")
    print("=" * 80)
    print()

    for dataset_name, dataset_config in config["datasets"].items():
        if dataset_config.get("status") == "experimental":
            print(f"⚠️  Skipping experimental dataset: {dataset_name}")
            continue

        dataset_path = raw_base / dataset_name
        if not dataset_path.exists():
            print(f"❌ Dataset directory not found: {dataset_name}")
            print()
            continue

        print(f"📊 Dataset: {dataset_name}")
        print(f"   Type: {dataset_config.get('type', 'N/A')}")
        print(f"   Description: {dataset_config.get('description', 'N/A')}")
        print()

        results = analyze_dataset(dataset_name, dataset_config, dataset_path)

        if not results:
            print("   No phases defined in config")
            print()
            continue

        # Create DataFrame
        df = pd.DataFrame(results)

        # Pivot to create model x phase matrix
        pivot = df.pivot_table(
            index='Model',
            columns='Phase',
            values='Status',
            aggfunc='first'
        )

        print(pivot.to_string())
        print()

        # Summary statistics
        missing_count = (df['Status'] == '✗').sum()
        total_count = len(df)
        completion_pct = ((total_count - missing_count) / total_count * 100) if total_count > 0 else 0

        print(f"   Completion: {total_count - missing_count}/{total_count} ({completion_pct:.1f}%)")

        if missing_count > 0:
            print(f"   ⚠️  Missing: {missing_count} experiments")
            missing_df = df[df['Status'] == '✗']
            for _, row in missing_df.iterrows():
                print(f"      - {row['Phase']}: {row['Model']}")

        print()
        print("-" * 80)
        print()

    # Overall summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Production datasets analyzed:")
    production_count = sum(1 for d in config["datasets"].values() if d.get("status") == "production")
    experimental_count = sum(1 for d in config["datasets"].values() if d.get("status") == "experimental")
    print(f"  - Production: {production_count}")
    print(f"  - Experimental: {experimental_count}")
    print()

if __name__ == "__main__":
    main()
