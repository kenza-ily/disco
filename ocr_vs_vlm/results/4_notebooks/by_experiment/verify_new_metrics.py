#!/usr/bin/env python3
"""
Verify that new metrics are accessible in the QA notebooks.
This script checks that the consolidation added the 4 new metrics.
"""

import pandas as pd
from pathlib import Path

# Test DocVQA
print("=" * 80)
print("VERIFYING NEW METRICS IN QA NOTEBOOKS")
print("=" * 80)

datasets = ['DocVQA_mini', 'InfographicVQA_mini']
phases_to_check = {
    'DocVQA_mini': 'QA1b',
    'InfographicVQA_mini': 'QA1a'
}

new_metrics = [
    'embedding_similarity',
    'substring_match',
    'prediction_in_ground_truth',
    'ground_truth_in_prediction'
]

for dataset in datasets:
    phase = phases_to_check[dataset]
    file_path = Path(f"../../../2_clean/{dataset}/{phase}.csv")

    print(f"\n📊 {dataset} - Phase {phase}")
    print("-" * 80)

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(df)} samples")

    # Check for each metric
    all_metrics_found = True
    for metric in new_metrics:
        cols = [c for c in df.columns if metric in c]
        if cols:
            # Check if populated
            sample_col = cols[0]
            populated = df[sample_col].notna().sum()
            mean_val = df[sample_col].mean()
            print(f"  ✓ {metric}: {len(cols)} columns, {populated}/{len(df)} populated, mean={mean_val:.3f}")
        else:
            print(f"  ❌ {metric}: NOT FOUND")
            all_metrics_found = False

    if all_metrics_found:
        print(f"\n✅ All new metrics available in {dataset}!")
    else:
        print(f"\n❌ Some metrics missing in {dataset}")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("\n💡 Run the notebooks to see the metrics in action:")
print("   jupyter notebook 01_docvqa_analysis.ipynb")
print("   jupyter notebook 02_infographicvqa_analysis.ipynb")
