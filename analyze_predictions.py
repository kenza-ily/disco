#!/usr/bin/env python3
"""Analyze VOC2007 predictions to check if they're identical across phases."""

import pandas as pd
from pathlib import Path

results_dir = Path("ocr_vs_vlm/results/2_clean/VOC2007")

# Load the CSV files
files = {
    'Pa': results_dir / 'Pa.csv',
    'Pb': results_dir / 'Pb.csv',
    'Pc': results_dir / 'Pc.csv',
    'Pd': results_dir / 'Pd.csv',
    'phase_2': results_dir / 'phase_2.csv',
    'phase_3a': results_dir / 'phase_3a.csv',
    'phase_4': results_dir / 'phase_4.csv',
}

dfs = {}
for name, path in files.items():
    try:
        dfs[name] = pd.read_csv(path)
        print(f"\n{name}: {len(dfs[name])} rows, columns: {list(dfs[name].columns)}")
    except Exception as e:
        print(f"Error loading {name}: {e}")

# Check if predictions are identical
print("\n" + "="*80)
print("CHECKING IF PREDICTIONS ARE IDENTICAL")
print("="*80)

# Look for prediction columns
for name, df in dfs.items():
    pred_cols = [col for col in df.columns if 'pred' in col.lower() or 'output' in col.lower() or 'text' in col.lower()]
    print(f"\n{name} - Potential prediction columns: {pred_cols}")
    
    if pred_cols:
        pred_col = pred_cols[0]
        # Show first 3 predictions
        print(f"  First 3 predictions:")
        for i, val in enumerate(df[pred_col].head(3)):
            print(f"    Row {i}: {str(val)[:100]}...")

# Check if Pa and Pb are identical
print("\n" + "="*80)
print("COMPARING SPECIFIC PHASES")
print("="*80)

if 'Pa' in dfs and 'Pb' in dfs:
    pa = dfs['Pa']
    pb = dfs['Pb']
    
    print(f"\nPa shape: {pa.shape}")
    print(f"Pb shape: {pb.shape}")
    print(f"Columns match: {list(pa.columns) == list(pb.columns)}")
    
    # Check if any columns are identical
    if list(pa.columns) == list(pb.columns):
        for col in pa.columns:
            identical = (pa[col] == pb[col]).all()
            matches = (pa[col] == pb[col]).sum()
            print(f"  Column '{col}': {matches}/{len(pa)} rows identical - {identical}")

# Check phase_2, phase_3a, phase_4
print(f"\n\nComparing newer phases:")
if 'phase_2' in dfs and 'phase_3a' in dfs and 'phase_4' in dfs:
    p2 = dfs['phase_2']
    p3 = dfs['phase_3a']
    p4 = dfs['phase_4']
    
    print(f"\nphase_2 shape: {p2.shape}")
    print(f"phase_3a shape: {p3.shape}")
    print(f"phase_4 shape: {p4.shape}")
    
    # Try to compare prediction columns if they exist
    for col in p2.columns:
        if col in p3.columns and col in p4.columns:
            matches_23 = (p2[col] == p3[col]).sum()
            matches_34 = (p3[col] == p4[col]).sum()
            matches_24 = (p2[col] == p4[col]).sum()
            print(f"\n  Column '{col}':")
            print(f"    phase_2 vs phase_3a: {matches_23}/{len(p2)} identical")
            print(f"    phase_3a vs phase_4: {matches_34}/{len(p3)} identical")
            print(f"    phase_2 vs phase_4: {matches_24}/{len(p2)} identical")

# Check modification times to see when runs were done
print("\n" + "="*80)
print("FILE MODIFICATION TIMES")
print("="*80)
import os
from datetime import datetime

for name, path in files.items():
    if path.exists():
        mtime = os.path.getmtime(path)
        dt = datetime.fromtimestamp(mtime)
        size = os.path.getsize(path)
        print(f"{name:15} - {dt} - {size:,} bytes")
