#!/usr/bin/env python3
"""
Split combined CSV file with multiple phases into separate phase-specific files.
"""
import csv
from pathlib import Path
from collections import defaultdict

# Source file
source_file = Path("/Users/kenzabenkirane/Documents/GitHub/research-playground/ocr_vs_vlm/benchmarks/results/raw/dude_mini/mistral_document_ai_gpt-5-mini/results_20260201_013610.csv")

# Target base directory
target_base = Path("/Users/kenzabenkirane/Documents/GitHub/research-playground/ocr_vs_vlm/results/1_raw/dude_mini")

# Model name for output files
model_name = "mistral_document_ai__gpt-5-mini"
timestamp = "20260201_013610"

def split_by_phase():
    """Split CSV by phase column into separate files."""
    print(f"Reading source file: {source_file}")

    # Read all rows and group by phase
    phase_rows = defaultdict(list)
    header = None

    with open(source_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Get header

        # Find phase column index (should be 8th column, index 7)
        try:
            phase_col_idx = header.index('phase')
        except ValueError:
            print("ERROR: 'phase' column not found in CSV")
            print(f"Available columns: {header}")
            return

        # Group rows by phase
        for row in reader:
            if len(row) <= phase_col_idx:
                print(f"WARNING: Skipping row with insufficient columns: {row[:3]}")
                continue
            phase = row[phase_col_idx]
            phase_rows[phase].append(row)

    print(f"\nFound phases: {list(phase_rows.keys())}")
    for phase, rows in phase_rows.items():
        print(f"  {phase}: {len(rows)} rows")

    # Write separate files for each phase
    for phase, rows in phase_rows.items():
        # Create phase directory if it doesn't exist
        phase_dir = target_base / phase
        phase_dir.mkdir(parents=True, exist_ok=True)

        # Output file path
        output_file = phase_dir / f"{model_name}_results_{timestamp}.csv"

        print(f"\nWriting {phase} to: {output_file}")
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)  # Write header
            writer.writerows(rows)   # Write data rows

        print(f"  Wrote {len(rows)} rows + header")

        # Verify
        line_count = sum(1 for _ in open(output_file))
        print(f"  Verified: {line_count} total lines")

    print("\n✅ Split complete!")

if __name__ == "__main__":
    split_by_phase()
