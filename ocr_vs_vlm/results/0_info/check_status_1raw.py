#!/usr/bin/env python3
"""Check status of ChartQAPro benchmark results in 1_raw folder."""
import csv
from pathlib import Path
from collections import defaultdict

results_dir = Path("ocr_vs_vlm/results/1_raw/chartqapro_mini")
expected_per_phase = 494

print("=" * 70)
print("CHARTQAPRO_MINI BENCHMARK STATUS (1_raw)")
print("=" * 70)
print(f"Expected samples per phase: {expected_per_phase}")
print()

completed = {
    "QA1": {"a": {}, "b": {}, "c": {}},
    "QA2": {"a": {}, "b": {}, "c": {}},
    "QA3": {"a": {}, "b": {}}
}

for model_dir in sorted(results_dir.iterdir()):
    if not model_dir.is_dir():
        continue
    
    for csv_file in sorted(model_dir.glob("*.csv")):
        try:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                phases = defaultdict(set)
                
                for row in reader:
                    phase = row.get("phase", "")
                    sample_id = row.get("sample_id", "")
                    
                    if phase and sample_id:
                        phases[phase].add(sample_id)
                
                for phase, samples in phases.items():
                    if phase.startswith("QA1"):
                        variant = phase[-1]
                        if model_dir.name not in completed["QA1"][variant]:
                            completed["QA1"][variant][model_dir.name] = 0
                        completed["QA1"][variant][model_dir.name] = max(completed["QA1"][variant][model_dir.name], len(samples))
                    elif phase.startswith("QA2"):
                        variant = phase[-1]
                        if model_dir.name not in completed["QA2"][variant]:
                            completed["QA2"][variant][model_dir.name] = 0
                        completed["QA2"][variant][model_dir.name] = max(completed["QA2"][variant][model_dir.name], len(samples))
                    elif phase.startswith("QA3"):
                        variant = phase[-1]
                        if model_dir.name not in completed["QA3"][variant]:
                            completed["QA3"][variant][model_dir.name] = 0
                        completed["QA3"][variant][model_dir.name] = max(completed["QA3"][variant][model_dir.name], len(samples))
                        
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

print("QA1 (OCR parsing + LLM answering):")
print("  Expected OCR models: azure_intelligence, mistral_document_ai")
for variant in ["a", "b", "c"]:
    prompt = {"a": "Simple", "b": "Detailed", "c": "CoT"}[variant]
    print(f"  QA1{variant} ({prompt}):")
    for model, count in completed["QA1"][variant].items():
        status = "✅ COMPLETE" if count >= expected_per_phase else f"❌ {count}/{expected_per_phase}"
        print(f"    {model}: {status}")
    if not completed["QA1"][variant]:
        print("    (none)")

print()
print("QA2 (VLM parsing + LLM answering):")
print("  Expected VLM models: gpt-5-mini, gpt-5-nano, claude_sonnet")
for variant in ["a", "b", "c"]:
    prompt = {"a": "Simple", "b": "Detailed", "c": "CoT"}[variant]
    print(f"  QA2{variant} ({prompt}):")
    for model, count in completed["QA2"][variant].items():
        status = "✅ COMPLETE" if count >= expected_per_phase else f"❌ {count}/{expected_per_phase}"
        print(f"    {model}: {status}")
    if not completed["QA2"][variant]:
        print("    (none)")

print()
print("QA3 (Direct VLM answering):")
print("  Expected VLM models: gpt-5-mini, gpt-5-nano, claude_sonnet")
for variant in ["a", "b"]:
    prompt = {"a": "Simple", "b": "Detailed"}[variant]
    print(f"  QA3{variant} ({prompt}):")
    for model, count in completed["QA3"][variant].items():
        status = "✅ COMPLETE" if count >= expected_per_phase else f"❌ {count}/{expected_per_phase}"
        print(f"    {model}: {status}")
    if not completed["QA3"][variant]:
        print("    (none)")

print()
print("=" * 70)
