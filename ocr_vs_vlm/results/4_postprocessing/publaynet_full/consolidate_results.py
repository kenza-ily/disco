"""
Consolidate PubLayNet benchmark results across all phases and models.

This script reads raw results CSVs from the benchmark and consolidates them into
phase-specific DataFrames with model-prefixed columns for easy comparison.

Structure:
  - Raw results: results/publaynet_full/{phase}/{model}/*.csv
  - Consolidated: results_postprocessing/publaynet_full/phase_{X}_consolidated.csv
  - Summary: results_postprocessing/publaynet_full/all_phases_summary.csv
"""

import json
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class PubLayNetConsolidator:
    """Consolidates PubLayNet benchmark results."""
    
    PHASES = ["P-A", "P-B", "P-C"]
    PA_MODELS = ["azure_intelligence", "mistral_document_ai"]
    PB_MODELS = ["gpt-5-mini", "gpt-5-nano"]
    PC_MODELS = ["gpt-5-mini", "gpt-5-nano"]
    
    CATEGORY_NAMES = {
        1: "Text",
        2: "Title",
        3: "List",
        4: "Table",
        5: "Figure"
    }
    
    def __init__(self, workspace_root: Path):
        """Initialize consolidator with workspace root path."""
        self.workspace_root = Path(workspace_root)
        self.results_dir = self.workspace_root / "ocr_vs_vlm" / "results" / "publaynet_full"
        self.output_dir = self.workspace_root / "ocr_vs_vlm" / "results_postprocessing" / "publaynet_full"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_models_for_phase(self, phase: str) -> List[str]:
        """Get models evaluated in a specific phase."""
        if phase == "P-A":
            return self.PA_MODELS
        elif phase == "P-B":
            return self.PB_MODELS
        elif phase == "P-C":
            return self.PC_MODELS
        return []
    
    def find_results_files(self) -> Dict[str, Dict[str, Path]]:
        """Find all results CSV files by phase and model.
        
        Returns:
            {phase: {model: csv_path}}
        """
        results_by_phase = {}
        
        for phase in self.PHASES:
            results_by_phase[phase] = {}
            phase_dir = self.results_dir / phase
            
            if not phase_dir.exists():
                print(f"⚠️  Phase directory not found: {phase_dir}")
                continue
            
            models = self.get_models_for_phase(phase)
            for model in models:
                model_dir = phase_dir / model
                csv_file = model_dir / f"{phase}_{model}_results.csv"
                
                if csv_file.exists():
                    results_by_phase[phase][model] = csv_file
                    print(f"✓ Found: {phase}/{model} -> {csv_file}")
                else:
                    print(f"✗ Missing: {phase}/{model} -> {csv_file}")
        
        return results_by_phase
    
    def read_csv_raw(self, csv_file: Path) -> List[Dict]:
        """Read CSV file with proper header handling.
        
        The benchmark writes CSV without headers, so we need to add them.
        """
        fieldnames = [
            'sample_id', 'image_path', 'model', 'phase',
            'ground_truth_boxes', 'predicted_boxes',
            'inference_time_ms', 'error', 'timestamp'
        ]
        
        rows = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            for row in reader:
                # Parse JSON fields
                try:
                    row['ground_truth_boxes'] = json.loads(row['ground_truth_boxes'])
                    row['predicted_boxes'] = json.loads(row['predicted_boxes'])
                except json.JSONDecodeError:
                    row['ground_truth_boxes'] = []
                    row['predicted_boxes'] = []
                
                rows.append(row)
        
        return rows
    
    def consolidate_phase_results(self, phase: str) -> Optional[dict]:
        """Consolidate results for a specific phase into a single DataFrame-like structure.
        
        Returns:
            {sample_id: {ground_truth_boxes, model1_predicted_boxes, model1_inference_time_ms, ...}}
        """
        print(f"\n{'='*80}")
        print(f"Consolidating Phase {phase}")
        print(f"{'='*80}")
        
        models = self.get_models_for_phase(phase)
        phase_dir = self.results_dir / phase
        
        # Load all results by model
        results_by_model = {}
        for model in models:
            csv_file = phase_dir / model / f"{phase}_{model}_results.csv"
            if csv_file.exists():
                rows = self.read_csv_raw(csv_file)
                results_by_model[model] = {r['sample_id']: r for r in rows}
                print(f"  Loaded {len(rows)} samples from {model}")
            else:
                print(f"  ⚠️  Missing results for {model}")
                return None
        
        # Find common sample IDs
        if not results_by_model:
            return None
        
        sample_ids = set(results_by_model[models[0]].keys())
        for model in models[1:]:
            sample_ids &= set(results_by_model[model].keys())
        
        sample_ids = sorted(sample_ids)
        print(f"  Common samples: {len(sample_ids)}")
        
        # Consolidate into single structure
        consolidated = {}
        for sample_id in sample_ids:
            consolidated[sample_id] = {
                'sample_id': sample_id,
                'ground_truth_boxes': results_by_model[models[0]][sample_id]['ground_truth_boxes'],
            }
            
            for model in models:
                result = results_by_model[model][sample_id]
                consolidated[sample_id][f'{model}_predicted_boxes'] = result['predicted_boxes']
                consolidated[sample_id][f'{model}_inference_time_ms'] = float(result['inference_time_ms'])
                if result.get('error'):
                    consolidated[sample_id][f'{model}_error'] = result['error']
        
        # Save consolidated results to CSV
        output_file = self.output_dir / f"{phase}_consolidated.csv"
        self._save_consolidated_csv(output_file, consolidated, models)
        
        # Calculate basic statistics
        self._print_phase_stats(consolidated, models, phase)
        
        return consolidated
    
    def _save_consolidated_csv(self, output_file: Path, consolidated: dict, models: List[str]):
        """Save consolidated results to CSV format."""
        if not consolidated:
            return
        
        fieldnames = ['sample_id', 'ground_truth_boxes']
        for model in models:
            fieldnames.extend([
                f'{model}_predicted_boxes',
                f'{model}_inference_time_ms'
            ])
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            
            for sample_id in sorted(consolidated.keys()):
                row = {'sample_id': sample_id}
                row['ground_truth_boxes'] = json.dumps(consolidated[sample_id]['ground_truth_boxes'])
                
                for model in models:
                    row[f'{model}_predicted_boxes'] = json.dumps(
                        consolidated[sample_id][f'{model}_predicted_boxes']
                    )
                    row[f'{model}_inference_time_ms'] = consolidated[sample_id][f'{model}_inference_time_ms']
                
                writer.writerow(row)
        
        print(f"  Saved consolidated results: {output_file}")
    
    def _print_phase_stats(self, consolidated: dict, models: List[str], phase: str):
        """Print basic statistics for a phase."""
        print(f"\n  Statistics for {phase}:")
        print(f"  {'-'*60}")
        
        # Box counts
        gt_counts = []
        for sample_id, data in consolidated.items():
            gt_counts.append(len(data['ground_truth_boxes']))
        
        if gt_counts:
            print(f"  Ground truth boxes per sample:")
            print(f"    Mean: {statistics.mean(gt_counts):.1f}, Median: {statistics.median(gt_counts):.1f}")
            print(f"    Min: {min(gt_counts):.0f}, Max: {max(gt_counts):.0f}")
        
        # Inference times by model
        for model in models:
            times = []
            for sample_id, data in consolidated.items():
                times.append(data[f'{model}_inference_time_ms'])
            
            if times:
                print(f"  {model} inference time (ms):")
                print(f"    Mean: {statistics.mean(times):.1f}, Median: {statistics.median(times):.1f}")
                print(f"    Min: {min(times):.1f}, Max: {max(times):.1f}")
        
        # Predicted box counts
        for model in models:
            pred_counts = []
            for sample_id, data in consolidated.items():
                pred_counts.append(len(data[f'{model}_predicted_boxes']))
            
            if pred_counts:
                print(f"  {model} predicted boxes per sample:")
                print(f"    Mean: {statistics.mean(pred_counts):.1f}, Median: {statistics.median(pred_counts):.1f}")
                print(f"    Min: {min(pred_counts):.0f}, Max: {max(pred_counts):.0f}")
    
    def create_all_phases_summary(self, all_consolidated: Dict[str, dict]) -> None:
        """Create a summary file with statistics across all phases."""
        print(f"\n{'='*80}")
        print("Creating All Phases Summary")
        print(f"{'='*80}\n")
        
        summary_rows = []
        all_fieldnames = {'phase', 'sample_id', 'ground_truth_box_count'}
        
        for phase in self.PHASES:
            if phase not in all_consolidated:
                continue
            
            consolidated = all_consolidated[phase]
            models = self.get_models_for_phase(phase)
            
            # Calculate per-phase statistics
            for sample_id in sorted(consolidated.keys()):
                data = consolidated[sample_id]
                row = {
                    'phase': phase,
                    'sample_id': sample_id,
                    'ground_truth_box_count': len(data['ground_truth_boxes']),
                }
                
                for model in models:
                    pred_boxes = data[f'{model}_predicted_boxes']
                    row[f'{model}_box_count'] = len(pred_boxes)
                    row[f'{model}_inference_time_ms'] = data[f'{model}_inference_time_ms']
                    
                    # Count boxes by category
                    for cat_id in range(1, 6):
                        count = sum(1 for b in pred_boxes if b.get('category') == cat_id)
                        row[f'{model}_cat{cat_id}_count'] = count
                
                summary_rows.append(row)
                all_fieldnames.update(row.keys())
        
        # Save summary
        if summary_rows:
            output_file = self.output_dir / "all_phases_summary.csv"
            # Order fieldnames: phase, sample_id, ground_truth, then rest alphabetically
            fieldnames = ['phase', 'sample_id', 'ground_truth_box_count']
            fieldnames.extend(sorted([f for f in all_fieldnames if f not in fieldnames]))
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)
            
            print(f"✓ Saved summary: {output_file}")
            print(f"  Total rows: {len(summary_rows)}")
    
    def run(self):
        """Run the complete consolidation pipeline."""
        print("\nPubLayNet Results Consolidation")
        print("=" * 80)
        
        # Find all results files
        results_files = self.find_results_files()
        
        # Consolidate each phase
        all_consolidated = {}
        for phase in self.PHASES:
            if phase in results_files:
                consolidated = self.consolidate_phase_results(phase)
                if consolidated:
                    all_consolidated[phase] = consolidated
        
        # Create summary
        self.create_all_phases_summary(all_consolidated)
        
        print(f"\n{'='*80}")
        print("✓ Consolidation Complete!")
        print(f"{'='*80}\n")


def main():
    """Main entry point."""
    import sys
    
    # Determine workspace root
    if len(sys.argv) > 1:
        workspace_root = Path(sys.argv[1])
    else:
        # Try to infer from current directory
        workspace_root = Path.cwd()
        if not (workspace_root / "ocr_vs_vlm").exists():
            workspace_root = workspace_root.parent
    
    consolidator = PubLayNetConsolidator(workspace_root)
    consolidator.run()


if __name__ == "__main__":
    main()
