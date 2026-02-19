#!/usr/bin/env python3
"""
Generate embeddings manifest file by scanning raw results directories.

This script scans the 1_raw directory for embedding files and creates a
centralized manifest tracking all embedding file locations.

Usage:
    python create_embeddings_manifest.py
"""
import json
from pathlib import Path
from datetime import datetime


def scan_embeddings():
    """Scan for embeddings files and create manifest."""
    results_dir = Path(__file__).parent.parent
    raw_dir = results_dir / "1_raw"

    manifest = {
        "last_updated": datetime.now().isoformat(),
        "embedding_model": "text-embedding-3-large",
        "embedding_dimension": 3072,
        "datasets": {}
    }

    # Scan QA datasets
    qa_datasets = ["DocVQA_mini", "InfographicVQA_mini"]

    for dataset in qa_datasets:
        dataset_dir = raw_dir / dataset
        if not dataset_dir.exists():
            print(f"  Warning: Dataset directory not found: {dataset_dir}")
            continue

        manifest["datasets"][dataset] = {}

        # Find all phase directories (QA1a, QA2b, etc.)
        phase_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("QA")])

        for phase_dir in phase_dirs:
            phase = phase_dir.name

            # Find embeddings files
            embeddings_files = list(phase_dir.glob("embeddings_*.json"))
            if not embeddings_files:
                continue

            # Take the most recent embeddings file
            emb_file = max(embeddings_files, key=lambda f: f.stat().st_mtime)

            try:
                # Load to get metadata
                with open(emb_file) as f:
                    emb_data = json.load(f)

                # Get list of models from predictions
                models_set = set()
                for sample_models in emb_data.get("predictions", {}).values():
                    models_set.update(sample_models.keys())

                manifest["datasets"][dataset][phase] = {
                    "embeddings_file": str(emb_file.relative_to(results_dir)),
                    "sample_count": len(emb_data.get("predictions", {})),
                    "unique_ground_truths": len(emb_data.get("ground_truths", {})),
                    "models": sorted(list(models_set))
                }

            except Exception as e:
                print(f"  Error processing {emb_file}: {e}")

    # Save manifest
    manifest_file = Path(__file__).parent / "embeddings_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print(f"✓ Created manifest: {manifest_file}")
    print(f"  Datasets: {len(manifest['datasets'])}")

    total_phases = sum(len(phases) for phases in manifest['datasets'].values())
    print(f"  Total phases: {total_phases}")

    for dataset, phases in manifest['datasets'].items():
        print(f"\n  {dataset}:")
        for phase, info in sorted(phases.items()):
            print(f"    {phase}: {info['sample_count']} samples, {len(info['models'])} models")


if __name__ == '__main__':
    scan_embeddings()
