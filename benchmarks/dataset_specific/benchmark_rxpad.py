#!/usr/bin/env python3
"""
RX-PAD Medical Prescription Parsing Benchmark

This script runs the OCR/VLM benchmark on the RX-PAD dataset containing
French medical prescription forms with field-level annotations.

Dataset Structure:
- Training Images: datasets/task1_parsing/RX-PAD/training_data/images/
- Testing Images: datasets/task1_parsing/RX-PAD/testing_data/images/
- Annotations: */annotations/*.json (field-level bounding boxes and text)
- Ground Truth: */ground_truth/

The benchmark evaluates:
- Phase 1: OCR baseline with pure OCR models
  - Extract text via OCR (Azure Document Intelligence, Mistral OCR)
  - Output: Extracted text from prescription fields

- Phase 2: VLM baseline with generic prompts
  - Send prescription image to VLM (GPT-5-mini, Claude Sonnet, etc.)
  - Generic prompt: "Extract all text from this prescription"
  - Output: Full text extraction

- Phase 3: VLM with context-aware prompts
  - Send prescription image to VLM with domain context
  - Context: Language (French), Document type (prescription), common fields
  - Detailed instructions for medical prescription structure
  - Output: Structured field extraction with better accuracy

Medical Prescription Fields (French):
- structure_name: Healthcare facility name
- prescriber_name: Doctor/prescriber name
- prescriber_id: Professional license number
- patient_name: Patient name
- patient_dob: Date of birth
- medication_name: Drug name
- dosage: Medication dosage
- route: Administration route
- duration: Treatment duration
- quantity: Quantity to dispense
- refills: Number of refills allowed
- date: Prescription date
- signature: Prescriber signature/authorization

Evaluation:
  - Metrics: CER (Character Error Rate), WER (Word Error Rate), substring matching
  - Field-level accuracy (does VLM correctly identify key fields?)
  - Saves results as CSV for post-processing and analysis
"""

import json
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.benchmark import BenchmarkRunner, create_benchmark_config

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_rxpad_benchmark(
    models: list | None = None,
    phases: list | None = None,
    sample_limit: int | None = None,
    phase_3_letter: str = 'a'
):
    """
    Run RX-PAD French Medical Prescription Benchmark.
    
    Args:
        models: List of models to test (default: all models)
                Examples: ['gpt-5-mini', 'claude_sonnet', 'azure_intelligence']
        phases: List of phases to run (default: [1, 2, 3])
                Phase 1 = OCR baseline
                Phase 2 = VLM baseline
                Phase 3 = VLM with context
        sample_limit: Max samples to process (default: None = all)
        phase_3_letter: Phase 3 variant letter (default: 'a')
    
    Returns:
        Execution summary dict
    """
    if models is None:
        # Default: test all models (both OCR and VLM)
        models = [
            'azure_intelligence',
            'mistral_document_ai',
            'gpt-5-mini',
            'gpt-5-nano',
            'claude_sonnet',
        ]
    
    if phases is None:
        phases = [1, 2, 3]  # All three phases
    
    logger.info("=" * 80)
    logger.info("RX-PAD Medical Prescription Parsing Benchmark (French)")
    logger.info("=" * 80)
    logger.info(f"Models: {models}")
    logger.info(f"Phases: {phases}")
    logger.info(f"Sample limit: {sample_limit or 'All samples'}")
    logger.info(f"Phase 3 letter: {phase_3_letter}")
    logger.info("=" * 80)
    
    # Print prompt information
    logger.info("\nPrompt Configuration:")
    logger.info("-" * 40)
    logger.info("Phase 1 (OCR Baseline):")
    logger.info("  Models: Azure Document Intelligence, Mistral OCR")
    logger.info("  Task: Extract all text from prescription image")
    logger.info("")
    logger.info("Phase 2 (VLM Baseline - Generic):")
    logger.info("  Prompt: 'Extract all text from this prescription document.'")
    logger.info("  Language: French (or detect from document)")
    logger.info("")
    logger.info("Phase 3 (VLM + Context):")
    logger.info("  Full context-aware prompt for Medical Prescriptions")
    logger.info("  Language: French (Français)")
    logger.info("  Document Type: Medical Prescription (Ordonnance Médicale)")
    logger.info("  Key fields to extract:")
    logger.info("    - Healthcare facility name")
    logger.info("    - Prescriber information")
    logger.info("    - Patient information")
    logger.info("    - Medication details (name, dosage, route, duration)")
    logger.info("    - Prescription date and signature")
    logger.info("  Output format: Structured field extraction")
    logger.info("-" * 40)
    
    from models import ModelRegistry
    
    ocr_models = ModelRegistry.list_ocr_models()
    vlm_models = ModelRegistry.list_vlm_models()
    
    all_summaries = {}
    
    for model_name in models:
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING MODEL: {model_name.upper()}")
        logger.info(f"{'='*80}\n")
        
        # Filter phases based on model type
        model_phases = []
        if model_name in ocr_models:
            # OCR models only support phase 1
            model_phases = [p for p in phases if p == 1]
        elif model_name in vlm_models:
            # VLM models only support phases 2 and 3
            model_phases = [p for p in phases if p in [2, 3]]
        else:
            logger.warning(f"Unknown model type: {model_name}. Skipping.")
            continue
        
        if not model_phases:
            logger.info(f"No compatible phases for {model_name}. Skipping.")
            continue
        
        # Create per-model results directory
        # Final path: ocr_vs_vlm/results/raw/RX-PAD/<model_name>/<timestamp>.csv
        # Use absolute path to ensure results go to correct location
        benchmark_dir = Path(__file__).parent
        results_base = benchmark_dir.parent / "results" / "raw"
        model_results_dir = str(results_base)
        
        # Create configuration for this model with filtered phases
        config = create_benchmark_config(
            datasets=['RX-PAD'],
            models=[model_name],
            phases=model_phases,
            sample_limit=sample_limit,
            results_dir=model_results_dir
        )
        
        config.phase_3_letter = phase_3_letter
        
        # Run benchmark
        try:
            runner = BenchmarkRunner(config)
            summary = runner.run_benchmark()
            all_summaries[model_name] = summary
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Summary for {model_name}:")
            logger.info(f"{'='*80}")
            logger.info(json.dumps(summary, indent=2, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"Error running benchmark for {model_name}: {e}")
            all_summaries[model_name] = {'error': str(e)}
    
    logger.info(f"\n{'='*80}")
    logger.info("All models completed!")
    logger.info(f"Results saved to: results/raw/RX-PAD/<model_name>/")
    logger.info(f"{'='*80}")
    
    return all_summaries


def main():
    """Main entry point for RX-PAD benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='RX-PAD French Medical Prescription OCR/VLM Benchmark'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=[
            'azure_intelligence',
            'mistral_document_ai',
            'gpt-5-mini',
            'gpt-5-nano',
            'claude_sonnet',
        ],
        help='Models to test (default: all models)'
    )
    parser.add_argument(
        '--phases', '-p',
        nargs='+',
        type=int,
        default=[1, 2, 3],
        help='Phases to run (default: 1 2 3)'
    )
    parser.add_argument(
        '--sample-limit', '-n',
        type=int,
        default=None,
        help='Maximum number of samples to process (default: all)'
    )
    parser.add_argument(
        '--phase-3-letter', '-l',
        default='a',
        help='Phase 3 variant letter (default: a)'
    )
    
    args = parser.parse_args()
    
    run_rxpad_benchmark(
        models=args.models,
        phases=args.phases,
        sample_limit=args.sample_limit,
        phase_3_letter=args.phase_3_letter
    )


if __name__ == '__main__':
    main()
