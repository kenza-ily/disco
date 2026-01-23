#!/usr/bin/env python3
"""
VOC2007 Medical Lab Reports Benchmark

This script runs the OCR/VLM benchmark on the VOC2007 dataset containing
Simplified Chinese medical laboratory reports.

Dataset Structure:
- Images: datasets/VOC2007/JPEGImages/*.jpg
- Ground Truth: datasets/VOC2007/labels_src.json (Chinese Unicode text)

The benchmark evaluates:
- Phase 2: VLM baseline with generic Chinese-aware prompt
- Phase 3: VLM with intermediate context (language + document type)
- Phase 4: VLM with detailed context-aware prompt for medical lab reports

All prompts explicitly request Unicode Chinese output to match ground truth format.
"""

import json
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_vs_vlm.benchmarks.benchmark import BenchmarkRunner, create_benchmark_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_voc2007_benchmark(
    models: list | None = None,
    phases: list | None = None,
    sample_limit: int | None = None,
    phase_3_letter: str = 'a'
):
    """
    Run VOC2007 Chinese Medical Lab Reports benchmark.
    
    Args:
        models: List of VLM models to test (default: GPT-4o-mini)
        phases: List of phases to run (default: [2, 3, 4])
        sample_limit: Max samples to process (default: None = all)
        phase_3_letter: Phase variant letter (default: 'a')
    
    Returns:
        Execution summary dict
    """
    if models is None:
        models = ['gpt-4o-mini']  # Default to GPT-4o-mini
    
    if phases is None:
        phases = [2, 3, 4]  # Phase 2 = VLM baseline, Phase 3 = Intermediate, Phase 4 = Detailed
    
    logger.info("=" * 80)
    logger.info("VOC2007 Medical Lab Reports Benchmark (Simplified Chinese)")
    logger.info("=" * 80)
    logger.info(f"Models: {models}")
    logger.info(f"Phases: {phases}")
    logger.info(f"Sample limit: {sample_limit or 'All samples'}")
    logger.info(f"Phase 3 letter: {phase_3_letter}")
    logger.info("=" * 80)
    
    # Print prompt information
    logger.info("\nPrompt Configuration:")
    logger.info("-" * 40)
    logger.info("Phase 2 Prompt (Generic):")
    logger.info("  Extract all text from this document image.")
    logger.info("  Output text in Simplified Chinese Unicode characters (UTF-8).")
    logger.info("")
    logger.info("Phase 3 Prompt (Intermediate):")
    logger.info("  Language: Simplified Chinese (简体中文)")
    logger.info("  Document Type: Medical Lab Report (医学检验报告)")
    logger.info("  Requests Unicode output with table structure")
    logger.info("")
    logger.info("Phase 4 Prompt (Detailed):")
    logger.info("  Full context-aware prompt for Medical Laboratory Reports")
    logger.info("  Lists common fields (报告时间, 姓名, 结果, etc.)")
    logger.info("  Detailed instructions for extraction")
    logger.info("-" * 40)
    
    all_summaries = {}
    
    for model_name in models:
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING MODEL: {model_name.upper()}")
        logger.info(f"{'='*80}\n")
        
        # Create per-model results directory
        model_results_dir = f"results/VOC2007/{model_name}"
        
        # Create configuration for this model
        config = create_benchmark_config(
            datasets=['VOC2007'],
            models=[model_name],
            phases=phases,
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
    logger.info(f"Results saved to: results/VOC2007/<model_name>/")
    logger.info(f"{'='*80}")
    
    return all_summaries


def main():
    """Main entry point for VOC2007 benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='VOC2007 Medical Lab Reports OCR/VLM Benchmark (Simplified Chinese)'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=['gpt-4o-mini'],
        help='VLM models to test (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--phases', '-p',
        nargs='+',
        type=int,
        default=[2, 3, 4],
        help='Phases to run (default: 2 3 4)'
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
    
    run_voc2007_benchmark(
        models=args.models,
        phases=args.phases,
        sample_limit=args.sample_limit,
        phase_3_letter=args.phase_3_letter
    )


if __name__ == '__main__':
    main()
