"""
IAM_mini Comprehensive Benchmark: Fair OCR/VLM Evaluation

Evaluates OCR and VLM models on the IAM_mini dataset with fairness controls:
1. Handwritten Phase: Models read only the handwritten portion (fair evaluation)
2. Printed Phase: Models read only the printed reference text (oracle evaluation)

This prevents models from "cheating" by reading the printed reference text during
fair handwritten evaluation.

Phase Configuration:
- OCR models (azure_intelligence, donut, mistral_document_ai): Phase 1 only
- VLM models (gpt-5-mini, gpt-5-nano): Phases 2 & 3

Results structure:
- ocr_vs_vlm/results/IAM_mini/<model>/phase_X_results.csv
"""

import json
import logging
import sys
import time
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

from tqdm import tqdm

from ocr_vs_vlm.datasets.dataset_loaders import DatasetRegistry, validate_dataset
from ocr_vs_vlm.unified_model_api import UnifiedModelAPI, ModelRegistry, ModelType
from ocr_vs_vlm import prompts as prompt_module

# Logs will be configured per run with the results directory
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _configure_logging(results_dir: Path):
    """Configure logging to write to results directory."""
    logger.handlers.clear()
    
    # Simple formatter for console (more readable)
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Detailed formatter for file
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler - INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler - DEBUG and above
    benchmark_log = logging.FileHandler(results_dir / 'benchmark_iammini.log')
    benchmark_log.setLevel(logging.DEBUG)
    benchmark_log.setFormatter(file_formatter)
    logger.addHandler(benchmark_log)


@dataclass
class BenchmarkResult:
    """Result from a single sample evaluation.
    
    For IAM_mini:
    - ground_truth: Text extracted from printed.png (the reference)
    - prediction: Text extracted from handwritten.png (what we're evaluating)
    """
    
    sample_id: str
    image_path: str  # Path to handwritten image (input for prediction)
    printed_image_path: str  # Path to printed image (input for ground truth)
    dataset: str
    model: str
    phase: int
    
    # Ground truth from printed image, prediction from handwritten image
    ground_truth: Optional[str] = None  # Extracted from printed.png
    prediction: Optional[str] = None  # Extracted from handwritten.png
    
    # Prompts used (for VLM models)
    prompt: Optional[str] = None
    
    # Inference metrics
    ground_truth_inference_time_ms: float = 0.0  # Time to extract from printed
    prediction_inference_time_ms: float = 0.0  # Time to extract from handwritten
    tokens_used: Optional[int] = None
    
    # Error tracking
    ground_truth_error: Optional[str] = None
    prediction_error: Optional[str] = None
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV serialization."""
        result = asdict(self)
        # Reorder for readability
        ordered = {}
        for key in ['sample_id', 'image_path', 'printed_image_path', 'dataset', 'model', 'phase',
                    'ground_truth', 'prediction', 'prompt',
                    'ground_truth_inference_time_ms', 'prediction_inference_time_ms',
                    'tokens_used', 'ground_truth_error', 'prediction_error', 'timestamp']:
            if key in result:
                ordered[key] = result[key]
        return ordered


class IAMMiniVLMBenchmark:
    """Comprehensive benchmark for OCR and VLM models on IAM_mini with fairness controls."""
    
    def __init__(self, models: List[str], sample_limit: Optional[int] = None):
        """
        Initialize IAM_mini benchmark.
        
        Args:
            models: List of model names to evaluate
            sample_limit: Max samples to process per dataset
        """
        self.models = models
        self.sample_limit = sample_limit
        
        # Create results directory
        self.results_dir = Path(__file__).parent / "results" / "IAM_mini"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Store log file path
        self.log_file = self.results_dir / "benchmark_iammini.log"
        
        # Configure logging to this results directory
        _configure_logging(self.results_dir)
        
        # Initialize API
        self.api = UnifiedModelAPI()
        
        logger.info(f"=" * 70)
        logger.info(f"IAMMini Benchmark Initialized")
        logger.info(f"=" * 70)
        logger.info(f"Models to evaluate: {models}")
        logger.info(f"Sample limit: {sample_limit if sample_limit else 'All samples'}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Log file: {self.log_file}")
    
    def run(self) -> Dict:
        """
        Run full benchmark across all models with appropriate phases.
        
        Returns:
            Summary of execution
        """
        start_time = time.time()
        logger.info(f"\n{'#'*70}")
        logger.info(f"# BENCHMARK START")
        logger.info(f"{'#'*70}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Log file: {self.log_file}")
        logger.info(f"Models to evaluate: {self.models}")
        logger.info(f"Sample limit: {self.sample_limit}")
        logger.info(f"{'#'*70}\n")
        
        summary = {
            'start_time': datetime.now().isoformat(),
            'models': self.models,
            'by_model': {},
        }
        
        # Load IAM_mini dataset
        dataset_root = Path(__file__).parent / "datasets_subsets"
        
        try:
            # Validate dataset
            logger.info("Validating dataset...")
            validation = validate_dataset('IAM_mini', str(dataset_root))
            if not validation['valid']:
                logger.error(f"Dataset validation failed: {validation}")
                return summary
            
            logger.info(f"✓ Dataset validated")
            
            # Load dataset
            logger.info("Loading dataset...")
            dataset = DatasetRegistry.get_dataset(
                'IAM_mini',
                str(dataset_root),
                sample_limit=self.sample_limit
            )
            logger.info(f"✓ Loaded {len(dataset)} samples from IAM_mini")
            
            # Show sample distribution
            logger.info(f"\nSample distribution:")
            hw_samples = sum(1 for s in dataset if 'handwritten' in str(s.image_path).lower())
            pr_samples = sum(1 for s in dataset if 'printed' in str(s.image_path).lower())
            logger.info(f"  - Handwritten images: {hw_samples}")
            logger.info(f"  - Printed images: {pr_samples}")
            
            # Run each model
            logger.info(f"\nStarting evaluation of {len(self.models)} models...\n")
            
            with tqdm(total=len(self.models), desc="Models", unit="model", leave=True,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                     file=sys.stdout) as model_pbar:
                for model_name in self.models:
                    try:
                        model_summary = self._run_model(model_name, dataset)
                        summary['by_model'][model_name] = model_summary
                        model_pbar.update(1)
                    except Exception as e:
                        logger.error(f"✗ Error with model {model_name}: {e}")
                        summary['by_model'][model_name] = {'error': str(e)}
                        model_pbar.update(1)
        
        except Exception as e:
            logger.error(f"✗ Failed to load dataset: {e}")
            return summary
        
        summary['end_time'] = datetime.now().isoformat()
        summary['total_time_seconds'] = time.time() - start_time
        
        # Final summary
        logger.info(f"\n{'#'*70}")
        logger.info(f"# BENCHMARK COMPLETE")
        logger.info(f"{'#'*70}")
        logger.info(f"Total time: {summary['total_time_seconds']:.1f} seconds")
        logger.info(f"Results saved to: {self.results_dir}")
        logger.info(f"{'#'*70}\n")
        
        # Save execution summary
        summary_file = self.results_dir / "execution_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved execution summary to {summary_file}")
        
        return summary
    
    def _run_model(self, model_name: str, dataset) -> Dict:
        """
        Run appropriate phases for one model based on its type.
        
        - OCR models: Phase 1 only (basic extraction, no prompt)
        - VLM models: Phases 2 & 3 (with prompts)
        
        Args:
            model_name: Model name
            dataset: Loaded dataset
        
        Returns:
            Model-level summary
        """
        model_dir = self.results_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"EVALUATING MODEL: {model_name}")
        logger.info(f"{'='*70}")
        
        model_type = ModelRegistry.get_model_type(model_name)
        logger.info(f"Model type: {model_type.name}")
        
        # Determine phases based on model type
        if model_type == ModelType.OCR:
            # OCR models: Phase 1 only (basic extraction)
            phases = [1]
        else:
            # VLM models: Phases 2 & 3 (with prompts)
            phases = [2, 3]
        logger.info(f"Phases to run: {phases}")
        
        model_summary = {
            'name': model_name,
            'type': str(model_type),
            'phases_completed': {}
        }
        
        # Run each phase
        for phase in phases:
            logger.info(f"\n{'─'*70}")
            logger.info(f"Phase {phase}")
            logger.info(f"{'─'*70}")
            
            try:
                phase_results = self._run_phase(model_name, phase, dataset)
                model_summary['phases_completed'][phase] = {
                    'status': 'completed',
                    'samples_processed': len(phase_results)
                }
                logger.info(f"✓ Phase {phase} completed: {len(phase_results)} samples processed")
            except Exception as e:
                logger.error(f"✗ Phase {phase} failed: {e}")
                model_summary['phases_completed'][phase] = {'status': 'failed', 'error': str(e)}
        
        return model_summary
    
    def _run_phase(self, model_name: str, phase: int, dataset) -> List[BenchmarkResult]:
        """
        Run single phase across all samples.
        
        For each sample:
        1. Extract GROUND TRUTH from printed image (clean typed text)
        2. Extract PREDICTION from handwritten image (what we're evaluating)
        
        This allows fair comparison:
        - Ground truth: what the model CAN read (printed text)
        - Prediction: what the model TRIES to read (handwritten text)
        
        Args:
            model_name: Model name (OCR or VLM)
            phase: Phase number (1 for basic extraction)
            dataset: Loaded dataset
        
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        # Add timestamp to filename to preserve previous runs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / model_name / f"phase_{phase}_results_{timestamp}.csv"
        
        # Load existing results if resuming
        existing_results = self._load_existing_results(results_file)
        processed_ids = {r.sample_id for r in existing_results}
        csv_headers_written = len(existing_results) > 0
        
        samples_to_process = [s for s in dataset if s.sample_id not in processed_ids]
        
        logger.info(f"[Phase {phase}] Total samples: {len(dataset)}")
        logger.info(f"[Phase {phase}] Already processed: {len(processed_ids)}")
        logger.info(f"[Phase {phase}] Samples to process: {len(samples_to_process)}")
        
        if not samples_to_process:
            logger.info(f"[Phase {phase}] ✓ All samples already processed, skipping")
            return existing_results
        
        # Process samples with detailed progress bar
        with tqdm(total=len(samples_to_process), 
                 desc=f"{model_name:20s} Phase {phase}",
                 unit="sample", leave=True, disable=False, 
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                 file=sys.stdout) as pbar:
            for sample in samples_to_process:
                try:
                    # Get image paths
                    hw_image_path = self._get_image_path(sample, 'handwritten')
                    pr_image_path = self._get_image_path(sample, 'printed')
                    
                    # 1. Extract GROUND TRUTH from printed image (use generic prompt)
                    pr_start = time.time()
                    gt_prompt, ground_truth, gt_error = self._call_model(
                        model_name, pr_image_path, phase, sample, is_ground_truth=True
                    )
                    pr_time_ms = (time.time() - pr_start) * 1000
                    
                    if ground_truth:
                        logger.debug(f"[{sample.sample_id}] Ground truth (printed): {len(ground_truth)} chars in {pr_time_ms:.1f}ms")
                    elif gt_error:
                        logger.debug(f"[{sample.sample_id}] Ground truth ERROR: {gt_error[:60]}")
                    
                    # 2. Extract PREDICTION from handwritten image (use phase-specific prompt)
                    hw_start = time.time()
                    pred_prompt, prediction, pred_error = self._call_model(
                        model_name, hw_image_path, phase, sample, is_ground_truth=False
                    )
                    hw_time_ms = (time.time() - hw_start) * 1000
                    
                    if prediction:
                        logger.debug(f"[{sample.sample_id}] Prediction (handwritten): {len(prediction)} chars in {hw_time_ms:.1f}ms")
                    elif pred_error:
                        logger.debug(f"[{sample.sample_id}] Prediction ERROR: {pred_error[:60]}")
                    
                    # Create result: ground_truth from printed, prediction from handwritten
                    result = BenchmarkResult(
                        sample_id=sample.sample_id,
                        image_path=str(hw_image_path),
                        printed_image_path=str(pr_image_path),
                        dataset='IAM_mini',
                        model=model_name,
                        phase=phase,
                        ground_truth=ground_truth,  # From printed image
                        prediction=prediction,  # From handwritten image
                        prompt=pred_prompt,  # Prompt used for prediction (phase-specific)
                        ground_truth_inference_time_ms=pr_time_ms,
                        prediction_inference_time_ms=hw_time_ms,
                        ground_truth_error=gt_error,
                        prediction_error=pred_error,
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(result)
                    
                    # Save incrementally every 50 samples
                    if len(results) % 50 == 0:
                        all_results = existing_results + results
                        self._save_results_csv(
                            results_file,
                            all_results,
                            write_headers=(not csv_headers_written)
                        )
                        logger.info(f"[Phase {phase}] ✓ Checkpoint saved ({len(all_results)} total)")
                        csv_headers_written = True
                        existing_results = all_results
                        results = []
                    
                    pbar.update(1)
                
                except Exception as e:
                    logger.warning(f"[Phase {phase}] ✗ Failed {sample.sample_id}: {str(e)[:100]}")
                    hw_image_path = self._get_image_path(sample, 'handwritten')
                    pr_image_path = self._get_image_path(sample, 'printed')
                    error_result = BenchmarkResult(
                        sample_id=sample.sample_id,
                        image_path=str(hw_image_path),
                        printed_image_path=str(pr_image_path),
                        dataset='IAM_mini',
                        model=model_name,
                        phase=phase,
                        prediction_error=str(e),
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(error_result)
                    pbar.update(1)
        
        # Final save
        if results:
            all_results = existing_results + results
            self._save_results_csv(
                results_file,
                all_results,
                write_headers=(not csv_headers_written)
            )
            logger.info(f"[Phase {phase}] ✓ Final save complete ({len(all_results)} total samples)")
        
        return existing_results + results
    
    def _get_image_path(self, sample, image_type: str) -> Path:
        """
        Get image path for the given type, constructing printed path from handwritten.
        
        Args:
            sample: Sample object
            image_type: 'handwritten' or 'printed'
        
        Returns:
            Path to the image file
        """
        base_path = Path(sample.image_path)
        
        if image_type == 'printed':
            # Replace handwritten.png with printed.png
            return base_path.parent / "printed.png"
        else:
            # Return handwritten as-is
            return base_path
    
    def _call_model(
        self, model_name: str, image_path: Path, phase: int, sample,
        is_ground_truth: bool = False
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Call model on a single image (OCR or VLM depending on model type).
        
        For ground truth extraction (printed image):
        - Always use generic prompt (Phase 2 style) to get clean baseline
        
        For prediction extraction (handwritten image):
        - Phase 1: Basic extraction prompt
        - Phase 2: Generic prompt
        - Phase 3: Context-aware prompt with dataset hints
        
        Args:
            model_name: Model name
            image_path: Path to image
            phase: Phase number
            sample: Sample object
            is_ground_truth: If True, use generic prompt (for printed image)
        
        Returns:
            Tuple of (prompt, output_text, error)
        """
        prompt: Optional[str] = None
        prediction = None
        error = None
        
        try:
            model_type = ModelRegistry.get_model_type(model_name)
            
            if model_type == ModelType.OCR:
                # OCR models: call without query/prompt
                response = self.api.process(str(image_path), model=model_name)
            else:
                # VLM models: build prompt
                if is_ground_truth:
                    # Ground truth always uses generic prompt (fair baseline)
                    prompt = prompt_module.get_phase_2_prompt()
                elif phase == 1:
                    # Phase 1: Basic extraction prompt
                    prompt = "Extract all text from this document image"
                elif phase == 2:
                    # Phase 2: Generic prompt
                    prompt = prompt_module.get_phase_2_prompt()
                elif phase == 3:
                    # Phase 3: Context-aware prompt for prediction only
                    prompt = prompt_module.get_phase_3_prompt(sample, 'IAM_mini', 'a')
                else:
                    raise ValueError(f"Unknown phase: {phase}")
                
                # Call VLM model with prompt
                response = self.api.process(str(image_path), model=model_name, query=prompt)
            
            if response.error:
                raise Exception(response.error)
            
            prediction = response.content
        
        except Exception as e:
            logger.warning(f"API call failed for {image_path.name}: {e}")
            error = str(e)
        
        return prompt, prediction, error
    
    def _save_results_csv(self, results_file: Path, results: List[BenchmarkResult], write_headers: bool = True):
        """Save results to CSV file."""
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            'sample_id', 'image_path', 'printed_image_path', 'dataset', 'model', 'phase',
            'ground_truth', 'prediction', 'prompt',
            'ground_truth_inference_time_ms', 'prediction_inference_time_ms',
            'tokens_used', 'ground_truth_error', 'prediction_error', 'timestamp'
        ]
        
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_headers:
                writer.writeheader()
            
            for result in results:
                writer.writerow(result.to_dict())
    
    def _load_existing_results(self, results_file: Path) -> List[BenchmarkResult]:
        """Load existing results if file exists (for resumability)."""
        if not results_file.exists():
            return []
        
        try:
            results = []
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    row['phase'] = int(row['phase'])
                    row['ground_truth_inference_time_ms'] = float(row.get('ground_truth_inference_time_ms', 0.0))
                    row['prediction_inference_time_ms'] = float(row.get('prediction_inference_time_ms', 0.0))
                    row['tokens_used'] = int(row['tokens_used']) if row.get('tokens_used') and row['tokens_used'] != 'None' else None
                    results.append(BenchmarkResult(**row))
            
            return results
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")
            return []


def main():
    """
    Run IAM_mini benchmark on VLM models with Phases 2 & 3.
    
    Phase 2 & 3 Evaluation:
    - Ground truth: VLM output on printed.png (clean text)
    - Prediction: VLM output on handwritten.png (handwritten text)
    - Phase 2: Generic prompt
    - Phase 3: Context-aware prompt with dataset hints
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='IAM_mini Handwriting OCR/VLM Benchmark')
    parser.add_argument(
        '--phases', '-p',
        nargs='+',
        default=["2", "3"],
        help='Phases to run (default: 2 3). Phase 2=generic, Phase 3=context-aware'
    )
    parser.add_argument(
        '--sample-limit', '-n',
        type=int,
        default=None,
        help='Maximum samples to process (default: all 500)'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=['gpt-5-mini', 'gpt-5-nano'],
        help='VLM models to test (default: gpt-5-mini, gpt-5-nano)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("IAM_mini Benchmark: VLM Evaluation")
    logger.info("=" * 80)
    logger.info(f"Models: {args.models}")
    logger.info(f"Phases: {args.phases}")
    logger.info(f"Sample limit: {args.sample_limit if args.sample_limit else 'All'}")
    
    benchmark = IAMMiniVLMBenchmark(
        models=args.models,
        sample_limit=args.sample_limit
    )
    
    summary = benchmark.run()
    
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(json.dumps(summary, indent=2))
    logger.info(f"\nResults saved to: {benchmark.results_dir}")


if __name__ == '__main__':
    main()
