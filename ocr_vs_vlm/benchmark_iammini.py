"""
IAM_mini Fairness Benchmark: Separate evaluation on handwritten vs printed images

This benchmark evaluates VLM models on the IAM_mini dataset with fairness controls:
1. Handwritten Phase: Models read only the handwritten portion (fair evaluation)
2. Printed Phase: Models read only the printed reference text (oracle evaluation)

This prevents models from "cheating" by reading the printed reference text during
fair handwritten evaluation.

Results structure:
- ocr_vs_vlm/results/IAM_mini/<date>/<model>/handwritten_phase_X.csv
- ocr_vs_vlm/results/IAM_mini/<date>/<model>/printed_phase_X.csv
"""

import json
import logging
import time
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

from tqdm import tqdm

from ocr_vs_vlm.dataset_loaders import DatasetRegistry, validate_dataset
from ocr_vs_vlm.unified_model_api import UnifiedModelAPI, ModelRegistry
from ocr_vs_vlm import prompts as prompt_module

# Logs will be configured per run with the results directory
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _configure_logging(results_dir: Path):
    """Configure logging to write to results directory."""
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    benchmark_log = logging.FileHandler(results_dir / 'benchmark_iammini.log')
    benchmark_log.setLevel(logging.DEBUG)
    benchmark_log.setFormatter(formatter)
    logger.addHandler(benchmark_log)


@dataclass
class BenchmarkResult:
    """Result from a single sample evaluation."""
    
    sample_id: str
    image_path: str
    dataset: str
    model: str
    phase: int
    image_type: str  # 'handwritten' or 'printed'
    ground_truth: str
    prediction: Optional[str] = None
    prompt: Optional[str] = None
    
    # Metadata
    inference_time_ms: float = 0.0
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV serialization."""
        result = asdict(self)
        # Reorder for readability
        ordered = {}
        for key in ['sample_id', 'image_path', 'dataset', 'model', 'phase', 'image_type',
                    'ground_truth', 'prediction', 'prompt', 'inference_time_ms', 'tokens_used',
                    'error', 'timestamp']:
            if key in result:
                ordered[key] = result[key]
        return ordered


class IAMMiniVLMBenchmark:
    """Benchmark VLM models on IAM_mini with fairness controls."""
    
    def __init__(self, models: List[str], phases: List[int] = None, sample_limit: Optional[int] = None):
        """
        Initialize IAM_mini benchmark.
        
        Args:
            models: List of model names (e.g., ['gpt-5-mini', 'gpt-5-nano'])
            phases: List of phases to run (default [2, 3])
            sample_limit: Max samples to process per image type
        """
        self.models = models
        self.phases = phases or [2, 3]
        self.sample_limit = sample_limit
        
        # Create results directory with date
        date_str = datetime.now().strftime("%Y%m%d")
        self.results_dir = Path(__file__).parent / "results" / "IAM_mini" / date_str
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging to this results directory
        _configure_logging(self.results_dir)
        
        # Initialize API
        self.api = UnifiedModelAPI()
        
        # Initialize prompts directory
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"IAMMiniVLMBenchmark initialized")
        logger.info(f"Models: {models}")
        logger.info(f"Phases: {self.phases}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def run(self) -> Dict:
        """
        Run full benchmark (handwritten then printed for each model).
        
        Returns:
            Summary of execution
        """
        start_time = time.time()
        summary = {
            'start_time': datetime.now().isoformat(),
            'models': self.models,
            'phases': self.phases,
            'by_model': {},
        }
        
        # Load IAM_mini dataset (will use handwritten images)
        dataset_root = Path(__file__).parent / "datasets_subsets"
        
        try:
            # Validate dataset
            validation = validate_dataset('IAM_mini', str(dataset_root))
            if not validation['valid']:
                logger.error(f"Dataset validation failed: {validation}")
                return summary
            
            logger.info(f"Dataset validated: {validation['checks']}")
            
            # Load dataset
            dataset = DatasetRegistry.get_dataset(
                'IAM_mini',
                str(dataset_root),
                sample_limit=self.sample_limit
            )
            logger.info(f"Loaded {len(dataset)} samples from IAM_mini")
            
            # Run each model
            for model_name in self.models:
                logger.info(f"\n{'='*70}")
                logger.info(f"RUNNING MODEL: {model_name.upper()}")
                logger.info(f"{'='*70}\n")
                
                try:
                    model_summary = self._run_model(model_name, dataset)
                    summary['by_model'][model_name] = model_summary
                except Exception as e:
                    logger.error(f"Error with model {model_name}: {e}")
                    summary['by_model'][model_name] = {'error': str(e)}
        
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return summary
        
        summary['end_time'] = datetime.now().isoformat()
        summary['total_time_seconds'] = time.time() - start_time
        
        # Save execution summary
        summary_file = self.results_dir / "execution_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved execution summary to {summary_file}")
        
        return summary
    
    def _run_model(self, model_name: str, dataset) -> Dict:
        """
        Run all phases for one model on both handwritten and printed images.
        
        Args:
            model_name: Model name
            dataset: Loaded dataset
        
        Returns:
            Model-level summary
        """
        model_dir = self.results_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_summary = {
            'name': model_name,
            'handwritten': {},
            'printed': {},
        }
        
        # Evaluate on handwritten images first
        logger.info(f"\n{'─'*70}")
        logger.info(f"Evaluating on HANDWRITTEN images (fair evaluation)")
        logger.info(f"{'─'*70}\n")
        
        try:
            handwritten_results = self._evaluate_on_images(
                model_name, dataset, image_type='handwritten'
            )
            model_summary['handwritten'] = {
                'status': 'completed',
                'samples_processed': sum(
                    len(results) for results in handwritten_results.values()
                )
            }
            logger.info(f"Handwritten evaluation completed")
        except Exception as e:
            logger.error(f"Error evaluating handwritten: {e}")
            model_summary['handwritten'] = {'status': 'failed', 'error': str(e)}
        
        # Evaluate on printed images
        logger.info(f"\n{'─'*70}")
        logger.info(f"Evaluating on PRINTED images (oracle evaluation)")
        logger.info(f"{'─'*70}\n")
        
        try:
            printed_results = self._evaluate_on_images(
                model_name, dataset, image_type='printed'
            )
            model_summary['printed'] = {
                'status': 'completed',
                'samples_processed': sum(
                    len(results) for results in printed_results.values()
                )
            }
            logger.info(f"Printed evaluation completed")
        except Exception as e:
            logger.error(f"Error evaluating printed: {e}")
            model_summary['printed'] = {'status': 'failed', 'error': str(e)}
        
        return model_summary
    
    def _evaluate_on_images(self, model_name: str, dataset, image_type: str) -> Dict[int, List[BenchmarkResult]]:
        """
        Evaluate model on either handwritten or printed images across all phases.
        
        Args:
            model_name: Model name
            dataset: Dataset with samples
            image_type: 'handwritten' or 'printed'
        
        Returns:
            Dict mapping phase -> list of results
        """
        model_dir = self.results_dir / model_name
        all_phase_results = {}
        
        for phase in self.phases:
            logger.info(f"  Phase {phase} ({image_type} images)...")
            
            results = []
            results_file = model_dir / f"{image_type}_phase_{phase}_results.csv"
            
            # Load existing results if resuming
            existing_results = self._load_existing_results(results_file)
            processed_ids = {(r.sample_id, r.image_type) for r in existing_results}
            csv_headers_written = len(existing_results) > 0
            
            samples_to_process = [
                s for s in dataset if (s.sample_id, image_type) not in processed_ids
            ]
            
            logger.info(f"    Processing {len(samples_to_process)} samples")
            
            # Process samples
            with tqdm(total=len(samples_to_process), desc=f"Phase {phase}",
                     unit="sample", leave=True, disable=False) as pbar:
                for sample in samples_to_process:
                    try:
                        # Get the appropriate image path based on image_type
                        image_path = self._get_image_path(sample, image_type)
                        
                        # Process sample
                        result = self._process_sample(
                            sample, model_name, phase, image_type, image_path
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
                            logger.info(f"Checkpoint saved ({len(all_results)} total)")
                            csv_headers_written = True
                            existing_results = all_results
                            results = []
                        
                        pbar.update(1)
                    
                    except Exception as e:
                        logger.warning(f"Failed {sample.sample_id}: {e}")
                        error_result = BenchmarkResult(
                            sample_id=sample.sample_id,
                            image_path=str(self._get_image_path(sample, image_type)),
                            dataset='IAM_mini',
                            model=model_name,
                            phase=phase,
                            image_type=image_type,
                            ground_truth=sample.ground_truth,
                            error=str(e),
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
                logger.info(f"Phase {phase} saved ({len(all_results)} total)")
            
            all_phase_results[phase] = existing_results + results
        
        return all_phase_results
    
    def _get_image_path(self, sample, image_type: str) -> Path:
        """
        Get the appropriate image path for the given image type.
        
        Args:
            sample: Sample object
            image_type: 'handwritten' or 'printed'
        
        Returns:
            Path to the image file
        """
        # The sample object should have metadata with paths to both image types
        if image_type == 'handwritten':
            # Use handwritten image if available, otherwise full image
            if hasattr(sample, 'handwritten_path') and sample.handwritten_path:
                return Path(sample.handwritten_path)
            elif hasattr(sample, 'metadata') and 'handwritten_path' in sample.metadata:
                return Path(sample.metadata['handwritten_path'])
        elif image_type == 'printed':
            # Use printed image if available
            if hasattr(sample, 'printed_path') and sample.printed_path:
                return Path(sample.printed_path)
            elif hasattr(sample, 'metadata') and 'printed_path' in sample.metadata:
                return Path(sample.metadata['printed_path'])
        
        # Fallback to image_path
        return Path(sample.image_path)
    
    def _process_sample(
        self, sample, model_name: str, phase: int,
        image_type: str, image_path: Path
    ) -> BenchmarkResult:
        """
        Process single sample through VLM API.
        
        Args:
            sample: Sample object
            model_name: Model name
            phase: Phase number
            image_type: 'handwritten' or 'printed'
            image_path: Path to image to use
        
        Returns:
            BenchmarkResult with prediction
        """
        start_time = time.time()
        prediction = None
        prompt: Optional[str] = None
        error = None
        
        try:
            # Validate model is a VLM
            if model_name not in ModelRegistry.list_vlm_models():
                raise ValueError(f"{model_name} is not a VLM model")
            
            # Build prompt based on phase
            if phase == 2:
                # Phase 2: Generic prompt
                prompt = prompt_module.get_phase_2_prompt()
            elif phase == 3:
                # Phase 3: Context-aware prompt
                prompt = prompt_module.get_phase_3_prompt(
                    sample, 'IAM_mini', 'a'  # Use 'a' as default suffix
                )
            else:
                raise ValueError(f"Unknown phase: {phase}")
            
            # Call VLM model
            response = self.api.process(str(image_path), model=model_name, query=prompt)
            
            if response.error:
                raise Exception(response.error)
            
            prediction = response.content
        
        except Exception as e:
            logger.warning(f"API call failed for {sample.sample_id}: {e}")
            error = str(e)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return BenchmarkResult(
            sample_id=sample.sample_id,
            image_path=str(image_path),
            dataset='IAM_mini',
            model=model_name,
            phase=phase,
            image_type=image_type,
            ground_truth=sample.ground_truth,
            prediction=prediction,
            prompt=prompt,
            inference_time_ms=elapsed_ms,
            error=error,
            timestamp=datetime.now().isoformat()
        )
    
    def _save_results_csv(self, results_file: Path, results: List[BenchmarkResult], write_headers: bool = True):
        """Save results to CSV file."""
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            'sample_id', 'image_path', 'dataset', 'model', 'phase', 'image_type',
            'ground_truth', 'prediction', 'prompt', 'inference_time_ms', 'tokens_used',
            'error', 'timestamp'
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
                    row['inference_time_ms'] = float(row['inference_time_ms']) if row.get('inference_time_ms') else 0.0
                    row['tokens_used'] = int(row['tokens_used']) if row.get('tokens_used') and row['tokens_used'] != 'None' else None
                    row['phase'] = int(row['phase'])
                    results.append(BenchmarkResult(**row))
            
            return results
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")
            return []


if __name__ == '__main__':
    """
    Run IAM_mini benchmark on gpt-5-mini and gpt-5-nano.
    
    Evaluates each model on:
    1. Handwritten images (fair evaluation - no reference text visible)
    2. Printed images (oracle evaluation - reference text only)
    
    Prevents cheating by keeping evaluations completely separate.
    """
    logger.info("=" * 80)
    logger.info("IAM_mini Fairness Benchmark: Handwritten vs Printed Images")
    logger.info("=" * 80)
    
    models = ['gpt-5-mini', 'gpt-5-nano']
    
    benchmark = IAMMiniVLMBenchmark(
        models=models,
        phases=[2, 3],
        sample_limit=None  # Use all samples
    )
    
    summary = benchmark.run()
    
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(json.dumps(summary, indent=2))
    logger.info(f"\nResults saved to: {benchmark.results_dir}")
