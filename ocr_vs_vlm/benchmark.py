"""
OCR vs VLM Benchmarking Pipeline

Orchestrates evaluation across three phases:
1. OCR Baseline: Pure OCR models
2. VLM Baseline: VLMs with generic prompts
3. VLM + Context: VLMs with task-aware prompts

Saves results incrementally for resumability and debugging.
"""

import json
import logging
import time
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import sys

from tqdm import tqdm

from .dataset_loaders import DatasetRegistry, validate_dataset, load_image
from .unified_model_api import UnifiedModelAPI, ModelRegistry
from . import prompts as prompt_module

# Create logs directory
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging with multiple handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

benchmark_log = logging.FileHandler(LOGS_DIR / 'benchmark.log')
benchmark_log.setLevel(logging.DEBUG)
benchmark_log.setFormatter(formatter)
logger.addHandler(benchmark_log)

benchmark_run = logging.FileHandler(LOGS_DIR / 'benchmark_run.txt')
benchmark_run.setLevel(logging.INFO)
benchmark_run.setFormatter(formatter)
logger.addHandler(benchmark_run)

benchmark_output = logging.FileHandler(LOGS_DIR / 'benchmark_output.log')
benchmark_output.setLevel(logging.INFO)
benchmark_output.setFormatter(formatter)
logger.addHandler(benchmark_output)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    
    # Dataset and model selection
    datasets: List[str]  # ['IAM', 'ICDAR', 'PubLayNet']
    models: List[str]  # OCR + VLM models to test
    
    # Phase control
    phases: List[int]  # [1, 2, 3] which phases to run
    phase_3_letter: Optional[str] = None  # Letter suffix for phase 3 (a, b, c, etc.)
    
    # Sample control
    sample_limit: Optional[int] = None  # Max samples per dataset
    batch_size: int = 50  # Save results every this many samples
    
    # Output paths
    results_dir: str = "results"
    checkpoint_file: str = "checkpoint.json"
    
    # API settings
    timeout_seconds: int = 60  # Per-API-call timeout
    retry_failed: bool = True
    max_retries: int = 2
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'BenchmarkConfig':
        """Create config from dictionary."""
        # Extract phase_3_letter if present, set default to None
        phase_3_letter = config_dict.pop('phase_3_letter', None)
        config = cls(**config_dict)
        config.phase_3_letter = phase_3_letter
        return config
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        # Note: phase_3_letter is not included in asdict if it's a property
        if hasattr(self, 'phase_3_letter'):
            config_dict['phase_3_letter'] = self.phase_3_letter
        return config_dict


@dataclass
class BenchmarkResult:
    """Result from a single sample evaluation."""
    
    sample_id: str
    image_path: str
    dataset: str
    model: str
    phase: int
    ground_truth: str
    prediction: Optional[str] = None
    prompt: Optional[str] = None
    
    # Metadata
    inference_time_ms: float = 0.0
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    timestamp: str = ""
    language: Optional[str] = None  # Language for ICDAR dataset
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization with image_path after sample_id."""
        result = asdict(self)
        # Reorder to put image_path right after sample_id
        ordered = {}
        for key in ['sample_id', 'image_path', 'dataset', 'model', 'phase', 'language', 'ground_truth', 
                    'prediction', 'prompt', 'inference_time_ms', 'tokens_used', 'error', 'timestamp']:
            if key in result:
                ordered[key] = result[key]
        return ordered


class BenchmarkRunner:
    """Main orchestrator for benchmark execution."""
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark runner.
        
        Args:
            config: BenchmarkConfig instance
        """
        self.config = config
        # Resolve results_dir relative to ocr_vs_vlm module
        if Path(config.results_dir).is_absolute():
            self.results_dir = Path(config.results_dir)
        else:
            self.results_dir = Path(__file__).parent / config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize prompts directory
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.results_dir / config.checkpoint_file
        self.checkpoint = self._load_checkpoint()
        
        # Initialize unified model API
        self.api = UnifiedModelAPI()
        
        logger.info(f"BenchmarkRunner initialized")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Prompts directory: {self.prompts_dir}")
        logger.info(f"Config: {config.to_dict()}")
    
    def run_benchmark(self) -> Dict:
        """
        Execute full benchmark across all datasets, models, and phases.
        
        Returns:
            Summary dict with execution statistics
        """
        start_time = time.time()
        execution_summary = {
            'start_time': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'by_dataset': {},
        }
        
        try:
            for dataset_name in self.config.datasets:
                logger.info(f"\n{'='*70}")
                logger.info(f"Processing dataset: {dataset_name}")
                logger.info(f"{'='*70}")
                
                try:
                    dataset_summary = self._run_dataset(dataset_name)
                    execution_summary['by_dataset'][dataset_name] = dataset_summary
                except Exception as e:
                    logger.error(f"Error processing dataset {dataset_name}: {e}")
                    execution_summary['by_dataset'][dataset_name] = {
                        'error': str(e),
                        'status': 'failed'
                    }
            
            execution_summary['end_time'] = datetime.now().isoformat()
            execution_summary['total_time_seconds'] = time.time() - start_time
            
            # Save final execution summary
            summary_file = self.results_dir / "execution_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(execution_summary, f, indent=2)
            logger.info(f"Saved execution summary to {summary_file}")
            
        finally:
            logger.info(f"Benchmark completed. Total time: {time.time() - start_time:.1f}s")
        
        return execution_summary
    
    def _run_dataset(self, dataset_name: str) -> Dict:
        """
        Run benchmark on single dataset across all models/phases.
        
        Args:
            dataset_name: Dataset name (IAM, ICDAR, PubLayNet)
        
        Returns:
            Dataset-level summary
        """
        dataset_start = time.time()
        dataset_summary = {
            'name': dataset_name,
            'status': 'completed',
            'total_samples': 0,
            'by_model': {},
        }
        
        # Validate dataset
        dataset_root = self._get_dataset_root(dataset_name)
        validation = validate_dataset(dataset_name, dataset_root)
        if not validation['valid']:
            logger.error(f"Dataset validation failed: {validation}")
            dataset_summary['status'] = 'validation_failed'
            dataset_summary['error'] = validation.get('error', 'Unknown error')
            return dataset_summary
        
        logger.info(f"Dataset validated. Checks: {validation['checks']}")
        
        # Load dataset
        try:
            dataset = DatasetRegistry.get_dataset(
                dataset_name,
                dataset_root,
                sample_limit=self.config.sample_limit
            )
            dataset_summary['total_samples'] = len(dataset)
            logger.info(f"Loaded {len(dataset)} samples from {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            dataset_summary['status'] = 'load_failed'
            dataset_summary['error'] = str(e)
            return dataset_summary
        
        # Run each model on this dataset
        for model_name in self.config.models:
            logger.info(f"\n  Model: {model_name}")
            
            try:
                model_summary = self._run_model_on_dataset(
                    dataset_name, model_name, dataset
                )
                dataset_summary['by_model'][model_name] = model_summary
            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
                dataset_summary['by_model'][model_name] = {'error': str(e)}
        
        dataset_summary['duration_seconds'] = time.time() - dataset_start
        
        # Save checkpoint after dataset
        self._save_checkpoint({
            'last_completed_dataset': dataset_name,
            'timestamp': datetime.now().isoformat()
        })
        
        return dataset_summary
    
    def _run_model_on_dataset(self, dataset_name: str, model_name: str, dataset) -> Dict:
        """
        Run all phases for one model on one dataset.
        
        Args:
            dataset_name: Dataset name
            model_name: Model name
            dataset: Loaded dataset instance
        
        Returns:
            Model-level summary
        """
        model_summary = {
            'name': model_name,
            'by_phase': {},
        }
        
        for phase in self.config.phases:
            logger.info(f"    Phase {phase}")
            
            try:
                phase_results = self._run_phase(
                    dataset_name, model_name, phase, dataset
                )
                
                model_summary['by_phase'][phase] = {
                    'status': 'completed',
                    'samples_processed': len(phase_results),
                    'results_file': f"{dataset_name}/{model_name}/phase_{phase}_results.json"
                }
            except Exception as e:
                logger.error(f"Error in phase {phase}: {e}")
                model_summary['by_phase'][phase] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return model_summary
    
    def _run_phase(self, dataset_name: str, model_name: str, phase: int, dataset) -> List[BenchmarkResult]:
        """
        Run single phase (OCR baseline, VLM baseline, or VLM+context).
        
        Args:
            dataset_name: Dataset name
            model_name: Model name
            phase: Phase number (1, 2, or 3)
            dataset: Loaded dataset instance
        
        Returns:
            List of BenchmarkResult objects
        """
        results = []
        # Create results filename with phase and phase 3 letter suffix
        phase_file_name = f"phase_{phase}"
        if phase == 3 and self.config.phase_3_letter:
            phase_file_name = f"phase_3{self.config.phase_3_letter}"
        
        output_dir = self.results_dir / dataset_name / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"{phase_file_name}_results.csv"
        
        # Load existing results if resuming
        existing_results = self._load_existing_results(results_file)
        processed_ids = {r.sample_id for r in existing_results}
        csv_headers_written = len(existing_results) > 0  # Track if headers written
        
        samples_to_process = [
            s for s in dataset if s.sample_id not in processed_ids
        ]
        
        logger.info(f"      Processing {len(samples_to_process)} samples (skipping {len(processed_ids)} existing)")
        
        # Process samples with progress bar
        with tqdm(total=len(samples_to_process), desc=f"Phase {phase} - {model_name}", 
                  unit="sample", leave=True, file=sys.stdout, disable=False) as pbar:
            for idx, sample in enumerate(samples_to_process):
                try:
                    # Generate result
                    result = self._process_sample(
                        sample, model_name, phase, dataset_name
                    )
                    results.append(result)
                    
                    # Save incrementally every batch_size samples
                    if (idx + 1) % self.config.batch_size == 0:
                        all_results = existing_results + results
                        self._save_phase_results_csv(
                            results_file,
                            all_results,
                            write_headers=(not csv_headers_written)
                        )
                        logger.info(f"Checkpoint saved ({len(all_results)} total)")
                        csv_headers_written = True
                        existing_results = all_results  # Update existing results to include newly saved batch
                        results = []
                    
                    pbar.update(1)
                
                except Exception as e:
                    logger.warning(f"Failed to process {sample.sample_id}: {e}")
                    # Create error result
                    error_result = BenchmarkResult(
                        sample_id=sample.sample_id,
                        image_path=sample.image_path,
                        dataset=dataset_name,
                        model=model_name,
                        phase=phase,
                        ground_truth=sample.ground_truth,
                        error=str(e),
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(error_result)
                    pbar.update(1)
        
        # Final save
        if results:
            self._save_phase_results_csv(
                results_file,
                existing_results + results,
                write_headers=(not csv_headers_written)
            )
            logger.info(f"Phase {phase} saved ({len(existing_results) + len(results)} total)")
        
        # Extract and save prompts for phase 3
        all_results = existing_results + results
        if phase == 3 and self.config.phase_3_letter:
            self._extract_and_save_prompts(all_results, self.config.phase_3_letter)
        
        return all_results
    
    def _process_sample(self, sample, model_name: str, phase: int, dataset_name: str) -> BenchmarkResult:
        """
        Process single sample through OCR/VLM API.
        
        Args:
            sample: Sample object
            model_name: Model name
            phase: Phase number
            dataset_name: Dataset name (for metadata)
        
        Returns:
            BenchmarkResult with prediction
        """
        start_time = time.time()
        prediction = None
        prompt: Optional[str] = None
        error = None
        tokens_used = None
        
        def execute_phase():
            """Execute the API call for this phase."""
            nonlocal prediction, prompt
            if phase == 1:
                # Phase 1: OCR Baseline
                prediction = self._call_ocr_model(model_name, sample.image_path)
            
            elif phase == 2:
                # Phase 2: VLM Baseline with generic prompt
                prompt = prompt_module.get_phase_2_prompt(dataset_name)
                prediction = self._call_vlm_model(model_name, sample.image_path, prompt)
            
            elif phase == 3:
                # Phase 3: VLM + Intermediate context-aware prompt
                prompt = self._build_context_aware_prompt(sample, dataset_name)
                prediction = self._call_vlm_model(model_name, sample.image_path, prompt)
            
            elif phase == 4:
                # Phase 4: VLM + Detailed context-aware prompt
                prompt = prompt_module.get_phase_4_prompt(sample, dataset_name, self.config.phase_3_letter)
                prediction = self._call_vlm_model(model_name, sample.image_path, prompt)
            
            else:
                raise ValueError(f"Unknown phase: {phase}")
        
        try:
            execute_phase()
        
        except Exception as e:
            logger.warning(f"API call failed for {sample.sample_id}: {e}")
            error = str(e)
            
            # Optionally retry
            if self.config.retry_failed:
                for retry in range(self.config.max_retries):
                    logger.info(f"  Retry {retry+1}/{self.config.max_retries}")
                    try:
                        time.sleep(2 ** retry)  # Exponential backoff
                        execute_phase()
                        error = None
                        logger.info(f"  Retry succeeded")
                        break
                    except Exception as retry_e:
                        logger.debug(f"  Retry failed: {retry_e}")
                        error = str(retry_e)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Extract language from metadata if available
        language = sample.metadata.get('language') if hasattr(sample, 'metadata') else None
        
        return BenchmarkResult(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            dataset=dataset_name,
            model=model_name,
            phase=phase,
            ground_truth=sample.ground_truth,
            prediction=prediction,
            prompt=prompt,
            inference_time_ms=elapsed_ms,
            tokens_used=tokens_used,
            error=error,
            timestamp=datetime.now().isoformat(),
            language=language
        )
    
    def _call_ocr_model(self, model_name: str, image_path: str) -> str:
        """
        Call OCR model.
        
        Args:
            model_name: OCR model name
            image_path: Path to image
        
        Returns:
            Extracted text
        
        Raises:
            Exception: If API call fails
        """
        try:
            # Validate model is an OCR model
            if model_name not in ModelRegistry.list_ocr_models():
                raise ValueError(f"{model_name} is not an OCR model. Available: {ModelRegistry.list_ocr_models()}")
            
            # Call unified API
            response = self.api.process(image_path, model=model_name)
            
            if response.error:
                raise Exception(response.error)
            
            return response.content
        
        except Exception as e:
            logger.error(f"OCR call failed: {e}")
            raise
    
    def _call_vlm_model(self, model_name: str, image_path: str, query: str) -> str:
        """
        Call VLM model.
        
        Args:
            model_name: VLM model name
            image_path: Path to image
            query: Query/prompt for VLM
        
        Returns:
            Model response
        
        Raises:
            Exception: If API call fails
        """
        try:
            # Validate model is a VLM
            if model_name not in ModelRegistry.list_vlm_models():
                raise ValueError(f"{model_name} is not a VLM model. Available: {ModelRegistry.list_vlm_models()}")
            
            # Call unified API
            response = self.api.process(image_path, model=model_name, query=query)
            
            if response.error:
                raise Exception(response.error)
            
            return response.content
        
        except Exception as e:
            logger.error(f"VLM call failed: {e}")
            raise
    
    def _build_context_aware_prompt(self, sample, dataset_name: str) -> str:
        """
        Build context-aware prompt for Phase 3.
        
        Args:
            sample: Sample object with metadata
            dataset_name: Dataset name
        
        Returns:
            Context-aware prompt string
        """
        return prompt_module.get_phase_3_prompt(
            sample, dataset_name, self.config.phase_3_letter
        )
    
    def _get_dataset_root(self, dataset_name: str) -> str:
        """Get dataset root directory path."""
        base_path = Path("/Users/kenzabenkirane/Documents/GitHub/research-playground/datasets/parsing")
        
        dataset_paths = {
            'IAM': base_path / "IAM",
            'ICDAR': base_path / "ICDAR",
            'ICDAR_mini': Path(__file__).parent / "datasets_subsets",  # ICDAR_mini is in datasets_subsets
            'IAM_mini': Path(__file__).parent / "datasets_subsets",  # IAM_mini is in datasets_subsets
            'PubLayNet': base_path / "PubLayNet",
            'VOC2007': Path("/Users/kenzabenkirane/Documents/GitHub/research-playground/datasets/VOC2007"),
        }
        
        if dataset_name not in dataset_paths:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return str(dataset_paths[dataset_name])
    
    def _extract_and_save_prompts(self, results: List[BenchmarkResult], phase_letter: str):
        """
        Extract unique prompts from results and save to prompts folder.
        
        Args:
            results: List of BenchmarkResult objects
            phase_letter: Letter suffix (a, b, c, etc.)
        """
        # Extract unique non-empty prompts
        unique_prompts = []
        seen_prompts = set()
        
        for result in results:
            if result.prompt and result.prompt.strip():
                prompt_text = result.prompt.strip()
                if prompt_text not in seen_prompts:
                    unique_prompts.append(prompt_text)
                    seen_prompts.add(prompt_text)
        
        # Save each unique prompt to a separate file
        if unique_prompts:
            prompt_file = self.prompts_dir / f"prompt{phase_letter}.txt"
            with open(prompt_file, 'w') as f:
                f.write(unique_prompts[0])  # Save the first (likely only) unique prompt
            logger.info(f"Saved {len(unique_prompts)} unique prompt(s) to {prompt_file}")
            
            # Log if there were multiple unique prompts
            if len(unique_prompts) > 1:
                logger.warning(f"Found {len(unique_prompts)} unique prompts for phase {phase_letter}. Saved first one.")
    
    def _save_phase_results_csv(self, results_file: Path, results: List[BenchmarkResult], write_headers: bool = True):
        """Save phase results to CSV file."""
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Define CSV column order
        fieldnames = [
            'sample_id', 'image_path', 'dataset', 'model', 'phase', 'language', 'ground_truth',
            'prediction', 'prompt', 'inference_time_ms', 'tokens_used', 'error', 'timestamp'
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
                    # Convert string values to appropriate types
                    row['inference_time_ms'] = float(row['inference_time_ms']) if row.get('inference_time_ms') else 0.0
                    row['tokens_used'] = int(row['tokens_used']) if row.get('tokens_used') and row['tokens_used'] != 'None' else None
                    row['phase'] = int(row['phase'])
                    # language is optional, keep as string or None
                    if 'language' not in row or not row.get('language'):
                        row['language'] = None
                    results.append(BenchmarkResult(**row))
            
            return results
        except Exception as e:
            logger.warning(f"Failed to load existing results {results_file}: {e}")
            return []
    
    def _save_checkpoint(self, checkpoint_data: Dict):
        """Save checkpoint for resumability."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self) -> Dict:
        """Load checkpoint if exists."""
        if not self.checkpoint_file.exists():
            return {}
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return {}


def create_benchmark_config(
    datasets: List[str],
    models: List[str],
    phases: List[int],
    sample_limit: Optional[int] = None,
    results_dir: str = "results"
) -> BenchmarkConfig:
    """
    Create benchmark configuration.
    
    Args:
        datasets: List of dataset names
        models: List of model names
        phases: List of phases to run
        sample_limit: Max samples per dataset
        results_dir: Output directory
    
    Returns:
        BenchmarkConfig instance
    """
    return BenchmarkConfig(
        datasets=datasets,
        models=models,
        phases=phases,
        sample_limit=sample_limit,
        results_dir=results_dir
    )


if __name__ == '__main__':
    """
    IAM_mini Dataset benchmark with GPT mini and nano models on phases 2 and 3a.
    Results include 'prediction' column with model outputs for post-processing comparison.
    Each model saves results to its own folder: results/<model_name>/
    """
    logger.info("=" * 80)
    logger.info("Starting IAM_mini Benchmark with GPT mini and nano (phases 2 & 3a)")
    logger.info("=" * 80)
    
    models = ['gpt-5-mini', 'gpt-5-nano']
    
    for model_name in models:
        logger.info(f"\n{'='*80}")
        logger.info(f"RUNNING MODEL: {model_name.upper()}")
        logger.info(f"{'='*80}\n")
        
        # Create per-model results directory
        model_results_dir = f"results/{model_name}"
        
        # Create configuration for this model
        config = create_benchmark_config(
            datasets=['IAM_mini'],  # Use IAM_mini dataset
            models=[model_name],  # Single model per run
            phases=[2, 3],  # Phase 2 = VLM baseline, Phase 3 = VLM + Context
            sample_limit=None,  # Use all IAM_mini samples
            results_dir=model_results_dir
        )
        
        config.phase_3_letter = 'a'  # Phase 3a
        
        # Run benchmark with progress bars
        runner = BenchmarkRunner(config)
        summary = runner.run_benchmark()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Summary for {model_name}:")
        logger.info(f"{'='*80}")
        logger.info(json.dumps(summary, indent=2))
    
    logger.info(f"\n{'='*80}")
    logger.info("All models completed!")
    logger.info(f"Results saved to: results/<model_name>/")
    logger.info(f"{'='*80}")
