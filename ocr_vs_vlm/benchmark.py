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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import sys

from dataset_loaders import DatasetRegistry, validate_dataset, load_image
from api_calls import call_ocr, call_vlm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    
    # Dataset and model selection
    datasets: List[str]  # ['IAM', 'ICDAR', 'PubLayNet']
    models: List[str]  # OCR + VLM models to test
    
    # Phase control
    phases: List[int]  # [1, 2, 3] which phases to run
    
    # Sample control
    sample_limit: Optional[int] = None  # Max samples per dataset
    batch_size: int = 10  # Batch API calls this many samples
    
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
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Result from a single sample evaluation."""
    
    sample_id: str
    dataset: str
    model: str
    phase: int
    image_path: str
    ground_truth: str
    prediction: Optional[str] = None
    prompt: Optional[str] = None
    
    # Metadata
    inference_time_ms: float = 0.0
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class BenchmarkRunner:
    """Main orchestrator for benchmark execution."""
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark runner.
        
        Args:
            config: BenchmarkConfig instance
        """
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.results_dir / config.checkpoint_file
        self.checkpoint = self._load_checkpoint()
        
        logger.info(f"BenchmarkRunner initialized")
        logger.info(f"Results directory: {self.results_dir}")
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
        output_dir = self.results_dir / dataset_name / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"phase_{phase}_results.json"
        
        # Load existing results if resuming
        existing_results = self._load_existing_results(results_file)
        processed_ids = {r['sample_id'] for r in existing_results}
        
        samples_to_process = [
            s for s in dataset if s.sample_id not in processed_ids
        ]
        
        logger.info(f"      Processing {len(samples_to_process)} samples (skipping {len(processed_ids)} existing)")
        
        # Process samples
        for idx, sample in enumerate(samples_to_process):
            # Progress
            progress = f"[{idx+1}/{len(samples_to_process)}]"
            logger.debug(f"{progress} {sample.sample_id}")
            
            try:
                # Generate result
                result = self._process_sample(
                    sample, model_name, phase, dataset_name
                )
                results.append(result)
                
                # Save incrementally every batch_size samples
                if (idx + 1) % self.config.batch_size == 0:
                    self._save_phase_results(
                        results_file,
                        existing_results + results
                    )
                    logger.info(f"{progress} Checkpoint saved ({len(existing_results) + len(results)} total)")
                    results = []
            
            except Exception as e:
                logger.warning(f"Failed to process {sample.sample_id}: {e}")
                # Create error result
                error_result = BenchmarkResult(
                    sample_id=sample.sample_id,
                    dataset=dataset_name,
                    model=model_name,
                    phase=phase,
                    image_path=sample.image_path,
                    ground_truth=sample.ground_truth,
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )
                results.append(error_result)
        
        # Final save
        if results:
            self._save_phase_results(
                results_file,
                existing_results + results
            )
            logger.info(f"Phase {phase} saved ({len(existing_results) + len(results)} total)")
        
        return existing_results + results
    
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
        prompt = None
        error = None
        tokens_used = None
        
        try:
            if phase == 1:
                # Phase 1: OCR Baseline
                prediction = self._call_ocr_model(model_name, sample.image_path)
            
            elif phase == 2:
                # Phase 2: VLM Baseline with generic prompt
                prompt = "Extract all text from this document image"
                prediction = self._call_vlm_model(model_name, sample.image_path, prompt)
            
            elif phase == 3:
                # Phase 3: VLM + Context-aware prompt
                prompt = self._build_context_aware_prompt(sample, dataset_name)
                prediction = self._call_vlm_model(model_name, sample.image_path, prompt)
            
            else:
                raise ValueError(f"Unknown phase: {phase}")
        
        except Exception as e:
            logger.warning(f"API call failed for {sample.sample_id}: {e}")
            error = str(e)
            
            # Optionally retry
            if self.config.retry_failed:
                for retry in range(self.config.max_retries):
                    logger.info(f"  Retry {retry+1}/{self.config.max_retries}")
                    try:
                        time.sleep(2 ** retry)  # Exponential backoff
                        
                        if phase == 1:
                            prediction = self._call_ocr_model(model_name, sample.image_path)
                        else:
                            prediction = self._call_vlm_model(model_name, sample.image_path, prompt)
                        
                        error = None
                        logger.info(f"  Retry succeeded")
                        break
                    except Exception as retry_e:
                        logger.debug(f"  Retry failed: {retry_e}")
                        error = str(retry_e)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return BenchmarkResult(
            sample_id=sample.sample_id,
            dataset=dataset_name,
            model=model_name,
            phase=phase,
            image_path=sample.image_path,
            ground_truth=sample.ground_truth,
            prediction=prediction,
            prompt=prompt,
            inference_time_ms=elapsed_ms,
            tokens_used=tokens_used,
            error=error,
            timestamp=datetime.now().isoformat()
        )
    
    def _call_ocr_model(self, model_name: str, image_path: str) -> str:
        """
        Call OCR model with timeout.
        
        Args:
            model_name: OCR model name
            image_path: Path to image
        
        Returns:
            Extracted text
        
        Raises:
            TimeoutError: If API call exceeds timeout
            Exception: Any API error
        """
        # Map model names to api_calls function names
        # For now, directly call based on model_name
        # In real implementation, may need mapping layer
        
        try:
            # Validate model is an OCR model
            ocr_models = ['azure_intelligence', 'donut', 'deepseek_ocr', 'mistral_ocr']
            if model_name not in ocr_models:
                raise ValueError(f"{model_name} is not an OCR model")
            
            # Call API
            documents = call_ocr(image_path, model=model_name)
            
            # Extract text from LangChain documents
            texts = [doc.page_content for doc in documents]
            return "\n".join(texts)
        
        except Exception as e:
            logger.error(f"OCR call failed: {e}")
            raise
    
    def _call_vlm_model(self, model_name: str, image_path: str, query: str) -> str:
        """
        Call VLM model with timeout.
        
        Args:
            model_name: VLM model name
            image_path: Path to image
            query: Query/prompt for VLM
        
        Returns:
            Model response
        
        Raises:
            TimeoutError: If API call exceeds timeout
            Exception: Any API error
        """
        try:
            # Validate model is a VLM
            vlm_models = ['gpt5_mini', 'gpt5_nano', 'claude_sonnet', 'claude_haiku', 'qwen_vl']
            if model_name not in vlm_models:
                raise ValueError(f"{model_name} is not a VLM model")
            
            # Call API
            result = call_vlm(image_path, model=model_name, query=query)
            return result
        
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
        prompt_parts = [
            f"Dataset: {dataset_name}",
            f"Task: Extract all text from this document image, preserving structure and layout"
        ]
        
        # Add dataset-specific context
        metadata = sample.metadata
        
        if 'languages' in metadata and metadata['languages']:
            langs = ", ".join(metadata['languages'])
            prompt_parts.append(f"Languages: {langs}")
        
        if 'num_text_lines' in metadata:
            prompt_parts.append(f"Expected approximately {metadata['num_text_lines']} text lines")
        
        if dataset_name == 'IAM':
            prompt_parts.append("This is handwritten text. Handle variations in writing style.")
        
        elif dataset_name == 'ICDAR':
            prompt_parts.append("This is multi-lingual scene text. Preserve script types and directions.")
        
        elif dataset_name == 'PubLayNet':
            prompt_parts.append("This is a document page. Preserve document structure and layout.")
        
        prompt_parts.append("Return ONLY the extracted text.")
        
        return "\n".join(prompt_parts)
    
    def _get_dataset_root(self, dataset_name: str) -> str:
        """Get dataset root directory path."""
        base_path = Path("/Users/kenzabenkirane/Documents/GitHub/research-playground/datasets/parsing")
        
        dataset_paths = {
            'IAM': base_path / "IAM",
            'ICDAR': base_path / "ICDAR",
            'PubLayNet': base_path / "PubLayNet",
        }
        
        if dataset_name not in dataset_paths:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return str(dataset_paths[dataset_name])
    
    def _save_phase_results(self, results_file: Path, results: List[BenchmarkResult]):
        """Save phase results to JSON file."""
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(
                [r.to_dict() for r in results],
                f,
                indent=2,
                default=str  # Handle any non-serializable objects
            )
    
    def _load_existing_results(self, results_file: Path) -> List[BenchmarkResult]:
        """Load existing results if file exists (for resumability)."""
        if not results_file.exists():
            return []
        
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            return [BenchmarkResult(**item) for item in data]
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
    Example benchmark execution.
    """
    logger.info("Starting OCR vs VLM Benchmark")
    
    # Create configuration
    config = create_benchmark_config(
        datasets=['ICDAR'],  # Start with ICDAR only
        models=['azure_intelligence', 'gpt5_mini', 'claude_haiku'],
        phases=[1, 2, 3],
        sample_limit=5,  # Small limit for testing
        results_dir="results"
    )
    
    # Run benchmark
    runner = BenchmarkRunner(config)
    summary = runner.run_benchmark()
    
    logger.info(f"\nBenchmark Summary:")
    logger.info(json.dumps(summary, indent=2))
