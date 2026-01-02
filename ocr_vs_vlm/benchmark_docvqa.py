#!/usr/bin/env python3
"""
DocVQA Benchmark: Document Visual Question Answering Evaluation

Evaluates VLM models on the DocVQA and InfographicVQA datasets for:
1. QA Mode: Question answering evaluation (primary use case)
2. Parsing Mode: Full document text extraction (uses standard benchmark pipeline)

Dataset Structure:
- DocVQA: Document images with question-answer pairs
- InfographicVQA: Infographic images with QA + pre-extracted OCR

Evaluation Metrics (QA Mode):
- ANLS: Average Normalized Levenshtein Similarity (standard DocVQA metric)
- Exact Match: Percentage of exact answer matches
- F1 Score: Token-level F1 between prediction and answers

Usage:
    # QA mode (default) - evaluates question answering
    python -m ocr_vs_vlm.benchmark_docvqa --models azure_intelligence mistral_document_ai
    
    # Parsing mode - evaluates text extraction
    python -m ocr_vs_vlm.benchmark_docvqa --mode parsing --models azure_intelligence
    
    # With sample limit
    python -m ocr_vs_vlm.benchmark_docvqa --models mistral_document_ai --sample-limit 100
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

from tqdm import tqdm

from ocr_vs_vlm.dataset_loaders import DatasetRegistry, QASample
from ocr_vs_vlm.unified_model_api import UnifiedModelAPI, ModelRegistry
from ocr_vs_vlm.benchmark import BenchmarkRunner, create_benchmark_config

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _configure_logging(results_dir: Path):
    """Configure logging to write to results directory."""
    logger.handlers.clear()
    
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    log_file = results_dir / 'benchmark_docvqa.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


@dataclass
class QAResult:
    """Result from a single QA evaluation."""
    
    sample_id: str
    image_path: str
    dataset: str
    model: str
    
    # QA fields
    question: str
    ground_truth_answers: List[str]  # All valid answers
    prediction: Optional[str] = None
    
    # Metrics
    anls_score: float = 0.0
    exact_match: bool = False
    
    # Metadata
    question_type: str = ""
    inference_time_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV serialization."""
        return {
            'sample_id': self.sample_id,
            'image_path': self.image_path,
            'dataset': self.dataset,
            'model': self.model,
            'question': self.question,
            'ground_truth_answers': json.dumps(self.ground_truth_answers),
            'prediction': self.prediction,
            'anls_score': self.anls_score,
            'exact_match': self.exact_match,
            'question_type': self.question_type,
            'inference_time_ms': self.inference_time_ms,
            'error': self.error,
            'timestamp': self.timestamp,
        }


def compute_anls(prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    """
    Compute Average Normalized Levenshtein Similarity (ANLS).
    
    This is the standard DocVQA evaluation metric.
    
    Args:
        prediction: Model's predicted answer
        ground_truths: List of valid ground truth answers
        threshold: Threshold below which score is 0 (default: 0.5)
    
    Returns:
        ANLS score in [0, 1]
    """
    if not prediction or not ground_truths:
        return 0.0
    
    try:
        import editdistance
    except ImportError:
        logger.warning("editdistance not installed, using basic comparison")
        # Fallback to exact match
        pred_lower = prediction.lower().strip()
        return 1.0 if any(pred_lower == gt.lower().strip() for gt in ground_truths) else 0.0
    
    pred = prediction.lower().strip()
    
    max_score = 0.0
    for gt in ground_truths:
        gt = gt.lower().strip()
        if not gt:
            continue
        
        # Compute normalized Levenshtein distance
        edit_dist = editdistance.eval(pred, gt)
        max_len = max(len(pred), len(gt))
        
        if max_len == 0:
            nls = 1.0
        else:
            nls = 1.0 - (edit_dist / max_len)
        
        # Apply threshold
        if nls < threshold:
            nls = 0.0
        
        max_score = max(max_score, nls)
    
    return max_score


def compute_exact_match(prediction: str, ground_truths: List[str]) -> bool:
    """Check if prediction exactly matches any ground truth."""
    if not prediction or not ground_truths:
        return False
    
    pred = prediction.lower().strip()
    return any(pred == gt.lower().strip() for gt in ground_truths)


class DocVQABenchmark:
    """DocVQA QA benchmark runner."""
    
    def __init__(
        self,
        models: List[str],
        dataset: str = 'DocVQA',  # 'DocVQA' or 'InfographicVQA'
        split: str = 'validation',
        sample_limit: Optional[int] = None,
        results_dir: Optional[str] = None,
    ):
        """
        Initialize DocVQA benchmark.
        
        Args:
            models: List of VLM model names
            dataset: Dataset name ('DocVQA' or 'InfographicVQA')
            split: Data split ('train', 'validation', 'test')
            sample_limit: Maximum samples to process
            results_dir: Output directory for results
        """
        self.models = models
        self.dataset_name = dataset
        self.split = split
        self.sample_limit = sample_limit
        
        # Setup results directory
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = Path(__file__).parent / "results" / dataset
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        _configure_logging(self.results_dir)
        
        # Initialize API
        self.api = UnifiedModelAPI()
        
        # Dataset root
        self.dataset_root = Path(__file__).parent.parent / "datasets" / "QA" / "DocVQA_hf"
        
        logger.info(f"DocVQA Benchmark initialized")
        logger.info(f"  Dataset: {dataset}")
        logger.info(f"  Split: {split}")
        logger.info(f"  Models: {models}")
        logger.info(f"  Sample limit: {sample_limit or 'All'}")
        logger.info(f"  Results dir: {self.results_dir}")
    
    def run(self) -> Dict:
        """
        Run QA benchmark on all models.
        
        Returns:
            Summary dict with metrics per model
        """
        start_time = time.time()
        summary = {
            'dataset': self.dataset_name,
            'split': self.split,
            'start_time': datetime.now().isoformat(),
            'models': {},
        }
        
        # Load dataset
        logger.info(f"\nLoading {self.dataset_name} dataset...")
        try:
            dataset = DatasetRegistry.get_dataset(
                self.dataset_name,
                str(self.dataset_root),
                sample_limit=self.sample_limit,
                mode='qa',
                split=self.split
            )
            logger.info(f"Loaded {len(dataset)} QA samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return {'error': str(e)}
        
        # Run each model
        for model_name in self.models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running model: {model_name}")
            logger.info(f"{'='*60}")
            
            try:
                model_results = self._run_model(model_name, dataset)
                summary['models'][model_name] = model_results
            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
                summary['models'][model_name] = {'error': str(e)}
        
        summary['end_time'] = datetime.now().isoformat()
        summary['total_time_seconds'] = time.time() - start_time
        
        # Save summary
        summary_file = self.results_dir / "qa_benchmark_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSaved summary to {summary_file}")
        
        return summary
    
    def _run_model(self, model_name: str, dataset) -> Dict:
        """Run QA evaluation for a single model."""
        results: List[QAResult] = []
        
        # Check if this is an OCR model (not ideal for QA)
        is_ocr_model = model_name in ModelRegistry.list_ocr_models()
        if is_ocr_model:
            logger.warning(f"⚠️  {model_name} is an OCR model, not a VLM.")
            logger.warning("   OCR models extract full text, not answers to questions.")
            logger.warning("   For QA evaluation, consider using VLM models: gpt-5-mini, gpt-5-nano, qwen_vl")
            logger.warning("   Proceeding with full document extraction + answer search...")
        
        # Create model output directory
        model_dir = self.results_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = model_dir / f"qa_results_{self.split}.csv"
        
        # Load existing results for resumability
        existing_ids = self._load_existing_ids(results_file)
        samples_to_process = [s for s in dataset if s.sample_id not in existing_ids]
        
        logger.info(f"Processing {len(samples_to_process)} samples (skipping {len(existing_ids)} existing)")
        
        # Build QA prompt template
        qa_prompt_template = self._get_qa_prompt_template()
        
        # Process samples
        with tqdm(total=len(samples_to_process), desc=model_name, unit="sample") as pbar:
            for sample in samples_to_process:
                result = self._process_qa_sample(sample, model_name, qa_prompt_template, is_ocr_model)
                results.append(result)
                
                # Save incrementally every 50 samples
                if len(results) % 50 == 0:
                    self._save_results(results_file, results, append=len(existing_ids) > 0)
                    logger.debug(f"Checkpoint: {len(results)} results saved")
                
                pbar.update(1)
        
        # Final save
        if results:
            self._save_results(results_file, results, append=len(existing_ids) > 0)
        
        # Compute aggregate metrics
        all_anls = [r.anls_score for r in results if r.error is None]
        all_exact = [r.exact_match for r in results if r.error is None]
        
        metrics = {
            'samples_processed': len(results),
            'samples_with_errors': sum(1 for r in results if r.error is not None),
            'mean_anls': sum(all_anls) / len(all_anls) if all_anls else 0.0,
            'exact_match_rate': sum(all_exact) / len(all_exact) if all_exact else 0.0,
            'results_file': str(results_file),
            'model_type': 'ocr' if is_ocr_model else 'vlm',
        }
        
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Mean ANLS: {metrics['mean_anls']:.4f}")
        logger.info(f"  Exact Match: {metrics['exact_match_rate']:.2%}")
        logger.info(f"  Errors: {metrics['samples_with_errors']}")
        
        return metrics
    
    def _process_qa_sample(self, sample: QASample, model_name: str, prompt_template: str, is_ocr_model: bool = False) -> QAResult:
        """Process a single QA sample."""
        start_time = time.time()
        prediction = None
        full_extraction = None
        error = None
        
        try:
            if is_ocr_model:
                # For OCR models: extract full text, then try to find answer
                response = self.api.process(
                    sample.image_path,
                    model=model_name
                )
                
                if response.error:
                    error = response.error
                else:
                    full_extraction = response.content
                    # Try to find the answer in the extracted text
                    prediction = self._extract_answer_from_text(
                        full_extraction, 
                        sample.question, 
                        sample.answers
                    )
            else:
                # For VLM models: use QA prompt
                prompt = prompt_template.format(question=sample.question)
                response = self.api.process(
                    sample.image_path,
                    model=model_name,
                    query=prompt
                )
                
                if response.error:
                    error = response.error
                else:
                    prediction = response.content
                
        except Exception as e:
            error = str(e)
            logger.warning(f"API error for {sample.sample_id}: {e}")
        
        # Compute metrics
        anls = compute_anls(prediction, sample.answers) if prediction else 0.0
        exact = compute_exact_match(prediction, sample.answers) if prediction else False
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return QAResult(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            dataset=self.dataset_name,
            model=model_name,
            question=sample.question,
            ground_truth_answers=sample.answers,
            prediction=prediction,
            anls_score=anls,
            exact_match=exact,
            question_type=sample.question_type,
            inference_time_ms=elapsed_ms,
            error=error,
            timestamp=datetime.now().isoformat(),
        )
    
    def _extract_answer_from_text(self, text: str, question: str, answers: List[str]) -> Optional[str]:
        """
        Try to extract an answer from full OCR text.
        
        For OCR models, we can't ask questions directly. Instead, we:
        1. Search for exact matches of known answers in the text
        2. Return the matching text if found
        
        Args:
            text: Full OCR extracted text
            question: The question (for context, not used in simple matching)
            answers: List of valid answers to search for
        
        Returns:
            Matched answer or None
        """
        if not text or not answers:
            return None
        
        text_lower = text.lower()
        
        # Try to find each answer in the text
        for answer in answers:
            answer_lower = answer.lower().strip()
            if answer_lower in text_lower:
                # Return the matched answer
                return answer
        
        return None
    
    def _get_qa_prompt_template(self) -> str:
        """Get the QA prompt template."""
        return (
            "Look at this document image and answer the following question.\n\n"
            "Question: {question}\n\n"
            "Provide only the answer, without any explanation or additional text. "
            "If the answer is a number, provide just the number. "
            "If the answer is text, provide just the text."
        )
    
    def _load_existing_ids(self, results_file: Path) -> set:
        """Load existing sample IDs from results file."""
        if not results_file.exists():
            return set()
        
        try:
            ids = set()
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ids.add(row['sample_id'])
            return ids
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")
            return set()
    
    def _save_results(self, results_file: Path, results: List[QAResult], append: bool = False):
        """Save results to CSV file."""
        fieldnames = [
            'sample_id', 'image_path', 'dataset', 'model', 'question',
            'ground_truth_answers', 'prediction', 'anls_score', 'exact_match',
            'question_type', 'inference_time_ms', 'error', 'timestamp'
        ]
        
        mode = 'a' if append else 'w'
        write_header = not append or not results_file.exists()
        
        with open(results_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())


def run_docvqa_parsing_benchmark(
    models: List[str],
    dataset: str = 'DocVQA',
    split: str = 'validation',
    sample_limit: Optional[int] = None,
    phases: Optional[List[int]] = None,
    phase_3_letter: str = 'a'
) -> Dict:
    """
    Run DocVQA in parsing mode using standard benchmark pipeline.
    
    This evaluates text extraction capability rather than QA.
    
    Args:
        models: List of model names
        dataset: 'DocVQA' or 'InfographicVQA'
        split: Data split
        sample_limit: Max samples
        phases: Phases to run (default: [1, 2])
        phase_3_letter: Phase variant letter
    
    Returns:
        Execution summary
    """
    if phases is None:
        phases = [1, 2]  # OCR baseline + VLM baseline
    
    logger.info("=" * 80)
    logger.info(f"{dataset} Parsing Benchmark")
    logger.info("=" * 80)
    
    # Note: Parsing mode requires adding DocVQA paths to _get_dataset_root in benchmark.py
    # For now, we'll run it directly here
    
    all_summaries = {}
    
    for model_name in models:
        logger.info(f"\nRunning model: {model_name}")
        
        model_results_dir = f"results/{dataset}/{model_name}"
        
        config = create_benchmark_config(
            datasets=[dataset],
            models=[model_name],
            phases=phases,
            sample_limit=sample_limit,
            results_dir=model_results_dir
        )
        config.phase_3_letter = phase_3_letter
        
        try:
            runner = BenchmarkRunner(config)
            summary = runner.run_benchmark()
            all_summaries[model_name] = summary
        except Exception as e:
            logger.error(f"Error with model {model_name}: {e}")
            all_summaries[model_name] = {'error': str(e)}
    
    return all_summaries


def main():
    """Main entry point for DocVQA benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DocVQA Visual Question Answering Benchmark'
    )
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        default=['azure_intelligence', 'mistral_document_ai'],
        help='Models to evaluate (default: azure_intelligence, mistral_document_ai)'
    )
    parser.add_argument(
        '--dataset', '-d',
        choices=['DocVQA', 'InfographicVQA'],
        default='DocVQA',
        help='Dataset to use (default: DocVQA)'
    )
    parser.add_argument(
        '--mode',
        choices=['qa', 'parsing'],
        default='qa',
        help='Evaluation mode: qa (question answering) or parsing (text extraction)'
    )
    parser.add_argument(
        '--split', '-s',
        choices=['train', 'validation', 'test'],
        default='validation',
        help='Data split to use (default: validation)'
    )
    parser.add_argument(
        '--sample-limit', '-n',
        type=int,
        default=None,
        help='Maximum samples to process (default: all)'
    )
    parser.add_argument(
        '--phases', '-p',
        nargs='+',
        type=int,
        default=None,
        help='Phases for parsing mode (default: 1 2)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("DocVQA Benchmark")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Sample limit: {args.sample_limit or 'All'}")
    logger.info("=" * 80)
    
    if args.mode == 'qa':
        # Run QA benchmark
        benchmark = DocVQABenchmark(
            models=args.models,
            dataset=args.dataset,
            split=args.split,
            sample_limit=args.sample_limit,
        )
        summary = benchmark.run()
    else:
        # Run parsing benchmark
        summary = run_docvqa_parsing_benchmark(
            models=args.models,
            dataset=args.dataset,
            split=args.split,
            sample_limit=args.sample_limit,
            phases=args.phases,
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(json.dumps(summary, indent=2))
    
    return summary


if __name__ == '__main__':
    main()
