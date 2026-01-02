#!/usr/bin/env python3
"""
InfographicVQA Benchmark: Infographic Visual Question Answering Evaluation

Evaluates OCR and VLM models on InfographicVQA_mini dataset using three approaches:

Phases:
- QA1a, QA1b, QA1c: OCR Pipeline (OCR extraction → LLM QA)
    - a: Simple prompt
    - b: Detailed prompt  
    - c: Chain-of-thought prompt
- QA2a, QA2b, QA2c: VLM Parse Pipeline (VLM extraction → LLM QA)
    - a: Simple prompt
    - b: Detailed prompt
    - c: Chain-of-thought prompt
- QA3a, QA3b: Direct VQA (VLM sees image + question)
    - a: Simple prompt
    - b: Detailed prompt
- QA4a, QA4b, QA4c: Pre-extracted OCR Pipeline (uses AWS Textract OCR from metadata)
    - a: Simple prompt
    - b: Detailed prompt
    - c: Chain-of-thought prompt

Note: InfographicVQA includes pre-extracted OCR text from AWS Textract in the metadata,
which enables QA4 phase (testing with pre-extracted OCR instead of live OCR).

Models:
- OCR: azure_intelligence, mistral_document_ai
- VLM: gpt-5-mini, gpt-5-nano

Metrics:
- ANLS: Average Normalized Levenshtein Similarity (standard DocVQA metric)
- Exact Match: Percentage of exact answer matches

Usage:
    # Run all phases
    python -m ocr_vs_vlm.benchmark_infographicvqa
    
    # Run specific phases
    python -m ocr_vs_vlm.benchmark_infographicvqa --phases QA1a QA3a QA4a
    
    # With sample limit
    python -m ocr_vs_vlm.benchmark_infographicvqa --sample-limit 50
"""

import json
import logging
import sys
import time
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .dataset_loaders_qa import InfographicVQAMiniDataset, QASample
from .unified_model_api import UnifiedModelAPI, ModelRegistry
from . import prompts_qa

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
    
    log_file = results_dir / 'benchmark_infographicvqa.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class QABenchmarkConfig:
    """Configuration for QA benchmark run."""
    
    # Dataset
    dataset_name: str = "InfographicVQA_mini"
    
    # Models for each approach
    ocr_models: List[str] = field(default_factory=lambda: ["azure_intelligence", "mistral_document_ai"])
    vlm_models: List[str] = field(default_factory=lambda: ["gpt-5-mini", "gpt-5-nano"])
    qa_model: str = "gpt-5-mini"  # Model for answering questions in pipeline approaches
    
    # Phases to run (includes QA4 for pre-extracted OCR)
    phases: List[str] = field(default_factory=lambda: [
        "QA1a", "QA1b", "QA1c",  # OCR Pipeline
        "QA2a", "QA2b", "QA2c",  # VLM Parse Pipeline  
        "QA3a", "QA3b",          # Direct VQA
        "QA4a", "QA4b", "QA4c"   # Pre-extracted OCR Pipeline
    ])
    
    # Sample control
    sample_limit: Optional[int] = None
    batch_size: int = 50
    
    # Output paths
    results_dir: str = "results/InfographicVQA_mini"
    
    # API settings
    retry_failed: bool = True
    max_retries: int = 2


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class QAResult:
    """Result from a single QA evaluation."""
    
    sample_id: str
    image_path: str
    question: str
    ground_truths: List[str]
    prediction: str
    
    # Phase info
    phase: str  # e.g., "QA1a", "QA2b", "QA3a", "QA4a"
    parsing_model: Optional[str] = None  # OCR/VLM for extraction (None for QA4)
    qa_model: Optional[str] = None  # LLM for answering
    
    # Intermediate outputs
    extracted_text: Optional[str] = None
    prompt_used: Optional[str] = None
    
    # Metrics
    anls_score: float = 0.0
    exact_match: float = 0.0
    
    # Metadata
    inference_time_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV serialization."""
        return {
            'sample_id': self.sample_id,
            'image_path': self.image_path,
            'question': self.question,
            'ground_truths': json.dumps(self.ground_truths),
            'prediction': self.prediction,
            'phase': self.phase,
            'parsing_model': self.parsing_model,
            'qa_model': self.qa_model,
            'extracted_text': self.extracted_text,
            'prompt_used': self.prompt_used,
            'anls_score': self.anls_score,
            'exact_match': self.exact_match,
            'inference_time_ms': self.inference_time_ms,
            'error': self.error,
            'timestamp': self.timestamp,
        }


# =============================================================================
# METRICS
# =============================================================================

def compute_anls(prediction: str, ground_truths: List[str], threshold: float = 0.5) -> float:
    """
    Compute Average Normalized Levenshtein Similarity (ANLS).
    
    This is the standard DocVQA/InfographicVQA evaluation metric.
    """
    if not prediction or not ground_truths:
        return 0.0
    
    try:
        import editdistance
    except ImportError:
        logger.warning("editdistance not installed, using basic comparison")
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


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Check if prediction exactly matches any ground truth. Returns 1.0 or 0.0."""
    if not prediction or not ground_truths:
        return 0.0
    
    pred = prediction.lower().strip()
    return 1.0 if any(pred == gt.lower().strip() for gt in ground_truths) else 0.0


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class InfographicVQABenchmark:
    """
    InfographicVQA Benchmark with four evaluation approaches.
    
    Approaches:
    1. OCR Pipeline (QA1): OCR extraction → LLM QA
    2. VLM Parse Pipeline (QA2): VLM extraction → LLM QA
    3. Direct VQA (QA3): VLM sees image + question together
    4. Pre-extracted OCR Pipeline (QA4): Uses AWS Textract OCR from metadata → LLM QA
    """
    
    def __init__(self, config: QABenchmarkConfig):
        """Initialize benchmark with configuration."""
        self.config = config
        
        # Setup results directory
        if Path(config.results_dir).is_absolute():
            self.results_dir = Path(config.results_dir)
        else:
            self.results_dir = Path(__file__).parent / config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        _configure_logging(self.results_dir)
        
        # Initialize API
        self.api = UnifiedModelAPI()
        
        logger.info(f"InfographicVQA Benchmark initialized")
        logger.info(f"  Phases: {config.phases}")
        logger.info(f"  OCR models: {config.ocr_models}")
        logger.info(f"  VLM models: {config.vlm_models}")
        logger.info(f"  QA model: {config.qa_model}")
        logger.info(f"  Sample limit: {config.sample_limit or 'All'}")
        logger.info(f"  Results dir: {self.results_dir}")
    
    def run(self) -> Dict:
        """Execute full QA benchmark across all phases."""
        start_time = time.time()
        
        summary = {
            'dataset': self.config.dataset_name,
            'start_time': datetime.now().isoformat(),
            'config': asdict(self.config),
            'phases': {},
            'metrics_summary': {}
        }
        
        # Load dataset
        logger.info(f"\nLoading {self.config.dataset_name}...")
        dataset_root = Path(__file__).parent / "datasets_subsets"
        dataset = InfographicVQAMiniDataset(
            str(dataset_root),
            sample_limit=self.config.sample_limit
        )
        logger.info(f"Loaded {len(dataset)} QA samples")
        
        # Run each phase
        all_results: Dict[str, List[QAResult]] = {}
        
        for phase in self.config.phases:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running phase: {phase}")
            logger.info(f"{'='*60}")
            
            try:
                phase_results = self._run_phase(phase, dataset)
                all_results[phase] = phase_results
                
                # Compute phase metrics
                metrics = self._compute_metrics(phase_results)
                summary['phases'][phase] = {
                    'status': 'completed',
                    'samples_processed': len(phase_results),
                    'metrics': metrics
                }
                
                logger.info(f"Phase {phase}: ANLS={metrics['anls']:.4f}, EM={metrics['exact_match']:.4f}")
                
            except Exception as e:
                logger.error(f"Error in phase {phase}: {e}")
                summary['phases'][phase] = {'status': 'failed', 'error': str(e)}
        
        # Aggregate metrics
        summary['metrics_summary'] = self._aggregate_metrics(all_results)
        summary['end_time'] = datetime.now().isoformat()
        summary['total_time_seconds'] = time.time() - start_time
        
        # Save summary
        summary_file = self.results_dir / "execution_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nSaved summary to {summary_file}")
        
        return summary
    
    def _run_phase(self, phase: str, dataset: InfographicVQAMiniDataset) -> List[QAResult]:
        """Run a single phase across all relevant models."""
        results = []
        
        # Parse phase type and variation
        phase_type = phase[:3]  # QA1, QA2, QA3, QA4
        variation = phase[3] if len(phase) > 3 else 'a'
        
        # Determine models for this phase
        if phase_type == "QA1":
            models = self.config.ocr_models
        elif phase_type == "QA2":
            models = self.config.vlm_models
        elif phase_type == "QA3":
            models = self.config.vlm_models
        else:  # QA4 - Pre-extracted OCR, only needs QA model
            models = ["pre_extracted_ocr"]  # Placeholder, uses metadata OCR
        
        # Setup output
        phase_dir = self.results_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        for model in models:
            model_results_file = phase_dir / f"{model}_results.csv"
            existing_ids = self._load_existing_ids(model_results_file)
            samples_to_process = [s for s in dataset if s.sample_id not in existing_ids]
            
            logger.info(f"  Model: {model} ({len(samples_to_process)} samples)")
            
            model_results = []
            with tqdm(total=len(samples_to_process), desc=f"{phase}/{model}", unit="sample") as pbar:
                for idx, sample in enumerate(samples_to_process):
                    try:
                        result = self._process_sample(sample, phase_type, variation, model)
                        model_results.append(result)
                        
                        # Save incrementally
                        if (idx + 1) % self.config.batch_size == 0:
                            self._save_results(model_results_file, model_results)
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.warning(f"Failed {sample.sample_id}: {e}")
                        error_result = QAResult(
                            sample_id=sample.sample_id,
                            image_path=sample.image_path,
                            question=sample.question,
                            ground_truths=sample.answers,
                            prediction="",
                            phase=phase,
                            parsing_model=model,
                            error=str(e),
                            timestamp=datetime.now().isoformat()
                        )
                        model_results.append(error_result)
                        pbar.update(1)
            
            # Final save
            if model_results:
                self._save_results(model_results_file, model_results)
            
            results.extend(model_results)
        
        return results
    
    def _process_sample(
        self,
        sample: QASample,
        phase_type: str,
        variation: str,
        parsing_model: str
    ) -> QAResult:
        """Process a single sample through the appropriate pipeline."""
        start_time = time.time()
        extracted_text = None
        prediction = ""
        prompt_used = None
        error = None
        qa_model_used = None
        
        try:
            if phase_type == "QA1":
                # OCR Pipeline: Extract with OCR, answer with LLM
                extracted_text = self._extract_with_ocr(sample.image_path, parsing_model)
                prompt_used = prompts_qa.get_pipeline_qa_prompt(
                    sample.question, extracted_text, variation, "InfographicVQA"
                )
                qa_model_used = self.config.qa_model
                prediction = self._answer_with_llm(prompt_used, qa_model_used)
                
            elif phase_type == "QA2":
                # VLM Parse Pipeline: Extract with VLM, answer with LLM
                parse_prompt = prompts_qa.get_parsing_prompt("InfographicVQA")
                extracted_text = self._extract_with_vlm(sample.image_path, parsing_model, parse_prompt)
                prompt_used = prompts_qa.get_pipeline_qa_prompt(
                    sample.question, extracted_text, variation, "InfographicVQA"
                )
                qa_model_used = self.config.qa_model
                prediction = self._answer_with_llm(prompt_used, qa_model_used)
                
            elif phase_type == "QA3":
                # Direct VQA: VLM sees image + question
                prompt_used = prompts_qa.get_direct_vqa_prompt(
                    sample.question, variation, "InfographicVQA"
                )
                prediction = self._direct_vqa(sample.image_path, parsing_model, prompt_used)
                
            elif phase_type == "QA4":
                # Pre-extracted OCR Pipeline: Use OCR from metadata
                extracted_text = sample.metadata.get('ocr_text', '')
                if not extracted_text:
                    raise ValueError("No pre-extracted OCR text in metadata")
                
                prompt_used = prompts_qa.get_pipeline_qa_prompt(
                    sample.question, extracted_text, variation, "InfographicVQA"
                )
                qa_model_used = self.config.qa_model
                prediction = self._answer_with_llm(prompt_used, qa_model_used)
            
        except Exception as e:
            logger.warning(f"Processing error: {e}")
            error = str(e)
            
            if self.config.retry_failed:
                for retry in range(self.config.max_retries):
                    try:
                        time.sleep(2 ** retry)
                        # Simplified retry for direct VQA
                        if phase_type == "QA3":
                            prompt_used = prompts_qa.get_direct_vqa_prompt(sample.question, variation, "InfographicVQA")
                            prediction = self._direct_vqa(sample.image_path, parsing_model, prompt_used)
                            error = None
                            break
                    except Exception:
                        pass
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Clean prediction
        prediction = self._clean_prediction(prediction)
        
        # Compute metrics
        anls = compute_anls(prediction, sample.answers)
        em = compute_exact_match(prediction, sample.answers)
        
        return QAResult(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            question=sample.question,
            ground_truths=sample.answers,
            prediction=prediction,
            phase=f"{phase_type}{variation}",
            parsing_model=parsing_model if parsing_model != "pre_extracted_ocr" else None,
            qa_model=qa_model_used,
            extracted_text=extracted_text,
            prompt_used=prompt_used,
            anls_score=anls,
            exact_match=em,
            inference_time_ms=elapsed_ms,
            error=error,
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_with_ocr(self, image_path: str, model: str) -> str:
        """Extract text using OCR model."""
        response = self.api.process(image_path, model=model)
        if response.error:
            raise Exception(response.error)
        return response.content
    
    def _extract_with_vlm(self, image_path: str, model: str, prompt: str) -> str:
        """Extract text using VLM with parsing prompt."""
        response = self.api.process(image_path, model=model, query=prompt)
        if response.error:
            raise Exception(response.error)
        return response.content
    
    def _answer_with_llm(self, prompt: str, model: str) -> str:
        """Answer question using LLM (text-only)."""
        try:
            client = self.api.azure_openai_client
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024  # Higher limit to account for reasoning tokens
            )
            return response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        except Exception as e:
            logger.error(f"LLM answering failed: {e}")
            raise
    
    def _direct_vqa(self, image_path: str, model: str, prompt: str) -> str:
        """Direct VQA: VLM sees image and answers question."""
        response = self.api.process(image_path, model=model, query=prompt)
        if response.error:
            raise Exception(response.error)
        return response.content
    
    def _clean_prediction(self, prediction: str) -> str:
        """Clean prediction by extracting answer from chain-of-thought."""
        if not prediction:
            return ""
        
        if "Final Answer:" in prediction:
            prediction = prediction.split("Final Answer:")[-1].strip()
        elif "Answer:" in prediction:
            prediction = prediction.split("Answer:")[-1].strip()
        
        return prediction.split("\n")[0].strip()
    
    def _compute_metrics(self, results: List[QAResult]) -> Dict:
        """Compute aggregate metrics for results."""
        if not results:
            return {'anls': 0.0, 'exact_match': 0.0}
        
        valid = [r for r in results if r.error is None]
        if not valid:
            return {'anls': 0.0, 'exact_match': 0.0}
        
        return {
            'anls': sum(r.anls_score for r in valid) / len(valid),
            'exact_match': sum(r.exact_match for r in valid) / len(valid),
            'total_samples': len(results),
            'valid_samples': len(valid),
            'error_rate': (len(results) - len(valid)) / len(results)
        }
    
    def _aggregate_metrics(self, all_results: Dict[str, List[QAResult]]) -> Dict:
        """Aggregate metrics across all phases."""
        summary = {}
        
        for phase, results in all_results.items():
            summary[phase] = self._compute_metrics(results)
        
        # Group by approach
        ocr_phases = [p for p in all_results if p.startswith("QA1")]
        vlm_parse_phases = [p for p in all_results if p.startswith("QA2")]
        direct_vqa_phases = [p for p in all_results if p.startswith("QA3")]
        pre_ocr_phases = [p for p in all_results if p.startswith("QA4")]
        
        def avg_metric(phases: List[str], metric: str) -> float:
            if not phases:
                return 0.0
            values = [summary.get(p, {}).get(metric, 0.0) for p in phases]
            return sum(values) / len(values) if values else 0.0
        
        summary['by_approach'] = {
            'ocr_pipeline': {
                'avg_anls': avg_metric(ocr_phases, 'anls'),
                'avg_em': avg_metric(ocr_phases, 'exact_match')
            },
            'vlm_parse_pipeline': {
                'avg_anls': avg_metric(vlm_parse_phases, 'anls'),
                'avg_em': avg_metric(vlm_parse_phases, 'exact_match')
            },
            'direct_vqa': {
                'avg_anls': avg_metric(direct_vqa_phases, 'anls'),
                'avg_em': avg_metric(direct_vqa_phases, 'exact_match')
            },
            'pre_extracted_ocr_pipeline': {
                'avg_anls': avg_metric(pre_ocr_phases, 'anls'),
                'avg_em': avg_metric(pre_ocr_phases, 'exact_match')
            }
        }
        
        return summary
    
    def _save_results(self, results_file: Path, results: List[QAResult]):
        """Save results to CSV."""
        fieldnames = [
            'sample_id', 'image_path', 'question', 'ground_truths', 'prediction',
            'phase', 'parsing_model', 'qa_model', 'extracted_text', 'prompt_used',
            'anls_score', 'exact_match', 'inference_time_ms', 'error', 'timestamp'
        ]
        
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())
    
    def _load_existing_ids(self, results_file: Path) -> set:
        """Load existing sample IDs from results file."""
        if not results_file.exists():
            return set()
        try:
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                return {row['sample_id'] for row in reader}
        except Exception:
            return set()


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run InfographicVQA benchmark from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='InfographicVQA QA Benchmark')
    parser.add_argument(
        '--phases', '-p',
        nargs='+',
        default=["QA1a", "QA1b", "QA1c", "QA2a", "QA2b", "QA2c", "QA3a", "QA3b", "QA4a", "QA4b", "QA4c"],
        help='Phases to run (default: all including QA4 pre-extracted OCR)'
    )
    parser.add_argument(
        '--sample-limit', '-n',
        type=int,
        default=None,
        help='Maximum samples to process'
    )
    parser.add_argument(
        '--ocr-models',
        nargs='+',
        default=["azure_intelligence", "mistral_document_ai"],
        help='OCR models for pipeline approach'
    )
    parser.add_argument(
        '--vlm-models',
        nargs='+',
        default=["gpt-5-mini", "gpt-5-nano"],
        help='VLM models for parsing/direct VQA'
    )
    parser.add_argument(
        '--qa-model',
        default="gpt-5-mini",
        help='Model for answering questions in pipeline'
    )
    
    args = parser.parse_args()
    
    config = QABenchmarkConfig(
        phases=args.phases,
        sample_limit=args.sample_limit,
        ocr_models=args.ocr_models,
        vlm_models=args.vlm_models,
        qa_model=args.qa_model
    )
    
    benchmark = InfographicVQABenchmark(config)
    summary = benchmark.run()
    
    # Print final summary
    print("\n" + "="*70)
    print("INFOGRAPHICVQA BENCHMARK SUMMARY")
    print("="*70)
    
    for phase, metrics in summary.get('metrics_summary', {}).items():
        if phase == 'by_approach':
            continue
        print(f"\n{phase}:")
        print(f"  ANLS:        {metrics.get('anls', 0):.4f}")
        print(f"  Exact Match: {metrics.get('exact_match', 0):.4f}")
    
    print("\n" + "-"*70)
    print("By Approach:")
    
    by_approach = summary.get('metrics_summary', {}).get('by_approach', {})
    for approach, metrics in by_approach.items():
        print(f"\n{approach}:")
        print(f"  Avg ANLS: {metrics.get('avg_anls', 0):.4f}")
        print(f"  Avg EM:   {metrics.get('avg_em', 0):.4f}")


if __name__ == '__main__':
    main()
