#!/usr/bin/env python3
"""
InfographicVQA Benchmark: Infographic Visual Question Answering Evaluation

Evaluates OCR and VLM models on InfographicVQA_mini dataset using three approaches:

Phases:
- QA1a, QA1b, QA1c: OCR Pipeline (OCR extraction → LLM QA)
    - a: Simple prompt
    - b: Detailed prompt
    - c: Chain-of-thought prompt
- QA2a, QA2b: VLM Parse Pipeline (VLM parsing → same VLM QA, text-only)
    - a: Simple parsing prompt → simple QA prompt
    - b: Detailed parsing prompt → simple QA prompt
- QA3a, QA3b: Direct VQA (VLM sees image + question)
    - a: Simple prompt
    - b: Detailed prompt
- QA4a, QA4b: Pre-extracted OCR Pipeline (uses AWS Textract OCR from metadata)
    - a: Simple QA prompt
    - b: Detailed QA prompt

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

from datasets.dataset_loaders_qa import InfographicVQAMiniDataset, QASample
from models import UnifiedModelAPI, ModelRegistry
from prompts import prompts_qa
from metrics.evaluation_metrics import (
    compute_anls,
    compute_exact_match,
    compute_substring_match,
    compute_prediction_in_ground_truth,
    compute_ground_truth_in_prediction,
    compute_embedding_similarity
)

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
    ocr_models: List[str] = field(default_factory=lambda: ["azure_intelligence", "mistral_document_ai", "mistral_ocr_3"])
    vlm_models: List[str] = field(default_factory=lambda: ["gpt-5-mini", "gpt-5-nano"])
    qa_model: str = "gpt-5-mini"  # Model for answering questions in pipeline approaches
    
    # Phases to run (includes QA4 for pre-extracted OCR)
    phases: List[str] = field(default_factory=lambda: [
        "QA1a", "QA1b", "QA1c",  # OCR Pipeline
        "QA2a", "QA2b",          # VLM Parse Pipeline
        "QA3a", "QA3b",          # Direct VQA
        "QA4a", "QA4b"           # Pre-extracted OCR Pipeline
    ])
    
    # Sample control
    sample_limit: Optional[int] = None
    batch_size: int = 10  # Save every 10 rows

    # Sample ID filtering (for targeted reruns)
    sample_ids_file: Optional[str] = None  # Path to JSON with missing sample IDs
    sample_ids_filter: Optional[set] = None  # Set of specific IDs to process

    # Output paths
    results_dir: str = "results/1_raw/InfographicVQA_mini"
    test_mode: bool = False  # If True, prefix result files with "test_"

    # API settings
    retry_failed: bool = True
    max_retries: int = 2

    # Embedding settings
    compute_embeddings: bool = False  # Set to True to compute embeddings during benchmark


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
    substring_match: float = 0.0
    prediction_in_ground_truth: float = 0.0
    ground_truth_in_prediction: float = 0.0
    embedding_similarity: float = 0.0
    
    # Embeddings
    prediction_embedding: Optional[List[float]] = None
    ground_truth_embeddings: Optional[List[List[float]]] = None
    
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
            'substring_match': self.substring_match,
            'prediction_in_ground_truth': self.prediction_in_ground_truth,
            'ground_truth_in_prediction': self.ground_truth_in_prediction,
            'embedding_similarity': self.embedding_similarity,
            'inference_time_ms': self.inference_time_ms,
            'error': self.error,
            'timestamp': self.timestamp,
        }


# =============================================================================
# METRICS
# =============================================================================
# Metrics are imported from evaluation_metrics module


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
        
        # Setup results directory - always use absolute path
        if Path(config.results_dir).is_absolute():
            self.results_dir = Path(config.results_dir).resolve()
        else:
            # Resolve relative to ocr_vs_vlm/ directory (parent.parent of this file)
            self.results_dir = (Path(__file__).parent.parent.parent / config.results_dir).resolve()
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

        # Convert config to dict, handling set -> list for JSON serialization
        config_dict = asdict(self.config)
        if config_dict.get('sample_ids_filter'):
            config_dict['sample_ids_filter'] = list(config_dict['sample_ids_filter'])

        summary = {
            'dataset': self.config.dataset_name,
            'start_time': datetime.now().isoformat(),
            'config': config_dict,
            'phases': {},
            'metrics_summary': {}
        }
        
        # Load dataset
        logger.info(f"\nLoading {self.config.dataset_name}...")
        dataset_root = Path(__file__).parent.parent.parent / "datasets" / "datasets_subsets"
        dataset = InfographicVQAMiniDataset(
            str(dataset_root.resolve()),
            sample_limit=self.config.sample_limit,
            sample_ids=self.config.sample_ids_filter
        )
        logger.info(f"Loaded {len(dataset)} QA samples")
        if self.config.sample_limit:
            logger.info(f"Sample limit enforced: {self.config.sample_limit} samples max")
        if self.config.sample_ids_filter:
            logger.info(f"Sample IDs filter active: {len(self.config.sample_ids_filter)} specific IDs to process")
        
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = "test_" if self.config.test_mode else ""

            # Find the most recent existing results file for this model
            existing_files = sorted(phase_dir.glob(f"{prefix}{model}_results_*.csv"), reverse=True)

            if existing_files and self.config.sample_ids_filter:
                # COMPLETION MODE: Copy existing file with "_complete" suffix
                source_file = existing_files[0]
                model_results_file = source_file.parent / f"{source_file.stem}_complete.csv"

                # Copy existing file to new complete file if it doesn't exist
                if not model_results_file.exists():
                    import shutil
                    shutil.copy2(source_file, model_results_file)
                    logger.info(f"  Copied {source_file.name} to {model_results_file.name}")

                model_results_file = model_results_file.resolve()
                existing_ids = self._load_existing_ids(model_results_file)

                # Filter to only missing sample IDs
                samples_to_process = [s for s in dataset
                                      if s.sample_id in self.config.sample_ids_filter
                                      and s.sample_id not in existing_ids]

                logger.info(f"  COMPLETION MODE: Adding {len(samples_to_process)} missing samples to existing file")

            else:
                # NORMAL MODE: Create new timestamped file
                model_results_file = phase_dir / f"{prefix}{model}_results_{timestamp}.csv"
                model_results_file = model_results_file.resolve()
                existing_ids = self._load_existing_ids(model_results_file)
                samples_to_process = [s for s in dataset if s.sample_id not in existing_ids]

                # Enforce sample limit strictly
                if self.config.sample_limit and len(samples_to_process) > self.config.sample_limit:
                    samples_to_process = samples_to_process[:self.config.sample_limit]
                    logger.info(f"  Enforcing sample limit: processing {self.config.sample_limit} samples")

            logger.info(f"  Model: {model} ({len(samples_to_process)} samples)")

            # Determine if we're in completion mode
            append_mode = self.config.sample_ids_filter is not None

            model_results = []
            with tqdm(total=len(samples_to_process), desc=f"{phase}/{model}", unit="sample") as pbar:
                for idx, sample in enumerate(samples_to_process):
                    try:
                        result = self._process_sample(sample, phase_type, variation, model)
                        model_results.append(result)

                        # Save incrementally every 10 rows
                        if (idx + 1) % self.config.batch_size == 0:
                            self._save_results(model_results_file, model_results, append_mode=append_mode)
                            logger.info(f"  Saved file with {len(model_results)} new rows in {model_results_file}")

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
                self._save_results(model_results_file, model_results, append_mode=append_mode)
                logger.info(f"  Final save: file with {len(model_results)} new rows in {model_results_file}")

            results.extend(model_results)

        # Save embeddings separately for this phase (only if computed)
        if results and self.config.compute_embeddings:
            self._save_embeddings(phase_dir, phase, results, timestamp)

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
                # VLM Parse Pipeline: 2-step process with same VLM
                # Step 1: Parsing (image + parsing_prompt → text)
                #   - QA2a: simple parsing prompt
                #   - QA2b: detailed parsing prompt
                parse_prompt = prompts_qa.get_parsing_prompt("InfographicVQA", variation)
                extracted_text = self._extract_with_vlm(sample.image_path, parsing_model, parse_prompt)

                # Step 2: QA (text + qa_prompt → answer)
                #   - Always uses simple QA prompt (variation 'a')
                #   - Same VLM, text-only input
                prompt_used = prompts_qa.get_pipeline_qa_prompt(
                    sample.question, extracted_text, "a", "InfographicVQA"
                )
                qa_model_used = parsing_model
                prediction = self._answer_with_vlm(prompt_used, parsing_model)
                
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
        
        # Compute all metrics
        anls = compute_anls(prediction, sample.answers)
        em = compute_exact_match(prediction, sample.answers)
        substring_match = compute_substring_match(prediction, sample.answers)
        pred_in_gt = compute_prediction_in_ground_truth(prediction, sample.answers)
        gt_in_pred = compute_ground_truth_in_prediction(prediction, sample.answers)
        
        # Compute embeddings (optional, controlled by config)
        pred_embedding = None
        gt_embeddings = None
        embedding_similarity = 0.0

        if self.config.compute_embeddings:
            pred_embedding, gt_embeddings = compute_embedding_similarity(prediction, sample.answers)
            if pred_embedding and gt_embeddings:
                try:
                    from scipy.spatial.distance import cosine
                    max_sim = 0.0
                    for gt_emb in gt_embeddings:
                        sim = 1 - cosine(pred_embedding, gt_emb)
                        max_sim = max(max_sim, sim)
                    embedding_similarity = float(max_sim)
                except Exception:
                    embedding_similarity = 0.0
        
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
            substring_match=substring_match,
            prediction_in_ground_truth=pred_in_gt,
            ground_truth_in_prediction=gt_in_pred,
            embedding_similarity=embedding_similarity,
            prediction_embedding=pred_embedding,
            ground_truth_embeddings=gt_embeddings,
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

    def _answer_with_vlm(self, prompt: str, model: str) -> str:
        """Answer question using VLM (text-only, no image)."""
        # For text-only VLM calls, use the appropriate API client directly
        if model == "claude_sonnet":
            # Use Bedrock for Claude Sonnet (text-only)
            import boto3
            import json

            client = boto3.client("bedrock-runtime", region_name="us-east-1")
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            })

            response = client.invoke_model(
                modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                body=body
            )
            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"].strip()

        elif model == "claude_haiku":
            # Use Bedrock for Claude Haiku (text-only)
            import boto3
            import json

            client = boto3.client("bedrock-runtime", region_name="us-east-1")
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            })

            response = client.invoke_model(
                modelId="us.anthropic.claude-3-5-haiku-20241022-v2:0",
                body=body
            )
            response_body = json.loads(response["body"].read())
            return response_body["content"][0]["text"].strip()
        else:
            # Use Azure OpenAI for GPT models
            client = self.api.azure_openai_client
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024
            )
            return response.choices[0].message.content.strip() if response.choices[0].message.content else ""

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
    
    def _save_results(self, results_file: Path, results: List[QAResult], append_mode: bool = False):
        """Save results to CSV.

        Args:
            results_file: Path to output CSV file
            results: List of results to save
            append_mode: If True, append to file; if False, overwrite
        """
        fieldnames = [
            'sample_id', 'image_path', 'question', 'ground_truths', 'prediction',
            'phase', 'parsing_model', 'qa_model', 'extracted_text', 'prompt_used',
            'anls_score', 'exact_match', 'substring_match',
            'prediction_in_ground_truth', 'ground_truth_in_prediction',
            'embedding_similarity', 'inference_time_ms', 'error', 'timestamp'
        ]

        mode = 'a' if append_mode else 'w'
        write_header = not append_mode or not results_file.exists()

        with open(results_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())

    def _save_embeddings(self, phase_dir: Path, phase: str, results: List[QAResult], timestamp: str):
        """Save embedding vectors to separate JSON file."""
        embeddings_data = {
            'metadata': {
                'dataset': self.config.dataset_name,
                'phase': phase,
                'model_embedding': 'text-embedding-3-large',
                'embedding_dimension': 3072,
                'timestamp': datetime.now().isoformat(),
            },
            'ground_truths': {},  # GT text -> embedding vectors
            'predictions': {},    # sample_id -> {model: embedding_vector}
        }

        # Collect embeddings from results
        for result in results:
            sample_id = result.sample_id

            # Store prediction embedding
            if result.prediction_embedding:
                model_key = f"{result.parsing_model}__{result.qa_model}" if result.qa_model else result.parsing_model
                if sample_id not in embeddings_data['predictions']:
                    embeddings_data['predictions'][sample_id] = {}
                embeddings_data['predictions'][sample_id][model_key] = result.prediction_embedding

            # Store ground truth embeddings (once per unique GT)
            gt_key = str(result.ground_truths)  # JSON string as key
            if gt_key not in embeddings_data['ground_truths'] and result.ground_truth_embeddings:
                embeddings_data['ground_truths'][gt_key] = result.ground_truth_embeddings

        # Save embeddings to JSON file
        embeddings_file = phase_dir / f"embeddings_{timestamp}.json"
        with open(embeddings_file, 'w') as f:
            json.dump(embeddings_data, f, indent=2)

        logger.info(f"  Saved embeddings to: {embeddings_file.name}")

    def _load_existing_ids(self, results_file: Path) -> set:
        """Load existing sample IDs from results file that are actually complete.
        
        A sample is considered complete if:
        - It has a non-empty prediction
        - It has no error
        """
        if not results_file.exists():
            return set()
        try:
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                complete_ids = set()
                for row in reader:
                    # Only include samples with non-empty predictions and no errors
                    has_prediction = row.get('prediction', '').strip() != ''
                    has_no_error = row.get('error', '').strip() == ''
                    if has_prediction and has_no_error:
                        complete_ids.add(row['sample_id'])
                return complete_ids
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
        default=["QA1a", "QA1b", "QA1c", "QA2a", "QA2b", "QA3a", "QA3b", "QA4a", "QA4b"],
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
    parser.add_argument(
        '--results-dir',
        type=str,
        default="results/1_raw/InfographicVQA_mini",
        help='Results directory'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: prefix result files with "test_"'
    )
    parser.add_argument(
        '--sample-ids-file',
        type=str,
        default=None,
        help='JSON file with specific sample IDs to process (from missing_samples.json)'
    )

    args = parser.parse_args()

    # Load sample IDs from file if provided
    sample_ids_filter = None
    if args.sample_ids_file:
        import json
        with open(args.sample_ids_file, 'r') as f:
            missing_data = json.load(f)

        # Extract IDs for this dataset/phase/model combination
        dataset_name = "InfographicVQA_mini"
        sample_ids_filter = set()

        for phase in args.phases:
            phase_data = missing_data.get(dataset_name, {}).get(phase, {})
            # Collect IDs from all models in this phase
            for model_name, ids_list in phase_data.items():
                # Check if this model is in our requested models
                if model_name in args.ocr_models or model_name in args.vlm_models:
                    sample_ids_filter.update(ids_list)

        logger.info(f"Loaded {len(sample_ids_filter)} sample IDs from {args.sample_ids_file}")

    config = QABenchmarkConfig(
        phases=args.phases,
        sample_limit=args.sample_limit,
        ocr_models=args.ocr_models,
        vlm_models=args.vlm_models,
        qa_model=args.qa_model,
        test_mode=args.test,
        results_dir=args.results_dir,
        sample_ids_filter=sample_ids_filter
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
