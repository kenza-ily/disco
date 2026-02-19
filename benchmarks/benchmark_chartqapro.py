#!/usr/bin/env python3
"""
ChartQAPro Benchmark: Chart Visual Question Answering Evaluation

8 Benchmark Phases Testing OCR vs VLM Approaches

**QA1 (OCR Pipeline)**: Azure Intelligence OCR → GPT-5-mini QA
  - QA1a: Simple prompt
  - QA1b: Detailed prompt
  - QA1c: Chain-of-thought prompt

**QA2 (VLM Pipeline)**: VLM extraction → VLM QA
  - QA2a: Simple extraction prompt
  - QA2b: Detailed extraction prompt

**QA3 (Direct VQA)**: GPT-5-mini sees image + question directly
  - QA3a: Simple prompt
  - QA3b: Detailed prompt

Usage:
    # Run all phases
    python -m ocr_vs_vlm.benchmarks.benchmark_chartqapro
    
    # Run specific phases
    python -m ocr_vs_vlm.benchmarks.benchmark_chartqapro --phases QA1a QA3a
    
    # With sample limit
    python -m ocr_vs_vlm.benchmarks.benchmark_chartqapro --sample-limit 50
"""

import json
import logging
import sys
import time
import csv
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from datasets.dataset_loaders_qa import QASample
from models import UnifiedModelAPI
from metrics.evaluation_metrics import (
    compute_anls,
    compute_exact_match,
    compute_substring_match,
    compute_prediction_in_ground_truth,
    compute_ground_truth_in_prediction,
    compute_embedding_similarity
)

# Suppress logging output
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

EXTRACTION_PROMPTS = {
    "simple": "Extract all text and data from this chart.",
    "detailed": "Extract all text, numbers, labels, axes, legends, and data values from this chart image. Include axis labels, scale values, legend items, and any annotations.",
}

QA_PROMPTS = {
    "simple": "Question: {question}\nBased on the extracted chart text:\n{extracted_text}\n\nAnswer:",
    "detailed": "You are analyzing a chart based on the following extracted information:\n\n{extracted_text}\n\nPlease answer the following question accurately and concisely:\nQuestion: {question}\n\nAnswer:",
    "cot": "You are analyzing a chart based on the following extracted information:\n\n{extracted_text}\n\nPlease answer the following question step by step:\nQuestion: {question}\n\n1. First, identify the relevant data in the chart.\n2. Then, perform any necessary calculations.\n3. Finally, provide the answer.\n\nAnswer:",
}

DIRECT_VQA_PROMPTS = {
    "simple": "Question: {question}\n\nAnswer:",
    "detailed": "You are analyzing a chart image. Please answer the following question accurately and concisely based on what you see in the chart.\n\nQuestion: {question}\n\nAnswer:",
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ChartQAProbenchmarkConfig:
    """Configuration for ChartQAPro QA benchmark run."""
    
    # Dataset
    dataset_name: str = "ChartQAPro_mini"
    
    # Models
    ocr_models: List[str] = field(default_factory=lambda: ["azure_intelligence"])
    vlm_models: List[str] = field(default_factory=lambda: ["gpt-5-mini"])
    
    # Phases to run
    phases: List[str] = field(default_factory=lambda: [
        "QA1a", "QA1b", "QA1c",
        "QA2a", "QA2b",
        "QA3a", "QA3b"
    ])
    
    # Sample control
    sample_limit: Optional[int] = None
    batch_size: int = 10  # Save results every 10 samples

    # Output paths
    results_dir: str = "results/1_raw/chartqapro_mini"  # Fixed: removed ocr_vs_vlm prefix to avoid double path
    
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
    phase: str  # e.g., "QA1a", "QA2b", "QA3a"
    parsing_model: Optional[str] = None
    qa_model: Optional[str] = None
    
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
# BENCHMARK RUNNER
# =============================================================================

class ChartQAProBenchmark:
    """
    ChartQAPro Benchmark with three evaluation approaches and prompt variants.
    """
    
    def __init__(self, config: ChartQAProbenchmarkConfig):
        """Initialize benchmark with configuration."""
        self.config = config
        
        # Setup results directory
        if Path(config.results_dir).is_absolute():
            self.results_dir = Path(config.results_dir)
        else:
            self.results_dir = Path(__file__).parent.parent / config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API
        self.api = UnifiedModelAPI()
        
        print(f"ChartQAPro Benchmark initialized")
        print(f"  Phases: {config.phases}")
        print(f"  Sample limit: {config.sample_limit or 'All'}")
        print(f"  Results dir: {self.results_dir}")
    
    def run(self) -> Dict:
        """Execute full QA benchmark across all phases."""
        start_time = time.time()
        
        # Load dataset
        dataset = self._load_dataset()
        print(f"\nLoaded {len(dataset)} samples from {self.config.dataset_name}")
        
        all_results = []
        
        print("\n" + "="*70)
        
        # Group phases by type
        qa1_phases = [p for p in self.config.phases if p.startswith("QA1")]
        qa2_phases = [p for p in self.config.phases if p.startswith("QA2")]
        qa3_phases = [p for p in self.config.phases if p.startswith("QA3")]
        
        # QA1: OCR Pipeline (for each OCR model)
        if qa1_phases:
            for ocr_model in self.config.ocr_models:
                print(f"\nQA1: OCR Pipeline ({ocr_model} → GPT-5-mini)")
                results = self._run_qa1(dataset, qa1_phases, ocr_model)
                all_results.extend(results)
        
        # QA2: VLM Pipeline (for each VLM model)
        if qa2_phases:
            for vlm_model in self.config.vlm_models:
                print(f"\nQA2: VLM Pipeline ({vlm_model} → {vlm_model})")
                results = self._run_qa2(dataset, qa2_phases, vlm_model)
                all_results.extend(results)
        
        # QA3: Direct VQA (for each VLM model)
        if qa3_phases:
            for vlm_model in self.config.vlm_models:
                print(f"\nQA3: Direct VQA ({vlm_model})")
                results = self._run_qa3(dataset, qa3_phases, vlm_model)
                all_results.extend(results)
        
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print(f"Benchmark completed in {elapsed:.1f}s")
        
        return {'results': all_results}
    
    def _load_dataset(self) -> List[QASample]:
        """Load ChartQAPro mini dataset."""
        dataset_root = Path(__file__).parent.parent / "datasets_subsets"
        index_file = dataset_root / "chartqapro_mini" / "chartqapro_mini_index.json"
        
        if not index_file.exists():
            raise FileNotFoundError(f"ChartQAPro mini index not found: {index_file}")
        
        with open(index_file, 'r') as f:
            data = json.load(f)
        
        images_dir = dataset_root / "chartqapro_mini"
        samples = []
        
        for sample_data in data.get('samples', []):
            # Build full image path
            image_path = images_dir / sample_data['image_path']
            
            # Extract answers
            answers = sample_data.get('answers', [])
            if isinstance(answers, str):
                answers = [answers]
            
            # Clean up answers if they're wrapped in lists
            if answers and isinstance(answers[0], list):
                answers = [a[0] if isinstance(a, list) else a for a in answers]
            
            # Get question
            question = sample_data.get('question', '')
            if isinstance(question, list):
                question = question[0] if question else ''
            
            # Get ground truth
            ground_truth = sample_data.get('ground_truth', answers[0] if answers else '')
            if isinstance(ground_truth, list):
                ground_truth = ground_truth[0] if ground_truth else ''
            
            sample = QASample(
                sample_id=sample_data['sample_id'],
                image_path=str(image_path),
                question=question,
                answers=answers,
                ground_truth=ground_truth,
                question_type=sample_data.get('question_type', ''),
                metadata=sample_data.get('metadata', {})
            )
            samples.append(sample)
        
        # Apply sample limit
        if self.config.sample_limit:
            samples = samples[:self.config.sample_limit]
        
        return samples
    
    def _run_qa1(self, dataset: List[QASample], phases: List[str], ocr_model: str) -> List[QAResult]:
        """Run QA1: OCR Pipeline with prompt variants."""
        all_results = []
        qa_model = "gpt-5-mini"  # Always use GPT-5-mini for answering
        
        for phase in phases:
            variant = phase[-1].lower()  # 'a', 'b', 'c'
            
            # Create phase directory
            phase_dir = self.results_dir / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            
            # Create CSV file with model name prefix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = phase_dir / f"{ocr_model}_results_{timestamp}.csv"
            
            results = []
            
            with tqdm(total=len(dataset), desc=f"{ocr_model} {phase}", unit="sample") as pbar:
                for idx, sample in enumerate(dataset):
                    try:
                        result = self._process_qa1_sample(sample, phase, variant, ocr_model, qa_model)
                        results.append(result)
                        all_results.append(result)
                        
                        # Save every batch_size samples
                        if (idx + 1) % self.config.batch_size == 0:
                            self._save_results(results_file, results)
                            results.clear()  # FIX: Clear results after saving to prevent duplication

                        pbar.update(1)
                    except Exception as e:
                        result = QAResult(
                            sample_id=sample.sample_id,
                            image_path=sample.image_path,
                            question=sample.question,
                            ground_truths=sample.answers,
                            prediction="",
                            phase=phase,
                            parsing_model=ocr_model,
                            qa_model=qa_model,
                            error=str(e),
                            timestamp=datetime.now().isoformat()
                        )
                        results.append(result)
                        all_results.append(result)
                        pbar.update(1)
            
            # Final save for this phase
            if results:
                self._save_results(results_file, results)
        
        return all_results
    
    def _run_qa2(self, dataset: List[QASample], phases: List[str], vlm_model: str) -> List[QAResult]:
        """Run QA2: VLM Pipeline with prompt variants."""
        all_results = []
        
        for phase in phases:
            variant = phase[-1].lower()  # 'a', 'b', 'c'
            
            # Create phase directory
            phase_dir = self.results_dir / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            
            # Create CSV file with model name prefix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = phase_dir / f"{vlm_model}_results_{timestamp}.csv"
            
            results = []
            
            with tqdm(total=len(dataset), desc=f"{vlm_model} {phase}", unit="sample") as pbar:
                for idx, sample in enumerate(dataset):
                    try:
                        result = self._process_qa2_sample(sample, phase, variant, vlm_model)
                        results.append(result)
                        all_results.append(result)
                        
                        # Save every batch_size samples
                        if (idx + 1) % self.config.batch_size == 0:
                            self._save_results(results_file, results)
                            results.clear()  # FIX: Clear results after saving to prevent duplication

                        pbar.update(1)
                    except Exception as e:
                        result = QAResult(
                            sample_id=sample.sample_id,
                            image_path=sample.image_path,
                            question=sample.question,
                            ground_truths=sample.answers,
                            prediction="",
                            phase=phase,
                            parsing_model=vlm_model,
                            qa_model=vlm_model,
                            error=str(e),
                            timestamp=datetime.now().isoformat()
                        )
                        results.append(result)
                        all_results.append(result)
                        pbar.update(1)
            
            # Final save for this phase
            if results:
                self._save_results(results_file, results)
        
        return all_results
    
    def _run_qa3(self, dataset: List[QASample], phases: List[str], vlm_model: str) -> List[QAResult]:
        """Run QA3: Direct VQA with prompt variants."""
        all_results = []
        
        for phase in phases:
            variant = phase[-1].lower()  # 'a', 'b'
            
            # Create phase directory
            phase_dir = self.results_dir / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            
            # Create CSV file with model name prefix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = phase_dir / f"{vlm_model}_results_{timestamp}.csv"
            
            results = []
            
            with tqdm(total=len(dataset), desc=f"{vlm_model} {phase}", unit="sample") as pbar:
                for idx, sample in enumerate(dataset):
                    try:
                        result = self._process_qa3_sample(sample, phase, variant, vlm_model)
                        results.append(result)
                        all_results.append(result)
                        
                        # Save every batch_size samples
                        if (idx + 1) % self.config.batch_size == 0:
                            self._save_results(results_file, results)
                            results.clear()  # FIX: Clear results after saving to prevent duplication

                        pbar.update(1)
                    except Exception as e:
                        result = QAResult(
                            sample_id=sample.sample_id,
                            image_path=sample.image_path,
                            question=sample.question,
                            ground_truths=sample.answers,
                            prediction="",
                            phase=phase,
                            parsing_model=vlm_model,
                            qa_model=vlm_model,
                            error=str(e),
                            timestamp=datetime.now().isoformat()
                        )
                        results.append(result)
                        all_results.append(result)
                        pbar.update(1)
            
            # Final save for this phase
            if results:
                self._save_results(results_file, results)
        
        return all_results
    
    def _process_qa1_sample(self, sample: QASample, phase: str, variant: str, ocr_model: str, qa_model: str) -> QAResult:
        """Process QA1: OCR + LLM answering."""
        start_time = time.time()
        extracted_text = None
        prediction = ""
        prompt_used = None
        error = None
        
        try:
            # Step 1: Extract with OCR
            extracted_text = self._extract_with_ocr(sample.image_path, ocr_model)
            
            # Step 2: Answer with LLM using extracted text
            qa_prompt_template = QA_PROMPTS.get(variant, QA_PROMPTS["simple"])
            prompt_used = qa_prompt_template.format(
                question=sample.question,
                extracted_text=extracted_text
            )
            
            prediction = self._answer_with_llm(prompt_used, qa_model)
            
        except Exception as e:
            error = str(e)
            if self.config.retry_failed:
                for retry in range(self.config.max_retries):
                    try:
                        time.sleep(2 ** retry)
                        extracted_text = self._extract_with_ocr(sample.image_path, ocr_model)
                        qa_prompt_template = QA_PROMPTS.get(variant, QA_PROMPTS["simple"])
                        prompt_used = qa_prompt_template.format(
                            question=sample.question,
                            extracted_text=extracted_text
                        )
                        prediction = self._answer_with_llm(prompt_used, qa_model)
                        error = None
                        break
                    except Exception:
                        pass
        
        elapsed_ms = (time.time() - start_time) * 1000
        prediction = self._clean_prediction(prediction)
        
        # Compute metrics
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
                        sim = 1.0 - cosine(pred_embedding, gt_emb)
                        max_sim = max(max_sim, sim)
                    embedding_similarity = max_sim
                except Exception:
                    pass
        
        return QAResult(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            question=sample.question,
            ground_truths=sample.answers,
            prediction=prediction,
            phase=phase,
            parsing_model=ocr_model,
            qa_model=qa_model,
            extracted_text=extracted_text,
            prompt_used=prompt_used,
            anls_score=anls,
            exact_match=em,
            substring_match=substring_match,
            prediction_in_ground_truth=pred_in_gt,
            ground_truth_in_prediction=gt_in_pred,
            embedding_similarity=embedding_similarity,
            inference_time_ms=elapsed_ms,
            error=error,
            timestamp=datetime.now().isoformat()
        )
    
    def _process_qa2_sample(self, sample: QASample, phase: str, variant: str, vlm_model: str) -> QAResult:
        """Process QA2: VLM extraction + LLM answering."""
        start_time = time.time()
        extracted_text = None
        prediction = ""
        prompt_used = None
        error = None
        
        try:
            # Step 1: Extract with VLM (QA2a=simple, QA2b=detailed extraction)
            extraction_key = "simple" if variant == "a" else "detailed"
            extraction_prompt = EXTRACTION_PROMPTS[extraction_key]
            extracted_text = self._extract_with_vlm(sample.image_path, vlm_model, extraction_prompt)
            
            # Step 2: Answer with LLM using extracted text (always simple QA prompt)
            qa_prompt_template = QA_PROMPTS["simple"]
            prompt_used = qa_prompt_template.format(
                question=sample.question,
                extracted_text=extracted_text
            )
            
            prediction = self._answer_with_llm(prompt_used, vlm_model)
            
        except Exception as e:
            error = str(e)
            if self.config.retry_failed:
                for retry in range(self.config.max_retries):
                    try:
                        time.sleep(2 ** retry)
                        extraction_key = "simple" if variant == "a" else "detailed"
                        extraction_prompt = EXTRACTION_PROMPTS[extraction_key]
                        extracted_text = self._extract_with_vlm(sample.image_path, vlm_model, extraction_prompt)
                        qa_prompt_template = QA_PROMPTS["simple"]
                        prompt_used = qa_prompt_template.format(
                            question=sample.question,
                            extracted_text=extracted_text
                        )
                        prediction = self._answer_with_llm(prompt_used, vlm_model)
                        error = None
                        break
                    except Exception:
                        pass
        
        elapsed_ms = (time.time() - start_time) * 1000
        prediction = self._clean_prediction(prediction)
        
        # Compute metrics
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
                        sim = 1.0 - cosine(pred_embedding, gt_emb)
                        max_sim = max(max_sim, sim)
                    embedding_similarity = max_sim
                except Exception:
                    pass
        
        return QAResult(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            question=sample.question,
            ground_truths=sample.answers,
            prediction=prediction,
            phase=phase,
            parsing_model=vlm_model,
            qa_model=vlm_model,
            extracted_text=extracted_text,
            prompt_used=prompt_used,
            anls_score=anls,
            exact_match=em,
            substring_match=substring_match,
            prediction_in_ground_truth=pred_in_gt,
            ground_truth_in_prediction=gt_in_pred,
            embedding_similarity=embedding_similarity,
            inference_time_ms=elapsed_ms,
            error=error,
            timestamp=datetime.now().isoformat()
        )
    
    def _process_qa3_sample(self, sample: QASample, phase: str, variant: str, vlm_model: str) -> QAResult:
        """Process QA3: Direct VQA."""
        start_time = time.time()
        prediction = ""
        prompt_used = None
        error = None
        
        try:
            # Direct VQA: VLM sees image + question
            prompt_template = DIRECT_VQA_PROMPTS.get(variant, DIRECT_VQA_PROMPTS["simple"])
            prompt_used = prompt_template.format(question=sample.question)
            
            prediction = self._direct_vqa(sample.image_path, vlm_model, prompt_used)
            
        except Exception as e:
            error = str(e)
            if self.config.retry_failed:
                for retry in range(self.config.max_retries):
                    try:
                        time.sleep(2 ** retry)
                        prompt_template = DIRECT_VQA_PROMPTS.get(variant, DIRECT_VQA_PROMPTS["simple"])
                        prompt_used = prompt_template.format(question=sample.question)
                        prediction = self._direct_vqa(sample.image_path, vlm_model, prompt_used)
                        error = None
                        break
                    except Exception:
                        pass
        
        elapsed_ms = (time.time() - start_time) * 1000
        prediction = self._clean_prediction(prediction)
        
        # Compute metrics
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
                        sim = 1.0 - cosine(pred_embedding, gt_emb)
                        max_sim = max(max_sim, sim)
                    embedding_similarity = max_sim
                except Exception:
                    pass
        
        return QAResult(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            question=sample.question,
            ground_truths=sample.answers,
            prediction=prediction,
            phase=phase,
            parsing_model=vlm_model,
            qa_model=vlm_model,
            prompt_used=prompt_used,
            anls_score=anls,
            exact_match=em,
            substring_match=substring_match,
            prediction_in_ground_truth=pred_in_gt,
            ground_truth_in_prediction=gt_in_pred,
            embedding_similarity=embedding_similarity,
            inference_time_ms=elapsed_ms,
            error=error,
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_with_ocr(self, image_path: str, model: str) -> str:
        """Extract text from image using OCR model."""
        response = self.api.process(image_path, model=model)
        if response.error:
            raise Exception(response.error)
        return response.content
    
    def _extract_with_vlm(self, image_path: str, model: str, prompt: str) -> str:
        """Extract text from image using VLM."""
        response = self.api.process(image_path, model=model, query=prompt)
        if response.error:
            raise Exception(response.error)
        return response.content
    
    def _answer_with_llm(self, prompt: str, model: str) -> str:
        """Answer question using LLM (text-only)."""
        try:
            # Import here to avoid issues if not available
            from openai import AzureOpenAI
            
            client = AzureOpenAI(
                api_key=os.environ.get("AZURE_OPENAI_KEY"),
                api_version="2024-05-01-preview",
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
            )
            
            # Map model names to deployment names
            deployment_map = {
                "gpt-5-mini": "gpt-5-mini",
                "gpt-5-nano": "gpt-5-nano",
                "claude_sonnet": "claude-3-5-sonnet",
            }
            
            deployment = deployment_map.get(model, model)
            
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024
            )
            
            return response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        except Exception as e:
            raise Exception(f"LLM answering failed: {e}")
    
    def _direct_vqa(self, image_path: str, model: str, prompt: str) -> str:
        """Direct VQA: VLM sees image and answers question."""
        response = self.api.process(image_path, model=model, query=prompt)
        if response.error:
            raise Exception(response.error)
        return response.content
    
    def _clean_prediction(self, prediction: str) -> str:
        """Clean prediction."""
        if not prediction:
            return ""
        
        prediction = prediction.strip()
        
        # Remove markdown code blocks
        if prediction.startswith('```'):
            lines = prediction.split('\n')
            prediction = '\n'.join(lines[1:-1]) if len(lines) > 2 else prediction
        
        # Remove quotes
        if (prediction.startswith('"') and prediction.endswith('"')) or \
           (prediction.startswith("'") and prediction.endswith("'")):
            prediction = prediction[1:-1]
        
        return prediction.strip()
    
    def _save_results(self, results_file: Path, results: List[QAResult]):
        """Save results to CSV."""
        fieldnames = [
            'sample_id', 'image_path', 'question', 'ground_truths', 'prediction',
            'phase', 'parsing_model', 'qa_model', 'extracted_text', 'prompt_used',
            'anls_score', 'exact_match', 'substring_match',
            'prediction_in_ground_truth', 'ground_truth_in_prediction',
            'embedding_similarity', 'inference_time_ms', 'error', 'timestamp'
        ]
        
        file_exists = results_file.exists()
        
        with open(results_file, 'a' if file_exists else 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for result in results:
                writer.writerow(result.to_dict())


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run ChartQAPro benchmark from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ChartQAPro QA Benchmark')
    parser.add_argument(
        '--phases', '-p',
        nargs='+',
        default=["QA1a", "QA1b", "QA1c", "QA2a", "QA2b", "QA2c", "QA3a", "QA3b"],
        help='Phases to run (default: all)'
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
        default=["azure_intelligence"],
        help='OCR models for QA1 approach'
    )
    parser.add_argument(
        '--vlm-models',
        nargs='+',
        default=["gpt-5-mini"],
        help='VLM models for QA2 and QA3'
    )
    
    args = parser.parse_args()
    
    config = ChartQAProbenchmarkConfig(
        phases=args.phases,
        sample_limit=args.sample_limit,
        ocr_models=args.ocr_models,
        vlm_models=args.vlm_models
    )
    
    benchmark = ChartQAProBenchmark(config)
    summary = benchmark.run()
    
    print("\n" + "="*70)
    print("CHARTQAPRO BENCHMARK COMPLETED")
    print("="*70)


if __name__ == '__main__':
    main()
