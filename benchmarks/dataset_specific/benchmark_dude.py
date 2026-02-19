#!/usr/bin/env python3
"""
DUDE Benchmark: Document Understanding for Diverse Environments QA Evaluation

Evaluates OCR and VLM models on DUDE_mini dataset using three approaches:

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

Dataset Characteristics:
- DUDE_mini: 404 samples stratified across 5 question families
- Question families: numeric_amount, date_time, lookup_entity, yes_no, multi_hop_other
- Real-world documents: diverse layouts, multiple languages, mixed quality
- Enables analysis of OCR vs VLM tradeoffs on challenging, real-world documents

Models:
- OCR: azure_intelligence, mistral_document_ai
- VLM: gpt-5-mini, gpt-5-nano

Metrics:
- ANLS: Average Normalized Levenshtein Similarity
- Exact Match: Percentage of exact answer matches
- Substring Match: Whether prediction/answer appear in each other
- Embedding Similarity: Semantic similarity via embeddings

Usage:
    # Run all phases
    python -m ocr_vs_vlm.benchmarks.benchmark_dude
    
    # Run specific phases
    python -m ocr_vs_vlm.benchmarks.benchmark_dude --phases QA1a QA3a
    
    # With sample limit
    python -m ocr_vs_vlm.benchmarks.benchmark_dude --sample-limit 50
    
    # Specific models
    python -m ocr_vs_vlm.benchmarks.benchmark_dude --vlm-models gpt-5-mini
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
import pandas as pd
from pdf2image import convert_from_path
import tempfile

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
    
    log_file = results_dir / 'benchmark_dude.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DUDEBenchmarkConfig:
    """Configuration for DUDE benchmark run."""
    
    # Dataset
    dataset_name: str = "DUDE_mini"
    
    # Models for each approach
    ocr_models: List[str] = field(default_factory=lambda: ["azure_intelligence", "mistral_document_ai", "mistral_ocr_3"])
    vlm_models: List[str] = field(default_factory=lambda: ["gpt-5-mini", "gpt-5-nano"])
    qa_model: str = "gpt-5-mini"  # Model for answering questions in pipeline approaches
    
    # Phases to run
    phases: List[str] = field(default_factory=lambda: [
        "QA1a", "QA1b", "QA1c",  # OCR Pipeline
        "QA2a", "QA2b", "QA2c",  # VLM Parse Pipeline  
        "QA3a", "QA3b"           # Direct VQA
    ])
    
    # Sample control
    sample_limit: Optional[int] = None
    batch_size: int = 10  # Changed from 50 to save every 10 rows consistently
    
    # Output paths
    results_dir: str = "results/raw/dude_mini"
    
    # API settings
    retry_failed: bool = True
    max_retries: int = 2

    # Embedding settings
    compute_embeddings: bool = False  # Set to True to compute embeddings during benchmark


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class DUDEQAResult:
    """Result from a single DUDE QA evaluation."""
    
    sample_id: str
    doc_id: str
    question: str
    question_family: str
    answer_type: str
    ground_truths: List[str]
    prediction: str
    
    # Phase info
    phase: str = ""  # e.g., "QA1a", "QA2b", "QA3a" - set after creation
    parsing_model: Optional[str] = None  # OCR/VLM for extraction
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
            'doc_id': self.doc_id,
            'question': self.question,
            'question_family': self.question_family,
            'answer_type': self.answer_type,
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
# DUDE DATASET LOADER
# =============================================================================

class DUDEMiniDataset:
    """
    DUDE Mini Dataset loader.
    
    Loads 404 QA samples from DUDE_mini, stratified across question families:
    - numeric_amount (20%)
    - date_time (15%)
    - lookup_entity (40%)
    - yes_no (15%)
    - multi_hop_other (10%)
    """
    
    def __init__(self, dataset_root: str, pdf_root: str, sample_limit: Optional[int] = None):
        """
        Initialize DUDE mini loader.
        
        Args:
            dataset_root: Root path to datasets_subsets folder
            pdf_root: Root path to DUDE PDF files
            sample_limit: Max samples to load (None = all 404)
        """
        self.dataset_root = Path(dataset_root)
        self.pdf_root = Path(pdf_root)
        self.sample_limit = sample_limit
        self.samples: List[Dict] = []
        self._load()
        
        if sample_limit and len(self.samples) > sample_limit:
            self.samples = self.samples[:sample_limit]
        
        logger.info(f"Loaded {len(self.samples)} samples from DUDE_mini")
    
    def _load(self):
        """Load samples from CSV file."""
        csv_file = self.dataset_root / "dude_mini" / "dude_mini.csv"
        
        if not csv_file.exists():
            raise FileNotFoundError(f"DUDE mini CSV not found: {csv_file}")
        
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            # Find the PDF file
            doc_id = row['docId']
            pdf_path = self.pdf_root / "sample_pdfs" / "sample" / f"{doc_id}.pdf"
            
            if not pdf_path.exists():
                logger.warning(f"PDF not found for doc_id {doc_id}, skipping")
                continue
            
            # Parse answers (they come as strings but are actually lists)
            try:
                if isinstance(row['answers'], str) and row['answers'].startswith('['):
                    answers = json.loads(row['answers'].replace("'", '"'))
                else:
                    answers = [row['answers']]
            except:
                answers = [str(row['answers'])]
            
            # Infer question family if not present
            question_family = self._infer_question_family(row['question'])
            
            sample = {
                'questionId': row['questionId'],
                'docId': row['docId'],
                'question': row['question'],
                'answers': answers,
                'answer_type': row['answer_type'],
                'question_family': question_family,
                'pdf_path': str(pdf_path),
            }
            self.samples.append(sample)
    
    def _infer_question_family(self, question: str) -> str:
        """Infer question family from question text."""
        ql = (question or "").strip().lower()
        
        if ql.startswith(("is ", "are ", "does ", "do ", "did ", "can ", "could ", "should ", "has ", "have ", "was ", "were ")):
            return "yes_no"
        if any(tok in ql for tok in [" when ", " date", " dated", " expiry", " effective", " issued"]):
            return "date_time"
        if any(tok in ql for tok in [" how many", " total", " amount", " sum", " price", " cost", " balance", " due", "£", "$", "€", " percent", "%"]):
            return "numeric_amount"
        if any(tok in ql for tok in [" between", " difference", " before", " after", " and ", " both ", " compared", " compare"]):
            return "multi_hop_other"
        return "lookup_entity"
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
    
    def get_stats(self) -> Dict:
        """Return dataset statistics."""
        question_families = {}
        answer_types = {}
        for s in self.samples:
            qf = s.get('question_family', 'unknown')
            question_families[qf] = question_families.get(qf, 0) + 1
            
            at = s.get('answer_type', 'unknown')
            answer_types[at] = answer_types.get(at, 0) + 1
        
        return {
            'total_samples': len(self.samples),
            'question_families': question_families,
            'answer_types': answer_types,
            'unique_documents': len(set(s['docId'] for s in self.samples)),
            'avg_question_length': sum(len(s['question']) for s in self.samples) / len(self.samples) if self.samples else 0,
        }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class DUDEBenchmark:
    """
    DUDE Benchmark with three evaluation approaches.
    
    Approaches:
    1. OCR Pipeline (QA1): OCR extraction → LLM QA
    2. VLM Parse Pipeline (QA2): VLM extraction → LLM QA
    3. Direct VQA (QA3): VLM sees image + question together
    
    Key advantage of DUDE: real-world document diversity enables analysis
    of where OCR pipelines excel (precise digit extraction) vs VLMs
    (visual context, multi-language handling).
    """
    
    def __init__(self, config: DUDEBenchmarkConfig):
        """Initialize benchmark with configuration."""
        self.config = config
        
        # Setup results directory
        if Path(config.results_dir).is_absolute():
            self.results_dir = Path(config.results_dir)
        else:
            # Resolve relative to project root (parent of benchmarks/)
            self.results_dir = Path(__file__).parent.parent / config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        _configure_logging(self.results_dir)
        
        # Initialize API
        self.api = UnifiedModelAPI()
        
        logger.info(f"DUDE Benchmark initialized")
        logger.info(f"  Phases: {config.phases}")
        logger.info(f"  OCR models: {config.ocr_models}")
        logger.info(f"  VLM models: {config.vlm_models}")
        logger.info(f"  QA model: {config.qa_model}")
        logger.info(f"  Sample limit: {config.sample_limit or 'All'}")
        logger.info(f"  Results dir: {self.results_dir}")
    
    def run(self) -> Dict:
        """Execute full DUDE benchmark across all phases."""
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
        # CSV is in ocr_vs_vlm/datasets/datasets_subsets/ (3 levels up from this file)
        dataset_root = Path(__file__).parent.parent.parent / "datasets" / "datasets_subsets"
        # PDFs are in repo_root/datasets/task2_QA/DUDE/ (4 levels up from this file)
        pdf_root = Path(__file__).parent.parent.parent.parent / "datasets" / "task2_QA" / "DUDE"
        
        try:
            dataset = DUDEMiniDataset(
                str(dataset_root),
                str(pdf_root),
                sample_limit=self.config.sample_limit
            )
        except Exception as e:
            logger.error(f"Failed to load DUDE dataset: {e}")
            raise
        
        logger.info(f"Loaded {len(dataset)} QA samples")
        stats = dataset.get_stats()
        logger.info(f"Dataset stats: {stats}")
        
        # Run each phase
        all_results: Dict[str, List[DUDEQAResult]] = {}
        
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
        
        # Save results grouped by model
        self._save_results_by_model(all_results)
        
        # Aggregate metrics
        summary['metrics_summary'] = self._aggregate_metrics(all_results)
        summary['end_time'] = datetime.now().isoformat()
        summary['total_time_seconds'] = time.time() - start_time
        
        # Save summary
        summary_file = self.results_dir / "execution_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nExecution summary saved to {summary_file}")
        
        return summary
    
    def _run_phase(self, phase: str, dataset: DUDEMiniDataset) -> List[DUDEQAResult]:
        """Execute a single phase across all samples."""
        results = []
        
        for sample in tqdm(dataset, desc=f"Processing {phase}", unit="sample"):
            sample_id = sample['questionId']
            
            try:
                result = self._process_sample(phase, sample)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing sample {sample_id}: {e}")
                result = DUDEQAResult(
                    sample_id=sample_id,
                    doc_id=sample['docId'],
                    question=sample['question'],
                    question_family=sample['question_family'],
                    answer_type=sample['answer_type'],
                    ground_truths=sample['answers'],
                    prediction="",
                    phase=phase,
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )
                results.append(result)
        
        return results
    
    def _process_sample(self, phase: str, sample: Dict) -> DUDEQAResult:
        """Process a single sample for a given phase."""
        sample_id = sample['questionId']
        start_time = time.time()
        
        try:
            # Convert PDF to image if needed
            image_path = self._get_image_path(sample['pdf_path'])
            
            # Extract phase components
            approach, prompt_variant = phase[:-1], phase[-1]  # e.g., "QA1" and "a"
            
            if approach == "QA1":
                # OCR Pipeline: OCR extraction → LLM QA
                result = self._qa1_ocr_pipeline(sample, prompt_variant, image_path)
            elif approach == "QA2":
                # VLM Parse Pipeline: VLM extraction → LLM QA
                result = self._qa2_vlm_pipeline(sample, prompt_variant, image_path)
            elif approach == "QA3":
                # Direct VQA: VLM sees image + question
                result = self._qa3_direct_vqa(sample, prompt_variant, image_path)
            else:
                raise ValueError(f"Unknown approach: {approach}")
            
            result.phase = phase
            result.inference_time_ms = (time.time() - start_time) * 1000
            result.timestamp = datetime.now().isoformat()
            
            # Compute metrics
            self._compute_result_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {phase} for sample {sample_id}: {e}", exc_info=True)
            try:
                error_result = DUDEQAResult(
                    sample_id=sample_id,
                    doc_id=sample['docId'],
                    question=sample['question'],
                    question_family=sample['question_family'],
                    answer_type=sample['answer_type'],
                    ground_truths=sample['answers'],
                    prediction="",
                    phase=phase,
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )
                error_result.inference_time_ms = (time.time() - start_time) * 1000
                return error_result
            except Exception as e2:
                logger.error(f"Failed to create error result: {e2}", exc_info=True)
                # Fallback: try a minimal error result
                error_result = DUDEQAResult(
                    sample_id=sample_id,
                    doc_id=sample.get('docId', 'unknown'),
                    question=sample.get('question', ''),
                    question_family=sample.get('question_family', 'unknown'),
                    answer_type=sample.get('answer_type', 'unknown'),
                    ground_truths=sample.get('answers', []),
                    prediction="",
                    phase=phase,
                    error=f"Double error: {e} | {e2}",
                    timestamp=datetime.now().isoformat()
                )
                error_result.inference_time_ms = (time.time() - start_time) * 1000
                return error_result
    
    def _get_image_path(self, pdf_path: str) -> str:
        """
        Convert PDF to image if needed, return image path.
        Uses first page of PDF if multi-page.
        """
        pdf_path = Path(pdf_path)
        
        if pdf_path.suffix.lower() == '.pdf':
            # Convert PDF to image
            try:
                images = convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=150)
                if images:
                    # Save as temporary PNG
                    temp_dir = Path(tempfile.gettempdir()) / "dude_benchmark"
                    temp_dir.mkdir(exist_ok=True)
                    
                    image_path = temp_dir / f"{pdf_path.stem}.png"
                    images[0].save(image_path, 'PNG')
                    return str(image_path)
            except Exception as e:
                logger.warning(f"Failed to convert PDF to image: {e}, using PDF directly")
        
        return str(pdf_path)
    
    def _qa1_ocr_pipeline(self, sample: Dict, prompt_variant: str, image_path: str) -> DUDEQAResult:
        """OCR Pipeline: Extract text with OCR, then LLM answers."""
        # Extract OCR model (use first available)
        ocr_model = self.config.ocr_models[0] if self.config.ocr_models else "azure_intelligence"
        
        # Process image with OCR to extract text
        try:
            ocr_response = self.api.process(image_path, model=ocr_model)
            extracted_text = ocr_response.content
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            extracted_text = ""
        
        # Get QA prompt based on variant
        if prompt_variant == 'a':
            from prompts.prompts_qa import get_qa_simple_prompt
            prompt = get_qa_simple_prompt(sample['question'], extracted_text)
        elif prompt_variant == 'b':
            from prompts.prompts_qa import get_qa_detailed_prompt
            prompt = get_qa_detailed_prompt(sample['question'], extracted_text)
        else:  # 'c'
            from prompts.prompts_qa import get_qa_cot_prompt
            prompt = get_qa_cot_prompt(sample['question'], extracted_text)
        
        # Answer question using LLM
        try:
            qa_response = self.api.process(
                image_path,  # For text-only queries, this is ignored but required
                model=self.config.qa_model,
                query=prompt
            )
            prediction = qa_response.content
        except Exception as e:
            logger.error(f"QA inference failed: {e}")
            prediction = ""
        
        return DUDEQAResult(
            sample_id=sample['questionId'],
            doc_id=sample['docId'],
            question=sample['question'],
            question_family=sample['question_family'],
            answer_type=sample['answer_type'],
            ground_truths=sample['answers'],
            prediction=prediction,
            parsing_model=ocr_model,
            qa_model=self.config.qa_model,
            extracted_text=extracted_text,
            prompt_used=prompt
        )
    
    def _qa2_vlm_pipeline(self, sample: Dict, prompt_variant: str, image_path: str) -> DUDEQAResult:
        """VLM Parse Pipeline: VLM extracts text, then LLM answers."""
        # Extract using VLM (use first available)
        vlm_model = self.config.vlm_models[0] if self.config.vlm_models else "gpt-5-mini"
        
        # Get extraction prompt
        extraction_query = "Extract all text and content from this document. Be comprehensive and preserve formatting."
        
        # Extract text from image using VLM
        try:
            vlm_response = self.api.process(
                image_path,
                model=vlm_model,
                query=extraction_query
            )
            extracted_text = vlm_response.content
        except Exception as e:
            logger.error(f"VLM extraction failed: {e}")
            extracted_text = ""
        
        # Get QA prompt based on variant
        if prompt_variant == 'a':
            from prompts.prompts_qa import get_qa_simple_prompt
            prompt = get_qa_simple_prompt(sample['question'], extracted_text)
        elif prompt_variant == 'b':
            from prompts.prompts_qa import get_qa_detailed_prompt
            prompt = get_qa_detailed_prompt(sample['question'], extracted_text)
        else:  # 'c'
            from prompts.prompts_qa import get_qa_cot_prompt
            prompt = get_qa_cot_prompt(sample['question'], extracted_text)
        
        # Answer question using LLM
        try:
            qa_response = self.api.process(
                image_path,
                model=self.config.qa_model,
                query=prompt
            )
            prediction = qa_response.content
        except Exception as e:
            logger.error(f"QA inference failed: {e}")
            prediction = ""
        
        return DUDEQAResult(
            sample_id=sample['questionId'],
            doc_id=sample['docId'],
            question=sample['question'],
            question_family=sample['question_family'],
            answer_type=sample['answer_type'],
            ground_truths=sample['answers'],
            prediction=prediction,
            parsing_model=vlm_model,
            qa_model=self.config.qa_model,
            extracted_text=extracted_text,
            prompt_used=prompt
        )
    
    def _qa3_direct_vqa(self, sample: Dict, prompt_variant: str, image_path: str) -> DUDEQAResult:
        """Direct VQA: VLM sees image + question together."""
        # Use VLM directly on image + question
        vlm_model = self.config.vlm_models[0] if self.config.vlm_models else "gpt-5-mini"
        
        # Get prompt for this variant
        if prompt_variant == 'a':
            from prompts.prompts_qa import get_direct_vqa_simple_prompt
            prompt = get_direct_vqa_simple_prompt(sample['question'])
        else:  # 'b'
            from prompts.prompts_qa import get_direct_vqa_detailed_prompt
            prompt = get_direct_vqa_detailed_prompt(sample['question'])
        
        # Query VLM with image and question
        try:
            vqa_response = self.api.process(
                image_path,
                model=vlm_model,
                query=prompt
            )
            prediction = vqa_response.content
        except Exception as e:
            logger.error(f"Direct VQA inference failed: {e}")
            prediction = ""
        
        return DUDEQAResult(
            sample_id=sample['questionId'],
            doc_id=sample['docId'],
            question=sample['question'],
            question_family=sample['question_family'],
            answer_type=sample['answer_type'],
            ground_truths=sample['answers'],
            prediction=prediction,
            parsing_model=None,  # Not applicable for direct VQA
            qa_model=vlm_model,
            prompt_used=prompt
        )
    
    def _compute_result_metrics(self, result: DUDEQAResult):
        """Compute evaluation metrics for a result."""
        result.anls_score = compute_anls(result.prediction, result.ground_truths)
        result.exact_match = compute_exact_match(result.prediction, result.ground_truths)
        result.substring_match = compute_substring_match(result.prediction, result.ground_truths)
        result.prediction_in_ground_truth = compute_prediction_in_ground_truth(result.prediction, result.ground_truths)
        result.ground_truth_in_prediction = compute_ground_truth_in_prediction(result.prediction, result.ground_truths)
        
        # Compute embedding similarity (optional, controlled by config)
        if self.config.compute_embeddings:
            try:
                result.embedding_similarity = compute_embedding_similarity(result.prediction, result.ground_truths)
            except:
                result.embedding_similarity = 0.0
        else:
            result.embedding_similarity = 0.0
    
    def _compute_metrics(self, results: List[DUDEQAResult]) -> Dict:
        """Compute aggregate metrics for a set of results."""
        if not results:
            return {}
        
        scores = {
            'anls': [r.anls_score for r in results if r.error is None],
            'exact_match': [r.exact_match for r in results if r.error is None],
            'substring_match': [r.substring_match for r in results if r.error is None],
        }
        
        return {
            'anls': sum(scores['anls']) / len(scores['anls']) if scores['anls'] else 0.0,
            'exact_match': sum(scores['exact_match']) / len(scores['exact_match']) if scores['exact_match'] else 0.0,
            'substring_match': sum(scores['substring_match']) / len(scores['substring_match']) if scores['substring_match'] else 0.0,
            'total_samples': len(results),
            'error_count': sum(1 for r in results if r.error),
        }
    
    def _aggregate_metrics(self, all_results: Dict[str, List[DUDEQAResult]]) -> Dict:
        """Aggregate metrics across all phases."""
        aggregated = {}
        
        for phase, results in all_results.items():
            aggregated[phase] = self._compute_metrics(results)
        
        # Group by approach
        approaches = {
            'OCR_Pipeline': ['QA1a', 'QA1b', 'QA1c'],
            'VLM_Pipeline': ['QA2a', 'QA2b', 'QA2c'],
            'Direct_VQA': ['QA3a', 'QA3b'],
        }
        
        by_approach = {}
        for approach_name, phases in approaches.items():
            relevant_results = []
            for phase in phases:
                if phase in all_results:
                    relevant_results.extend(all_results[phase])
            
            if relevant_results:
                metrics = self._compute_metrics(relevant_results)
                by_approach[approach_name] = {
                    'avg_anls': metrics['anls'],
                    'avg_em': metrics['exact_match'],
                    'samples': metrics['total_samples'],
                }
        
        aggregated['by_approach'] = by_approach
        
        # By question family
        by_family = {}
        for family in ['numeric_amount', 'date_time', 'lookup_entity', 'yes_no', 'multi_hop_other']:
            family_results = []
            for results in all_results.values():
                family_results.extend([r for r in results if r.question_family == family])
            
            if family_results:
                metrics = self._compute_metrics(family_results)
                by_family[family] = {
                    'anls': metrics['anls'],
                    'exact_match': metrics['exact_match'],
                    'samples': metrics['total_samples'],
                }
        
        aggregated['by_question_family'] = by_family
        
        return aggregated
    
    def _save_phase_results(self, phase: str, results: List[DUDEQAResult]):
        """Save results for a phase to CSV."""
        phase_dir = self.results_dir / f"phase_{phase}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = phase_dir / "results.csv"
        
        # Convert results to dicts
        rows = [r.to_dict() for r in results]
        
        if rows:
            keys = rows[0].keys()
            with open(results_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"Saved {len(rows)} results to {results_file}")
    
    def _save_results_by_model(self, all_results: Dict[str, List[DUDEQAResult]]):
        """
        Save results grouped by model.
        
        Directory structure:
        results_dir/
          {model_name}/
            results_{timestamp}.csv
        """
        # Collect all results
        all_flat_results = []
        for phase_results in all_results.values():
            all_flat_results.extend(phase_results)
        
        # Group by model (using qa_model for QA3, and qa_model for QA1/QA2)
        results_by_model = {}
        for result in all_flat_results:
            # Determine which model this result used
            if result.phase.startswith("QA1"):
                # OCR + QA model
                if result.parsing_model and result.qa_model:
                    model_key = f"{result.parsing_model}_{result.qa_model}"
                else:
                    model_key = "error"
            elif result.phase.startswith("QA2"):
                # VLM + QA model
                if result.parsing_model and result.qa_model:
                    model_key = f"{result.parsing_model}_{result.qa_model}"
                else:
                    model_key = "error"
            elif result.phase.startswith("QA3"):
                # Direct VLM
                if result.qa_model:
                    model_key = result.qa_model
                else:
                    model_key = "error"
            else:
                model_key = "unknown"
            
            if model_key not in results_by_model:
                results_by_model[model_key] = []
            results_by_model[model_key].append(result)
        
        # Save each model's results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_key, model_results in results_by_model.items():
            # Create model directory
            model_dir = self.results_dir / str(model_key)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results with timestamp
            results_file = model_dir / f"results_{timestamp}.csv"
            
            rows = [r.to_dict() for r in model_results]
            
            if rows:
                keys = rows[0].keys()
                with open(results_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(rows)
                
                logger.info(f"Saved {len(rows)} results to {results_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run DUDE benchmark from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DUDE QA Benchmark')
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
    
    config = DUDEBenchmarkConfig(
        phases=args.phases,
        sample_limit=args.sample_limit,
        ocr_models=args.ocr_models,
        vlm_models=args.vlm_models,
        qa_model=args.qa_model
    )
    
    benchmark = DUDEBenchmark(config)
    summary = benchmark.run()
    
    # Print final summary
    print("\n" + "="*70)
    print("DUDE BENCHMARK SUMMARY")
    print("="*70)
    
    for phase, phase_info in summary.get('phases', {}).items():
        if phase_info.get('status') != 'completed':
            continue
        metrics = phase_info.get('metrics', {})
        print(f"\n{phase}:")
        print(f"  ANLS:        {metrics.get('anls', 0):.4f}")
        print(f"  Exact Match: {metrics.get('exact_match', 0):.4f}")
        print(f"  Samples:     {metrics.get('total_samples', 0)}")
    
    print("\n" + "-"*70)
    print("By Approach:")
    
    by_approach = summary.get('metrics_summary', {}).get('by_approach', {})
    for approach, metrics in by_approach.items():
        print(f"\n{approach}:")
        print(f"  Avg ANLS: {metrics.get('avg_anls', 0):.4f}")
        print(f"  Avg EM:   {metrics.get('avg_em', 0):.4f}")
        print(f"  Samples:  {metrics.get('samples', 0)}")
    
    print("\n" + "-"*70)
    print("By Question Family:")
    
    by_family = summary.get('metrics_summary', {}).get('by_question_family', {})
    for family, metrics in by_family.items():
        print(f"\n{family}:")
        print(f"  ANLS: {metrics.get('anls', 0):.4f}")
        print(f"  EM:   {metrics.get('exact_match', 0):.4f}")


if __name__ == '__main__':
    main()
