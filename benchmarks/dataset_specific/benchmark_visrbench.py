#!/usr/bin/env python3
"""
VisR-Bench Mini Benchmark: Multi-Page Document Question Answering

Evaluates VLM and OCR performance on multi-page documents using evidence page approach.

**QA1 (OCR Pipeline)**: Azure Intelligence OCR (evidence page) → GPT-5-mini QA
  - QA1a: Simple prompt
  - QA1b: Detailed prompt with metadata
  - QA1c: Chain-of-thought prompt with metadata

**QA2 (VLM Pipeline)**: GPT-5-mini extraction (evidence page) → GPT-5-mini QA
  - QA2a: Simple prompt
  - QA2b: Detailed prompt with metadata
  - QA2c: Chain-of-thought prompt with metadata

**QA3 (Direct VQA)**: GPT-5-mini sees evidence page image + question directly
  - QA3a: Simple prompt
  - QA3b: Detailed prompt with metadata

**QA4 (BM25 Sparse Retrieval)**: BM25 retrieval → Ground-truth markdown → GPT-5-mini QA
  - QA4a: Simple prompt
  - QA4b: Detailed prompt with metadata
  - QA4c: Chain-of-thought prompt with metadata
  - QA4d: Structured JSON output (answer + reasoning)

**QA5 (BGE-M3 Dense Retrieval)**: BGE-M3 retrieval → Ground-truth markdown → GPT-5-mini QA
  - QA5a: Simple prompt
  - QA5b: Detailed prompt with metadata
  - QA5c: Chain-of-thought prompt with metadata
  - QA5d: Structured JSON output (answer + reasoning)

**QA6 (Hybrid Retrieval)**: Hybrid (BM25+BGE-M3) retrieval → Ground-truth markdown → GPT-5-mini QA
  - QA6a: Simple prompt
  - QA6b: Detailed prompt with metadata
  - QA6c: Chain-of-thought prompt with metadata
  - QA6d: Structured JSON output (answer + reasoning)

Metadata-Enhanced Prompts:
  - Phases with 'b', 'c', or 'd' suffix include document metadata in prompts:
    - Content Type: figure/table/text/multilingual
    - Detected Language: language code
    - Evidence Page: page index in document
  - Hypothesis: Metadata improves QA accuracy by providing document context

Retrieval Evaluation:
  - QA4/QA5 use BM25 to retrieve relevant pages instead of ground truth
  - Compares retrieval accuracy impact on QA performance
  - Metrics: retrieved_page_index, is_correct_page, retrieval_rank, bm25_score

Cost-Aware Model Selection:
  - OCR: azure_intelligence (recommended), mistral_document_2505 (alternative)
  - VLM: gpt-5-mini (primary), gpt-5-nano (budget), claude_sonnet (baseline)

Sampling Strategy:
  - Samples QAs per document to ensure diversity
  - Default: 5 random QAs per document
  - Can filter by content_type (figure, table, text, multilingual)

Usage:
    # Run all phases on full dataset
    python -m ocr_vs_vlm.benchmarks.benchmark_visrbench

    # Run specific phases with sample limit
    python -m ocr_vs_vlm.benchmarks.benchmark_visrbench --phases QA1a QA3a --sample-limit 50

    # Run BM25 retrieval phases
    python -m ocr_vs_vlm.benchmarks.benchmark_visrbench --phases QA4a QA4b QA4c QA4d --sample-limit 10

    # Development run: 1 sample
    python -m ocr_vs_vlm.benchmarks.benchmark_visrbench --sample-limit 1

    # Compare simple vs metadata-enhanced prompts
    python -m ocr_vs_vlm.benchmarks.benchmark_visrbench --phases QA1a QA1b QA3a QA3b --sample-limit 20

    # Stratified run: 100 samples per content type, 5 per doc
    python -m ocr_vs_vlm.benchmarks.benchmark_visrbench --sample-limit 100 --qa-per-doc 5 --content-type multilingual
"""

import json
import logging
import sys
import time
import csv
import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from datasets.dataset_loaders_qa import VisRBenchMiniDataset, VisRBenchSample
from models import UnifiedModelAPI
from metrics.evaluation_metrics import (
    compute_anls,
    compute_exact_match,
    compute_substring_match,
    compute_prediction_in_ground_truth,
    compute_ground_truth_in_prediction,
    compute_embedding_similarity
)
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, BGEM3Retriever

# Suppress debug logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

EXTRACTION_PROMPTS = {
    "simple": "Extract all text from this page.",
    "detailed": "Extract all text, numbers, labels, tables, and structured information from this page. Preserve formatting and hierarchy where possible.",
}

QA_PROMPTS = {
    "simple": """Based on the following document text, answer the question concisely.

Document Text:
{extracted_text}

Question: {question}

Answer:""",
    "detailed": """You are answering a question based on the following document content.

Document Metadata:
- Content Type: {content_type}
- Detected Language: {detected_language}
- Page Number: {page_index}

Document Text:
{extracted_text}

Please answer accurately and concisely based on the text above.

Question: {question}

Answer:""",
    "cot": """You are answering a question based on the following document content.

Document Metadata:
- Content Type: {content_type}
- Detected Language: {detected_language}
- Page Number: {page_index}

Document Text:
{extracted_text}

Please answer step by step:

Question: {question}

Think through this carefully:
1. Identify relevant information from the text
2. Reason through the answer
3. Provide your final answer

Answer:""",
    "structured": """You are answering a question based on the following document content.

Document Metadata:
- Content Type: {content_type}
- Detected Language: {detected_language}
- Page Number: {page_index}

Document Text:
{extracted_text}

Please provide your answer in JSON format with your reasoning.

Question: {question}

Respond in this format:
{{
  "answer": "your concise answer",
  "reasoning": "brief explanation"
}}

JSON Response:""",
}

DIRECT_VQA_PROMPTS = {
    "simple": "Question: {question}\n\nAnswer:",
    "detailed": """You are analyzing a document page.

Document Metadata:
- Content Type: {content_type} (figure/table/text/multilingual)
- Detected Language: {detected_language}
- Evidence Page: {page_index}

Please answer the following question accurately based on what you see.

Question: {question}

Answer:""",
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VisRBenchConfig:
    """Configuration for VisR-Bench QA benchmark run."""

    # Dataset
    dataset_name: str = "VisR-Bench_mini"
    dataset_root: str = ""  # Set in _load_dataset() if not provided

    # HuggingFace loading
    use_huggingface: bool = False
    hf_dataset_name: str = "kenza-ily/visr-bench-mini"

    # Models for different approaches
    ocr_models: List[str] = field(default_factory=lambda: ["azure_intelligence"])
    vlm_models: List[str] = field(default_factory=lambda: ["gpt-5-mini"])

    # Phases to run
    phases: List[str] = field(default_factory=lambda: [
        "QA1a", "QA1b", "QA1c",  # OCR pipeline (ground truth page)
        "QA2a", "QA2b", "QA2c",  # VLM pipeline (ground truth page)
        "QA3a", "QA3b",          # Direct VQA (ground truth page)
    ])

    # Retrieval configuration
    retrieval_mode: bool = False  # If True, use BM25; if False, use ground truth
    retrieval_top_k: int = 1

    # Content type filter
    content_type: Optional[str] = None

    # Sampling strategy
    sample_limit: Optional[int] = None
    qa_per_doc: int = 5  # Random QAs per document
    batch_size: int = 5  # Save results every N samples

    # Output paths
    results_dir: str = "ocr_vs_vlm/results/1_raw/visrbench_mini"

    # Embedding settings
    compute_embeddings: bool = False  # Set to True to compute embeddings during benchmark


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class VisRBenchResult:
    """Result from a single VisR-Bench QA evaluation."""

    sample_id: str
    doc_id: str
    question: str
    page_index: int  # Ground truth evidence page
    detected_language: str
    ground_truths: List[str]  # List with single answer
    prediction: str

    # Phase info
    phase: str  # e.g., "QA1a", "QA2b"
    parsing_model: Optional[str] = None
    qa_model: Optional[str] = None

    # Intermediate outputs
    extracted_text: Optional[str] = None
    prompt_used: Optional[str] = None

    # QA metrics
    anls_score: float = 0.0
    exact_match: float = 0.0
    substring_match: float = 0.0
    embedding_similarity: float = 0.0

    # Retrieval metrics (for QA4/QA5/QA6)
    retrieved_page_index: Optional[int] = None
    is_correct_page: bool = False
    retrieval_rank: Optional[int] = None
    retrieval_score: Optional[float] = None  # BM25, cosine, or hybrid score
    retrieval_method: Optional[str] = None  # "bm25", "dense", or "hybrid"

    # Metadata
    content_type: str = ""
    total_pages: int = 0
    inference_time_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class VisRBenchBenchmark:
    """
    VisR-Bench Benchmark runner.
    
    Evaluates models on long-document QA with retrieval focus.
    """
    
    def __init__(self, config: VisRBenchConfig):
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.api = UnifiedModelAPI()

        # For text-only QA in retrieval phases
        try:
            from models.settings import get_azure_openai_client
        except ImportError:
            from models.settings import get_azure_openai_client
        self.text_qa_client = get_azure_openai_client()
        
        print(f"\nVisR-Bench Benchmark initialized")
        print(f"  Content type: {config.content_type or 'all'}")
        print(f"  QA per doc: {config.qa_per_doc}")
        print(f"  Phases: {config.phases}")
        print(f"  Sample limit: {config.sample_limit or 'All'}")
        print(f"  Results dir: {self.results_dir}")
    
    def run(self) -> Dict:
        """Execute full benchmark across all phases."""
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
        qa4_phases = [p for p in self.config.phases if p.startswith("QA4")]
        qa5_phases = [p for p in self.config.phases if p.startswith("QA5")]
        qa6_phases = [p for p in self.config.phases if p.startswith("QA6")]

        # QA1: OCR Pipeline (evidence page only)
        if qa1_phases:
            for ocr_model in self.config.ocr_models:
                print(f"\nQA1: OCR Pipeline ({ocr_model} → GPT-5-mini) [evidence page only]")
                results = self._run_qa1(dataset, qa1_phases, ocr_model)
                all_results.extend(results)

        # QA2: VLM Pipeline (evidence page only)
        if qa2_phases:
            for vlm_model in self.config.vlm_models:
                print(f"\nQA2: VLM Pipeline ({vlm_model} → {vlm_model}) [evidence page only]")
                results = self._run_qa2(dataset, qa2_phases, vlm_model)
                all_results.extend(results)

        # QA3: Direct VQA (evidence page only)
        if qa3_phases:
            for vlm_model in self.config.vlm_models:
                print(f"\nQA3: Direct VQA ({vlm_model}) [evidence page only]")
                results = self._run_qa3(dataset, qa3_phases, vlm_model)
                all_results.extend(results)

        # QA4: BM25 Sparse Retrieval
        if qa4_phases:
            print(f"\nQA4: BM25 Sparse Retrieval → Ground-truth markdown → GPT-5-mini")
            results = self._run_qa4(dataset, qa4_phases)
            all_results.extend(results)

        # QA5: Dense Semantic Retrieval
        if qa5_phases:
            print(f"\nQA5: Dense Semantic Retrieval → Ground-truth markdown → GPT-5-mini")
            results = self._run_qa5(dataset, qa5_phases)
            all_results.extend(results)

        # QA6: Hybrid Retrieval
        if qa6_phases:
            print(f"\nQA6: Hybrid Retrieval (BM25+Dense) → Ground-truth markdown → GPT-5-mini")
            results = self._run_qa6(dataset, qa6_phases)
            all_results.extend(results)
        
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print(f"Benchmark completed in {elapsed:.1f}s")
        
        # Save all results
        self._save_results(all_results)
        
        return {'results': all_results}
    
    def _load_dataset(self) -> List[VisRBenchSample]:
        """Load VisR-Bench mini dataset."""
        # Set dataset_root if not provided (only needed for local loading)
        if not self.config.dataset_root:
            self.config.dataset_root = str(Path(__file__).parent.parent / "datasets_subsets")

        loader = VisRBenchMiniDataset(
            dataset_root=self.config.dataset_root,
            content_type=self.config.content_type,
            sample_limit=self.config.sample_limit,
            qa_per_doc=self.config.qa_per_doc,
            use_huggingface=self.config.use_huggingface,
            hf_dataset_name=self.config.hf_dataset_name
        )

        return list(loader)
    
    def _run_qa1(self, dataset: List[VisRBenchSample], phases: List[str], ocr_model: str) -> List[VisRBenchResult]:
        """QA1: OCR Pipeline using evidence page only."""
        results = []
        qa_model = "gpt-5-mini"
        
        for sample in tqdm(dataset, desc=f"QA1 ({ocr_model})"):
            for phase in phases:
                prompt_variant = phase[-1].lower()  # 'a', 'b', 'c'
                
                result = self._process_sample(
                    sample=sample,
                    phase=phase,
                    approach="qa1_ocr",
                    parsing_model=ocr_model,
                    qa_model=qa_model,
                    prompt_variant=prompt_variant,
                    use_evidence_page_only=True
                )
                results.append(result)
            
            if len(results) % self.config.batch_size == 0:
                self._save_batch_results(results[-self.config.batch_size:], ocr_model)
        
        return results
    
    def _run_qa2(self, dataset: List[VisRBenchSample], phases: List[str], vlm_model: str) -> List[VisRBenchResult]:
        """QA2: VLM Parse Pipeline using evidence page only."""
        results = []
        
        for sample in tqdm(dataset, desc=f"QA2 ({vlm_model})"):
            for phase in phases:
                prompt_variant = phase[-1].lower()
                
                result = self._process_sample(
                    sample=sample,
                    phase=phase,
                    approach="qa2_vlm",
                    parsing_model=vlm_model,
                    qa_model=vlm_model,
                    prompt_variant=prompt_variant,
                    use_evidence_page_only=True
                )
                results.append(result)
            
            if len(results) % self.config.batch_size == 0:
                self._save_batch_results(results[-self.config.batch_size:], vlm_model)
        
        return results
    
    def _run_qa3(self, dataset: List[VisRBenchSample], phases: List[str], vlm_model: str) -> List[VisRBenchResult]:
        """QA3: Direct VQA on evidence page."""
        results = []

        for sample in tqdm(dataset, desc=f"QA3 ({vlm_model})"):
            for phase in phases:
                prompt_variant = phase[-1].lower()

                result = self._process_sample(
                    sample=sample,
                    phase=phase,
                    approach="qa3_direct",
                    parsing_model=None,
                    qa_model=vlm_model,
                    prompt_variant=prompt_variant,
                    use_evidence_page_only=True
                )
                results.append(result)

            if len(results) % self.config.batch_size == 0:
                self._save_batch_results(results[-self.config.batch_size:], vlm_model)

        return results

    def _run_qa4(self, dataset: List[VisRBenchSample], phases: List[str]) -> List[VisRBenchResult]:
        """QA4: BM25 Sparse Retrieval → Ground-truth markdown → QA."""
        results = []
        qa_model = "gpt-5-mini"
        retriever = BM25Retriever()

        for sample in tqdm(dataset, desc="QA4 (BM25 Sparse)"):
            # Use BM25 to retrieve the most relevant page
            if sample.all_page_md_str:
                top_results, gt_rank = retriever.retrieve_with_rank(
                    query=sample.question,
                    documents=sample.all_page_md_str,
                    ground_truth_idx=sample.page_index,
                    top_k=self.config.retrieval_top_k
                )
                retrieved_page_idx, retrieval_score = top_results[0] if top_results else (sample.page_index, 0.0)
            else:
                # Fallback to ground truth if no markdown available
                retrieved_page_idx = sample.page_index
                retrieval_score = 0.0
                gt_rank = 1

            for phase in phases:
                prompt_variant = phase[-1].lower()

                # Use pre-extracted markdown (NO API CALL for extraction)
                result = self._process_retrieval_sample(
                    sample=sample,
                    phase=phase,
                    qa_model=qa_model,
                    prompt_variant=prompt_variant,
                    retrieved_page_idx=retrieved_page_idx,
                    retrieval_score=retrieval_score,
                    retrieval_rank=gt_rank,
                    retrieval_method="bm25"
                )
                results.append(result)

            if len(results) % self.config.batch_size == 0:
                self._save_batch_results(results[-self.config.batch_size:], qa_model)

        return results

    def _run_qa5(self, dataset: List[VisRBenchSample], phases: List[str]) -> List[VisRBenchResult]:
        """QA5: BGE-M3 Dense Retrieval → Ground-truth markdown → QA."""
        results = []
        qa_model = "gpt-5-mini"
        retriever = BGEM3Retriever()

        for sample in tqdm(dataset, desc="QA5 (Dense Semantic)"):
            # Use dense retrieval to find the most relevant page
            if sample.all_page_md_str:
                top_results, gt_rank = retriever.retrieve_with_rank(
                    query=sample.question,
                    documents=sample.all_page_md_str,
                    ground_truth_idx=sample.page_index,
                    top_k=self.config.retrieval_top_k
                )
                retrieved_page_idx, retrieval_score = top_results[0] if top_results else (sample.page_index, 0.0)
            else:
                # Fallback to ground truth if no markdown available
                retrieved_page_idx = sample.page_index
                retrieval_score = 0.0
                gt_rank = 1

            for phase in phases:
                prompt_variant = phase[-1].lower()

                # Use pre-extracted markdown (NO API CALL for extraction)
                result = self._process_retrieval_sample(
                    sample=sample,
                    phase=phase,
                    qa_model=qa_model,
                    prompt_variant=prompt_variant,
                    retrieved_page_idx=retrieved_page_idx,
                    retrieval_score=retrieval_score,
                    retrieval_rank=gt_rank,
                    retrieval_method="bge_m3"
                )
                results.append(result)

            if len(results) % self.config.batch_size == 0:
                self._save_batch_results(results[-self.config.batch_size:], qa_model)

        return results

    def _run_qa6(self, dataset: List[VisRBenchSample], phases: List[str]) -> List[VisRBenchResult]:
        """QA6: Hybrid Retrieval (BM25+Dense) → Ground-truth markdown → QA."""
        results = []
        qa_model = "gpt-5-mini"
        retriever = HybridRetriever()

        for sample in tqdm(dataset, desc="QA6 (Hybrid)"):
            # Use hybrid retrieval to find the most relevant page
            if sample.all_page_md_str:
                top_results, gt_rank = retriever.retrieve_with_rank(
                    query=sample.question,
                    documents=sample.all_page_md_str,
                    ground_truth_idx=sample.page_index,
                    top_k=self.config.retrieval_top_k
                )
                retrieved_page_idx, retrieval_score = top_results[0] if top_results else (sample.page_index, 0.0)
            else:
                # Fallback to ground truth if no markdown available
                retrieved_page_idx = sample.page_index
                retrieval_score = 0.0
                gt_rank = 1

            for phase in phases:
                prompt_variant = phase[-1].lower()

                # Use pre-extracted markdown (NO API CALL for extraction)
                result = self._process_retrieval_sample(
                    sample=sample,
                    phase=phase,
                    qa_model=qa_model,
                    prompt_variant=prompt_variant,
                    retrieved_page_idx=retrieved_page_idx,
                    retrieval_score=retrieval_score,
                    retrieval_rank=gt_rank,
                    retrieval_method="hybrid"
                )
                results.append(result)

            if len(results) % self.config.batch_size == 0:
                self._save_batch_results(results[-self.config.batch_size:], qa_model)

        return results
    
    def _process_sample(
        self,
        sample: VisRBenchSample,
        phase: str,
        approach: str,
        parsing_model: Optional[str],
        qa_model: str,
        prompt_variant: str,
        use_evidence_page_only: bool = True,
        retrieved_page_index: Optional[int] = None,
        bm25_score: Optional[float] = None,
        retrieval_rank: Optional[int] = None
    ) -> VisRBenchResult:
        """Process a single sample for QA1-QA5 (with optional retrieval)."""
        start_time = time.time()

        # Determine which page to use
        page_to_use = retrieved_page_index if retrieved_page_index is not None else sample.page_index
        is_correct_page = (retrieved_page_index == sample.page_index) if retrieved_page_index is not None else False

        try:
            # Get page image (evidence page or retrieved page)
            from datasets.dataset_loaders_qa import VisRBenchMiniDataset
            image_path = None
            fallback_markdown = None

            try:
                loader = VisRBenchMiniDataset(
                    dataset_root=str(Path(__file__).parent.parent / "datasets_subsets"),
                    content_type=None
                )

                # Get image for the page to use
                if retrieved_page_index is not None:
                    # For retrieval phases (QA4/QA5), get the retrieved page image
                    all_page_images = loader.get_all_page_images(sample)
                    if all_page_images and page_to_use < len(all_page_images):
                        candidate_path = all_page_images[page_to_use]
                        # Verify image exists before using it
                        if candidate_path and Path(candidate_path).exists():
                            image_path = candidate_path
                        else:
                            logger.warning(f"Image not found: {candidate_path}, will use markdown fallback")
                            image_path = None
                else:
                    # For ground truth phases (QA1/QA2/QA3), get the evidence page image
                    image_path = loader.get_evidence_page_image(sample)
                    # Check if image exists
                    if image_path and not Path(image_path).exists():
                        logger.warning(f"Image not found: {image_path}, using markdown fallback")
                        image_path = None

                # Get markdown for the page as fallback
                if not image_path and sample.all_page_md_str and page_to_use < len(sample.all_page_md_str):
                    fallback_markdown = sample.all_page_md_str[page_to_use]

            except Exception as e:
                logger.warning(f"Could not load page image: {e}")
                # Try to use markdown as fallback
                if sample.all_page_md_str and page_to_use < len(sample.all_page_md_str):
                    fallback_markdown = sample.all_page_md_str[page_to_use]
            
            extracted_text = ""
            prediction = ""
            
            if approach == "qa3_direct":
                # QA3: Pass image + question directly to VLM (no extraction step)
                # Skip if no image available (QA3 requires image)
                if not image_path:
                    logger.warning(f"QA3 skipped for {sample.sample_id}: no image available")
                    elapsed_ms = (time.time() - start_time) * 1000
                    return VisRBenchResult(
                        sample_id=sample.sample_id,
                        doc_id=sample.doc_id,
                        question=sample.question,
                        page_index=sample.page_index,
                        detected_language=sample.detected_language,
                        ground_truths=[sample.answer],
                        prediction="[SKIPPED: image not available]",
                        phase=phase,
                        parsing_model=parsing_model,
                        qa_model=qa_model,
                        anls_score=0.0,
                        exact_match=0.0,
                        substring_match=0.0,
                        embedding_similarity=0.0,
                        retrieved_page_index=retrieved_page_index,
                        is_correct_page=is_correct_page,
                        retrieval_rank=retrieval_rank,
                        bm25_score=bm25_score,
                        content_type=sample.content_type,
                        total_pages=sample.total_pages,
                        inference_time_ms=elapsed_ms,
                        timestamp=datetime.now().isoformat()
                    )
                
                qa_prompt_template = DIRECT_VQA_PROMPTS.get(
                    "detailed" if prompt_variant != "a" else "simple"
                )

                # Format prompt with metadata if template expects it (detailed/cot variants)
                try:
                    qa_prompt = qa_prompt_template.format(
                        question=sample.question,
                        content_type=sample.content_type,
                        detected_language=sample.detected_language,
                        page_index=sample.page_index
                    )
                except KeyError:
                    # Template doesn't expect metadata (simple prompt)
                    qa_prompt = qa_prompt_template.format(question=sample.question)
                
                # Generate answer
                qa_response = self.api.process(
                    image_path,
                    model=qa_model,
                    query=qa_prompt
                )
                prediction = qa_response.content if qa_response else ""
            
            else:
                # QA1 or QA2: Extract text from evidence page, then answer question
                
                # If no image but have markdown, use it as extracted_text (skip extraction step)
                if not image_path and fallback_markdown:
                    extracted_text = fallback_markdown
                    logger.info(f"Using markdown fallback for {sample.sample_id}")
                elif image_path:
                    # Extract text using OCR (QA1) or VLM (QA2)
                    extraction_prompt = EXTRACTION_PROMPTS.get(
                        "detailed" if prompt_variant != "a" else "simple"
                    )
                    
                    response = self.api.process(
                        image_path,
                        model=parsing_model,
                        query=extraction_prompt
                    )
                    extracted_text = response.content if response else ""
                else:
                    # No image and no markdown fallback
                    logger.warning(f"QA1/QA2 skipped for {sample.sample_id}: no image or markdown available")
                    elapsed_ms = (time.time() - start_time) * 1000
                    return VisRBenchResult(
                        sample_id=sample.sample_id,
                        doc_id=sample.doc_id,
                        question=sample.question,
                        page_index=sample.page_index,
                        detected_language=sample.detected_language,
                        ground_truths=[sample.answer],
                        prediction="[SKIPPED: no content available]",
                        phase=phase,
                        parsing_model=parsing_model,
                        qa_model=qa_model,
                        anls_score=0.0,
                        exact_match=0.0,
                        substring_match=0.0,
                        embedding_similarity=0.0,
                        retrieved_page_index=retrieved_page_index,
                        is_correct_page=is_correct_page,
                        retrieval_rank=retrieval_rank,
                        bm25_score=bm25_score,
                        content_type=sample.content_type,
                        total_pages=sample.total_pages,
                        inference_time_ms=elapsed_ms,
                        timestamp=datetime.now().isoformat()
                    )
                
                # Get QA prompt
                qa_prompt_template = QA_PROMPTS.get(prompt_variant, QA_PROMPTS["simple"])

                # Format prompt with metadata if template expects it (detailed/cot variants)
                try:
                    qa_prompt = qa_prompt_template.format(
                        question=sample.question,
                        extracted_text=extracted_text[:2000],  # Limit context
                        content_type=sample.content_type,
                        detected_language=sample.detected_language,
                        page_index=page_to_use
                    )
                except KeyError:
                    # Template doesn't expect metadata (simple prompt)
                    qa_prompt = qa_prompt_template.format(
                        question=sample.question,
                        extracted_text=extracted_text[:2000]
                    )

                # For QA step: try to get an image for the API call
                # If we don't have one, try to get the evidence page image as fallback
                qa_image_path = image_path
                if not qa_image_path:
                    try:
                        # Try to get evidence page image for QA call
                        qa_image_path = loader.get_evidence_page_image(sample)
                        if qa_image_path and not Path(qa_image_path).exists():
                            qa_image_path = None
                    except:
                        qa_image_path = None

                # If still no image, we can't make the QA call (API requires image)
                if not qa_image_path:
                    logger.warning(f"QA1/QA2/QA4/QA5 answering skipped for {sample.sample_id}: no image available for QA call")
                    elapsed_ms = (time.time() - start_time) * 1000
                    return VisRBenchResult(
                        sample_id=sample.sample_id,
                        doc_id=sample.doc_id,
                        question=sample.question,
                        page_index=sample.page_index,
                        detected_language=sample.detected_language,
                        ground_truths=[sample.answer],
                        prediction="[SKIPPED: no image available for QA call]",
                        phase=phase,
                        parsing_model=parsing_model,
                        qa_model=qa_model,
                        extracted_text=extracted_text[:100] if extracted_text else None,
                        anls_score=0.0,
                        exact_match=0.0,
                        substring_match=0.0,
                        embedding_similarity=0.0,
                        retrieved_page_index=retrieved_page_index,
                        is_correct_page=is_correct_page,
                        retrieval_rank=retrieval_rank,
                        bm25_score=bm25_score,
                        content_type=sample.content_type,
                        total_pages=sample.total_pages,
                        inference_time_ms=elapsed_ms,
                        timestamp=datetime.now().isoformat()
                    )

                qa_response = self.api.process(
                    qa_image_path,
                    model=qa_model,
                    query=qa_prompt
                )
                prediction = qa_response.content if qa_response else ""
            
            # Compute metrics
            ground_truths = [sample.answer]
            anls = compute_anls(prediction, ground_truths)
            exact = compute_exact_match(prediction, ground_truths)
            substring = compute_substring_match(prediction, ground_truths)
            embedding_sim = compute_embedding_similarity(prediction, ground_truths[0]) if self.config.compute_embeddings else 0.0

            elapsed_ms = (time.time() - start_time) * 1000

            return VisRBenchResult(
                sample_id=sample.sample_id,
                doc_id=sample.doc_id,
                question=sample.question,
                page_index=sample.page_index,
                detected_language=sample.detected_language,
                ground_truths=ground_truths,
                prediction=prediction,
                phase=phase,
                parsing_model=parsing_model,
                qa_model=qa_model,
                anls_score=anls,
                exact_match=exact,
                substring_match=substring,
                embedding_similarity=embedding_sim,
                retrieved_page_index=retrieved_page_index,
                is_correct_page=is_correct_page,
                retrieval_rank=retrieval_rank,
                bm25_score=bm25_score,
                content_type=sample.content_type,
                total_pages=sample.total_pages,
                inference_time_ms=elapsed_ms,
                timestamp=datetime.now().isoformat()
            )
        
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return VisRBenchResult(
                sample_id=sample.sample_id,
                doc_id=sample.doc_id,
                question=sample.question,
                page_index=sample.page_index,
                detected_language=sample.detected_language,
                ground_truths=[sample.answer],
                prediction="",
                phase=phase,
                parsing_model=parsing_model,
                qa_model=qa_model,
                error=str(e),
                retrieved_page_index=retrieved_page_index,
                is_correct_page=is_correct_page,
                retrieval_rank=retrieval_rank,
                bm25_score=bm25_score,
                content_type=sample.content_type,
                total_pages=sample.total_pages,
                inference_time_ms=elapsed_ms,
                timestamp=datetime.now().isoformat()
            )

    def _process_retrieval_sample(
        self,
        sample: VisRBenchSample,
        phase: str,
        qa_model: str,
        prompt_variant: str,
        retrieved_page_idx: int,
        retrieval_score: float,
        retrieval_rank: int,
        retrieval_method: str
    ) -> VisRBenchResult:
        """
        Process a retrieval-based sample (QA4/QA5/QA6).

        Uses pre-extracted markdown from retrieved page - NO OCR/VLM API calls.
        Only makes QA API call with the extracted text.
        """
        start_time = time.time()

        try:
            # Get pre-extracted markdown for retrieved page (NO API CALL)
            if not sample.all_page_md_str or retrieved_page_idx >= len(sample.all_page_md_str):
                # No markdown available
                elapsed_ms = (time.time() - start_time) * 1000
                return VisRBenchResult(
                    sample_id=sample.sample_id,
                    doc_id=sample.doc_id,
                    question=sample.question,
                    page_index=sample.page_index,
                    detected_language=sample.detected_language,
                    ground_truths=[sample.answer],
                    prediction="[SKIPPED: no markdown available]",
                    phase=phase,
                    parsing_model=None,
                    qa_model=qa_model,
                    anls_score=0.0,
                    exact_match=0.0,
                    substring_match=0.0,
                    embedding_similarity=0.0,
                    retrieved_page_index=retrieved_page_idx,
                    is_correct_page=(retrieved_page_idx == sample.page_index),
                    retrieval_rank=retrieval_rank,
                    retrieval_score=retrieval_score,
                    retrieval_method=retrieval_method,
                    content_type=sample.content_type,
                    total_pages=sample.total_pages,
                    inference_time_ms=elapsed_ms,
                    timestamp=datetime.now().isoformat()
                )

            # Get the pre-extracted markdown text
            extracted_text = sample.all_page_md_str[retrieved_page_idx]

            # Format QA prompt with extracted text (TEXT-ONLY, NO IMAGE)
            qa_prompt_template = QA_PROMPTS.get(prompt_variant, QA_PROMPTS["simple"])

            try:
                qa_prompt = qa_prompt_template.format(
                    question=sample.question,
                    extracted_text=extracted_text[:4000],  # More context for text-only
                    content_type=sample.content_type,
                    detected_language=sample.detected_language,
                    page_index=retrieved_page_idx
                )
            except KeyError:
                # Template doesn't expect metadata (simple prompt)
                qa_prompt = qa_prompt_template.format(
                    question=sample.question,
                    extracted_text=extracted_text[:4000]
                )

            # Make TEXT-ONLY QA API call (no image needed!)
            try:
                # Map model name to deployment (use actual Azure deployment names)
                deployment_map = {
                    "gpt-5-mini": "gpt-5-mini",
                    "gpt-5-nano": "gpt-5-nano",
                    "gpt-4": "gpt-4"
                }
                deployment = deployment_map.get(qa_model, "gpt-5-mini")

                response = self.text_qa_client.chat.completions.create(
                    model=deployment,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document text."},
                        {"role": "user", "content": qa_prompt}
                    ],
                    max_completion_tokens=500,
                    temperature=0.0
                )
                prediction = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Text QA API call failed: {e}")
                prediction = f"[ERROR: {str(e)}]"

            # Compute metrics
            ground_truths = [sample.answer]
            anls = compute_anls(prediction, ground_truths)
            exact = compute_exact_match(prediction, ground_truths)
            substring = compute_substring_match(prediction, ground_truths)
            embedding_sim = compute_embedding_similarity(prediction, ground_truths[0]) if self.config.compute_embeddings else 0.0

            elapsed_ms = (time.time() - start_time) * 1000

            return VisRBenchResult(
                sample_id=sample.sample_id,
                doc_id=sample.doc_id,
                question=sample.question,
                page_index=sample.page_index,
                detected_language=sample.detected_language,
                ground_truths=ground_truths,
                prediction=prediction,
                phase=phase,
                parsing_model=None,  # No parsing model used (pre-extracted markdown)
                qa_model=qa_model,
                extracted_text=extracted_text[:100] if extracted_text else None,
                anls_score=anls,
                exact_match=exact,
                substring_match=substring,
                embedding_similarity=embedding_sim,
                retrieved_page_index=retrieved_page_idx,
                is_correct_page=(retrieved_page_idx == sample.page_index),
                retrieval_rank=retrieval_rank,
                retrieval_score=retrieval_score,
                retrieval_method=retrieval_method,
                content_type=sample.content_type,
                total_pages=sample.total_pages,
                inference_time_ms=elapsed_ms,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return VisRBenchResult(
                sample_id=sample.sample_id,
                doc_id=sample.doc_id,
                question=sample.question,
                page_index=sample.page_index,
                detected_language=sample.detected_language,
                ground_truths=[sample.answer],
                prediction="",
                phase=phase,
                parsing_model=None,
                qa_model=qa_model,
                error=str(e),
                retrieved_page_index=retrieved_page_idx,
                is_correct_page=(retrieved_page_idx == sample.page_index),
                retrieval_rank=retrieval_rank,
                retrieval_score=retrieval_score,
                retrieval_method=retrieval_method,
                content_type=sample.content_type,
                total_pages=sample.total_pages,
                inference_time_ms=elapsed_ms,
                timestamp=datetime.now().isoformat()
            )

    def _save_batch_results(self, results: List[VisRBenchResult], model_name: str):
        """Save batch of results to CSV."""
        if not results:
            return
        
        model_dir = self.results_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = model_dir / f"results_{timestamp}.csv"
        
        fieldnames = [
            'sample_id', 'doc_id', 'question', 'page_index', 'detected_language',
            'prediction', 'ground_truths', 'phase', 'parsing_model', 'qa_model',
            'anls_score', 'exact_match', 'substring_match', 'embedding_similarity',
            'retrieved_page_index', 'is_correct_page', 'retrieval_rank', 'retrieval_score', 'retrieval_method',
            'content_type', 'total_pages', 'inference_time_ms', 'error', 'timestamp'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval='', extrasaction='ignore')
            writer.writeheader()
            
            for result in results:
                row = asdict(result)
                # Convert lists to JSON strings
                if row.get('ground_truths'):
                    row['ground_truths'] = json.dumps(row['ground_truths'])
                # Remove fields not in fieldnames
                row = {k: v for k, v in row.items() if k in fieldnames}
                writer.writerow(row)
    
    def _save_results(self, results: List[VisRBenchResult]):
        """Save all results, grouped by model."""
        by_model = {}
        
        for result in results:
            model = result.parsing_model or result.qa_model or "unknown"
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)
        
        for model, model_results in by_model.items():
            self._save_batch_results(model_results, model)
        
        print(f"\nSaved {len(results)} results to {self.results_dir}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VisR-Bench Benchmark")
    parser.add_argument("--phases", nargs="+", help="Phases to run (e.g., QA1a QA3a QA4a QA5a)")
    parser.add_argument("--sample-limit", type=int, help="Max samples to load")
    parser.add_argument("--qa-per-doc", type=int, default=5, help="QAs to sample per document")
    parser.add_argument("--content-type", choices=["figure", "table", "text", "multilingual"], help="Filter by content type")
    parser.add_argument("--ocr-models", nargs="+", default=["azure_intelligence"], help="OCR models")
    parser.add_argument("--vlm-models", nargs="+", default=["gpt-5-mini"], help="VLM models")
    parser.add_argument("--retrieval-mode", action="store_true", help="Use BM25 retrieval instead of ground truth pages")
    parser.add_argument("--retrieval-top-k", type=int, default=1, help="Number of pages to retrieve (default: 1)")
    parser.add_argument("--use-huggingface", action="store_true", help="Load dataset from HuggingFace instead of local files")
    parser.add_argument("--hf-dataset-name", type=str, default="kenza-ily/visr-bench-mini", help="HuggingFace dataset name")

    args = parser.parse_args()

    config = VisRBenchConfig(
        phases=args.phases or VisRBenchConfig().phases,
        sample_limit=args.sample_limit,
        qa_per_doc=args.qa_per_doc,
        content_type=args.content_type,
        ocr_models=args.ocr_models,
        vlm_models=args.vlm_models,
        retrieval_mode=args.retrieval_mode,
        retrieval_top_k=args.retrieval_top_k,
        use_huggingface=args.use_huggingface,
        hf_dataset_name=args.hf_dataset_name,
    )
    
    benchmark = VisRBenchBenchmark(config)
    benchmark.run()


if __name__ == "__main__":
    main()
