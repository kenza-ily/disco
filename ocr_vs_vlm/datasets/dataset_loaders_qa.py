"""
QA Dataset Loaders for DocVQA_mini, InfographicVQA_mini, and VisR-Bench_mini.

Loads the mini datasets from the datasets_subsets folder for QA benchmarking.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QASample:
    """
    Sample for Question-Answering tasks.
    
    Attributes:
        sample_id: Unique identifier
        image_path: Path to the image file
        question: The question to answer
        answers: List of valid answers (for multi-answer evaluation)
        ground_truth: Primary answer (first in answers list)
        question_type: Category of question
        metadata: Additional info (ocr_text for InfographicVQA, etc.)
    """
    sample_id: str
    image_path: str
    question: str
    answers: List[str]
    ground_truth: str
    question_type: str = ""
    metadata: Dict = field(default_factory=dict)


class DocVQAMiniDataset:
    """
    DocVQA Mini Dataset loader.
    
    Loads 500 QA samples from the DocVQA validation set.
    Each sample has a document image, question, and multiple valid answers.
    """
    
    def __init__(self, dataset_root: str, sample_limit: Optional[int] = None):
        """
        Initialize DocVQA mini loader.
        
        Args:
            dataset_root: Root path to datasets_subsets folder
            sample_limit: Max samples to load (None = all 500)
        """
        self.dataset_root = Path(dataset_root)
        self.sample_limit = sample_limit
        self.samples: List[QASample] = []
        self._load()
        
        if sample_limit and len(self.samples) > sample_limit:
            self.samples = self.samples[:sample_limit]
        
        logger.info(f"Loaded {len(self.samples)} samples from DocVQA_mini")
    
    def _load(self):
        """Load samples from the index JSON file."""
        index_file = self.dataset_root / "docvqa_mini" / "docvqa_mini_index.json"
        
        if not index_file.exists():
            raise FileNotFoundError(f"DocVQA mini index not found: {index_file}")
        
        with open(index_file, 'r') as f:
            data = json.load(f)
        
        images_dir = self.dataset_root / "docvqa_mini"
        
        for sample_data in data.get('samples', []):
            # Build full image path
            image_path = images_dir / sample_data['image_path']
            
            # Extract answers (ensure it's a list)
            answers = sample_data.get('answers', [])
            if isinstance(answers, str):
                answers = [answers]
            
            sample = QASample(
                sample_id=sample_data['sample_id'],
                image_path=str(image_path),
                question=sample_data['question'],
                answers=answers,
                ground_truth=sample_data.get('ground_truth', answers[0] if answers else ''),
                question_type=sample_data.get('question_type', ''),
                metadata=sample_data.get('metadata', {})
            )
            self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[QASample]:
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> QASample:
        return self.samples[idx]
    
    def get_stats(self) -> Dict:
        """Return dataset statistics."""
        question_types = {}
        for s in self.samples:
            qt = s.question_type or 'unknown'
            question_types[qt] = question_types.get(qt, 0) + 1
        
        return {
            'total_samples': len(self.samples),
            'question_types': question_types,
            'avg_question_length': sum(len(s.question) for s in self.samples) / len(self.samples) if self.samples else 0,
            'avg_answer_count': sum(len(s.answers) for s in self.samples) / len(self.samples) if self.samples else 0,
        }


class InfographicVQAMiniDataset:
    """
    InfographicVQA Mini Dataset loader.
    
    Loads 500 QA samples from the InfographicVQA validation set.
    Key difference from DocVQA: includes pre-extracted OCR text in metadata.
    """
    
    def __init__(self, dataset_root: str, sample_limit: Optional[int] = None):
        """
        Initialize InfographicVQA mini loader.
        
        Args:
            dataset_root: Root path to datasets_subsets folder
            sample_limit: Max samples to load (None = all 500)
        """
        self.dataset_root = Path(dataset_root)
        self.sample_limit = sample_limit
        self.samples: List[QASample] = []
        self._load()
        
        if sample_limit and len(self.samples) > sample_limit:
            self.samples = self.samples[:sample_limit]
        
        logger.info(f"Loaded {len(self.samples)} samples from InfographicVQA_mini")
    
    def _load(self):
        """Load samples from the index JSON file."""
        index_file = self.dataset_root / "infographicvqa_mini" / "infographicvqa_mini_index.json"
        
        if not index_file.exists():
            raise FileNotFoundError(f"InfographicVQA mini index not found: {index_file}")
        
        with open(index_file, 'r') as f:
            data = json.load(f)
        
        images_dir = self.dataset_root / "infographicvqa_mini"
        
        for sample_data in data.get('samples', []):
            # Build full image path
            image_path = images_dir / sample_data['image_path']
            
            # Extract answers (ensure it's a list)
            answers = sample_data.get('answers', [])
            if isinstance(answers, str):
                answers = [answers]
            
            # InfographicVQA has OCR text in metadata
            metadata = sample_data.get('metadata', {})
            
            sample = QASample(
                sample_id=sample_data['sample_id'],
                image_path=str(image_path),
                question=sample_data['question'],
                answers=answers,
                ground_truth=sample_data.get('ground_truth', answers[0] if answers else ''),
                question_type=sample_data.get('question_type', ''),
                metadata=metadata
            )
            self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[QASample]:
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> QASample:
        return self.samples[idx]
    
    def get_stats(self) -> Dict:
        """Return dataset statistics."""
        question_types = {}
        has_ocr_count = 0
        
        for s in self.samples:
            qt = s.question_type or 'unknown'
            question_types[qt] = question_types.get(qt, 0) + 1
            if s.metadata.get('ocr_text'):
                has_ocr_count += 1
        
        return {
            'total_samples': len(self.samples),
            'question_types': question_types,
            'samples_with_ocr': has_ocr_count,
            'avg_question_length': sum(len(s.question) for s in self.samples) / len(self.samples) if self.samples else 0,
            'avg_answer_count': sum(len(s.answers) for s in self.samples) / len(self.samples) if self.samples else 0,
        }


def validate_qa_dataset(dataset_name: str, dataset_root: str) -> Dict:
    """
    Validate QA dataset integrity.
    
    Args:
        dataset_name: 'DocVQA_mini' or 'InfographicVQA_mini'
        dataset_root: Root path to datasets_subsets folder
    
    Returns:
        Dict with validation results
    """
    root_path = Path(dataset_root)
    
    if dataset_name == 'DocVQA_mini':
        dataset_dir = root_path / "docvqa_mini"
        index_file = dataset_dir / "docvqa_mini_index.json"
    elif dataset_name == 'InfographicVQA_mini':
        dataset_dir = root_path / "infographicvqa_mini"
        index_file = dataset_dir / "infographicvqa_mini_index.json"
    else:
        return {'valid': False, 'error': f"Unknown dataset: {dataset_name}"}
    
    checks = {}
    
    # Check directory exists
    checks['directory_exists'] = dataset_dir.exists()
    if not checks['directory_exists']:
        return {'valid': False, 'error': f"Dataset directory not found: {dataset_dir}", 'checks': checks}
    
    # Check index file exists
    checks['index_exists'] = index_file.exists()
    if not checks['index_exists']:
        return {'valid': False, 'error': f"Index file not found: {index_file}", 'checks': checks}
    
    # Check images directory exists
    images_dir = dataset_dir / "images"
    checks['images_dir_exists'] = images_dir.exists()
    if not checks['images_dir_exists']:
        return {'valid': False, 'error': f"Images directory not found: {images_dir}", 'checks': checks}
    
    # Load and validate index
    try:
        with open(index_file, 'r') as f:
            data = json.load(f)
        checks['index_valid'] = True
        checks['total_samples'] = len(data.get('samples', []))
    except Exception as e:
        checks['index_valid'] = False
        return {'valid': False, 'error': f"Failed to parse index: {e}", 'checks': checks}
    
    # Check a few sample images exist
    samples = data.get('samples', [])[:5]
    images_found = 0
    for sample in samples:
        img_path = dataset_dir / sample['image_path']
        if img_path.exists():
            images_found += 1
    checks['sample_images_found'] = f"{images_found}/{len(samples)}"
    
    return {
        'valid': True,
        'checks': checks
    }


@dataclass
class VisRBenchSample:
    """Sample for VisR-Bench retrieval + QA task."""
    
    sample_id: str
    doc_id: str
    question: str
    answer: str  # Ground truth answer
    page_index: int  # Index of the evidence page
    content_type: str  # "figure", "table", "text", or "multilingual"
    detected_language: str
    
    # Multi-page document info
    all_page_images: List[str]  # List of image filenames for all pages
    all_page_md_str: List[str]  # Pre-extracted markdown for each page
    total_pages: int
    
    # Paths resolved at load time
    images_dir: Optional[str] = None


class VisRBenchMiniDataset:
    """
    VisR-Bench Mini Dataset loader for retrieval + QA evaluation.
    
    Multi-page document QA with evidence page grounding.
    - 498 documents across 4 content types
    - 17,045 total QA pairs
    - Each QA includes page_index for retrieval evaluation
    - Pre-extracted markdown available (all_page_md_str)
    
    Sampling Strategy:
    - qa_per_doc: Cap random QAs per document to ensure diversity
    - Default: 5 QAs per doc to avoid over-weighting documents with many questions
    """
    
    def __init__(
        self,
        dataset_root: str,
        content_type: Optional[str] = None,
        sample_limit: Optional[int] = None,
        qa_per_doc: int = 5,
        seed: int = 42,
        use_huggingface: bool = False,
        hf_dataset_name: str = "kenza-ily/visr-bench-mini"
    ):
        """
        Initialize VisR-Bench mini loader.

        Args:
            dataset_root: Root path to datasets_subsets folder (used when use_huggingface=False)
            content_type: Filter by type: "figure", "table", "text", "multilingual", or None for all
            sample_limit: Max total samples to load (None = all)
            qa_per_doc: Max QAs to sample randomly per document (default 5 for development)
            seed: Random seed for reproducible sampling
            use_huggingface: Load from HuggingFace Hub instead of local files
            hf_dataset_name: HuggingFace dataset name (default: kenza-ily/visr-bench-mini)
        """
        self.dataset_root = Path(dataset_root)
        self.content_type = content_type
        self.sample_limit = sample_limit
        self.qa_per_doc = qa_per_doc
        self.seed = seed
        self.use_huggingface = use_huggingface
        self.hf_dataset_name = hf_dataset_name
        self.samples: List[VisRBenchSample] = []

        random.seed(seed)
        self._load()

        source = "HuggingFace" if use_huggingface else "local"
        logger.info(f"Loaded {len(self.samples)} samples from VisR-Bench_mini ({source}) "
                   f"(content_type={content_type}, qa_per_doc={qa_per_doc})")
    
    def _load(self):
        """Load samples from either local files or HuggingFace."""
        if self.use_huggingface:
            self._load_from_huggingface()
        else:
            self._load_from_local()

    def _load_from_huggingface(self):
        """Load dataset from HuggingFace Hub (or local Parquet if available)."""
        from huggingface_hub import hf_hub_download
        import pandas as pd

        # Check for local Parquet first (faster and more reliable)
        local_parquet = self.dataset_root / "visr_bench_mini" / "visr_bench_mini.parquet"

        if local_parquet.exists():
            logger.info(f"Loading from local Parquet: {local_parquet}")
            df = pd.read_parquet(local_parquet)
        else:
            # Fall back to HuggingFace download
            logger.info(f"Loading dataset from HuggingFace: {self.hf_dataset_name}")
            try:
                parquet_path = hf_hub_download(
                    repo_id=self.hf_dataset_name,
                    filename="visr_bench_mini.parquet",
                    repo_type="dataset"
                )
                df = pd.read_parquet(parquet_path)
            except Exception as e:
                logger.error(f"Failed to download from HuggingFace: {e}")
                logger.info("Falling back to local JSON files...")
                self._load_from_local()
                return

        # Filter by content type if specified
        if self.content_type:
            df = df[df['content_type'] == self.content_type]

        # Validate answer coverage
        has_answer = df['answer'].notna() & (df['answer'] != '')
        if has_answer.sum() < len(df):
            missing = len(df) - has_answer.sum()
            logger.warning(f"Dataset has {missing}/{len(df)} QA pairs with empty answers. "
                         f"These will be skipped. Consider using local JSON files instead.")
            # Filter out empty answers
            df = df[has_answer]

        # Group by doc_id and sample QAs per document
        total_loaded = 0

        for doc_id, group in df.groupby('doc_id'):
            qas = group.to_dict('records')

            # Sample up to qa_per_doc questions per document
            if len(qas) > self.qa_per_doc:
                qas = random.sample(qas, self.qa_per_doc)

            for qa_idx, qa in enumerate(qas):
                sample_id = f"{doc_id}_{qa_idx}"

                # Download images on-demand (cached by huggingface_hub)
                image_files = qa['image_files'].split(',')
                image_paths = []

                for img_file in image_files:
                    img_path = hf_hub_download(
                        repo_id=self.hf_dataset_name,
                        filename=f"{qa['image_dir']}/{img_file}",
                        repo_type="dataset"
                    )
                    image_paths.append(img_file)  # Store filename for compatibility

                # Get all page markdown if available
                all_page_md_str = []
                if qa.get('all_page_images'):
                    all_page_images = qa['all_page_images'].split(',')
                else:
                    all_page_images = image_files

                sample = VisRBenchSample(
                    sample_id=sample_id,
                    doc_id=doc_id,
                    question=qa['question'],
                    answer=qa['answer'],
                    page_index=qa['page_index'],
                    detected_language=qa['detected_language'],
                    content_type=qa['content_type'],
                    all_page_images=all_page_images,
                    all_page_md_str=all_page_md_str,  # Not available in HF dataset
                    total_pages=len(all_page_images),
                    images_dir=f"hf://{self.hf_dataset_name}/{qa['image_dir']}"  # Virtual path for HF
                )

                self.samples.append(sample)
                total_loaded += 1

                # Apply sample limit
                if self.sample_limit and total_loaded >= self.sample_limit:
                    return

    def _load_from_local(self):
        """Load samples from local JSON files with per-document QA sampling."""
        visr_dir = self.dataset_root / "visr_bench_mini"
        content_types_to_load = [self.content_type] if self.content_type else ["figure", "table", "text", "multilingual"]
        
        # Content type to HF folder mapping
        content_to_hf_folder = {
            "figure": "Multimodal",
            "table": "Multimodal",
            "text": "Multimodal",
            "multilingual": "Multilingual"
        }
        
        # Local images directory
        local_images_dir = visr_dir / "images"
        
        total_loaded = 0
        
        for ctype in content_types_to_load:
            qa_file = visr_dir / f"{ctype}_QA_mini.json"
            
            if not qa_file.exists():
                logger.warning(f"VisR-Bench {ctype} JSON not found: {qa_file}")
                continue
            
            with open(qa_file) as f:
                docs = json.load(f)
            
            hf_folder = content_to_hf_folder[ctype]
            
            for doc in docs:
                doc_id = doc['file_name']
                all_page_images = doc.get('all_page_images', [])
                all_page_md_str = doc.get('all_page_md_str', [])
                qa_list = doc.get('qa_list', [])
                
                # Filter QAs to only include those with non-empty ground truth answers
                valid_qas = [qa for qa in qa_list if qa.get('answer', '').strip()]
                
                if not valid_qas:
                    # Skip documents with no valid QAs
                    continue
                
                # Sample random QAs per document (capped at qa_per_doc)
                if len(valid_qas) > self.qa_per_doc:
                    sampled_qas = random.sample(valid_qas, self.qa_per_doc)
                else:
                    sampled_qas = valid_qas
                
                # Create one sample per QA
                for qa_idx, qa in enumerate(sampled_qas):
                    sample_id = f"{doc_id}_{qa_idx}"
                    
                    # Use the new images directory structure
                    images_dir = local_images_dir / hf_folder / doc_id
                    
                    sample = VisRBenchSample(
                        sample_id=sample_id,
                        doc_id=doc_id,
                        question=qa.get('question', ''),
                        answer=qa.get('answer', ''),
                        page_index=qa.get('page_index', 0),
                        detected_language=qa.get('detected_language', 'unknown'),
                        content_type=ctype,
                        all_page_images=all_page_images,
                        all_page_md_str=all_page_md_str,
                        total_pages=len(all_page_images),
                        images_dir=str(images_dir)
                    )
                    
                    self.samples.append(sample)
                    total_loaded += 1
                    
                    if self.sample_limit and total_loaded >= self.sample_limit:
                        return
    
    def get_evidence_page_image(self, sample: VisRBenchSample) -> Optional[str]:
        """
        Get the file path to the evidence page image for a sample.

        Returns:
            Full path to the evidence page image, or None if not found
        """
        page_idx = sample.page_index
        image_filename = sample.all_page_images[page_idx]

        # Handle HuggingFace virtual path
        if sample.images_dir.startswith("hf://"):
            from huggingface_hub import hf_hub_download
            # Parse: "hf://kenza-ily/visr-bench-mini/images/multilingual/0001"
            path_without_prefix = sample.images_dir.replace("hf://", "")
            parts = path_without_prefix.split("/")

            if len(parts) < 2:
                logger.error(f"Invalid HF path: {sample.images_dir}")
                return None

            repo_id = f"{parts[0]}/{parts[1]}"  # "kenza-ily/visr-bench-mini"
            image_dir = "/".join(parts[2:])  # "images/multilingual/doc_id"

            try:
                image_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{image_dir}/{image_filename}",
                    repo_type="dataset"
                )
                return image_path
            except Exception as e:
                logger.warning(f"Failed to download image from HuggingFace: {e}")
                return None

        # Handle local path
        images_dir = Path(sample.images_dir)
        image_path = images_dir / image_filename
        if image_path.exists():
            return str(image_path)

        # Image not found - return None to signal fallback to markdown
        return None
    
    def get_all_page_images(self, sample: VisRBenchSample) -> List[str]:
        """
        Get file paths to all page images for a sample.

        Args:
            sample: VisRBenchSample

        Returns:
            List of paths to all page images in order
        """
        # Handle HuggingFace virtual path
        if sample.images_dir.startswith("hf://"):
            from huggingface_hub import hf_hub_download
            # Parse: "hf://kenza-ily/visr-bench-mini/images/multilingual/0001"
            path_without_prefix = sample.images_dir.replace("hf://", "")
            parts = path_without_prefix.split("/")

            if len(parts) < 2:
                logger.error(f"Invalid HF path: {sample.images_dir}")
                return []

            repo_id = f"{parts[0]}/{parts[1]}"  # "kenza-ily/visr-bench-mini"
            image_dir = "/".join(parts[2:])  # "images/multilingual/doc_id"

            image_paths = []
            for img_filename in sample.all_page_images:
                try:
                    img_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=f"{image_dir}/{img_filename}",
                        repo_type="dataset"
                    )
                    image_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"Failed to download image {img_filename} from HuggingFace: {e}")
            return image_paths

        # Handle local path
        images_dir = Path(sample.images_dir)
        return [str(images_dir / img) for img in sample.all_page_images]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> VisRBenchSample:
        return self.samples[idx]
    
    def get_stats(self) -> Dict:
        """Return dataset statistics."""
        stats = {
            'total_samples': len(self.samples),
            'unique_documents': len(set(s.doc_id for s in self.samples)),
            'content_type_distribution': {},
            'language_distribution': {},
            'avg_pages_per_doc': 0,
            'avg_qa_per_doc': 0,
        }
        
        # Content type distribution
        for s in self.samples:
            ct = s.content_type
            stats['content_type_distribution'][ct] = stats['content_type_distribution'].get(ct, 0) + 1
            
            lang = s.detected_language
            stats['language_distribution'][lang] = stats['language_distribution'].get(lang, 0) + 1
        
        # Average pages
        if self.samples:
            total_pages = sum(s.total_pages for s in self.samples)
            unique_docs = len(set(s.doc_id for s in self.samples))
            stats['avg_pages_per_doc'] = total_pages / unique_docs if unique_docs > 0 else 0
            stats['avg_qa_per_doc'] = len(self.samples) / unique_docs if unique_docs > 0 else 0
        
        return stats
