"""
QA Dataset Loaders for DocVQA_mini and InfographicVQA_mini.

Loads the mini datasets from the datasets_subsets folder for QA benchmarking.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Iterator

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
