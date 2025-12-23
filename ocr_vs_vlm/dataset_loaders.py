"""
Unified dataset loaders for OCR/VLM benchmarking.

All loaders return a consistent interface: Sample(image_path, ground_truth, metadata)
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """
    Unified representation of a dataset sample.
    
    Attributes:
        sample_id: Unique identifier (e.g., "iam_000_a01_000_00")
        image_path: Full path to image file
        ground_truth: Expected text output
        metadata: Additional info (language, difficulty, bbox, etc.)
    """
    sample_id: str
    image_path: str
    ground_truth: str
    metadata: Dict


class Dataset(ABC):
    """Abstract base class for all dataset loaders."""
    
    def __init__(self, dataset_root: str, sample_limit: Optional[int] = None):
        """
        Initialize dataset loader.
        
        Args:
            dataset_root: Root path to dataset
            sample_limit: Max samples to load (None = all)
        """
        self.dataset_root = Path(dataset_root)
        self.sample_limit = sample_limit
        self.samples: List[Sample] = []
        self._load()
        
        if sample_limit and len(self.samples) > sample_limit:
            self.samples = self.samples[:sample_limit]
        
        logger.info(f"Loaded {len(self.samples)} samples from {self.__class__.__name__}")
    
    @abstractmethod
    def _load(self):
        """Load all samples from dataset. Must set self.samples."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)
    
    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]
    
    def get_stats(self) -> Dict:
        """Return dataset statistics."""
        return {
            'total_samples': len(self.samples),
            'avg_text_length': np.mean([len(s.ground_truth) for s in self.samples]),
            'min_text_length': min([len(s.ground_truth) for s in self.samples]) if self.samples else 0,
            'max_text_length': max([len(s.ground_truth) for s in self.samples]) if self.samples else 0,
        }


class IAMDataset(Dataset):
    """
    IAM Handwriting Database loader.
    
    Structure:
    - data/{000-100}/a01_000_00.png  (image)
    - We assume corresponding ground truth files exist or are provided
    
    Each sample = handwritten text line
    """
    
    def _load(self):
        """Walk through IAM data directories and load samples."""
        data_dir = self.dataset_root / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"IAM data directory not found: {data_dir}")
        
        # Walk through numbered directories (000-100+)
        for dir_path in sorted(data_dir.glob("*")):
            if not dir_path.is_dir():
                continue
            
            # Load all PNG images in this directory
            for image_path in sorted(dir_path.glob("*.png")):
                sample_id = f"iam_{dir_path.name}_{image_path.stem}"
                
                # For now, ground truth is either from .txt file with same name,
                # or we extract from database metadata if available
                # Since IAM typically requires registration, we'll make this optional
                
                ground_truth = self._get_iam_ground_truth(image_path)
                if ground_truth is None:
                    continue  # Skip if no ground truth available
                
                metadata = {
                    'dataset': 'IAM',
                    'writer_id': image_path.stem.split('_')[0],  # a01 from a01_000_00
                    'image_size': self._get_image_size(image_path),
                }
                
                self.samples.append(Sample(
                    sample_id=sample_id,
                    image_path=str(image_path),
                    ground_truth=ground_truth,
                    metadata=metadata
                ))
    
    def _get_iam_ground_truth(self, image_path: Path) -> Optional[str]:
        """
        Attempt to retrieve ground truth for IAM image.
        
        Tries multiple strategies:
        1. Look for .txt file with same name
        2. Check metadata files in parent directory
        3. Return None if not found
        """
        # Strategy 1: Look for .txt file
        txt_path = image_path.with_suffix('.txt')
        if txt_path.exists():
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to read {txt_path}: {e}")
                return None
        
        # If no .txt file, we may need to use IAM's official metadata
        # For now, return None (user should provide ground truth files)
        return None
    
    @staticmethod
    def _get_image_size(image_path: Path) -> Tuple[int, int]:
        """Get image dimensions without loading full image."""
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return (0, 0)


class ICDARDataset(Dataset):
    """
    ICDAR 2019 MLT (Multi-Lingual Text) Dataset loader.
    
    Structure:
    - ImagesPart{1,2}/*.jpg  (training images)
    - train_gt_t13/*.txt     (annotations)
    
    Annotation format per line:
    x1,y1,x2,y2,x3,y3,x4,y4,language,text
    """
    
    def _load(self):
        """Load ICDAR images and annotations."""
        # Collect all image paths
        images_dir1 = self.dataset_root / "ImagesPart1"
        images_dir2 = self.dataset_root / "ImagesPart2"
        
        if not images_dir1.exists() and not images_dir2.exists():
            raise FileNotFoundError(f"ICDAR images directories not found")
        
        gt_dir = self.dataset_root / "train_gt_t13"
        if not gt_dir.exists():
            logger.warning(f"ICDAR ground truth directory not found: {gt_dir}")
            return
        
        # Load annotations
        annotations = self._load_annotations(gt_dir)
        
        # Process images
        image_dirs = [d for d in [images_dir1, images_dir2] if d.exists()]
        
        for images_dir in image_dirs:
            for image_path in sorted(images_dir.glob("*.*")):
                if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                # Get corresponding annotation
                image_id = image_path.stem
                if image_id not in annotations:
                    logger.debug(f"No annotation for {image_id}")
                    continue
                
                anno = annotations[image_id]
                sample_id = f"icdar_{image_id}"
                
                # Concatenate all text entries, preserving order (top-to-bottom)
                texts = [entry['text'] for entry in sorted(anno, key=lambda x: x['y_min'])]
                ground_truth = "\n".join(texts)  # Preserve line breaks
                
                # Extract metadata
                languages = [entry['language'] for entry in anno]
                metadata = {
                    'dataset': 'ICDAR',
                    'languages': list(set(languages)),
                    'num_text_lines': len(anno),
                    'image_size': self._get_image_size(image_path),
                    'bounding_boxes': [entry['bbox'] for entry in anno],
                }
                
                self.samples.append(Sample(
                    sample_id=sample_id,
                    image_path=str(image_path),
                    ground_truth=ground_truth,
                    metadata=metadata
                ))
    
    def _load_annotations(self, gt_dir: Path) -> Dict:
        """
        Load all annotation files.
        
        Returns:
            Dict mapping image_id -> List of text entries with bbox and language
        """
        annotations = {}
        
        for gt_file in sorted(gt_dir.glob("*.txt")):
            image_id = gt_file.stem
            entries = []
            
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split(',')
                        if len(parts) < 9:
                            logger.debug(f"Malformed annotation line in {gt_file}")
                            continue
                        
                        # Parse: x1,y1,x2,y2,x3,y3,x4,y4,language,text
                        coords = [int(p) for p in parts[:8]]
                        language = parts[8]
                        text = ','.join(parts[9:]) if len(parts) > 9 else ""
                        
                        if not text:
                            continue
                        
                        # Store bbox as quadrilateral points
                        bbox = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
                        y_coords = [coords[i+1] for i in range(0, 8, 2)]
                        
                        entries.append({
                            'text': text,
                            'language': language,
                            'bbox': bbox,
                            'y_min': min(y_coords),  # For sorting by top-to-bottom
                        })
                
                if entries:
                    annotations[image_id] = entries
            
            except Exception as e:
                logger.warning(f"Failed to load annotation {gt_file}: {e}")
                continue
        
        return annotations
    
    @staticmethod
    def _get_image_size(image_path: Path) -> Tuple[int, int]:
        """Get image dimensions."""
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return (0, 0)


class PubLayNetDataset(Dataset):
    """
    PubLayNet Document Layout Dataset loader.
    
    Structure:
    - data/*.parquet  (HuggingFace parquet format)
    
    Each sample contains:
    - Image: document page image
    - Annotations: layout elements (text blocks, tables, figures)
    """
    
    def _load(self):
        """Load PubLayNet from parquet files."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("pyarrow not installed. Cannot load PubLayNet. Install with: pip install pyarrow")
            return
        
        data_dir = self.dataset_root / "data"
        if not data_dir.exists():
            logger.warning(f"PubLayNet data directory not found: {data_dir}")
            return
        
        parquet_files = sorted(data_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"No parquet files found in {data_dir}")
            return
        
        # Load from parquet files
        for parquet_file in parquet_files:
            try:
                table = pq.read_table(parquet_file)
                df = table.to_pandas()
                
                for idx, row in df.iterrows():
                    # Extract ground truth and metadata from row
                    sample_id = f"publaynet_{parquet_file.stem}_{idx}"
                    
                    # PubLayNet structure: image + layout annotations
                    # Try to extract text from annotations or use image filename
                    ground_truth = self._extract_ground_truth(row)
                    
                    # Try to load image
                    try:
                        # PubLayNet typically stores images as bytes
                        if 'image' in row:
                            # Image data available
                            image_data = row['image']
                            # Store as temporary path or reference
                            image_path = f"publaynet_sample_{idx}"
                        else:
                            continue
                    except Exception as e:
                        logger.debug(f"Cannot load image for sample {sample_id}: {e}")
                        continue
                    
                    metadata = {
                        'dataset': 'PubLayNet',
                        'split': row.get('split', 'train'),
                        'document_type': row.get('document_type', 'unknown'),
                    }
                    
                    self.samples.append(Sample(
                        sample_id=sample_id,
                        image_path=image_path,
                        ground_truth=ground_truth,
                        metadata=metadata
                    ))
            
            except Exception as e:
                logger.warning(f"Failed to load parquet file {parquet_file}: {e}")
                continue
    
    @staticmethod
    def _extract_ground_truth(row) -> str:
        """Extract text from PubLayNet annotation structure."""
        # PubLayNet stores layout annotations with text content
        # Strategy: Concatenate all text from layout elements
        
        texts = []
        
        if 'texts' in row and row['texts']:
            texts.extend(row['texts'])
        
        if 'layout_text' in row and row['layout_text']:
            texts.extend(row['layout_text'])
        
        # Join with line breaks
        return "\n".join(texts) if texts else ""


class DatasetRegistry:
    """
    Central registry for dataset instantiation.
    
    Provides unified interface: get_dataset("IAM", path, limit)
    """
    
    _DATASETS = {
        'IAM': IAMDataset,
        'ICDAR': ICDARDataset,
        'PubLayNet': PubLayNetDataset,
    }
    
    @classmethod
    def register(cls, name: str, dataset_class):
        """Register a new dataset loader."""
        cls._DATASETS[name] = dataset_class
    
    @classmethod
    def get_dataset(cls, name: str, root: str, sample_limit: Optional[int] = None) -> Dataset:
        """
        Get dataset instance by name.
        
        Args:
            name: Dataset name (IAM, ICDAR, PubLayNet)
            root: Root directory path
            sample_limit: Max samples to load
        
        Returns:
            Dataset instance
        
        Raises:
            ValueError: If dataset name not recognized
        """
        if name not in cls._DATASETS:
            available = ", ".join(cls._DATASETS.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        
        dataset_class = cls._DATASETS[name]
        return dataset_class(root, sample_limit=sample_limit)
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered datasets."""
        return list(cls._DATASETS.keys())


def validate_dataset(dataset_name: str, root: str) -> Dict:
    """
    Validate dataset integrity.
    
    Checks:
    - Root directory exists
    - Required subdirectories exist
    - Sample files are readable
    
    Returns:
        Dict with validation results
    """
    root_path = Path(root)
    
    if not root_path.exists():
        return {'valid': False, 'error': f"Root path does not exist: {root}"}
    
    results = {
        'valid': True,
        'dataset': dataset_name,
        'root': str(root_path),
        'checks': {}
    }
    
    try:
        if dataset_name == 'IAM':
            data_dir = root_path / "data"
            results['checks']['data_dir_exists'] = data_dir.exists()
            if data_dir.exists():
                subdirs = list(data_dir.glob("*"))
                results['checks']['subdirs_count'] = len(subdirs)
                results['checks']['sample_images'] = len(list(data_dir.glob("*/*.png")))
        
        elif dataset_name == 'ICDAR':
            images1 = root_path / "ImagesPart1"
            images2 = root_path / "ImagesPart2"
            gt = root_path / "train_gt_t13"
            
            results['checks']['ImagesPart1_exists'] = images1.exists()
            results['checks']['ImagesPart2_exists'] = images2.exists()
            results['checks']['train_gt_t13_exists'] = gt.exists()
            
            if images1.exists():
                results['checks']['ImagesPart1_images'] = len(list(images1.glob("*.*")))
            if images2.exists():
                results['checks']['ImagesPart2_images'] = len(list(images2.glob("*.*")))
            if gt.exists():
                results['checks']['train_gt_t13_files'] = len(list(gt.glob("*.txt")))
        
        elif dataset_name == 'PubLayNet':
            data_dir = root_path / "data"
            results['checks']['data_dir_exists'] = data_dir.exists()
            if data_dir.exists():
                results['checks']['parquet_files'] = len(list(data_dir.glob("*.parquet")))
        
        # Try to instantiate
        try:
            dataset = DatasetRegistry.get_dataset(dataset_name, str(root_path), sample_limit=1)
            results['checks']['instantiation'] = True
            results['checks']['sample_loadable'] = len(dataset) > 0
        except Exception as e:
            results['checks']['instantiation'] = False
            results['error'] = str(e)
            results['valid'] = False
    
    except Exception as e:
        results['valid'] = False
        results['error'] = str(e)
    
    return results


# Utility functions for common operations

def load_image(image_path: str) -> Image.Image:
    """Load image and ensure RGB format."""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def get_dataset_stats(dataset: Dataset) -> Dict:
    """Get comprehensive statistics about a dataset."""
    stats = dataset.get_stats()
    
    # Add language distribution for ICDAR
    if dataset.samples and 'languages' in dataset.samples[0].metadata:
        all_langs = []
        for sample in dataset.samples:
            all_langs.extend(sample.metadata.get('languages', []))
        stats['languages'] = {}
        for lang in set(all_langs):
            stats['languages'][lang] = all_langs.count(lang)
    
    return stats


if __name__ == '__main__':
    """Test loaders with local datasets."""
    logging.basicConfig(level=logging.INFO)
    
    # Test ICDAR
    icdar_root = "/Users/kenzabenkirane/Documents/GitHub/research-playground/datasets/parsing/ICDAR"
    print(f"\n=== Testing ICDAR Dataset ===")
    print(f"Validation: {validate_dataset('ICDAR', icdar_root)}")
    
    try:
        icdar = DatasetRegistry.get_dataset('ICDAR', icdar_root, sample_limit=5)
        print(f"Loaded {len(icdar)} samples")
        print(f"Stats: {get_dataset_stats(icdar)}")
        
        for i, sample in enumerate(icdar):
            print(f"\nSample {i}: {sample.sample_id}")
            print(f"  Ground truth (first 100 chars): {sample.ground_truth[:100]}...")
            print(f"  Languages: {sample.metadata.get('languages', [])}")
    except Exception as e:
        print(f"Error: {e}")
