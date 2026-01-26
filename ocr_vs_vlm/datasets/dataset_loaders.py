"""
Unified dataset loaders for OCR/VLM benchmarking.

All loaders return a consistent interface: Sample(image_path, ground_truth, metadata)
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple, Literal
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


@dataclass
class QASample(Sample):
    """
    Extended sample for Question-Answering tasks.
    
    Inherits from Sample for backward compatibility with parsing tasks.
    For parsing mode: ground_truth contains full document text
    For QA mode: ground_truth contains the primary answer, answers contains all valid answers
    
    Attributes:
        question: The question to answer about the image
        answers: List of valid answers (for QA evaluation with multiple correct answers)
        question_type: Category of question (layout, table/list, form, etc.)
    """
    question: str = ""
    answers: List[str] = field(default_factory=list)
    question_type: str = ""


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
                
                # Try to load ground truth if available, otherwise use empty string
                # Ground truth will be populated by models during evaluation
                ground_truth = self._get_iam_ground_truth(image_path) or ""
                
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
                unique_languages = list(set(languages))
                metadata = {
                    'dataset': 'ICDAR',
                    'languages': unique_languages,
                    'language': ', '.join(unique_languages),  # Single string for easy access
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


class ICDARMiniDataset(Dataset):
    """
    ICDAR Mini Dataset loader - loads pre-sampled balanced ICDAR data.
    
    Loads from JSON files in datasets_subsets folder.
    """
    
    def _load(self):
        """Load ICDAR mini samples from JSON files."""
        # Find the icdar_mini subfolder in datasets_subsets
        # Try relative to this file, then try common paths
        possible_paths = [
            Path(__file__).parent / "datasets_subsets" / "icdar_mini",
            Path(__file__).parent.parent / "ocr_vs_vlm" / "datasets_subsets" / "icdar_mini",
            self.dataset_root / "icdar_mini" if self.dataset_root else None,
        ]
        
        subsets_dir = None
        for path in possible_paths:
            if path and path.exists():
                subsets_dir = path
                break
        
        if not subsets_dir:
            raise FileNotFoundError(
                f"ICDAR_mini icdar_mini folder not found. "
                f"Tried: {[str(p) for p in possible_paths if p]}"
            )
        
        # Load all language JSON files
        json_files = sorted(subsets_dir.glob("icdar_mini_*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No ICDAR_mini JSON files found in {subsets_dir}")
        
        # Skip the index file, load language files
        for json_file in json_files:
            if json_file.name == "icdar_mini_index.json":
                continue
            
            try:
                with open(json_file, 'r') as f:
                    lang_data = json.load(f)
                
                language = lang_data.get('language', 'unknown')
                
                for sample_data in lang_data.get('samples', []):
                    # Ensure image_path is absolute
                    image_path = sample_data['image_path']
                    if not Path(image_path).is_absolute():
                        # Make it absolute if stored relatively
                        image_path = str(Path(image_path))
                    
                    metadata = sample_data.get('metadata', {})
                    metadata['dataset'] = 'ICDAR_mini'
                    
                    self.samples.append(Sample(
                        sample_id=sample_data['sample_id'],
                        image_path=image_path,
                        ground_truth=sample_data['ground_truth'],
                        metadata=metadata
                    ))
            
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.samples)} samples from ICDAR_mini")
    
    @staticmethod
    def _get_image_size(image_path: Path) -> Tuple[int, int]:
        """Get image dimensions."""
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return (0, 0)


class IAMMiniDataset(Dataset):
    """
    IAM Mini Dataset loader - loads pre-sampled IAM handwriting data.
    
    Loads from JSON files in datasets_subsets folder.
    
    Each sample contains:
    - handwritten.png: The handwritten text image (input to VLM for prediction)
    - printed.png: The printed reference text (ground truth image)
    
    The VLM should read the handwritten image and produce text that matches
    what appears in the printed reference.
    """
    
    def __init__(self, dataset_root: Path, crop_handwritten_only: bool = False, **kwargs):
        """
        Initialize IAMMiniDataset.
        
        Args:
            dataset_root: Root directory of dataset
            crop_handwritten_only: If True, automatically extract handwritten portions only
            **kwargs: Additional arguments passed to parent Dataset
        """
        self.crop_handwritten_only = crop_handwritten_only
        super().__init__(dataset_root, **kwargs)
    
    def _load(self):
        """Load IAM mini samples from folder structure."""
        # Find the iam_mini subfolder in datasets_subsets
        # Try relative to this file, then try common paths
        possible_paths = [
            Path(__file__).parent / "datasets_subsets" / "iam_mini",
            Path(__file__).parent.parent / "ocr_vs_vlm" / "datasets_subsets" / "iam_mini",
            self.dataset_root / "iam_mini" if self.dataset_root else None,
        ]
        
        subsets_dir = None
        for path in possible_paths:
            if path and path.exists():
                subsets_dir = path
                break
        
        if not subsets_dir:
            raise FileNotFoundError(
                f"IAM_mini iam_mini folder not found. "
                f"Tried: {[str(p) for p in possible_paths if p]}"
            )
        
        # Load from the new folder structure with index.json
        index_file = subsets_dir / "iam_mini_index.json"
        
        # If new structure exists (iam_mini_index.json), use it
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index_data = json.load(f)
                
                for sample_data in index_data.get('samples', []):
                    sample_id = sample_data['sample_id']
                    folder = sample_data['folder']
                    
                    # Handwritten image = input for VLM prediction
                    handwritten_path = subsets_dir / folder / "handwritten.png"
                    # Printed image = ground truth reference
                    printed_path = subsets_dir / folder / "printed.png"
                    
                    if not handwritten_path.exists():
                        logger.debug(f"Handwritten image not found for {sample_id}: {handwritten_path}")
                        continue
                    
                    metadata = sample_data.get('metadata', {})
                    metadata['dataset'] = 'IAM_mini'
                    metadata['folder'] = folder
                    # Store printed image path for ground truth reference
                    metadata['printed_image_path'] = str(printed_path) if printed_path.exists() else None
                    metadata['handwritten_image_path'] = str(handwritten_path)
                    metadata['crop_valid'] = True  # Pre-cropped images are valid
                    
                    self.samples.append(Sample(
                        sample_id=sample_id,
                        image_path=str(handwritten_path),  # VLM reads handwritten image
                        ground_truth='',  # Text ground truth - empty, use printed_image_path for image-based GT
                        metadata=metadata
                    ))
            
            except Exception as e:
                raise FileNotFoundError(f"Failed to load {index_file}: {e}")
        
        else:
            # Fall back to old iam_mini.json format
            json_file = subsets_dir / "iam_mini.json"
            if not json_file.exists():
                raise FileNotFoundError(f"No iam_mini_index.json or iam_mini.json found in {subsets_dir}")
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for sample_data in data.get('samples', []):
                    # Ensure image_path is absolute
                    image_path = sample_data['image_path']
                    if not Path(image_path).is_absolute():
                        # Make it absolute if stored relatively
                        image_path = str(Path(image_path))
                    
                    metadata = sample_data.get('metadata', {})
                    metadata['dataset'] = 'IAM_mini'
                    
                    # If cropping enabled, detect Line 2 boundary
                    crop_metadata = {}
                    if self.crop_handwritten_only:
                        try:
                            from .datasets_subsets.iam_mini.line2_detection import find_line2, validate_crop
                            line2 = find_line2(image_path)
                            crop_metadata['line2'] = line2
                            crop_metadata['crop_valid'] = validate_crop(image_path, line2)
                        except Exception as e:
                            logger.debug(f"Could not detect Line 2 for {image_path}: {e}")
                            crop_metadata['line2'] = None
                            crop_metadata['crop_valid'] = False
                    
                    metadata.update(crop_metadata)
                    
                    self.samples.append(Sample(
                        sample_id=sample_data['sample_id'],
                        image_path=image_path,
                        ground_truth=sample_data.get('ground_truth', ''),
                        metadata=metadata
                    ))
            
            except Exception as e:
                raise FileNotFoundError(f"Failed to load {json_file}: {e}")
        
        mode_str = "(handwritten pre-cropped)" if len(self.samples) > 0 and (subsets_dir / index_file.name).exists() else ""
        logger.info(f"Loaded {len(self.samples)} samples from IAM_mini {mode_str}")
    
    def __getitem__(self, idx: int) -> Sample:
        """
        Get a sample, optionally cropping to handwritten portion only.
        
        If crop_handwritten_only is enabled, returns a Sample with:
        - image_path: Temporary path to cropped handwritten-only image
        - metadata: Includes 'line2' and 'crop_valid' fields
        
        Args:
            idx: Sample index
        
        Returns:
            Sample with potentially cropped image
        """
        sample = self.samples[idx]
        
        if not self.crop_handwritten_only or sample.metadata.get('line2') is None:
            return sample
        
        # Crop to handwritten portion
        try:
            from .datasets_subsets.iam_mini.line2_detection import crop_handwritten_only
            
            line2 = sample.metadata['line2']
            hw_img = crop_handwritten_only(sample.image_path, line2)
            
            # Save cropped image to temporary file
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "iam_cropped"
            temp_dir.mkdir(exist_ok=True)
            
            stem = Path(sample.image_path).stem
            temp_path = temp_dir / f"{stem}_hw.png"
            hw_img.save(temp_path)
            
            # Return sample with cropped image path
            cropped_sample = Sample(
                sample_id=sample.sample_id,
                image_path=str(temp_path),
                ground_truth=sample.ground_truth,
                metadata=sample.metadata
            )
            return cropped_sample
        
        except Exception as e:
            logger.warning(f"Failed to crop sample {sample.sample_id}: {e}")
            return sample
    
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


class VOC2007Dataset(Dataset):
    """
    VOC2007 Medical Lab Reports Dataset loader.
    
    Structure:
    - JPEGImages/*.jpg (images)
    - labels_src.json (annotations with Chinese text)
    
    This dataset contains Simplified Chinese medical laboratory reports.
    Each image has multiple text annotations in table format.
    """
    
    def _load(self):
        """Load VOC2007 images and annotations."""
        images_dir = self.dataset_root / "JPEGImages"
        labels_file = self.dataset_root / "labels_src.json"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"VOC2007 images directory not found: {images_dir}")
        
        if not labels_file.exists():
            raise FileNotFoundError(f"VOC2007 labels file not found: {labels_file}")
        
        # Load annotations
        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load labels_src.json: {e}")
        
        # Build filename -> annotation mapping
        filename_to_annotation = {}
        for entry in annotations:
            filename = entry.get('filename')
            if filename:
                filename_to_annotation[filename] = entry
        
        # Process each image
        for image_path in sorted(images_dir.glob("*.jpg")):
            filename = image_path.name
            
            if filename not in filename_to_annotation:
                logger.debug(f"No annotation for {filename}")
                continue
            
            entry = filename_to_annotation[filename]
            sample_id = f"voc2007_{image_path.stem}"
            
            # Extract all text from annotations
            ground_truth = self._extract_ground_truth(entry)
            
            metadata = {
                'dataset': 'VOC2007',
                'language': 'zh-CN',  # Simplified Chinese
                'document_type': 'medical_lab_report',
                'num_annotations': len(entry.get('annotations', [])),
                'image_size': self._get_image_size(image_path),
            }
            
            self.samples.append(Sample(
                sample_id=sample_id,
                image_path=str(image_path),
                ground_truth=ground_truth,
                metadata=metadata
            ))
    
    def _extract_ground_truth(self, entry: Dict) -> str:
        """
        Extract all text from annotations.
        
        Args:
            entry: Annotation entry with 'annotations' list
        
        Returns:
            Concatenated text preserving structure (Unicode Chinese)
        """
        texts = []
        annotations = entry.get('annotations', [])
        
        # Sort by table_no, then by y position (top to bottom), then x (left to right)
        sorted_annotations = sorted(
            annotations,
            key=lambda a: (
                int(a.get('table_no', 0)),
                int(a.get('cell_row', 0)),
                int(a.get('cell_line', 0)),
                float(a.get('y', 0)),
                float(a.get('x', 0))
            )
        )
        
        for anno in sorted_annotations:
            text = anno.get('text', '').strip()
            if text:
                texts.append(text)
        
        # Join texts - use newline for structure
        return '\n'.join(texts)
    
    @staticmethod
    def _get_image_size(image_path: Path) -> Tuple[int, int]:
        """Get image dimensions."""
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return (0, 0)


class RXPADDataset(Dataset):
    """
    RX-PAD French Medical Prescription Dataset loader.
    
    Structure:
    - training_data/images/*.png (images)
    - training_data/annotations/*.json (field-level annotations)
    - testing_data/images/*.png (images)
    - testing_data/annotations/*.json (field-level annotations)
    
    This dataset contains French medical prescription forms with field-level
    bounding boxes and text annotations for key fields (prescriber, patient,
    medication, dosage, etc.).
    """
    
    def _load(self):
        """Load RX-PAD images and annotations from both training and testing sets."""
        self.samples = []
        
        # Load both training and testing data
        for split in ['training_data', 'testing_data']:
            split_dir = self.dataset_root / split
            images_dir = split_dir / "images"
            annotations_dir = split_dir / "annotations"
            
            if not images_dir.exists():
                logger.warning(f"RX-PAD {split} images directory not found: {images_dir}")
                continue
            
            if not annotations_dir.exists():
                logger.warning(f"RX-PAD {split} annotations directory not found: {annotations_dir}")
                continue
            
            # Process each image
            for image_path in sorted(images_dir.glob("*.png")):
                # Find corresponding annotation file
                annotation_name = image_path.stem + ".json"
                annotation_path = annotations_dir / annotation_name
                
                if not annotation_path.exists():
                    logger.debug(f"No annotation for {image_path.name}")
                    continue
                
                sample_id = f"rxpad_{split}_{image_path.stem}"
                
                # Load and extract ground truth
                try:
                    with open(annotation_path, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                    ground_truth = self._extract_ground_truth(annotation)
                except Exception as e:
                    logger.warning(f"Failed to load annotation {annotation_path}: {e}")
                    continue
                
                metadata = {
                    'dataset': 'RX-PAD',
                    'language': 'fr',  # French
                    'document_type': 'medical_prescription',
                    'split': split,
                    'image_size': self._get_image_size(image_path),
                }
                
                self.samples.append(Sample(
                    sample_id=sample_id,
                    image_path=str(image_path),
                    ground_truth=ground_truth,
                    metadata=metadata
                ))
    
    def _extract_ground_truth(self, annotation: Dict) -> str:
        """
        Extract all text from prescription annotation.
        
        Args:
            annotation: Annotation dict with 'prescr' list containing fields
        
        Returns:
            Concatenated text with field structure
        """
        texts = []
        prescr_fields = annotation.get('prescr', [])
        
        for field in prescr_fields:
            label = field.get('label', '')
            text = field.get('text', '').strip()
            
            if text:
                if label:
                    # Include label for structured output
                    texts.append(f"{label}: {text}")
                else:
                    texts.append(text)
        
        # Join texts - use newline for structure
        return '\n'.join(texts)
    
    @staticmethod
    def _get_image_size(image_path: Path) -> Tuple[int, int]:
        """Get image dimensions."""
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return (0, 0)


class DocVQADataset(Dataset):
    """
    DocVQA Dataset loader supporting both parsing and QA tasks.
    
    Loads from HuggingFace parquet format with embedded images.
    
    Structure:
    - DocVQA/{split}-*.parquet
    
    Modes:
    - 'qa': Each question-answer pair is a sample (returns QASample)
    - 'parsing': Each unique document is a sample (returns Sample with full text)
    
    Args:
        dataset_root: Root path to DocVQA_hf folder
        mode: 'qa' or 'parsing'
        split: 'train', 'validation', or 'test'
        sample_limit: Max samples to load
    """
    
    def __init__(
        self,
        dataset_root: str,
        mode: Literal['qa', 'parsing'] = 'qa',
        split: str = 'validation',
        sample_limit: Optional[int] = None,
        **kwargs
    ):
        self.mode = mode
        self.split = split
        self._cache_dir: Optional[Path] = None
        super().__init__(dataset_root, sample_limit=sample_limit)
    
    def _get_cache_dir(self) -> Path:
        """Get or create image cache directory."""
        if self._cache_dir is None:
            self._cache_dir = self.dataset_root / ".cache" / "images" / "DocVQA" / self.split
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir
    
    def _extract_image(self, image_data: Dict, image_name: str) -> str:
        """
        Extract image bytes to disk and return path.
        
        Args:
            image_data: Dict with 'bytes' and 'path' keys
            image_name: Fallback name if path is empty
        
        Returns:
            Absolute path to extracted image file
        """
        cache_dir = self._get_cache_dir()
        
        # Use original path or fallback to provided name
        filename = image_data.get('path') or f"{image_name}.png"
        image_path = cache_dir / filename
        
        # Only extract if not already cached
        if not image_path.exists():
            image_bytes = image_data.get('bytes', b'')
            if image_bytes:
                image_path.write_bytes(image_bytes)
        
        return str(image_path)
    
    def _load(self):
        """Load DocVQA samples from parquet files."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("pyarrow not installed. Install with: pip install pyarrow")
            return
        
        data_dir = self.dataset_root / "DocVQA"
        if not data_dir.exists():
            raise FileNotFoundError(f"DocVQA directory not found: {data_dir}")
        
        # Find parquet files for the split
        parquet_files = sorted(data_dir.glob(f"{self.split}-*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found for split '{self.split}' in {data_dir}")
        
        if self.mode == 'qa':
            self._load_qa_mode(parquet_files)
        else:
            self._load_parsing_mode(parquet_files)
    
    def _load_qa_mode(self, parquet_files: List[Path]):
        """Load in QA mode: each question is a sample."""
        import pyarrow.parquet as pq
        
        for parquet_file in parquet_files:
            try:
                df = pq.read_table(parquet_file).to_pandas()
                
                for _, row in df.iterrows():
                    # Extract image to cache
                    image_path = self._extract_image(
                        row['image'],
                        f"doc_{row['docId']}_{row['questionId']}"
                    )
                    
                    # Get answers list (handle numpy arrays)
                    raw_answers = row['answers']
                    answers = list(raw_answers) if hasattr(raw_answers, '__iter__') and not isinstance(raw_answers, str) else [raw_answers]
                    
                    # Get question types (handle numpy arrays)
                    raw_q_types = row.get('question_types', [])
                    q_types = list(raw_q_types) if hasattr(raw_q_types, '__iter__') and not isinstance(raw_q_types, str) else []
                    question_type = q_types[0] if len(q_types) > 0 else 'unknown'
                    
                    sample = QASample(
                        sample_id=f"docvqa_{row['questionId']}",
                        image_path=image_path,
                        ground_truth=answers[0] if answers else "",  # Primary answer for backward compat
                        metadata={
                            'dataset': 'DocVQA',
                            'mode': 'qa',
                            'split': self.split,
                            'docId': row['docId'],
                            'ucsf_document_id': row.get('ucsf_document_id', ''),
                            'ucsf_document_page_no': row.get('ucsf_document_page_no', ''),
                            'question_types': q_types,
                        },
                        question=row['question'],
                        answers=answers,
                        question_type=question_type,
                    )
                    self.samples.append(sample)
                    
            except Exception as e:
                logger.warning(f"Failed to load {parquet_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.samples)} QA samples from DocVQA ({self.split})")
    
    def _load_parsing_mode(self, parquet_files: List[Path]):
        """Load in parsing mode: each unique document is a sample."""
        import pyarrow.parquet as pq
        
        # Collect all data first to group by document
        doc_data: Dict[int, Dict] = {}
        
        for parquet_file in parquet_files:
            try:
                df = pq.read_table(parquet_file).to_pandas()
                
                for _, row in df.iterrows():
                    doc_id = row['docId']
                    
                    if doc_id not in doc_data:
                        # First occurrence of this document
                        image_path = self._extract_image(
                            row['image'],
                            f"doc_{doc_id}"
                        )
                        doc_data[doc_id] = {
                            'image_path': image_path,
                            'answers': [],
                            'questions': [],
                            'ucsf_document_id': row.get('ucsf_document_id', ''),
                            'ucsf_document_page_no': row.get('ucsf_document_page_no', ''),
                        }
                    
                    # Collect all answers as text content
                    raw_answers = row['answers']
                    answers = list(raw_answers) if hasattr(raw_answers, '__iter__') and not isinstance(raw_answers, str) else [raw_answers]
                    doc_data[doc_id]['answers'].extend(answers)
                    doc_data[doc_id]['questions'].append(row['question'])
                    
            except Exception as e:
                logger.warning(f"Failed to load {parquet_file}: {e}")
                continue
        
        # Create samples from grouped documents
        for doc_id, data in doc_data.items():
            # Ground truth is all unique text snippets from answers
            unique_texts = list(dict.fromkeys(data['answers']))  # Preserve order, remove dupes
            ground_truth = '\n'.join(unique_texts)
            
            sample = Sample(
                sample_id=f"docvqa_doc_{doc_id}",
                image_path=data['image_path'],
                ground_truth=ground_truth,
                metadata={
                    'dataset': 'DocVQA',
                    'mode': 'parsing',
                    'split': self.split,
                    'docId': doc_id,
                    'ucsf_document_id': data['ucsf_document_id'],
                    'ucsf_document_page_no': data['ucsf_document_page_no'],
                    'num_questions': len(data['questions']),
                    'num_text_snippets': len(unique_texts),
                },
            )
            self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} document samples from DocVQA ({self.split})")


class InfographicVQADataset(Dataset):
    """
    InfographicVQA Dataset loader supporting both parsing and QA tasks.
    
    Similar to DocVQA but includes pre-extracted OCR from AWS Textract.
    
    Structure:
    - InfographicVQA/{split}-*.parquet
    
    Modes:
    - 'qa': Each question-answer pair is a sample (returns QASample)
    - 'parsing': Each unique image is a sample with OCR as ground truth
    
    Args:
        dataset_root: Root path to DocVQA_hf folder
        mode: 'qa' or 'parsing'
        split: 'train', 'validation', or 'test'
        sample_limit: Max samples to load
    """
    
    def __init__(
        self,
        dataset_root: str,
        mode: Literal['qa', 'parsing'] = 'qa',
        split: str = 'validation',
        sample_limit: Optional[int] = None,
        **kwargs
    ):
        self.mode = mode
        self.split = split
        self._cache_dir: Optional[Path] = None
        super().__init__(dataset_root, sample_limit=sample_limit)
    
    def _get_cache_dir(self) -> Path:
        """Get or create image cache directory."""
        if self._cache_dir is None:
            self._cache_dir = self.dataset_root / ".cache" / "images" / "InfographicVQA" / self.split
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir
    
    def _extract_image(self, image_data: Dict, image_name: str) -> str:
        """Extract image bytes to disk and return path."""
        cache_dir = self._get_cache_dir()
        
        filename = image_data.get('path') or f"{image_name}.png"
        image_path = cache_dir / filename
        
        if not image_path.exists():
            image_bytes = image_data.get('bytes', b'')
            if image_bytes:
                image_path.write_bytes(image_bytes)
        
        return str(image_path)
    
    def _extract_ocr_text(self, ocr_json_str: str) -> str:
        """
        Extract plain text from AWS Textract OCR JSON.
        
        Args:
            ocr_json_str: JSON string containing Textract output
        
        Returns:
            Extracted plain text
        """
        if not ocr_json_str:
            return ""
        
        try:
            import ast
            
            # The OCR field may be a Python list repr string like "['...json...']"
            if isinstance(ocr_json_str, str) and ocr_json_str.startswith('['):
                try:
                    parsed = ast.literal_eval(ocr_json_str)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        ocr_data = json.loads(parsed[0])
                    else:
                        return ""
                except (ValueError, SyntaxError):
                    ocr_data = json.loads(ocr_json_str)
            else:
                ocr_data = json.loads(ocr_json_str) if isinstance(ocr_json_str, str) else ocr_json_str
            
            # Handle list wrapper
            if isinstance(ocr_data, list) and len(ocr_data) > 0:
                ocr_data = json.loads(ocr_data[0]) if isinstance(ocr_data[0], str) else ocr_data[0]
            
            # Extract LINE text from Textract format
            # Textract stores LINE blocks in a separate 'LINE' key, not nested in 'PAGE'
            texts = []
            
            # Try LINE key first (common Textract format)
            if 'LINE' in ocr_data:
                for block in ocr_data.get('LINE', []):
                    text = block.get('Text', '')
                    if text:
                        texts.append(text)
            
            # Fallback: check PAGE for LINE blocks
            if not texts and 'PAGE' in ocr_data:
                for block in ocr_data.get('PAGE', []):
                    if block.get('BlockType') == 'LINE':
                        text = block.get('Text', '')
                        if text:
                            texts.append(text)
            
            return '\n'.join(texts)
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.debug(f"Failed to parse OCR JSON: {e}")
            return ""
    
    def _load(self):
        """Load InfographicVQA samples from parquet files."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            logger.error("pyarrow not installed. Install with: pip install pyarrow")
            return
        
        data_dir = self.dataset_root / "InfographicVQA"
        if not data_dir.exists():
            raise FileNotFoundError(f"InfographicVQA directory not found: {data_dir}")
        
        parquet_files = sorted(data_dir.glob(f"{self.split}-*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found for split '{self.split}' in {data_dir}")
        
        if self.mode == 'qa':
            self._load_qa_mode(parquet_files)
        else:
            self._load_parsing_mode(parquet_files)
    
    def _load_qa_mode(self, parquet_files: List[Path]):
        """Load in QA mode: each question is a sample."""
        import pyarrow.parquet as pq
        
        for parquet_file in parquet_files:
            try:
                df = pq.read_table(parquet_file).to_pandas()
                
                for _, row in df.iterrows():
                    image_path = self._extract_image(
                        row['image'],
                        f"infovqa_{row['questionId']}"
                    )
                    
                    # Handle numpy arrays
                    raw_answers = row['answers']
                    answers = list(raw_answers) if hasattr(raw_answers, '__iter__') and not isinstance(raw_answers, str) else [raw_answers]
                    
                    raw_answer_types = row.get('answer_type', [])
                    answer_types = list(raw_answer_types) if hasattr(raw_answer_types, '__iter__') and not isinstance(raw_answer_types, str) else []
                    question_type = answer_types[0] if len(answer_types) > 0 else 'unknown'
                    
                    # Handle operations/reasoning field
                    raw_ops = row.get('operation/reasoning', [])
                    operations = list(raw_ops) if hasattr(raw_ops, '__iter__') and not isinstance(raw_ops, str) else []
                    
                    # Extract OCR text for reference
                    ocr_text = self._extract_ocr_text(row.get('ocr', ''))
                    
                    sample = QASample(
                        sample_id=f"infovqa_{row['questionId']}",
                        image_path=image_path,
                        ground_truth=answers[0] if answers else "",
                        metadata={
                            'dataset': 'InfographicVQA',
                            'mode': 'qa',
                            'split': self.split,
                            'image_url': row.get('image_url', ''),
                            'answer_types': answer_types,
                            'operations': operations,
                            'ocr_text': ocr_text,  # Pre-extracted OCR available
                        },
                        question=row['question'],
                        answers=answers,
                        question_type=question_type,
                    )
                    self.samples.append(sample)
                    
            except Exception as e:
                logger.warning(f"Failed to load {parquet_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.samples)} QA samples from InfographicVQA ({self.split})")
    
    def _load_parsing_mode(self, parquet_files: List[Path]):
        """Load in parsing mode: each unique image is a sample with OCR as ground truth."""
        import pyarrow.parquet as pq
        
        # Group by image URL (unique identifier for infographics)
        image_data: Dict[str, Dict] = {}
        
        for parquet_file in parquet_files:
            try:
                df = pq.read_table(parquet_file).to_pandas()
                
                for _, row in df.iterrows():
                    image_url = row.get('image_url', row['questionId'])
                    
                    if image_url not in image_data:
                        image_path = self._extract_image(
                            row['image'],
                            f"infovqa_{hash(image_url) % 10**8}"
                        )
                        ocr_text = self._extract_ocr_text(row.get('ocr', ''))
                        
                        image_data[image_url] = {
                            'image_path': image_path,
                            'ocr_text': ocr_text,
                            'image_url': image_url,
                            'questions': [],
                            'answers': [],
                        }
                    
                    raw_answers = row['answers']
                    answers = list(raw_answers) if hasattr(raw_answers, '__iter__') and not isinstance(raw_answers, str) else [raw_answers]
                    image_data[image_url]['answers'].extend(answers)
                    image_data[image_url]['questions'].append(row['question'])
                    
            except Exception as e:
                logger.warning(f"Failed to load {parquet_file}: {e}")
                continue
        
        # Create samples
        for idx, (image_url, data) in enumerate(image_data.items()):
            # Use OCR text as ground truth for parsing evaluation
            ground_truth = data['ocr_text'] if data['ocr_text'] else '\n'.join(dict.fromkeys(data['answers']))
            
            sample = Sample(
                sample_id=f"infovqa_img_{idx}",
                image_path=data['image_path'],
                ground_truth=ground_truth,
                metadata={
                    'dataset': 'InfographicVQA',
                    'mode': 'parsing',
                    'split': self.split,
                    'image_url': data['image_url'],
                    'num_questions': len(data['questions']),
                    'has_ocr': bool(data['ocr_text']),
                },
            )
            self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} image samples from InfographicVQA ({self.split})")


class DatasetRegistry:
    """
    Central registry for dataset instantiation.
    
    Provides unified interface: get_dataset("IAM", path, limit)
    """
    
    _DATASETS = {
        'IAM': IAMDataset,
        'ICDAR': ICDARDataset,
        'ICDAR_mini': ICDARMiniDataset,
        'IAM_mini': IAMMiniDataset,
        'PubLayNet': PubLayNetDataset,
        'VOC2007': VOC2007Dataset,
        'RX-PAD': RXPADDataset,
        'DocVQA': DocVQADataset,
        'InfographicVQA': InfographicVQADataset,
    }
    
    @classmethod
    def register(cls, name: str, dataset_class):
        """Register a new dataset loader."""
        cls._DATASETS[name] = dataset_class
    
    @classmethod
    def get_dataset(cls, name: str, root: str, sample_limit: Optional[int] = None, **kwargs) -> Dataset:
        """
        Get dataset instance by name.
        
        Args:
            name: Dataset name (IAM, ICDAR, PubLayNet, DocVQA, InfographicVQA)
            root: Root directory path
            sample_limit: Max samples to load
            **kwargs: Additional arguments for specific loaders:
                - mode: 'qa' or 'parsing' (for DocVQA/InfographicVQA)
                - split: 'train', 'validation', or 'test' (for DocVQA/InfographicVQA)
        
        Returns:
            Dataset instance
        
        Raises:
            ValueError: If dataset name not recognized
        """
        if name not in cls._DATASETS:
            available = ", ".join(cls._DATASETS.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        
        dataset_class = cls._DATASETS[name]
        return dataset_class(root, sample_limit=sample_limit, **kwargs)
    
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
        
        elif dataset_name == 'VOC2007':
            images_dir = root_path / "JPEGImages"
            labels_file = root_path / "labels_src.json"
            
            results['checks']['JPEGImages_exists'] = images_dir.exists()
            results['checks']['labels_src_exists'] = labels_file.exists()
            
            if images_dir.exists():
                results['checks']['images_count'] = len(list(images_dir.glob("*.jpg")))
        
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


def create_icdar_mini(icdar_root: str, samples_per_language: int = 50) -> Dict:
    """
    Create a mini version of ICDAR dataset with balanced sampling per language.
    
    Args:
        icdar_root: Root path to ICDAR dataset
        samples_per_language: Number of samples to select per language
    
    Returns:
        Dict mapping language -> List[Sample] with balanced representation
    """
    # Load full ICDAR dataset
    icdar = DatasetRegistry.get_dataset('ICDAR', icdar_root)
    
    # Group samples by language
    language_samples = {}
    for sample in icdar:
        languages = sample.metadata.get('languages', [])
        # A sample can have multiple languages, assign to all
        for lang in languages:
            if lang not in language_samples:
                language_samples[lang] = []
            language_samples[lang].append(sample)
    
    # Sample up to N items per language
    mini_dataset = {}
    for lang, samples in language_samples.items():
        # Randomly sample up to samples_per_language items
        if len(samples) > samples_per_language:
            import random
            mini_dataset[lang] = random.sample(samples, samples_per_language)
        else:
            mini_dataset[lang] = samples
    
    return mini_dataset


def save_icdar_mini_index(mini_dataset: Dict, output_dir: str):
    """
    Save ICDAR mini dataset index as JSON for reference.
    
    Args:
        mini_dataset: Dict mapping language -> List[Sample]
        output_dir: Directory to save index files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metadata for each language
    for lang, samples in mini_dataset.items():
        lang_file = output_path / f"icdar_mini_{lang}.json"
        
        lang_data = {
            'language': lang,
            'total_samples': len(samples),
            'samples': []
        }
        
        for sample in samples:
            lang_data['samples'].append({
                'sample_id': sample.sample_id,
                'image_path': sample.image_path,
                'ground_truth': sample.ground_truth,
                'metadata': sample.metadata
            })
        
        with open(lang_file, 'w') as f:
            json.dump(lang_data, f, indent=2)
        
        logger.info(f"Saved {len(samples)} samples for language {lang} to {lang_file}")
    
    # Save master index
    index_file = output_path / "icdar_mini_index.json"
    index_data = {
        'dataset': 'ICDAR_mini',
        'total_languages': len(mini_dataset),
        'languages': {lang: len(samples) for lang, samples in mini_dataset.items()},
        'total_samples': sum(len(samples) for samples in mini_dataset.values())
    }
    
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    logger.info(f"Saved ICDAR_mini index with {index_data['total_languages']} languages and {index_data['total_samples']} total samples")
    
    return index_file

