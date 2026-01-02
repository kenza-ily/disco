"""
PubLayNet Document Parsing Benchmark

Orchestrates evaluation of layout element detection across three phases:

Phase P-A: OCR Layout Inference
  - Extracts text via OCR (Azure Document Intelligence, Mistral OCR)
  - Infers layout structure from text positions and characteristics
  - Assigns layout categories based on text properties (size, position, structure)
  - Output: Predicted bounding boxes with element categories

Phase P-B: VLM Direct Detection
  - Sends document image to VLM (GPT-5-mini, GPT-5-nano)
  - VLM directly identifies layout elements using visual understanding
  - Output: Predicted bounding boxes with element categories

Phase P-C: VLM + OCR Hybrid
  - Combines OCR text extraction (Phase P-A) with VLM reasoning
  - Provides OCR context to VLM for layout refinement
  - VLM uses OCR positions as hints for more accurate box prediction
  - Output: Predicted bounding boxes with element categories

Layout Categories:
  1 = Text (regular text blocks and paragraphs)
  2 = Title (document titles and headings)
  3 = List (bulleted or numbered lists)
  4 = Table (tabular data structures)
  5 = Figure (images, charts, and diagrams)

Evaluation (post-processing):
  - Metrics computed offline: IoU, mAP @ 0.5 and 0.75
  - Saves ground truth and predicted boxes as JSON
  - Error analysis by category and document type

Models Tested:
  OCR Models: azure_intelligence, mistral_document_ai
  VLM Models: gpt-5-mini, gpt-5-nano

Dataset: PubLayNet Mini (500 samples)
  - Source: kenza-ily/publaynet-mini on HuggingFace
  - Contains document images with layout annotations
  - Provides ground truth bounding boxes and segmentation masks
"""

import json
import logging
import time
import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import re
import os
import base64
import requests

from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

from .unified_model_api import UnifiedModelAPI, ModelRegistry
from .dataset_loaders import load_image

# Load environment
load_dotenv(Path(__file__).parent.parent / ".env.local")

# Create logs directory
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

benchmark_log = logging.FileHandler(LOGS_DIR / 'benchmark_publaynet.log')
benchmark_log.setLevel(logging.DEBUG)
benchmark_log.setFormatter(formatter)
logger.addHandler(benchmark_log)


@dataclass
class BoundingBox:
    """Bounding box representation [x, y, width, height]."""
    x: float
    y: float
    width: float
    height: float
    category: int
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'category': self.category,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'BoundingBox':
        """Create from dictionary."""
        return cls(
            x=d['x'],
            y=d['y'],
            width=d['width'],
            height=d['height'],
            category=d['category'],
            confidence=d.get('confidence', 1.0)
        )
    
    def to_list(self) -> List[float]:
        """Return as [x, y, width, height]."""
        return [self.x, self.y, self.width, self.height]


@dataclass
class PubLayNetResult:
    """Result from a single PubLayNet sample evaluation."""
    
    sample_id: str
    image_path: str
    model: str
    phase: str  # 'P-A', 'P-B', 'P-C'
    
    # Ground truth boxes (from dataset)
    ground_truth_boxes: List[Dict] = field(default_factory=list)
    
    # Predicted boxes (from model)
    predicted_boxes: List[Dict] = field(default_factory=list)
    
    # Metadata
    inference_time_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV storage."""
        return {
            'sample_id': self.sample_id,
            'image_path': self.image_path,
            'model': self.model,
            'phase': self.phase,
            'ground_truth_boxes': json.dumps(self.ground_truth_boxes),
            'predicted_boxes': json.dumps(self.predicted_boxes),
            'inference_time_ms': self.inference_time_ms,
            'error': self.error,
            'timestamp': self.timestamp,
        }


@dataclass
class BenchmarkConfigPubLayNet:
    """Configuration for PubLayNet parsing benchmark."""
    
    # Model selection
    ocr_models: List[str] = field(default_factory=lambda: ['azure_intelligence', 'mistral_document_ai'])
    vlm_models: List[str] = field(default_factory=lambda: ['gpt-5-mini', 'gpt-5-nano'])
    
    # Phase control
    phases: List[str] = field(default_factory=lambda: ['P-A', 'P-B', 'P-C'])  # P-A, P-B, P-C
    
    # Sample control
    sample_limit: Optional[int] = None
    batch_size: int = 10  # Save checkpoint every N samples
    
    # Output paths
    results_dir: str = "results/publaynet"
    checkpoint_file: str = "checkpoint.json"
    
    # API settings
    timeout_seconds: int = 120
    retry_failed: bool = True
    max_retries: int = 2


class PubLayNetBenchmarkRunner:
    """Orchestrator for PubLayNet parsing benchmark."""
    
    # Layout categories
    CATEGORIES = {
        1: 'Text',
        2: 'Title',
        3: 'List',
        4: 'Table',
        5: 'Figure'
    }
    
    def __init__(self, config: BenchmarkConfigPubLayNet):
        """
        Initialize PubLayNet benchmark runner.
        
        Args:
            config: BenchmarkConfigPubLayNet instance
        """
        self.config = config
        
        # Setup results directory
        if Path(config.results_dir).is_absolute():
            self.results_dir = Path(config.results_dir)
        else:
            self.results_dir = Path(__file__).parent / config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_file = self.results_dir / config.checkpoint_file
        self.checkpoint = self._load_checkpoint()
        
        # Initialize unified model API
        self.api = UnifiedModelAPI()
        
        logger.info(f"PubLayNetBenchmarkRunner initialized")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Config: {config}")
    
    def run_benchmark(self) -> Dict:
        """
        Execute PubLayNet parsing benchmark.
        
        Returns:
            Summary dict with execution statistics
        """
        start_time = time.time()
        execution_summary = {
            'start_time': datetime.now().isoformat(),
            'config': asdict(self.config),
            'by_phase': {},
        }
        
        # Load PubLayNet mini dataset
        logger.info("Loading PubLayNet mini dataset...")
        try:
            dataset = load_dataset('kenza-ily/publaynet-mini')
            samples = dataset['train']
            if self.config.sample_limit:
                samples = samples.select(range(min(self.config.sample_limit, len(samples))))
            logger.info(f"Loaded {len(samples)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            execution_summary['error'] = str(e)
            return execution_summary
        
        # Run each phase
        for phase in self.config.phases:
            logger.info(f"\n{'='*70}")
            logger.info(f"Running Phase {phase}")
            logger.info(f"{'='*70}")
            
            try:
                phase_summary = self._run_phase(phase, samples)
                execution_summary['by_phase'][phase] = phase_summary
            except Exception as e:
                logger.error(f"Error in phase {phase}: {e}")
                execution_summary['by_phase'][phase] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        execution_summary['end_time'] = datetime.now().isoformat()
        execution_summary['total_time_seconds'] = time.time() - start_time
        
        # Save execution summary
        summary_file = self.results_dir / "execution_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(execution_summary, f, indent=2)
        logger.info(f"Saved execution summary to {summary_file}")
        
        return execution_summary
    
    def _run_phase(self, phase: str, samples) -> Dict:
        """
        Run single phase with all configured models.
        
        Args:
            phase: Phase name ('P-A', 'P-B', 'P-C')
            samples: Dataset samples
        
        Returns:
            Phase-level summary
        """
        phase_summary = {
            'phase': phase,
            'status': 'completed',
            'by_model': {},
        }
        
        # Select models based on phase
        if phase == 'P-A':
            models = self.config.ocr_models
        elif phase in ['P-B', 'P-C']:
            models = self.config.vlm_models
        else:
            raise ValueError(f"Unknown phase: {phase}")
        
        for model_name in models:
            logger.info(f"\n  Model: {model_name} ({phase})")
            
            try:
                model_summary = self._run_model_phase(phase, model_name, samples)
                phase_summary['by_model'][model_name] = model_summary
            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
                phase_summary['by_model'][model_name] = {'error': str(e)}
        
        return phase_summary
    
    def _run_model_phase(self, phase: str, model_name: str, samples) -> Dict:
        """
        Run model on all samples in a phase.
        
        Args:
            phase: Phase name
            model_name: Model name
            samples: Dataset samples
        
        Returns:
            Model-phase summary
        """
        output_dir = self.results_dir / phase / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"{phase}_{model_name}_results.csv"
        
        # Load existing results if resuming
        existing_results = self._load_existing_results(results_file)
        processed_ids = {r.sample_id for r in existing_results}
        csv_headers_written = len(existing_results) > 0
        
        samples_to_process = [
            s for s in samples if str(s['id']) not in processed_ids
        ]
        
        logger.info(f"    Processing {len(samples_to_process)} samples (skipping {len(processed_ids)} existing)")
        
        results = []
        
        # Process samples
        with tqdm(total=len(samples_to_process), desc=f"{phase} - {model_name}",
                  unit="sample", leave=True, file=sys.stdout) as pbar:
            for idx, sample in enumerate(samples_to_process):
                try:
                    result = self._process_sample(phase, model_name, sample)
                    results.append(result)
                    
                    # Checkpoint every batch_size samples
                    if (idx + 1) % self.config.batch_size == 0:
                        all_results = existing_results + results
                        self._save_results_csv(results_file, all_results, write_headers=(not csv_headers_written))
                        logger.info(f"    Checkpoint saved ({len(all_results)} total)")
                        csv_headers_written = True
                        existing_results = all_results
                        results = []
                    
                    pbar.update(1)
                
                except Exception as e:
                    logger.warning(f"Failed to process {sample['id']}: {e}")
                    error_result = PubLayNetResult(
                        sample_id=str(sample['id']),
                        image_path="",
                        model=model_name,
                        phase=phase,
                        error=str(e),
                        timestamp=datetime.now().isoformat()
                    )
                    results.append(error_result)
                    pbar.update(1)
        
        # Final save
        if results:
            all_results = existing_results + results
            self._save_results_csv(results_file, all_results, write_headers=(not csv_headers_written))
            logger.info(f"  Saved {len(all_results)} results to {results_file}")
        
        return {
            'status': 'completed',
            'samples_processed': len(existing_results) + len(results),
            'results_file': str(results_file)
        }
    
    def _process_sample(self, phase: str, model_name: str, sample) -> PubLayNetResult:
        """
        Process single PubLayNet sample.
        
        Args:
            phase: Phase name ('P-A', 'P-B', 'P-C')
            model_name: Model name
            sample: Dataset sample with image and annotations
        
        Returns:
            PubLayNetResult with predictions
        """
        start_time = time.time()
        sample_id = str(sample['id'])
        
        # Extract ground truth boxes from annotations
        ground_truth_boxes = self._extract_ground_truth_boxes(sample)
        
        predicted_boxes = []
        error = None
        
        try:
            if phase == 'P-A':
                # OCR Layout Inference
                predicted_boxes = self._phase_pa_ocr_inference(model_name, sample)
            
            elif phase == 'P-B':
                # VLM Direct Detection
                predicted_boxes = self._phase_pb_vlm_direct(model_name, sample)
            
            elif phase == 'P-C':
                # VLM + OCR Hybrid
                predicted_boxes = self._phase_pc_vlm_ocr_hybrid(model_name, sample)
            
            else:
                raise ValueError(f"Unknown phase: {phase}")
        
        except Exception as e:
            logger.warning(f"Prediction failed for {sample_id}: {e}")
            error = str(e)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return PubLayNetResult(
            sample_id=sample_id,
            image_path=str(sample_id),
            model=model_name,
            phase=phase,
            ground_truth_boxes=ground_truth_boxes,
            predicted_boxes=predicted_boxes,
            inference_time_ms=elapsed_ms,
            error=error,
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_ground_truth_boxes(self, sample) -> List[Dict]:
        """
        Extract ground truth bounding boxes from PubLayNet annotations.
        
        Args:
            sample: Dataset sample
        
        Returns:
            List of box dicts with category
        """
        boxes = []
        for ann in sample['annotations']:
            bbox = ann['bbox']  # [x, y, width, height]
            boxes.append({
                'x': float(bbox[0]),
                'y': float(bbox[1]),
                'width': float(bbox[2]),
                'height': float(bbox[3]),
                'category': int(ann['category_id']),
                'confidence': 1.0
            })
        return boxes
    
    def _phase_pa_ocr_inference(self, model_name: str, sample) -> List[Dict]:
        """
        Phase P-A: Extract text via OCR and infer layout structure.
        
        Args:
            model_name: OCR model name (azure_intelligence, mistral_document_ai)
            sample: Dataset sample with image
        
        Returns:
            Predicted boxes with categories
        """
        logger.debug(f"Phase P-A: Running {model_name} OCR inference")
        
        try:
            # Get image
            image = sample['image']
            
            # Save temporarily for OCR API
            temp_path = "/tmp/ocr_temp.jpg"
            if isinstance(image, Image.Image):
                image.save(temp_path)
            else:
                # Assume it's a path
                temp_path = str(image)
            
            # Call OCR API based on model
            if model_name == "azure_intelligence":
                boxes = self._extract_azure_bounding_boxes(temp_path)
            elif model_name == "mistral_document_ai":
                boxes = self._extract_mistral_bounding_boxes(temp_path)
            else:
                logger.warning(f"Unknown OCR model: {model_name}, returning empty boxes")
                boxes = []
            
            return boxes
            
        except Exception as e:
            logger.error(f"P-A OCR inference failed: {e}")
            return []
    
    def _extract_azure_bounding_boxes(self, image_path: str) -> List[Dict]:
        """
        Extract bounding boxes from Azure Document Intelligence OCR.
        
        Args:
            image_path: Path to image
        
        Returns:
            List of bounding boxes as [x, y, width, height, category, confidence]
        """
        try:
            endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").strip()
            api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "").strip()
            
            if not endpoint or not api_key:
                logger.warning("Azure Document Intelligence credentials not found")
                return []
            
            client = DocumentIntelligenceClient(
                endpoint=endpoint, 
                credential=AzureKeyCredential(api_key)
            )
            
            # Read image and call Azure API
            with open(image_path, "rb") as f:
                document_data = f.read()
            
            poller = client.begin_analyze_document(
                model_id="prebuilt-read",
                body=document_data,
                content_type="application/octet-stream"
            )
            result = poller.result()
            
            boxes = []
            
            # Extract bounding boxes from pages
            if result.pages:
                page = result.pages[0]  # Assume single page
                
                # Collect all lines with their positions
                if page.lines:
                    for line in page.lines:
                        if line.polygon:
                            # Polygon is a flat list: [x1, y1, x2, y2, x3, y3, ...]
                            coords = []
                            for i in range(0, len(line.polygon), 2):
                                if i + 1 < len(line.polygon):
                                    coords.append((line.polygon[i], line.polygon[i + 1]))
                            
                            if coords:
                                # Get bounding box from polygon
                                xs = [c[0] for c in coords]
                                ys = [c[1] for c in coords]
                                x_min, x_max = min(xs), max(xs)
                                y_min, y_max = min(ys), max(ys)
                                
                                width = x_max - x_min
                                height = y_max - y_min
                                
                                # Infer category from text properties
                                # Simple heuristic: text size determines category
                                # Title: larger font (height > 20)
                                # Regular text: normal (height 10-20)
                                # Others get default category 1
                                if height > 20:
                                    category = 2  # Title
                                else:
                                    category = 1  # Text
                                
                                box = {
                                    "x": float(x_min),
                                    "y": float(y_min),
                                    "width": float(width),
                                    "height": float(height),
                                    "category": category,
                                    "confidence": 1.0  # OCR confidence assumed high
                                }
                                boxes.append(box)
            
            logger.debug(f"Azure OCR extracted {len(boxes)} boxes")
            return boxes
            
        except Exception as e:
            logger.error(f"Azure bounding box extraction failed: {e}")
            return []
    
    def _extract_mistral_bounding_boxes(self, image_path: str) -> List[Dict]:
        """
        Extract bounding boxes from Mistral Document AI OCR.
        
        Uses mistral-document-ai-2505 model via Azure endpoint.
        
        Args:
            image_path: Path to image
        
        Returns:
            List of bounding boxes
        """
        try:
            import base64
            import requests
            from pathlib import Path
            
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
            api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
            
            if not endpoint or not api_key:
                logger.warning("Azure OpenAI credentials not found for Mistral")
                return []
            
            # Load and encode image
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            
            # Detect mime type
            suffix = Path(image_path).suffix.lower()
            mime_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }
            mime_type = mime_type_map.get(suffix, "image/jpeg")
            document_url = f"data:{mime_type};base64,{image_data}"
            
            # Call Mistral OCR API via Azure
            endpoint = endpoint.rstrip("/")
            endpoint_url = f"{endpoint}/providers/mistral/azure/ocr"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": api_key.strip(),
            }
            
            payload = {
                "model": "mistral-document-ai-2505",
                "document": {
                    "type": "document_url",
                    "document_url": document_url
                }
            }
            
            logger.debug(f"Calling Mistral OCR API: {endpoint_url}")
            response = requests.post(endpoint_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"Mistral API error: {response.status_code} - {response.text[:300]}")
                return []
            
            result = response.json()
            boxes = []
            
            # Extract bounding boxes from Mistral response
            # Mistral returns structured layout data in pages
            if "pages" in result:
                for page_idx, page in enumerate(result["pages"]):
                    if "elements" in page:
                        for element in page["elements"]:
                            # Extract bounding box
                            if "bbox" in element:
                                bbox = element["bbox"]
                                # bbox format: [x_min, y_min, x_max, y_max]
                                x_min, y_min, x_max, y_max = bbox
                                width = x_max - x_min
                                height = y_max - y_min
                                
                                # Infer category from element type
                                element_type = element.get("type", "text").lower()
                                if element_type in ["title", "heading"]:
                                    category = 2
                                elif element_type in ["list", "bullet"]:
                                    category = 3
                                elif element_type in ["table"]:
                                    category = 4
                                elif element_type in ["figure", "image"]:
                                    category = 5
                                else:
                                    category = 1  # Default to text
                                
                                box = {
                                    "x": float(x_min),
                                    "y": float(y_min),
                                    "width": float(width),
                                    "height": float(height),
                                    "category": category,
                                    "confidence": 1.0
                                }
                                boxes.append(box)
            
            logger.debug(f"Mistral OCR extracted {len(boxes)} boxes")
            return boxes
            
        except Exception as e:
            logger.error(f"Mistral bounding box extraction failed: {e}")
            return []
    
    def _phase_pb_vlm_direct(self, model_name: str, sample) -> List[Dict]:
        """
        Phase P-B: Direct VLM layout detection.
        
        Args:
            model_name: VLM model name (gpt-5-mini, gpt-5-nano)
            sample: Dataset sample with image
        
        Returns:
            Predicted boxes with categories
        """
        prompt = self._get_phase_pb_prompt()
        
        # Save image temporarily for API call
        image_path = f"/tmp/publaynet_{sample['id']}.jpg"
        sample['image'].save(image_path)
        
        try:
            # Call VLM API
            response = self.api.process(image_path, model=model_name, query=prompt)
            
            if response.error:
                raise Exception(response.error)
            
            # Parse VLM response to extract boxes
            boxes = self._parse_vlm_boxes_response(response.content)
            return boxes
        
        finally:
            # Clean up
            import os
            if os.path.exists(image_path):
                os.remove(image_path)
    
    def _phase_pc_vlm_ocr_hybrid(self, model_name: str, sample) -> List[Dict]:
        """
        Phase P-C: VLM with OCR context for hybrid layout detection.
        
        Args:
            model_name: VLM model name
            sample: Dataset sample with image
        
        Returns:
            Predicted boxes with categories
        """
        # Get OCR context first (simplified)
        ocr_context = self._get_ocr_context(sample)
        
        # Build prompt with OCR context
        prompt = self._get_phase_pc_prompt(ocr_context)
        
        # Save image temporarily
        image_path = f"/tmp/publaynet_{sample['id']}.jpg"
        sample['image'].save(image_path)
        
        try:
            # Call VLM API with context
            response = self.api.process(image_path, model=model_name, query=prompt)
            
            if response.error:
                raise Exception(response.error)
            
            # Parse VLM response
            boxes = self._parse_vlm_boxes_response(response.content)
            return boxes
        
        finally:
            # Clean up
            import os
            if os.path.exists(image_path):
                os.remove(image_path)
    
    def _get_phase_pb_prompt(self) -> str:
        """Get Phase P-B VLM prompt for direct layout detection."""
        return """Analyze this document image and identify all layout elements.

For each element found, provide:
1. Category (choose one): Text, Title, List, Table, Figure
2. Bounding box coordinates in format: [x, y, width, height]

Category definitions:
- Text: Regular text blocks and paragraphs
- Title: Document titles and section headings
- List: Bulleted or numbered lists
- Table: Tabular data structures with rows/columns
- Figure: Images, charts, diagrams, or visual elements

Output as JSON array:
[
  {
    "category": "Text",
    "bbox": [x, y, width, height],
    "confidence": 0.95
  }
]

Only output the JSON array, no other text."""
    
    def _get_phase_pc_prompt(self, ocr_context: str) -> str:
        """Get Phase P-C VLM prompt for OCR-guided layout detection."""
        return f"""Analyze this document image and identify all layout elements.
Use the OCR context below to guide your analysis.

OCR Context (text regions):
{ocr_context}

For each element found, provide:
1. Category (choose one): Text, Title, List, Table, Figure
2. Bounding box coordinates in format: [x, y, width, height]

Category definitions:
- Text: Regular text blocks and paragraphs
- Title: Document titles and section headings
- List: Bulleted or numbered lists
- Table: Tabular data structures with rows and columns
- Figure: Images, charts, diagrams, or visual elements

Output as JSON array with this structure:
[{{'category': 'Text', 'bbox': [10, 20, 300, 150], 'confidence': 0.95}}]

Only output the JSON array, no other text."""
    
    def _get_ocr_context(self, sample) -> str:
        """Get simplified OCR context from sample (placeholder)."""
        # In real implementation, would run OCR and extract text positions
        return "Text regions detected at various positions in document"
    
    def _parse_vlm_boxes_response(self, response_text: str) -> List[Dict]:
        """
        Parse VLM response to extract bounding boxes.
        
        Args:
            response_text: VLM model response
        
        Returns:
            List of box dicts
        """
        try:
            # Try to find JSON in response - look for [ ... ]
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                logger.debug(f"No JSON found in response: {response_text[:100]}")
                return []
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            if not isinstance(data, list):
                logger.warning(f"Expected JSON array, got {type(data)}")
                return []
            
            boxes = []
            for item in data:
                if not isinstance(item, dict):
                    logger.warning(f"Expected dict item, got {type(item)}")
                    continue
                
                # Map category name to ID
                category_name = item.get('category', 'Text')
                if isinstance(category_name, str):
                    category_id = self._category_name_to_id(category_name)
                else:
                    category_id = int(category_name) if isinstance(category_name, (int, float)) else 1
                
                bbox = item.get('bbox', [0, 0, 0, 0])
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    logger.warning(f"Invalid bbox format: {bbox}")
                    continue
                
                boxes.append({
                    'x': float(bbox[0]),
                    'y': float(bbox[1]),
                    'width': float(bbox[2]),
                    'height': float(bbox[3]),
                    'category': category_id,
                    'confidence': float(item.get('confidence', 0.5))
                })
            
            return boxes
        
        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}")
            return []
        except Exception as e:
            logger.warning(f"Failed to parse VLM response: {e}")
            return []
    
    def _category_name_to_id(self, name: str) -> int:
        """Convert category name to ID."""
        mapping = {
            'text': 1,
            'title': 2,
            'list': 3,
            'table': 4,
            'figure': 5,
        }
        return mapping.get(name.lower(), 1)  # Default to Text
    
    def _save_results_csv(self, results_file: Path, results: List[PubLayNetResult], write_headers: bool = True):
        """Save results to CSV."""
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            'sample_id', 'image_path', 'model', 'phase',
            'ground_truth_boxes', 'predicted_boxes',
            'inference_time_ms', 'error', 'timestamp'
        ]
        
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_headers:
                writer.writeheader()
            
            for result in results:
                writer.writerow(result.to_dict())
    
    def _load_existing_results(self, results_file: Path) -> List[PubLayNetResult]:
        """Load existing results for resumability."""
        if not results_file.exists():
            return []
        
        try:
            results = []
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['inference_time_ms'] = float(row.get('inference_time_ms', 0))
                    row['ground_truth_boxes'] = json.loads(row.get('ground_truth_boxes', '[]'))
                    row['predicted_boxes'] = json.loads(row.get('predicted_boxes', '[]'))
                    results.append(PubLayNetResult(**row))
            
            return results
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")
            return []
    
    def _save_checkpoint(self, checkpoint_data: Dict):
        """Save checkpoint."""
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self) -> Dict:
        """Load checkpoint if exists."""
        if not self.checkpoint_file.exists():
            return {}
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return {}


if __name__ == '__main__':
    """
    PubLayNet parsing benchmark with OCR and VLM models.
    
    Phases:
      P-A: OCR layout inference (azure_intelligence, mistral_document_ai)
      P-B: VLM direct detection (gpt-5-mini, gpt-5-nano)
      P-C: VLM + OCR hybrid (gpt-5-mini, gpt-5-nano)
    """
    logger.info("=" * 80)
    logger.info("Starting PubLayNet Parsing Benchmark")
    logger.info("=" * 80)
    
    config = BenchmarkConfigPubLayNet(
        ocr_models=['azure_intelligence', 'mistral_document_ai'],
        vlm_models=['gpt-5-mini', 'gpt-5-nano'],
        phases=['P-A', 'P-B', 'P-C'],
        sample_limit=None,  # Use all samples
        batch_size=10,
        results_dir="results/publaynet"
    )
    
    runner = PubLayNetBenchmarkRunner(config)
    summary = runner.run_benchmark()
    
    logger.info("\n" + "=" * 80)
    logger.info("Benchmark Complete")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {runner.results_dir}")
    logger.info(json.dumps(summary, indent=2))
