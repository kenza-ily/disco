import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent.parent / "utils"))
from embedding_utils import get_chunked_embed_fn
"""
Embedding cache utilities for notebooks.

Provides functions to load and save embeddings for analysis notebooks,
avoiding repeated API calls for the same texts.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Default embeddings directory
DEFAULT_EMBEDDINGS_DIR = Path(__file__).parent.parent / "results" / "3_embeddings"

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


def get_embeddings_dir() -> Path:
    """Get the default embeddings directory."""
    return DEFAULT_EMBEDDINGS_DIR


def load_embeddings_for_dataset(
    dataset_name: str,
    embeddings_base_dir: Optional[Path] = None
) -> Dict[str, Dict]:
    """
    Load all cached embeddings for a dataset.
    
    Args:
        dataset_name: Name of dataset (e.g., "IAM_mini")
        embeddings_base_dir: Base directory for embeddings (default: 3_embeddings/)
        
    Returns:
        Dict with structure:
        {
            'Pa': {'ground_truths': {...}, 'predictions': {...}},
            'Pb': {'ground_truths': {...}, 'predictions': {...}},
            ...
        }
    """
    if embeddings_base_dir is None:
        embeddings_base_dir = DEFAULT_EMBEDDINGS_DIR
    
    embeddings_base_dir = Path(embeddings_base_dir)
    dataset_dir = embeddings_base_dir / dataset_name
    
    cache = {}
    
    if not dataset_dir.exists():
        logger.info(f"No embeddings directory found for {dataset_name}, will generate on-the-fly")
        return cache
    
    # Find all embedding files for this dataset
    embedding_files = list(dataset_dir.glob("*_embeddings_*.json"))
    
    if not embedding_files:
        logger.info(f"No embedding files found in {dataset_dir}, will generate on-the-fly")
        return cache
    
    # Group by phase and load most recent for each
    phase_files: Dict[str, List[Path]] = {}
    for f in embedding_files:
        # Extract phase from filename: Pa_embeddings_model_timestamp.json
        phase = f.name.split("_embeddings_")[0]
        if phase not in phase_files:
            phase_files[phase] = []
        phase_files[phase].append(f)
    
    for phase, files in phase_files.items():
        # Load most recent file for this phase
        most_recent = max(files, key=lambda f: f.stat().st_mtime)
        try:
            with open(most_recent, 'r') as f:
                cache[phase] = json.load(f)
            logger.info(f"Loaded embeddings for {phase} from {most_recent.name}")
        except Exception as e:
            logger.warning(f"Failed to load embeddings from {most_recent}: {e}")
            cache[phase] = {'ground_truths': {}, 'predictions': {}}
    
    return cache


def save_embeddings_for_phase(
    dataset_name: str,
    phase: str,
    embeddings_dict: Dict,
    embeddings_base_dir: Optional[Path] = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL
) -> Path:
    """
    Save embeddings for a phase to disk.
    
    Args:
        dataset_name: Name of dataset (e.g., "IAM_mini")
        phase: Phase name (e.g., "Pa", "Pb", "Pc")
        embeddings_dict: Dict with 'ground_truths' and 'predictions' keys
        embeddings_base_dir: Base directory for embeddings (default: 3_embeddings/)
        model_name: Embedding model name for filename
        
    Returns:
        Path to saved file
    """
    if embeddings_base_dir is None:
        embeddings_base_dir = DEFAULT_EMBEDDINGS_DIR
    
    embeddings_base_dir = Path(embeddings_base_dir)
    dataset_dir = embeddings_base_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize model name for filename
    model_safe = model_name.replace("/", "_").replace(" ", "_")
    filename = f"{phase}_embeddings_{model_safe}_{timestamp}.json"
    
    filepath = dataset_dir / filename
    
    # Add metadata
    embeddings_dict['_metadata'] = {
        'dataset': dataset_name,
        'phase': phase,
        'model': model_name,
        'timestamp': timestamp,
        'num_ground_truths': len(embeddings_dict.get('ground_truths', {})),
        'num_predictions': sum(
            len(models) for models in embeddings_dict.get('predictions', {}).values()
        ),
    }
    
    with open(filepath, 'w') as f:
        json.dump(embeddings_dict, f)
    
    logger.info(f"Saved embeddings to {filepath}")
    return filepath


def get_cached_embedding(
    cache: Dict,
    phase: str,
    text_type: str,
    key: str,
    model: Optional[str] = None
) -> Optional[List[float]]:
    """
    Get a cached embedding if available.
    
    Args:
        cache: Loaded embeddings cache
        phase: Phase name (e.g., "Pa")
        text_type: Either "ground_truths" or "predictions"
        key: For ground_truths: the ground truth string
             For predictions: the sample_id
        model: Model name (required for predictions)
        
    Returns:
        Embedding vector or None if not cached
    """
    if phase not in cache:
        return None
    
    phase_cache = cache[phase]
    
    if text_type == "ground_truths":
        return phase_cache.get('ground_truths', {}).get(key)
    elif text_type == "predictions":
        if model is None:
            return None
        sample_cache = phase_cache.get('predictions', {}).get(key, {})
        return sample_cache.get(model)
    
    return None


def store_embedding_in_cache(
    cache: Dict,
    phase: str,
    text_type: str,
    key: str,
    embedding: List[float],
    model: Optional[str] = None
) -> None:
    """
    Store an embedding in the cache.
    
    Args:
        cache: Embeddings cache to update
        phase: Phase name (e.g., "Pa")
        text_type: Either "ground_truths" or "predictions"
        key: For ground_truths: the ground truth string
             For predictions: the sample_id
        embedding: Embedding vector to store
        model: Model name (required for predictions)
    """
    if phase not in cache:
        cache[phase] = {'ground_truths': {}, 'predictions': {}}
    
    if text_type == "ground_truths":
        if 'ground_truths' not in cache[phase]:
            cache[phase]['ground_truths'] = {}
        
        # Note: Duplicate embeddings can occur when different samples have identical text
        # (e.g., template-based medical reports). This is expected behavior - the cache
        # key is unique per sample, but identical text produces identical embeddings.
        
        cache[phase]['ground_truths'][key] = embedding
    elif text_type == "predictions":
        if 'predictions' not in cache[phase]:
            cache[phase]['predictions'] = {}
        if key not in cache[phase]['predictions']:
            cache[phase]['predictions'][key] = {}
        if model:
            cache[phase]['predictions'][key][model] = embedding


class EmbeddingCacheManager:
    """
    Manager for embedding cache with progress tracking.
    
    Usage in notebooks:
        manager = EmbeddingCacheManager("IAM_mini")
        
        # Get embedding (from cache or compute)
        emb = manager.get_embedding(phase, "ground_truth", text)
        emb = manager.get_embedding(phase, "prediction", text, sample_id=sid, model=model)
        
        # Save at the end
        manager.save_new_embeddings()
    """
    
    def __init__(
        self,
        dataset_name: str,
        embeddings_base_dir: Optional[Path] = None,
        embedding_calculator=None
    ):
        """
        Initialize cache manager.
        
        Args:
            dataset_name: Name of dataset
            embeddings_base_dir: Base directory for embeddings
            embedding_calculator: Optional pre-initialized EmbeddingCalculator
        """
        self.dataset_name = dataset_name
        self.embeddings_base_dir = embeddings_base_dir or DEFAULT_EMBEDDINGS_DIR
        
        # Load existing cache
        self.cache = load_embeddings_for_dataset(dataset_name, self.embeddings_base_dir)
        
        # Track which phases have new embeddings
        self.modified_phases: set = set()
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Lazy-load embedding calculator
        self._calculator = embedding_calculator
    
    @property
    def calculator(self):
        """Lazy-load embedding calculator."""
        if self._calculator is None:
            from models.embeddings import EmbeddingCalculator
            self._calculator = EmbeddingCalculator()
        return self._calculator
    
    def get_ground_truth_embedding(
        self,
        phase: str,
        ground_truth: str,
        sample_id: Optional[str] = None
    ) -> List[float]:
        """
        Get embedding for a ground truth text.
        
        Args:
            phase: Phase name
            ground_truth: Ground truth text
            sample_id: Optional sample identifier for unique caching per sample.
                       If provided, cache key is sample_id (recommended for section-level).
                       If None, cache key is ground_truth text (legacy behavior).
            
        Returns:
            Embedding vector
        """
        # Determine cache key: use sample_id if provided, otherwise use text content
        # Using sample_id prevents cache collisions when different samples have identical text
        cache_key = sample_id if sample_id else ground_truth
        
        # Check cache
        cached = get_cached_embedding(self.cache, phase, "ground_truths", cache_key)
        if cached is not None:
            self.cache_hits += 1
            logger.debug(f"GT cache hit for phase {phase}, key {cache_key[:50] if len(cache_key) > 50 else cache_key}")
            return cached
        
        # Compute
        self.cache_misses += 1
        logger.debug(f"GT cache miss for phase {phase}, computing embedding for key {cache_key[:50] if len(cache_key) > 50 else cache_key}")
        # Use safe embedding for long texts
        result = self.calculator.embed_text(ground_truth, safe=True, max_tokens=8000)
        embedding = result.embedding if result else []
        
        # Store in cache
        if embedding:
            store_embedding_in_cache(self.cache, phase, "ground_truths", cache_key, embedding)
            self.modified_phases.add(phase)
        
        return embedding
    
    def get_prediction_embedding(
        self,
        phase: str,
        prediction: str,
        sample_id: str,
        model: str
    ) -> List[float]:
        """
        Get embedding for a prediction.
        
        Args:
            phase: Phase name
            prediction: Prediction text
            sample_id: Sample identifier
            model: Model name
            
        Returns:
            Embedding vector
        """
        # Check cache
        cached = get_cached_embedding(self.cache, phase, "predictions", sample_id, model)
        if cached is not None:
            self.cache_hits += 1
            return cached
        
        # Compute
        self.cache_misses += 1
        # Use safe embedding for long texts
        result = self.calculator.embed_text(prediction, safe=True, max_tokens=8000)
        embedding = result.embedding if result else []
        
        # Store in cache
        if embedding:
            store_embedding_in_cache(self.cache, phase, "predictions", sample_id, embedding, model)
            self.modified_phases.add(phase)
        
        return embedding
    
    def compute_cosine_similarity(
        self,
        phase: str,
        ground_truth: str,
        prediction: str,
        sample_id: str,
        model: str,
        gt_sample_id: Optional[str] = None
    ) -> float:
        """
        Compute cosine similarity with caching.
        
        Args:
            phase: Phase name
            ground_truth: Ground truth text
            prediction: Prediction text
            sample_id: Sample identifier (used for prediction caching)
            model: Model name
            gt_sample_id: Optional sample ID for ground truth caching.
                          If provided, ground truth embedding is cached by this ID
                          instead of by text content. Use for section-level analysis
                          where different samples may have identical section text.
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not prediction or not ground_truth:
            return 0.0
        
        gt_emb = self.get_ground_truth_embedding(phase, ground_truth, sample_id=gt_sample_id)
        pred_emb = self.get_prediction_embedding(phase, prediction, sample_id, model)
        
        if not gt_emb or not pred_emb:
            logger.warning(f"Missing embeddings for sample {sample_id}: gt_emb={len(gt_emb) if gt_emb else 0}, pred_emb={len(pred_emb) if pred_emb else 0}")
            return 0.0
        
        # Check if embeddings are identical (debugging)
        if gt_emb == pred_emb:
            logger.warning(f"Identical embeddings for sample {sample_id}, model {model}")
            return 1.0  # Perfect match
        
        try:
            from scipy.spatial.distance import cosine
            similarity = 1 - cosine(pred_emb, gt_emb)
            
            # Log suspicious values for debugging
            if similarity < 0 or similarity > 1:
                logger.warning(f"Invalid cosine similarity {similarity} for sample {sample_id}")
                return max(0.0, min(1.0, similarity))  # Clamp to valid range
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing cosine similarity for sample {sample_id}: {e}")
            return 0.0
    
    def save_new_embeddings(self) -> List[Path]:
        """
        Save any newly computed embeddings to disk.
        
        Returns:
            List of paths to saved files
        """
        saved_files = []
        
        for phase in self.modified_phases:
            if phase in self.cache:
                filepath = save_embeddings_for_phase(
                    self.dataset_name,
                    phase,
                    self.cache[phase],
                    self.embeddings_base_dir
                )
                saved_files.append(filepath)
        
        return saved_files
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total,
            'hit_rate': hit_rate,
            'modified_phases': list(self.modified_phases),
            'loaded_phases': list(self.cache.keys()),
        }
    
    def print_statistics(self) -> None:
        """Print cache statistics."""
        stats = self.get_statistics()
        print(f"\n📊 Embedding Cache Statistics:")
        print(f"   Cache hits: {stats['cache_hits']}")
        print(f"   Cache misses: {stats['cache_misses']}")
        print(f"   Hit rate: {stats['hit_rate']:.1%}")
        if stats['modified_phases']:
            print(f"   Modified phases: {', '.join(stats['modified_phases'])}")
