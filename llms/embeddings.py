"""
Embedding calculation pipeline using Azure OpenAI.

Provides utilities for calculating and managing embeddings for text data.
Supports batch processing with checkpointing and caching.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from datetime import datetime
import time

from .llm_settings import get_settings
from openai import AzureOpenAI, RateLimitError, APIError

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model: str = "text-embedding-3-large"
    batch_size: int = 100
    max_retries: int = 3
    timeout_seconds: int = 60
    checkpoint_interval: int = 100
    cache_dir: Optional[Path] = None


@dataclass
class EmbeddingResult:
    """Result of embedding a single text."""
    text: str
    embedding: List[float]
    model: str
    timestamp: str
    input_tokens: int = 0


@dataclass
class EmbeddingBatch:
    """Batch of embeddings with metadata."""
    texts: List[str]
    embeddings: List[List[float]]
    model: str
    total_tokens: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'texts': self.texts,
            'embeddings': self.embeddings,
            'model': self.model,
            'total_tokens': self.total_tokens,
            'timestamp': self.timestamp,
            'config': self.config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingBatch':
        """Create from dictionary."""
        return cls(
            texts=data['texts'],
            embeddings=data['embeddings'],
            model=data['model'],
            total_tokens=data.get('total_tokens', 0),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            config=data.get('config', {}),
        )


class EmbeddingCalculator:
    """
    Calculate embeddings using Azure OpenAI API.

    Features:
    - Batch processing for efficiency
    - Automatic retry logic with exponential backoff
    - Checkpoint saving for recovery from failures
    - Caching to avoid redundant API calls
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding calculator.
        
        Args:
            config: Embedding configuration (uses defaults if None)
        """
        self.config = config or EmbeddingConfig()
        self.client = self._get_client()
        self.cache: Dict[str, List[float]] = {}
        self._load_cache()
        logger.info(f"EmbeddingCalculator initialized with model: {self.config.model}")
    
    def _get_client(self) -> AzureOpenAI:
        """Get Azure OpenAI client."""
        settings = get_settings()
        return AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
    
    def _load_cache(self):
        """Load cached embeddings if cache directory is configured."""
        if self.config.cache_dir:
            cache_dir = Path(self.config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f".embedding_cache_{self.config.model}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        self.cache = json.load(f)
                    logger.info(f"Loaded {len(self.cache)} cached embeddings")
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        if self.config.cache_dir:
            cache_dir = Path(self.config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f".embedding_cache_{self.config.model}.json"
            
            try:
                with open(cache_file, 'w') as f:
                    json.dump(self.cache, f)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    def embed_text(self, text: str, safe: bool = False, max_tokens: int = 8000, encoding_name: str = "cl100k_base") -> EmbeddingResult:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with embedding vector
        """
        # If safe=True, use chunked embedding utility
        if safe:
            safe_embed = self.get_safe_embed_fn(max_tokens=max_tokens, encoding_name=encoding_name)
            arr = safe_embed(text)
            return EmbeddingResult(
                text=text,
                embedding=arr.tolist(),
                model=self.config.model,
                timestamp=datetime.now().isoformat(),
            )
        # Check cache first
        text_hash = str(hash(text))
        if text_hash in self.cache:
            logger.debug(f"Cache hit for text")
            return EmbeddingResult(
                text=text,
                embedding=self.cache[text_hash],
                model=self.config.model,
                timestamp=datetime.now().isoformat(),
            )
        # Call API with retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.config.model,
                )
                embedding = response.data[0].embedding
                self.cache[text_hash] = embedding
                return EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model=self.config.model,
                    timestamp=datetime.now().isoformat(),
                    input_tokens=response.usage.prompt_tokens,
                )
            except RateLimitError as e:
                wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                logger.warning(f"Rate limit hit, retrying in {wait_time:.1f}s (attempt {attempt + 1})")
                time.sleep(wait_time)
            except APIError as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = (2 ** attempt)
                    logger.warning(f"API error: {e}, retrying in {wait_time}s (attempt {attempt + 1})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to embed text after {self.config.max_retries} attempts: {e}")
                    raise
        raise RuntimeError(f"Failed to embed text after {self.config.max_retries} attempts")
    
    def embed_batch(self, texts: List[str]) -> EmbeddingBatch:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            EmbeddingBatch with all embeddings
        """
        embeddings = []
        total_tokens = 0
        
        for text in texts:
            result = self.embed_text(text)
            embeddings.append(result.embedding)
            total_tokens += result.input_tokens
        
        # Save cache periodically
        self._save_cache()
        
        return EmbeddingBatch(
            texts=texts,
            embeddings=embeddings,
            model=self.config.model,
            total_tokens=total_tokens,
            config=asdict(self.config),
        )

    def get_safe_embed_fn(self, max_tokens: int = 8000, encoding_name: str = "cl100k_base"):
        """
        Returns a function that safely embeds any text (with chunking/aggregation if needed).
        Usage:
            safe_embed = self.get_safe_embed_fn(max_tokens=8000)
            embedding = safe_embed(long_text)
        """
        from utils.embedding_utils import get_chunked_embed_fn
        return get_chunked_embed_fn(self._embed_text_array, max_tokens=max_tokens, encoding_name=encoding_name)

    def _embed_text_array(self, text: str):
        """Return embedding as np.ndarray for chunked embedding utility."""
        result = self.embed_text(text)
        return np.array(result.embedding)


class EmbeddingPipeline:
    """
    Pipeline for embedding requests and saving results.
    
    Manages:
    - Loading requests from various sources
    - Batch processing with progress tracking
    - Checkpoint saving for fault tolerance
    - Results saving with configurable formats
    """
    
    def __init__(self, output_dir: Path, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding pipeline.
        
        Args:
            output_dir: Directory to save embeddings
            config: Embedding configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or EmbeddingConfig()
        self.calculator = EmbeddingCalculator(self.config)
        self.checkpoint_file = self.output_dir / "embedding_checkpoint.json"
        self.results_file = self.output_dir / "embeddings_results.json"
        self.embeddings_file = self.output_dir / "embeddings.npy"
        self.metadata_file = self.output_dir / "embeddings_metadata.json"
        
        logger.info(f"EmbeddingPipeline initialized")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Model: {self.config.model}")
    
    def embed_texts(self, texts: List[str], resume: bool = True) -> Dict[str, Any]:
        """
        Embed a list of texts with checkpointing.
        
        Args:
            texts: List of texts to embed
            resume: If True, resume from last checkpoint
            
        Returns:
            Dictionary with results and statistics
        """
        # Load checkpoint if resuming
        processed_texts = []
        processed_embeddings = []
        start_idx = 0
        
        if resume and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                processed_texts = checkpoint.get('texts', [])
                processed_embeddings = checkpoint.get('embeddings', [])
                start_idx = len(processed_texts)
                logger.info(f"Resumed from checkpoint: {start_idx}/{len(texts)} texts processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting fresh")
        
        # Process remaining texts
        remaining_texts = texts[start_idx:]
        all_embeddings = processed_embeddings.copy()
        all_texts = processed_texts.copy()
        
        if remaining_texts:
            logger.info(f"Processing {len(remaining_texts)} texts...")
            
            for i in range(0, len(remaining_texts), self.config.batch_size):
                batch_texts = remaining_texts[i:i + self.config.batch_size]
                
                try:
                    batch_result = self.calculator.embed_batch(batch_texts)
                    all_embeddings.extend(batch_result.embeddings)
                    all_texts.extend(batch_texts)
                    
                    # Save checkpoint
                    self._save_checkpoint(all_texts, all_embeddings)
                    
                    logger.info(f"Processed {len(all_texts)}/{len(texts)} texts")
                
                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")
                    raise
        
        # Save final results
        results = self._save_results(all_texts, all_embeddings)
        
        return results
    
    def _save_checkpoint(self, texts: List[str], embeddings: List[List[float]]):
        """Save checkpoint for recovery."""
        checkpoint = {
            'texts': texts,
            'embeddings': embeddings,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
    
    def _save_results(self, texts: List[str], embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Save results in multiple formats.
        
        Saves:
        - embeddings.npy: NumPy array of embeddings
        - embeddings_metadata.json: Texts and metadata
        - embeddings_results.json: Full results
        """
        # Save embeddings as NumPy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        np.save(self.embeddings_file, embeddings_array)
        
        # Save metadata
        metadata = {
            'texts': texts,
            'model': self.config.model,
            'embedding_dim': embeddings_array.shape[1] if embeddings_array.size > 0 else 0,
            'num_texts': len(texts),
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save full results
        results = {
            'config': asdict(self.config),
            'summary': {
                'num_texts': len(texts),
                'embedding_dimension': embeddings_array.shape[1] if embeddings_array.size > 0 else 0,
                'model': self.config.model,
                'timestamp': datetime.now().isoformat(),
            },
            'metadata': metadata,
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results:")
        logger.info(f"  Embeddings: {self.embeddings_file} ({embeddings_array.shape})")
        logger.info(f"  Metadata: {self.metadata_file}")
        logger.info(f"  Results: {self.results_file}")
        
        return results


def create_embeddings_for_dataset(
    dataset_name: str,
    texts: List[str],
    dataset_dir: Path,
    config: Optional[EmbeddingConfig] = None,
) -> Path:
    """
    Create embeddings for all texts in a dataset and save to the dataset directory.
    
    Saves embeddings with "_embeddings" suffix in the same directory as the dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "DocVQA_mini")
        texts: List of texts to embed
        dataset_dir: Root directory of the dataset
        config: Embedding configuration
        
    Returns:
        Path to the embeddings directory
    """
    # Create embeddings directory with _embeddings suffix
    embeddings_dir = dataset_dir / f"{dataset_name}_embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating embeddings for {dataset_name}")
    logger.info(f"  Texts to embed: {len(texts)}")
    logger.info(f"  Output directory: {embeddings_dir}")
    
    pipeline = EmbeddingPipeline(embeddings_dir, config)
    results = pipeline.embed_texts(texts)
    
    logger.info(f"Embedding pipeline completed")
    logger.info(f"  Results: {results['summary']}")
    
    return embeddings_dir


def load_embeddings(embeddings_dir: Path) -> Dict[str, Any]:
    """
    Load embeddings and metadata from a results directory.
    
    Args:
        embeddings_dir: Directory containing embeddings
        
    Returns:
        Dictionary with embeddings, texts, and metadata
    """
    embeddings_file = embeddings_dir / "embeddings.npy"
    metadata_file = embeddings_dir / "embeddings_metadata.json"
    
    if not embeddings_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(f"Embeddings not found in {embeddings_dir}")
    
    # Load embeddings array
    embeddings = np.load(embeddings_file)
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return {
        'embeddings': embeddings,
        'texts': metadata['texts'],
        'model': metadata['model'],
        'embedding_dim': metadata['embedding_dim'],
        'timestamp': metadata['timestamp'],
    }
