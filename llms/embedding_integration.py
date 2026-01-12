"""
Integration module for using embeddings in benchmark pipelines.

Provides utilities for:
- Loading pre-computed embeddings
- Computing similarity scores between texts
- Using embeddings for semantic search and clustering
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
from scipy.spatial.distance import cosine
from scipy import stats

from llms.embeddings import load_embeddings

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    Store and query embeddings for efficient similarity computations.
    """
    
    def __init__(self, embeddings_dir: Path):
        """
        Initialize embedding store.
        
        Args:
            embeddings_dir: Directory containing embeddings
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.data = load_embeddings(self.embeddings_dir)
        self.embeddings = self.data['embeddings']
        self.texts = self.data['texts']
        self.model = self.data['model']
        
        logger.info(f"Loaded {len(self.texts)} embeddings from {embeddings_dir}")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Dimension: {self.data['embedding_dim']}")
    
    def find_similar(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar texts to a query using cosine similarity.
        
        Args:
            query_text: Text to query
            top_k: Number of results to return
            
        Returns:
            List of (text, similarity_score) tuples
        """
        # Find exact match first
        try:
            query_idx = self.texts.index(query_text)
            query_embedding = self.embeddings[query_idx]
        except ValueError:
            logger.warning(f"Query text not found in store: {query_text[:50]}...")
            return []
        
        # Compute similarities
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            if i == query_idx:
                continue
            
            similarity = 1 - cosine(query_embedding, embedding)
            similarities.append((i, similarity))
        
        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = [(self.texts[i], score) for i, score in similarities[:top_k]]
        
        return results
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """
        Compute pairwise similarity matrix.
        
        Returns:
            Matrix of cosine similarities
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(self.embeddings)
    
    def cluster_by_similarity(self, threshold: float = 0.7) -> Dict[int, List[Tuple[str, int]]]:
        """
        Cluster texts by similarity.
        
        Args:
            threshold: Similarity threshold for clustering
            
        Returns:
            Dictionary mapping cluster_id to list of (text, original_idx) tuples
        """
        similarity_matrix = self.compute_similarity_matrix()
        
        clusters = {}
        cluster_id = 0
        visited = set()
        
        for i in range(len(self.texts)):
            if i in visited:
                continue
            
            cluster = [(self.texts[i], i)]
            visited.add(i)
            
            # Find all similar texts
            for j in range(i + 1, len(self.texts)):
                if j in visited:
                    continue
                
                if similarity_matrix[i, j] >= threshold:
                    cluster.append((self.texts[j], j))
                    visited.add(j)
            
            if len(cluster) > 1:  # Only add non-trivial clusters
                clusters[cluster_id] = cluster
                cluster_id += 1
        
        logger.info(f"Found {len(clusters)} clusters with similarity >= {threshold}")
        return clusters


class EmbeddingAnalyzer:
    """
    Analyze embeddings and their properties.
    """
    
    def __init__(self, store: EmbeddingStore):
        """Initialize analyzer."""
        self.store = store
    
    def compute_statistics(self) -> Dict[str, float]:
        """Compute statistics on embeddings."""
        embeddings = self.store.embeddings
        
        # Compute pairwise distances
        similarity_matrix = self.store.compute_similarity_matrix()
        similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        stats_dict = {
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'median_similarity': float(np.median(similarities)),
            'num_embeddings': len(embeddings),
            'embedding_dim': embeddings.shape[1],
        }
        
        logger.info(f"Embedding statistics:")
        for key, value in stats_dict.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return stats_dict
    
    def find_duplicate_questions(self, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Find highly similar (near-duplicate) questions.
        
        Args:
            threshold: Similarity threshold
            
        Returns:
            List of (text1, text2, similarity) tuples
        """
        similarity_matrix = self.store.compute_similarity_matrix()
        
        duplicates = []
        for i in range(len(self.store.texts)):
            for j in range(i + 1, len(self.store.texts)):
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    duplicates.append((
                        self.store.texts[i],
                        self.store.texts[j],
                        float(sim)
                    ))
        
        logger.info(f"Found {len(duplicates)} near-duplicate question pairs")
        return sorted(duplicates, key=lambda x: x[2], reverse=True)


def load_and_analyze_embeddings(embeddings_dir: Path) -> Dict:
    """
    Load embeddings and perform analysis.
    
    Args:
        embeddings_dir: Directory containing embeddings
        
    Returns:
        Dictionary with loaded data and analysis results
    """
    store = EmbeddingStore(embeddings_dir)
    analyzer = EmbeddingAnalyzer(store)
    
    stats = analyzer.compute_statistics()
    duplicates = analyzer.find_duplicate_questions()
    
    return {
        'store': store,
        'analyzer': analyzer,
        'statistics': stats,
        'duplicates': duplicates,
    }
