#!/usr/bin/env python3
"""
Retrieval Methods for Multi-Page Documents

Provides multiple retrieval strategies to find relevant pages from multi-page
documents given a query (e.g., a question in a QA task).

Available Methods:
- BM25Retriever: Sparse keyword-based retrieval (BM25Okapi)
- DenseRetriever: Dense semantic retrieval using embeddings
- HybridRetriever: Combines BM25 and dense retrieval

Usage:
    # BM25 (sparse)
    retriever = BM25Retriever()
    results, gt_rank = retriever.retrieve_with_rank(
        query="What is the total revenue?",
        documents=list_of_page_texts,
        ground_truth_idx=2,
        top_k=1
    )

    # Dense (semantic)
    retriever = DenseRetriever()
    results, gt_rank = retriever.retrieve_with_rank(...)

    # Hybrid (combined)
    retriever = HybridRetriever()
    results, gt_rank = retriever.retrieve_with_rank(...)
"""

from typing import List, Tuple, Callable, Optional
import numpy as np
from rank_bm25 import BM25Okapi


def default_tokenizer(text: str) -> List[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


class BM25Retriever:
    """
    BM25-based retriever for finding relevant pages in multi-page documents.

    Uses BM25Okapi algorithm with configurable tokenization.
    """

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        """
        Initialize BM25 retriever.

        Args:
            tokenizer: Function to tokenize text. Defaults to whitespace splitting.
        """
        self.tokenizer = tokenizer or default_tokenizer

    def retrieve(
        self,
        query: str,
        documents: List[str],
        top_k: int = 1
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-K most relevant documents using BM25.

        Args:
            query: Search query (e.g., question text)
            documents: List of document texts (e.g., extracted page text)
            top_k: Number of top results to return

        Returns:
            List of (page_index, bm25_score) tuples, sorted by score descending.
            page_index is 0-indexed position in the documents list.

        Example:
            >>> retriever = BM25Retriever()
            >>> docs = ["revenue was 100M", "expenses were 50M", "profit was 50M"]
            >>> results = retriever.retrieve("What was the revenue?", docs, top_k=1)
            >>> results
            [(0, 1.234)]
        """
        if not documents:
            return []

        # Tokenize documents and query
        tokenized_docs = [self.tokenizer(doc) for doc in documents]
        tokenized_query = self.tokenizer(query)

        # Create BM25 index
        bm25 = BM25Okapi(tokenized_docs)

        # Get BM25 scores for all documents
        scores = bm25.get_scores(tokenized_query)

        # Create (index, score) tuples and sort by score descending
        results = [(idx, float(score)) for idx, score in enumerate(scores)]
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-K results
        return results[:top_k]

    def retrieve_with_rank(
        self,
        query: str,
        documents: List[str],
        ground_truth_idx: int,
        top_k: int = 1
    ) -> Tuple[List[Tuple[int, float]], int]:
        """
        Retrieve top-K documents and compute rank of ground truth page.

        Args:
            query: Search query
            documents: List of document texts
            ground_truth_idx: Index of ground truth evidence page
            top_k: Number of top results to return

        Returns:
            Tuple of:
                - List of (page_index, bm25_score) for top-K results
                - Rank of ground truth page (1-indexed, 1=best)

        Example:
            >>> retriever = BM25Retriever()
            >>> docs = ["revenue was 100M", "expenses were 50M", "profit was 50M"]
            >>> top_results, gt_rank = retriever.retrieve_with_rank(
            ...     "What was the profit?", docs, ground_truth_idx=2, top_k=1
            ... )
            >>> gt_rank
            1  # Ground truth is top-ranked
        """
        # Get all results sorted by score
        all_results = self.retrieve(query, documents, top_k=len(documents))

        # Find rank of ground truth page (1-indexed)
        gt_rank = None
        for rank, (idx, score) in enumerate(all_results, start=1):
            if idx == ground_truth_idx:
                gt_rank = rank
                break

        # Return top-K and ground truth rank
        return all_results[:top_k], gt_rank if gt_rank else len(documents) + 1


class DenseRetriever:
    """
    Dense retrieval using embeddings and cosine similarity.

    Uses OpenAI embeddings (text-embedding-3-small) to convert query and documents
    into semantic vectors, then ranks by cosine similarity.
    """

    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize dense retriever.

        Args:
            embedding_model: OpenAI embedding model to use.
        """
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        documents: List[str],
        top_k: int = 1
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-K most relevant documents using dense embeddings.

        Args:
            query: Search query (e.g., question text)
            documents: List of document texts (e.g., extracted page text)
            top_k: Number of top results to return

        Returns:
            List of (page_index, cosine_score) tuples, sorted by score descending.
            Cosine scores range from -1 to 1, typically 0.3-0.9 for relevant docs.

        Example:
            >>> retriever = DenseRetriever()
            >>> docs = ["revenue was 100M", "expenses were 50M", "profit was 50M"]
            >>> results = retriever.retrieve("What was the revenue?", docs, top_k=1)
            >>> results
            [(0, 0.87)]
        """
        if not documents:
            return []

        # Import here to avoid circular dependency and allow standalone usage
        try:
            from models.embeddings import EmbeddingCalculator, EmbeddingConfig
        except ImportError:
            from models.embeddings import EmbeddingCalculator, EmbeddingConfig

        # Get embeddings (batch API call for efficiency)
        config = EmbeddingConfig(model=self.embedding_model)
        calculator = EmbeddingCalculator(config=config)
        all_texts = [query] + documents
        embedding_batch = calculator.embed_batch(all_texts)

        # Extract vectors
        query_emb = np.array(embedding_batch.embeddings[0])
        doc_embs = [np.array(emb) for emb in embedding_batch.embeddings[1:]]

        # Compute cosine similarities
        # cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
        # For normalized vectors: cosine_sim(a, b) = dot(a, b)
        query_norm = np.linalg.norm(query_emb)
        doc_norms = np.linalg.norm(doc_embs, axis=1)

        scores = []
        for i, doc_emb in enumerate(doc_embs):
            dot_product = np.dot(query_emb, doc_emb)
            cosine_score = dot_product / (query_norm * doc_norms[i])
            scores.append(float(cosine_score))

        # Create (index, score) tuples and sort descending
        results = [(idx, score) for idx, score in enumerate(scores)]
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-K results
        return results[:top_k]

    def retrieve_with_rank(
        self,
        query: str,
        documents: List[str],
        ground_truth_idx: int,
        top_k: int = 1
    ) -> Tuple[List[Tuple[int, float]], int]:
        """
        Retrieve top-K documents and compute rank of ground truth page.

        Args:
            query: Search query
            documents: List of document texts
            ground_truth_idx: Index of ground truth evidence page
            top_k: Number of top results to return

        Returns:
            Tuple of:
                - List of (page_index, cosine_score) for top-K results
                - Rank of ground truth page (1-indexed, 1=best)

        Example:
            >>> retriever = DenseRetriever()
            >>> docs = ["revenue was 100M", "expenses were 50M", "profit was 50M"]
            >>> top_results, gt_rank = retriever.retrieve_with_rank(
            ...     "What was the profit?", docs, ground_truth_idx=2, top_k=1
            ... )
            >>> gt_rank
            1  # Ground truth is top-ranked
        """
        # Get all results sorted by score
        all_results = self.retrieve(query, documents, top_k=len(documents))

        # Find rank of ground truth page (1-indexed)
        gt_rank = None
        for rank, (idx, score) in enumerate(all_results, start=1):
            if idx == ground_truth_idx:
                gt_rank = rank
                break

        # Return top-K and ground truth rank
        return all_results[:top_k], gt_rank if gt_rank else len(documents) + 1


class BGEM3Retriever:
    """
    Dense semantic retrieval using BGE-M3 embeddings.

    Aligns with VisR-Bench standard: https://github.com/puar-playground/VisR-Bench
    Model: BAAI/bge-m3
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None, batch_size: int = 16):
        """
        Initialize BGE-M3 retriever.

        Args:
            model_name: HuggingFace model ID (default: BAAI/bge-m3)
            device: Torch device (cuda/cpu, auto-detects if None)
            batch_size: Batch size for encoding (default: 16)
        """
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings using CLS token + L2 normalization.

        Returns:
            Normalized embeddings array (num_texts, embedding_dim)
        """
        import torch

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token (first token) as sentence embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                # L2 normalization
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def retrieve(
        self,
        query: str,
        documents: List[str],
        top_k: int = 1
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-K most relevant documents using BGE-M3 embeddings.

        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results

        Returns:
            List of (page_index, cosine_score) tuples, sorted descending
            Scores range: -1 to 1 (typically 0.3-0.9 for relevant docs)
        """
        if not documents:
            return []

        # Encode query and documents
        query_emb = self.encode_texts([query])[0]
        doc_embs = self.encode_texts(documents)

        # Compute cosine similarities (embeddings are already normalized)
        # cosine_sim = dot(query, doc) for normalized vectors
        scores = np.dot(doc_embs, query_emb)

        # Create (index, score) tuples and sort descending
        results = [(idx, float(score)) for idx, score in enumerate(scores)]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def retrieve_with_rank(
        self,
        query: str,
        documents: List[str],
        ground_truth_idx: int,
        top_k: int = 1
    ) -> Tuple[List[Tuple[int, float]], int]:
        """
        Retrieve documents and return rank of ground truth page.

        Returns:
            (top_k_results, ground_truth_rank)
            ground_truth_rank is 1-indexed (1 = best)
        """
        # Get all scores
        all_results = self.retrieve(query, documents, top_k=len(documents))

        # Find rank of ground truth (1-indexed)
        gt_rank = None
        for rank, (idx, _) in enumerate(all_results, start=1):
            if idx == ground_truth_idx:
                gt_rank = rank
                break

        return all_results[:top_k], gt_rank if gt_rank else len(documents) + 1


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (sparse) and BGE-M3 (dense).

    Fuses normalized scores with configurable weighting.
    Aligns with VisR-Bench repository standards.
    """

    def __init__(
        self,
        bm25_weight: float = 0.5,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        bge_model_name: str = "BAAI/bge-m3",
        bge_device: str = None,
        bge_batch_size: int = 16
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_weight: Weight for BM25 scores (default: 0.5)
                        BGE-M3 weight = 1 - bm25_weight
            tokenizer: BM25 tokenizer function
            bge_model_name: BGE-M3 model name
            bge_device: Device for BGE-M3 model
            bge_batch_size: Batch size for BGE-M3 encoding
        """
        self.bm25_weight = bm25_weight
        self.bge_weight = 1.0 - bm25_weight

        # Initialize both retrievers
        self.bm25_retriever = BM25Retriever(tokenizer=tokenizer)
        self.bge_retriever = BGEM3Retriever(
            model_name=bge_model_name,
            device=bge_device,
            batch_size=bge_batch_size
        )

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Min-max normalize scores to [0, 1] range."""
        if not scores or len(scores) == 1:
            return [1.0] * len(scores)

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def retrieve(
        self,
        query: str,
        documents: List[str],
        top_k: int = 1
    ) -> List[Tuple[int, float]]:
        """
        Retrieve using hybrid BM25 + BGE-M3 fusion.

        Returns:
            List of (page_index, hybrid_score) tuples
        """
        if not documents:
            return []

        # Get all scores from both methods
        bm25_results = self.bm25_retriever.retrieve(query, documents, top_k=len(documents))
        bge_results = self.bge_retriever.retrieve(query, documents, top_k=len(documents))

        # Create score dictionaries
        bm25_scores = {idx: score for idx, score in bm25_results}
        bge_scores = {idx: score for idx, score in bge_results}

        # Normalize scores
        bm25_normalized = self._normalize_scores(list(bm25_scores.values()))
        bge_normalized = self._normalize_scores(list(bge_scores.values()))

        # Compute hybrid scores
        hybrid_scores = {}
        for idx in range(len(documents)):
            bm25_norm = bm25_normalized[idx]
            bge_norm = bge_normalized[idx]
            hybrid_scores[idx] = (
                self.bm25_weight * bm25_norm +
                self.bge_weight * bge_norm
            )

        # Sort by hybrid score descending
        results = [(idx, score) for idx, score in hybrid_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def retrieve_with_rank(
        self,
        query: str,
        documents: List[str],
        ground_truth_idx: int,
        top_k: int = 1
    ) -> Tuple[List[Tuple[int, float]], int]:
        """
        Retrieve top-K documents and compute rank of ground truth page.

        Args:
            query: Search query
            documents: List of document texts
            ground_truth_idx: Index of ground truth evidence page
            top_k: Number of top results to return

        Returns:
            Tuple of:
                - List of (page_index, hybrid_score) for top-K results
                - Rank of ground truth page (1-indexed, 1=best)

        Example:
            >>> retriever = HybridRetriever()
            >>> docs = ["revenue was 100M", "expenses were 50M", "profit was 50M"]
            >>> top_results, gt_rank = retriever.retrieve_with_rank(
            ...     "What was the profit?", docs, ground_truth_idx=2, top_k=1
            ... )
            >>> gt_rank
            1  # Ground truth is top-ranked
        """
        # Get all results sorted by score
        all_results = self.retrieve(query, documents, top_k=len(documents))

        # Find rank of ground truth page (1-indexed)
        gt_rank = None
        for rank, (idx, score) in enumerate(all_results, start=1):
            if idx == ground_truth_idx:
                gt_rank = rank
                break

        # Return top-K and ground truth rank
        return all_results[:top_k], gt_rank if gt_rank else len(documents) + 1
