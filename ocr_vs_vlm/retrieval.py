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
            from ..llms.embeddings import EmbeddingCalculator, EmbeddingConfig
        except ImportError:
            from llms.embeddings import EmbeddingCalculator, EmbeddingConfig

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


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (sparse) and dense (semantic) methods.

    Normalizes scores from both methods and combines them with a weighted average.
    """

    def __init__(
        self,
        bm25_weight: float = 0.5,
        embedding_model: str = "text-embedding-3-small",
        tokenizer: Optional[Callable[[str], List[str]]] = None
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_weight: Weight for BM25 scores (0-1). Dense weight = 1 - bm25_weight.
            embedding_model: OpenAI embedding model to use.
            tokenizer: Function to tokenize text for BM25.
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = 1.0 - bm25_weight
        self.bm25_retriever = BM25Retriever(tokenizer=tokenizer)
        self.dense_retriever = DenseRetriever(embedding_model=embedding_model)

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
        Retrieve top-K most relevant documents using hybrid method.

        Args:
            query: Search query (e.g., question text)
            documents: List of document texts (e.g., extracted page text)
            top_k: Number of top results to return

        Returns:
            List of (page_index, hybrid_score) tuples, sorted by score descending.
            Hybrid scores are normalized to [0, 1] range.

        Example:
            >>> retriever = HybridRetriever(bm25_weight=0.5)
            >>> docs = ["revenue was 100M", "expenses were 50M", "profit was 50M"]
            >>> results = retriever.retrieve("What was the revenue?", docs, top_k=1)
            >>> results
            [(0, 0.92)]
        """
        if not documents:
            return []

        # Get BM25 results (all documents)
        bm25_results = self.bm25_retriever.retrieve(query, documents, top_k=len(documents))
        bm25_scores_dict = {idx: score for idx, score in bm25_results}

        # Get dense results (all documents)
        dense_results = self.dense_retriever.retrieve(query, documents, top_k=len(documents))
        dense_scores_dict = {idx: score for idx, score in dense_results}

        # Normalize scores
        bm25_scores = [bm25_scores_dict[i] for i in range(len(documents))]
        dense_scores = [dense_scores_dict[i] for i in range(len(documents))]

        norm_bm25 = self._normalize_scores(bm25_scores)
        norm_dense = self._normalize_scores(dense_scores)

        # Combine scores
        hybrid_scores = []
        for i in range(len(documents)):
            hybrid_score = (self.bm25_weight * norm_bm25[i] +
                           self.dense_weight * norm_dense[i])
            hybrid_scores.append((i, float(hybrid_score)))

        # Sort by hybrid score descending
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-K results
        return hybrid_scores[:top_k]

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
