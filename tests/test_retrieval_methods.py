#!/usr/bin/env python3
"""
Test script for retrieval methods (BM25, Dense, Hybrid).

Validates that all three retrieval methods work correctly and
can rank documents based on query relevance.
"""

from ocr_vs_vlm.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


def test_bm25_retrieval():
    """Test BM25 sparse retrieval."""
    print("\n=== Testing BM25 Retrieval ===")

    retriever = BM25Retriever()

    # Sample documents (page texts)
    documents = [
        "The total revenue for Q4 2023 was 150 million dollars.",
        "Operating expenses increased by 10% this quarter.",
        "Net profit margins remained stable at 15%.",
        "Revenue growth was driven by increased sales in Asia.",
    ]

    # Query
    query = "What was the revenue in Q4?"
    ground_truth_idx = 0  # First document is the correct answer

    # Retrieve
    top_results, gt_rank = retriever.retrieve_with_rank(
        query=query,
        documents=documents,
        ground_truth_idx=ground_truth_idx,
        top_k=2
    )

    print(f"Query: {query}")
    print(f"Top results: {top_results}")
    print(f"Ground truth rank: {gt_rank}")

    # Validate - BM25 should rank ground truth in top 2
    assert gt_rank <= 2, f"Expected ground truth in top 2, got rank {gt_rank}"

    print(f"✓ BM25 retrieval passed (ground truth at rank {gt_rank})")
    return True


def test_dense_retrieval():
    """Test Dense semantic retrieval."""
    print("\n=== Testing Dense Retrieval ===")

    retriever = DenseRetriever()

    # Sample documents - semantic similarity test
    documents = [
        "The company's income reached 150M in the fourth quarter.",  # Semantic match
        "Expenses for operations went up 10%.",
        "Profit margins stayed consistent.",
        "Sales in Asia boosted financial performance.",
    ]

    # Query (different words, same meaning as doc 0)
    query = "What was the revenue in Q4?"
    ground_truth_idx = 0

    # Retrieve
    top_results, gt_rank = retriever.retrieve_with_rank(
        query=query,
        documents=documents,
        ground_truth_idx=ground_truth_idx,
        top_k=2
    )

    print(f"Query: {query}")
    print(f"Top results: {top_results}")
    print(f"Ground truth rank: {gt_rank}")

    # Validate
    assert gt_rank <= 2, f"Expected ground truth in top 2, got rank {gt_rank}"

    print(f"✓ Dense retrieval passed (ground truth at rank {gt_rank})")
    return True


def test_hybrid_retrieval():
    """Test Hybrid retrieval (BM25 + Dense)."""
    print("\n=== Testing Hybrid Retrieval ===")

    retriever = HybridRetriever(bm25_weight=0.5)

    # Sample documents
    documents = [
        "Q4 revenue was 150 million dollars.",
        "Operating costs increased.",
        "Profit margins stable.",
        "Asia sales growth.",
    ]

    query = "What was the revenue in Q4?"
    ground_truth_idx = 0

    # Retrieve
    top_results, gt_rank = retriever.retrieve_with_rank(
        query=query,
        documents=documents,
        ground_truth_idx=ground_truth_idx,
        top_k=2
    )

    print(f"Query: {query}")
    print(f"Top results: {top_results}")
    print(f"Ground truth rank: {gt_rank}")

    # Validate
    assert gt_rank <= 2, f"Expected ground truth in top 2, got rank {gt_rank}"

    print(f"✓ Hybrid retrieval passed (ground truth at rank {gt_rank})")
    return True


def test_retrieval_comparison():
    """Compare all three methods on same data."""
    print("\n=== Comparing Retrieval Methods ===")

    documents = [
        "The annual revenue reached 500 million USD.",
        "Quarterly expenses totaled 200 million.",
        "Company profits increased by 20%.",
        "Revenue from European markets declined.",
    ]

    query = "What was the total revenue?"
    ground_truth_idx = 0

    # BM25
    bm25_retriever = BM25Retriever()
    bm25_results, bm25_rank = bm25_retriever.retrieve_with_rank(
        query, documents, ground_truth_idx, top_k=1
    )

    # Dense
    dense_retriever = DenseRetriever()
    dense_results, dense_rank = dense_retriever.retrieve_with_rank(
        query, documents, ground_truth_idx, top_k=1
    )

    # Hybrid
    hybrid_retriever = HybridRetriever()
    hybrid_results, hybrid_rank = hybrid_retriever.retrieve_with_rank(
        query, documents, ground_truth_idx, top_k=1
    )

    print(f"\nQuery: {query}")
    print(f"BM25 rank: {bm25_rank}, score: {bm25_results[0][1]:.3f}")
    print(f"Dense rank: {dense_rank}, score: {dense_results[0][1]:.3f}")
    print(f"Hybrid rank: {hybrid_rank}, score: {hybrid_results[0][1]:.3f}")

    # All should rank ground truth highly
    assert bm25_rank <= 2, f"BM25 rank too low: {bm25_rank}"
    assert dense_rank <= 2, f"Dense rank too low: {dense_rank}"
    assert hybrid_rank <= 2, f"Hybrid rank too low: {hybrid_rank}"

    print("✓ All methods ranked ground truth in top 2")
    return True


if __name__ == "__main__":
    print("="*70)
    print("Retrieval Methods Validation")
    print("="*70)

    try:
        test_bm25_retrieval()
        test_dense_retrieval()
        test_hybrid_retrieval()
        test_retrieval_comparison()

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
