#!/usr/bin/env python3
"""
Test script for QA4 (OCR + BM25 retrieval) phase.

This script validates the QA4 implementation by running a single sample
through the benchmark pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr_vs_vlm.benchmarks.benchmark_visrbench import VisRBenchConfig, VisRBenchBenchmark

def main():
    """Run QA4 validation test with 5 samples."""
    print("=" * 70)
    print("QA4 Validation Test: OCR + BM25 Retrieval")
    print("=" * 70)

    config = VisRBenchConfig(
        phases=["QA4a"],  # Test just QA4a for speed
        sample_limit=5,   # 5 samples as requested
        qa_per_doc=1,
        retrieval_mode=True,
        retrieval_top_k=1,
        use_huggingface=False
    )

    print(f"\nConfiguration:")
    print(f"  Phases: {config.phases}")
    print(f"  Sample limit: {config.sample_limit}")
    print(f"  Retrieval mode: {config.retrieval_mode}")
    print(f"  Retrieval top-K: {config.retrieval_top_k}")

    try:
        benchmark = VisRBenchBenchmark(config)
        results = benchmark.run()

        print("\n" + "=" * 70)
        print("Validation Results:")
        print("=" * 70)

        if results and 'results' in results:
            result_list = results['results']
            if result_list:
                print(f"\nProcessed {len(result_list)} samples:")
                print("-" * 70)

                retrieval_correct = 0
                for i, r in enumerate(result_list, 1):
                    print(f"\n[Sample {i}/{len(result_list)}]")
                    print(f"  Sample ID: {r.sample_id}")
                    print(f"  Question: {r.question[:60]}...")
                    print(f"  Ground truth page: {r.page_index}")
                    print(f"  Retrieved page: {r.retrieved_page_index}")
                    print(f"  Is correct: {r.is_correct_page} (rank: {r.retrieval_rank})")
                    print(f"  BM25 score: {r.bm25_score:.3f}" if r.bm25_score else "  BM25 score: None")
                    print(f"  ANLS: {r.anls_score:.3f}")

                    if r.is_correct_page:
                        retrieval_correct += 1

                print("\n" + "=" * 70)
                print(f"Summary:")
                print(f"  Retrieval accuracy: {retrieval_correct}/{len(result_list)} ({100*retrieval_correct/len(result_list):.1f}%)")
                print(f"  Average ANLS: {sum(r.anls_score for r in result_list)/len(result_list):.3f}")
                print(f"\n✅ QA4 validation test completed successfully!")
            else:
                print("\n⚠️  No results returned")
        else:
            print("\n⚠️  Results dictionary empty")

    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
