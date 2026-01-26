#!/usr/bin/env python3
"""
Create a representative mini version of VisR-Bench dataset.

Strategy:
1. Allocate 500 samples proportionally by content type:
   - Figure: ~40 docs
   - Table: ~67 docs
   - Text: ~99 docs
   - Multilingual: ~294 docs

2. For text documents: Stratify by length (quartiles)
   - Short (≤Q1): 25% of text docs
   - Medium (Q1-Q2): 25% of text docs
   - Long (Q2-Q3): 25% of text docs
   - Very long (>Q3): 25% of text docs

3. For multilingual: Preserve language distribution
   - Include top 8-10 languages
   - Maintain proportional representation

4. Preserve all QA pairs from selected documents
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

def load_dataset(filepath):
    """Load JSON dataset."""
    with open(filepath) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def create_visr_bench_mini(source_dir, output_dir, max_samples=500, seed=42):
    """Create mini dataset with stratified sampling."""
    
    random.seed(seed)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load all datasets
    qa_files = {
        "figure": "figure_QA.json",
        "table": "table_QA.json",
        "text": "text_QA.json",
        "multilingual": "multilingual_QA.json"
    }
    
    # Calculate allocations
    allocations = {
        "figure": int(max_samples * 125 / 1558),      # ~40
        "table": int(max_samples * 210 / 1558),       # ~67
        "text": int(max_samples * 310 / 1558),        # ~99
    }
    allocations["multilingual"] = max_samples - sum(allocations.values())
    
    print("=" * 80)
    print("CREATING VISR-BENCH MINI DATASET")
    print("=" * 80)
    print(f"\nTarget allocation (total: {max_samples}):")
    for qa_type, count in allocations.items():
        print(f"  {qa_type:15s}: {count:3d} documents")
    
    mini_datasets = {}
    metadata = {
        "dataset": "VisR-Bench Mini",
        "version": "1.0",
        "total_samples": max_samples,
        "source_dataset": "VisR-Bench (full)",
        "creation_strategy": "Stratified sampling preserving diversity",
        "subsets": {}
    }
    
    # 1. FIGURE DATASET
    print(f"\n1. Processing FIGURE dataset...")
    figure_path = os.path.join(source_dir, qa_files["figure"])
    figure_data = load_dataset(figure_path)
    
    # Random sampling
    sampled_figures = random.sample(figure_data, min(allocations["figure"], len(figure_data)))
    mini_datasets["figure"] = sampled_figures
    
    fig_qa_count = sum(len(d.get('qa_list', [])) for d in sampled_figures if isinstance(d, dict))
    print(f"   ✓ Selected {len(sampled_figures)} documents ({fig_qa_count} QA pairs)")
    
    metadata["subsets"]["figure"] = {
        "documents": len(sampled_figures),
        "qa_pairs": fig_qa_count,
        "allocation": f"{allocations['figure']}/1558 (8%)"
    }
    
    # 2. TABLE DATASET
    print(f"\n2. Processing TABLE dataset...")
    table_path = os.path.join(source_dir, qa_files["table"])
    table_data = load_dataset(table_path)
    
    # Random sampling
    sampled_tables = random.sample(table_data, min(allocations["table"], len(table_data)))
    mini_datasets["table"] = sampled_tables
    
    table_qa_count = sum(len(d.get('qa_list', [])) for d in sampled_tables if isinstance(d, dict))
    print(f"   ✓ Selected {len(sampled_tables)} documents ({table_qa_count} QA pairs)")
    
    metadata["subsets"]["table"] = {
        "documents": len(sampled_tables),
        "qa_pairs": table_qa_count,
        "allocation": f"{allocations['table']}/1558 (13%)"
    }
    
    # 3. TEXT DATASET (Stratified by document length)
    print(f"\n3. Processing TEXT dataset (stratified by length)...")
    text_path = os.path.join(source_dir, qa_files["text"])
    text_data = load_dataset(text_path)
    
    # Get page counts for all documents
    doc_pages = [(doc, len(doc.get('all_page_images', []))) for doc in text_data]
    doc_pages.sort(key=lambda x: x[1])
    
    # Calculate quartile boundaries
    q1_idx = len(doc_pages) // 4
    q2_idx = len(doc_pages) // 2
    q3_idx = 3 * len(doc_pages) // 4
    
    q1_val = doc_pages[q1_idx][1] if q1_idx < len(doc_pages) else doc_pages[-1][1]
    q2_val = doc_pages[q2_idx][1] if q2_idx < len(doc_pages) else doc_pages[-1][1]
    q3_val = doc_pages[q3_idx][1] if q3_idx < len(doc_pages) else doc_pages[-1][1]
    
    # Stratify documents
    text_target = allocations["text"]
    strata = {
        "short": [],
        "medium": [],
        "long": [],
        "very_long": []
    }
    
    for doc, pages in doc_pages:
        if pages <= q1_val:
            strata["short"].append(doc)
        elif pages <= q2_val:
            strata["medium"].append(doc)
        elif pages <= q3_val:
            strata["long"].append(doc)
        else:
            strata["very_long"].append(doc)
    
    # Allocate proportionally
    sampled_text = []
    for stratum, docs in strata.items():
        target_count = int(text_target * len(docs) / len(text_data))
        sampled = random.sample(docs, min(target_count, len(docs)))
        sampled_text.extend(sampled)
        print(f"   - {stratum:12s}: {len(sampled):2d} from {len(docs):3d} documents")
    
    mini_datasets["text"] = sampled_text
    
    text_qa_count = sum(len(d.get('qa_list', [])) for d in sampled_text if isinstance(d, dict))
    print(f"   ✓ Selected {len(sampled_text)} documents ({text_qa_count} QA pairs)")
    
    # Analyze stratification
    pages_mini = [len(d.get('all_page_images', [])) for d in sampled_text]
    
    metadata["subsets"]["text"] = {
        "documents": len(sampled_text),
        "qa_pairs": text_qa_count,
        "allocation": f"{allocations['text']}/1558 (20%)",
        "length_stats": {
            "min_pages": min(pages_mini),
            "max_pages": max(pages_mini),
            "mean_pages": sum(pages_mini) / len(pages_mini) if pages_mini else 0,
            "stratification": {
                "short": f"≤{q1_val}p",
                "medium": f"{q1_val}-{q2_val}p",
                "long": f"{q2_val}-{q3_val}p",
                "very_long": f">{q3_val}p"
            }
        }
    }
    
    # 4. MULTILINGUAL DATASET (Preserve language distribution)
    print(f"\n4. Processing MULTILINGUAL dataset (language-stratified)...")
    multi_path = os.path.join(source_dir, qa_files["multilingual"])
    multi_data = load_dataset(multi_path)
    
    # Group by language
    lang_groups = defaultdict(list)
    for doc in multi_data:
        if 'qa_list' in doc and len(doc['qa_list']) > 0:
            lang = doc['qa_list'][0].get('detected_language', 'unknown')
            lang_groups[lang].append(doc)
    
    # Calculate language-based allocation
    multi_target = allocations["multilingual"]
    sampled_multi = []
    lang_stats = {}
    
    sorted_langs = sorted(lang_groups.items(), key=lambda x: -len(x[1]))
    
    for lang, docs in sorted_langs[:10]:  # Top 10 languages
        target_count = int(multi_target * len(docs) / len(multi_data))
        sampled = random.sample(docs, min(target_count, len(docs)))
        sampled_multi.extend(sampled)
        lang_stats[lang] = {
            "sampled": len(sampled),
            "total": len(docs),
            "pct": f"{len(sampled)/target_count*100:.1f}%"
        }
        print(f"   - {lang:8s}: {len(sampled):3d} documents")
    
    # If not enough samples, add more from other languages
    if len(sampled_multi) < multi_target:
        remaining = multi_target - len(sampled_multi)
        other_docs = [d for lang, docs in sorted_langs[10:] for d in docs]
        sampled_multi.extend(random.sample(other_docs, min(remaining, len(other_docs))))
    
    mini_datasets["multilingual"] = sampled_multi[:multi_target]
    
    multi_qa_count = sum(len(d.get('qa_list', [])) for d in mini_datasets["multilingual"] if isinstance(d, dict))
    print(f"   ✓ Selected {len(mini_datasets['multilingual'])} documents ({multi_qa_count} QA pairs)")
    
    metadata["subsets"]["multilingual"] = {
        "documents": len(mini_datasets["multilingual"]),
        "qa_pairs": multi_qa_count,
        "allocation": f"{allocations['multilingual']}/1558 (59%)",
        "languages": lang_stats
    }
    
    # 5. SAVE MINI DATASETS
    print(f"\n5. Saving mini datasets...")
    
    for qa_type, docs in mini_datasets.items():
        output_file = os.path.join(output_dir, f"{qa_type}_QA_mini.json")
        with open(output_file, 'w') as f:
            json.dump(docs, f, indent=2)
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"   ✓ {output_file}: {len(docs):3d} docs ({size_mb:.2f} MB)")
    
    # Save metadata
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ {metadata_file}: {os.path.getsize(metadata_file) / 1024:.1f} KB")
    
    # Summary
    total_docs = sum(len(v) for v in mini_datasets.values())
    total_qa = sum(metadata["subsets"][k]["qa_pairs"] for k in mini_datasets.keys())
    
    print("\n" + "=" * 80)
    print("MINI DATASET CREATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFinal Summary:")
    print(f"  Total documents: {total_docs}")
    print(f"  Total QA pairs: {total_qa}")
    print(f"  Output directory: {output_dir}")
    print(f"\nBreakdown:")
    print(f"  Figure:      {len(mini_datasets['figure']):3d} docs ({metadata['subsets']['figure']['qa_pairs']:5d} QAs)")
    print(f"  Table:       {len(mini_datasets['table']):3d} docs ({metadata['subsets']['table']['qa_pairs']:5d} QAs)")
    print(f"  Text:        {len(mini_datasets['text']):3d} docs ({metadata['subsets']['text']['qa_pairs']:5d} QAs)")
    print(f"  Multilingual:{len(mini_datasets['multilingual']):3d} docs ({metadata['subsets']['multilingual']['qa_pairs']:5d} QAs)")
    
    return mini_datasets, metadata

if __name__ == "__main__":
    source_dir = "datasets/task2_QA/VisR-Bench/QA"
    output_dir = "ocr_vs_vlm/datasets/datasets_subsets/visr_bench_mini"
    
    mini_data, meta = create_visr_bench_mini(source_dir, output_dir, max_samples=500)
