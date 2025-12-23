#!/usr/bin/env python3
"""
Unified data loader for small-scale datasets.
Usage:
    from load_small_datasets import SmallDatasetLoader

    loader = SmallDatasetLoader('small_scale_datasets')

    # Load specific dataset
    docvqa_data = loader.load_dataset('DocVQA')

    # Load all datasets
    all_data = loader.load_all_datasets()

    # Iterate through a dataset
    for sample in loader.iterate_dataset('DocVQA'):
        print(sample)
"""

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List


class SmallDatasetLoader:
    """Unified loader for small-scale datasets with consistent interface."""

    def __init__(self, base_dir: str = "small_scale_datasets"):
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise ValueError(f"Dataset directory '{base_dir}' does not exist")

        self.datasets = self._discover_datasets()

    def _discover_datasets(self) -> List[str]:
        datasets = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "annotations" / "data.json").exists():
                datasets.append(item.name)
        return sorted(datasets)

    def list_datasets(self) -> List[str]:
        return self.datasets.copy()

    def load_dataset(self, dataset_name: str, load_images: bool = False) -> List[Dict[str, Any]]:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {self.datasets}")

        dataset_dir = self.base_dir / dataset_name
        annotations_file = dataset_dir / "annotations" / "data.json"

        with open(annotations_file, 'r') as f:
            data = json.load(f)

        if load_images:
            data = self._enrich_with_files(data, dataset_dir)

        return data

    def _enrich_with_files(self, data: List[Dict], dataset_dir: Path) -> List[Dict]:
        enriched_data = []

        for sample in data:
            enriched_sample = sample.copy()

            if 'image' in sample and sample['image']:
                image_path = dataset_dir / "images" / sample['image']
                if image_path.exists():
                    enriched_sample['image_data'] = self._load_file_as_base64(image_path)

            if 'document' in sample and sample['document']:
                doc_path = dataset_dir / "documents" / sample['document']
                if doc_path.exists():
                    enriched_sample['document_data'] = self._load_file_as_base64(doc_path)

            if 'ocr_file' in sample and sample['ocr_file']:
                ocr_path = dataset_dir / "ocr" / sample['ocr_file']
                if ocr_path.exists():
                    with open(ocr_path, 'r') as f:
                        enriched_sample['ocr_text'] = f.read()

            if 'pdf' in sample and sample['pdf']:
                pdf_path = dataset_dir / "pdfs" / sample['pdf']
                if pdf_path.exists():
                    enriched_sample['pdf_data'] = self._load_file_as_base64(pdf_path)

            if 'images' in sample and sample['images']:
                enriched_sample['images_data'] = []
                for img in sample['images']:
                    img_path = dataset_dir / "images" / img
                    if img_path.exists():
                        enriched_sample['images_data'].append(
                            self._load_file_as_base64(img_path)
                        )

            enriched_data.append(enriched_sample)

        return enriched_data

    def _load_file_as_base64(self, file_path: Path) -> str:
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def iterate_dataset(self, dataset_name: str, load_images: bool = False) -> Iterator[Dict[str, Any]]:
        data = self.load_dataset(dataset_name, load_images=load_images)
        for sample in data:
            yield sample

    def load_all_datasets(self, load_images: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        all_data = {}
        for dataset_name in self.datasets:
            try:
                all_data[dataset_name] = self.load_dataset(dataset_name, load_images=load_images)
            except Exception as e:
                print(f"Warning: Failed to load {dataset_name}: {e}")
        return all_data

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        dataset_dir = self.base_dir / dataset_name
        data = self.load_dataset(dataset_name, load_images=False)

        num_images = len(list((dataset_dir / "images").glob("*"))) if (dataset_dir / "images").exists() else 0
        num_docs = len(list((dataset_dir / "documents").glob("*"))) if (dataset_dir / "documents").exists() else 0
        num_ocr = len(list((dataset_dir / "ocr").glob("*"))) if (dataset_dir / "ocr").exists() else 0
        num_pdfs = len(list((dataset_dir / "pdfs").glob("*"))) if (dataset_dir / "pdfs").exists() else 0

        return {
            'name': dataset_name,
            'num_samples': len(data),
            'num_images': num_images,
            'num_documents': num_docs,
            'num_ocr_files': num_ocr,
            'num_pdfs': num_pdfs,
            'fields': list(data[0].keys()) if data else [],
            'path': str(dataset_dir)
        }

    def get_all_dataset_info(self) -> List[Dict[str, Any]]:
        return [self.get_dataset_info(name) for name in self.datasets]

    def get_sample(self, dataset_name: str, index: int = 0, load_images: bool = False) -> Dict[str, Any]:
        data = self.load_dataset(dataset_name, load_images=load_images)
        if index >= len(data):
            raise IndexError(f"Index {index} out of range for dataset '{dataset_name}' (size: {len(data)})")
        return data[index]


def print_dataset_summary(loader: SmallDatasetLoader):
    print("\n" + "="*60)
    print("SMALL-SCALE DATASETS SUMMARY")
    print("="*60 + "\n")

    all_info = loader.get_all_dataset_info()

    for info in all_info:
        print(f"Dataset: {info['name']}")
        print(f"  Samples: {info['num_samples']}")
        if info['num_images'] > 0:
            print(f"  Images: {info['num_images']}")
        if info['num_documents'] > 0:
            print(f"  Documents: {info['num_documents']}")
        if info['num_ocr_files'] > 0:
            print(f"  OCR files: {info['num_ocr_files']}")
        if info['num_pdfs'] > 0:
            print(f"  PDFs: {info['num_pdfs']}")
        print(f"  Fields: {', '.join(info['fields'])}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Load small-scale datasets')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Name of specific dataset to load (default: show summary of all)')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='Index of sample to display (default: 0)')
    parser.add_argument('--base_dir', type=str, default='small_scale_datasets',
                        help='Base directory containing datasets (default: small_scale_datasets)')

    args = parser.parse_args()

    try:
        loader = SmallDatasetLoader(args.base_dir)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.dataset:
        try:
            print(f"\nLoading dataset: {args.dataset}")
            data = loader.load_dataset(args.dataset)
            print(f"Loaded {len(data)} samples\n")

            sample = loader.get_sample(args.dataset, args.sample_index)
            print(f"Sample {args.sample_index}:")
            print(json.dumps(sample, indent=2))

        except (ValueError, IndexError) as e:
            print(f"Error: {e}")
    else:
        print_dataset_summary(loader)


if __name__ == "__main__":
    main()
