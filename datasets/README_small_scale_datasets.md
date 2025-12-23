# Small-Scale Datasets

Extract and load small-scale versions of document understanding datasets with consistent formatting.

## Datasets

- **CheckboxQA**: Checkbox-based question answering on form documents
- **DocFinQA**: Financial document question answering with context
- **DocVQA**: Visual question answering on document images
- **DUDE**: Document understanding dataset with PDFs and OCR
- **InfographicVQA**: Visual question answering on infographics with OCR
- **MMDocIR**: Multi-modal document information retrieval with images and PDFs
- **VisR-Bench-QA**: Visual reasoning QA with complete multi-page documents

## Usage


### Load Datasets

```bash
# Show summary of all datasets
python3 load_small_datasets.py

# Load specific dataset
python3 load_small_datasets.py --dataset DocVQA

# View specific sample
python3 load_small_datasets.py --dataset DocVQA --sample_index 0
```

### Upload data

aws sso login --profile eu-dev
aws s3 ls s3://research-playground-datasets/
aws s3 sync small_scale_datasets s3://research-playground-datasets/small_scale_datasets/
aws s3 sync s3://research-playground-datasets/small_scale_datasets/ small_scale_datasets

### Create Small-Scale Datasets

```bash
# Extract 10 samples from each dataset (default)
python3 create_small_scale_datasets.py

# Extract custom number of samples
python3 create_small_scale_datasets.py --num_samples 20
```

### Python API

```python
from load_small_datasets import SmallDatasetLoader

# Initialize loader
loader = SmallDatasetLoader('small_scale_datasets')

# List all datasets
datasets = loader.list_datasets()

# Load a dataset
data = loader.load_dataset('DocVQA')

# Get a specific sample
sample = loader.get_sample('DocVQA', index=0)

# Load all datasets
all_data = loader.load_all_datasets()
```
