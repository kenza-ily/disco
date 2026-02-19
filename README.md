# DISCO - Document Intelligence Suite Comparison

Benchmark OCR and Vision-Language Models for document understanding tasks.

**Research Question:** To what extent do OCRs remain key parsing tools, and in which ways are VLMs better suited for specific tasks?

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/kenza-ily/disco.git
cd disco
uv sync
```

### 2. Configure Credentials

```bash
cp .env.example .env.local
# Edit .env.local with your API keys
```

**You only need credentials for the providers you want to use:**
- Azure OpenAI (GPT-5, Document Intelligence OCR)
- Anthropic Direct API (Claude models - recommended)
- AWS Bedrock (alternative for Claude models)
- Mistral API (Mistral OCR)

### 3. Authenticate with HuggingFace

Datasets are loaded automatically from HuggingFace Hub:

```bash
huggingface-cli login
```

### 4. Run Your First Benchmark

```bash
# Option 1: Use unified runner (recommended)
uv run python scripts/run_benchmark.py \
    --dataset publaynet \
    --models claude_sonnet \
    --phases P-B \
    --sample-limit 10

# Option 2: Call benchmark directly
uv run python -m benchmarks.dataset_specific.benchmark_publaynet \
    --models claude_sonnet \
    --phases P-B \
    --sample-limit 10
```

## Supported Models

### Vision-Language Models (VLMs)
- **Claude Sonnet 4** - High-accuracy vision LLM (Direct API or AWS Bedrock)
- **Claude Haiku 4** - Fast vision LLM (Direct API or AWS Bedrock)
- **GPT-5 mini/nano** - Azure OpenAI vision models
- **Qwen-VL** - Open source vision-language model

### OCR Models
- **Azure Document Intelligence** - Enterprise-grade OCR
- **Mistral OCR 3** - Advanced document parsing with markdown tables
- **Donut** - Open source document understanding
- **DeepSeek-OCR** - Multilingual OCR model

## Datasets

All datasets load automatically from HuggingFace (no local storage needed):

| Dataset | Task | Samples | HuggingFace URL |
|---------|------|---------|-----------------|
| IAM | Handwriting recognition | 500 | `kenza-ily/iam_disco` |
| DocVQA | Document question answering | 500 | `kenza-ily/docvqa_disco` |
| InfographicVQA | Infographic QA | 500 | `kenza-ily/infographicvqa_disco` |
| DUDE | Diverse documents QA | 500 | `kenza-ily/dude_disco` |
| ChartQA Pro | Chart question answering | 500 | `kenza-ily/chartqapro_disco` |
| PubLayNet | Document layout parsing | 500 | `kenza-ily/publaynet_disco` |
| VisRBench | Visual reasoning | 500 | `kenza-ily/visrbench_disco` |
| ICDAR | Multilingual OCR | 500 | `kenza-ily/icdar_disco` |

## Benchmark Pipeline

The evaluation follows a three-phase approach:

- **Phase P-A**: OCR baseline - pure OCR models extract text
- **Phase P-B**: VLM baseline - VLMs with generic prompts
- **Phase P-C**: VLM + context - VLMs with task-aware prompts

### Phase Naming Conventions

**QA Benchmarks (DocVQA, InfographicVQA, DUDE, ChartQA):**
- **QA1a**: OCR extraction → simple QA prompt
- **QA1b**: OCR extraction → detailed QA prompt
- **QA1c**: OCR extraction → chain-of-thought QA
- **QA2a**: Direct VLM with simple prompt
- **QA2b**: Direct VLM with detailed prompt
- **QA3a**: Hybrid (OCR + VLM reasoning)
- **QA4a**: Multi-page with retrieval (VisRBench only)

**Parsing Benchmarks (PubLayNet, RX-PAD, IAM, VOC2007, ICDAR):**
- **P-A**: Pure OCR baseline
- **P-B**: Direct VLM extraction
- **P-C**: Hybrid (OCR + VLM refinement)
- **1, 2, 3**: IAM uses integer phases (equivalent to P-A, P-B, P-C)

### Example: Run Full DocVQA Benchmark

```bash
# Run all phases with Claude Sonnet
uv run python -m benchmarks.benchmark_docvqa \
    --models claude_sonnet \
    --phases P-A P-B P-C \
    --sample-limit 50
```

### Run Multiple Models

```bash
uv run python -m benchmarks.benchmark_publaynet \
    --models azure_intelligence gpt5_mini claude_sonnet \
    --phases P-B \
    --sample-limit 100
```

## Results Analysis

Results are saved in `results/` with a structured pipeline:

```
results/
├── 0_info/          # Pipeline scripts and documentation
├── 1_raw/           # Raw experimental outputs (CSV per model)
├── 2_clean/         # Consolidated results (CSV per phase)
├── 3_notebooks/     # Analysis notebooks
└── 4_postprocessing/  # Aggregated statistics
```

**Run full analysis pipeline:**

```bash
cd results
python 0_info/run_pipeline.py --full
```

**Open analysis notebooks:**

```bash
cd results/3_notebooks
jupyter notebook 00_master_evaluation.ipynb
```

## Project Structure

- **benchmarks/** - Benchmark scripts for all datasets
  - `benchmark.py` - Base framework
  - `dataset_specific/` - Dataset implementations
- **datasets/** - HuggingFace dataset loaders
- **metrics/** - Evaluation metrics (CER, WER, ANLS)
- **models/** - Unified API for OCR + VLM models
- **prompts/** - Task-specific prompts
- **results/** - Benchmark results and analysis
  - `0_info/` - Pipeline documentation
  - `1_raw/` - Raw benchmark outputs
  - `2_clean/` - Consolidated results
  - `3_embeddings/` - Pre-computed embeddings
- **scripts/** - Utility scripts
  - `run_benchmark.py` - Unified benchmark runner
- **utils/** - Shared utilities
- **archive/** - Archived code and old experiments

## Evaluation Metrics

- **CER** (Character Error Rate) - Character-level edit distance
- **WER** (Word Error Rate) - Word-level edit distance
- **ANLS** (Average Normalized Levenshtein Similarity) - Standard VQA metric
- **EM** (Exact Match) - Binary exact match
- **Cosine Similarity** - Semantic similarity using embeddings
- **Substring Match** - Fuzzy matching for VQA

## Development

```bash
# Run tests
make test

# Format code
make format

# Lint code
make lint

# Type checking
make typecheck

# Run all quality checks
make all
```

## Credential Setup Examples

### Option 1: Azure (Free Credits Available)

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key-here
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=your-key-here
```

### Option 2: Anthropic Direct API (Recommended for Claude)

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Option 3: AWS Bedrock (Alternative for Claude)

```env
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
```

### Option 4: Mistral OCR

```env
MISTRAL_API_KEY=your-key-here
```

## Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@misc{disco2026,
  title={DISCO: Document Intelligence Suite Comparison},
  author={Your Name},
  year={2026},
  url={https://github.com/kenza-ily/disco}
}
```

## License

MIT License - See LICENSE file for details.
