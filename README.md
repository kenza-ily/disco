# DISCO: Document Intelligence Suite for COmparative Evaluation

Benchmark OCR and Vision-Language Models for document understanding tasks.

**Research Question:** When should practitioners use OCR pipelines versus end-to-end VLMs for document intelligence tasks?

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
    --dataset docvqa \
    --models claude_sonnet \
    --phases P-B \
    --sample-limit 10

# Option 2: Call benchmark directly
uv run python -m benchmarks.dataset_specific.benchmark_docvqa \
    --models claude_sonnet \
    --phases P-B \
    --sample-limit 10
```

## Supported Models

### Models Evaluated in the Paper

**Vision-Language Models (VLMs):**
- **GPT-5 mini** (`gpt-5-mini`) - Azure OpenAI vision model (primary)
- **GPT-5 nano** (`gpt-5-nano`) - Lightweight Azure OpenAI vision model
- **Claude 3.5 Sonnet** (`claude-3-5-sonnet`) - Anthropic vision-language model

**OCR Systems:**
- **Azure Document Intelligence** (`azure-ai-documentintelligence`) - Enterprise-grade OCR with layout analysis
- **Mistral OCR 2** (`mistral-ocr-2505`) - Document parsing with markdown output
- **Mistral OCR 3** (`mistral-ocr-2512`) - Newer Mistral OCR version

### Additional Models Supported by the Codebase

- **Claude Sonnet / Haiku** - Via Anthropic Direct API or AWS Bedrock
- **Qwen-VL** - Open source vision-language model
- **Donut** - Open source document understanding
- **DeepSeek-OCR** - Multilingual OCR model

## Datasets

All datasets load automatically from HuggingFace (no local storage needed). Full collection: [kenza-ily/disco](https://huggingface.co/collections/kenza-ily/disco)

| Dataset | Task | Samples | HuggingFace URL |
|---------|------|---------|-----------------|
| IAM | Handwriting recognition | 500 | `kenza-ily/iam_disco` |
| DocVQA | Document question answering | 500 | `kenza-ily/docvqa_disco` |
| InfographicVQA | Infographic QA | 500 | `kenza-ily/infographicvqa_disco` |
| DUDE | Diverse documents QA | 404 | `kenza-ily/dude_disco` |
| ChartQA Pro | Chart question answering | 494 | `kenza-ily/chartqapro_disco` |
| PubLayNet | Document layout parsing | 500 | `kenza-ily/publaynet_disco` |
| VisRBench | Visual reasoning (multi-page) | 498 | `kenza-ily/visrbench_disco` |
| ICDAR | Multilingual OCR (10 languages) | 500 | `kenza-ily/icdar_disco` |
| RxPad | Medical prescription parsing (French) | 200 | `kenza-ily/rxpad_disco` |

## Benchmark Pipeline

The evaluation separates **text parsing** from **downstream question answering** across three pipeline architectures:

### Parsing Pipelines (IAM, ICDAR, RxPad, PubLayNet)

| Paper Name | Code Phase | Description |
|-----------|-----------|-------------|
| P_OCR | P-A | Pure OCR baseline — specialized OCR extracts text |
| P_VLM-base | P-B | VLM with generic text extraction prompt |
| P_VLM-task | P-C | VLM with task-aware, domain-specific prompt |

### QA Pipelines (DocVQA, InfographicVQA, DUDE, ChartQA, VisRBench)

| Paper Name | Code Phase | Description |
|-----------|-----------|-------------|
| QA_OCR | QA1a/QA1b/QA1c | Specialized OCR → LLM reasoning (with simple, detailed, or CoT prompt) |
| QA_VLM-2stage | QA2a/QA2b | VLM extracts text → VLM performs QA (two-stage) |
| QA_VLM-direct | QA3a | Single-step VLM answers directly from the image |
| Multi-page retrieval | QA4a | Multi-page with retrieval (VisRBench only) |

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

## Key Findings

Empirical guidance from our evaluation across 9 datasets:

| Document Type | Recommended Approach | Notes |
|--------------|---------------------|-------|
| Handwritten text (IAM) | OCR pipeline | VLMs lag by 5–9% CER even with task-aware prompting |
| Multilingual documents (ICDAR) | VLM (generic prompt) | 87% CER reduction vs OCR; OCR fails on non-Latin scripts |
| Single-page visual QA (DocVQA, InfographicVQA) | Direct VQA | Highest GT-in-Pred (~0.91); fewer error propagation stages |
| Multi-page documents (DUDE) | OCR pipeline | More reliable text grounding; VLMs struggle with long context |
| Medical prescriptions (RxPad) | Either | Similar accuracy; VLMs produce structured key-value output |

**Additional insights:**
- Task-aware prompting yields **heterogeneous effects** — substantially improves multilingual parsing but can degrade performance on diverse inputs
- **OCR system selection matters**: Azure Document Intelligence consistently outperforms Mistral OCR on structured documents
- Mistral OCR 3 (2512) shows a **23-point regression** vs Mistral OCR 2 (2505) on DocVQA — a newer version number does not guarantee improvement
- Direct VQA achieves the best **speed-accuracy frontier** (0.87–0.91 GT-in-Pred, 4–10s latency vs 17–35s for two-stage pipelines)

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

- **GT-in-Pred** (Ground-Truth-in-Prediction) - Primary QA metric; binary indicator whether the ground-truth answer appears in the model's prediction
- **CER** (Character Error Rate) - Character-level edit distance; primary parsing metric
- **WER** (Word Error Rate) - Word-level edit distance
- **ANLS** (Average Normalized Levenshtein Similarity) - String similarity for format compliance
- **Cosine Similarity** - Semantic similarity using embeddings
- **EM** (Exact Match) - Binary exact match
- **Substring Match** - Fuzzy matching for VQA

## Development

```bash
# Run tests
make test

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
@misc{benkirane2026disco,
  title={DISCO: Document Intelligence Suite for COmparative Evaluation},
  author={Benkirane, Kenza and Goldwater, Dan and Asenov, Martin and Ghodsi, Aneiss},
  year={2026},
  note={ICLR 2026 submission},
  url={https://github.com/kenza-ily/disco}
}
```

## License

MIT License - See LICENSE file for details.
