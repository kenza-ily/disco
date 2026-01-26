# Experiments

## Overview

Benchmark comparing **OCR Pipeline** vs **VLM Direct** approaches across two main tasks:

### Task 1: Question Answering (VQA)
Answer questions about document content using either parsed text or direct vision understanding.

**Datasets:**
- **DocVQA_mini** (500 samples) — Questions about business/legal documents
- **InfographicVQA_mini** (500 samples) — Questions about infographics with charts/statistics

### Task 2: Text Parsing (OCR)
Extract and structure text from images with varying complexity.

**Datasets:**
- **IAM_mini** (500 samples) — Handwritten text recognition
- **ICDAR_mini** (500 samples) — Scene text recognition  
- **PubLayNet** (500 samples) — Document layout detection
- **VOC2007** (238 samples) — Natural image object detection

## VQA Experiments (DocVQA, InfographicVQA)

### Phase Naming

| Phase | Approach | Parsing | QA Model | Prompt |
|-------|----------|---------|----------|--------|
| **QA1a** | OCR → LLM | Azure DocIntel | GPT-5 | Simple |
| **QA1b** | OCR → LLM | Azure DocIntel | GPT-5 | Detailed |
| **QA1c** | OCR → LLM | Azure DocIntel | GPT-5 | Chain-of-Thought |
| **QA2a** | VLM → LLM | GPT-5-mini/nano | GPT-5 | Simple |
| **QA2b** | VLM → LLM | GPT-5-mini/nano | GPT-5 | Detailed |
| **QA2c** | VLM → LLM | GPT-5-mini/nano | GPT-5 | Chain-of-Thought |
| **QA3a** | Direct VQA | — | VLM | Simple |
| **QA3b** | Direct VQA | — | VLM | Detailed |

**Alternative Names:**
- QA1a = QA-OCR_LLM_simple
- QA2a = QA-VLM_LLM_simple  
- QA3a = QA-VLM_direct_simple

### Models

**OCR Parsing:**
- `azure_intelligence` — Azure AI Document Intelligence
- `mistral_document_ai` — Mistral Document AI

**VLM (Parsing + QA):**
- `gpt-5-mini`, `gpt-5-nano` — OpenAI GPT-5
- `claude_sonnet` — Anthropic Claude Sonnet 4

### Prompt Types

**Simple** — Minimal instruction, just extracted text + question

**Detailed** — Task-specific context (forms/tables for DocVQA, charts/stats for InfographicVQA)

**Chain-of-Thought** — Step-by-step reasoning requested before answer

### Example Prompts

**QA1a (Simple):**
```
Based on the following document text, answer the question.
Document text: {text}
Question: {question}
Answer:
```

**QA3b (Direct Detailed):**
```
This is a document image that may contain forms, tables, or text.
Pay attention to form fields, table data, and document structure.
Question: {question}
Answer:
```

## Parsing Experiments (IAM, ICDAR, VOC2007)

### Phase Naming

| Phase | Description | Models |
|-------|-------------|--------|
| **phase_1** | Basic text extraction | azure_intelligence, mistral_document_ai, gpt-5-mini, gpt-5-nano |
| **phase_2** | Structure-preserving extraction | azure_intelligence, mistral_document_ai, gpt-5-mini, gpt-5-nano |
| **phase_3** | Advanced formatting instructions | azure_intelligence, mistral_document_ai, gpt-5-mini, gpt-5-nano |

## Layout Detection (PubLayNet)

### Phase Naming

| Phase | Description | Models |
|-------|-------------|--------|
| **P-A** | Basic layout detection | azure_intelligence, mistral_document_ai, gpt-5-mini, gpt-5-nano |
| **P-B** | Detailed detection instructions | azure_intelligence, mistral_document_ai, gpt-5-mini, gpt-5-nano |
| **P-C** | Bounding box format specification | azure_intelligence, mistral_document_ai, gpt-5-mini, gpt-5-nano |

**Output Format:**
```json
{"boxes": [
  {"label": "text", "bbox": [x1, y1, x2, y2]},
  {"label": "title", "bbox": [x1, y1, x2, y2]}
]}
```

## Results Structure

```
results_clean/
├── DocVQA_mini/
│   ├── QA1a.csv  # azure_intelligence + mistral_document_ai
│   ├── QA2a.csv  # gpt-5-mini + gpt-5-nano
│   ├── QA3a.csv  # gpt-5-mini + gpt-5-nano
│   └── ...
├── InfographicVQA_mini/
│   └── ... (same structure)
├── IAM_mini/
│   ├── phase_1.csv
│   ├── phase_2.csv
│   └── phase_3.csv
└── publaynet/
    ├── P-A.csv
    ├── P-B.csv
    └── P-C.csv
```
Findings

### By Approach
- **OCR Pipeline (QA1)**: Best for documents with clear text structure
- **VLM Pipeline (QA2)**: Handles complex layouts, lower accuracy
- **Direct VQA (QA3)**: Fastest, competitive accuracy, no parsing overhead

### By Prompt Type
- **Simple**: Fastest, often sufficient
- **Detailed**: Better on complex documents
- **Chain-of-Thought**: Best for multi-step reasoning, slower

### By Model
- **GPT-5-mini**: Highst, often sufficient
- **Detailed**: 10-15% improvement on complex docs
- **Chain-of-Thought**: Best for multi-step reasoning, slower

### By Model
- **GPT-5-mini**: Best accuracy, higher cost
- **GPT-5-nano**: Good balance, faster
- **Claude Sonnet**: Strong visual understanding
- **Azure DocIntel**: Most reliable OCR
- **Mistral Document AI**: Fast, occasional connection issues

---

## VisR-Bench Experiments (Multi-Page Long-Document Retrieval + QA) ⭐

### Task: Document Retrieval + Question Answering

**Key Difference from Short-Doc Tasks:**
- **Short docs (DocVQA, InfographicVQA)**: Parse entire page → Answer question
- **Long docs (VisR-Bench)**: Find relevant page in multi-page doc → Parse → Answer question

VisR-Bench tests **information retrieval** as a primary challenge, not just parsing and reasoning.

### Phase Naming (Evidence Page vs Retrieval)

**Phases QA1–QA4: Evidence Page Only** (ground truth page provided)
- Tests parsing and QA quality in isolation
- Same approach as short-doc benchmarks for continuity

| Phase | Approach | Parsing | QA Model | Prompt | Use Case |
|-------|----------|---------|----------|--------|----------|
| **QA1a** | OCR → LLM | Azure DocIntel | GPT-5-mini | Simple | Text extraction baseline |
| **QA1b** | OCR → LLM | Azure DocIntel | GPT-5-mini | Detailed | Structure-aware extraction |
| **QA1c** | OCR → LLM | Azure DocIntel | GPT-5-mini | Chain-of-Thought | Multi-step reasoning |
| **QA2a** | VLM → LLM | GPT-5-mini | GPT-5-mini | Simple | Visual parsing baseline |
| **QA2b** | VLM → LLM | GPT-5-mini | GPT-5-mini | Detailed | Context-aware VLM parsing |
| **QA2c** | VLM → LLM | GPT-5-mini | GPT-5-mini | Chain-of-Thought | Reasoning + visual layout |
| **QA3a** | Direct VQA | — | GPT-5-mini | Simple | End-to-end vision + text |
| **QA3b** | Direct VQA | — | GPT-5-mini | Detailed | Structured document understanding |
| **QA4a** | Gold Text → LLM | (pre-extracted) | GPT-5-mini | Simple | Pure QA without parsing error |
| **QA4b** | Gold Text → LLM | (pre-extracted) | GPT-5-mini | Detailed | QA with full document context |
| **QA4c** | Gold Text → LLM | (pre-extracted) | GPT-5-mini | Chain-of-Thought | QA with step-by-step reasoning |

**Phases QA5: Retrieval + QA** (all pages available, must retrieve then answer)
- Tests document-level information retrieval
- Evaluates page ranking and context selection

| Phase | Approach | Retrieval Method | Top-K | Purpose |
|-------|----------|------------------|-------|---------|
| **QA5a** | Retrieval → QA | Dense embeddings | 1 | High-precision retrieval |
| **QA5b** | Retrieval → QA | Dense embeddings | 5 | Precision-recall trade-off |
| **QA5c** | Retrieval → QA | BM25 (keyword) | 1 | Sparse retrieval baseline |
| **QA5d** | Retrieval → QA | BM25 (keyword) | 5 | Sparse multi-page context |

### Models (Cost-Aware Selection)

**OCR Extraction:**
- `azure_intelligence` — Azure AI Document Intelligence (reliable baseline)
- `mistral_document_2505` — Mistral Document AI 2505 (fast alternative)

**VLM (Parsing + QA):**
- `gpt-5-mini` — Primary model (best balance of speed/quality)
- `gpt-5-nano` — Budget alternative (faster, lower cost)
- `claude_sonnet` — Qualitative baseline (strong visual understanding)

### Sampling Strategy (Development → Production)

**Development Runs (Recommended for Testing):**
```bash
python -m ocr_vs_vlm.benchmarks.benchmark_visrbench \
  --sample-limit 10 \
  --qa-per-doc 1 \
  --content-type multilingual \
  --phases QA1a QA3a QA5a
```
- 1 QA per document (ensures document diversity)
- Single content type (faster iteration)
- Single phase per approach (targeted testing)
- Est. runtime: 5–10 min

**Stratified Runs (Recommended for Analysis):**
```bash
python -m ocr_vs_vlm.benchmarks.benchmark_visrbench \
  --sample-limit 100 \
  --qa-per-doc 5 \
  --phases QA1a QA2a QA3a QA4a QA5a QA5c
```
- 5 QAs per document (capped to avoid over-weighting)
- Full dataset (all content types)
- Mix of approaches (QA1, QA2, QA3, QA4, QA5)
- Est. runtime: 30–60 min per model

**Full Production Run (After Analysis):**
```bash
python -m ocr_vs_vlm.benchmarks.benchmark_visrbench \
  --qa-per-doc 5
```
- All 17,045 QA pairs (capped 5 per doc for diversity)
- All content types + languages
- All phases: QA1–QA5 × OCR/VLM variants
- Est. runtime: 2–4 hours for full evaluation

### Metrics by Phase

**QA1–QA4 (Evidence Page Only):**
- ANLS (Alphanumeric Overlap) — primary metric
- Exact Match
- Substring Match
- Embedding Similarity

**QA5 (Retrieval RAG):**
- **Retrieval Accuracy**: Evidence page in top-k? (binary)
- **Retrieval Rank**: Position of correct page in ranking
- **Retrieval MRR**: Mean Reciprocal Rank across samples
- Plus QA metrics: ANLS, Exact Match, etc.

### Why VisR-Bench Matters

1. **New Failure Mode**: Models can understand content correctly but fail to locate it
2. **Scalability Test**: 74–85% of QAs target pages beyond page 5 (deep retrieval needle-in-haystack)
3. **Multi-Modal Challenge**: Mix of figures (66.9% long answers), tables (52.3% short), text, and multilingual (13,237 QAs)
4. **Production Realism**: Long documents are common in enterprise scenarios

### Expected Insights

- **OCR vs VLM on evidence page (QA1–QA4)**: Repeat findings from short-doc benchmarks
- **Retrieval impact (QA5)**: Accuracy drops due to incorrect page selection
- **Dense vs BM25 (QA5a/b vs QA5c/d)**: Trade-offs in precision/recall by content type
- **Language effects**: Multilingual subset (13K QAs) vs English (figure/table/text subsets)
- **Page depth**: Do retrieval methods struggle more as documents get longer?
