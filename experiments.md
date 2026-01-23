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
