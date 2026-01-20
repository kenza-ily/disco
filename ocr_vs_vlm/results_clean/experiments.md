# OCR vs VLM Benchmark Experiments

This document describes all experiments conducted as part of the OCR vs VLM benchmark comparison.

## Overview

The benchmark evaluates two main approaches to document understanding:

1. **Pipeline Approach**: OCR/VLM extracts text → LLM answers question
2. **Direct VQA Approach**: VLM sees image + question, answers directly

## Datasets

| Dataset | Task Type | Samples | Description |
|---------|-----------|---------|-------------|
| **DocVQA_mini** | Question Answering | 500 | Business/legal document VQA |
| **InfographicVQA_mini** | Question Answering | 500 | Infographic VQA with charts/stats |
| **IAM_mini** | Text Parsing | 500 | Handwritten text recognition |
| **ICDAR_mini** | Text Parsing | 500 | Scene text recognition |
| **PubLayNet** | Layout Detection | 500 | Document layout analysis |
| **VOC2007** | Object Detection | 238 | Natural image understanding |

---

## QA Experiments (DocVQA, InfographicVQA)

### Phase Naming Convention

| Phase | Approach | Parsing Model | QA Model | Prompt Type |
|-------|----------|---------------|----------|-------------|
| **QA1a** | OCR Pipeline | Azure Document Intelligence | LLM | Simple |
| **QA1b** | OCR Pipeline | Azure Document Intelligence | LLM | Detailed |
| **QA1c** | OCR Pipeline | Azure Document Intelligence | LLM | Chain-of-Thought |
| **QA2a** | VLM Pipeline | GPT-5-mini/nano | LLM | Simple |
| **QA2b** | VLM Pipeline | GPT-5-mini/nano | LLM | Detailed |
| **QA2c** | VLM Pipeline | GPT-5-mini/nano | LLM | Chain-of-Thought |
| **QA3a** | Direct VQA | — | VLM | Simple |
| **QA3b** | Direct VQA | — | VLM | Detailed |
| **QA4a** | Pre-extracted OCR | External OCR | LLM | Simple |
| **QA4b** | Pre-extracted OCR | External OCR | LLM | Detailed |
| **QA4c** | Pre-extracted OCR | External OCR | LLM | Chain-of-Thought |

### Models Used

**OCR Models (Parsing):**
- `azure_intelligence` — Azure AI Document Intelligence (Read API)
- `mistral_document_ai` — Mistral Document AI 2505

**VLM Models (Parsing + Direct VQA):**
- `gpt-5-mini` — OpenAI GPT-5 Mini
- `gpt-5-nano` — OpenAI GPT-5 Nano
- `claude_sonnet` — Anthropic Claude Sonnet 4

---

## Prompt Details

### Pipeline QA Prompts

#### QA Simple (`QA1a`, `QA2a`, `QA4a`)

```text
Based on the following document text, answer the question.

Document text:
{extracted_text}

Question: {question}

Answer:
```

#### QA Detailed (`QA1b`, `QA2b`, `QA4b`)

For **DocVQA**:
```text
You are a document understanding assistant. Answer the question based ONLY on the provided document text.

This text was extracted from a business/legal document. Documents may contain:
- Forms with labels and values
- Tables with headers and data
- Handwritten annotations
- Multiple sections with headers

Look for the specific information requested in the question.

Document text:
{extracted_text}

Question: {question}

Instructions:
- Answer concisely with just the requested information
- If the answer is a number, include any units if present
- If the answer cannot be found in the text, respond with "NOT FOUND"
- Do not include explanations unless specifically asked

Answer:
```

For **InfographicVQA**:
```text
You are a document understanding assistant. Answer the question based ONLY on the provided document text.

This text was extracted from an infographic. Infographics often contain:
- Statistics and percentages
- Comparisons and rankings
- Data visualizations described in text
- Short phrases rather than complete sentences

When answering, consider that some information may require inference from multiple parts of the text.

Document text:
{extracted_text}

Question: {question}

Instructions:
- Answer concisely with just the requested information
- If the answer is a number, include any units if present
- If the answer cannot be found in the text, respond with "NOT FOUND"
- Do not include explanations unless specifically asked

Answer:
```

#### QA Chain-of-Thought (`QA1c`, `QA2c`, `QA4c`)

```text
You are analyzing a document to answer a question. [task_hint]

Document text:
{extracted_text}

Question: {question}

Let's solve this step by step:
1. First, identify what information the question is asking for
2. Search the document text for relevant information
3. Extract the specific answer

Reasoning:
[Your step-by-step reasoning here]

Final Answer:
```

### Direct VQA Prompts

#### Direct VQA Simple (`QA3a`)

```text
Look at this document image and answer the question.

Question: {question}

Answer:
```

#### Direct VQA Detailed (`QA3b`)

For **DocVQA**:
```text
This is a document image that may contain forms, tables, or text.
Pay attention to:
- Form fields and their values
- Table headers and data cells
- Handwritten text or annotations
- Document structure and sections

Carefully examine the image to answer the following question.

Question: {question}

Instructions:
- Provide a concise, direct answer
- Include units or context if relevant
- If you cannot find the answer, respond with "NOT FOUND"

Answer:
```

For **InfographicVQA**:
```text
This is an infographic image containing data visualizations, statistics, and text.
Pay attention to:
- Charts, graphs, and visual data representations
- Numbers, percentages, and statistics
- Labels, titles, and annotations
- Relationships between different elements

Carefully examine the image to answer the following question.

Question: {question}

Instructions:
- Provide a concise, direct answer
- Include units or context if relevant
- If you cannot find the answer, respond with "NOT FOUND"

Answer:
```

---

## Parsing Experiments (IAM, ICDAR, VOC2007)

### Phase Naming Convention

| Phase | Description |
|-------|-------------|
| **phase_1** | Basic text extraction prompt |
| **phase_2** | Detailed text extraction with structure preservation |
| **phase_3** | Advanced extraction with formatting instructions |

### Parsing Prompts

#### DocVQA Parsing Prompt (used in QA1, QA2)

```text
Extract ALL text from this document image.

Instructions:
- Preserve the structure and layout of the document
- Include form labels and their values
- Include table headers and cell contents
- Include any handwritten text or annotations
- Maintain reading order (top to bottom, left to right)

Return ONLY the extracted text, no commentary.
```

#### InfographicVQA Parsing Prompt (used in QA1, QA2)

```text
Extract ALL text from this infographic image.

Instructions:
- Include all visible text: titles, labels, data values, annotations
- Preserve numerical data and statistics exactly as shown
- Include text from charts, graphs, and diagrams
- Capture any footnotes or source citations
- Read text from all areas: headers, body, legends, captions

Return ONLY the extracted text, no commentary.
```

---

## PubLayNet Experiments (Layout Detection)

### Phase Naming Convention

| Phase | Description | Model Type |
|-------|-------------|------------|
| **P-A** | Layout detection with basic prompt | OCR/VLM |
| **P-B** | Layout detection with detailed instructions | OCR/VLM |
| **P-C** | Layout detection with bounding box format specifications | OCR/VLM |

### Output Format

PubLayNet experiments output bounding boxes in the format:
```json
{
  "boxes": [
    {"label": "text", "bbox": [x1, y1, x2, y2]},
    {"label": "title", "bbox": [x1, y1, x2, y2]},
    ...
  ]
}
```

---

## Evaluation Metrics

### QA Tasks (DocVQA, InfographicVQA)

| Metric | Description |
|--------|-------------|
| **ANLS** | Average Normalized Levenshtein Similarity (0-1) |
| **Exact Match** | Binary match after normalization (0 or 1) |

### Parsing Tasks (IAM, ICDAR)

| Metric | Description |
|--------|-------------|
| **CER** | Character Error Rate |
| **WER** | Word Error Rate |
| **BLEU** | BLEU score for text similarity |

### Layout Detection (PubLayNet)

| Metric | Description |
|--------|-------------|
| **mAP** | Mean Average Precision at various IoU thresholds |
| **IoU** | Intersection over Union for bounding boxes |

---

## File Structure

After consolidation, results are organized as:

```
results_clean/
├── DocVQA_mini/
│   ├── QA1a.csv
│   ├── QA1b.csv
│   ├── QA1c.csv
│   ├── QA2a.csv
│   ├── QA2b.csv
│   ├── QA2c.csv
│   ├── QA3a.csv
│   └── QA3b.csv
├── InfographicVQA_mini/
│   └── ... (similar structure)
├── IAM_mini/
│   ├── phase_1.csv
│   ├── phase_2.csv
│   └── phase_3.csv
├── ICDAR_mini/
│   └── ... (similar structure)
├── publaynet/
│   ├── P-A.csv
│   ├── P-B.csv
│   └── P-C.csv
└── VOC2007/
    └── ... (similar structure)
```

Each consolidated CSV contains columns from all models:
- Shared columns: `sample_id`, `image_path`, `question`, `ground_truths`
- Model-specific: `prediction_{model}`, `anls_score_{model}`, `inference_time_ms_{model}`
