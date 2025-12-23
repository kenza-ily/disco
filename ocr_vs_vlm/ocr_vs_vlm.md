# OCR VS VLMs


## Purpose

Testing OCR models vs VLMs and evaluate to which extent OCRs remain key parsing tools - and it which way VLMs are more suit to some tasks

### Research questions

Parsing:

1. OCR vs. VLM direct parsing
Compare structured field extraction from a document using:
- OCR-based pipeline (e.g. Azure/Mistral OCR)
- VLM-only extraction (e.g. GPT-4, Claude), directly prompted to parse fields from image

2. OCR + VLM with prompt-based parsing
- Use OCR to extract plain text → feed to VLM with structured extraction prompt
Compare to VLM parsing from image directly
Useful to test how much structure is gained/lost in OCR-only text

QA:
1. OCR + VLM QA
OCR for raw text → VLM answers question based on that text
Simulates traditional pipeline with reasoning LLM downstream
2. End-to-end VLM QA
VLM answers question directly from document image
Tests multimodal reasoning and OCR+language integration

### Paper
[Overleaf link](https://www.overleaf.com/project/691c47e001b29217e22fb39c)



## Datasets


| Name (link)        | Description                                                                 | Pros                                                                 | Cons                                                                 | Year | Task    | Dataset available |
|-------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|------|---------|-------------------|
| RVL-CDIP          | Large-scale collection of scanned business documents across many categories (letters, forms, reports). | Very large scale; strong diversity of real documents; standard reference for document OCR and layout learning. | Weak fine-grained annotations; little handwriting; mostly English.   | 2015 | Parsing | Hugging Face: aharley/rvl_cdip |
| PubLayNet         | Document layout dataset with high-quality annotations for text blocks, tables, figures, and lists. | Excellent layout annotations; clean ground truth; strong for OCR + layout interaction. | Mostly digitally generated PDFs; limited scan noise; no handwriting. | 2019 | Parsing |                   |
| ICDAR 2019 MLT    | Multi-lingual scene text dataset covering 10 languages and multiple scripts. | Strong multi-script coverage; good stress test for multilingual OCR; widely used benchmark. | Focused on scene text rather than documents; limited long-form structure. | 2019 | Parsing | Y                 |
| FUNSD             | Noisy scanned forms with word-level boxes, entity labels, and semantic links. | Real scan artefacts; precise spatial annotations; good for form-style OCR evaluation. | Small scale; English only; limited document variety.                 | 2019 | Parsing |                   |
| IAM Handwriting   | Handwritten English text with line- and word-level transcriptions from hundreds of writers. | Canonical handwriting benchmark; clean and detailed ground truth; strong writer variability. | No complex layouts; no multilingual text; handwriting only.          | 2002 | Parsing |                   |
| DocVQA            | QA benchmark over scanned documents such as forms, letters, and contracts.   | OCR quality directly impacts answers; realistic layouts; widely recognised evaluation benchmark. | OCR and reasoning errors are entangled; no explicit OCR ground-truth alignment. | 2021 | QA      |                   |
| DUDE              | Multi-page document QA dataset built from diverse PDFs with dense layouts.    | Long documents; tables and complex structure; good for testing context handling. | Many documents are digitally clean; OCR difficulty varies; indirect OCR signal. | 2023 | QA      |                   |
| DSL-QA            | Curated dataset of 70 real-world document pages with 800 QA pairs targeting difficult layouts. | Designed to expose OCR failure modes; controlled difficulty; clean separation of OCR vs reasoning. | Small scale; high annotation cost; not a public benchmark yet.       | 2024 | QA      |                   |




## Models


| Model name                                 | Model size                      | Open / Closed | Rationale for choosing it                                                                                                                                                                          |
| ------------------------------------------ | ------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mistral OCR 3                              | undisclosed                     | closed        | Latest generation OCR from Mistral, designed for high-fidelity document parsing with strong layout and table retention. Used as a high-end OCR baseline for structured documents and noisy inputs. |
| Mistral OCR                                | undisclosed                     | closed        | Strong general-purpose OCR with good multilingual and layout handling. Serves as a stable reference OCR system and as the OCR stage for open pipelines (e.g. with Mistral 7B).                     |
| Azure Document Intelligence                | undisclosed                     | closed        | Widely used enterprise document OCR with built-in receipt and form models. Useful baseline for structured field extraction and cost/latency comparison against LLM-based systems.                  |
| Donut (Document Understanding Transformer) | ~400M–1B (varies by checkpoint) | open          | End-to-end document understanding without a separate OCR step. Suitable for fine-tuning on receipts and forms, and a strong open baseline for image-to-JSON tasks.                                 |
| GPT-5 mini                                 | undisclosed                     | closed        | Multimodal LLM with direct image input. Chosen to evaluate high-quality document understanding at a lower latency and cost than flagship models.                                                   |
| GPT-5 nano                                 | undisclosed                     | closed        | Smallest GPT-5 variant, used to study quality–cost trade-offs for document parsing and QA under strict latency or budget constraints.                                                              |
| Claude Sonnet 4.5                          | undisclosed                     | closed        | High-accuracy multimodal model with strong reasoning and structured output. Used as a top-tier proprietary VLM baseline for document QA and form understanding.                                    |
| Claude Haiku 4.5                           | undisclosed                     | closed        | Lightweight Claude variant optimised for speed and cost. Included to compare fast VLMs against OCR-centric pipelines.                                                                              |
| Mistral 7B                                 | 7B                              | open          | Compact open LLM suitable for fine-tuning. Used with OCR text input to simulate an open, controllable document understanding pipeline.                                                             |
| Qwen-VL 14B                                | 14B                             | open          | Open multimodal model with direct image input and strong OCR reasoning. Serves as an open alternative to proprietary VLMs for document QA and structured extraction.                               |


## Evaluation