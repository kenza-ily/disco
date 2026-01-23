````md
# Mistral OCR – usage and batching guide

This document explains how to run `mistral_model.py` for single documents and how to organise data for batch OCR to reduce costs.

The file wraps **Mistral Document AI OCR** using the latest model: `mistral-ocr-latest`.

---

## 1. Prerequisites

### 1.1 Install dependencies

Make sure the Mistral Python SDK is installed:

```bash
pip install mistralai
````

### 1.2 Set your API key

Export your Mistral API key as an environment variable:

```bash
export MISTRAL_API_KEY="your_api_key_here"
```

The class will fail fast if the key is missing.

---

## 2. Running OCR on a single document

### 2.1 Supported inputs

The `ocr()` method accepts:

* Local files (PDFs or images)
* Remote URLs (PDF or image URLs)

The code automatically:

* Uploads local files and uses `file_id`
* Chooses `document_url` for PDFs and documents
* Chooses `image_url` for images

### 2.2 Example: local PDF

```python
from mistral_model import MistralOCR

ocr = MistralOCR()

result = ocr.ocr("data/invoices/sample.pdf")
print(result)
```

### 2.3 Example: local image

```python
result = ocr.ocr("data/receipts/receipt_01.jpg")
```

### 2.4 Example: remote URL

```python
result = ocr.ocr(
    "https://example.com/scanned_report.pdf"
)
```

### 2.5 Page selection (PDFs only)

You can restrict OCR to specific pages (0-based indexing):

```python
result = ocr.ocr(
    "data/report.pdf",
    pages=[0, 1, 2]
)
```

---

## 3. Output format

The OCR response includes:

* Extracted text blocks
* Layout information
* Optional base64-encoded page images (enabled by default)

This mirrors the raw API output so you can post-process as needed.

---

## 4. Why batching matters

Batch inference is the recommended way to:

* Reduce per-document cost
* Process large volumes of documents
* Avoid repeated request overhead

Instead of sending one OCR request at a time, you:

1. Build a `.jsonl` file describing many OCR requests
2. Upload it once
3. Run a batch job against the `/v1/ocr` endpoint

---

## 5. Organising your data for batching

A clean and predictable layout helps a lot.

### 5.1 Recommended directory structure

```text
data/
├── pdfs/
│   ├── report_001.pdf
│   ├── report_002.pdf
│   └── ...
├── images/
│   ├── receipt_001.jpg
│   ├── receipt_002.png
│   └── ...
└── batches/
    ├── ocr_inputs.jsonl
    └── ...
```

Keep PDFs and images separate if possible. Mixed inputs still work, but separation makes debugging easier.

---

## 6. Creating a batch input file

### 6.1 Basic idea

Each line in the `.jsonl` file represents **one OCR request**:

```json
{
  "custom_id": "0",
  "body": {
    "document": { ... },
    "include_image_base64": true
  }
}
```

`custom_id` lets you map outputs back to inputs.

### 6.2 Using the helper method

The class provides a helper to build this file for you.

#### Example: batch from local files (recommended)

```python
from mistral_model import MistralOCR

ocr = MistralOCR()

inputs = [
    "data/pdfs/report_001.pdf",
    "data/pdfs/report_002.pdf",
    "data/images/receipt_001.jpg",
]

jsonl_path = ocr.write_batch_jsonl(
    inputs=inputs,
    output_path="data/batches/ocr_inputs.jsonl",
    mode="upload_then_file_id",
)
```

What happens:

* Each local file is uploaded once
* The batch file references `file_id`s
* Large base64 payloads are avoided

This is the safest option for PDFs and large images.

---

### 6.3 Alternative: data URL batching

You can inline files as `data:` URLs:

```python
jsonl_path = ocr.write_batch_jsonl(
    inputs=inputs,
    output_path="data/batches/ocr_inputs.jsonl",
    mode="data_url",
)
```

Notes:

* Images become `data:image/...;base64,...`
* PDFs become `data:application/pdf;base64,...`
* File size matters; very large PDFs can create huge batch files

---

## 7. Submitting a batch job

Once the `.jsonl` file is ready:

```python
job = ocr.create_batch_job(
    batch_jsonl_path="data/batches/ocr_inputs.jsonl",
    metadata={"project": "invoice-ocr"},
    timeout_hours=24,
)

print(job.id)
```

Mistral will process all OCR requests asynchronously.

---

## 8. Reading batch results

When the batch completes:

* Each output line corresponds to a `custom_id`
* You can join results back to your original file list using that ID

Typical workflow:

1. Store `(custom_id → file path)` mapping locally
2. Download batch outputs
3. Parse and align results with your dataset

---

## 9. Practical tips

* Batch PDFs separately from images if page counts vary a lot
* Use `upload_then_file_id` for anything larger than a few MB
* Keep one document per batch line (do not merge files)
* Start with small batches to validate output format

---

## 10. Summary

* `mistral_model.py` handles single OCR cleanly
* Batch inference is the cost-efficient path for scale
* Use structured folders and generated `.jsonl` files
* Let `custom_id` be your ground truth link between inputs and outputs
