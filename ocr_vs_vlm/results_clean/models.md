# Model Pricing and Performance Analysis

This document provides pricing and inference time analysis for all models used in the OCR vs VLM benchmark.

## Model Pricing Summary

### OCR Models (Per-Page Billing)

| Model | Provider | Price per 1000 Pages | Price per Page | Notes |
|-------|----------|---------------------|----------------|-------|
| **Azure Document Intelligence** | Azure | $1.50 | $0.0015 | 0-1M pages/month tier |
| **Mistral Document AI** | Mistral | $3.00 | $0.0030 | Azure AI Foundry (Global) |
| **Mistral OCR** | Mistral | $1.00 | $0.0010 | Basic OCR |

### VLM Models (Token-Based Billing)

| Model | Provider | Input (per 1M tokens) | Output (per 1M tokens) | Est. per Page |
|-------|----------|----------------------|------------------------|---------------|
| **GPT-5-mini** | OpenAI | $0.25 | $2.00 | ~$0.0024 |
| **GPT-5-nano** | OpenAI | $0.05 | $0.40 | ~$0.0005 |
| **Claude Sonnet 4** | Anthropic | $3.00 | $15.00 | ~$0.005-0.01 |

### Donut Model (Self-Hosted)

| Model | Provider | Notes |
|-------|----------|-------|
| **Donut** | Self-hosted | Free (compute costs only) |

---

## Cost Estimation per Dataset

### Assumptions
- Each page generates ~900 output tokens (for text extraction)
- VLM image input: ~2,500 tokens per page (GPT-5-mini), ~3,700 tokens (GPT-5-nano)
- Pipeline QA: parsing cost + additional QA inference cost
- Direct VQA: single inference cost per sample

### DocVQA_mini (500 samples)

| Phase | Model | Estimated Cost |
|-------|-------|----------------|
| **QA1a-c** (OCR Pipeline) | Azure + LLM | ~$0.75 (OCR) + QA cost |
| **QA1a-c** (OCR Pipeline) | Mistral + LLM | ~$1.50 (OCR) + QA cost |
| **QA2a-c** (VLM Pipeline) | GPT-5-mini | ~$1.21 (parsing) + QA cost |
| **QA2a-c** (VLM Pipeline) | GPT-5-nano | ~$0.27 (parsing) + QA cost |
| **QA3a-b** (Direct VQA) | GPT-5-mini | ~$1.21 |
| **QA3a-b** (Direct VQA) | GPT-5-nano | ~$0.27 |
| **QA3a-b** (Direct VQA) | Claude Sonnet | ~$2.50-5.00 |

### InfographicVQA_mini (500 samples)

Similar cost structure to DocVQA_mini.

### IAM_mini (500 samples)

| Model | Estimated Cost |
|-------|----------------|
| **Azure Document Intelligence** | ~$0.75 |
| **Mistral Document AI** | ~$1.50 |
| **GPT-5-mini** | ~$1.21 |
| **GPT-5-nano** | ~$0.27 |
| **Claude Sonnet** | ~$2.50-5.00 |

### ICDAR_mini (500 samples)

Similar cost structure to IAM_mini.

### PubLayNet (500 samples)

| Model | Estimated Cost |
|-------|----------------|
| **Azure Document Intelligence** | ~$0.75 |
| **Mistral Document AI** | ~$1.50 |

### VOC2007 (238 samples)

| Model | Estimated Cost |
|-------|----------------|
| **GPT-5-mini** | ~$0.57 |
| **GPT-5-nano** | ~$0.13 |

---

## Inference Time Analysis

Inference time data is extracted from the `inference_time_ms` column in result files.

### Expected Inference Times (per sample)

| Model | Typical Range | Notes |
|-------|--------------|-------|
| **Azure Document Intelligence** | 500-2000ms | Depends on document complexity |
| **Mistral Document AI** | 1000-3000ms | Network latency included |
| **GPT-5-mini** | 2000-5000ms | Vision tasks slower |
| **GPT-5-nano** | 1500-4000ms | Slightly faster than mini |
| **Claude Sonnet** | 3000-8000ms | More thorough analysis |
| **Donut** | 500-1500ms | Self-hosted, GPU dependent |

### Aggregate Statistics Template

After running `clean_files.py`, populate this table with actual data:

| Dataset | Model | Phase | Avg Time (ms) | Total Time (s) | Cost ($) |
|---------|-------|-------|---------------|----------------|----------|
| DocVQA_mini | azure_intelligence | QA1a | — | — | — |
| DocVQA_mini | mistral_document_ai | QA1a | — | — | — |
| DocVQA_mini | gpt-5-mini | QA2a | — | — | — |
| DocVQA_mini | gpt-5-nano | QA2a | — | — | — |
| ... | ... | ... | ... | ... | ... |

---

## Token Usage Estimation

For models without `tokens_used` column, we use tiktoken to estimate:

```python
import tiktoken

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

### Average Token Estimates

| Content Type | Avg Tokens |
|--------------|------------|
| Document page (text) | 500-1500 |
| Document page (image) | 2000-4000 |
| Question | 10-50 |
| Answer | 5-50 |
| CoT Response | 100-300 |

---

## Cost-Performance Tradeoffs

### Recommendations by Use Case

| Use Case | Recommended Model | Reasoning |
|----------|-------------------|-----------|
| **Bulk OCR (cost-sensitive)** | Mistral OCR | $1.00/1000 pages, good accuracy |
| **High-accuracy OCR** | Azure Document Intelligence | Best form/table extraction |
| **Quick VQA (cost-sensitive)** | GPT-5-nano Direct VQA | Cheapest VLM option |
| **Accurate VQA** | Claude Sonnet Direct VQA | Best reasoning, but expensive |
| **Pipeline VQA** | Azure + GPT-5-mini | Good balance of accuracy/cost |

### Quality vs Cost Matrix

```
High Quality
    ^
    |  Claude Sonnet
    |      ●
    |          GPT-5-mini
    |              ●
    |  Azure DI        GPT-5-nano
    |      ●               ●
    |          Mistral DAI
    |              ●
    |                  Donut
    |                    ●
    +-------------------------> Low Cost
```

---

## Scripts for Analysis

### Calculate Actual Costs

```python
"""
Calculate actual costs from consolidated result files.
Uses inference_time_ms and tokens_used columns where available,
estimates with tiktoken otherwise.
"""

import csv
import json
from pathlib import Path
import tiktoken

def load_prices():
    with open("../../llms/prices.json") as f:
        return json.load(f)

def calculate_dataset_costs(dataset_dir: Path, prices: dict):
    """Calculate costs for all experiments in a dataset."""
    results = {}
    
    for csv_file in dataset_dir.glob("*.csv"):
        experiment = csv_file.stem
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Extract model names from columns
        models = set()
        for col in reader.fieldnames:
            if col.startswith("prediction_"):
                models.add(col.replace("prediction_", ""))
        
        # Calculate per-model metrics
        for model in models:
            total_time = 0
            token_count = 0
            
            for row in rows:
                time_col = f"inference_time_ms_{model}"
                token_col = f"tokens_used_{model}"
                
                if time_col in row and row[time_col]:
                    total_time += float(row[time_col])
                
                if token_col in row and row[token_col]:
                    token_count += int(row[token_col])
            
            results[(experiment, model)] = {
                "total_time_ms": total_time,
                "total_time_s": total_time / 1000,
                "token_count": token_count,
                "sample_count": len(rows),
            }
    
    return results

if __name__ == "__main__":
    prices = load_prices()
    
    for dataset_dir in Path(".").iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith("_"):
            print(f"\n=== {dataset_dir.name} ===")
            costs = calculate_dataset_costs(dataset_dir, prices)
            for (exp, model), metrics in sorted(costs.items()):
                print(f"  {exp} / {model}:")
                print(f"    Time: {metrics['total_time_s']:.1f}s")
                print(f"    Samples: {metrics['sample_count']}")
```

---

## Data Sources

- **Pricing data**: [`/llms/prices.json`](../../llms/prices.json)
- **As of date**: 2026-01-13
- **Currency**: USD

---

## Notes

1. **Token estimates are approximate** — actual costs vary with document complexity
2. **Inference times include network latency** — self-hosted models will differ
3. **Volume discounts available** — Azure offers $0.60/1000 for 1M+ pages
4. **Cached prompts may reduce costs** — some providers offer prompt caching
