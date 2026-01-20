# Experiment Registry

## QA Experiments (DocVQA, InfographicVQA)

| Phase | Approach | Parsing | QA Model | Prompt |
|-------|----------|---------|----------|--------|
| **QA1a** | OCR→LLM | Azure DocIntel | GPT-5-mini | Simple |
| **QA1b** | OCR→LLM | Azure DocIntel | GPT-5-mini | Detailed |
| **QA1c** | OCR→LLM | Azure DocIntel | GPT-5-mini | CoT |
| **QA2a** | VLM→LLM | GPT-5-mini | GPT-5-mini | Simple |
| **QA2b** | VLM→LLM | GPT-5-mini | GPT-5-mini | Detailed |
| **QA2c** | VLM→LLM | GPT-5-mini | GPT-5-mini | CoT |
| **QA3a** | Direct VQA | — | VLM | Simple |
| **QA3b** | Direct VQA | — | VLM | Detailed |
| **QA4a-c** | ExtOCR→LLM | AWS Textract | GPT-5-mini | Simple/Det/CoT |

## Parsing Experiments (IAM, ICDAR, VOC2007)

| Phase | Description | Models |
|-------|-------------|--------|
| **P1** | OCR baseline | azure_intelligence, mistral_document_ai |
| **P2** | VLM generic prompt | gpt-5-mini, gpt-5-nano |
| **P3** | VLM task-aware prompt | gpt-5-mini, gpt-5-nano |
| **P4** | VLM domain-specific | gpt-5-mini, gpt-5-nano |

## Layout Detection (PubLayNet)

| Phase | Description | Models |
|-------|-------------|--------|
| **PL-A** | OCR layout inference | azure_intelligence, mistral_document_ai |
| **PL-B** | VLM direct detection | gpt-5-mini, gpt-5-nano |
| **PL-C** | VLM + OCR hybrid | gpt-5-mini, gpt-5-nano |

## Metrics by Task

- **QA**: ANLS (primary), Exact Match, Substring Match
- **Parsing**: CER↓, WER↓, ANLS↑
- **Layout**: IoU, mAP, F1
