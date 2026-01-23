# OCR vs VLMs Research

## Research Questions

**Can OCR models remain competitive with Vision-Language Models for document understanding, and where is each approach optimal?**

### Parsing Tasks
1. **OCR vs. Direct VLM**: Compare structured field extraction using OCR pipeline vs VLM-only image analysis
2. **OCR + VLM Pipeline**: Extract text with OCR, then feed to VLM for reasoning — does this outperform end-to-end VLMs?

### QA Tasks
1. **OCR → LLM QA**: Traditional pipeline — OCR extracts text, then LLM answers questions
2. **Direct VLM QA**: End-to-end approach — VLM answers directly from image

---

## Documentation

See detailed specifications in the consolidated files:

- **[Experiments](../experiments.md)** — Phase naming, models used, prompts, results structure
- **[Datasets](../datasets.md)** — Dataset specifications (7 datasets: DocVQA, InfographicVQA, DUDE, IAM, ICDAR, PubLayNet, VOC2007)
- **[Models](../models.md)** — Model details, pricing, speed, strengths (5 actively tested models)
- **[Metrics](../metrics.md)** — Evaluation metrics: ANLS, EM, CER, WER, substring matching, embeddings

### Future Consideration

**Datasets not currently tested:**
- RVL-CDIP (large-scale document collection)
- FUNSD (noisy forms)
- DSL-QA (controlled OCR difficulty evaluation)

**Models for future work:**
- Donut (document understanding transformer, fine-tuning potential)
- Claude Haiku (cost-optimized VLM)
- Mistral 7B (open-source LLM for OCR pipelines)
- Qwen-VL (open-source multimodal model)

---

## Paper

[Research paper on Overleaf](https://www.overleaf.com/project/691c47e001b29217e22fb39c)