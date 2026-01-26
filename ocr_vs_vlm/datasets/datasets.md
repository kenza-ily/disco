# Datasets

## Task 1: Question Answering (VQA)

### DocVQA_mini
- **Type**: Scanned forms, receipts, letters
- **Samples**: 500 QA pairs
- **Complexity**: Moderate — structured documents, printed text, variable quality scans

### InfographicVQA_mini
- **Type**: Infographics and data visualizations
- **Samples**: 500 QA pairs  
- **Complexity**: High — graphs, charts, numerical values, legends, mixed text orientations
- **Special**: Pre-extracted OCR from AWS Textract included

### DUDE_mini
- **Type**: Real-world documents (diverse layouts)
- **Samples**: 404 (stratified across question families)
- **Complexity**: Very High — multiple languages, tables, forms, receipts, invoices, mixed quality
- **Stratification**: Question family (numeric, date/time, lookup, yes/no, multi-hop), answer type, document ID (max 5 per doc)

### ChartQAPro_mini
- **Type**: Charts, infographics, dashboards, and data visualizations
- **Samples**: 494 QA pairs (stratified from 1,948 total)
- **Complexity**: Very High — requires numerical reasoning, trend analysis, multi-step inference, diverse question types
- **Question Types**: Factoid (55.9%), Conversational (16%), Fact Checking (12.8%), Multi Choice (10.7%), Hypothetical (4.7%)
- **Answer Types**: short_text (38.3%), numeric (37.7%), boolean (13.2%), multiple_choice (8.9%), long_text (2%)
- **Special**: 
  - Conversational samples have 2-6 follow-up questions (multi-turn QA)
  - 12.6% include paragraph context (pre-extracted text)
  - 4.3% require temporal/year reasoning
  - Modern benchmark (published 2025, shows VLM saturation - Claude 3.5 Sonnet: 55.8% vs 90.5% on older ChartQA)
- **Stratification**: Question Type (proportional), Answer Type (within question type), multi-turn depth, context availability

### VisR-Bench_mini ⭐ Retrieval + QA
- **Type**: Long multi-page documents (PDFs) with figures, tables, text, and multilingual content
- **Samples**: 498 documents, 17,045 QA pairs
- **Task**: **Document Retrieval + Question Answering** — Models must first locate relevant pages in long documents, then answer questions based on retrieved evidence
- **Complexity**: Very High — multi-page reasoning (median 7 pages, max 417), information retrieval across document, evidence grounding
- **Why Retrieval Matters**:
  - 74–85% of QAs target pages **beyond page 5** (tests needle-in-haystack retrieval)
  - Each QA includes `evidence_pages` field for retrieval evaluation
  - Tests VLM ability to handle long-context documents (up to 417 pages)
- **Content Types**:
  - **Figure** (40 docs, 142 QA): Scientific figures, diagrams, visualizations
  - **Table** (67 docs, 1,512 QA): Tabular data extraction and reasoning
  - **Text** (97 docs, 2,154 QA): Dense textual content understanding
  - **Multilingual** (294 docs, 13,237 QA): 15 languages (es 18.9%, fr 11.4%, nl 10.0%, it 8.5%, vi 8.2%, de 7.0%, pt 6.4%, pl 4.7%, ru 4.5%, ja 4.4%, zh 3.5%, ko 3.2%, ar 2.9%, uk 0.3%, th 0.2%)
- **Document Statistics**:
  - Pages: 2–417 (mean 21.2, median 7.0)
  - File sizes: 23.60 MB total (multilingual 15.57 MB, text 3.39 MB, table 3.09 MB, figure 1.54 MB)
- **Answer Types**:
  - Figure: 66.9% long, 26.1% medium, 7.0% short
  - Table: 52.3% short, 27.6% medium, 20.1% long
  - Text: 42.9% medium, 32.6% short, 24.5% long
- **Special**:
  - Modern benchmark (CVPR 2025) testing VLM limits on long-context document understanding
  - Page-level images: All pages stored as separate image files
  - Ideal for RAG pipeline evaluation and retrieval-augmented QA
- **HuggingFace**: [kenza-ily/visr-bench-mini](https://huggingface.co/datasets/kenza-ily/visr-bench-mini)

## Task 2: Text Parsing (OCR)

### IAM_mini
- **Type**: Handwritten text
- **Samples**: 500
- **Complexity**: High — handwritten English, writer variability

### ICDAR_mini
- **Type**: Scene text (natural images)
- **Samples**: 500 (50 per language, stratified)
- **Complexity**: Extreme — 10 languages (Arabic, Bangla, Chinese, Hindi, Japanese, Korean, Latin, Mixed, Symbols, None)

### PubLayNet
- **Type**: Document pages (scientific papers, books, magazines)
- **Samples**: 500
- **Complexity**: Moderate — structured layouts, multi-column text, tables, figures, titles
- **Categories**: Text, Title, List, Table, Figure
- **Task**: Layout analysis — identify and classify semantic regions

### VOC2007
- **Type**: Chinese medical laboratory reports
- **Samples**: 238 (full dataset)
- **Complexity**: High — medical domain, Chinese characters, handwritten annotations, medical symbols, structured forms

### RX-PAD
- **Type**: French medical prescriptions
- **Samples**: 200 (150 training + 50 testing)
- **Complexity**: High — medical domain, French text, handwritten sections, structured forms, field-level extraction
- **Image Size**: 1,474 × 1,995 px average
- **Annotations**: Field-level bounding boxes and text labels for prescription parsing
