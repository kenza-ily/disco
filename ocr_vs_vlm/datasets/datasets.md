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
