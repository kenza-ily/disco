# research-playground
Playground to explore PoCs and research ideas

## Setup
1. `aws-login`
2. `codeartifact-login`
3. `make setup-env`
4. `uv sync`

## LLM Settings
- Synchronous and asynchronous Azure OpenAI clients for flexible API calls
- Structured responses using Pydantic models for type-safe LLM outputs
- Environment-based configuration loaded from `.env.local` file
- Azure Document Intelligence client for OCR and document processing

## Usage

### Authentication Setup
Before running any scripts, ensure you have valid AWS credentials:

```bash
aws-login
codeartifact-login
make setup-env
uv sync
```

### LLM Script
Run a simple LLM call using Azure OpenAI:

```bash
uv run python sandbox/llm_call.py
```

**Output**: Saves response to `sandbox/output/output.md`

### Document Analysis
Analyze PDF documents using Azure Document Intelligence:

```bash
# Analyze a PDF in the datasets folder
uv run python sandbox/document_analysis.py "datasets/your-document.pdf"

# Or use an absolute path
uv run python sandbox/document_analysis.py "/full/path/to/document.pdf"
```

**Outputs**:
- `sandbox/output/document_analysis.json` - Detailed structured data with text and bounding boxes
- `sandbox/output/document_text.txt` - Clean extracted text organized by page

**Example**:
```bash
# Healthcare eBook analysis
uv run python sandbox/document_analysis.py "datasets/Generative AI in Healthcare eBook.pdf"

