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
- `uv run python sandbox/llm_call.py` - Run LLM script and save output to `sandbox/output/output.md`
- `uv run python sandbox/document_analysis.py` - Analyze PDF documents with Azure Document Intelligence
