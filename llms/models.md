# Models

## OCR Models (Parsing)

### azure_intelligence
- **Provider**: Azure AI Document Intelligence
- **Type**: Enterprise OCR
- **Price**: $0.0015/page ($1.50/1000 pages)
- **Speed**: 500-2000ms/sample
- **Strengths**: Best form/table extraction, reliable, widely used

### mistral_document_ai
- **Provider**: Mistral (via Azure AI Foundry)
- **Type**: Document OCR
- **Price**: $0.0030/page ($3.00/1000 pages)
- **Speed**: 1000-3000ms/sample
- **Strengths**: Good general-purpose OCR, multilingual, layout handling

## VLM Models (Parsing + QA)

### gpt-5-mini
- **Provider**: OpenAI
- **Type**: Multimodal LLM
- **Price**: $0.25/1M input tokens, $2.00/1M output tokens (~$0.0024/page)
- **Image input**: ~2,500 tokens/page
- **Speed**: 2000-5000ms/sample (vision tasks)
- **Strengths**: Best accuracy, good balance of cost/performance

### gpt-5-nano
- **Provider**: OpenAI
- **Type**: Multimodal LLM (smallest GPT-5 variant)
- **Price**: $0.05/1M input tokens, $0.40/1M output tokens (~$0.0005/page)
- **Image input**: ~3,700 tokens/page
- **Speed**: 1500-4000ms/sample
- **Strengths**: Cost-effective, faster than mini, good balance

### claude_sonnet
- **Provider**: Anthropic Claude Sonnet 4
- **Type**: Multimodal LLM
- **Price**: $3.00/1M input tokens, $15.00/1M output tokens (~$0.005-0.01/page)
- **Speed**: 3000-8000ms/sample
- **Strengths**: Best reasoning, strong visual understanding, structured output
