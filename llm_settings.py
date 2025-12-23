from pydantic_settings import BaseSettings
from openai import AzureOpenAI, AsyncAzureOpenAI
from typing import Any, TypeVar, Union
from pathlib import Path
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

T = TypeVar('T')


class LLMModel:
    def __init__(self, name: str, reasoning_effort: str = None):
        self.name = name
        self.reasoning_effort = reasoning_effort


class LLMSettings(BaseSettings):
    azure_api_version: str = "2024-02-01"
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_document_intelligence_endpoint: str
    azure_document_intelligence_key: str
    max_retries_openai: int = 3

    model_config = {
        "extra": "ignore",
        "env_file": ".env.local"
    }


def get_settings() -> LLMSettings:
    return LLMSettings()


def get_azure_openai_client() -> AzureOpenAI:
    settings = get_settings()
    return AzureOpenAI(
        api_version=settings.azure_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        max_retries=settings.max_retries_openai,
    )


def get_async_azure_openai_client() -> AsyncAzureOpenAI:
    settings = get_settings()
    return AsyncAzureOpenAI(
        api_version=settings.azure_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        max_retries=settings.max_retries_openai,
    )


def get_azure_document_intelligence_client() -> DocumentIntelligenceClient:
    settings = get_settings()
    credential = AzureKeyCredential(settings.azure_document_intelligence_key)
    return DocumentIntelligenceClient(
        endpoint=settings.azure_document_intelligence_endpoint,
        credential=credential
    )


async def azure_openai_structured_response(
    system_prompt: str,
    user_input: Union[str, list[dict[str, Any]]],
    pydantic_model: type[T],
    model: LLMModel,
    pdf_path: Path | None = None,
) -> T:
    """Handle Azure OpenAI structured response using beta chat completions parse API."""
    client = get_async_azure_openai_client()

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    if isinstance(user_input, str):
        messages.append({"role": "user", "content": user_input})
    else:
        messages.extend(user_input)

    # Use the beta chat completions parse API for structured responses
    response = await client.beta.chat.completions.parse(
        model=model.name,
        messages=messages,
        response_format=pydantic_model,
    )

    if not response.choices or not response.choices[0].message.parsed:
        raise ValueError("Failed to get a valid structured response from the model")

    return response.choices[0].message.parsed