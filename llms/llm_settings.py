"""
Unified settings and credential management for LLM/OCR models.

Credentials are loaded from .env.local via pydantic-settings.
Direct API clients are used - no LangChain wrappers.
"""

from pydantic_settings import BaseSettings
from openai import AzureOpenAI
from pathlib import Path


class LLMSettings(BaseSettings):
    """Configuration for Azure OpenAI and Document Intelligence."""

    azure_api_version: str = "2024-02-01"
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_document_intelligence_endpoint: str
    azure_document_intelligence_key: str
    max_retries_openai: int = 3

    model_config = {
        "extra": "ignore",
        "env_file": str(Path(__file__).parent.parent / ".env.local")
    }


def get_settings() -> LLMSettings:
    """Get settings instance from .env.local."""
    return LLMSettings()


def get_azure_openai_client() -> AzureOpenAI:
    """Get Azure OpenAI client for direct API calls."""
    settings = get_settings()
    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )
