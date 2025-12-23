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
    huggingface_api_key: str = ""  # Optional for Hugging Face API

    model_config = {
        "extra": "ignore",
        "env_file": ".env.local"
    }


def get_settings() -> LLMSettings:
    return LLMSettings()