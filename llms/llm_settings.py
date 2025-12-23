from pydantic_settings import BaseSettings
from openai import AzureOpenAI, AsyncAzureOpenAI
from typing import Any, TypeVar, Union
from pathlib import Path
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from transformers import pipeline
import os

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
        "env_file": str(Path(__file__).parent.parent / ".env.local")
    }


def get_settings() -> LLMSettings:
    return LLMSettings()


def get_langchain_azure_openai(model_name: str = "gpt-4") -> AzureChatOpenAI:
    """Get LangChain Azure OpenAI client."""
    settings = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_deployment=model_name,
        api_version=settings.azure_api_version,
        api_key=settings.azure_openai_api_key,
        max_retries=settings.max_retries_openai,
    )


def get_langchain_huggingface_pipeline(model_name: str = "google/gemma-3-27b-it") -> HuggingFacePipeline:
    """Get LangChain Hugging Face pipeline."""
    settings = get_settings()
    token = settings.huggingface_api_key if settings.huggingface_api_key else None
    
    # Create a text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name,
        token=token,
        max_length=100,
        do_sample=True,
        temperature=0.7,
    )
    
    return HuggingFacePipeline(pipeline=pipe)


def get_langchain_azure_document_intelligence_loader(file_path: str) -> AzureAIDocumentIntelligenceLoader:
    """Get LangChain Azure Document Intelligence loader."""
    settings = get_settings()
    return AzureAIDocumentIntelligenceLoader(
        api_endpoint=settings.azure_document_intelligence_endpoint,
        api_key=settings.azure_document_intelligence_key,
        file_path=file_path,
        api_model="prebuilt-read",
    )
