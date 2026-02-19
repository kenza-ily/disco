"""
Unified settings and credential management for all models.

Credentials are loaded from .env.local via pydantic-settings.
All credentials are optional - only require what you need for your chosen providers.
"""

from pydantic_settings import BaseSettings
from openai import AzureOpenAI
from pathlib import Path
import boto3


class Settings(BaseSettings):
    """Configuration for all model providers (Azure, Anthropic, AWS, Mistral)."""

    # Azure OpenAI + Document Intelligence
    azure_api_version: str = "2024-02-01"
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_document_intelligence_endpoint: str | None = None
    azure_document_intelligence_key: str | None = None
    max_retries_openai: int = 3

    # Anthropic Direct API (NEW)
    anthropic_api_key: str | None = None

    # AWS Bedrock (alternative to direct Anthropic API)
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"
    aws_profile: str | None = None

    # Mistral API
    mistral_api_key: str | None = None

    model_config = {
        "extra": "ignore",
        "env_file": str(Path(__file__).parent.parent / ".env.local")
    }


def get_settings() -> Settings:
    """Get settings instance from .env.local."""
    return Settings()


def get_azure_openai_client() -> AzureOpenAI:
    """Get Azure OpenAI client for direct API calls."""
    settings = get_settings()

    if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
        raise ValueError(
            "Azure OpenAI credentials not configured. "
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env.local"
        )

    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
    )


def get_bedrock_client(profile_name: str | None = None):
    """Get AWS Bedrock Runtime client for Claude models.

    Args:
        profile_name: AWS profile name (e.g., 'eu-dev', 'eu-prod').
                     If None, uses default credential chain.
    """
    settings = get_settings()

    client_kwargs = {
        'service_name': 'bedrock-runtime',
        'region_name': settings.aws_region
    }

    # Priority: profile_name > explicit credentials > default credential chain
    if profile_name:
        session = boto3.Session(profile_name=profile_name)
        return session.client(**client_kwargs)
    elif settings.aws_access_key_id and settings.aws_secret_access_key:
        client_kwargs['aws_access_key_id'] = settings.aws_access_key_id
        client_kwargs['aws_secret_access_key'] = settings.aws_secret_access_key

    return boto3.client(**client_kwargs)
