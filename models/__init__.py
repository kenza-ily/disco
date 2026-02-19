"""
Models package - Unified API for OCR and VLM models.

Exports:
    - UnifiedModelAPI: Main interface for all models
    - ModelRegistry: Registry of available models
    - ModelResponse: Standard response format
    - ModelType: Model type enum (OCR/VLM)
    - Settings: Configuration from .env.local
"""

from .unified_api import UnifiedModelAPI, ModelRegistry, ModelResponse, ModelType
from .settings import Settings, get_settings, get_azure_openai_client, get_bedrock_client

__all__ = [
    "UnifiedModelAPI",
    "ModelRegistry",
    "ModelResponse",
    "ModelType",
    "Settings",
    "get_settings",
    "get_azure_openai_client",
    "get_bedrock_client",
]
