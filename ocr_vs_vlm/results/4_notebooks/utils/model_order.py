"""
Model ordering configuration for consistent display across all notebooks.

This module defines the canonical ordering of models for visualizations and tables.
Models are ordered by provider group and capability level.
"""

from typing import List

# Canonical model ordering for all visualizations
# Order: OCR models (Azure, Mistral), then VLMs by size/capability (nano, mini, full)
MODEL_ORDER: List[str] = [
    # OCR Models
    'azure_intelligence',
    'mistral_document_ai',
    
    # VLM Models - Nano tier
    'gpt-5-nano',
    
    # VLM Models - Mini tier
    'gpt-5-mini',
    
    # VLM Models - Full/Advanced tier
    'claude_sonnet',
]

# Alternative names mapping (for data cleaning/normalization)
MODEL_ALIASES = {
    'azure_document_intelligence': 'azure_intelligence',
    'azure_di': 'azure_intelligence',
    'mistral': 'mistral_document_ai',
    'mistral_pixtral': 'mistral_document_ai',
    'gpt5nano': 'gpt-5-nano',
    'gpt5mini': 'gpt-5-mini',
    'claude': 'claude_sonnet',
    'claude-sonnet': 'claude_sonnet',
}


def sort_models(models: List[str]) -> List[str]:
    """
    Sort a list of models according to the canonical MODEL_ORDER.
    
    Models not in MODEL_ORDER will be appended at the end in alphabetical order.
    
    Args:
        models: List of model names to sort
        
    Returns:
        Sorted list of models (preserving original names, just reordered)
    """
    # Sort by position in MODEL_ORDER
    def sort_key(model: str) -> tuple:
        # Normalize for comparison but preserve original name
        normalized = MODEL_ALIASES.get(model, model)
        if normalized in MODEL_ORDER:
            return (0, MODEL_ORDER.index(normalized))
        else:
            return (1, model)  # Unknown models go last, alphabetically
    
    return sorted(models, key=sort_key)


def get_model_display_name(model: str) -> str:
    """
    Get a human-readable display name for a model.
    
    Args:
        model: Model identifier
        
    Returns:
        Display name
    """
    display_names = {
        'azure_intelligence': 'Azure Document Intelligence',
        'mistral_document_ai': 'Mistral Document AI',
        'gpt-5-nano': 'GPT-5 Nano',
        'gpt-5-mini': 'GPT-5 Mini',
        'claude_sonnet': 'Claude Sonnet',
    }
    return display_names.get(model, model)
