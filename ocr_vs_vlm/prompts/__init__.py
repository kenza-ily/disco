"""
Prompt management module for OCR vs VLM benchmark.

Handles generation and retrieval of prompts for different phases and datasets.
"""

from typing import Optional


def get_phase_2_prompt() -> str:
    """
    Get phase 2 prompt (VLM baseline with generic prompt).
    
    Returns:
        Generic VLM prompt for text extraction
    """
    return "Extract all text from this document image"


def get_phase_3_prompt(
    sample,
    dataset_name: str,
    phase_letter: Optional[str] = None
) -> str:
    """
    Get phase 3 prompt (VLM + context-aware prompt).
    
    Args:
        sample: Sample object with metadata
        dataset_name: Dataset name (IAM, ICDAR, PubLayNet)
        phase_letter: Letter suffix for phase 3 variant (a, b, c, etc.)
    
    Returns:
        Context-aware prompt string
    """
    prompt_parts = [
        f"Dataset: {dataset_name}",
        f"Task: Extract all text from this document image, preserving structure and layout"
    ]
    
    # Add dataset-specific context
    metadata = sample.metadata if hasattr(sample, 'metadata') else {}
    
    if 'languages' in metadata and metadata['languages']:
        langs = ", ".join(metadata['languages'])
        prompt_parts.append(f"Languages: {langs}")
    
    if 'num_text_lines' in metadata:
        prompt_parts.append(f"Expected approximately {metadata['num_text_lines']} text lines")
    
    if dataset_name == 'IAM':
        prompt_parts.append("This is handwritten text. Handle variations in writing style.")
    
    elif dataset_name == 'ICDAR':
        prompt_parts.append("This is multi-lingual scene text. Preserve script types and directions.")
    
    elif dataset_name == 'PubLayNet':
        prompt_parts.append("This is a document page. Preserve document structure and layout.")
    
    prompt_parts.append("Return ONLY the extracted text.")
    
    return "\n".join(prompt_parts)


def get_prompt(
    phase: int,
    sample=None,
    dataset_name: str = '',
    phase_letter: Optional[str] = None
) -> str:
    """
    Get prompt for a specific phase.
    
    Args:
        phase: Phase number (1, 2, or 3)
        sample: Sample object (required for phase 3)
        dataset_name: Dataset name (required for phase 3)
        phase_letter: Letter suffix for phase 3 variant
    
    Returns:
        Prompt string for the phase
    
    Raises:
        ValueError: If phase is invalid or required parameters missing
    """
    if phase == 1:
        raise ValueError("Phase 1 (OCR) does not use prompts")
    
    elif phase == 2:
        return get_phase_2_prompt()
    
    elif phase == 3:
        if sample is None:
            raise ValueError("Phase 3 requires sample parameter")
        if not dataset_name:
            raise ValueError("Phase 3 requires dataset_name parameter")
        return get_phase_3_prompt(sample, dataset_name, phase_letter)
    
    else:
        raise ValueError(f"Unknown phase: {phase}")
