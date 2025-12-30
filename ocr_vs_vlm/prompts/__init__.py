"""
Prompt management module for OCR vs VLM benchmark.

Handles generation and retrieval of prompts for different phases and datasets.
"""

from typing import Optional


def get_phase_2_prompt(dataset_name: str = '') -> str:
    """
    Get phase 2 prompt (VLM baseline with generic prompt).
    
    Args:
        dataset_name: Optional dataset name for language-specific prompts
    
    Returns:
        Generic VLM prompt for text extraction
    """
    # For Chinese datasets, specify Unicode output
    if dataset_name == 'VOC2007':
        return "Extract all text from this document image. Output text in Simplified Chinese Unicode characters (UTF-8)."
    
    return "Extract all text from this document image"


def get_phase_3_prompt(
    sample,
    dataset_name: str,
    phase_letter: Optional[str] = None
) -> str:
    """
    Get phase 3 prompt (VLM + context-aware prompt - intermediate detail).
    
    Args:
        sample: Sample object with metadata
        dataset_name: Dataset name (IAM, ICDAR, PubLayNet, VOC2007)
        phase_letter: Letter suffix for phase 3 variant (a, b, c, etc.)
    
    Returns:
        Context-aware prompt string
    """
    # Special handling for VOC2007 (Chinese medical lab reports)
    if dataset_name == 'VOC2007':
        return _get_voc2007_phase3_prompt(sample, phase_letter)
    
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


def get_phase_4_prompt(
    sample,
    dataset_name: str,
    phase_letter: Optional[str] = None
) -> str:
    """
    Get phase 4 prompt (VLM + detailed context-aware prompt).
    
    Args:
        sample: Sample object with metadata
        dataset_name: Dataset name (IAM, ICDAR, PubLayNet, VOC2007)
        phase_letter: Letter suffix for phase 4 variant (a, b, c, etc.)
    
    Returns:
        Detailed context-aware prompt string
    """
    # Special handling for VOC2007 (Chinese medical lab reports)
    if dataset_name == 'VOC2007':
        return _get_voc2007_phase4_prompt(sample, phase_letter)
    
    # For other datasets, use enhanced version of phase 3
    prompt_parts = [
        f"Dataset: {dataset_name}",
        f"Task: Extract all text from this document image with maximum accuracy",
        "Instructions:",
        "1. Preserve the exact structure and layout of the document",
        "2. Maintain all formatting including line breaks and spacing",
        "3. Extract every piece of text visible in the image"
    ]
    
    metadata = sample.metadata if hasattr(sample, 'metadata') else {}
    
    if 'languages' in metadata and metadata['languages']:
        langs = ", ".join(metadata['languages'])
        prompt_parts.append(f"Languages present: {langs}")
    
    if dataset_name == 'IAM':
        prompt_parts.extend([
            "Document type: Handwritten text",
            "Handle variations in writing style and potential ambiguous characters"
        ])
    elif dataset_name == 'ICDAR':
        prompt_parts.extend([
            "Document type: Multi-lingual scene text",
            "Preserve script types and reading directions"
        ])
    elif dataset_name == 'PubLayNet':
        prompt_parts.extend([
            "Document type: Structured document page",
            "Preserve document hierarchy and layout structure"
        ])
    
    prompt_parts.append("Return ONLY the extracted text.")
    
    return "\n".join(prompt_parts)


def _get_voc2007_phase3_prompt(sample, phase_letter: Optional[str] = None) -> str:
    """
    Get intermediate context-aware prompt for VOC2007 Chinese medical lab reports.
    
    Args:
        sample: Sample object with metadata
        phase_letter: Letter suffix for phase 3 variant
    
    Returns:
        Intermediate context-aware prompt for Chinese medical documents
    """
    prompt = """Extract all text from this Medical Laboratory Report.

Language: Simplified Chinese (简体中文)
Document Type: Medical Lab Report (医学检验报告)

Output text in Simplified Chinese Unicode characters (UTF-8).
Preserve the table structure and layout.
Return ONLY the extracted Chinese text."""

    return prompt


def _get_voc2007_phase4_prompt(sample, phase_letter: Optional[str] = None) -> str:
    """
    Get detailed context-aware prompt for VOC2007 Chinese medical lab reports.
    
    Args:
        sample: Sample object with metadata
        phase_letter: Letter suffix for phase 4 variant
    
    Returns:
        Detailed context-aware prompt for Chinese medical documents
    """
    prompt = """You are extracting text from a Medical Laboratory Report written in Simplified Chinese (简体中文).

Document Type: Medical Laboratory Test Report (医学检验报告)
Language: Simplified Chinese (简体中文)

Instructions:
1. Extract ALL text from the document image
2. Output text in Simplified Chinese Unicode characters (UTF-8)
3. Preserve the table structure and layout
4. Include headers, patient information, test names, results, units, and reference ranges
5. Maintain proper Chinese character encoding (do not use ASCII/Pinyin substitutions)

Common fields in these reports:
- 报告时间 (Report Time)
- 报告类型 (Report Type)
- 姓名 (Name)
- 性别 (Gender)
- 年龄 (Age)
- 项目名称 (Test Name)
- 结果 (Result)
- 单位 (Unit)
- 参考范围 (Reference Range)

Return ONLY the extracted Chinese text in Unicode format."""

    return prompt


def get_prompt(
    phase: int,
    sample=None,
    dataset_name: str = '',
    phase_letter: Optional[str] = None
) -> str:
    """
    Get prompt for a specific phase.
    
    Args:
        phase: Phase number (1, 2, 3, or 4)
        sample: Sample object (required for phase 3 and 4)
        dataset_name: Dataset name (required for phase 3 and 4)
        phase_letter: Letter suffix for phase variant
    
    Returns:
        Prompt string for the phase
    
    Raises:
        ValueError: If phase is invalid or required parameters missing
    """
    if phase == 1:
        raise ValueError("Phase 1 (OCR) does not use prompts")
    
    elif phase == 2:
        return get_phase_2_prompt(dataset_name)
    
    elif phase == 3:
        if sample is None:
            raise ValueError("Phase 3 requires sample parameter")
        if not dataset_name:
            raise ValueError("Phase 3 requires dataset_name parameter")
        return get_phase_3_prompt(sample, dataset_name, phase_letter)
    
    elif phase == 4:
        if sample is None:
            raise ValueError("Phase 4 requires sample parameter")
        if not dataset_name:
            raise ValueError("Phase 4 requires dataset_name parameter")
        return get_phase_4_prompt(sample, dataset_name, phase_letter)
    
    else:
        raise ValueError(f"Unknown phase: {phase}")
