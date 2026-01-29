"""
QA Prompts for DocVQA and InfographicVQA benchmarks.

Defines prompt templates for:
- Pipeline approach: OCR/VLM parsing + LLM QA
- Direct VQA approach: VLM sees image + question together

Phase naming convention:
- QA1a, QA1b, QA1c: OCR Pipeline + QA variations
- QA2a, QA2b: VLM Pipeline (varies parsing prompt, same QA prompt)
- QA3a, QA3b: Direct VQA variations
"""

from typing import Optional, List


# =============================================================================
# PIPELINE QA PROMPTS (Given extracted text, answer the question)
# =============================================================================

def get_qa_simple_prompt(question: str, extracted_text: str) -> str:
    """
    QA1a/QA2a: Simple QA prompt - minimal instructions.
    
    Args:
        question: The question to answer
        extracted_text: Text extracted from document via OCR/VLM
    
    Returns:
        Simple prompt for QA
    """
    return f"""Based on the following document text, answer the question.

Document text:
{extracted_text}

Question: {question}

Answer:"""


def get_qa_detailed_prompt(question: str, extracted_text: str, dataset_name: str = "DocVQA") -> str:
    """
    QA1b/QA2b: Detailed QA prompt - with specific instructions.
    
    Args:
        question: The question to answer
        extracted_text: Text extracted from document via OCR/VLM
        dataset_name: Dataset name for context-specific instructions
    
    Returns:
        Detailed prompt for QA
    """
    if dataset_name == "InfographicVQA":
        context = """This text was extracted from an infographic. Infographics often contain:
- Statistics and percentages
- Comparisons and rankings
- Data visualizations described in text
- Short phrases rather than complete sentences

When answering, consider that some information may require inference from multiple parts of the text."""
    else:
        context = """This text was extracted from a business/legal document. Documents may contain:
- Forms with labels and values
- Tables with headers and data
- Handwritten annotations
- Multiple sections with headers

Look for the specific information requested in the question."""

    return f"""You are a document understanding assistant. Answer the question based ONLY on the provided document text.

{context}

Document text:
{extracted_text}

Question: {question}

Instructions:
- Answer concisely with just the requested information
- If the answer is a number, include any units if present
- If the answer cannot be found in the text, respond with "NOT FOUND"
- Do not include explanations unless specifically asked

Answer:"""


def get_qa_cot_prompt(question: str, extracted_text: str, dataset_name: str = "DocVQA") -> str:
    """
    QA1c: Chain-of-thought QA prompt - reasoning before answering.
    
    Args:
        question: The question to answer
        extracted_text: Text extracted from document via OCR/VLM
        dataset_name: Dataset name for context
    
    Returns:
        CoT prompt for QA with reasoning
    """
    if dataset_name == "InfographicVQA":
        task_hint = "This is an infographic which may require finding and combining information from different parts."
    else:
        task_hint = "This is a document which may have structured information in tables, forms, or sections."

    return f"""You are analyzing a document to answer a question. {task_hint}

Document text:
{extracted_text}

Question: {question}

Let's solve this step by step:
1. First, identify what information the question is asking for
2. Search the document text for relevant information
3. Extract the specific answer

Reasoning:
[Your step-by-step reasoning here]

Final Answer:"""


# =============================================================================
# DIRECT VQA PROMPTS (VLM sees image directly)
# =============================================================================

def get_direct_vqa_simple_prompt(question: str) -> str:
    """
    QA3a: Simple direct VQA prompt.
    
    Args:
        question: The question to answer about the image
    
    Returns:
        Simple prompt for direct VQA
    """
    return f"""Look at this document image and answer the question.

Question: {question}

Answer:"""


def get_direct_vqa_detailed_prompt(question: str, dataset_name: str = "DocVQA") -> str:
    """
    QA3b: Detailed direct VQA prompt with instructions.
    
    Args:
        question: The question to answer about the image
        dataset_name: Dataset name for context
    
    Returns:
        Detailed prompt for direct VQA
    """
    if dataset_name == "InfographicVQA":
        context = """This is an infographic image containing data visualizations, statistics, and text.
Pay attention to:
- Charts, graphs, and visual data representations
- Numbers, percentages, and statistics
- Labels, titles, and annotations
- Relationships between different elements"""
    else:
        context = """This is a document image that may contain forms, tables, or text.
Pay attention to:
- Form fields and their values
- Table headers and data cells
- Handwritten text or annotations
- Document structure and sections"""

    return f"""{context}

Carefully examine the image to answer the following question.

Question: {question}

Instructions:
- Provide a concise, direct answer
- Include units or context if relevant
- If you cannot find the answer, respond with "NOT FOUND"

Answer:"""


# =============================================================================
# PARSING PROMPTS (For text extraction phase in pipeline approach)
# =============================================================================

def get_parsing_prompt_docvqa() -> str:
    """
    Parsing prompt for DocVQA documents.
    
    Returns:
        Prompt for extracting text from document images
    """
    return """Extract ALL text from this document image.

Instructions:
- Preserve the structure and layout of the document
- Include form labels and their values
- Include table headers and cell contents
- Include any handwritten text or annotations
- Maintain reading order (top to bottom, left to right)

Return ONLY the extracted text, no commentary."""


def get_parsing_prompt_infographicvqa() -> str:
    """
    Parsing prompt for InfographicVQA infographics.
    
    Returns:
        Prompt for extracting text from infographic images
    """
    return """Extract ALL text from this infographic image.

Instructions:
- Include all visible text: titles, labels, data values, annotations
- Preserve numerical data and statistics exactly as shown
- Include text from charts, graphs, and diagrams
- Capture any footnotes or source citations
- Read text from all areas: headers, body, legends, captions

Return ONLY the extracted text, no commentary."""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_pipeline_qa_prompt(
    question: str,
    extracted_text: str,
    variation: str,
    dataset_name: str = "DocVQA"
) -> str:
    """
    Get pipeline QA prompt based on variation.
    
    Args:
        question: The question to answer
        extracted_text: Text extracted from document
        variation: 'a' (simple), 'b' (detailed), or 'c' (cot)
        dataset_name: Dataset name for context
    
    Returns:
        Appropriate QA prompt
    """
    if variation == 'a':
        return get_qa_simple_prompt(question, extracted_text)
    elif variation == 'b':
        return get_qa_detailed_prompt(question, extracted_text, dataset_name)
    elif variation == 'c':
        return get_qa_cot_prompt(question, extracted_text, dataset_name)
    else:
        raise ValueError(f"Unknown variation: {variation}. Use 'a', 'b', or 'c'")


def get_direct_vqa_prompt(
    question: str,
    variation: str,
    dataset_name: str = "DocVQA"
) -> str:
    """
    Get direct VQA prompt based on variation.
    
    Args:
        question: The question to answer
        variation: 'a' (simple) or 'b' (detailed)
        dataset_name: Dataset name for context
    
    Returns:
        Appropriate direct VQA prompt
    """
    if variation == 'a':
        return get_direct_vqa_simple_prompt(question)
    elif variation == 'b':
        return get_direct_vqa_detailed_prompt(question, dataset_name)
    else:
        raise ValueError(f"Unknown variation: {variation}. Use 'a' or 'b'")


def get_parsing_prompt(dataset_name: str, variation: str = "a") -> str:
    """
    Get parsing prompt for dataset.
    
    Args:
        dataset_name: 'DocVQA' or 'InfographicVQA'
        variation: 'a' (simple extraction) or 'b' (detailed extraction)
    
    Returns:
        Appropriate parsing prompt
    """
    if variation == "b":
        # Detailed extraction prompt
        if dataset_name == "InfographicVQA":
            return """Extract ALL text and data from this infographic image with maximum detail.

Instructions:
- Include all visible text: titles, subtitles, labels, data values, annotations
- Preserve ALL numerical data and statistics exactly as shown
- Include text from charts, graphs, and diagrams
- Capture data series names and their values
- Include axis labels and scale values
- Capture any footnotes, sources, or citations
- Read text from all areas: headers, body, legends, captions
- For tables, preserve row and column structure
- Include any icons or symbols with text labels

Return ONLY the extracted text, no commentary."""
        else:
            return """Extract ALL text and data from this document image with maximum detail.

Instructions:
- Preserve the complete structure and layout of the document
- Include ALL form labels and their corresponding values
- Include ALL table headers and cell contents with structure
- Include any handwritten text or annotations
- Capture watermarks, stamps, or signatures if present
- Maintain reading order (top to bottom, left to right)
- For forms, use "Label: Value" format
- For tables, indicate row/column structure
- Include page numbers, headers, footers if present

Return ONLY the extracted text, no commentary."""
    else:
        # Simple extraction prompt (variation 'a')
        if dataset_name == "InfographicVQA":
            return get_parsing_prompt_infographicvqa()
        else:
            return get_parsing_prompt_docvqa()
