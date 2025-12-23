"""
OCR and VLM API calls for document analysis
"""

import os
from pathlib import Path
from typing import List

# All imports at the top
from PIL import Image
from langchain_core.documents import Document
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    AutoModel,
    AutoTokenizer,
)
import pdf2image


def get_settings():
    """Import settings from llm_settings."""
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import llms
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from llms.llm_settings import LLMSettings
    return LLMSettings()


def get_langchain_clients():
    """Get LangChain client helper functions from llm_settings."""
    import sys
    from pathlib import Path
    
    # Add parent directory to path to import llms
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from llms.llm_settings import (
        get_langchain_azure_openai,
        get_langchain_huggingface_pipeline,
        get_langchain_azure_document_intelligence_loader,
    )
    return {
        "get_azure_openai": get_langchain_azure_openai,
        "get_huggingface_pipeline": get_langchain_huggingface_pipeline,
        "get_document_intelligence_loader": get_langchain_azure_document_intelligence_loader,
    }


def call_ocr(file_path: str, model: str = "azure_intelligence") -> List[Document]:
    """
    Call OCR service on a document file.
    
    Args:
        file_path: Path to the document file (PDF, image, etc.)
        model: OCR model to use. Options: 
            - "azure_intelligence": Azure Document Intelligence
            - "mistral_ocr": Mistral OCR via Azure OpenAI
            - "mistral_ocr_3": Mistral OCR 3 (not yet available)
            - "donut": Donut (naver-clova-ix/donut-base) from Hugging Face
            - "deepseek_ocr": DeepSeek-OCR from Hugging Face
    
    Returns:
        List of LangChain Document objects with extracted text
    """
    if model == "azure_intelligence":
        clients = get_langchain_clients()
        loader = clients["get_document_intelligence_loader"](file_path)
        return loader.load()
    
    elif model == "mistral_ocr":
        # Mistral OCR via Azure OpenAI
        # TODO: Implement proper vision API call to Mistral OCR via Azure OpenAI
        # Requires azure-ai-openai with vision capabilities
        raise NotImplementedError("Mistral OCR via Azure OpenAI requires vision API implementation with proper file encoding")
    
    elif model == "mistral_ocr_3":
        # TODO: Implement when Mistral OCR 3 becomes available
        raise NotImplementedError("Mistral OCR 3 is not yet available. Please check Azure for updates.")
    
    elif model == "donut":
        # Donut: Document Understanding Transformer from Hugging Face
        # Handle PDF files by converting to images
        file_ext = Path(file_path).suffix.lower()
        if file_ext == ".pdf":
            # Convert first page of PDF to image
            images = pdf2image.convert_from_path(file_path, first_page=1, last_page=1)
            image = images[0]
        else:
            # Load image directly
            image = Image.open(file_path).convert("RGB")
        
        # Load processor and model from Hugging Face
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        model_obj = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        
        # Process image
        pixel_values = processor(image, return_tensors="pt").pixel_values
        
        # Generate OCR output
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        outputs = model_obj.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=768)
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Convert to LangChain Document format
        doc = Document(page_content=result, metadata={"source": file_path, "model": "donut"})
        return [doc]
    
    elif model == "deepseek_ocr":
        # DeepSeek-OCR from Hugging Face
        # Load image
        image = Image.open(file_path).convert("RGB")
        
        # Load model and tokenizer
        model_obj = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
        
        # Process image
        text = model_obj.ocr(image, tokenizer)
        
        # Convert to LangChain Document format
        doc = Document(page_content=text, metadata={"source": file_path, "model": "deepseek_ocr"})
        return [doc]
    
    else:
        raise ValueError(f"Unknown OCR model: {model}. Options: azure_intelligence, mistral_ocr, mistral_ocr_3, donut, deepseek_ocr")


def call_vlm(file_path: str, model: str = "gpt5_mini", query: str = None) -> str:
    """
    Call Vision Language Model (VLM) on a document file.
    
    Args:
        file_path: Path to the document file (PDF, image, etc.)
        model: VLM model to use. Options: "gpt5_mini", "gpt5_nano", "claude_sonnet", "claude_haiku", "qwen_vl"
        query: Optional query for the VLM (e.g., "Extract all text from this document")
    
    Returns:
        String output from the VLM
    """
    if model == "gpt5_mini":
        clients = get_langchain_clients()
        llm = clients["get_azure_openai"]("gpt-5-mini")
        # TODO: Implement image loading and vision capability
        raise NotImplementedError("GPT-5 mini integration not yet fully implemented")
    
    elif model == "claude_sonnet":
        # TODO: Implement Claude Sonnet integration
        raise NotImplementedError("Claude Sonnet integration not yet implemented")
    
    elif model == "claude_haiku":
        # TODO: Implement Claude Haiku integration
        raise NotImplementedError("Claude Haiku integration not yet implemented")
    
    elif model == "qwen_vl":
        # LangChain Hugging Face integration for Qwen-VL
        clients = get_langchain_clients()
        vlm = clients["get_huggingface_pipeline"]("Qwen/Qwen-VL-Chat")
        result = vlm.invoke(f"Question: {query if query else 'Extract all text'}")
        return result
    
    else:
        raise ValueError(f"Unknown VLM model: {model}. Options: gpt5_mini, gpt5_nano, claude_sonnet, claude_haiku, qwen_vl")
