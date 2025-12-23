"""
Unified Model API for OCR and VLM document analysis.

This module provides a single, clean interface for all document processing models:
- OCR models: Azure Document Intelligence, Donut, DeepSeek-OCR, Mistral Document AI
- VLM models: GPT-5 (mini/nano), Claude (Sonnet/Haiku), Qwen-VL, Mistral models

All responses use Pydantic for consistent output formatting.
"""

import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Optional, List
from enum import Enum

# Load environment variables from .env.local if it exists
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env.local")

from pydantic import BaseModel, Field
from PIL import Image
from openai import AzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import pdf2image

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Model type classification."""
    OCR = "ocr"
    VLM = "vlm"


class ModelResponse(BaseModel):
    """Unified response format for all models."""
    
    model_name: str = Field(..., description="Name of the model used")
    model_type: ModelType = Field(..., description="Type: OCR or VLM")
    content: str = Field(..., description="Main output: extracted text or model response")
    source: str = Field(..., description="Original image path")
    query: Optional[str] = Field(None, description="Query sent to VLM (VLMs only)")
    tokens_used: Optional[int] = Field(None, description="Token count if available")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "gpt-5-mini",
                "model_type": "vlm",
                "content": "Extracted text from document",
                "source": "/path/to/image.jpg",
                "query": "Extract all text",
                "tokens_used": 150,
                "error": None
            }
        }


class ModelRegistry:
    """Registry of available models and their metadata."""
    
    # OCR Models
    OCR_MODELS = {
        "azure_intelligence": {
            "type": ModelType.OCR,
            "requires": ["azure_document_intelligence_endpoint", "azure_document_intelligence_key"],
            "description": "Azure Document Intelligence - Enterprise OCR"
        },
        "mistral_document_ai": {
            "type": ModelType.OCR,
            "requires": ["azure_openai_endpoint", "azure_openai_api_key", "azure_api_version"],
            "description": "Mistral Document AI 2505 - Advanced document parsing"
        },
        "donut": {
            "type": ModelType.OCR,
            "requires": [],
            "description": "Donut - Open source document understanding"
        },
        "deepseek_ocr": {
            "type": ModelType.OCR,
            "requires": [],
            "description": "DeepSeek-OCR - Open source multilingual OCR"
        },
    }
    
    # VLM Models
    VLM_MODELS = {
        "gpt-5-mini": {
            "type": ModelType.VLM,
            "requires": ["azure_openai_endpoint", "azure_openai_api_key", "azure_api_version"],
            "description": "GPT-5 mini - Fast vision LLM"
        },
        "gpt-5-nano": {
            "type": ModelType.VLM,
            "requires": ["azure_openai_endpoint", "azure_openai_api_key", "azure_api_version"],
            "description": "GPT-5 nano - Ultra-fast vision LLM"
        },
        "mistral": {
            "type": ModelType.VLM,
            "requires": ["azure_openai_endpoint", "azure_openai_api_key", "azure_api_version"],
            "description": "Mistral - Open vision LLM"
        },
        "claude_sonnet": {
            "type": ModelType.VLM,
            "requires": ["anthropic_api_key"],
            "description": "Claude Sonnet 4.5 - High-accuracy vision LLM"
        },
        "claude_haiku": {
            "type": ModelType.VLM,
            "requires": ["anthropic_api_key"],
            "description": "Claude Haiku 4.5 - Fast vision LLM"
        },
        "qwen_vl": {
            "type": ModelType.VLM,
            "requires": [],
            "description": "Qwen-VL - Open vision LLM"
        },
    }
    
    @classmethod
    def get_model_type(cls, model_name: str) -> ModelType:
        """Get type of model."""
        if model_name in cls.OCR_MODELS:
            return ModelType.OCR
        elif model_name in cls.VLM_MODELS:
            return ModelType.VLM
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    @classmethod
    def list_ocr_models(cls) -> List[str]:
        """List all available OCR models."""
        return list(cls.OCR_MODELS.keys())
    
    @classmethod
    def list_vlm_models(cls) -> List[str]:
        """List all available VLM models."""
        return list(cls.VLM_MODELS.keys())
    
    @classmethod
    def list_all_models(cls) -> List[str]:
        """List all available models."""
        return cls.list_ocr_models() + cls.list_vlm_models()


class UnifiedModelAPI:
    """
    Unified API for all document processing models.
    
    Usage:
        api = UnifiedModelAPI()
        
        # OCR
        response = api.process("path/to/image.jpg", model="azure_intelligence")
        print(response.content)  # Extracted text
        
        # VLM
        response = api.process(
            "path/to/image.jpg",
            model="gpt-5-mini",
            query="What is the title of this document?"
        )
        print(response.content)  # Model response
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize API with optional configuration.
        
        Args:
            config: Dict with keys like 'azure_openai_api_key', 'azure_openai_endpoint', etc.
                   If None, will load from environment variables.
        """
        self.config = config or self._load_env_config()
        self._azure_openai_client = None
        self._document_intelligence_client = None
    
    def _load_env_config(self) -> dict:
        """Load configuration from environment variables."""
        import os
        return {
            "azure_openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "azure_api_version": os.getenv("AZURE_API_VERSION", "2024-02-01"),
            "azure_document_intelligence_endpoint": os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
            "azure_document_intelligence_key": os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        }
    
    @property
    def azure_openai_client(self) -> AzureOpenAI:
        """Lazy-load Azure OpenAI client."""
        if self._azure_openai_client is None:
            self._azure_openai_client = AzureOpenAI(
                api_key=self.config.get("azure_openai_api_key"),
                api_version=self.config.get("azure_api_version", "2024-02-01"),
                azure_endpoint=self.config.get("azure_openai_endpoint"),
            )
        return self._azure_openai_client
    
    @property
    def document_intelligence_client(self) -> DocumentIntelligenceClient:
        """Lazy-load Azure Document Intelligence client."""
        if self._document_intelligence_client is None:
            credential = AzureKeyCredential(
                self.config.get("azure_document_intelligence_key")
            )
            self._document_intelligence_client = DocumentIntelligenceClient(
                endpoint=self.config.get("azure_document_intelligence_endpoint"),
                credential=credential
            )
        return self._document_intelligence_client
    
    def process(
        self,
        image_path: str,
        model: str,
        query: Optional[str] = None,
    ) -> ModelResponse:
        """
        Process document image with specified model.
        
        Args:
            image_path: Path to image file
            model: Model name from ModelRegistry
            query: Query for VLM models (ignored for OCR)
        
        Returns:
            ModelResponse with consistent output format
        """
        try:
            model_type = ModelRegistry.get_model_type(model)
            
            if model_type == ModelType.OCR:
                return self._process_ocr(image_path, model)
            else:  # VLM
                return self._process_vlm(image_path, model, query)
        
        except Exception as e:
            logger.error(f"Error processing {image_path} with {model}: {e}")
            return ModelResponse(
                model_name=model,
                model_type=ModelRegistry.get_model_type(model),
                content="",
                source=image_path,
                error=str(e)
            )
    
    def _process_ocr(self, image_path: str, model: str) -> ModelResponse:
        """Process image with OCR model."""
        logger.info(f"OCR: {model} on {image_path}")
        
        if model == "azure_intelligence":
            return self._ocr_azure_intelligence(image_path)
        elif model == "mistral_document_ai":
            return self._ocr_mistral_document_ai(image_path)
        elif model == "donut":
            return self._ocr_donut(image_path)
        elif model == "deepseek_ocr":
            return self._ocr_deepseek(image_path)
        else:
            raise ValueError(f"Unknown OCR model: {model}")
    
    def _process_vlm(self, image_path: str, model: str, query: Optional[str]) -> ModelResponse:
        """Process image with VLM model."""
        logger.info(f"VLM: {model} on {image_path}")
        
        if model in ["gpt-5-mini", "gpt-5-nano"]:
            return self._vlm_gpt5(image_path, model, query)
        elif model == "mistral":
            return self._vlm_mistral(image_path, query)
        elif model == "claude_sonnet":
            return self._vlm_claude(image_path, "claude-3-5-sonnet-20241022", query)
        elif model == "claude_haiku":
            return self._vlm_claude(image_path, "claude-3-5-haiku-20241022", query)
        elif model == "qwen_vl":
            return self._vlm_qwen(image_path, query)
        else:
            raise ValueError(f"Unknown VLM model: {model}")
    
    # ========== OCR IMPLEMENTATIONS ==========
    
    def _ocr_azure_intelligence(self, image_path: str) -> ModelResponse:
        """Azure Document Intelligence OCR."""
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
        
        with open(image_path, "rb") as f:
            document_data = f.read()
        
        poller = self.document_intelligence_client.begin_analyze_document(
            "prebuilt-read",
            document=document_data,
            content_type="image/jpeg"
        )
        result = poller.result()
        
        text_lines = []
        for page in result.pages:
            for line in page.lines:
                text_lines.append(line.content)
        
        return ModelResponse(
            model_name="azure_intelligence",
            model_type=ModelType.OCR,
            content="\n".join(text_lines),
            source=image_path,
        )
    
    def _ocr_mistral_document_ai(self, image_path: str) -> ModelResponse:
        """Mistral Document AI via Azure OpenAI."""
        # Load and encode image
        image = Image.open(image_path).convert("RGB")
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        
        # Call Mistral Document AI
        response = self.azure_openai_client.chat.completions.create(
            model="mistral-document-ai-2505",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this document image. Return ONLY the extracted text."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000
        )
        
        return ModelResponse(
            model_name="mistral_document_ai",
            model_type=ModelType.OCR,
            content=response.choices[0].message.content,
            source=image_path,
            tokens_used=response.usage.total_tokens,
        )
    
    def _ocr_donut(self, image_path: str) -> ModelResponse:
        """Donut OCR from Hugging Face."""
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        
        # Load image
        file_ext = Path(image_path).suffix.lower()
        if file_ext == ".pdf":
            images = pdf2image.convert_from_path(image_path, first_page=1, last_page=1)
            image = images[0]
        else:
            image = Image.open(image_path).convert("RGB")
        
        # Load model
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        
        # Process
        pixel_values = processor(image, return_tensors="pt").pixel_values
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        outputs = model.generate(
            pixel_values, decoder_input_ids=decoder_input_ids, max_length=768
        )
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return ModelResponse(
            model_name="donut",
            model_type=ModelType.OCR,
            content=result,
            source=image_path,
        )
    
    def _ocr_deepseek(self, image_path: str) -> ModelResponse:
        """DeepSeek-OCR from Hugging Face."""
        from transformers import AutoModel, AutoTokenizer
        
        image = Image.open(image_path).convert("RGB")
        
        model = AutoModel.from_pretrained(
            "deepseek-ai/DeepSeek-OCR", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-OCR", trust_remote_code=True
        )
        
        text = model.ocr(image, tokenizer)
        
        return ModelResponse(
            model_name="deepseek_ocr",
            model_type=ModelType.OCR,
            content=text,
            source=image_path,
        )
    
    # ========== VLM IMPLEMENTATIONS ==========
    
    def _vlm_gpt5(self, image_path: str, model: str, query: Optional[str]) -> ModelResponse:
        """GPT-5 mini/nano vision LLM."""
        # Load and encode image
        image = Image.open(image_path).convert("RGB")
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        
        # Prepare prompt
        if not query:
            query = "Extract all text from this document image"
        
        # Call API
        response = self.azure_openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        }
                    ]
                }
            ],
            max_completion_tokens=2000
        )
        
        return ModelResponse(
            model_name=model,
            model_type=ModelType.VLM,
            content=response.choices[0].message.content,
            source=image_path,
            query=query,
            tokens_used=response.usage.total_tokens,
        )
    
    def _vlm_mistral(self, image_path: str, query: Optional[str]) -> ModelResponse:
        """Mistral vision LLM."""
        # Load and encode image
        image = Image.open(image_path).convert("RGB")
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")
        
        if not query:
            query = "Extract all text from this document image"
        
        response = self.azure_openai_client.chat.completions.create(
            model="mistral",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        
        return ModelResponse(
            model_name="mistral",
            model_type=ModelType.VLM,
            content=response.choices[0].message.content,
            source=image_path,
            query=query,
            tokens_used=response.usage.total_tokens,
        )
    
    def _vlm_claude(self, image_path: str, model_id: str, query: Optional[str]) -> ModelResponse:
        """Claude vision LLM."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("Claude requires 'anthropic' package. Install with: pip install anthropic")
        
        client = anthropic.Anthropic(api_key=self.config.get("anthropic_api_key"))
        
        # Load image
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        # Detect media type
        suffix = Path(image_path).suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")
        
        if not query:
            query = "Extract all text from this document image"
        
        response = client.messages.create(
            model=model_id,
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": query}
                    ],
                }
            ],
        )
        
        return ModelResponse(
            model_name="claude_sonnet" if "sonnet" in model_id else "claude_haiku",
            model_type=ModelType.VLM,
            content=response.content[0].text,
            source=image_path,
            query=query,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        )
    
    def _vlm_qwen(self, image_path: str, query: Optional[str]) -> ModelResponse:
        """Qwen-VL open vision LLM."""
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        except ImportError:
            raise ImportError("Qwen-VL requires transformers. Install with: pip install transformers")
        
        if not query:
            query = "Extract all text from this document image"
        
        # Load model and processor
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        
        # Load image
        image = Image.open(image_path)
        
        # Prepare input
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            }
        ]
        
        # Process and generate
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = processor.process_images([image])
        inputs = processor(text=text, images=image_inputs, videos=video_inputs, return_tensors="pt")
        
        generated_ids = model.generate(**inputs, max_new_tokens=1000)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return ModelResponse(
            model_name="qwen_vl",
            model_type=ModelType.VLM,
            content=generated_text,
            source=image_path,
            query=query,
        )
