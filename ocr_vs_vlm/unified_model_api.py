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
import re
from io import BytesIO
from pathlib import Path
from typing import Optional, List
from enum import Enum
import json
import requests

# Load environment variables from .env.local if it exists
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env.local")

from pydantic import BaseModel, Field
from PIL import Image
from openai import AzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
import pdf2image

# Lazy imports for optional dependencies
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None

try:
    from transformers import (
        DonutProcessor,
        VisionEncoderDecoderModel,
        AutoModel,
        AutoTokenizer,
        AutoProcessor,
        Qwen2VLForConditionalGeneration,
    )
except ImportError:
    DonutProcessor = None
    VisionEncoderDecoderModel = None
    AutoModel = None
    AutoTokenizer = None
    AutoProcessor = None
    Qwen2VLForConditionalGeneration = None

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
        "claude_sonnet": {
            "type": ModelType.VLM,
            "requires": [],
            "description": "Claude Sonnet 4 - High-accuracy vision LLM via AWS Bedrock"
        },
        "claude_haiku": {
            "type": ModelType.VLM,
            "requires": [],
            "description": "Claude Haiku 4 - Fast vision LLM via AWS Bedrock"
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
            return self._vlm_claude(image_path, "us.anthropic.claude-3-5-sonnet-20241022-v2:0", query)
        elif model == "claude_haiku":
            return self._vlm_claude(image_path, "us.anthropic.claude-3-5-haiku-20241022-v2:0", query)
        elif model == "qwen_vl":
            return self._vlm_qwen(image_path, query)
        else:
            raise ValueError(f"Unknown VLM model: {model}")
    
    # ========== OCR IMPLEMENTATIONS ==========
    
    def _ocr_azure_intelligence(self, image_path: str) -> ModelResponse:
        """Azure Document Intelligence OCR."""
        with open(image_path, "rb") as f:
            document_data = f.read()
        
        # Call with body parameter using the correct API signature
        poller = self.document_intelligence_client.begin_analyze_document(
            model_id="prebuilt-read",
            body=document_data,
            content_type="application/octet-stream"
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
        """Mistral Document AI via direct API endpoint."""
        try:
            # Load and encode image to base64
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            image_data = base64.b64encode(image_bytes).decode("utf-8")
            
            # Detect mime type from file extension
            suffix = Path(image_path).suffix.lower()
            mime_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".pdf": "application/pdf",
            }
            mime_type = mime_type_map.get(suffix, "image/jpeg")
            
            # Create data URL for the document
            document_url = f"data:{mime_type};base64,{image_data}"
            
            # Call Mistral OCR API via Azure endpoint
            endpoint = self.config.get("azure_openai_endpoint")
            api_key = self.config.get("azure_openai_api_key")
            
            if not endpoint or not api_key:
                raise ValueError("Azure endpoint or API key not configured")
            
            # Remove trailing slash from endpoint to avoid double slash
            endpoint = endpoint.rstrip("/")
            endpoint_url = f"{endpoint}/providers/mistral/azure/ocr"
            
            headers = {
                "Content-Type": "application/json",
                "api-key": api_key.strip(),
            }
            
            payload = {
                "model": "mistral-document-ai-2505",
                "document": {
                    "type": "document_url",
                    "document_url": document_url
                }
            }
            
            logger.info(f"Calling Mistral OCR API: {endpoint_url}")
            response = requests.post(
                endpoint_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"API response: {response.status_code} - {response.text[:500]}")
            
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Mistral API response keys: {result.keys()}")
            
            # Extract text from the response - Mistral returns markdown in pages
            extracted_text = ""
            if "pages" in result:
                for page in result["pages"]:
                    if "markdown" in page:
                        extracted_text += page["markdown"] + "\n"
            elif "content" in result:
                extracted_text = result["content"]
            elif "text" in result:
                extracted_text = result["text"]
            
            return ModelResponse(
                model_name="mistral_document_ai",
                model_type=ModelType.OCR,
                content=extracted_text.strip(),
                source=image_path,
            )
        
        except Exception as e:
            logger.error(f"Mistral OCR API call failed: {e}")
            raise
    
    def _ocr_donut(self, image_path: str) -> ModelResponse:
        """Donut OCR from Hugging Face with GPU support."""
        if Accelerator is None:
            raise ImportError("Donut requires 'accelerate' package. Install with: pip install accelerate")
        if DonutProcessor is None or VisionEncoderDecoderModel is None:
            raise ImportError("Donut requires 'transformers' package. Install with: pip install transformers")
        
        # Load image
        logger.info(f"📷 Loading image: {image_path}")
        file_ext = Path(image_path).suffix.lower()
        if file_ext == ".pdf":
            images = pdf2image.convert_from_path(image_path, first_page=1, last_page=1)
            image = images[0]
        else:
            image = Image.open(image_path).convert("RGB")
        logger.info(f"✓ Image loaded")
        
        # Load model (cache these to avoid reloading)
        if not hasattr(self, '_donut_processor'):
            logger.info(f"📥 Loading Donut processor...")
            self._donut_processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
            logger.info(f"✓ Processor loaded")
        
        if not hasattr(self, '_donut_model'):
            logger.info(f"📥 Loading Donut model...")
            self._donut_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
            logger.info(f"✓ Model loaded")
        
        processor = self._donut_processor
        model = self._donut_model
        
        # Get device from Accelerator (handles MacBook GPU automatically)
        if not hasattr(self, '_donut_device'):
            logger.info(f"🚀 Detecting device (GPU/CPU)...")
            self._donut_device = Accelerator().device
            logger.info(f"✓ Using device: {self._donut_device}")
            model.to(self._donut_device)
        
        device = self._donut_device
        
        # Process image
        logger.info(f"🔄 Processing image...")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        
        # Prepare decoder inputs
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        
        # Generate with device handling
        logger.info(f"🤖 Running inference on {device}...")
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        
        # Process output
        logger.info(f"📝 Decoding output...")
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        logger.info(f"✓ Extraction complete")
        
        return ModelResponse(
            model_name="donut",
            model_type=ModelType.OCR,
            content=sequence,
            source=image_path,
        )
    
    def _ocr_deepseek(self, image_path: str) -> ModelResponse:
        """DeepSeek-OCR from Hugging Face."""
        if AutoModel is None or AutoTokenizer is None:
            raise ImportError("DeepSeek-OCR requires 'transformers' package. Install with: pip install transformers")
        
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
            max_completion_tokens=16000  # Increased for complex infographics with lots of text
        )

        # Check if content is empty and log the reason
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason

        if not content and finish_reason:
            logger.warning(f"Empty content from {model}. Finish reason: {finish_reason}")
            if finish_reason == "content_filter":
                logger.warning(f"Content filtered for image: {image_path}")

        return ModelResponse(
            model_name=model,
            model_type=ModelType.VLM,
            content=content,
            source=image_path,
            query=query,
            tokens_used=response.usage.total_tokens if response.usage else 0,
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
            max_tokens=4000  # Mistral can handle extended responses
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
        """Claude vision LLM via AWS Bedrock."""
        try:
            import boto3
            import json
        except ImportError:
            raise ImportError("Claude via Bedrock requires 'boto3'. Install with: pip install boto3")
        
        # Load image and encode
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        # Detect media type
        suffix = Path(image_path).suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".gif": "image/gif", ".webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")
        
        if not query:
            query = "Extract all text from this document image"
        
        # Use AWS Bedrock client
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        
        # Prepare message for Claude
        message_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            },
            {"type": "text", "text": query}
        ]
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
        })
        
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        response_body = json.loads(response["body"].read().decode("utf-8"))
        
        return ModelResponse(
            model_name="claude_sonnet" if "sonnet" in model_id else "claude_haiku",
            model_type=ModelType.VLM,
            content=response_body["content"][0]["text"],
            source=image_path,
            query=query,
            tokens_used=response_body["usage"]["input_tokens"] + response_body["usage"]["output_tokens"],
        )
    
    def _vlm_qwen(self, image_path: str, query: Optional[str]) -> ModelResponse:
        """Qwen-VL open vision LLM."""
        if AutoProcessor is None or Qwen2VLForConditionalGeneration is None:
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
        
        generated_ids = model.generate(**inputs, max_new_tokens=2000)  # Allow extended outputs
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return ModelResponse(
            model_name="qwen_vl",
            model_type=ModelType.VLM,
            content=generated_text,
            source=image_path,
            query=query,
        )
