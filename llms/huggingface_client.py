from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from .llm_settings import get_settings


def load_huggingface_model(model_name: str, trust_remote_code: bool = True, dtype: str = "auto"):
    """Load a model directly from Hugging Face."""
    settings = get_settings()
    token = settings.huggingface_api_key if settings.huggingface_api_key else None
    return AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code, dtype=dtype, token=token)


def load_text_generation_model(model_name: str, trust_remote_code: bool = True):
    """Load tokenizer and model for text generation."""
    settings = get_settings()
    token = settings.huggingface_api_key if settings.huggingface_api_key else None
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code, token=token)
    return tokenizer, model


def generate_text(prompt: str, model_name: str = "google/gemma-3-27b-it", max_length: int = 100):
    """Generate text using a Hugging Face model."""
    tokenizer, model = load_text_generation_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example usage:
# model = load_huggingface_model("deepseek-ai/DeepSeek-OCR")
# text = generate_text("Hello, how are you?")