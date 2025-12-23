from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from .llm_settings import get_settings, get_langchain_huggingface_pipeline
from langchain_core.prompts import PromptTemplate


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


def langchain_generate_text(prompt: str, model_name: str = "google/gemma-3-27b-it") -> str:
    """Generate text using LangChain Hugging Face pipeline."""
    llm = get_langchain_huggingface_pipeline(model_name)
    
    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="{prompt}"
    )
    
    # Create chain using pipe operator (LangChain v0.1+)
    chain = prompt_template | llm
    
    # Run chain
    result = chain.invoke({"prompt": prompt})
    
    return result