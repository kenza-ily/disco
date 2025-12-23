from .llm_settings import get_settings, LLMModel, get_langchain_azure_openai
from openai import AzureOpenAI, AsyncAzureOpenAI
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from typing import Any, TypeVar, Union
from pathlib import Path
from .azure_models import AVAILABLE_AZURE_MODELS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

T = TypeVar('T')


def get_azure_openai_client() -> AzureOpenAI:
    settings = get_settings()
    return AzureOpenAI(
        api_version=settings.azure_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        max_retries=settings.max_retries_openai,
    )


def get_async_azure_openai_client() -> AsyncAzureOpenAI:
    settings = get_settings()
    return AsyncAzureOpenAI(
        api_version=settings.azure_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        max_retries=settings.max_retries_openai,
    )


def get_azure_document_intelligence_client() -> DocumentIntelligenceClient:
    settings = get_settings()
    credential = AzureKeyCredential(settings.azure_document_intelligence_key)
    return DocumentIntelligenceClient(
        endpoint=settings.azure_document_intelligence_endpoint,
        credential=credential
    )


async def azure_openai_structured_response(
    system_prompt: str,
    user_input: Union[str, list[dict[str, Any]]],
    pydantic_model: type[T],
    model: LLMModel,
    pdf_path: Path | None = None,
) -> T:
    """Handle Azure OpenAI structured response using beta chat completions parse API."""
    if model.name not in AVAILABLE_AZURE_MODELS:
        raise ValueError(f"Model '{model.name}' is not in the list of available Azure models. Available models: {AVAILABLE_AZURE_MODELS}")
    
    client = get_async_azure_openai_client()

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    if isinstance(user_input, str):
        messages.append({"role": "user", "content": user_input})
    else:
        messages.extend(user_input)

    # Use the beta chat completions parse API for structured responses
    response = await client.beta.chat.completions.parse(
        model=model.name,
        messages=messages,
        response_format=pydantic_model,
    )

    if not response.choices or not response.choices[0].message.parsed:
        raise ValueError("Failed to get a valid structured response from the model")

    return response.choices[0].message.parsed


async def langchain_azure_openai_structured_response(
    system_prompt: str,
    user_input: Union[str, list[dict[str, Any]]],
    pydantic_model: type[T],
    model_name: str = "gpt-4",
) -> T:
    """Handle Azure OpenAI structured response using LangChain."""
    if model_name not in AVAILABLE_AZURE_MODELS:
        raise ValueError(f"Model '{model_name}' is not in the list of available Azure models. Available models: {AVAILABLE_AZURE_MODELS}")
    
    llm = get_langchain_azure_openai(model_name)
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\n{format_instructions}"),
        ("human", "{input}")
    ])
    
    # Create chain
    chain = prompt | llm | parser
    
    # Prepare input
    if isinstance(user_input, str):
        input_text = user_input
    else:
        # Convert list of dicts to string
        input_text = "\n".join([msg.get("content", "") for msg in user_input if isinstance(msg, dict)])
    
    # Run chain
    result = await chain.ainvoke({
        "input": input_text,
        "format_instructions": parser.get_format_instructions()
    })
    
    return result