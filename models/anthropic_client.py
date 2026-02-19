"""
Direct Anthropic API client for Claude models.

Provides direct access to Claude models via Anthropic API (alternative to AWS Bedrock).
"""

import base64
import logging
from pathlib import Path
from typing import Optional

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from .settings import get_settings

logger = logging.getLogger(__name__)


def get_anthropic_client() -> "Anthropic":
    """Get Anthropic client for direct API calls.

    Raises:
        ImportError: If anthropic package not installed
        ValueError: If ANTHROPIC_API_KEY not configured
    """
    if Anthropic is None:
        raise ImportError(
            "Anthropic package not installed. Install with: pip install anthropic"
        )

    settings = get_settings()

    if not settings.anthropic_api_key:
        raise ValueError(
            "Anthropic API key not configured. "
            "Set ANTHROPIC_API_KEY in .env.local"
        )

    return Anthropic(api_key=settings.anthropic_api_key)


def call_claude_direct(
    image_path: str,
    model_id: str,
    query: str,
    max_tokens: int = 4000
) -> dict:
    """Call Claude via direct Anthropic API.

    Args:
        image_path: Path to image file
        model_id: Anthropic model ID (e.g., "claude-3-5-sonnet-20241022")
        query: Text query/prompt
        max_tokens: Maximum tokens in response

    Returns:
        dict with 'content' and 'usage' keys
    """
    client = get_anthropic_client()

    # Load and encode image
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

    # Call Anthropic API
    message = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
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

    # Extract response
    content = message.content[0].text if message.content else ""
    usage = {
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
    }

    return {
        "content": content,
        "usage": usage,
    }
