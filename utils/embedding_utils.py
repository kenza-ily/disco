"""
embedding_utils.py
Reusable utility for safe embedding of long texts with chunking and aggregation.
"""
import numpy as np
from typing import List, Callable

# You may need to install tiktoken or use a tokenizer from your embedding model's library
try:
    import tiktoken
except ImportError:
    tiktoken = None

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Estimate the number of tokens in a string using tiktoken (OpenAI tokenizer)."""
    if tiktoken is None:
        # Fallback: estimate 4 chars per token
        return max(1, len(text) // 4)
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))

def chunk_text_by_tokens(text: str, max_tokens: int, encoding_name: str = "cl100k_base") -> List[str]:
    """Split text into chunks, each <= max_tokens tokens."""
    if tiktoken is None:
        # Fallback: split by characters
        chunk_size = max_tokens * 4
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [enc.decode(chunk) for chunk in chunks]

def embed_long_text(
    text: str,
    embed_fn: Callable[[str], np.ndarray],
    max_tokens: int = 8000,
    encoding_name: str = "cl100k_base",
    agg_fn: Callable[[List[np.ndarray]], np.ndarray] = None,
) -> np.ndarray:
    """
    Embed a long text safely by chunking and aggregating.
    - text: input string
    - embed_fn: function that takes a string and returns an embedding (np.ndarray)
    - max_tokens: max tokens per chunk (should be < model limit)
    - encoding_name: tokenizer encoding name
    - agg_fn: aggregation function (default: mean)
    """
    chunks = chunk_text_by_tokens(text, max_tokens, encoding_name)
    embeddings = [embed_fn(chunk) for chunk in chunks]
    if agg_fn is None:
        agg_fn = lambda arrs: np.mean(arrs, axis=0)
    return agg_fn(embeddings)

# Example usage (in your embedding manager):
# from utils.embedding_utils import embed_long_text
# embedding = embed_long_text(long_text, embed_fn=your_embed_function, max_tokens=8000)

def get_chunked_embed_fn(embed_fn: Callable[[str], np.ndarray], max_tokens: int = 8000, encoding_name: str = "cl100k_base", agg_fn: Callable[[List[np.ndarray]], np.ndarray] = None):
    """
    Returns a function that safely embeds any text (with chunking/aggregation if needed).
    Usage:
        safe_embed = get_chunked_embed_fn(embed_fn, max_tokens=8000)
        embedding = safe_embed(long_text)
    """
    def safe_embed(text: str) -> np.ndarray:
        return embed_long_text(
            text,
            embed_fn=embed_fn,
            max_tokens=max_tokens,
            encoding_name=encoding_name,
            agg_fn=agg_fn,
        )
    return safe_embed
