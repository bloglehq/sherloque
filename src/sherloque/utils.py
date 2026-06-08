import asyncio
import math
from functools import lru_cache

import httpx
from nltk import word_tokenize, WordNetLemmatizer
from openai import AsyncOpenAI, RateLimitError

from sherloque.config import get_settings

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

ALLOWED_TABLE_FIELDS = {
    "document": ["url", "title"],
    "token": ["token"],
    "token_position": ["doc_id", "token_id", "position"],
    "term_doc_stats": ["token_id", "doc_id", "tf"],
    "link": ["from_doc_id", "to_doc_id"],
}


def preprocess_text(text: str) -> list[str]:
    """Lowercase, tokenize, drop non-alphanumeric tokens, and lemmatize."""
    lemmatizer = WordNetLemmatizer()
    return [
        lemmatizer.lemmatize(tok)
        for tok in word_tokenize(text.lower())
        if any(ch.isalnum() for ch in tok)
    ]


@lru_cache(maxsize=1)
def _embeddings_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=get_settings().fireworks_api_key,
        base_url=FIREWORKS_BASE_URL,
    )


def _l2_normalize(vec: list[float]) -> list[float]:
    """Scale a vector to unit L2 norm. Cosine ranking is scale-invariant, but we
    normalize so stored/queried vectors are clean unit vectors and cosine == dot.
    Qwen3-Embedding returns un-normalized vectors (norm ~60), unlike nomic."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


async def get_embedding(
    *,
    model: str,
    text: str,
    dimensions: int | None = None,
    normalize: bool = True,
) -> list[float]:
    """Embed a single text via the Fireworks (OpenAI-compatible) API.

    `dimensions`, when set, is passed through to the API. For Matryoshka (MRL)
    models like qwen3-embedding-8b this truncates the native 4096-dim vector to
    the requested size (e.g. 768) server-side.

    `normalize` L2-normalizes the (possibly truncated) vector. Kept True by
    default so the backfill and query paths stay consistent.
    """
    kwargs: dict = {"model": model, "input": text}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions

    # Retry with exponential backoff on Fireworks rate limits (429).
    delay = 2.0
    max_attempts = 12
    for attempt in range(max_attempts):
        try:
            resp = await _embeddings_client().embeddings.create(**kwargs)
            break
        except RateLimitError:
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

    vec = resp.data[0].embedding
    return _l2_normalize(vec) if normalize else vec


async def rerank(
    *,
    model: str,
    query: str,
    documents: list[str],
) -> list[tuple[int, float]]:
    """Rerank `documents` against `query` via the Fireworks /v1/rerank endpoint.

    Returns a list of `(original_index, relevance_score)` pairs, sorted by score
    descending. `original_index` is the position of the document in the
    `documents` list you passed in.

    IMPORTANT: the API reorders results by score and does NOT preserve input
    order. Associate each result back to your data via `original_index` — NEVER
    by position in the returned list. So the caller should keep its docs in a
    stable order (e.g. a parallel `[doc_id, ...]` list) and look up `doc_ids[original_index]`.

    Note: the OpenAI SDK has no rerank method, so this is a raw POST. Scores are
    peaky "yes"-probabilities (relevant ~0.7, irrelevant ~1e-6) — use for
    ordering, not as calibrated confidence.
    """
    payload: dict = {
        "model": model,
        "query": query,
        "documents": documents,
        "return_documents": False,
    }

    headers = {
        "Authorization": f"Bearer {get_settings().fireworks_api_key}",
        "Content-Type": "application/json",
    }

    # Retry with exponential backoff on Fireworks rate limits (429).
    delay = 2.0
    max_attempts = 12
    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(max_attempts):
            resp = await client.post(
                f"{FIREWORKS_BASE_URL}/rerank", headers=headers, json=payload
            )
            if resp.status_code == 429 and attempt < max_attempts - 1:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            resp.raise_for_status()
            data = resp.json()["data"]
            # API returns sorted desc already; sort defensively to be safe.
            pairs = [(item["index"], item["relevance_score"]) for item in data]
            return sorted(pairs, key=lambda p: p[1], reverse=True)
    raise RuntimeError("unreachable")


__all__ = [
    "preprocess_text",
    "get_embedding",
    "rerank",
]
