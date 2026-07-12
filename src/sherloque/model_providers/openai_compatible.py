from typing import Any

import httpx

from .base import BaseModelProvider


class OpenAICompatibleModelProvider(BaseModelProvider):
    """Embedding provider for OpenAI and OpenAI-compatible HTTP APIs.

    OpenAI does not expose a reranking endpoint. To satisfy Sherloque's model
    provider interface portably, :meth:`rerank` embeds the query and documents
    and ranks the documents by cosine similarity.
    """

    BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        base_url: str = BASE_URL,
        api_key: str | None = None,
        embed_model: str = "text-embedding-3-small",
        rerank_model: str | None = None,
        timeout: float | httpx.Timeout = 60.0,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.BASE_URL = base_url.rstrip("/")
        self.api_key = api_key
        self.embed_model = embed_model
        self.rerank_model = rerank_model or embed_model
        owns_http_client = http_client is None

        if http_client is None:
            headers = {"Content-Type": "application/json"}
            if api_key is not None:
                headers["Authorization"] = f"Bearer {api_key}"
            http_client = httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                base_url=self.BASE_URL,
            )

        super().__init__(
            http_client=http_client,
            owns_http_client=owns_http_client,
        )

    async def embed(
        self,
        *,
        documents: list[str],
        dimensions: int | None = None,
        normalize: bool = True,
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Embed ``documents`` with the configured OpenAI-compatible model."""
        if not documents:
            return []

        # Float encoding is required because callers consume numeric vectors.
        # Keep model and input under provider control rather than allowing
        # arbitrary keyword arguments to silently replace them.
        payload = dict(kwargs)
        payload.update(
            {
                "model": model or self.embed_model,
                "input": documents,
                "encoding_format": "float",
            }
        )
        if dimensions is not None:
            if dimensions <= 0:
                raise ValueError("dimensions must be greater than zero")
            payload["dimensions"] = dimensions

        response = await self._post("/embeddings", payload)
        response_data = response.json()
        try:
            data = response_data["data"]
            ordered = sorted(data, key=lambda item: item["index"])
            vectors = [item["embedding"] for item in ordered]
        except (KeyError, TypeError) as exc:
            raise ValueError("Invalid OpenAI-compatible embeddings response") from exc

        if len(vectors) != len(documents):
            raise ValueError(
                "OpenAI-compatible embeddings response returned "
                f"{len(vectors)} vectors for {len(documents)} inputs"
            )
        if normalize:
            vectors = [self._l2_normalize(vector) for vector in vectors]
        return vectors

    async def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """Rank documents by cosine similarity to ``query``.

        The returned indexes always refer to the original ``documents`` list.
        A separate ``rerank_model`` can be configured, but it must implement
        the standard embeddings endpoint.
        """
        if not documents or top_n == 0:
            return []
        if top_n is not None and top_n < 0:
            raise ValueError("top_n must be non-negative")

        vectors = await self.embed(
            documents=[query, *documents],
            model=self.rerank_model,
            dimensions=dimensions,
            normalize=True,
            **kwargs,
        )
        query_vector, *document_vectors = vectors
        results = [
            (index, sum(q * d for q, d in zip(query_vector, document_vector)))
            for index, document_vector in enumerate(document_vectors)
        ]
        results.sort(key=lambda item: item[1], reverse=True)
        return results if top_n is None else results[:top_n]


__all__ = [
    "OpenAICompatibleModelProvider",
]
