import math

import httpx

from .base import BaseModelProvider


class FireworksModelProvider(BaseModelProvider):
    BASE_URL = "https://api.fireworks.ai/inference/v1"
    INSTRUCTION = (
        "Given a search query, retrieve relevant documents that answer the query"
    )

    def __init__(
        self,
        api_key: str | None = None,
        rerank_model: str = "fireworks/qwen3-reranker-8b",
        embed_model: str = "fireworks/qwen3-embedding-8b",
    ):
        self.api_key = api_key
        self.rerank_model = rerank_model
        self.embed_model = embed_model
        super().__init__(
            http_client=httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
                base_url=self.BASE_URL,
            )
        )

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        """Scale a vector to unit L2 norm."""
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    async def embed(
        self,
        *,
        documents: list[str],
        dimensions: int | None = None,
        normalize: bool = True,
        **kwargs,
    ) -> list[list[float]]:
        """
        Embeds documents using the Fireworks API.

        `dimensions`, when set, is passed to the API. For Matryoshka (MRL) models
        like qwen3-embedding-8b this truncates the native 4096-dim vector to the
        requested size (e.g. 768) server-side.

        `normalize` L2-normalizes each returned vector (default True); Fireworks
        returns un-normalized vectors, so this is required for cosine consistency
        with how documents were stored.

        Returns one embedding vector per input document, in the same order.
        """
        payload = {
            "model": self.embed_model,
            "input": documents,
            **kwargs,
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions
        response = await self._post("/embeddings", payload)
        data = response.json()["data"]
        vectors = [item["embedding"] for item in data]
        if normalize:
            vectors = [self._l2_normalize(v) for v in vectors]
        return vectors

    async def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        token_false_id: int = 2753,
        token_true_id: int = 9454,
        **kwargs,
    ) -> list[tuple[int, float]]:
        """
        Reranks documents based on the query using the Fireworks API.

        Currently, only the `qwen3-reranker-8b` model is supported.
        """
        prompts = [
            f"<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"
            for doc in documents
        ]
        payload = {
            "model": self.rerank_model,
            "input": prompts,
            "return_logits": [token_false_id, token_true_id],
            "normalize": True,  # Applies softmax to the selected logits
        }
        response = await self._post("/embeddings", payload)
        results: list[tuple[int, float]] = []
        for i, item in enumerate(response.json()["data"]):
            probs = item["embedding"]
            relevance_score = probs[1]  # yes probability is the relevance score
            results.append((i, relevance_score))

        results.sort(key=lambda x: x[1], reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results


__all__ = [
    "FireworksModelProvider",
]
