from dataclasses import dataclass

from .base import BaseRetriever, RetrieverResult
from sherloque.index import PostgresIndex
from sherloque.model_providers import BaseModelProvider

# Qwen3-Embedding is asymmetric via INSTRUCTIONS (not nomic-style prefixes):
# documents are embedded raw, queries are wrapped as
#   `Instruct: {task}\nQuery: {query}`.
# DB embeddings are MRL-truncated to 768 dims and L2-normalized; the query
# vector must match (same dimensions + normalization) for cosine to be valid.
QWEN3_QUERY_TASK = (
    "Given a search query, retrieve relevant documents that answer the query"
)
EMBED_DIM = PostgresIndex.EMBEDDING_DIMENSIONS


@dataclass
class VectorRetrieverConfig:
    top_k: int = 10


class VectorRetriever(BaseRetriever):
    def __init__(self, index: PostgresIndex, config: VectorRetrieverConfig, model_client: BaseModelProvider):
        self.index = index
        self.config = config
        self.model_client = model_client

    async def retrieve(self, *, query: str, top_k: int | None = None) -> list[RetrieverResult]:
        top_k = self.config.top_k if top_k is None else top_k

        embeddings = await self.model_client.embed(
            documents=[f"Instruct: {QWEN3_QUERY_TASK}\nQuery: {query}"],
            dimensions=EMBED_DIM,
            normalize=True,
        )
        query_embedding = embeddings[0]
        rows = await self.index.retrieve_vector(query_embedding=query_embedding, top_k=top_k)
        return [
            RetrieverResult(
                doc_id=row.doc_id,
                doc_title=row.doc_title or "",
                score=1 - row.distance,
            )
            for row in rows
        ]


__all__ = [
    "VectorRetriever",
    "VectorRetrieverConfig",
]
