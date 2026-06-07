from dataclasses import dataclass
from sqlalchemy import text

from sqlalchemy.ext.asyncio import AsyncEngine

from .base import Retriever, RetrieverResult
from sherloque.utils import get_embedding


# Qwen3-Embedding is asymmetric via INSTRUCTIONS (not nomic-style prefixes):
# documents are embedded raw, queries are wrapped as
#   `Instruct: {task}\nQuery: {query}`.
# DB embeddings are MRL-truncated to 768 dims and L2-normalized; the query
# vector must match (same dimensions + normalization) for cosine to be valid.
QWEN3_QUERY_TASK = (
    "Given a search query, retrieve relevant documents that answer the query"
)
EMBED_DIM = 768


def _format_query(query: str) -> str:
    return f"Instruct: {QWEN3_QUERY_TASK}\nQuery: {query}"


@dataclass
class VectorRetrieverConfig:
    embedding_model: str
    top_k: int = 10


class VectorRetriever(Retriever):
    VECTOR_RETRIEVE_SQL = text(
        # @formatter:off
        """
        SELECT d._id AS doc_id, d.title AS doc_title, 1 - (d.embedding <=> :query_embedding) AS score
        FROM document d
        ORDER BY d.embedding <=> :query_embedding
        LIMIT :top_k
        """
        # @formatter:on
    )

    def __init__(self, engine: AsyncEngine, config: VectorRetrieverConfig):
        self.engine = engine
        self.config = config

    async def retrieve(self, *, query: str, top_k: int | None = None) -> list[RetrieverResult]:
        top_k = top_k or self.config.top_k

        query_embedding = await get_embedding(
            model=self.config.embedding_model,
            text=_format_query(query),
            dimensions=EMBED_DIM,
            normalize=True,
        )
        async with self.engine.connect() as conn:
            cur = await conn.execute(
                self.VECTOR_RETRIEVE_SQL,
                {"query_embedding": query_embedding, "top_k": top_k},
            )
            rows = [record for record in cur]
        results = [RetrieverResult(doc_id=row.doc_id, doc_title=row.doc_title, score=row.score) for row in rows]
        return results


__all__ = [
    "VectorRetrieverConfig",
    "VectorRetriever",
]
