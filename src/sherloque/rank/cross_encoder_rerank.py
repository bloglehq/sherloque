from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from sherloque.retrieve import RetrieverResult
from sherloque.model_providers import BaseClient


class CrossEncoderReRanker:
    FETCH_DOCS_SQL = text(
        # @formatter:off
        """
        SELECT d.full_text AS doc_full_text, d._id AS doc_id
        FROM document d
        WHERE d._id = ANY(:doc_ids)
        """
        # @formatter:on
    )

    def __init__(self, engine: AsyncEngine, model_client: BaseClient):
        self.engine = engine
        self.model_client = model_client

    async def rank(self, *, query: str, candidates: list[RetrieverResult], top_k: int) -> list[RetrieverResult]:
        if not candidates:
            return []

        async with self.engine.connect() as conn:
            cur = await conn.execute(
                self.FETCH_DOCS_SQL,
                {"doc_ids": [c.doc_id for c in candidates]},
            )
            id_to_doc_texts: dict[int, str] = {row.doc_id: row.doc_full_text for row in cur}

        # db results may be in different order than candidates, so map back to `candidates` order
        doc_texts = [id_to_doc_texts.get(c.doc_id, "") for c in candidates]

        rerank_results = await self.model_client.rerank(
            query=query,
            documents=doc_texts,
        )
        reranked_candidates = []
        for i, score in rerank_results[:top_k]:
            reranked_candidates.append(
                RetrieverResult(
                    doc_id=candidates[i].doc_id,
                    doc_title=candidates[i].doc_title,
                    score=score,
                )
            )
        return reranked_candidates


__all__ = [
    "CrossEncoderReRanker",
]
