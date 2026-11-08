from sqlalchemy.ext.asyncio import AsyncEngine

from sherloque.document import Document
from sherloque.model_providers import BaseModelProvider


class CrossEncoderReRanker:
    def __init__(self, engine: AsyncEngine, model_client: BaseModelProvider):
        self.engine = engine
        self.model_client = model_client

    async def rank(self, *, query: str, candidates: list[Document], top_k: int) -> list[Document]:
        if not candidates:
            return []

        doc_texts = [c.text for c in candidates]
        rerank_results = await self.model_client.rerank(
            query=query,
            documents=doc_texts,
        )
        reranked_candidates = []
        for i, score in rerank_results[:top_k]:
            reranked_candidates.append(
                Document(
                    id=candidates[i].id,
                    title=candidates[i].title,
                    text=candidates[i].text,
                    score=score,
                )
            )
        return reranked_candidates


__all__ = [
    "CrossEncoderReRanker",
]
