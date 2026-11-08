from sherloque.document import Document

from .crawler import *
from .index import PostgresIndex
from .rank import CrossEncoderReRanker
from .retrieve import BaseRetriever, reciprocal_rank_fusion


class Sherloque:
    def __init__(
        self,
        index: PostgresIndex,
        retrievers: list[BaseRetriever],
        rankers: list[CrossEncoderReRanker],
    ):
        self.index = index
        self.retrievers = retrievers
        self.rankers = rankers

    async def search(
        self,
        *,
        query: str,
        limit: int,
    ):
        assert limit > 0, "`limit` has to be >0"

        fused_candidates = await self.retrieve_fused(query=query)
        return await self.rank(
            query=query,
            documents=fused_candidates,
            limit=limit,
        )

    async def rank(self, *, query: str, documents: list[Document], limit: int = 20):
        ranked = documents.copy()
        for ranker in self.rankers:
            ranked = await ranker.rank(
                query=query,
                candidates=ranked,
                top_k=limit,
            )
        return ranked[:limit]

    async def retrieve(self, *, query: str) -> list[list[Document]]:
        retrieved = []
        for retriever in self.retrievers:
            retrieved.append(await retriever.retrieve(query=query))
        return retrieved

    async def retrieve_fused(self, *, query: str) -> list[Document]:
        retrieved = await self.retrieve(query=query)
        return reciprocal_rank_fusion(retrieved)



__all__ = [
    "Sherloque",
]
