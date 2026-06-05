import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio.engine import AsyncEngine

from sherloque import log_config
# from sherloque.search_engine.ranker import Ranker
from sherloque.retrieve.bm25 import Retriever, BM25RetrieverResult

log_config.setup()

LOG = logging.getLogger(__name__)

class DummyFusion:
    async def fuse(self, retriever_results: list[list[BM25RetrieverResult]]):
        return [a for b in retriever_results for a in b]



class QueryEngine:
    def __init__(self, engine: AsyncEngine, retrievers: list[Retriever]):
        self.engine = engine
        self.retrievers = retrievers
        self.fuser = DummyFusion()

    async def search(
        self,
        *,
        query: str,
    ):
        retriever_results = []
        for retriever in self.retrievers:
            retriever_results.append(await retriever.retrieve(query=query))

        return await self.fuser.fuse(retriever_results)


__all__ = [
    "QueryEngine",
]
