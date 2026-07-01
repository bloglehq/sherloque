import logging

from sherloque import log_config
from sherloque.rank import CrossEncoderReRanker
from sherloque.retrieve import BaseRetriever, reciprocal_rank_fusion

log_config.setup()

LOG = logging.getLogger(__name__)


class QueryEngine:
    def __init__(self, retrievers: list[BaseRetriever], rankers: list[CrossEncoderReRanker]):
        self.retrievers = retrievers
        self.rankers = rankers

    async def search(
        self,
        *,
        query: str,
    ):
        retriever_results = []
        for retriever in self.retrievers:
            retriever_results.append(await retriever.retrieve(query=query))

        fused_candidates = reciprocal_rank_fusion(retriever_results)

        for ranker in self.rankers:
            fused_candidates = await ranker.rank(
                query=query,
                candidates=fused_candidates,
                top_k=20,
            )
        return fused_candidates


__all__ = [
    "QueryEngine",
]
