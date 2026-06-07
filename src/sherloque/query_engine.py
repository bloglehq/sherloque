import logging

from sherloque import log_config
from sherloque.retrieve import Retriever, reciprocal_rank_fusion

log_config.setup()

LOG = logging.getLogger(__name__)


class QueryEngine:
    def __init__(self, retrievers: list[Retriever]):
        self.retrievers = retrievers

    async def search(
        self,
        *,
        query: str,
    ):
        retriever_results = []
        for retriever in self.retrievers:
            retriever_results.append(await retriever.retrieve(query=query))

        return reciprocal_rank_fusion(retriever_results)


__all__ = [
    "QueryEngine",
]
