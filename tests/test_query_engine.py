import unittest
from unittest.mock import AsyncMock, MagicMock

from sherloque.query_engine import QueryEngine
from sherloque.retrieve import RetrieverResult


class QueryEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_fuses_retrieval_results_before_reranking(self) -> None:
        first_retriever = MagicMock()
        first_retriever.retrieve = AsyncMock(
            return_value=[
                RetrieverResult(doc_id=1, doc_title="One", score=0.9),
                RetrieverResult(doc_id=2, doc_title="Two", score=0.8),
            ]
        )
        second_retriever = MagicMock()
        second_retriever.retrieve = AsyncMock(
            return_value=[
                RetrieverResult(doc_id=2, doc_title="Two", score=0.95),
                RetrieverResult(doc_id=3, doc_title="Three", score=0.7),
            ]
        )
        expected = [RetrieverResult(doc_id=2, doc_title="Two", score=0.99)]
        ranker = MagicMock()
        ranker.rank = AsyncMock(return_value=expected)
        engine = QueryEngine([first_retriever, second_retriever], [ranker])

        results = await engine.search(query="query")

        first_retriever.retrieve.assert_awaited_once_with(query="query")
        second_retriever.retrieve.assert_awaited_once_with(query="query")
        ranker.rank.assert_awaited_once()
        ranker_call = ranker.rank.await_args.kwargs
        self.assertEqual(ranker_call["query"], "query")
        self.assertEqual(ranker_call["top_k"], 20)
        self.assertEqual(
            [candidate.doc_id for candidate in ranker_call["candidates"]],
            [2, 1, 3],
        )
        self.assertEqual(results, expected)

    async def test_returns_fused_results_without_rankers(self) -> None:
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(
            return_value=[RetrieverResult(doc_id=1, doc_title="One", score=0.9)]
        )
        engine = QueryEngine([retriever], [])

        results = await engine.search(query="query")

        self.assertEqual([result.doc_id for result in results], [1])
