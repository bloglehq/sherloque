import unittest
from unittest.mock import AsyncMock, MagicMock

from sherloque.document import Document
from sherloque.query_engine import QueryEngine


class QueryEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_fuses_retrieval_results_before_reranking(self) -> None:
        first_retriever = MagicMock()
        first_retriever.retrieve = AsyncMock(
            return_value=[
                Document(id="one", title="One", text="First document", score=0.9),
                Document(id="two", title="Two", text="Second document", score=0.8),
            ]
        )
        second_retriever = MagicMock()
        second_retriever.retrieve = AsyncMock(
            return_value=[
                Document(id="two", title="Two", text="Second document", score=0.95),
                Document(id="three", title="Three", text="Third document", score=0.7),
            ]
        )
        expected = [Document(id="two", title="Two", text="Second document", score=0.99)]
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
            [candidate.id for candidate in ranker_call["candidates"]],
            ["two", "one", "three"],
        )
        self.assertEqual(ranker_call["candidates"][0].text, "Second document")
        self.assertEqual(results, expected)

    async def test_returns_fused_results_without_rankers(self) -> None:
        retriever = MagicMock()
        retriever.retrieve = AsyncMock(
            return_value=[Document(id="one", title="One", text="First document", score=0.9)]
        )
        engine = QueryEngine([retriever], [])

        results = await engine.search(query="query")

        self.assertEqual([result.id for result in results], ["one"])
        self.assertEqual(results[0].text, "First document")
