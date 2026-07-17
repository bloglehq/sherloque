import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from sherloque.rank.cross_encoder_rerank import CrossEncoderReRanker
from sherloque.retrieve import RetrieverResult


class CrossEncoderReRankerTests(unittest.IsolatedAsyncioTestCase):
    async def test_maps_model_indexes_to_original_candidates(self) -> None:
        candidates = [
            RetrieverResult(doc_id=10, doc_title="Ten", score=0.3),
            RetrieverResult(doc_id=20, doc_title="Twenty", score=0.2),
            RetrieverResult(doc_id=30, doc_title="Thirty", score=0.1),
        ]
        rows = [
            SimpleNamespace(doc_id=30, doc_full_text="text thirty"),
            SimpleNamespace(doc_id=10, doc_full_text="text ten"),
            SimpleNamespace(doc_id=20, doc_full_text="text twenty"),
        ]
        connection = AsyncMock()
        connection.execute.return_value = rows
        engine = MagicMock()
        engine.connect.return_value.__aenter__.return_value = connection
        model = MagicMock()
        model.rerank = AsyncMock(
            return_value=[(2, 0.95), (0, 0.8), (1, 0.1)]
        )
        reranker = CrossEncoderReRanker(engine, model)

        results = await reranker.rank(
            query="query",
            candidates=candidates,
            top_k=2,
        )

        model.rerank.assert_awaited_once_with(
            query="query",
            documents=["text ten", "text twenty", "text thirty"],
        )
        self.assertEqual([result.doc_id for result in results], [30, 10])
        self.assertEqual([result.score for result in results], [0.95, 0.8])

    async def test_empty_candidates_skip_database_and_model(self) -> None:
        engine = MagicMock()
        model = MagicMock()
        model.rerank = AsyncMock()
        reranker = CrossEncoderReRanker(engine, model)

        results = await reranker.rank(query="query", candidates=[], top_k=5)

        self.assertEqual(results, [])
        engine.connect.assert_not_called()
        model.rerank.assert_not_awaited()
