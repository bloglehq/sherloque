import unittest
from unittest.mock import AsyncMock, MagicMock

from sherloque.document import Document
from sherloque.rank.cross_encoder_rerank import CrossEncoderReRanker


class CrossEncoderReRankerTests(unittest.IsolatedAsyncioTestCase):
    async def test_maps_model_indexes_to_original_candidates(self) -> None:
        candidates = [
            Document(id="ten", title="Ten", text="text ten", score=0.3),
            Document(id="twenty", title="Twenty", text="text twenty", score=0.2),
            Document(id="thirty", title="Thirty", text="text thirty", score=0.1),
        ]
        original = [document.model_dump() for document in candidates]
        engine = MagicMock()
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
        self.assertEqual([result.id for result in results], ["thirty", "ten"])
        self.assertEqual([result.title for result in results], ["Thirty", "Ten"])
        self.assertEqual([result.text for result in results], ["text thirty", "text ten"])
        self.assertEqual([result.score for result in results], [0.95, 0.8])
        self.assertEqual([document.model_dump() for document in candidates], original)
        self.assertIsNot(results[0], candidates[2])
        engine.connect.assert_not_called()

    async def test_empty_candidates_skip_database_and_model(self) -> None:
        engine = MagicMock()
        model = MagicMock()
        model.rerank = AsyncMock()
        reranker = CrossEncoderReRanker(engine, model)

        results = await reranker.rank(query="query", candidates=[], top_k=5)

        self.assertEqual(results, [])
        engine.connect.assert_not_called()
        model.rerank.assert_not_awaited()
