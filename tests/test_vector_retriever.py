import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import text

from sherloque.index import PostgresIndex
from sherloque.retrieve.vector import (
    EMBED_DIM,
    QWEN3_QUERY_TASK,
    VectorRetriever,
    VectorRetrieverConfig,
)
from tests.database import DatabaseTestCase


class VectorRetrieverUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_requests_a_normalized_qwen_query_embedding(self) -> None:
        index = MagicMock(spec=PostgresIndex)
        index.retrieve_vector = AsyncMock(return_value=[])
        model = MagicMock()
        model.embed = AsyncMock(
            return_value=[[1.0] + [0.0] * (EMBED_DIM - 1)]
        )
        retriever = VectorRetriever(
            index,
            VectorRetrieverConfig(),
            model,
        )

        await retriever.retrieve(query="banana bread")

        model.embed.assert_awaited_once_with(
            documents=[f"Instruct: {QWEN3_QUERY_TASK}\nQuery: banana bread"],
            dimensions=EMBED_DIM,
            normalize=True,
        )
        index.retrieve_vector.assert_awaited_once_with(
            query_embedding=[1.0] + [0.0] * (EMBED_DIM - 1), top_k=10
        )

    async def test_preserves_zero_limit_override(self) -> None:
        index = MagicMock(spec=PostgresIndex)
        index.retrieve_vector = AsyncMock(return_value=[])
        model = MagicMock()
        model.embed = AsyncMock(return_value=[[1.0] * EMBED_DIM])
        retriever = VectorRetriever(index, VectorRetrieverConfig(), model)

        self.assertEqual(await retriever.retrieve(query="anything", top_k=0), [])

        self.assertEqual(index.retrieve_vector.await_args.kwargs["top_k"], 0)

    async def test_converts_distances_to_scores_without_reordering(self) -> None:
        index = MagicMock(spec=PostgresIndex)
        index.retrieve_vector = AsyncMock(return_value=[
            SimpleNamespace(doc_id=1, doc_title=None, distance=0.0),
            SimpleNamespace(doc_id=2, doc_title="Related", distance=0.25),
            SimpleNamespace(doc_id=3, doc_title="Unrelated", distance=1.0),
            SimpleNamespace(doc_id=4, doc_title="Opposite", distance=2.0),
        ])
        model = MagicMock()
        model.embed = AsyncMock(return_value=[[1.0] + [0.0] * (EMBED_DIM - 1)])
        retriever = VectorRetriever(index, VectorRetrieverConfig(), model)

        results = await retriever.retrieve(query="anything")

        self.assertEqual([result.doc_id for result in results], [1, 2, 3, 4])
        self.assertEqual([result.score for result in results], [1.0, 0.75, 0.0, -1.0])
        self.assertEqual(results[0].doc_title, "")


class VectorRetrieverIntegrationTests(DatabaseTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        root_half = math.sqrt(0.5)
        self.query_vector = [1.0] + [0.0] * (EMBED_DIM - 1)
        documents = [
            ("Closest", self.query_vector),
            ("Unrelated", [0.0, 1.0] + [0.0] * (EMBED_DIM - 2)),
            ("Related", [root_half, root_half] + [0.0] * (EMBED_DIM - 2)),
        ]
        async with self.engine.begin() as connection:
            for title, embedding in documents:
                await connection.execute(
                    text(
                        """
                        INSERT INTO document (title, full_text, len, embedding)
                        VALUES (:title, :title, 1, :embedding)
                        """
                    ),
                    {"title": title, "embedding": embedding},
                )

    async def test_orders_documents_by_cosine_similarity(self) -> None:
        model = MagicMock()
        model.embed = AsyncMock(return_value=[self.query_vector])
        retriever = VectorRetriever(
            self.index,
            VectorRetrieverConfig(),
            model,
        )

        results = await retriever.retrieve(query="anything", top_k=2)

        self.assertEqual(
            [result.doc_title for result in results],
            ["Closest", "Related"],
        )
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertAlmostEqual(results[1].score, math.sqrt(0.5), places=6)
        self.assertGreater(results[0].score, results[1].score)

        rows = await self.index.retrieve_vector(self.query_vector, top_k=3)
        self.assertEqual([row.doc_title for row in rows], ["Closest", "Related", "Unrelated"])
        for row, expected in zip(rows, (0.0, 1 - math.sqrt(0.5), 1.0), strict=True):
            self.assertAlmostEqual(row.distance, expected, places=6)

    async def test_returns_empty_title_for_an_untitled_document(self) -> None:
        async with self.engine.begin() as connection:
            await connection.execute(
                text(
                    """
                    INSERT INTO document (title, full_text, len, embedding)
                    VALUES (NULL, 'Untitled', 1, :embedding)
                    """
                ),
                {"embedding": self.query_vector},
            )
        model = MagicMock()
        model.embed = AsyncMock(return_value=[self.query_vector])
        retriever = VectorRetriever(self.index, VectorRetrieverConfig(), model)

        results = await retriever.retrieve(query="anything", top_k=4)

        self.assertIn("", [result.doc_title for result in results])

    async def test_excludes_documents_without_embeddings(self) -> None:
        async with self.engine.begin() as connection:
            await connection.execute(
                text("INSERT INTO document (full_text, len) VALUES ('Not embedded', 2)")
            )
        model = MagicMock()
        model.embed = AsyncMock(return_value=[self.query_vector])
        retriever = VectorRetriever(self.index, VectorRetrieverConfig(), model)

        results = await retriever.retrieve(query="anything")

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.score is not None for result in results))
        self.assertEqual(await retriever.retrieve(query="anything", top_k=0), [])
