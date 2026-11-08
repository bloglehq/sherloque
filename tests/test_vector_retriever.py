import math
import unittest
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy import text

from sherloque.document import Document
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

    async def test_preserves_documents_scores_and_order_from_the_index(self) -> None:
        index = MagicMock(spec=PostgresIndex)
        documents = [
            Document(id="closest", text="Closest document", score=1.0),
            Document(id="related", title="Related", text="Related document", score=0.75),
            Document(id="unrelated", title="Unrelated", text="Unrelated document", score=0.0),
            Document(id="opposite", title="Opposite", text="Opposite document", score=-1.0),
        ]
        index.retrieve_vector = AsyncMock(return_value=documents)
        model = MagicMock()
        model.embed = AsyncMock(return_value=[[1.0] + [0.0] * (EMBED_DIM - 1)])
        retriever = VectorRetriever(index, VectorRetrieverConfig(), model)

        results = await retriever.retrieve(query="anything")

        self.assertEqual(results, documents)
        self.assertEqual([result.score for result in results], [1.0, 0.75, 0.0, -1.0])
        self.assertIsNone(results[0].title)


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
            [result.title for result in results],
            ["Closest", "Related"],
        )
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertAlmostEqual(results[1].score, math.sqrt(0.5), places=6)
        self.assertGreater(results[0].score, results[1].score)

        rows = await self.index.retrieve_vector(self.query_vector, top_k=3)
        self.assertEqual([row.title for row in rows], ["Closest", "Related", "Unrelated"])
        for row, expected in zip(rows, (1.0, math.sqrt(0.5), 0.0), strict=True):
            self.assertIsInstance(row, Document)
            self.assertAlmostEqual(row.score, expected, places=6)

    async def test_returns_none_title_for_an_untitled_document(self) -> None:
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

        untitled = next(result for result in results if result.text == "Untitled")
        self.assertIsNone(untitled.title)

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
