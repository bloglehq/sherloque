import math
import unittest
from collections import Counter
from unittest.mock import AsyncMock, MagicMock

from sherloque.document import Document
from sherloque.index import PostgresIndex
from sherloque.query_engine import QueryEngine
from sherloque.retrieve import (
    BM25Retriever,
    BM25RetrieverConfig,
    TFIDFRetriever,
    TFIDFRetrieverConfig,
)
from tests.database import DatabaseTestCase


class TFIDFRetrieverUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_preprocesses_query_and_uses_config(self) -> None:
        index = MagicMock(spec=PostgresIndex)
        document = Document(id="banana", text="banana recipe", score=0.8)
        index.retrieve_tfidf = AsyncMock(return_value=[document])
        retriever = TFIDFRetriever(
            index, TFIDFRetrieverConfig(top_k=4, sublinear_tf=True)
        )

        results = await retriever.retrieve(query="BANANA recipes, banana!")

        index.retrieve_tfidf.assert_awaited_once_with(
            tokens=["banana", "recipe", "banana"], top_k=4, sublinear_tf=True
        )
        self.assertEqual(results, [document])

    async def test_preserves_zero_and_false_overrides(self) -> None:
        index = MagicMock(spec=PostgresIndex)
        index.retrieve_tfidf = AsyncMock(return_value=[])
        retriever = TFIDFRetriever(
            index, TFIDFRetrieverConfig(top_k=4, sublinear_tf=True)
        )

        self.assertEqual(
            await retriever.retrieve(query="banana", top_k=0, sublinear_tf=False), []
        )

        index.retrieve_tfidf.assert_awaited_once_with(
            tokens=["banana"], top_k=0, sublinear_tf=False
        )

    async def test_empty_inputs_skip_database_and_negative_limit_is_rejected(self) -> None:
        engine = MagicMock()
        index = PostgresIndex(engine, "test")

        self.assertEqual(await index.retrieve_tfidf([], 10), [])
        self.assertEqual(await index.retrieve_tfidf(["word"], 0), [])
        with self.assertRaisesRegex(ValueError, "top_k"):
            await index.retrieve_tfidf(["word"], -1)
        engine.connect.assert_not_called()


class TFIDFRetrieverTests(DatabaseTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        documents = [
            ("Apple heavy", "apple apple banana"),
            ("Banana heavy", "apple banana banana"),
            (None, "apple carrot carrot carrot"),
            ("Unrelated", "date"),
            ("Empty", ""),
        ]
        frequencies = [Counter(body.split()) for _, body in documents]
        doc_freq = Counter(term for terms in frequencies for term in terms)
        self.doc_ids = []
        async with self.engine.begin() as connection:
            token_ids = {}
            for term, frequency in doc_freq.items():
                token_ids[term] = (
                    await connection.execute(
                        self.index._token.insert()
                        .values(token=term, doc_freq=frequency)
                        .returning(self.index._token.c._id)
                    )
                ).scalar_one()
            for (title, body), terms in zip(documents, frequencies, strict=True):
                doc_id = (
                    await connection.execute(
                        self.index._document.insert()
                        .values(title=title, full_text=body, len=sum(terms.values()))
                        .returning(self.index._document.c._id)
                    )
                ).scalar_one()
                self.doc_ids.append(doc_id)
                for term, frequency in terms.items():
                    await connection.execute(
                        self.index._term_doc_stats.insert().inline().values(
                            doc_id=doc_id, token_id=token_ids[term], tf=frequency
                        )
                    )
            zero_token_id = (
                await connection.execute(
                    self.index._token.insert().values(token="unused", doc_freq=0)
                    .returning(self.index._token.c._id)
                )
            ).scalar_one()
            await connection.execute(
                self.index._term_doc_stats.insert().inline().values(
                    doc_id=self.doc_ids[-1], token_id=zero_token_id, tf=0
                )
            )
        self.retriever = TFIDFRetriever(self.index, TFIDFRetrieverConfig())

    async def test_cosine_scores_use_full_document_norms_and_smoothed_idf(self) -> None:
        results = await self.retriever.retrieve(query="banana")

        apple_idf = math.log(6 / 4) + 1
        banana_idf = math.log(6 / 3) + 1
        self.assertEqual([doc.title for doc in results], ["Banana heavy", "Apple heavy"])
        self.assertAlmostEqual(
            results[0].score,
            2 * banana_idf / math.hypot(apple_idf, 2 * banana_idf),
        )
        self.assertAlmostEqual(
            results[1].score,
            banana_idf / math.hypot(2 * apple_idf, banana_idf),
        )
        self.assertEqual(results[0].text, "apple banana banana")
        self.assertEqual(
            results[0].id, self.index._doc_hash(self.doc_ids[1], results[0].text)
        )

    async def test_query_frequency_changes_ranking_and_identical_vectors_score_one(self) -> None:
        apple_results = await self.retriever.retrieve(query="apple apple banana")
        banana_results = await self.retriever.retrieve(query="apple banana banana")

        self.assertEqual(apple_results[0].title, "Apple heavy")
        self.assertEqual(banana_results[0].title, "Banana heavy")
        self.assertAlmostEqual(apple_results[0].score, 1.0)
        self.assertAlmostEqual(banana_results[0].score, 1.0)
        self.assertIsNone(apple_results[-1].title)
        self.assertTrue(all(0 < doc.score <= 1 + 1e-12 for doc in apple_results))

    async def test_sublinear_tf_applies_to_documents_and_queries(self) -> None:
        results = await self.retriever.retrieve(query="banana", sublinear_tf=True)
        apple_idf = math.log(6 / 4) + 1
        banana_idf = math.log(6 / 3) + 1
        log_tf = 1 + math.log(2)
        self.assertAlmostEqual(
            results[0].score,
            log_tf * banana_idf / math.hypot(apple_idf, log_tf * banana_idf),
        )
        self.assertAlmostEqual(
            results[1].score,
            banana_idf / math.hypot(log_tf * apple_idf, banana_idf),
        )
        identical = await self.retriever.retrieve(
            query="apple apple banana", sublinear_tf=True
        )
        self.assertEqual(identical[0].title, "Apple heavy")
        self.assertAlmostEqual(identical[0].score, 1.0)

    async def test_limits_empty_queries_and_unknown_vocabulary(self) -> None:
        for query in ("", "?!", "astronomy", "unused"):
            with self.subTest(query=query):
                self.assertEqual(await self.retriever.retrieve(query=query), [])
        self.assertEqual(await self.retriever.retrieve(query="apple", top_k=0), [])
        with self.assertRaisesRegex(ValueError, "top_k"):
            await self.retriever.retrieve(query="apple", top_k=-1)
        results = await self.retriever.retrieve(query="apple", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Apple heavy")
        self.assertEqual(
            await self.retriever.retrieve(query="banana astronomy"),
            await self.retriever.retrieve(query="banana"),
        )

    async def test_empty_corpus_returns_no_results(self) -> None:
        async with self.engine.begin() as connection:
            await connection.execute(self.index._term_doc_stats.delete())
            await connection.execute(self.index._token.delete())
            await connection.execute(self.index._document.delete())

        self.assertEqual(await self.retriever.retrieve(query="banana"), [])

    async def test_fuses_with_bm25_without_duplicate_documents(self) -> None:
        bm25 = BM25Retriever(self.index, BM25RetrieverConfig())
        engine = QueryEngine([bm25, self.retriever], [])

        results = await engine.search(query="banana")
        tfidf_results = await self.retriever.retrieve(query="banana")

        self.assertEqual(len(results), 2)
        self.assertEqual({doc.id for doc in results}, {doc.id for doc in tfidf_results})
        self.assertEqual(results[0].title, "Banana heavy")
        self.assertAlmostEqual(results[0].score, 2 / 61)
