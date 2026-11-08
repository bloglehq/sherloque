import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from sqlalchemy import text

from sherloque.model_providers import (
    FireworksModelProvider,
    OpenRouterModelProvider,
)
from tests.database import DatabaseTestCase


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


eval_retrieval = load_script(
    "eval_retrieval_test_module",
    "scratchpad/scripts/eval_retrieval.py",
)
reembed_documents = load_script(
    "reembed_documents_test_module",
    "scratchpad/scripts/reembed_documents.py",
)


class FakeProvider:
    def __init__(self, embed_model: str, rerank_model: str) -> None:
        self.embed_model = embed_model
        self.rerank_model = rerank_model


class EvaluationCheckpointTests(unittest.IsolatedAsyncioTestCase):
    def test_model_selection_reuses_one_openrouter_provider(self) -> None:
        args = Namespace(
            embedding_provider="openrouter",
            reranker_provider="openrouter",
            openrouter_embed_model="embed-model",
            openrouter_rerank_model="rerank-model",
        )
        settings = Namespace(
            openrouter_api_key="secret",
            openrouter_embed_model="default-embed",
            openrouter_rerank_model="default-rerank",
            fireworks_api_key="fireworks-secret",
        )

        embedding, reranker = eval_retrieval.create_models(args, settings)
        self.addAsyncCleanup(embedding.aclose)

        self.assertIsInstance(embedding, OpenRouterModelProvider)
        self.assertIs(embedding, reranker)
        self.assertEqual(embedding.embed_model, "embed-model")
        self.assertEqual(embedding.rerank_model, "rerank-model")

    def test_model_selection_keeps_mixed_providers_separate(self) -> None:
        args = Namespace(
            embedding_provider="fireworks",
            reranker_provider="openrouter",
            openrouter_embed_model=None,
            openrouter_rerank_model=None,
        )
        settings = Namespace(
            openrouter_api_key="secret",
            openrouter_embed_model="default-embed",
            openrouter_rerank_model="default-rerank",
            fireworks_api_key="fireworks-secret",
        )

        embedding, reranker = eval_retrieval.create_models(args, settings)
        self.addAsyncCleanup(embedding.aclose)
        self.addAsyncCleanup(reranker.aclose)

        self.assertIsInstance(embedding, FireworksModelProvider)
        self.assertIsInstance(reranker, OpenRouterModelProvider)
        self.assertIsNot(embedding, reranker)

    def test_checkpoint_rejects_different_configuration(self) -> None:
        provider = FakeProvider("embed-a", "rerank-a")
        metadata = eval_retrieval.checkpoint_metadata(
            dataset_name="dataset-a",
            k1=1.5,
            b=0.75,
            top_k=100,
            rrf_k=60,
            embedding_model=provider,
            rerank_model=provider,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            eval_retrieval.save_evaluation_checkpoint(
                path,
                metadata=metadata,
                runs={"bm25": {}},
                latencies={},
            )
            changed = {**metadata, "top_k": 20}
            with self.assertRaisesRegex(ValueError, "does not match"):
                eval_retrieval.load_evaluation_checkpoint(path, changed)

    async def test_backoff_retries_transient_200_error_envelope(self) -> None:
        attempts = 0
        sleeps: list[float] = []

        async def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Invalid OpenRouter embeddings response")
            return "ok"

        async def sleep(delay: float) -> None:
            sleeps.append(delay)

        result = await eval_retrieval.with_provider_backoff(
            operation,
            sleep=sleep,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual(sleeps, [2.0, 4.0])

    async def test_backoff_does_not_retry_permanent_http_error(self) -> None:
        request = httpx.Request("POST", "https://provider.test")
        response = httpx.Response(401, request=request)
        attempts = 0

        async def operation():
            nonlocal attempts
            attempts += 1
            raise httpx.HTTPStatusError(
                "unauthorized",
                request=request,
                response=response,
            )

        with self.assertRaises(httpx.HTTPStatusError):
            await eval_retrieval.with_provider_backoff(operation)
        self.assertEqual(attempts, 1)


class EvaluationDocumentMappingTests(DatabaseTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.query_embedding = [1.0] + [0.0] * 767
        async with self.engine.begin() as connection:
            await connection.execute(text("ALTER TABLE document ADD COLUMN url TEXT"))
            token_id = (
                await connection.execute(
                    text("""
                        INSERT INTO token (token, doc_freq)
                        VALUES ('search', 2) RETURNING _id
                    """)
                )
            ).scalar_one()
            for title, body, url, embedding in (
                ("Answer", "search asyncio answer", "beir-answer", self.query_embedding),
                ("Other", "search other answer", "beir-other", [0.0, 1.0] + [0.0] * 766),
            ):
                doc_id = (
                    await connection.execute(
                        text("""
                            INSERT INTO document (title, full_text, url, len, embedding)
                            VALUES (:title, :body, :url, 3, :embedding)
                            RETURNING _id
                        """),
                        {"title": title, "body": body, "url": url, "embedding": embedding},
                    )
                ).scalar_one()
                await connection.execute(
                    text("""
                        INSERT INTO term_doc_stats (doc_id, token_id, tf)
                        VALUES (:doc_id, :token_id, 1)
                    """),
                    {"doc_id": doc_id, "token_id": token_id},
                )

    async def test_lookup_keys_match_public_document_ids(self) -> None:
        documents = await self.index.retrieve_bm25(["search"], 1.5, 0.75, 10)

        mapping = await eval_retrieval.load_id_to_url(self.engine)

        self.assertEqual(mapping, {
            document.id: "beir-answer" if document.title == "Answer" else "beir-other"
            for document in documents
        })

    async def test_all_modes_emit_original_benchmark_ids(self) -> None:
        async def rerank(*, query, documents):
            return sorted(
                [(i, 1.0 if "asyncio" in body else 0.1) for i, body in enumerate(documents)],
                key=lambda item: item[1], reverse=True,
            )

        provider = SimpleNamespace(
            embed_model="fixture-embedding", rerank_model="fixture-reranker",
            embed=AsyncMock(return_value=[self.query_embedding]),
            rerank=AsyncMock(side_effect=rerank),
        )
        dataset = SimpleNamespace(
            queries_iter=lambda: iter([SimpleNamespace(query_id="query-1", text="search")])
        )
        mapping = await eval_retrieval.load_id_to_url(self.engine)

        with patch.object(eval_retrieval.ir_datasets, "load", return_value=dataset):
            runs, _ = await eval_retrieval.run_all_modes(
                self.engine, provider, provider, "fixture", mapping,
                k1=1.5, b=0.75, top_k=10, rrf_k=60, concurrency=1,
            )

        for mode in ("bm25", "vector", "rrf", "reranked"):
            with self.subTest(mode=mode):
                self.assertEqual(set(runs[mode]["query-1"]), {"beir-answer", "beir-other"})
        self.assertEqual(runs["vector"]["query-1"]["beir-answer"], 1.0)
        self.assertEqual(runs["vector"]["query-1"]["beir-other"], 0.0)
        self.assertEqual(runs["reranked"]["query-1"], {"beir-answer": 1.0, "beir-other": 0.1})


class ReembeddingCheckpointTests(unittest.TestCase):
    def test_checkpoint_rejects_a_different_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text(
                json.dumps(
                    {
                        "model": "model-a",
                        "dimensions": 768,
                        "corpus_count": 10,
                        "corpus_max_id": 10,
                        "last_id": 4,
                        "updated": 4,
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "model/corpus"):
                reembed_documents.load_checkpoint(
                    path,
                    "model-a",
                    768,
                    11,
                    11,
                )

    def test_destructive_run_requires_explicit_confirmation(self) -> None:
        args = Namespace(
            yes_replace_all=False,
            dimensions=768,
            batch_size=20,
            request_delay=0.0,
        )

        with self.assertRaisesRegex(ValueError, "--yes-replace-all"):
            import asyncio

            asyncio.run(reembed_documents.async_main(args))
