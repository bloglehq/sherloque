import json
import unittest

import httpx

from sherloque.model_providers import FireworksModelProvider


class FireworksModelProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests = []

        async def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            body = json.loads(request.content)
            if "return_logits" in body:
                scores = [0.9, 0.2, 0.7]
                data = [
                    {"index": index, "embedding": [1 - score, score]}
                    for index, score in enumerate(scores)
                ]
            else:
                data = [
                    {"index": 0, "embedding": [3.0, 4.0]},
                    {"index": 1, "embedding": [0.0, 0.0]},
                ]
            return httpx.Response(200, json={"data": data})

        self.provider = FireworksModelProvider(
            base_url="https://fireworks.test/v1",
            api_key="secret",
        )
        await self.provider.http_client.aclose()
        self.provider.http_client = httpx.AsyncClient(
            base_url="https://fireworks.test/v1",
            headers={"Authorization": "Bearer secret"},
            transport=httpx.MockTransport(handler),
        )

    async def asyncTearDown(self) -> None:
        await self.provider.aclose()

    async def test_embed_sends_payload_and_normalizes_vectors(self) -> None:
        vectors = await self.provider.embed(
            documents=["one", "two"],
            dimensions=2,
            user="test-user",
        )

        body = json.loads(self.requests[0].content)
        self.assertEqual(
            body,
            {
                "model": "fireworks/qwen3-embedding-8b",
                "input": ["one", "two"],
                "dimensions": 2,
                "user": "test-user",
            },
        )
        self.assertEqual(vectors, [[0.6, 0.8], [0.0, 0.0]])
        self.assertEqual(self.requests[0].headers["authorization"], "Bearer secret")

    async def test_rerank_builds_prompts_and_returns_top_results(self) -> None:
        results = await self.provider.rerank(
            query="query",
            documents=["one", "two", "three"],
            top_n=2,
        )

        body = json.loads(self.requests[0].content)
        self.assertEqual(body["model"], "fireworks/qwen3-reranker-8b")
        self.assertEqual(body["return_logits"], [2753, 9454])
        self.assertTrue(body["normalize"])
        self.assertEqual(
            body["input"][0],
            f"<Instruct>: {self.provider.INSTRUCTION}\n<Query>: query\n<Document>: one",
        )
        self.assertEqual(results, [(0, 0.9), (2, 0.7)])
