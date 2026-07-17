import json
import math
import unittest

import httpx

from sherloque.model_providers import (
    FireworksModelProvider,
    OpenAICompatibleModelProvider,
)


def mock_transport() -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "return_logits" in body:
            data = [
                {"index": 0, "embedding": [0.1, 0.9]},
                {"index": 1, "embedding": [0.8, 0.2]},
            ]
        else:
            vectors = {
                "query": [1.0, 0.0],
                "best": [1.0, 0.0],
                "worst": [-1.0, 0.0],
                "one": [3.0, 4.0],
                "two": [4.0, 3.0],
            }
            data = [
                {"index": index, "embedding": vectors[text]}
                for index, text in enumerate(body["input"])
            ]
        return httpx.Response(200, json={"data": data})

    return httpx.MockTransport(handler)


class ModelProviderContract:
    provider: FireworksModelProvider | OpenAICompatibleModelProvider

    async def test_embed_contract(self) -> None:
        vectors = await self.provider.embed(documents=["one", "two"])

        self.assertEqual(len(vectors), 2)
        for vector in vectors:
            self.assertAlmostEqual(math.sqrt(sum(value**2 for value in vector)), 1.0)

    async def test_rerank_contract(self) -> None:
        results = await self.provider.rerank(
            query="query",
            documents=["best", "worst"],
            top_n=1,
        )

        self.assertEqual(len(results), 1)
        self.assertIn(results[0][0], range(2))
        self.assertIsInstance(results[0][1], float)


class OpenAICompatibleContractTests(
    ModelProviderContract,
    unittest.IsolatedAsyncioTestCase,
):
    async def asyncSetUp(self) -> None:
        self.client = httpx.AsyncClient(
            base_url="https://compatible.test/v1",
            transport=mock_transport(),
        )
        self.provider = OpenAICompatibleModelProvider(
            base_url="https://compatible.test/v1",
            http_client=self.client,
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()


class FireworksContractTests(
    ModelProviderContract,
    unittest.IsolatedAsyncioTestCase,
):
    async def asyncSetUp(self) -> None:
        self.provider = FireworksModelProvider(
            base_url="https://fireworks.test/v1",
            api_key="secret",
        )
        await self.provider.http_client.aclose()
        self.provider.http_client = httpx.AsyncClient(
            base_url="https://fireworks.test/v1",
            transport=mock_transport(),
        )

    async def asyncTearDown(self) -> None:
        await self.provider.aclose()
