import json
import math
import unittest

import httpx

from sherloque.model_providers.openai_compatible import (
    OpenAICompatibleModelProvider,
)


class OpenAICompatibleModelProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            body = json.loads(request.content)
            embeddings = {
                "query": [1.0, 0.0],
                "best": [2.0, 0.0],
                "worst": [-1.0, 0.0],
                "middle": [1.0, 1.0],
            }
            # Reverse the response to ensure indexes, not response order, are
            # used to reconstruct the input order.
            data = [
                {
                    "object": "embedding",
                    "embedding": embeddings.get(text, [3.0, 4.0]),
                    "index": index,
                }
                for index, text in reversed(list(enumerate(body["input"])))
            ]
            return httpx.Response(200, json={"object": "list", "data": data})

        self.client = httpx.AsyncClient(
            base_url="http://compatible.test/v1",
            transport=httpx.MockTransport(handler),
        )
        self.provider = OpenAICompatibleModelProvider(
            base_url="http://compatible.test/v1",
            embed_model="embedding-model",
            http_client=self.client,
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_embed_sends_standard_payload_and_normalizes(self) -> None:
        result = await self.provider.embed(
            documents=["one", "two"], dimensions=2, user="test-user"
        )

        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0][0], 0.6)
        self.assertAlmostEqual(result[0][1], 0.8)
        body = json.loads(self.requests[0].content)
        self.assertEqual(self.requests[0].url.path, "/v1/embeddings")
        self.assertEqual(
            body,
            {
                "model": "embedding-model",
                "input": ["one", "two"],
                "encoding_format": "float",
                "dimensions": 2,
                "user": "test-user",
            },
        )

    async def test_embed_can_return_raw_vectors(self) -> None:
        result = await self.provider.embed(documents=["one"], normalize=False)

        self.assertEqual(result, [[3.0, 4.0]])

    async def test_embed_can_override_the_configured_model(self) -> None:
        await self.provider.embed(documents=["one"], model="override-model")

        body = json.loads(self.requests[0].content)
        self.assertEqual(body["model"], "override-model")

    async def test_empty_embed_does_not_make_a_request(self) -> None:
        self.assertEqual(await self.provider.embed(documents=[]), [])
        self.assertEqual(self.requests, [])

    async def test_rerank_uses_cosine_similarity_and_original_indexes(self) -> None:
        self.provider.rerank_model = "rerank-model"
        result = await self.provider.rerank(
            query="query",
            documents=["worst", "middle", "best"],
            top_n=2,
        )

        self.assertEqual([index for index, _ in result], [2, 1])
        self.assertAlmostEqual(result[0][1], 1.0)
        self.assertAlmostEqual(result[1][1], 1 / math.sqrt(2))
        self.assertEqual(json.loads(self.requests[0].content)["model"], "rerank-model")

    async def test_provider_does_not_close_an_injected_client(self) -> None:
        await self.provider.aclose()

        self.assertFalse(self.client.is_closed)

    async def test_invalid_arguments_fail_before_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimensions"):
            await self.provider.embed(documents=["one"], dimensions=0)
        with self.assertRaisesRegex(ValueError, "top_n"):
            await self.provider.rerank(query="query", documents=["one"], top_n=-1)

        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
