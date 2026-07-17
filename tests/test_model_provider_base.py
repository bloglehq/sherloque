import unittest

from sherloque.model_providers import FireworksModelProvider


class BaseModelProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_subclasses_inherit_async_client_lifecycle(self) -> None:
        provider = FireworksModelProvider(api_key="test")

        async with provider as entered_provider:
            self.assertIs(entered_provider, provider)
            self.assertFalse(provider.http_client.is_closed)

        self.assertTrue(provider.http_client.is_closed)

    def test_l2_normalization_is_shared_by_subclasses(self) -> None:
        self.assertEqual(
            FireworksModelProvider._l2_normalize([3.0, 4.0]),
            [0.6, 0.8],
        )
        self.assertEqual(
            FireworksModelProvider._l2_normalize([0.0, 0.0]),
            [0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
