import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

import httpx

from ._retry import with_retry


class BaseModelProvider(ABC):
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        *,
        owns_http_client: bool = True,
    ):
        self.http_client = http_client
        self._owns_http_client = owns_http_client

    @staticmethod
    def _l2_normalize(vector: Sequence[float]) -> list[float]:
        """Return a unit-length copy of ``vector``."""
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return list(vector)
        return [value / norm for value in vector]

    @with_retry(max_retries=3, delay=0.1)
    async def _post(self, url: str, payload: dict) -> httpx.Response:
        response = await self.http_client.post(url, json=payload)
        response.raise_for_status()
        return response

    async def aclose(self) -> None:
        """Close the HTTP client when it is owned by this provider."""
        if self._owns_http_client:
            await self.http_client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    @abstractmethod
    async def embed(self, *, documents: list[str], **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def rerank(self, *, query: str, documents: list[str], **kwargs):
        raise NotImplementedError


__all__ = [
    "BaseModelProvider",
]
