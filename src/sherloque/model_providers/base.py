from abc import ABC, abstractmethod

import httpx

from ._retry import with_retry


class BaseModelProvider(ABC):
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client

    @with_retry(max_retries=3, delay=0.1)
    async def _post(self, url: str, payload: dict) -> httpx.Response:
        response = await self.http_client.post(url, json=payload)
        response.raise_for_status()
        return response

    @abstractmethod
    async def embed(self, *, documents: list[str], **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def rerank(self, *, query: str, documents: list[str], **kwargs):
        raise NotImplementedError


__all__ = [
    "BaseModelProvider",
]
