from typing import Any

from pydantic import BaseModel


class QueryResponse(BaseModel):
    results: list[Any] | None


__all__ = [
    "QueryResponse",
]
