from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    title: str | None = None
    text: str
    metadata: dict[str, Any] | None = None
    score: float | None = None


__all__ = [
    "Document",
]
