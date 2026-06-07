from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

class RetrieverResult(BaseModel):
    doc_id: int = Field(..., description="The unique identifier of the retrieved document.")
    doc_title: str = Field(..., description="The title of the retrieved document.")
    score: float = Field(..., description="The score of the retrieved document.")

class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, *, query: str, **kwargs) -> list[RetrieverResult]: ...


__all__ = [
    "RetrieverResult",
    "Retriever",
]
