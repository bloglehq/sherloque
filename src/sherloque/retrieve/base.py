from abc import ABC, abstractmethod
from collections import defaultdict

from pydantic import BaseModel, Field


class RetrieverResult(BaseModel):
    doc_id: int = Field(..., description="The unique identifier of the retrieved document.")
    doc_title: str = Field(..., description="The title of the retrieved document.")
    score: float = Field(..., description="The score of the retrieved document.")


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, *, query: str, **kwargs) -> list[RetrieverResult]: ...


def reciprocal_rank_fusion(
    retrievers_results: list[list[RetrieverResult]],
    k: int = 60,
) -> list[RetrieverResult]:
    scores: dict[int, float] = defaultdict(float)
    doc_titles: dict[int, str] = {}

    for results in retrievers_results:
        for rank, result in enumerate(results, start=1):
            scores[result.doc_id] += 1 / (k + rank)
            doc_titles[result.doc_id] = result.doc_title

    fused = [
        RetrieverResult(doc_id=doc_id, doc_title=doc_titles[doc_id], score=score)
        for doc_id, score in scores.items()
    ]
    return sorted(fused, key=lambda o: o.score, reverse=True)


__all__ = [
    "RetrieverResult",
    "BaseRetriever",
    "reciprocal_rank_fusion",
]
