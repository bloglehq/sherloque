from abc import ABC, abstractmethod
from collections import defaultdict

from sherloque.document import Document


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, *, query: str) -> list[Document]: ...


def reciprocal_rank_fusion(
    retrievers_results: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    doc_titles: dict[str, str | None] = {}
    doc_texts : dict[str, str] = {}

    for results in retrievers_results:
        for rank, result in enumerate(results, start=1):
            scores[result.id] += 1 / (k + rank)
            doc_titles[result.id] = result.title
            doc_texts[result.id] = result.text


    fused = [
        Document(id=doc_id, title=doc_titles[doc_id], text=doc_texts[doc_id], score=score)
        for doc_id, score in scores.items()
    ]
    return sorted(fused, key=lambda o: o.score, reverse=True)  # ty: ignore


__all__ = [
    "BaseRetriever",
    "reciprocal_rank_fusion",
]
