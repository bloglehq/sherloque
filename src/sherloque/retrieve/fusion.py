from collections import defaultdict

from .base import RetrieverResult


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
    "reciprocal_rank_fusion",
]
