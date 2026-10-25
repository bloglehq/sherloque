from dataclasses import dataclass

from nltk import word_tokenize, WordNetLemmatizer

from .base import RetrieverResult, BaseRetriever
from sherloque.index import PostgresIndex


@dataclass
class BM25RetrieverConfig:
    k1: float = 1.5
    b: float = 0.75
    top_k: int = 10


class BM25Retriever(BaseRetriever):
    def __init__(self, index: PostgresIndex, config: BM25RetrieverConfig):
        self.index = index
        self.config = config
        self.lemmatizer = WordNetLemmatizer()

    def _preprocess_query(self, query: str) -> list[str]:
        """Lowercase, tokenize, drop non-alphanumeric tokens, and lemmatize."""
        return [
            self.lemmatizer.lemmatize(tok)
            for tok in word_tokenize(query.lower())
            if any(ch.isalnum() for ch in tok)
        ]

    async def retrieve(
        self,
        *,
        query: str,
        k1: float | None = None,
        b: float | None = None,
        top_k: int | None = None
    ) -> list[RetrieverResult]:
        k1 = self.config.k1 if k1 is None else k1
        b = self.config.b if b is None else b
        top_k = self.config.top_k if top_k is None else top_k

        tokens = self._preprocess_query(query)
        rows = await self.index.retrieve_bm25(tokens=tokens, k1=k1, b=b, top_k=top_k)
        results = [RetrieverResult(doc_id=row.doc_id, doc_title=row.doc_title or "", score=row.score) for row in rows]
        return results


__all__ = [
    "BM25RetrieverConfig",
    "BM25Retriever",
]
