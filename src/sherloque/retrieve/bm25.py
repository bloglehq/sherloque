from nltk import word_tokenize, WordNetLemmatizer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from pydantic import BaseModel, Field
from typing import Protocol


class BM25RetrieverResult(BaseModel):
    # doc_id:
    doc_title: str = Field(..., description="The title of the retrieved document.")
    score: float = Field(..., description="The BM25 score of the retrieved document.")


class Retriever(Protocol):
    async def retrieve(self, query: str, *args) -> list[BM25RetrieverResult]: ...


class BM25Retriever:
    BM25_SQL = text(
        """
        WITH q_tokens AS (SELECT t._id AS token_id, t.doc_freq
                          FROM token t
                          WHERE t.token = ANY(:tokens)),
             stats AS (SELECT COUNT(*)::float AS total_docs, AVG(len) ::float AS avgdl
                       FROM document)
        SELECT d.title AS doc_title,
               SUM(
                       LN((stats.total_docs - q.doc_freq + 0.5) / (q.doc_freq + 0.5) + 1)
                           * (tds.tf * (:k1 + 1))
                           / (
                           tds.tf
                               + :k1 * (1 - :b + :b * d.len / stats.avgdl)
                           )
               ) ::float AS score
        FROM term_doc_stats tds
                 JOIN q_tokens q ON tds.token_id = q.token_id
                 JOIN document d ON tds.doc_id = d._id
                 CROSS JOIN stats
        GROUP BY d.title
        ORDER BY score DESC
        LIMIT :top_k
        """
    )

    def __init__(self, engine: AsyncEngine, ):
        self.engine = engine

    def _preprocess_query(self, query: str) -> list[str]:
        """Lowercase, tokenize, drop non-alphanumeric tokens, and lemmatize."""
        lemmatizer = WordNetLemmatizer()
        return [
            lemmatizer.lemmatize(tok)
            for tok in word_tokenize(query.lower())
            if any(ch.isalnum() for ch in tok)
        ]

    async def retrieve(
        self,
        *,
        query: str,
        k1: float = 1.5,
        b: float = 0.75,
        top_k: int = 10
    ) -> list[BM25RetrieverResult]:
        tokens = self._preprocess_query(query)
        async with self.engine.connect() as conn:
            cur = await conn.execute(
                self.BM25_SQL,
                {"tokens": tokens, "k1": k1, "b": b, "top_k": top_k},
            )
        rows = [record for record in cur]
        results = [BM25RetrieverResult(doc_title=row.doc_title, score=row.score) for row in rows]

        return sorted(results, key=lambda r: r.score, reverse=True)


__all__ = [
    "BM25RetrieverResult",
    "Retriever",
    "BM25Retriever",
]
