from dataclasses import dataclass

from sherloque.document import Document
from sherloque.index import PostgresIndex
from sherloque.utils import preprocess_text

from .base import BaseRetriever


@dataclass
class TFIDFRetrieverConfig:
    top_k: int = 10
    sublinear_tf: bool = False


class TFIDFRetriever(BaseRetriever):
    """Cosine retrieval with smoothed IDF over the existing lexical index."""

    def __init__(self, index: PostgresIndex, config: TFIDFRetrieverConfig):
        self.index = index
        self.config = config

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int | None = None,
        sublinear_tf: bool | None = None,
    ) -> list[Document]:
        top_k = self.config.top_k if top_k is None else top_k
        sublinear_tf = self.config.sublinear_tf if sublinear_tf is None else sublinear_tf

        return await self.index.retrieve_tfidf(
            tokens=preprocess_text(query), top_k=top_k, sublinear_tf=sublinear_tf
        )


__all__ = [
    "TFIDFRetriever",
    "TFIDFRetrieverConfig",
]
