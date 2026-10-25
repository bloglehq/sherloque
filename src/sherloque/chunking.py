from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

# isort: off
from chonkie import (
    BaseChunker as ChonkieBaseChunker,
    RecursiveChunker as ChonkieRecursiveChunker,
    SentenceChunker as ChonkieSentenceChunker,
)
# isort: on


@dataclass
class SherloqueChunk:
    id: str = field(default_factory=lambda: f"chunk_{uuid4().hex}")
    index: int = field(default=0)
    text: str = field(default="")
    start_index: int = field(default=0)
    end_index: int = field(default=0)
    metadata: dict[str, Any] | None = field(default=None)


class Chunker:
    _chunker: ChonkieBaseChunker

    def chunk(self, text: str) -> list[SherloqueChunk]:
        chunks = self._chunker.chunk(text)
        sher_chunks = []
        for i, chunk in enumerate(chunks):
            if text[chunk.start_index : chunk.end_index] != chunk.text:
                raise ValueError(
                    "Chonkie chunk text does not match its reported source offsets"
                )
            sher_chunks.append(
                SherloqueChunk(
                    index=i,
                    text=chunk.text,
                    start_index=chunk.start_index,
                    end_index=chunk.end_index,
                )
            )
        return sher_chunks


class RecursiveChunker(Chunker):
    def __init__(self, *, chunk_size_tokens: int):
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        self.chunk_size_tokens = chunk_size_tokens
        self._chunker = ChonkieRecursiveChunker(
            tokenizer="gpt2",
            chunk_size=chunk_size_tokens,
        )


class SentenceChunker(Chunker):
    def __init__(self, *, chunk_size_tokens: int, chunk_overlap_tokens: int):
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if not 0 <= chunk_overlap_tokens < chunk_size_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be non-negative and smaller than "
                "chunk_size_tokens"
            )

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self._chunker = ChonkieSentenceChunker(
            tokenizer="gpt2",
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
            min_sentences_per_chunk=1,
        )


__all__ = [
    "RecursiveChunker",
    "SentenceChunker",
    "SherloqueChunk",
]
