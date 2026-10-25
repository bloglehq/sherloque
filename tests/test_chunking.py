import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import UUID

from sherloque.chunking import RecursiveChunker, SentenceChunker, SherloqueChunk


class SherloqueChunkTests(unittest.TestCase):
    def test_defaults_and_unique_ids(self) -> None:
        chunks = [SherloqueChunk(), SherloqueChunk()]

        self.assertNotEqual(chunks[0].id, chunks[1].id)
        for chunk in chunks:
            self.assertRegex(chunk.id, r"^chunk_[0-9a-f]{32}$")
            self.assertEqual(UUID(chunk.id.removeprefix("chunk_")).version, 4)
            self.assertEqual(chunk.index, 0)
            self.assertEqual(chunk.text, "")
            self.assertEqual(chunk.start_index, 0)
            self.assertEqual(chunk.end_index, 0)
            self.assertIsNone(chunk.metadata)

    def test_explicit_fields_and_metadata(self) -> None:
        metadata = {"document_id": "doc-1", "page": 2}
        chunk = SherloqueChunk(
            id="chunk_custom", index=3, text="Passage.",
            start_index=10, end_index=18, metadata=metadata,
        )

        self.assertEqual(chunk.id, "chunk_custom")
        self.assertEqual(chunk.index, 3)
        self.assertEqual(chunk.text, "Passage.")
        self.assertEqual(chunk.start_index, 10)
        self.assertEqual(chunk.end_index, 18)
        self.assertEqual(chunk.metadata, metadata)
        self.assertIsNone(SherloqueChunk().metadata)


class ChunkerAdapterTests(unittest.TestCase):
    def test_recursive_configuration(self) -> None:
        with patch("sherloque.chunking.ChonkieRecursiveChunker") as native:
            chunker = RecursiveChunker(chunk_size_tokens=256)

        native.assert_called_once_with(tokenizer="gpt2", chunk_size=256)
        self.assertEqual(chunker.chunk_size_tokens, 256)

    def test_sentence_configuration(self) -> None:
        with patch("sherloque.chunking.ChonkieSentenceChunker") as native:
            chunker = SentenceChunker(chunk_size_tokens=128, chunk_overlap_tokens=32)

        native.assert_called_once_with(
            tokenizer="gpt2", chunk_size=128, chunk_overlap=32,
            min_sentences_per_chunk=1,
        )
        self.assertEqual(chunker.chunk_size_tokens, 128)
        self.assertEqual(chunker.chunk_overlap_tokens, 32)

    def test_adapters_preserve_native_output_and_assign_indices_and_ids(self) -> None:
        text = "xxxxFirst. Second."
        native_chunks = [
            SimpleNamespace(text="First. ", start_index=4, end_index=11),
            SimpleNamespace(text="Second.", start_index=11, end_index=18),
        ]
        configurations = (
            (RecursiveChunker, "ChonkieRecursiveChunker", {"chunk_size_tokens": 20}),
            (SentenceChunker, "ChonkieSentenceChunker",
             {"chunk_size_tokens": 20, "chunk_overlap_tokens": 0}),
        )
        for adapter, native_name, kwargs in configurations:
            with self.subTest(adapter=adapter.__name__):
                fake = Mock()
                fake.chunk.return_value = native_chunks
                with patch(f"sherloque.chunking.{native_name}", return_value=fake):
                    chunker = adapter(**kwargs)
                    chunks = chunker.chunk(text)
                    repeated_chunks = chunker.chunk(text)
                    fake.chunk.assert_called_with(text)
                    self.assertEqual(fake.chunk.call_count, 2)
                    self.assertEqual([chunk.index for chunk in chunks], [0, 1])
                    self.assertEqual([chunk.index for chunk in repeated_chunks], [0, 1])
                    self.assertEqual(len({c.id for c in chunks + repeated_chunks}), 4)
                    for chunk, native_chunk in zip(chunks, native_chunks, strict=True):
                        self.assertIsInstance(chunk, SherloqueChunk)
                        self.assertEqual(chunk.text, native_chunk.text)
                        self.assertEqual(chunk.start_index, native_chunk.start_index)
                        self.assertEqual(chunk.end_index, native_chunk.end_index)
                        self.assertIsNone(chunk.metadata)

                    fake.chunk.return_value = []
                    self.assertEqual(chunker.chunk(text), [])


class RecursiveChunkerTests(unittest.TestCase):
    def test_chunks_keep_offsets_and_respect_target_size(self) -> None:
        text = " ".join(
            f"Sentence {index} has enough words for a repeatable chunking test."
            for index in range(30)
        )
        chunker = RecursiveChunker(chunk_size_tokens=100)

        chunks = chunker.chunk(text)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assertEqual([chunk.index for chunk in chunks], list(range(len(chunks))))
        for chunk in chunks:
            self.assertEqual(text[chunk.start_index : chunk.end_index], chunk.text)
            self.assertLessEqual(chunker._chunker.tokenizer.count_tokens(chunk.text), 100)

    def test_unicode_chunks_preserve_source_characters(self) -> None:
        text = " ".join(["You’ll receive £545 — depending on eligibility."] * 40)
        chunks = RecursiveChunker(chunk_size_tokens=30).chunk(text)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        for chunk in chunks:
            self.assertTrue(chunk.text)
            self.assertEqual(text[chunk.start_index : chunk.end_index], chunk.text)

    def test_empty_text_produces_no_chunks(self) -> None:
        self.assertEqual(RecursiveChunker(chunk_size_tokens=10).chunk(""), [])

    def test_short_text_produces_one_chunk(self) -> None:
        chunks = RecursiveChunker(chunk_size_tokens=100).chunk("Short passage.")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Short passage.")
        self.assertEqual(chunks[0].start_index, 0)
        self.assertEqual(chunks[0].end_index, len("Short passage."))

    def test_rejects_non_positive_chunk_size_before_constructing_backend(self) -> None:
        with patch("sherloque.chunking.ChonkieRecursiveChunker") as native:
            for size in (0, -1):
                with self.subTest(size=size):
                    with self.assertRaisesRegex(ValueError, "positive"):
                        RecursiveChunker(chunk_size_tokens=size)
            native.assert_not_called()


class SentenceChunkerTests(unittest.TestCase):
    text = (
        "First sentence has enough words to be useful and self contained. "
        "Second sentence continues the same explanation with added detail. "
        "Third sentence begins a different idea with relevant context. "
        "Fourth sentence provides a little more supporting information."
    )

    def test_chunks_keep_source_offsets_and_overlap(self) -> None:
        chunker = SentenceChunker(chunk_size_tokens=28, chunk_overlap_tokens=14)

        chunks = chunker.chunk(self.text)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].start_index, 0)
        self.assertEqual(chunks[-1].end_index, len(self.text))
        self.assertEqual([chunk.index for chunk in chunks], list(range(len(chunks))))
        for chunk in chunks:
            self.assertEqual(self.text[chunk.start_index : chunk.end_index], chunk.text)
            self.assertLessEqual(chunker._chunker.tokenizer.count_tokens(chunk.text), 28)
        self.assertLess(chunks[1].start_index, chunks[0].end_index)

    def test_zero_overlap_preserves_source_without_duplication(self) -> None:
        chunks = SentenceChunker(
            chunk_size_tokens=28, chunk_overlap_tokens=0
        ).chunk(self.text)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.text for chunk in chunks), self.text)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous.end_index, current.start_index)

    def test_unicode_chunks_preserve_source_offsets(self) -> None:
        text = " ".join(["You’ll receive £545 — depending on eligibility."] * 40)
        chunks = SentenceChunker(
            chunk_size_tokens=30, chunk_overlap_tokens=0
        ).chunk(text)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        for chunk in chunks:
            self.assertTrue(chunk.text)
            self.assertEqual(text[chunk.start_index : chunk.end_index], chunk.text)

    def test_empty_text_produces_no_chunks(self) -> None:
        chunker = SentenceChunker(chunk_size_tokens=10, chunk_overlap_tokens=2)

        self.assertEqual(chunker.chunk(""), [])

    def test_short_text_produces_one_chunk(self) -> None:
        chunks = SentenceChunker(
            chunk_size_tokens=100, chunk_overlap_tokens=0
        ).chunk("Short passage.")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Short passage.")
        self.assertEqual(chunks[0].start_index, 0)
        self.assertEqual(chunks[0].end_index, len("Short passage."))

    def test_rejects_invalid_parameters_before_constructing_backend(self) -> None:
        cases = (
            (0, 0, "positive"), (-1, 0, "positive"),
            (10, -1, "non-negative"), (10, 10, "smaller"), (10, 11, "smaller"),
        )
        with patch("sherloque.chunking.ChonkieSentenceChunker") as native:
            for size, overlap, message in cases:
                with self.subTest(size=size, overlap=overlap):
                    with self.assertRaisesRegex(ValueError, message):
                        SentenceChunker(
                            chunk_size_tokens=size, chunk_overlap_tokens=overlap
                        )
            native.assert_not_called()

    def test_rejects_native_chunks_with_incorrect_source_offsets(self) -> None:
        native_chunk = SimpleNamespace(
            text="Different text", start_index=0, end_index=5
        )
        fake = Mock()
        fake.chunk.return_value = [native_chunk]

        with patch("sherloque.chunking.ChonkieSentenceChunker", return_value=fake):
            chunker = SentenceChunker(
                chunk_size_tokens=20, chunk_overlap_tokens=0
            )

        with self.assertRaisesRegex(ValueError, "source offsets"):
            chunker.chunk("Hello world")
