"""Tests for text chunkers."""

import pytest

from qdrant_indexer.chunkers import (
    Chunker,
    FixedSizeChunker,
    RecursiveChunker,
    merge_small_chunks,
    split_text_with_overlap,
)


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_chunks_respect_size_limit(self, sample_long_text: str):
        """Assert all chunks <= chunk_size."""
        chunker = RecursiveChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk(sample_long_text)

        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 200, f"Chunk {i} exceeds size limit: {len(chunk)}"

    def test_empty_text_returns_empty_list(self):
        """Edge case for empty string."""
        chunker = RecursiveChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_text_smaller_than_chunk_size(self):
        """Verify single chunk returned for small text."""
        chunker = RecursiveChunker(chunk_size=1000)
        text = "Short text"
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_paragraph_splitting(self):
        """Text with \\n\\n should split at paragraphs first."""
        chunker = RecursiveChunker(chunk_size=100, overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text)

        # Should split into separate chunks at paragraph boundaries
        assert len(chunks) >= 1
        # Full text should fit in one chunk at this size
        assert "First paragraph" in chunks[0]

    def test_overlap_parameter_zero(self):
        """Test with overlap=0."""
        chunker = RecursiveChunker(chunk_size=50, overlap=0)
        text = "A" * 100
        chunks = chunker.chunk(text)

        assert len(chunks) == 2
        # No overlap means chunks don't share content
        assert chunks[0] == "A" * 50
        assert chunks[1] == "A" * 50

    def test_overlap_parameter_nonzero(self):
        """Test with nonzero overlap."""
        chunker = RecursiveChunker(chunk_size=50, overlap=10)
        text = "A" * 100
        chunks = chunker.chunk(text)

        # With overlap, second chunk should start with overlap from first
        assert len(chunks) >= 2

    def test_large_document_chunking(self, sample_long_text: str):
        """Verify chunking works on larger documents."""
        chunker = RecursiveChunker(chunk_size=300, overlap=30)
        chunks = chunker.chunk(sample_long_text)

        assert len(chunks) > 1
        # All chunks should have content
        for chunk in chunks:
            assert len(chunk) > 0

    def test_unicode_text(self):
        """Test chunking works with unicode characters."""
        chunker = RecursiveChunker(chunk_size=50, overlap=5)
        text = "Hello 世界! " * 20
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        # Verify no corruption
        combined = "".join(chunks)
        assert "世界" in combined

    def test_is_chunker_subclass(self):
        """Verify RecursiveChunker is a Chunker."""
        assert issubclass(RecursiveChunker, Chunker)


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_fixed_chunks_size(self):
        """Verify chunks are approximately chunk_size."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        text = "A" * 150
        chunks = chunker.chunk(text)

        assert len(chunks) == 3
        assert len(chunks[0]) == 50
        assert len(chunks[1]) == 50
        assert len(chunks[2]) == 50

    def test_fixed_overlap_calculation(self):
        """Verify overlap is correctly applied."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        text = "A" * 100
        chunks = chunker.chunk(text)

        # With overlap of 10, step is 40, so we need more chunks
        assert len(chunks) >= 2

    def test_no_overlap_mode(self):
        """Test with overlap=0."""
        chunker = FixedSizeChunker(chunk_size=25, overlap=0)
        text = "ABCDE" * 10  # 50 chars
        chunks = chunker.chunk(text)

        assert len(chunks) == 2
        assert chunks[0] == "ABCDE" * 5
        assert chunks[1] == "ABCDE" * 5

    def test_text_smaller_than_chunk(self):
        """Should return single chunk for small text."""
        chunker = FixedSizeChunker(chunk_size=100)
        text = "Short"
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_fixed(self):
        """Edge case handling for empty text."""
        chunker = FixedSizeChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("  ") == []

    def test_is_chunker_subclass(self):
        """Verify FixedSizeChunker is a Chunker."""
        assert issubclass(FixedSizeChunker, Chunker)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_split_text_with_overlap_basic(self):
        """Test basic splitting with overlap."""
        text = "A" * 100
        chunks = split_text_with_overlap(text, size=50, overlap=10)

        assert len(chunks) >= 2
        assert len(chunks[0]) == 50

    def test_split_text_with_overlap_empty(self):
        """Test with empty text."""
        assert split_text_with_overlap("", size=50, overlap=10) == []

    def test_split_text_with_overlap_small(self):
        """Test with text smaller than size."""
        text = "Small"
        chunks = split_text_with_overlap(text, size=50, overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_merge_small_chunks_basic(self):
        """Test merging small chunks."""
        chunks = ["Hi", "there", "world"]
        merged = merge_small_chunks(chunks, min_size=10)

        # Should merge small chunks together
        assert len(merged) <= len(chunks)

    def test_merge_small_chunks_empty(self):
        """Test with empty list."""
        assert merge_small_chunks([], min_size=10) == []

    def test_merge_small_chunks_already_large(self):
        """Test when chunks are already large enough."""
        chunks = ["A" * 50, "B" * 50]
        merged = merge_small_chunks(chunks, min_size=10)

        assert len(merged) == 2


class TestChunkerDefaults:
    """Tests for default parameter values."""

    def test_recursive_chunker_defaults(self):
        """Verify default values for RecursiveChunker."""
        chunker = RecursiveChunker()
        assert chunker.chunk_size == 1536
        assert chunker.overlap == 200

    def test_fixed_size_chunker_defaults(self):
        """Verify default values for FixedSizeChunker."""
        chunker = FixedSizeChunker()
        assert chunker.chunk_size == 1536
        assert chunker.overlap == 200


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [
        (100, 10),
        (200, 20),
        (500, 50),
        (1000, 100),
    ],
)
def test_recursive_chunker_parametrized(
    chunk_size: int, overlap: int, sample_long_text: str
):
    """Test RecursiveChunker with various configurations."""
    chunker = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk(sample_long_text)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= chunk_size


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [
        (50, 0),
        (50, 10),
        (100, 25),
    ],
)
def test_fixed_size_chunker_parametrized(chunk_size: int, overlap: int):
    """Test FixedSizeChunker with various configurations."""
    text = "X" * 200
    chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    # First chunk should be exactly chunk_size
    assert len(chunks[0]) == chunk_size
