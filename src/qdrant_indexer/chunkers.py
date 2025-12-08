"""Text chunking strategies for document processing."""

from abc import ABC, abstractmethod

from qdrant_indexer.models import CodeSymbol


class Chunker(ABC):
    """Abstract base class for text chunking strategies."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: The text to split into chunks.

        Returns:
            List of text chunks.
        """
        pass


class RecursiveChunker(Chunker):
    """Recursive character text splitter with semantic awareness.

    Splits text hierarchically using separators in order of preference:
    paragraphs -> lines -> sentences -> words.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """Initialize the chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            overlap: Number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " "]

    def chunk(self, text: str) -> list[str]:
        """Split text into chunks using recursive splitting."""
        if not text or not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = self._split_recursive(text, self.separators)
        return self._merge_with_overlap(chunks)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the given separators."""
        if not separators:
            # No more separators, force split by chunk_size
            return self._force_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        parts = text.split(separator)

        chunks = []
        current_chunk = ""

        for part in parts:
            # Add separator back except for first part
            if current_chunk:
                test_chunk = current_chunk + separator + part
            else:
                test_chunk = part

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Current chunk is full, save it
                if current_chunk:
                    chunks.append(current_chunk)

                # Check if part itself needs splitting
                if len(part) > self.chunk_size:
                    # Recursively split with next separator
                    sub_chunks = self._split_recursive(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _force_split(self, text: str) -> list[str]:
        """Force split text into chunk_size pieces when no separators work."""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def _merge_with_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1 or self.overlap <= 0:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Get overlap from end of previous chunk
            overlap_text = prev_chunk[-self.overlap :] if len(prev_chunk) >= self.overlap else prev_chunk

            # Prepend overlap to current chunk
            merged = overlap_text + current_chunk

            # Trim if exceeds chunk_size
            if len(merged) > self.chunk_size:
                merged = merged[: self.chunk_size]

            result.append(merged)

        return result


class FixedSizeChunker(Chunker):
    """Simple fixed-size chunker without semantic awareness."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """Initialize the chunker.

        Args:
            chunk_size: Size of each chunk in characters.
            overlap: Number of characters to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        """Split text into fixed-size chunks with overlap."""
        if not text or not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text]

        return split_text_with_overlap(text, self.chunk_size, self.overlap)


def split_text_with_overlap(text: str, size: int, overlap: int) -> list[str]:
    """Split text into chunks of given size with overlap.

    Args:
        text: The text to split.
        size: Maximum size of each chunk.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks with overlap.
    """
    if not text:
        return []

    if len(text) <= size:
        return [text]

    chunks = []
    step = size - overlap
    if step <= 0:
        step = size  # Prevent infinite loop if overlap >= size

    for i in range(0, len(text), step):
        chunk = text[i : i + size]
        chunks.append(chunk)
        if i + size >= len(text):
            break

    return chunks


def merge_small_chunks(chunks: list[str], min_size: int, max_size: int | None = None) -> list[str]:
    """Merge consecutive chunks that are smaller than min_size.

    Args:
        chunks: List of text chunks.
        min_size: Minimum size for a chunk.
        max_size: Maximum size after merging (optional).

    Returns:
        List of merged chunks.
    """
    if not chunks:
        return []

    if max_size is None:
        max_size = min_size * 3

    result = []
    current = ""

    for chunk in chunks:
        if not current:
            current = chunk
        elif len(current) < min_size:
            # Try to merge
            merged = current + " " + chunk
            if len(merged) <= max_size:
                current = merged
            else:
                result.append(current)
                current = chunk
        else:
            result.append(current)
            current = chunk

    if current:
        result.append(current)

    return result


class CodeChunker(Chunker):
    """Code-aware chunker that respects symbol boundaries.

    Primary strategy: One chunk per symbol.
    Fallback: Split large symbols at logical points while preserving signature.
    """

    def __init__(
        self, chunk_size: int = 512, overlap: int = 50, include_source: bool = True
    ):
        """Initialize the code chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            overlap: Number of characters to overlap for split symbols.
            include_source: Include full source code (vs docstring only).
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.include_source = include_source

    def chunk(self, text: str) -> list[str]:
        """Chunk text (fallback for non-code documents).

        For code documents, use chunk_symbols() instead.

        Args:
            text: Plain text to chunk.

        Returns:
            List of text chunks.
        """
        # Fallback to recursive chunking for plain text
        return RecursiveChunker(self.chunk_size, self.overlap).chunk(text)

    def chunk_symbol(self, symbol: CodeSymbol) -> list[str]:
        """Chunk a single code symbol.

        Args:
            symbol: The code symbol to chunk.

        Returns:
            List of text chunks for this symbol.
        """
        # Build header with symbol info
        header_parts = [f"{symbol.symbol_type}: {symbol.qualified_name}"]

        if symbol.signature:
            header_parts.append(symbol.signature)

        if symbol.docstring:
            header_parts.append(f"\nDocumentation:\n{symbol.docstring}")

        header = "\n".join(header_parts)

        # Choose content to chunk
        if self.include_source:
            full_content = f"{header}\n\nSource:\n{symbol.content}"
        else:
            full_content = header

        # If fits in one chunk, return as-is
        if len(full_content) <= self.chunk_size:
            return [full_content]

        # Symbol too large, need to split
        chunks: list[str] = []

        if self.include_source:
            # Calculate space for content in first chunk
            header_with_prefix = f"{header}\n\nSource:\n"
            remaining_space = self.chunk_size - len(header_with_prefix)

            if remaining_space > 0:
                first_chunk = header_with_prefix + symbol.content[:remaining_space]
                chunks.append(first_chunk)
                remaining = symbol.content[remaining_space:]
            else:
                # Header alone is too large, split it
                chunks = split_text_with_overlap(
                    header, self.chunk_size, self.overlap
                )
                remaining = symbol.content

            # Split remaining source code
            if remaining:
                source_chunks = split_text_with_overlap(
                    remaining, self.chunk_size, self.overlap
                )
                chunks.extend(source_chunks)
        else:
            # Docstring only mode: split header if needed
            chunks = split_text_with_overlap(
                full_content, self.chunk_size, self.overlap
            )

        return chunks

    def chunk_symbols(
        self, symbols: list[CodeSymbol]
    ) -> list[tuple[str, CodeSymbol]]:
        """Chunk a list of code symbols.

        Args:
            symbols: List of code symbols to chunk.

        Returns:
            List of (chunk_text, source_symbol) tuples.
        """
        result: list[tuple[str, CodeSymbol]] = []
        for symbol in symbols:
            symbol_chunks = self.chunk_symbol(symbol)
            for chunk_text in symbol_chunks:
                result.append((chunk_text, symbol))
        return result
