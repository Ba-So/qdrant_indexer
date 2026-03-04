"""Text chunking strategies for document processing."""

import re
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

    def __init__(self, chunk_size: int = 1532, overlap: int = 200):
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
            overlap_text = (
                prev_chunk[-self.overlap :]
                if len(prev_chunk) >= self.overlap
                else prev_chunk
            )

            # Prepend overlap to current chunk
            merged = overlap_text + current_chunk

            # Trim if exceeds chunk_size
            if len(merged) > self.chunk_size:
                merged = merged[: self.chunk_size]

            result.append(merged)

        return result


class FixedSizeChunker(Chunker):
    """Simple fixed-size chunker without semantic awareness."""

    def __init__(self, chunk_size: int = 1536, overlap: int = 50):
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


def merge_small_chunks(
    chunks: list[str], min_size: int, max_size: int | None = None
) -> list[str]:
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


class MarkdownChunker(Chunker):
    """Markdown-aware chunker that splits on header boundaries.

    Splits markdown documents at header boundaries (# through ######)
    while respecting chunk size limits. Falls back to RecursiveChunker
    for oversized sections or documents without headers.
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        overlap: int = 100,
        min_section_size: int = 100,
    ):
        """Initialize the markdown chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            overlap: Number of characters to overlap between chunks.
            min_section_size: Minimum section size before merging with neighbors.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_section_size = min_section_size

    def _extract_frontmatter(self, text: str) -> tuple[str | None, str]:
        """Extract YAML frontmatter from text.

        Args:
            text: The markdown text.

        Returns:
            Tuple of (frontmatter content or None, remaining text).
        """
        pattern = r"^---\n(.*?)\n---\n"
        match = re.match(pattern, text, re.DOTALL)
        if match:
            frontmatter = match.group(0).strip()
            remaining = text[match.end() :]
            return frontmatter, remaining
        return None, text

    def _find_code_blocks(self, text: str) -> list[tuple[int, int]]:
        """Find code block positions in text.

        Args:
            text: The markdown text.

        Returns:
            List of (start, end) positions for code blocks.
        """
        positions = []
        pattern = r"```.*?```"
        for match in re.finditer(pattern, text, re.DOTALL):
            positions.append((match.start(), match.end()))
        return positions

    def _is_inside_code_block(
        self, pos: int, code_blocks: list[tuple[int, int]]
    ) -> bool:
        """Check if a position is inside a code block.

        Args:
            pos: Position in text.
            code_blocks: List of (start, end) code block positions.

        Returns:
            True if position is inside a code block.
        """
        for start, end in code_blocks:
            if start <= pos < end:
                return True
        return False

    def _parse_sections(self, text: str) -> list[dict[str, int | str]]:
        """Parse document into sections with headers.

        Args:
            text: The markdown text (without frontmatter).

        Returns:
            List of section dicts with keys: level, title, content, start_pos, end_pos.
        """
        if not text.strip():
            return []

        code_blocks = self._find_code_blocks(text)
        header_pattern = r"^(#{1,6})\s+(.+)$"
        sections: list[dict[str, int | str]] = []

        # Find all headers that are not inside code blocks
        header_matches = []
        for match in re.finditer(header_pattern, text, re.MULTILINE):
            if not self._is_inside_code_block(match.start(), code_blocks):
                header_matches.append(match)

        if not header_matches:
            # No headers found, return entire text as level 0
            return [
                {
                    "level": 0,
                    "title": "",
                    "content": text,
                    "start_pos": 0,
                    "end_pos": len(text),
                }
            ]

        # Handle content before first header
        first_header_pos = header_matches[0].start()
        if first_header_pos > 0:
            pre_content = text[:first_header_pos].strip()
            if pre_content:
                sections.append(
                    {
                        "level": 0,
                        "title": "",
                        "content": pre_content,
                        "start_pos": 0,
                        "end_pos": first_header_pos,
                    }
                )

        # Process each header
        for i, match in enumerate(header_matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = match.start()

            # Content ends at next header or end of text
            if i + 1 < len(header_matches):
                end_pos = header_matches[i + 1].start()
            else:
                end_pos = len(text)

            content = text[start_pos:end_pos].strip()
            sections.append(
                {
                    "level": level,
                    "title": title,
                    "content": content,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                }
            )

        return sections

    def _build_header_context(
        self, sections: list[dict[str, int | str]], current_idx: int
    ) -> str:
        """Build parent header breadcrumb for context.

        Args:
            sections: List of parsed sections.
            current_idx: Index of current section.

        Returns:
            Header context string like "# Title > ## Section > ### Subsection".
        """
        if current_idx < 0 or current_idx >= len(sections):
            return ""

        current_level = sections[current_idx]["level"]
        if current_level == 0:
            return ""

        # Collect parent headers
        parents = []
        target_level = int(current_level) - 1

        for i in range(current_idx - 1, -1, -1):
            section = sections[i]
            level = section["level"]
            if level == 0:
                continue
            if int(level) <= target_level:
                header_marker = "#" * int(level)
                parents.append(f"{header_marker} {section['title']}")
                target_level = int(level) - 1
                if target_level < 1:
                    break

        parents.reverse()
        return " > ".join(parents) if parents else ""

    def chunk(self, text: str) -> list[str]:
        """Split markdown text into chunks respecting header boundaries.

        Args:
            text: The markdown text to chunk.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        # Extract frontmatter
        frontmatter, remaining = self._extract_frontmatter(text)

        # Parse sections
        sections = self._parse_sections(remaining)

        # No headers? Fall back to RecursiveChunker
        if not sections or all(s["level"] == 0 for s in sections):
            return RecursiveChunker(self.chunk_size, self.overlap).chunk(text)

        chunks: list[str] = []

        # Add frontmatter as separate chunk if present and not empty
        if frontmatter and frontmatter.strip():
            chunks.append(frontmatter)

        # Merge small consecutive sections
        merged_sections = self._merge_small_sections(sections)

        # Process each section
        for i, section in enumerate(merged_sections):
            content = str(section["content"])
            if not content.strip():
                continue

            if len(content) <= self.chunk_size:
                chunks.append(content)
            else:
                # Section too large, use RecursiveChunker with header context
                header_context = self._build_header_context(sections, i)
                sub_chunks = RecursiveChunker(self.chunk_size, self.overlap).chunk(
                    content
                )

                for j, sub_chunk in enumerate(sub_chunks):
                    if header_context and j > 0:
                        # Prepend context to continuation chunks
                        context_prefix = f"[Context: {header_context}]\n\n"
                        if len(context_prefix) + len(sub_chunk) <= self.chunk_size:
                            sub_chunk = context_prefix + sub_chunk
                    chunks.append(sub_chunk)

        return chunks

    def _merge_small_sections(
        self, sections: list[dict[str, int | str]]
    ) -> list[dict[str, int | str]]:
        """Merge consecutive small sections.

        Args:
            sections: List of parsed sections.

        Returns:
            List of sections with small ones merged.
        """
        if not sections:
            return []

        merged: list[dict[str, int | str]] = []
        current: dict[str, int | str] | None = None

        for section in sections:
            content = str(section["content"])

            if current is None:
                current = dict(section)
                continue

            current_content = str(current["content"])

            # If current section is small, try to merge
            if len(current_content) < self.min_section_size:
                merged_content = current_content + "\n\n" + content
                if len(merged_content) <= self.chunk_size:
                    current["content"] = merged_content
                    current["end_pos"] = section["end_pos"]
                    continue

            # Can't merge, save current and start new
            merged.append(current)
            current = dict(section)

        if current is not None:
            merged.append(current)

        return merged


class CodeChunker(Chunker):
    """Code-aware chunker that respects symbol boundaries.

    Primary strategy: One chunk per symbol.
    Fallback: Split large symbols at logical points while preserving signature.
    """

    def __init__(
        self, chunk_size: int = 1536, overlap: int = 200, include_source: bool = True
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
                chunks = split_text_with_overlap(header, self.chunk_size, self.overlap)
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

    def chunk_symbols(self, symbols: list[CodeSymbol]) -> list[tuple[str, CodeSymbol]]:
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
