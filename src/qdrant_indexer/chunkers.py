"""Text chunking strategies for document processing."""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup, Tag
from fastembed import TextEmbedding

from qdrant_indexer.models import CodeSymbol

logger = logging.getLogger(__name__)


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


class HTMLChunker(Chunker):
    """HTML-aware chunker that splits on semantic tag boundaries.

    Splits HTML documents at semantic tag boundaries (article, section, main, etc.)
    or heading tags (h1-h6) while respecting chunk size limits. Falls back to
    RecursiveChunker for oversized sections or documents without structure.
    Output is clean text with HTML tags stripped.
    """

    # Semantic tags to split on (in order of preference)
    SEMANTIC_TAGS = ["article", "section", "main", "aside", "header", "footer", "nav"]
    # Heading tags for fallback splitting
    HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
    # Tags to remove entirely
    REMOVE_TAGS = ["script", "style", "noscript"]

    def __init__(
        self,
        chunk_size: int = 1500,
        overlap: int = 100,
        preserve_tags: list[str] | None = None,
    ):
        """Initialize the HTML chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            overlap: Number of characters to overlap between chunks.
            preserve_tags: Optional list of tags whose structure to preserve
                          (not currently used, reserved for future).
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.preserve_tags = preserve_tags

    def _clean_soup(self, soup: BeautifulSoup) -> None:
        """Remove script, style, and other unwanted tags from soup in place.

        Args:
            soup: BeautifulSoup object to clean.
        """
        for tag_name in self.REMOVE_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

    def _extract_text(self, element: Tag | BeautifulSoup) -> str:
        """Extract clean text from an HTML element.

        Args:
            element: BeautifulSoup Tag or soup object.

        Returns:
            Clean text with whitespace normalized.
        """
        text = element.get_text(separator=" ", strip=True)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _split_by_semantic_tags(self, soup: BeautifulSoup) -> list[str]:
        """Split HTML by semantic tags (article, section, main, etc.).

        Args:
            soup: Cleaned BeautifulSoup object.

        Returns:
            List of text chunks, or empty list if no semantic tags found.
        """
        chunks: list[str] = []

        # Find all semantic tags
        semantic_elements = soup.find_all(self.SEMANTIC_TAGS)

        if not semantic_elements:
            return []

        for element in semantic_elements:
            text = self._extract_text(element)
            if text:
                chunks.append(text)

        return chunks

    def _split_by_headings(self, soup: BeautifulSoup) -> list[str]:
        """Split HTML by heading tags (h1-h6).

        Args:
            soup: Cleaned BeautifulSoup object.

        Returns:
            List of text chunks, or empty list if no headings found.
        """
        chunks: list[str] = []

        # Find all headings
        headings = soup.find_all(self.HEADING_TAGS)

        if not headings:
            return []

        # Collect content between headings
        for i, heading in enumerate(headings):
            section_text_parts = [self._extract_text(heading)]

            # Collect siblings until next heading or end
            current = heading.next_sibling
            while current:
                if isinstance(current, Tag):
                    if current.name in self.HEADING_TAGS:
                        break
                    # Skip if this element contains a heading (nested structure)
                    if current.find(self.HEADING_TAGS):
                        break
                    text = self._extract_text(current)
                    if text:
                        section_text_parts.append(text)
                current = current.next_sibling

            section_text = " ".join(section_text_parts)
            if section_text.strip():
                chunks.append(section_text.strip())

        return chunks

    def _handle_table(self, table: Tag) -> list[str]:
        """Handle a table element, keeping it together or splitting by rows.

        Args:
            table: A table Tag element.

        Returns:
            List of text chunks from the table.
        """
        full_text = self._extract_text(table)

        # If table fits in one chunk, return as-is
        if len(full_text) <= self.chunk_size:
            return [full_text] if full_text else []

        # Table too large, split by rows
        chunks: list[str] = []
        current_chunk = ""

        rows = table.find_all("tr")
        for row in rows:
            row_text = self._extract_text(row)
            if not row_text:
                continue

            # Try to add row to current chunk
            if current_chunk:
                test_chunk = current_chunk + " | " + row_text
            else:
                test_chunk = row_text

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)

                # Check if row itself is too large
                if len(row_text) > self.chunk_size:
                    # Split the row text
                    row_chunks = RecursiveChunker(
                        self.chunk_size, self.overlap
                    ).chunk(row_text)
                    chunks.extend(row_chunks)
                    current_chunk = ""
                else:
                    current_chunk = row_text

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _process_tables(self, soup: BeautifulSoup) -> list[str]:
        """Process all tables in the document.

        Args:
            soup: Cleaned BeautifulSoup object.

        Returns:
            List of text chunks from tables.
        """
        chunks: list[str] = []
        for table in soup.find_all("table"):
            table_chunks = self._handle_table(table)
            chunks.extend(table_chunks)
            # Remove table from soup to avoid double-processing
            table.decompose()
        return chunks

    def _split_oversized_chunks(self, chunks: list[str]) -> list[str]:
        """Split any chunks that exceed chunk_size using RecursiveChunker.

        Args:
            chunks: List of text chunks.

        Returns:
            List of chunks all within size limit.
        """
        result: list[str] = []
        recursive = RecursiveChunker(self.chunk_size, self.overlap)

        for chunk in chunks:
            if len(chunk) <= self.chunk_size:
                result.append(chunk)
            else:
                sub_chunks = recursive.chunk(chunk)
                result.extend(sub_chunks)

        return result

    def chunk(self, text: str) -> list[str]:
        """Split HTML text into chunks respecting semantic boundaries.

        The chunking strategy follows this order:
        1. Split on semantic tags (article, section, main, etc.)
        2. If no semantic tags, split on heading tags (h1-h6)
        3. If no structure, fall back to RecursiveChunker

        Tables are handled specially - kept together if possible,
        otherwise split by row.

        Args:
            text: The HTML text to chunk.

        Returns:
            List of plain text chunks (HTML tags stripped).
        """
        if not text or not text.strip():
            return []

        # Parse HTML with lxml for robustness
        soup = BeautifulSoup(text, "lxml")

        # Remove script, style, and other unwanted tags
        self._clean_soup(soup)

        # Process tables first (they get special handling)
        table_chunks = self._process_tables(soup)

        # Try semantic tag splitting first
        chunks = self._split_by_semantic_tags(soup)

        # If no semantic tags, try heading-based splitting
        if not chunks:
            chunks = self._split_by_headings(soup)

        # If still no structure, extract all text and use RecursiveChunker
        if not chunks:
            full_text = self._extract_text(soup)
            if full_text:
                chunks = RecursiveChunker(self.chunk_size, self.overlap).chunk(
                    full_text
                )

        # Add table chunks
        chunks.extend(table_chunks)

        # Split any oversized chunks
        chunks = self._split_oversized_chunks(chunks)

        # Filter empty chunks
        chunks = [c for c in chunks if c.strip()]

        return chunks


class SemanticChunker(Chunker):
    """Semantic chunker that splits text based on embedding similarity.

    Uses fastembed embeddings to compute semantic similarity between text
    segments and splits at points where similarity drops below a threshold.
    Falls back to RecursiveChunker when semantic splitting is not effective.
    """

    # Class-level model cache
    _model: TextEmbedding | None = None
    _model_name: str | None = None

    def __init__(
        self,
        chunk_size: int = 1500,
        min_chunk_size: int = 200,
        similarity_threshold: float = 0.5,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize the semantic chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters.
            min_chunk_size: Minimum chunk size; smaller chunks are merged.
            similarity_threshold: Cosine similarity threshold below which to split.
            embedding_model: Name of the fastembed model to use.
        """
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

    @classmethod
    def _get_model(cls, model_name: str) -> TextEmbedding:
        """Get or create cached TextEmbedding model.

        Args:
            model_name: Name of the embedding model.

        Returns:
            Cached or newly created TextEmbedding instance.
        """
        if cls._model is None or cls._model_name != model_name:
            cls._model = TextEmbedding(model_name=model_name)
            cls._model_name = model_name
        return cls._model

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text on double newlines.

        Args:
            text: The text to split.

        Returns:
            List of paragraph strings.
        """
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex.

        Args:
            text: The text to split.

        Returns:
            List of sentence strings.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_embeddings(self, segments: list[str]) -> list[np.ndarray]:
        """Compute embeddings for a list of text segments.

        Args:
            segments: List of text segments to embed.

        Returns:
            List of embedding vectors as numpy arrays.
        """
        model = self._get_model(self.embedding_model)
        embeddings = list(model.embed(segments))
        return [np.array(e) for e in embeddings]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Cosine similarity value between -1 and 1.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _find_split_points(
        self, embeddings: list[np.ndarray], threshold: float
    ) -> list[int]:
        """Find indices where similarity drops below threshold.

        Args:
            embeddings: List of embedding vectors.
            threshold: Similarity threshold for splitting.

        Returns:
            List of indices where splits should occur (after these indices).
        """
        split_points = []
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            if similarity < threshold:
                split_points.append(i)
        return split_points

    def _merge_segments(
        self, segments: list[str], split_points: list[int], separator: str
    ) -> list[str]:
        """Merge segments between split points.

        Args:
            segments: List of text segments.
            split_points: Indices where splits occur.
            separator: String to join segments with.

        Returns:
            List of merged chunks.
        """
        if not segments:
            return []

        if not split_points:
            return [separator.join(segments)]

        chunks = []
        start = 0

        for split_idx in split_points:
            chunk_segments = segments[start : split_idx + 1]
            if chunk_segments:
                chunks.append(separator.join(chunk_segments))
            start = split_idx + 1

        # Add remaining segments
        if start < len(segments):
            remaining = segments[start:]
            if remaining:
                chunks.append(separator.join(remaining))

        return chunks

    def _enforce_size_constraints(self, chunks: list[str]) -> list[str]:
        """Merge small chunks and split oversized ones.

        Args:
            chunks: List of text chunks.

        Returns:
            List of chunks respecting size constraints.
        """
        if not chunks:
            return []

        result = []
        current = ""

        for chunk in chunks:
            if not current:
                current = chunk
            elif len(current) < self.min_chunk_size:
                # Try to merge small chunks
                merged = current + "\n\n" + chunk
                if len(merged) <= self.chunk_size:
                    current = merged
                else:
                    # Current is small but merging exceeds limit
                    result.append(current)
                    current = chunk
            else:
                result.append(current)
                current = chunk

        if current:
            result.append(current)

        # Split any remaining oversized chunks
        final_result = []
        for chunk in result:
            if len(chunk) <= self.chunk_size:
                final_result.append(chunk)
            else:
                sub_chunks = self._fallback_chunk(chunk)
                final_result.extend(sub_chunks)

        return final_result

    def _fallback_chunk(self, text: str) -> list[str]:
        """Fall back to RecursiveChunker for text.

        Args:
            text: Text to chunk.

        Returns:
            List of chunks from RecursiveChunker.
        """
        return RecursiveChunker(self.chunk_size, overlap=0).chunk(text)

    def chunk(self, text: str) -> list[str]:
        """Split text into semantically coherent chunks.

        Algorithm:
        1. Handle edge cases (empty text, small text)
        2. Try paragraph-level splitting first
        3. Fall back to sentence-level if needed
        4. Compute embeddings and find semantic boundaries
        5. Merge segments between boundaries
        6. Enforce size constraints

        Args:
            text: The text to chunk.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        if len(text) <= self.chunk_size:
            return [text]

        # Try paragraph-level splitting first
        paragraphs = self._split_into_paragraphs(text)

        # Use paragraphs if we have enough of them with reasonable size
        if len(paragraphs) >= 3:
            avg_size = sum(len(p) for p in paragraphs) / len(paragraphs)
            if avg_size >= 100:
                return self._semantic_split(paragraphs, separator="\n\n")

        # Fall back to sentence-level splitting
        sentences = self._split_into_sentences(text)
        if len(sentences) >= 3:
            return self._semantic_split(sentences, separator=" ")

        # If we have very few segments, use fallback
        return self._fallback_chunk(text)

    def _semantic_split(self, segments: list[str], separator: str) -> list[str]:
        """Perform semantic splitting on segments.

        Args:
            segments: List of text segments (paragraphs or sentences).
            separator: String to join segments with.

        Returns:
            List of semantically coherent chunks.
        """
        try:
            embeddings = self._compute_embeddings(segments)
        except Exception as e:
            logger.warning(f"Embedding error, falling back to recursive: {e}")
            return self._fallback_chunk(separator.join(segments))

        # Find split points at similarity threshold
        split_points = self._find_split_points(embeddings, self.similarity_threshold)

        # If no split points found, try with a lower threshold
        if not split_points:
            lower_threshold = self.similarity_threshold * 0.7
            split_points = self._find_split_points(embeddings, lower_threshold)

        # If still no split points (uniform similarity), use fallback
        if not split_points:
            return self._fallback_chunk(separator.join(segments))

        # Merge segments between split points
        chunks = self._merge_segments(segments, split_points, separator)

        # Enforce size constraints
        return self._enforce_size_constraints(chunks)


# Chunker strategy registry
CHUNKERS: dict[str, type[Chunker]] = {
    "recursive": RecursiveChunker,
    "fixed": FixedSizeChunker,
    "markdown": MarkdownChunker,
    "html": HTMLChunker,
    "semantic": SemanticChunker,
    "code": CodeChunker,
}


def get_chunker(strategy: str, **kwargs) -> Chunker:
    """Get a chunker instance by strategy name.

    Args:
        strategy: Chunker strategy name. One of: 'recursive', 'fixed',
                  'markdown', 'html', 'semantic', 'code'.
        **kwargs: Arguments to pass to the chunker constructor.
                  Unsupported kwargs are silently filtered out.

    Returns:
        Configured chunker instance.

    Raises:
        ValueError: If strategy is unknown or 'auto' (which requires file_path).
    """
    import inspect

    if strategy == "auto":
        raise ValueError(
            "Strategy 'auto' requires a file path. Use get_chunker_for_file() instead."
        )

    if strategy not in CHUNKERS:
        valid = ", ".join(sorted(CHUNKERS.keys()))
        raise ValueError(f"Unknown chunker strategy '{strategy}'. Valid: {valid}")

    chunker_cls = CHUNKERS[strategy]

    # Filter kwargs to only those accepted by the chunker's __init__
    sig = inspect.signature(chunker_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return chunker_cls(**filtered_kwargs)


def get_chunker_for_file(file_path: Path, **kwargs) -> Chunker:
    """Get the appropriate chunker for a file based on its type.

    Uses the loader's preferred_chunker attribute to determine the best
    chunking strategy for the file type.

    Args:
        file_path: Path to the file (used to determine type).
        **kwargs: Arguments to pass to the chunker constructor.

    Returns:
        Configured chunker instance appropriate for the file type.
    """
    from qdrant_indexer.loaders import get_loader

    loader = get_loader(file_path)
    strategy = loader.preferred_chunker
    return get_chunker(strategy, **kwargs)
