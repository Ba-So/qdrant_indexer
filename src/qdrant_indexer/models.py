"""Data models for Qdrant Indexer."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """Represents a loaded document with its content and metadata.

    Attributes:
        content: The text content of the document.
        source_path: The original file path.
        metadata: Additional metadata (frontmatter, file info, etc.).
    """

    content: str
    source_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedImage:
    """Represents an image extracted from a PDF document.

    Attributes:
        image_data: Raw image data as bytes (PNG format).
        page_number: Page number where the image was found (1-indexed).
        bbox: Bounding box as (x0, y0, x1, y1) in PDF coordinates.
        width: Image width in pixels.
        height: Image height in pixels.
        surrounding_text: Text content near the image for context.
        caption: Detected caption (e.g., "Figure 1: ...").
        image_hash: MD5 hash of image data for deduplication.
    """

    image_data: bytes
    page_number: int
    bbox: tuple[float, float, float, float]
    width: int
    height: int
    surrounding_text: str | None = None
    caption: str | None = None
    image_hash: str | None = None


@dataclass
class Chunk:
    """Represents a chunk of text from a document.

    Attributes:
        text: The chunk content.
        index: The chunk position (0-indexed).
        total_chunks: Total number of chunks from the source document.
        metadata: Metadata inherited from document plus chunk-specific info.
    """

    text: str
    index: int
    total_chunks: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexedPoint:
    """Represents a point to be indexed in Qdrant.

    Attributes:
        id: Unique identifier for the point.
        vector: The embedding vector.
        payload: Metadata stored with the point.
    """

    id: int
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class CodeSymbol:
    """Represents an extracted code symbol (function, class, method, constant).

    Used by code loaders to represent parsed symbols from Python, PHP, and Rust
    source files before chunking and indexing.

    Attributes:
        name: Symbol name (e.g., 'parse_segment').
        qualified_name: Fully qualified name (e.g., 'MyClass.parse_segment').
        symbol_type: Type of symbol ('function', 'class', 'method', 'constant', 'module').
        content: Full source code of the symbol.
        language: Source language ('python', 'php', or 'rust').
        docstring: Extracted documentation (Python docstring or PHPDoc).
        signature: Function/method signature (e.g., '(data: bytes) -> Segment').
        line_start: Starting line number in source file (1-indexed).
        line_end: Ending line number in source file (1-indexed).
        parent: Parent class name for methods, None for top-level symbols.
        visibility: Access modifier for PHP/Rust ('public', 'private', 'protected', 'pub'),
            None for Python.
        metadata: Additional language-specific metadata (e.g., decorators, base classes).
    """

    name: str
    qualified_name: str
    symbol_type: str
    content: str
    language: str
    docstring: str | None = None
    signature: str | None = None
    line_start: int = 1
    line_end: int = 1
    parent: str | None = None
    visibility: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexedFileState:
    """Tracks the state of an indexed file.

    Attributes:
        path: Absolute path to file.
        content_hash: SHA-256 of file content.
        indexed_at: ISO 8601 timestamp.
        chunk_count: Number of chunks created.
        chunk_ids: Point IDs in Qdrant.
        mtime: File modification time (for fast change detection).
        image_count: Number of images extracted and indexed from the file.
        image_ids: Point IDs for image embeddings in Qdrant.
    """

    path: str
    content_hash: str
    indexed_at: str
    chunk_count: int
    chunk_ids: list[int]
    mtime: float | None = None
    image_count: int = 0
    image_ids: list[int] = field(default_factory=list)


@dataclass
class IndexResult:
    """Result of a full directory indexing operation.

    Attributes:
        total_files: Number of files successfully indexed.
        total_chunks: Total chunks created across all files.
        failed_files: Paths of files that failed to process.
        skipped_files: Number of files skipped by exclusion patterns.
    """

    total_files: int
    total_chunks: int
    failed_files: list[str] = field(default_factory=list)
    skipped_files: int = 0


@dataclass
class SyncResult:
    """Result of a directory synchronization operation.

    Attributes:
        added: New files indexed.
        updated: Modified files re-indexed.
        deleted: Deleted files removed from DB.
        unchanged: Files skipped (no changes).
        failed: Files that failed to process.
    """

    added: int
    updated: int
    deleted: int
    unchanged: int
    failed: list[str] = field(default_factory=list)
