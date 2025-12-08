"""Data models for Qdrant Indexer."""

from dataclasses import dataclass, field
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

    Used by code loaders to represent parsed symbols from Python and PHP source
    files before chunking and indexing.

    Attributes:
        name: Symbol name (e.g., 'parse_segment').
        qualified_name: Fully qualified name (e.g., 'MyClass.parse_segment').
        symbol_type: Type of symbol ('function', 'class', 'method', 'constant', 'module').
        content: Full source code of the symbol.
        docstring: Extracted documentation (Python docstring or PHPDoc).
        signature: Function/method signature (e.g., '(data: bytes) -> Segment').
        line_start: Starting line number in source file (1-indexed).
        line_end: Ending line number in source file (1-indexed).
        parent: Parent class name for methods, None for top-level symbols.
        visibility: Access modifier for PHP ('public', 'private', 'protected'), None for Python.
        language: Source language ('python' or 'php').
        metadata: Additional language-specific metadata (e.g., decorators, base classes).
    """

    name: str
    qualified_name: str
    symbol_type: str
    content: str
    docstring: str | None
    signature: str | None
    line_start: int
    line_end: int
    parent: str | None
    visibility: str | None
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)
