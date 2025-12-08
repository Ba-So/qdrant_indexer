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
