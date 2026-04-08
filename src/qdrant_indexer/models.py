"""Data models for Qdrant Indexer."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, TypedDict


class DocumentMetadata(TypedDict, total=False):
    """Known keys that loaders populate on :attr:`Document.metadata`.

    All keys are optional (``total=False``) because different loaders produce
    different subsets and frontmatter can contribute arbitrary extra keys.

    Common across loaders
    ---------------------
    filename:       Base name of the source file (e.g. ``"report.pdf"``).
    extension:      File suffix including the dot (e.g. ``".md"``).
    size:           File size in bytes.
    modified_time:  File modification time as a POSIX timestamp (float).

    Markdown (via python-frontmatter — any frontmatter key may appear)
    ------------------------------------------------------------------
    title:          Document title from frontmatter.
    author:         Document author from frontmatter.

    PDF (PyMuPDF metadata)
    ----------------------
    title:          PDF title metadata field.
    author:         PDF author metadata field.
    subject:        PDF subject metadata field.
    keywords:       PDF keywords metadata field.
    creator:        Application that created the PDF.
    producer:       PDF producer / library.
    creationDate:   Creation date, normalised to ``YYYY-MM-DD`` string.
    modDate:        Modification date, normalised to ``YYYY-MM-DD`` string.
    page_count:     Total number of pages.
    doi:            DOI extracted from document content (e.g. ``"10.1000/xyz"``)

    HTML / RustdocLoader
    --------------------
    description:    ``<meta name="description">`` content.
    keywords:       ``<meta name="keywords">`` content.
    doc_type:       Fixed ``"rustdoc"`` for rustdoc-generated pages.
    module_path:    Fully-qualified Rust module path from ``.fqn`` element.
    item_type:      Rust item kind (``"struct"``, ``"fn"``, ``"trait"`` …).
    signature:      Rust item signature extracted from ``.rust`` code block.

    Code loaders (PythonCodeLoader, PHPCodeLoader, RustCodeLoader)
    ---------------------------------------------------------------
    is_code:        Always ``True`` for files processed by a CodeLoader.
    symbols:        List of :class:`CodeSymbol` objects extracted from the file.
    """

    filename: str
    extension: str
    size: int
    modified_time: float
    title: str
    author: str
    subject: str
    keywords: str
    creator: str
    producer: str
    creationDate: str
    modDate: str
    page_count: int
    doi: str
    description: str
    doc_type: str
    module_path: str
    item_type: str
    signature: str
    is_code: bool
    symbols: list  # list[CodeSymbol] — typed as list to avoid a forward-ref cycle


class ChunkMetadata(DocumentMetadata, total=False):
    """Known keys on :attr:`Chunk.metadata`.

    Chunks inherit all keys from :class:`DocumentMetadata` (the document
    loader's metadata is copied verbatim onto every chunk produced from that
    document), so the same optional key set applies.  See
    :class:`DocumentMetadata` for field descriptions.
    """

    pass


class SymbolMetadata(TypedDict, total=False):
    """Known keys on :attr:`CodeSymbol.metadata`.

    Each code loader populates a different subset of these keys.  All are
    optional because no single symbol type uses all of them.

    Python (PythonCodeLoader)
    -------------------------
    decorators:   List of decorator source strings (e.g. ``["@staticmethod"]``).
    is_async:     ``True`` if the function/method is declared with ``async``.
    bases:        List of base-class name strings for class symbols.

    PHP (PHPCodeLoader)
    -------------------
    extends:      Name of the parent class (string), or ``None``.
    implements:   List of interface names the class implements.

    Rust (RustCodeLoader)
    ---------------------
    is_async:       ``True`` for ``async fn`` items.
    is_unsafe:      ``True`` for ``unsafe fn`` or ``unsafe impl`` items.
    is_const:       ``True`` for ``const fn`` items.
    generics:       List of generic parameter strings.
    lifetimes:      List of lifetime parameter strings (e.g. ``["'a"]``).
    derives:        List of derive macro names (e.g. ``["Debug", "Clone"]``).
    variants:       List of enum variant name strings.
    where_clause:   Raw ``where`` clause source text, or ``None``.
    attributes:     List of outer attribute strings (e.g. ``["#[cfg(test)]"]``).
    supertraits:    List of supertrait name strings for trait items.
    self_type:      Implementing type name for ``impl`` blocks.
    trait:          Trait being implemented (for ``impl Trait for Type`` blocks).
    aliased_type:   Right-hand side of a ``type`` alias.
    const_type:     Type annotation of a ``const`` item.
    static_type:    Type annotation of a ``static`` item.
    is_mutable:     ``True`` for ``static mut`` items.
    is_exported:    ``True`` when a ``mod`` item is ``pub``.
    """

    # Python
    decorators: list[str]
    is_async: bool
    bases: list[str]
    # PHP
    extends: str | None
    implements: list[str]
    # Rust
    is_unsafe: bool
    is_const: bool
    generics: list[str]
    lifetimes: list[str]
    derives: list[str]
    variants: list[str]
    where_clause: str | None
    attributes: list[str]
    supertraits: list[str]
    self_type: str
    trait: str | None
    aliased_type: str
    const_type: str
    static_type: str
    is_mutable: bool
    is_exported: bool


class ProgressEvent(StrEnum):
    """Progress event names emitted by the indexer via the on_progress callback.

    All events follow the signature: (event, current, total, message).
    """

    # Full-index events (index_directory)
    DISCOVERY = "discovery"
    LOADING = "loading"
    FILE_LOADED = "file_loaded"
    FILE_ERROR = "file_error"
    EMBEDDING = "embedding"
    PREPARING = "preparing"
    UPLOADING = "uploading"

    # Per-file upload events (internal to _index_regular_file / _index_code_file)
    UPLOAD = "upload"

    # Incremental-sync events (sync_directory)
    SYNC_DISCOVERY = "sync_discovery"
    SYNC_CHECKING = "sync_checking"
    SYNC_INDEXING = "sync_indexing"
    SYNC_DELETING = "sync_deleting"


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
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)


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
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)


class PointPayload(TypedDict, total=False):
    """Top-level payload stored with every Qdrant point.

    The payload structure is produced by ``_build_payload``,
    ``_build_code_payload``, and ``_build_image_payload`` in the indexer and
    must conform to the schema expected by mcp-server-qdrant.

    document
        The embedded text content (required by qdrant-mcp).
    metadata
        Nested mapping carrying all other fields.  Known keys inside
        ``metadata`` include:

        *All points*: ``source``, ``chunk_index``, ``total_chunks``,
        ``timestamp``, plus any keys from :class:`DocumentMetadata`
        (e.g. ``filename``, ``title``, ``author``).

        *Code points* (``_build_code_payload``): additionally
        ``language``, ``symbol_type``, ``symbol_name``,
        ``symbol_qualified_name``, ``signature``, ``docstring``,
        ``line_start``, ``line_end``, ``parent_class``, ``visibility``.

        *Image points* (``_build_image_payload``): additionally
        ``content_type`` (``"image"``), ``image_index``, ``total_images``,
        ``page_number``, ``width``, ``height``, ``bbox``, ``caption``,
        ``surrounding_text``, ``image_hash``.
    """

    document: str
    metadata: dict[str, Any]


@dataclass
class IndexedPoint:
    """Represents a point to be indexed in Qdrant.

    Attributes:
        id: Unique identifier for the point.
        vector: The embedding vector.
        payload: Metadata stored with the point.  See :class:`PointPayload`
            for the documented payload structure.
    """

    id: int
    vector: list[float]
    payload: PointPayload


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
    metadata: SymbolMetadata = field(default_factory=SymbolMetadata)


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
