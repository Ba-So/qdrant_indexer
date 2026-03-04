"""Core indexer for uploading documents to Qdrant."""

import hashlib
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from qdrant_indexer.chunkers import Chunker, RecursiveChunker
from qdrant_indexer.filters import filter_files
from qdrant_indexer.loaders import get_loader
from qdrant_indexer.models import CodeSymbol, IndexedFileState, SyncResult
from qdrant_indexer.state import IndexState, compute_file_hash, get_file_mtime

logger = logging.getLogger(__name__)

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_model_info(model_name: str) -> dict:
    """Get model information from FastEmbed.

    Args:
        model_name: FastEmbed model name.

    Returns:
        Dict with 'dim' (vector dimension) and 'model' (canonical name).

    Raises:
        ValueError: If model is not supported by FastEmbed.
    """
    supported = TextEmbedding.list_supported_models()
    for model in supported:
        if model["model"] == model_name:
            return model
    # List available models in error message
    available = [m["model"] for m in supported]
    raise ValueError(
        f"Unsupported embedding model: {model_name}\n"
        f"Available models: {', '.join(available[:10])}..."
    )


def model_to_vector_name(model_name: str) -> str:
    """Convert model name to a valid Qdrant vector name.

    Compatible with mcp-server-qdrant's FastEmbed naming convention:
    'fast-{model_name}' where model_name is the part after the last '/'.

    Args:
        model_name: FastEmbed model name (e.g., 'jinaai/jina-embeddings-v3').

    Returns:
        Sanitized vector name (e.g., 'fast-jina-embeddings-v3').
    """
    # Use the same naming convention as mcp-server-qdrant
    # Extract the model name after the last '/' and prefix with 'fast-'
    name = model_name.split("/")[-1].lower()
    return f"fast-{name}"


# Progress callback type: (event, current, total, message)
ProgressCallback = Callable[[str, int, int, str], None]

# Default number of workers for parallel processing
DEFAULT_WORKERS = min(4, (os.cpu_count() or 1))

# Default batch size for embedding - smaller batches use less GPU memory
DEFAULT_EMBEDDING_BATCH_SIZE = 64

# PDF extensions that need process-based parallelism (PyMuPDF is not thread-safe)
PDF_EXTENSIONS = {".pdf"}


def _load_pdf_file(args: tuple) -> dict:
    """Load a PDF file in a separate process.

    PyMuPDF is not thread-safe, so PDF files must be processed in separate
    processes rather than threads. This function is designed to be called
    via ProcessPoolExecutor.

    Args:
        args: Tuple of (file_path_str, chunk_size, overlap)

    Returns:
        Dict with 'file_path', 'chunks' (list of chunk dicts), and 'error'.
    """
    file_path_str, chunk_size, overlap = args
    file_path = Path(file_path_str)

    try:
        # Import here to avoid issues with process spawning
        from qdrant_indexer.chunkers import RecursiveChunker
        from qdrant_indexer.loaders import PDFLoader

        loader = PDFLoader()
        doc = loader.load(file_path)

        chunker = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
        chunks = chunker.chunk(doc.content)

        # Convert to serializable format
        prepared_chunks = []
        for i, chunk in enumerate(chunks):
            prepared_chunks.append(
                {
                    "text": chunk,
                    "file_path": str(file_path),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "metadata": doc.metadata,
                    "symbol": None,
                }
            )

        return {
            "file_path": str(file_path),
            "chunks": prepared_chunks,
            "error": None,
        }

    except Exception as e:
        return {
            "file_path": str(file_path),
            "chunks": [],
            "error": str(e),
        }


@dataclass
class PreparedChunk:
    """A chunk prepared for embedding and upload."""

    text: str
    file_path: Path
    chunk_index: int
    total_chunks: int
    metadata: dict
    symbol: CodeSymbol | None = None


@dataclass
class LoadedFile:
    """Result of loading and chunking a file."""

    file_path: Path
    chunks: list[PreparedChunk]
    error: str | None = None


def is_cuda_available() -> bool:
    """Check if CUDA is available for ONNX Runtime.

    Returns:
        True if CUDAExecutionProvider is available.
    """
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def get_default_providers(use_cuda: bool = False) -> list[str]:
    """Get the list of execution providers to use.

    Args:
        use_cuda: Whether to attempt using CUDA if available.

    Returns:
        List of provider names in priority order.
    """
    if use_cuda:
        if is_cuda_available():
            logger.info("CUDA is available, using GPU acceleration")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")

    return ["CPUExecutionProvider"]


class QdrantIndexer:
    """Orchestrates document loading, chunking, embedding, and uploading to Qdrant.

    Attributes:
        client: QdrantClient instance for database operations.
        collection: Name of the Qdrant collection to use.
        embeddings: TextEmbedding model for generating vectors.
        use_cuda: Whether GPU acceleration is enabled.
    """

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        use_cuda: bool | None = None,
    ):
        """Initialize the indexer.

        Args:
            qdrant_url: URL of the Qdrant server.
            collection_name: Name of the collection to index into.
            embedding_model: FastEmbed model name for embeddings.
            use_cuda: Enable CUDA/GPU acceleration. If None, auto-detect from
                      QDRANT_INDEXER_USE_CUDA environment variable.
        """
        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection_name
        self.embedding_model = embedding_model

        # Auto-detect CUDA from environment if not explicitly set
        if use_cuda is None:
            use_cuda = os.environ.get("QDRANT_INDEXER_USE_CUDA", "").lower() in (
                "1",
                "true",
                "yes",
            )

        self.use_cuda = use_cuda

        # Get model info for vector dimensions
        model_info = get_model_info(embedding_model)
        self._vector_size = model_info["dim"]
        self._vector_name = model_to_vector_name(embedding_model)

        # Configure execution providers
        providers = get_default_providers(use_cuda)

        self.embeddings = TextEmbedding(
            model_name=embedding_model,
            providers=providers,
        )

        logger.debug(
            f"Initialized indexer for collection '{collection_name}' at {qdrant_url} "
            f"with model '{embedding_model}' (dim={self._vector_size}, cuda={self.use_cuda})"
        )

    def ensure_collection(self) -> bool:
        """Ensure the collection exists, creating it if necessary.

        Returns:
            True if collection was created, False if it already existed.
        """
        if self.client.collection_exists(self.collection):
            logger.debug(f"Collection '{self.collection}' already exists")
            return False

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                self._vector_name: VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            },
        )
        logger.info(f"Created collection '{self.collection}'")
        return True

    def delete_file_chunks(self, file_path: Path) -> int:
        """Delete all chunks for a file from the database.

        Uses filter query on source field to find and delete points.

        Args:
            file_path: Path to the file whose chunks should be deleted.

        Returns:
            Number of points deleted.
        """
        source = str(file_path.absolute())

        # Query to count points before deletion
        result = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
            limit=10000,  # Reasonable limit
            with_payload=False,
            with_vectors=False,
        )

        point_ids = [point.id for point in result[0]]

        if point_ids:
            self.client.delete(
                collection_name=self.collection,
                points_selector=point_ids,
            )
            logger.info(f"Deleted {len(point_ids)} chunks for {file_path.name}")

        return len(point_ids)

    def delete_points_by_ids(self, point_ids: list[int]) -> None:
        """Delete specific points by their IDs.

        Args:
            point_ids: List of point IDs to delete.
        """
        if not point_ids:
            return

        self.client.delete(
            collection_name=self.collection,
            points_selector=point_ids,
        )
        logger.debug(f"Deleted {len(point_ids)} points")

    def index_file(
        self,
        file_path: Path,
        chunker: Chunker,
        batch_size: int = 100,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[int, list[int]]:
        """Index a single file into Qdrant.

        Args:
            file_path: Path to the file to index.
            chunker: Chunker instance to split the document.
            batch_size: Number of points to upload per batch.
            on_progress: Optional callback for progress updates.

        Returns:
            Tuple of (chunk_count, list of point IDs).
        """
        logger.debug(f"Loading file: {file_path}")
        loader = get_loader(file_path)
        doc = loader.load(file_path)

        # Check if this is a code document with symbols
        if doc.metadata.get("is_code") and "symbols" in doc.metadata:
            return self._index_code_file(
                doc, file_path, chunker, batch_size, on_progress
            )
        else:
            return self._index_regular_file(
                doc, file_path, chunker, batch_size, on_progress
            )

    def _index_regular_file(
        self,
        doc,
        file_path: Path,
        chunker: Chunker,
        batch_size: int,
        on_progress: ProgressCallback | None,
    ) -> tuple[int, list[int]]:
        """Index regular document (non-code).

        Args:
            doc: Loaded document.
            file_path: Path to the file.
            chunker: Chunker instance to split the document.
            batch_size: Number of points to upload per batch.
            on_progress: Optional callback for progress updates.

        Returns:
            Tuple of (chunk_count, list of point IDs).
        """
        logger.debug(f"Chunking content ({len(doc.content)} chars)")
        chunks = chunker.chunk(doc.content)
        if not chunks:
            logger.debug(f"No chunks generated for {file_path}")
            return 0, []

        total_chunks = len(chunks)
        points_batch: list[PointStruct] = []
        point_ids: list[int] = []

        if on_progress:
            on_progress("embedding", 0, total_chunks, f"Embedding {file_path.name}")

        # Generate embeddings for all chunks at once (more efficient)
        logger.debug(f"Generating embeddings for {total_chunks} chunks")
        embeddings = list(self.embeddings.embed(chunks))

        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = self._generate_point_id(file_path, i)
            point_ids.append(point_id)
            payload = self._build_payload(
                chunk=chunk,
                file_path=file_path,
                chunk_index=i,
                total_chunks=total_chunks,
                metadata=doc.metadata,
            )

            points_batch.append(
                PointStruct(
                    id=point_id,
                    vector={self._vector_name: list(vector)},
                    payload=payload,
                )
            )

            if len(points_batch) >= batch_size:
                logger.debug(f"Uploading batch of {len(points_batch)} points")
                self.client.upsert(
                    collection_name=self.collection,
                    points=points_batch,
                )
                points_batch = []

                if on_progress:
                    on_progress(
                        "upload",
                        i + 1,
                        total_chunks,
                        f"Uploaded {i + 1}/{total_chunks}",
                    )

        # Upload remaining points
        if points_batch:
            logger.debug(f"Uploading final batch of {len(points_batch)} points")
            self.client.upsert(
                collection_name=self.collection,
                points=points_batch,
            )

        logger.info(f"Indexed {file_path.name}: {total_chunks} chunks")
        return total_chunks, point_ids

    def _index_code_file(
        self,
        doc,
        file_path: Path,
        chunker: Chunker,
        batch_size: int,
        on_progress: ProgressCallback | None,
    ) -> tuple[int, list[int]]:
        """Index code file with symbol metadata.

        Args:
            doc: Loaded document with symbols.
            file_path: Path to the file.
            chunker: Chunker instance (may be code-aware).
            batch_size: Number of points to upload per batch.
            on_progress: Optional callback for progress updates.

        Returns:
            Tuple of (chunk_count, list of point IDs).
        """
        symbols = doc.metadata["symbols"]
        if not symbols:
            logger.debug(f"No symbols extracted from {file_path}")
            return 0, []

        # Try to use CodeChunker if available
        try:
            from qdrant_indexer.chunkers import CodeChunker

            if isinstance(chunker, CodeChunker):
                # Use specialized code chunker
                chunks_with_symbols = chunker.chunk_symbols(symbols)
            else:
                # Fallback: convert symbols to text for regular chunker
                chunks_with_symbols = self._fallback_chunk_symbols(symbols, chunker)
        except ImportError:
            # CodeChunker not available, use fallback
            chunks_with_symbols = self._fallback_chunk_symbols(symbols, chunker)

        if not chunks_with_symbols:
            logger.debug(f"No chunks generated from symbols in {file_path}")
            return 0, []

        total_chunks = len(chunks_with_symbols)
        points_batch: list[PointStruct] = []
        point_ids: list[int] = []

        if on_progress:
            on_progress("embedding", 0, total_chunks, f"Embedding {file_path.name}")

        # Generate embeddings for all chunks
        logger.debug(f"Generating embeddings for {total_chunks} code chunks")
        chunk_texts = [chunk_text for chunk_text, _ in chunks_with_symbols]
        embeddings = list(self.embeddings.embed(chunk_texts))

        for i, ((chunk_text, symbol), vector) in enumerate(
            zip(chunks_with_symbols, embeddings)
        ):
            point_id = self._generate_point_id(file_path, i)
            point_ids.append(point_id)
            payload = self._build_code_payload(
                chunk=chunk_text,
                symbol=symbol,
                file_path=file_path,
                chunk_index=i,
                total_chunks=total_chunks,
                metadata=doc.metadata,
            )

            points_batch.append(
                PointStruct(
                    id=point_id,
                    vector={self._vector_name: list(vector)},
                    payload=payload,
                )
            )

            if len(points_batch) >= batch_size:
                logger.debug(f"Uploading batch of {len(points_batch)} points")
                self.client.upsert(
                    collection_name=self.collection,
                    points=points_batch,
                )
                points_batch = []

                if on_progress:
                    on_progress(
                        "upload",
                        i + 1,
                        total_chunks,
                        f"Uploaded {i + 1}/{total_chunks}",
                    )

        # Upload remaining points
        if points_batch:
            logger.debug(f"Uploading final batch of {len(points_batch)} points")
            self.client.upsert(
                collection_name=self.collection,
                points=points_batch,
            )

        logger.info(f"Indexed {file_path.name}: {total_chunks} code chunks")
        return total_chunks, point_ids

    def _fallback_chunk_symbols(
        self, symbols: list[CodeSymbol], chunker: Chunker
    ) -> list[tuple[str, CodeSymbol]]:
        """Fallback chunking for symbols when CodeChunker is not available.

        Args:
            symbols: List of code symbols.
            chunker: Regular chunker to use.

        Returns:
            List of (chunk_text, symbol) tuples.
        """
        chunks_with_symbols = []
        for symbol in symbols:
            # Create searchable text from symbol
            context = f"{symbol.symbol_type}: {symbol.qualified_name}\n"
            if symbol.signature:
                context += f"{symbol.signature}\n"
            if symbol.docstring:
                context += f"\n{symbol.docstring}"

            # Chunk the context (though typically symbols won't be too long)
            chunks = chunker.chunk(context)
            for chunk in chunks:
                chunks_with_symbols.append((chunk, symbol))

        return chunks_with_symbols

    def _load_and_chunk_file(self, file_path: Path, chunker: Chunker) -> LoadedFile:
        """Load a file and prepare chunks for embedding.

        This method is designed to be called in parallel threads.

        Args:
            file_path: Path to the file to load.
            chunker: Chunker instance for splitting content.

        Returns:
            LoadedFile with prepared chunks or error.
        """
        try:
            loader = get_loader(file_path)
            doc = loader.load(file_path)

            prepared_chunks: list[PreparedChunk] = []

            # Check if this is a code document with symbols
            if doc.metadata.get("is_code") and "symbols" in doc.metadata:
                symbols = doc.metadata["symbols"]
                if symbols:
                    # Try to use CodeChunker if available
                    try:
                        from qdrant_indexer.chunkers import CodeChunker

                        if isinstance(chunker, CodeChunker):
                            chunks_with_symbols = chunker.chunk_symbols(symbols)
                        else:
                            chunks_with_symbols = self._fallback_chunk_symbols(
                                symbols, chunker
                            )
                    except ImportError:
                        chunks_with_symbols = self._fallback_chunk_symbols(
                            symbols, chunker
                        )

                    for i, (chunk_text, symbol) in enumerate(chunks_with_symbols):
                        prepared_chunks.append(
                            PreparedChunk(
                                text=chunk_text,
                                file_path=file_path,
                                chunk_index=i,
                                total_chunks=len(chunks_with_symbols),
                                metadata=doc.metadata,
                                symbol=symbol,
                            )
                        )
            else:
                # Regular document
                chunks = chunker.chunk(doc.content)
                for i, chunk in enumerate(chunks):
                    prepared_chunks.append(
                        PreparedChunk(
                            text=chunk,
                            file_path=file_path,
                            chunk_index=i,
                            total_chunks=len(chunks),
                            metadata=doc.metadata,
                            symbol=None,
                        )
                    )

            return LoadedFile(file_path=file_path, chunks=prepared_chunks)

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return LoadedFile(file_path=file_path, chunks=[], error=str(e))

    def index_directory(
        self,
        path: Path,
        patterns: list[str] | None = None,
        chunker: Chunker | None = None,
        batch_size: int = 100,
        exclude_patterns: list[str] | None = None,
        on_progress: ProgressCallback | None = None,
        workers: int = DEFAULT_WORKERS,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> dict:
        """Index all matching files in a directory with parallel processing.

        Args:
            path: Directory path to index.
            patterns: Glob patterns for file matching (defaults to common doc types).
            chunker: Chunker instance (defaults to RecursiveChunker).
            batch_size: Number of points to upload per batch.
            exclude_patterns: Additional glob patterns to exclude.
            on_progress: Optional callback for progress updates.
            workers: Number of parallel workers for file loading (default: CPU count, max 4).
            embedding_batch_size: Number of chunks to embed at once (default: 64).
                Smaller values use less GPU memory.

        Returns:
            Summary dict with total_files, total_chunks, failed_files, and skipped_files.
        """
        if patterns is None:
            patterns = ["**/*.md", "**/*.txt", "**/*.pdf", "**/*.rst"]

        if chunker is None:
            chunker = RecursiveChunker()

        # Discover files first
        all_files = []
        seen = set()
        for pattern in patterns:
            for f in path.glob(pattern):
                if f.is_file() and f not in seen:
                    all_files.append(f)
                    seen.add(f)

        # Apply exclusion filters
        files, skipped = filter_files(all_files, path, exclude_patterns)
        total_files_to_process = len(files)

        if skipped:
            logger.info(f"Skipped {len(skipped)} files due to exclusion patterns")

        patterns_str = ", ".join(patterns)
        logger.info(
            f"Found {total_files_to_process} files matching patterns: {patterns_str}"
        )

        if on_progress:
            on_progress(
                "discovery",
                total_files_to_process,
                total_files_to_process,
                f"Found {total_files_to_process} files",
            )

        # Phase 1: Parallel file loading and chunking
        # Separate PDF files (need ProcessPoolExecutor) from other files (ThreadPoolExecutor)
        pdf_files = [f for f in files if f.suffix.lower() in PDF_EXTENSIONS]
        other_files = [f for f in files if f.suffix.lower() not in PDF_EXTENSIONS]

        logger.info(f"Loading files with {workers} workers...")
        if pdf_files:
            logger.info(f"  {len(pdf_files)} PDF files (process pool)")
        if other_files:
            logger.info(f"  {len(other_files)} other files (thread pool)")

        if on_progress:
            on_progress("loading", 0, total_files_to_process, "Loading files...")

        loaded_files: list[LoadedFile] = []
        failed_files: list[str] = []
        files_loaded = 0

        # Get chunker settings for PDF process pool
        chunk_size = chunker.chunk_size if hasattr(chunker, "chunk_size") else 1536
        overlap = chunker.overlap if hasattr(chunker, "overlap") else 200

        # Process PDF files with ProcessPoolExecutor (PyMuPDF is not thread-safe)
        if pdf_files:
            pdf_args = [(str(f), chunk_size, overlap) for f in pdf_files]

            with ProcessPoolExecutor(max_workers=workers) as executor:
                for result in executor.map(_load_pdf_file, pdf_args):
                    files_loaded += 1
                    file_path = Path(result["file_path"])

                    if result["error"]:
                        failed_files.append(f"{file_path}: {result['error']}")
                        if on_progress:
                            on_progress(
                                "file_error",
                                files_loaded,
                                total_files_to_process,
                                f"Failed: {file_path.name}",
                            )
                    else:
                        # Convert dict chunks back to PreparedChunk objects
                        chunks = [
                            PreparedChunk(
                                text=c["text"],
                                file_path=Path(c["file_path"]),
                                chunk_index=c["chunk_index"],
                                total_chunks=c["total_chunks"],
                                metadata=c["metadata"],
                                symbol=c["symbol"],
                            )
                            for c in result["chunks"]
                        ]
                        loaded_files.append(
                            LoadedFile(file_path=file_path, chunks=chunks)
                        )
                        if on_progress:
                            on_progress(
                                "file_loaded",
                                files_loaded,
                                total_files_to_process,
                                f"Loaded {file_path.name}: {len(chunks)} chunks",
                            )

        # Process other files with ThreadPoolExecutor
        if other_files:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all file loading tasks
                future_to_file = {
                    executor.submit(self._load_and_chunk_file, f, chunker): f
                    for f in other_files
                }

                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    files_loaded += 1

                    try:
                        result = future.result()
                        if result.error:
                            failed_files.append(f"{result.file_path}: {result.error}")
                            if on_progress:
                                on_progress(
                                    "file_error",
                                    files_loaded,
                                    total_files_to_process,
                                    f"Failed: {file_path.name}",
                                )
                        else:
                            loaded_files.append(result)
                            if on_progress:
                                on_progress(
                                    "file_loaded",
                                    files_loaded,
                                    total_files_to_process,
                                    f"Loaded {file_path.name}: {len(result.chunks)} chunks",
                                )
                    except Exception as e:
                        failed_files.append(f"{file_path}: {e}")
                        logger.error(f"Failed to load {file_path}: {e}")

        # Phase 2: Batch embedding across all files
        # Collect all chunks for efficient batch embedding
        all_chunks: list[PreparedChunk] = []
        for loaded_file in loaded_files:
            all_chunks.extend(loaded_file.chunks)

        if not all_chunks:
            logger.info("No chunks to index")
            return {
                "total_files": 0,
                "total_chunks": 0,
                "failed_files": failed_files,
                "skipped_files": len(skipped),
            }

        total_chunks = len(all_chunks)
        logger.info(
            f"Generating embeddings for {total_chunks} chunks (batch size: {embedding_batch_size})..."
        )

        if on_progress:
            on_progress(
                "embedding", 0, total_chunks, f"Embedding {total_chunks} chunks..."
            )

        # Generate embeddings in batches to avoid GPU OOM
        chunk_texts = [c.text for c in all_chunks]
        embeddings: list = []

        for i in range(0, len(chunk_texts), embedding_batch_size):
            batch = chunk_texts[i : i + embedding_batch_size]
            batch_embeddings = list(self.embeddings.embed(batch))
            embeddings.extend(batch_embeddings)

            if on_progress:
                completed = min(i + embedding_batch_size, total_chunks)
                on_progress(
                    "embedding",
                    completed,
                    total_chunks,
                    f"Embedding {completed}/{total_chunks} chunks...",
                )

        if on_progress:
            on_progress("embedding", total_chunks, total_chunks, "Embeddings complete")

        # Phase 3: Build points
        logger.info("Preparing points for upload...")
        if on_progress:
            on_progress("preparing", 0, total_chunks, "Preparing points...")

        all_points: list[PointStruct] = []
        for i, (chunk, vector) in enumerate(zip(all_chunks, embeddings)):
            point_id = self._generate_point_id(chunk.file_path, chunk.chunk_index)

            if chunk.symbol:
                payload = self._build_code_payload(
                    chunk=chunk.text,
                    symbol=chunk.symbol,
                    file_path=chunk.file_path,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    metadata=chunk.metadata,
                )
            else:
                payload = self._build_payload(
                    chunk=chunk.text,
                    file_path=chunk.file_path,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    metadata=chunk.metadata,
                )

            all_points.append(
                PointStruct(
                    id=point_id,
                    vector={self._vector_name: list(vector)},
                    payload=payload,
                )
            )

            # Update progress every 100 points
            if on_progress and (i + 1) % 100 == 0:
                on_progress(
                    "preparing",
                    i + 1,
                    total_chunks,
                    f"Preparing {i + 1}/{total_chunks} points...",
                )

        if on_progress:
            on_progress("preparing", total_chunks, total_chunks, "Points prepared")

        # Phase 4: Upload in batches
        logger.info(f"Uploading to Qdrant in batches of {batch_size}...")
        if on_progress:
            on_progress("uploading", 0, total_chunks, "Uploading to Qdrant...")

        uploaded_count = 0
        for i in range(0, len(all_points), batch_size):
            points_batch = all_points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection,
                points=points_batch,
            )
            uploaded_count += len(points_batch)

            if on_progress:
                on_progress(
                    "uploading",
                    uploaded_count,
                    total_chunks,
                    f"Uploaded {uploaded_count}/{total_chunks}",
                )

        if on_progress:
            on_progress("uploading", total_chunks, total_chunks, "Upload complete")

        logger.info(
            f"Indexing complete: {len(loaded_files)} files, {total_chunks} chunks"
        )
        return {
            "total_files": len(loaded_files),
            "total_chunks": total_chunks,
            "failed_files": failed_files,
            "skipped_files": len(skipped),
        }

    def sync_directory(
        self,
        path: Path,
        patterns: list[str] | None = None,
        chunker: Chunker | None = None,
        batch_size: int = 100,
        exclude_patterns: list[str] | None = None,
        state_file: Path | None = None,
        force: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> SyncResult:
        """Synchronize a directory with the database.

        Detects new, modified, and deleted files for incremental updates.

        Args:
            path: Directory path to synchronize.
            patterns: Glob patterns for file matching (defaults to common doc types).
            chunker: Chunker instance (defaults to RecursiveChunker).
            batch_size: Number of points to upload per batch.
            exclude_patterns: Additional glob patterns to exclude.
            state_file: Path to state file (defaults to .qdrant-index-state.json in path).
            force: Force re-indexing of all files even if unchanged.
            on_progress: Optional callback for progress updates.

        Returns:
            SyncResult with counts of added, updated, deleted, unchanged, and failed files.
        """
        if state_file is None:
            state_file = path / ".qdrant-index-state.json"

        if chunker is None:
            chunker = RecursiveChunker()

        # Load existing state
        state = IndexState(state_file)
        state.load()

        # Discover current files
        if patterns is None:
            patterns = [
                "**/*.md",
                "**/*.txt",
                "**/*.pdf",
                "**/*.rst",
                "**/*.py",
                "**/*.php",
            ]

        all_files = []
        seen = set()
        for pattern in patterns:
            for f in path.glob(pattern):
                if f.is_file() and f not in seen:
                    all_files.append(f)
                    seen.add(f)

        # Apply exclusion filters
        files, _ = filter_files(all_files, path, exclude_patterns)

        # Report discovery complete
        if on_progress:
            on_progress("sync_discovery", 0, len(files), f"Found {len(files)} files")

        # Detect changes
        current_paths = {str(f.absolute()) for f in files}
        tracked_paths = state.get_all_paths()

        added = 0
        updated = 0
        deleted = 0
        unchanged = 0
        failed = []
        files_to_process = []  # Track files that need indexing

        # Phase 1: Check which files need processing
        for i, file_path in enumerate(files):
            if on_progress:
                on_progress("sync_checking", i + 1, len(files), file_path.name)
            abs_path = str(file_path.absolute())
            current_mtime = get_file_mtime(file_path)
            file_state = state.get_file_state(file_path)

            # Determine if file needs processing using mtime pre-filter
            if file_state is None:
                # New file - must index
                content_hash = compute_file_hash(file_path)
                files_to_process.append(
                    (file_path, "new", content_hash, current_mtime, None)
                )
            elif force:
                # Forced re-index
                content_hash = compute_file_hash(file_path)
                files_to_process.append(
                    (file_path, "modified", content_hash, current_mtime, file_state)
                )
            elif file_state.mtime is None or current_mtime != file_state.mtime:
                # Mtime changed or not tracked - compute hash to confirm
                content_hash = compute_file_hash(file_path)
                if file_state.content_hash != content_hash:
                    files_to_process.append(
                        (file_path, "modified", content_hash, current_mtime, file_state)
                    )
                else:
                    # Mtime changed but content same (touched file)
                    # Update mtime in state to avoid future false positives
                    file_state.mtime = current_mtime
                    state.set_file_state(file_path, file_state)
                    unchanged += 1
            else:
                # Mtime unchanged - skip hash computation (fast path)
                unchanged += 1

        # Phase 2: Index files that need processing
        for i, (
            file_path,
            status,
            content_hash,
            current_mtime,
            file_state,
        ) in enumerate(files_to_process):
            if on_progress:
                on_progress(
                    "sync_indexing", i + 1, len(files_to_process), file_path.name
                )

            abs_path = str(file_path.absolute())
            try:
                if status == "modified" and file_state:
                    # Delete old chunks first
                    self.delete_points_by_ids(file_state.chunk_ids)

                # Index file (don't pass on_progress to index_file to avoid confusing the sync progress)
                chunk_count, chunk_ids = self.index_file(
                    file_path, chunker, batch_size, None
                )

                # Update state with mtime
                new_state = IndexedFileState(
                    path=abs_path,
                    content_hash=content_hash,
                    indexed_at=datetime.now().isoformat(),
                    chunk_count=chunk_count,
                    chunk_ids=chunk_ids,
                    mtime=current_mtime,
                )
                state.set_file_state(file_path, new_state)

                if status == "new":
                    added += 1
                    logger.info(f"Added new file: {file_path.name}")
                else:
                    updated += 1
                    logger.info(f"Updated modified file: {file_path.name}")

            except Exception as e:
                logger.error(f"Failed to sync {file_path}: {e}")
                failed.append(f"{file_path}: {e}")

        # Phase 3: Handle deleted files
        deleted_paths = list(tracked_paths - current_paths)
        for i, deleted_path in enumerate(deleted_paths):
            file_path = Path(deleted_path)
            file_state = state.get_file_state(file_path)

            if on_progress and deleted_paths:
                on_progress("sync_deleting", i + 1, len(deleted_paths), file_path.name)

            if file_state:
                try:
                    self.delete_points_by_ids(file_state.chunk_ids)
                    state.remove_file(file_path)
                    deleted += 1
                    logger.info(f"Removed deleted file: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to remove deleted file {file_path}: {e}")
                    failed.append(f"{file_path}: {e}")

        # Save updated state
        state.save()

        logger.info(
            f"Sync complete: {added} added, {updated} updated, "
            f"{deleted} deleted, {unchanged} unchanged"
        )
        return SyncResult(
            added=added,
            updated=updated,
            deleted=deleted,
            unchanged=unchanged,
            failed=failed,
        )

    def _generate_point_id(self, file_path: Path, chunk_index: int) -> int:
        """Generate a stable point ID from file path and chunk index.

        Args:
            file_path: Path to the source file.
            chunk_index: Index of the chunk within the file.

        Returns:
            Positive int64 ID.
        """
        key = f"{file_path.absolute()}-{chunk_index}"
        hash_obj = hashlib.sha256(key.encode())
        # Convert first 8 bytes to int64 and ensure positive
        return int.from_bytes(hash_obj.digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF

    def _build_payload(
        self,
        chunk: str,
        file_path: Path,
        chunk_index: int,
        total_chunks: int,
        metadata: dict,
    ) -> dict:
        """Build the payload dict for a Qdrant point.

        Args:
            chunk: The text content of the chunk.
            file_path: Path to the source file.
            chunk_index: Index of this chunk.
            total_chunks: Total number of chunks from the source.
            metadata: Additional metadata from the document loader.

        Returns:
            Payload dict with all fields.
        """
        payload = {
            "document": chunk,  # Field name required by qdrant-mcp
            "source": str(file_path.absolute()),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "timestamp": datetime.now().isoformat(),
        }
        # Merge document metadata
        payload.update(metadata)
        return payload

    def _build_code_payload(
        self,
        chunk: str,
        symbol: CodeSymbol,
        file_path: Path,
        chunk_index: int,
        total_chunks: int,
        metadata: dict,
    ) -> dict:
        """Build the payload dict for a code symbol point.

        Args:
            chunk: The text content of the chunk.
            symbol: The code symbol this chunk represents.
            file_path: Path to the source file.
            chunk_index: Index of this chunk.
            total_chunks: Total number of chunks from the source.
            metadata: Additional metadata from the document loader.

        Returns:
            Payload dict with all fields including code-specific metadata.
        """
        payload = {
            "document": chunk,  # Field name required by qdrant-mcp
            "source": str(file_path.absolute()),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "timestamp": datetime.now().isoformat(),
            # Code-specific metadata
            "language": symbol.language,
            "symbol_type": symbol.symbol_type,
            "symbol_name": symbol.name,
            "symbol_qualified_name": symbol.qualified_name,
            "signature": symbol.signature or "",
            "docstring": symbol.docstring or "",
            "line_start": symbol.line_start,
            "line_end": symbol.line_end,
            "parent_class": symbol.parent or "",
            "visibility": symbol.visibility or "",
        }
        # Merge document metadata (excluding symbols to avoid large payload)
        for key, value in metadata.items():
            if key != "symbols":
                payload[key] = value
        return payload
