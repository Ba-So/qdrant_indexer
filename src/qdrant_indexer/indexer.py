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

from qdrant_indexer.chunkers import Chunker, RecursiveChunker, get_chunker_for_file
from qdrant_indexer.filters import DEFAULT_INDEX_PATTERNS, glob_and_dedup, filter_files
from qdrant_indexer.loaders import get_loader
from qdrant_indexer.models import CodeSymbol, ExtractedImage, IndexedFileState, IndexResult, ProgressEvent, SyncResult
from qdrant_indexer.state import IndexState, compute_file_hash, get_file_mtime

logger = logging.getLogger(__name__)

from .config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_WORKERS, PDF_EXTENSIONS

# Default CLIP vision model for image embeddings
DEFAULT_CLIP_VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"


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


def get_clip_model_info(model_name: str) -> dict:
    """Get CLIP image model information from FastEmbed.

    Args:
        model_name: FastEmbed CLIP model name.

    Returns:
        Dict with 'dim' (vector dimension) and 'model' (canonical name).

    Raises:
        ValueError: If model is not supported by FastEmbed.
    """
    from fastembed import ImageEmbedding

    supported = ImageEmbedding.list_supported_models()
    for model in supported:
        if model["model"] == model_name:
            return model
    # List available models in error message
    available = [m["model"] for m in supported]
    raise ValueError(
        f"Unsupported CLIP model: {model_name}\n"
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


def clip_model_to_vector_name(model_name: str) -> str:
    """Convert CLIP model name to a valid Qdrant vector name.

    Uses 'clip-{model_name}' convention for image embeddings.

    Args:
        model_name: FastEmbed CLIP model name (e.g., 'Qdrant/clip-ViT-B-32-vision').

    Returns:
        Sanitized vector name (e.g., 'clip-clip-vit-b-32-vision').
    """
    name = model_name.split("/")[-1].lower()
    return f"clip-{name}"


# Progress callback type: (event, current, total, message)
ProgressCallback = Callable[[ProgressEvent, int, int, str], None]


class EmbeddingService:
    """Encapsulates text and image embedding model initialization and inference.

    Owns the FastEmbed TextEmbedding and (optionally) ImageEmbedding models.
    Has no knowledge of Qdrant, file loading, or chunking.

    Attributes:
        text_vector_size: Dimensionality of the text embedding vectors.
        text_vector_name: Qdrant-compatible vector name for text embeddings.
        clip_vector_size: Dimensionality of the CLIP image vectors (0 if disabled).
        clip_vector_name: Qdrant-compatible vector name for image embeddings ('' if disabled).
    """

    def __init__(
        self,
        embedding_model: str,
        providers: list[str],
        enable_images: bool = False,
        clip_vision_model: str = DEFAULT_CLIP_VISION_MODEL,
    ):
        """Initialize embedding models.

        Args:
            embedding_model: FastEmbed model name for text embeddings.
            providers: ONNX execution providers in priority order (e.g. CUDA, CPU).
            enable_images: Whether to prepare for CLIP image embedding.
            clip_vision_model: FastEmbed CLIP model name (used only when enable_images=True).
        """
        model_info = get_model_info(embedding_model)
        self.text_vector_size: int = model_info["dim"]
        self.text_vector_name: str = model_to_vector_name(embedding_model)

        self._text_model = TextEmbedding(model_name=embedding_model, providers=providers)
        self._providers = providers
        self._enable_images = enable_images
        self._clip_vision_model = clip_vision_model
        self._image_model = None  # lazy-initialized on first use

        if enable_images:
            clip_info = get_clip_model_info(clip_vision_model)
            self.clip_vector_size: int = clip_info["dim"]
            self.clip_vector_name: str = clip_model_to_vector_name(clip_vision_model)
        else:
            self.clip_vector_size = 0
            self.clip_vector_name = ""

    def embed_texts(self, texts: list[str]) -> list:
        """Generate text embedding vectors for a list of strings.

        Args:
            texts: Strings to embed.

        Returns:
            List of embedding vectors (one per text) in the same order.
        """
        return list(self._text_model.embed(texts))

    def embed_images(self, pil_images: list) -> list:
        """Generate CLIP embedding vectors for a list of PIL images.

        The image model is lazy-initialized on the first call.

        Args:
            pil_images: PIL Image objects to embed.

        Returns:
            List of embedding vectors (one per image) in the same order.

        Raises:
            RuntimeError: If called when enable_images=False.
        """
        if not self._enable_images:
            raise RuntimeError("Image embedding is not enabled for this EmbeddingService")
        if self._image_model is None:
            from fastembed import ImageEmbedding

            self._image_model = ImageEmbedding(
                model_name=self._clip_vision_model,
                providers=self._providers,
            )
        return list(self._image_model.embed(pil_images))


def _load_pdf_file(args: tuple) -> dict:
    """Load a PDF file in a separate process.

    PyMuPDF is not thread-safe, so PDF files must be processed in separate
    processes rather than threads. This function is designed to be called
    via ProcessPoolExecutor.

    Args:
        args: Tuple of (file_path_str, chunk_size, overlap, chunker_strategy)
              chunker_strategy can be "auto" or a specific strategy name.

    Returns:
        Dict with 'file_path', 'chunks' (list of chunk dicts), 'chunker_used', and 'error'.
    """
    file_path_str, chunk_size, overlap, chunker_strategy = args
    file_path = Path(file_path_str)

    try:
        # Import here to avoid issues with process spawning
        from qdrant_indexer.chunkers import get_chunker, get_chunker_for_file
        from qdrant_indexer.loaders import PDFLoader

        loader = PDFLoader()
        doc = loader.load(file_path)

        # Select chunker based on strategy
        if chunker_strategy == "auto":
            chunker = get_chunker_for_file(
                file_path, chunk_size=chunk_size, overlap=overlap
            )
        else:
            chunker = get_chunker(
                chunker_strategy, chunk_size=chunk_size, overlap=overlap
            )

        chunker_used = type(chunker).__name__
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
            "chunker_used": chunker_used,
            "error": None,
        }

    except Exception as e:
        return {
            "file_path": str(file_path),
            "chunks": [],
            "chunker_used": None,
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
        embedder: EmbeddingService handling text and image embedding.
        use_cuda: Whether GPU acceleration is enabled.
        enable_image_embeddings: Whether image embedding is enabled.
        clip_vision_model: CLIP model name for image embeddings.
    """

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        use_cuda: bool | None = None,
        enable_image_embeddings: bool = False,
        clip_vision_model: str = DEFAULT_CLIP_VISION_MODEL,
        min_image_size: int = 100,
    ):
        """Initialize the indexer.

        Args:
            qdrant_url: URL of the Qdrant server.
            collection_name: Name of the collection to index into.
            embedding_model: FastEmbed model name for embeddings.
            use_cuda: Enable CUDA/GPU acceleration. If None, auto-detect from
                      QDRANT_INDEXER_USE_CUDA environment variable.
            enable_image_embeddings: Enable CLIP-based image embedding for PDFs.
            clip_vision_model: FastEmbed CLIP model for image embeddings.
            min_image_size: Minimum image dimension in pixels for extraction.
        """
        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection_name
        self.embedding_model = embedding_model
        self.enable_image_embeddings = enable_image_embeddings
        self.clip_vision_model = clip_vision_model
        self.min_image_size = min_image_size

        # Auto-detect CUDA from environment if not explicitly set
        if use_cuda is None:
            use_cuda = os.environ.get("QDRANT_INDEXER_USE_CUDA", "").lower() in (
                "1",
                "true",
                "yes",
            )

        self.use_cuda = use_cuda

        providers = get_default_providers(use_cuda)
        self.embedder = EmbeddingService(
            embedding_model=embedding_model,
            providers=providers,
            enable_images=enable_image_embeddings,
            clip_vision_model=clip_vision_model,
        )

        # Expose vector names/sizes for collection management
        self._vector_size = self.embedder.text_vector_size
        self._vector_name = self.embedder.text_vector_name
        if enable_image_embeddings:
            self._clip_vector_size = self.embedder.clip_vector_size
            self._clip_vector_name = self.embedder.clip_vector_name

        logger.debug(
            f"Initialized indexer for collection '{collection_name}' at {qdrant_url} "
            f"with model '{embedding_model}' (dim={self._vector_size}, cuda={self.use_cuda})"
        )
        if enable_image_embeddings:
            logger.debug(
                f"Image embeddings enabled with CLIP model '{clip_vision_model}'"
            )

    def ensure_collection(self) -> bool:
        """Ensure the collection exists, creating it if necessary.

        Returns:
            True if collection was created, False if it already existed.
        """
        if self.client.collection_exists(self.collection):
            # Check if we need to add CLIP vectors to existing collection
            if self.enable_image_embeddings:
                self._ensure_clip_vectors()
            logger.debug(f"Collection '{self.collection}' already exists")
            return False

        # Build vectors config
        vectors_config = {
            self._vector_name: VectorParams(
                size=self._vector_size,
                distance=Distance.COSINE,
            ),
        }

        # Add CLIP vector config if image embeddings are enabled
        if self.enable_image_embeddings:
            vectors_config[self._clip_vector_name] = VectorParams(
                size=self._clip_vector_size,
                distance=Distance.COSINE,
            )

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=vectors_config,
        )
        logger.info(f"Created collection '{self.collection}'")
        return True

    def _ensure_clip_vectors(self) -> None:
        """Ensure CLIP vectors exist in an existing collection.

        Adds the CLIP vector configuration if it doesn't exist yet.
        """
        collection_info = self.client.get_collection(self.collection)
        vectors_config = collection_info.config.params.vectors

        # Check if CLIP vector already exists
        if isinstance(vectors_config, dict):
            if self._clip_vector_name in vectors_config:
                logger.debug(f"CLIP vector '{self._clip_vector_name}' already exists")
                return

        # Add CLIP vector to collection
        try:
            self.client.update_collection(
                collection_name=self.collection,
                vectors_config={
                    self._clip_vector_name: VectorParams(
                        size=self._clip_vector_size,
                        distance=Distance.COSINE,
                    ),
                },
            )
            logger.info(f"Added CLIP vector '{self._clip_vector_name}' to collection")
        except Exception as e:
            logger.warning(f"Could not add CLIP vectors to existing collection: {e}")

    def delete_file_chunks(self, file_path: Path) -> int:
        """Delete all chunks for a file from the database.

        Uses filter query on source field to find and delete points.

        Args:
            file_path: Path to the file whose chunks should be deleted.

        Returns:
            Number of points deleted.
        """
        source = str(file_path.absolute())
        scroll_filter = Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
        )
        point_ids: list[int] = []
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.extend(point.id for point in batch)
            if next_offset is None:
                break
            offset = next_offset

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
        chunker: Chunker | None,
        batch_size: int = 100,
        on_progress: ProgressCallback | None = None,
        chunk_size: int = 1536,
        overlap: int = 200,
    ) -> tuple[int, list[int], int, list[int]]:
        """Index a single file into Qdrant.

        Args:
            file_path: Path to the file to index.
            chunker: Chunker instance to split the document, or None for auto-selection.
            batch_size: Number of points to upload per batch.
            on_progress: Optional callback for progress updates.
            chunk_size: Chunk size for auto-selected chunker (used when chunker is None).
            overlap: Overlap for auto-selected chunker (used when chunker is None).

        Returns:
            Tuple of (chunk_count, chunk_ids, image_count, image_ids).
        """
        logger.debug(f"Loading file: {file_path}")
        loader = get_loader(file_path)
        doc = loader.load(file_path)

        # Auto-select chunker based on file type if None
        if chunker is None:
            chunker = get_chunker_for_file(
                file_path, chunk_size=chunk_size, overlap=overlap
            )
            logger.debug(
                f"Auto-selected {type(chunker).__name__} for {file_path.name}"
            )

        # Index text content
        if doc.metadata.get("is_code") and "symbols" in doc.metadata:
            chunk_count, chunk_ids = self._index_code_file(
                doc, file_path, chunker, batch_size, on_progress
            )
        else:
            chunk_count, chunk_ids = self._index_regular_file(
                doc, file_path, chunker, batch_size, on_progress
            )

        # Extract and index images from PDFs if enabled
        image_count = 0
        image_ids: list[int] = []
        if self.enable_image_embeddings and file_path.suffix.lower() == ".pdf":
            from qdrant_indexer.loaders import PDFLoader

            pdf_loader = PDFLoader(
                extract_images=True,
                min_image_size=self.min_image_size,
            )
            images = pdf_loader.extract_images(file_path)
            if images:
                image_count, image_ids = self._index_images(
                    images, file_path, doc.metadata, batch_size
                )

        return chunk_count, chunk_ids, image_count, image_ids

    def _upsert_batched(
        self,
        collection_name: str,
        points: list[PointStruct],
        batch_size: int,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Upload a list of PointStructs to Qdrant in fixed-size batches.

        Args:
            collection_name: Target Qdrant collection.
            points: All points to upload.
            batch_size: Maximum number of points per upsert call.
            on_progress: Optional callback invoked after each batch with
                         (uploaded_so_far, total) so callers can translate to
                         their own progress event vocabulary.

        Returns:
            Total number of points uploaded.
        """
        total = len(points)
        uploaded = 0
        for i in range(0, total, batch_size):
            batch = points[i : i + batch_size]
            logger.debug(f"Uploading batch of {len(batch)} points")
            self.client.upsert(collection_name=collection_name, points=batch)
            uploaded += len(batch)
            if on_progress:
                on_progress(uploaded, total)
        return uploaded

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
        point_ids: list[int] = []

        if on_progress:
            on_progress(ProgressEvent.EMBEDDING, 0, total_chunks, f"Embedding {file_path.name}")

        # Generate embeddings for all chunks at once (more efficient)
        logger.debug(f"Generating embeddings for {total_chunks} chunks")
        embeddings = self.embedder.embed_texts(chunks)

        all_points: list[PointStruct] = []
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
            all_points.append(
                PointStruct(
                    id=point_id,
                    vector={self._vector_name: list(vector)},
                    payload=payload,
                )
            )

        def _progress(uploaded: int, total: int) -> None:
            if on_progress:
                on_progress(ProgressEvent.UPLOAD, uploaded, total, f"Uploaded {uploaded}/{total}")

        self._upsert_batched(self.collection, all_points, batch_size, _progress)

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
        point_ids: list[int] = []

        if on_progress:
            on_progress(ProgressEvent.EMBEDDING, 0, total_chunks, f"Embedding {file_path.name}")

        # Generate embeddings for all chunks
        logger.debug(f"Generating embeddings for {total_chunks} code chunks")
        chunk_texts = [chunk_text for chunk_text, _ in chunks_with_symbols]
        embeddings = self.embedder.embed_texts(chunk_texts)

        all_points: list[PointStruct] = []
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
            all_points.append(
                PointStruct(
                    id=point_id,
                    vector={self._vector_name: list(vector)},
                    payload=payload,
                )
            )

        def _progress(uploaded: int, total: int) -> None:
            if on_progress:
                on_progress(ProgressEvent.UPLOAD, uploaded, total, f"Uploaded {uploaded}/{total}")

        self._upsert_batched(self.collection, all_points, batch_size, _progress)

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

    def _load_and_chunk_file(
        self,
        file_path: Path,
        chunker: Chunker | None,
        chunk_size: int = 1536,
        overlap: int = 200,
    ) -> LoadedFile:
        """Load a file and prepare chunks for embedding.

        This method is designed to be called in parallel threads.

        Args:
            file_path: Path to the file to load.
            chunker: Chunker instance for splitting content, or None for auto-selection.
            chunk_size: Chunk size for auto-selected chunker (used when chunker is None).
            overlap: Overlap for auto-selected chunker (used when chunker is None).

        Returns:
            LoadedFile with prepared chunks or error.
        """
        try:
            loader = get_loader(file_path)
            doc = loader.load(file_path)

            # Auto-select chunker based on file type if None
            if chunker is None:
                chunker = get_chunker_for_file(
                    file_path, chunk_size=chunk_size, overlap=overlap
                )
                logger.debug(
                    f"Auto-selected {type(chunker).__name__} for {file_path.name}"
                )

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

    def _discover_files(
        self,
        path: Path,
        patterns: list[str],
        exclude_patterns: list[str] | None,
    ) -> tuple[list[Path], list[Path]]:
        """Glob for files matching patterns under path, deduplicate, then apply exclusions.

        Delegates globbing and deduplication to :func:`glob_and_dedup` and
        returns both included and skipped lists for caller logging.

        Returns:
            Tuple of (included_files, skipped_files).
        """
        all_files = glob_and_dedup(path, patterns)
        return filter_files(all_files, path, exclude_patterns)

    def _load_files_parallel(
        self,
        files: list[Path],
        chunker: Chunker | None,
        chunk_size: int,
        overlap: int,
        workers: int,
        on_progress: ProgressCallback | None,
        total_files: int,
    ) -> tuple[list[LoadedFile], list[str]]:
        """Load and chunk all files in parallel, routing PDFs to a process pool.

        PyMuPDF is not thread-safe, so PDF files are processed via
        ``ProcessPoolExecutor`` while all other file types use
        ``ThreadPoolExecutor``.

        Args:
            files: Files to load.
            chunker: Chunker instance, or None for per-file auto-selection.
            chunk_size: Chunk size forwarded to auto-selected chunkers.
            overlap: Overlap forwarded to auto-selected chunkers.
            workers: Maximum number of parallel workers for each executor.
            on_progress: Optional progress callback.
            total_files: Total file count used for progress reporting (may be
                larger than ``len(files)`` when called after discovery).

        Returns:
            Tuple of (loaded_files, failed_file_messages).
        """
        pdf_files = [f for f in files if f.suffix.lower() in PDF_EXTENSIONS]
        other_files = [f for f in files if f.suffix.lower() not in PDF_EXTENSIONS]

        logger.info(f"Loading files with {workers} workers...")
        if pdf_files:
            logger.info(f"  {len(pdf_files)} PDF files (process pool)")
        if other_files:
            logger.info(f"  {len(other_files)} other files (thread pool)")

        if on_progress:
            on_progress(ProgressEvent.LOADING, 0, total_files, "Loading files...")

        # Derive chunker parameters for the PDF process pool (must be serialisable)
        if chunker is not None:
            pdf_chunk_size = chunker.chunk_size if hasattr(chunker, "chunk_size") else chunk_size
            pdf_overlap = chunker.overlap if hasattr(chunker, "overlap") else overlap
            chunker_strategy = chunker.strategy
        else:
            pdf_chunk_size = chunk_size
            pdf_overlap = overlap
            chunker_strategy = "auto"

        loaded_files: list[LoadedFile] = []
        failed_files: list[str] = []
        files_loaded = 0

        # Process PDF files with ProcessPoolExecutor (PyMuPDF is not thread-safe)
        if pdf_files:
            pdf_args = [
                (str(f), pdf_chunk_size, pdf_overlap, chunker_strategy)
                for f in pdf_files
            ]

            with ProcessPoolExecutor(max_workers=workers) as executor:
                for result in executor.map(_load_pdf_file, pdf_args):
                    files_loaded += 1
                    file_path = Path(result["file_path"])

                    if result["error"]:
                        failed_files.append(f"{file_path}: {result['error']}")
                        if on_progress:
                            on_progress(
                                ProgressEvent.FILE_ERROR,
                                files_loaded,
                                total_files,
                                f"Failed: {file_path.name}",
                            )
                    else:
                        if result.get("chunker_used"):
                            logger.debug(
                                f"Used {result['chunker_used']} for {file_path.name}"
                            )
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
                        loaded_files.append(LoadedFile(file_path=file_path, chunks=chunks))
                        if on_progress:
                            on_progress(
                                ProgressEvent.FILE_LOADED,
                                files_loaded,
                                total_files,
                                file_path.name,
                            )

        # Process other files with ThreadPoolExecutor
        if other_files:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_file = {
                    executor.submit(
                        self._load_and_chunk_file, f, chunker, chunk_size, overlap
                    ): f
                    for f in other_files
                }

                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    files_loaded += 1
                    try:
                        result = future.result()
                        if result.error:
                            failed_files.append(f"{result.file_path}: {result.error}")
                            if on_progress:
                                on_progress(
                                    ProgressEvent.FILE_ERROR,
                                    files_loaded,
                                    total_files,
                                    f"Failed: {file_path.name}",
                                )
                        else:
                            loaded_files.append(result)
                            if on_progress:
                                on_progress(
                                    ProgressEvent.FILE_LOADED,
                                    files_loaded,
                                    total_files,
                                    file_path.name,
                                )
                    except Exception as e:
                        failed_files.append(f"{file_path}: {e}")
                        logger.error(f"Failed to load {file_path}: {e}")

        return loaded_files, failed_files

    def _embed_chunks_batched(
        self,
        all_chunks: list[PreparedChunk],
        embedding_batch_size: int,
        on_progress: ProgressCallback | None,
    ) -> list[list[float]]:
        """Generate embeddings for all chunks in fixed-size batches.

        Batching avoids GPU out-of-memory errors when the chunk list is large.

        Args:
            all_chunks: Chunks whose text should be embedded.
            embedding_batch_size: Number of chunks to embed per batch.
            on_progress: Optional progress callback.

        Returns:
            List of embedding vectors in the same order as ``all_chunks``.
        """
        total_chunks = len(all_chunks)
        logger.info(
            f"Generating embeddings for {total_chunks} chunks "
            f"(batch size: {embedding_batch_size})..."
        )

        if on_progress:
            on_progress(
                ProgressEvent.EMBEDDING, 0, total_chunks, f"Embedding {total_chunks} chunks..."
            )

        chunk_texts = [c.text for c in all_chunks]
        embeddings: list = []

        for i in range(0, len(chunk_texts), embedding_batch_size):
            batch = chunk_texts[i : i + embedding_batch_size]
            embeddings.extend(self.embedder.embed_texts(batch))

            if on_progress:
                completed = min(i + embedding_batch_size, total_chunks)
                on_progress(
                    ProgressEvent.EMBEDDING,
                    completed,
                    total_chunks,
                    f"Embedding {completed}/{total_chunks} chunks...",
                )

        if on_progress:
            on_progress(ProgressEvent.EMBEDDING, total_chunks, total_chunks, "Embeddings complete")

        return embeddings

    def _build_and_upload_points(
        self,
        all_chunks: list[PreparedChunk],
        embeddings: list,
        batch_size: int,
        on_progress: ProgressCallback | None,
    ) -> int:
        """Build PointStruct objects and upload them to Qdrant in batches.

        Args:
            all_chunks: Chunks paired positionally with ``embeddings``.
            embeddings: Embedding vectors, one per chunk.
            batch_size: Number of points per upsert call.
            on_progress: Optional progress callback.

        Returns:
            Total number of points uploaded.
        """
        total_chunks = len(all_chunks)
        logger.info("Preparing points for upload...")

        if on_progress:
            on_progress(ProgressEvent.PREPARING, 0, total_chunks, "Preparing points...")

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

            if on_progress and (i + 1) % 100 == 0:
                on_progress(
                    ProgressEvent.PREPARING,
                    i + 1,
                    total_chunks,
                    f"Preparing {i + 1}/{total_chunks} points...",
                )

        if on_progress:
            on_progress(ProgressEvent.PREPARING, total_chunks, total_chunks, "Points prepared")

        logger.info(f"Uploading to Qdrant in batches of {batch_size}...")
        if on_progress:
            on_progress(ProgressEvent.UPLOADING, 0, total_chunks, "Uploading to Qdrant...")

        def _progress(uploaded: int, total: int) -> None:
            if on_progress:
                on_progress(
                    ProgressEvent.UPLOADING,
                    uploaded,
                    total,
                    f"Uploaded {uploaded}/{total}",
                )

        uploaded_count = self._upsert_batched(
            self.collection, all_points, batch_size, _progress
        )

        if on_progress:
            on_progress(ProgressEvent.UPLOADING, total_chunks, total_chunks, "Upload complete")

        return uploaded_count

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
        chunk_size: int = 1536,
        overlap: int = 200,
    ) -> IndexResult:
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
            chunk_size: Chunk size for auto-selected chunker (used when chunker is None).
            overlap: Overlap for auto-selected chunker (used when chunker is None).

        Returns:
            IndexResult with total_files, total_chunks, failed_files, and skipped_files.
        """
        if patterns is None:
            patterns = DEFAULT_INDEX_PATTERNS

        # Note: chunker=None signals auto-selection per file
        # (handled in _load_and_chunk_file)

        # Phase 1: File discovery
        files, skipped = self._discover_files(path, patterns, exclude_patterns)
        total_files_to_process = len(files)

        if skipped:
            logger.info(f"Skipped {len(skipped)} files due to exclusion patterns")

        patterns_str = ", ".join(patterns)
        logger.info(
            f"Found {total_files_to_process} files matching patterns: {patterns_str}"
        )

        if on_progress:
            on_progress(
                ProgressEvent.DISCOVERY,
                total_files_to_process,
                total_files_to_process,
                f"Found {total_files_to_process} files",
            )

        # Phase 2: Parallel loading and chunking
        loaded_files, failed_files = self._load_files_parallel(
            files=files,
            chunker=chunker,
            chunk_size=chunk_size,
            overlap=overlap,
            workers=workers,
            on_progress=on_progress,
            total_files=total_files_to_process,
        )

        # Phase 3: Batch embedding
        all_chunks: list[PreparedChunk] = []
        for loaded_file in loaded_files:
            all_chunks.extend(loaded_file.chunks)

        if not all_chunks:
            logger.info("No chunks to index")
            return IndexResult(
                total_files=0,
                total_chunks=0,
                failed_files=failed_files,
                skipped_files=len(skipped),
            )

        embeddings = self._embed_chunks_batched(all_chunks, embedding_batch_size, on_progress)

        # Phase 4: Build points and upload
        total_chunks = len(all_chunks)
        self._build_and_upload_points(all_chunks, embeddings, batch_size, on_progress)

        logger.info(
            f"Indexing complete: {len(loaded_files)} files, {total_chunks} chunks"
        )
        return IndexResult(
            total_files=len(loaded_files),
            total_chunks=total_chunks,
            failed_files=failed_files,
            skipped_files=len(skipped),
        )

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
        chunk_size: int = 1536,
        overlap: int = 200,
    ) -> SyncResult:
        """Synchronize a directory with the database.

        Detects new, modified, and deleted files for incremental updates.

        Args:
            path: Directory path to synchronize.
            patterns: Glob patterns for file matching (defaults to common doc types).
            chunker: Chunker instance, or None for auto-selection per file.
            batch_size: Number of points to upload per batch.
            exclude_patterns: Additional glob patterns to exclude.
            state_file: Path to state file (defaults to .qdrant-index-state.json in path).
            force: Force re-indexing of all files even if unchanged.
            on_progress: Optional callback for progress updates.
            chunk_size: Chunk size for auto-selected chunker (used when chunker is None).
            overlap: Overlap for auto-selected chunker (used when chunker is None).

        Returns:
            SyncResult with counts of added, updated, deleted, unchanged, and failed files.
        """
        if state_file is None:
            state_file = path / ".qdrant-index-state.json"

        # Note: chunker=None signals auto-selection per file
        # (handled in index_file)

        # Load existing state
        state = IndexState(state_file)
        state.load()

        # Discover current files
        if patterns is None:
            patterns = DEFAULT_INDEX_PATTERNS

        files, _ = self._discover_files(path, patterns, exclude_patterns)

        # Report discovery complete
        if on_progress:
            on_progress(ProgressEvent.SYNC_DISCOVERY, 0, len(files), f"Found {len(files)} files")

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
                on_progress(ProgressEvent.SYNC_CHECKING, i + 1, len(files), file_path.name)
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
                    ProgressEvent.SYNC_INDEXING, i + 1, len(files_to_process), file_path.name
                )

            abs_path = str(file_path.absolute())
            try:
                if status == "modified" and file_state:
                    # Delete old chunks and images first
                    self.delete_points_by_ids(file_state.chunk_ids)
                    if file_state.image_ids:
                        self.delete_points_by_ids(file_state.image_ids)

                # Index file (don't pass on_progress to index_file to avoid confusing the sync progress)
                chunk_count, chunk_ids, image_count, image_ids = self.index_file(
                    file_path,
                    chunker,
                    batch_size,
                    None,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )

                # Update state with mtime
                new_state = IndexedFileState(
                    path=abs_path,
                    content_hash=content_hash,
                    indexed_at=datetime.now().isoformat(),
                    chunk_count=chunk_count,
                    chunk_ids=chunk_ids,
                    mtime=current_mtime,
                    image_count=image_count,
                    image_ids=image_ids,
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
                on_progress(ProgressEvent.SYNC_DELETING, i + 1, len(deleted_paths), file_path.name)

            if file_state:
                try:
                    self.delete_points_by_ids(file_state.chunk_ids)
                    if file_state.image_ids:
                        self.delete_points_by_ids(file_state.image_ids)
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
        # Build nested metadata object (required by mcp-server-qdrant)
        nested_metadata = {
            "source": str(file_path.absolute()),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "timestamp": datetime.now().isoformat(),
        }
        # Merge document metadata into nested metadata
        nested_metadata.update(metadata)

        return {
            "document": chunk,  # Field name required by qdrant-mcp
            "metadata": nested_metadata,
        }

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
        # Build nested metadata object (required by mcp-server-qdrant)
        nested_metadata = {
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
                nested_metadata[key] = value

        return {
            "document": chunk,  # Field name required by qdrant-mcp
            "metadata": nested_metadata,
        }

    def _generate_image_point_id(self, file_path: Path, image_index: int) -> int:
        """Generate a stable point ID for an image.

        Uses a different namespace than text chunks to avoid collisions.

        Args:
            file_path: Path to the source file.
            image_index: Index of the image within the file.

        Returns:
            Positive int64 ID.
        """
        key = f"image:{file_path.absolute()}-{image_index}"
        hash_obj = hashlib.sha256(key.encode())
        return int.from_bytes(hash_obj.digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF

    def _build_image_payload(
        self,
        image: ExtractedImage,
        file_path: Path,
        image_index: int,
        total_images: int,
        metadata: dict,
    ) -> dict:
        """Build the payload dict for an image point.

        Args:
            image: ExtractedImage object with image data and context.
            file_path: Path to the source file.
            image_index: Index of this image.
            total_images: Total number of images from the source.
            metadata: Additional metadata from the document loader.

        Returns:
            Payload dict with all fields including image-specific metadata.
        """
        # Build document text from caption and surrounding text
        doc_parts = []
        if image.caption:
            doc_parts.append(image.caption)
        if image.surrounding_text:
            doc_parts.append(image.surrounding_text)
        document_text = " ".join(doc_parts) if doc_parts else ""

        # Build nested metadata object (required by mcp-server-qdrant)
        nested_metadata = {
            "source": str(file_path.absolute()),
            "content_type": "image",
            "image_index": image_index,
            "total_images": total_images,
            "page_number": image.page_number,
            "width": image.width,
            "height": image.height,
            "bbox": list(image.bbox),
            "caption": image.caption or "",
            "surrounding_text": image.surrounding_text or "",
            "image_hash": image.image_hash or "",
            "timestamp": datetime.now().isoformat(),
        }
        # Merge document metadata (excluding symbols)
        for key, value in metadata.items():
            if key != "symbols":
                nested_metadata[key] = value

        return {
            "document": document_text,
            "metadata": nested_metadata,
        }

    def _index_images(
        self,
        images: list[ExtractedImage],
        file_path: Path,
        metadata: dict,
        batch_size: int = 100,
    ) -> tuple[int, list[int]]:
        """Index extracted images into Qdrant.

        Args:
            images: List of ExtractedImage objects.
            file_path: Path to the source file.
            metadata: Document metadata.
            batch_size: Number of points to upload per batch.

        Returns:
            Tuple of (image_count, list of point IDs).
        """
        import io

        from PIL import Image

        if not images:
            return 0, []

        total_images = len(images)
        point_ids: list[int] = []

        logger.debug(f"Generating CLIP embeddings for {total_images} images")

        # Process images and generate embeddings
        # FastEmbed ImageEmbedding accepts PIL Images
        pil_images = []
        for img in images:
            pil_img = Image.open(io.BytesIO(img.image_data))
            pil_images.append(pil_img)

        # Generate embeddings
        embeddings = self.embedder.embed_images(pil_images)

        all_points: list[PointStruct] = []
        for i, (image, vector) in enumerate(zip(images, embeddings)):
            point_id = self._generate_image_point_id(file_path, i)
            point_ids.append(point_id)

            payload = self._build_image_payload(
                image=image,
                file_path=file_path,
                image_index=i,
                total_images=total_images,
                metadata=metadata,
            )

            all_points.append(
                PointStruct(
                    id=point_id,
                    vector={self._clip_vector_name: list(vector)},
                    payload=payload,
                )
            )

        self._upsert_batched(self.collection, all_points, batch_size)

        logger.info(f"Indexed {total_images} images from {file_path.name}")
        return total_images, point_ids
