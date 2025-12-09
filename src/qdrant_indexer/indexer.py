"""Core indexer for uploading documents to Qdrant."""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from qdrant_indexer.chunkers import Chunker, RecursiveChunker
from qdrant_indexer.filters import filter_files
from qdrant_indexer.loaders import get_loader
from qdrant_indexer.models import CodeSymbol

logger = logging.getLogger(__name__)

# Progress callback type: (event, current, total, message)
ProgressCallback = Callable[[str, int, int, str], None]


class QdrantIndexer:
    """Orchestrates document loading, chunking, embedding, and uploading to Qdrant.

    Attributes:
        client: QdrantClient instance for database operations.
        collection: Name of the Qdrant collection to use.
        embeddings: TextEmbedding model for generating vectors.
    """

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize the indexer.

        Args:
            qdrant_url: URL of the Qdrant server.
            collection_name: Name of the collection to index into.
            embedding_model: FastEmbed model name for embeddings.
        """
        self.client = QdrantClient(url=qdrant_url)
        self.collection = collection_name
        # Explicitly use CPU provider to avoid GPU initialization warnings
        self.embeddings = TextEmbedding(
            model_name=embedding_model,
            providers=["CPUExecutionProvider"],
        )
        self._vector_size = 384  # all-MiniLM-L6-v2 dimension
        self._vector_name = "fast-all-minilm-l6-v2"  # Required by qdrant-mcp
        logger.debug(f"Initialized indexer for collection '{collection_name}' at {qdrant_url}")

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

    def index_file(
        self,
        file_path: Path,
        chunker: Chunker,
        batch_size: int = 100,
        on_progress: ProgressCallback | None = None,
    ) -> int:
        """Index a single file into Qdrant.

        Args:
            file_path: Path to the file to index.
            chunker: Chunker instance to split the document.
            batch_size: Number of points to upload per batch.
            on_progress: Optional callback for progress updates.

        Returns:
            Number of chunks indexed.
        """
        logger.debug(f"Loading file: {file_path}")
        loader = get_loader(file_path)
        doc = loader.load(file_path)

        # Check if this is a code document with symbols
        if doc.metadata.get("is_code") and "symbols" in doc.metadata:
            return self._index_code_file(doc, file_path, chunker, batch_size, on_progress)
        else:
            return self._index_regular_file(doc, file_path, chunker, batch_size, on_progress)

    def _index_regular_file(
        self,
        doc,
        file_path: Path,
        chunker: Chunker,
        batch_size: int,
        on_progress: ProgressCallback | None,
    ) -> int:
        """Index regular document (non-code).

        Args:
            doc: Loaded document.
            file_path: Path to the file.
            chunker: Chunker instance to split the document.
            batch_size: Number of points to upload per batch.
            on_progress: Optional callback for progress updates.

        Returns:
            Number of chunks indexed.
        """
        logger.debug(f"Chunking content ({len(doc.content)} chars)")
        chunks = chunker.chunk(doc.content)
        if not chunks:
            logger.debug(f"No chunks generated for {file_path}")
            return 0

        total_chunks = len(chunks)
        points_batch: list[PointStruct] = []

        if on_progress:
            on_progress("embedding", 0, total_chunks, f"Embedding {file_path.name}")

        # Generate embeddings for all chunks at once (more efficient)
        logger.debug(f"Generating embeddings for {total_chunks} chunks")
        embeddings = list(self.embeddings.embed(chunks))

        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = self._generate_point_id(file_path, i)
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
                    on_progress("upload", i + 1, total_chunks, f"Uploaded {i + 1}/{total_chunks}")

        # Upload remaining points
        if points_batch:
            logger.debug(f"Uploading final batch of {len(points_batch)} points")
            self.client.upsert(
                collection_name=self.collection,
                points=points_batch,
            )

        logger.info(f"Indexed {file_path.name}: {total_chunks} chunks")
        return total_chunks

    def _index_code_file(
        self,
        doc,
        file_path: Path,
        chunker: Chunker,
        batch_size: int,
        on_progress: ProgressCallback | None,
    ) -> int:
        """Index code file with symbol metadata.

        Args:
            doc: Loaded document with symbols.
            file_path: Path to the file.
            chunker: Chunker instance (may be code-aware).
            batch_size: Number of points to upload per batch.
            on_progress: Optional callback for progress updates.

        Returns:
            Number of chunks indexed.
        """
        symbols = doc.metadata["symbols"]
        if not symbols:
            logger.debug(f"No symbols extracted from {file_path}")
            return 0

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
            return 0

        total_chunks = len(chunks_with_symbols)
        points_batch: list[PointStruct] = []

        if on_progress:
            on_progress("embedding", 0, total_chunks, f"Embedding {file_path.name}")

        # Generate embeddings for all chunks
        logger.debug(f"Generating embeddings for {total_chunks} code chunks")
        chunk_texts = [chunk_text for chunk_text, _ in chunks_with_symbols]
        embeddings = list(self.embeddings.embed(chunk_texts))

        for i, ((chunk_text, symbol), vector) in enumerate(zip(chunks_with_symbols, embeddings)):
            point_id = self._generate_point_id(file_path, i)
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
                    on_progress("upload", i + 1, total_chunks, f"Uploaded {i + 1}/{total_chunks}")

        # Upload remaining points
        if points_batch:
            logger.debug(f"Uploading final batch of {len(points_batch)} points")
            self.client.upsert(
                collection_name=self.collection,
                points=points_batch,
            )

        logger.info(f"Indexed {file_path.name}: {total_chunks} code chunks")
        return total_chunks

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

    def index_directory(
        self,
        path: Path,
        patterns: list[str] | None = None,
        chunker: Chunker | None = None,
        batch_size: int = 100,
        exclude_patterns: list[str] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Index all matching files in a directory.

        Args:
            path: Directory path to index.
            patterns: Glob patterns for file matching (defaults to common doc types).
            chunker: Chunker instance (defaults to RecursiveChunker).
            batch_size: Number of points to upload per batch.
            exclude_patterns: Additional glob patterns to exclude.
            on_progress: Optional callback for progress updates.

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
        logger.info(f"Found {total_files_to_process} files matching patterns: {patterns_str}")

        if on_progress:
            on_progress("discovery", total_files_to_process, total_files_to_process, f"Found {total_files_to_process} files")

        total_files = 0
        total_chunks = 0
        failed_files: list[str] = []

        for idx, file_path in enumerate(files):
            if on_progress:
                on_progress("file", idx, total_files_to_process, f"Processing {file_path.name}")

            try:
                chunks_count = self.index_file(file_path, chunker, batch_size)
                total_files += 1
                total_chunks += chunks_count

                if on_progress:
                    on_progress("file_done", idx + 1, total_files_to_process, f"Indexed {file_path.name}: {chunks_count} chunks")

            except Exception as e:
                error_msg = f"{file_path}: {e}"
                logger.error(f"Failed to index {file_path}: {e}")
                failed_files.append(error_msg)

                if on_progress:
                    on_progress("file_error", idx + 1, total_files_to_process, f"Failed: {file_path.name}")

        logger.info(f"Indexing complete: {total_files} files, {total_chunks} chunks")
        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "failed_files": failed_files,
            "skipped_files": len(skipped),
        }

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
