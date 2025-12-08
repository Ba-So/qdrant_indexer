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
        self.embeddings = TextEmbedding(model_name=embedding_model)
        self._vector_size = 384  # all-MiniLM-L6-v2 dimension
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
            vectors_config=VectorParams(
                size=self._vector_size,
                distance=Distance.COSINE,
            ),
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
                    vector=list(vector),
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

    def index_directory(
        self,
        path: Path,
        pattern: str = "**/*.md",
        chunker: Chunker | None = None,
        batch_size: int = 100,
        exclude_patterns: list[str] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> dict:
        """Index all matching files in a directory.

        Args:
            path: Directory path to index.
            pattern: Glob pattern for file matching.
            chunker: Chunker instance (defaults to RecursiveChunker).
            batch_size: Number of points to upload per batch.
            exclude_patterns: Additional glob patterns to exclude.
            on_progress: Optional callback for progress updates.

        Returns:
            Summary dict with total_files, total_chunks, failed_files, and skipped_files.
        """
        if chunker is None:
            chunker = RecursiveChunker()

        # Discover files first
        all_files = [f for f in path.glob(pattern) if f.is_file()]

        # Apply exclusion filters
        files, skipped = filter_files(all_files, path, exclude_patterns)
        total_files_to_process = len(files)

        if skipped:
            logger.info(f"Skipped {len(skipped)} files due to exclusion patterns")

        logger.info(f"Found {total_files_to_process} files matching '{pattern}'")

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
            "text": chunk,
            "source": str(file_path.absolute()),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "timestamp": datetime.now().isoformat(),
        }
        # Merge document metadata
        payload.update(metadata)
        return payload
