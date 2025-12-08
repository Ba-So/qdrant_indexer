"""Core indexer for uploading documents to Qdrant."""

import hashlib
from datetime import datetime
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from qdrant_indexer.chunkers import Chunker, RecursiveChunker
from qdrant_indexer.loaders import get_loader


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

    def ensure_collection(self) -> bool:
        """Ensure the collection exists, creating it if necessary.

        Returns:
            True if collection was created, False if it already existed.
        """
        if self.client.collection_exists(self.collection):
            return False

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self._vector_size,
                distance=Distance.COSINE,
            ),
        )
        return True

    def index_file(
        self,
        file_path: Path,
        chunker: Chunker,
        batch_size: int = 100,
    ) -> int:
        """Index a single file into Qdrant.

        Args:
            file_path: Path to the file to index.
            chunker: Chunker instance to split the document.
            batch_size: Number of points to upload per batch.

        Returns:
            Number of chunks indexed.
        """
        loader = get_loader(file_path)
        doc = loader.load(file_path)

        chunks = chunker.chunk(doc.content)
        if not chunks:
            return 0

        total_chunks = len(chunks)
        points_batch: list[PointStruct] = []

        # Generate embeddings for all chunks at once (more efficient)
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
                self.client.upsert(
                    collection_name=self.collection,
                    points=points_batch,
                )
                points_batch = []

        # Upload remaining points
        if points_batch:
            self.client.upsert(
                collection_name=self.collection,
                points=points_batch,
            )

        return total_chunks

    def index_directory(
        self,
        path: Path,
        pattern: str = "**/*.md",
        chunker: Chunker | None = None,
        batch_size: int = 100,
    ) -> dict:
        """Index all matching files in a directory.

        Args:
            path: Directory path to index.
            pattern: Glob pattern for file matching.
            chunker: Chunker instance (defaults to RecursiveChunker).
            batch_size: Number of points to upload per batch.

        Returns:
            Summary dict with total_files, total_chunks, and failed_files.
        """
        if chunker is None:
            chunker = RecursiveChunker()

        total_files = 0
        total_chunks = 0
        failed_files: list[str] = []

        for file_path in path.glob(pattern):
            if not file_path.is_file():
                continue

            try:
                chunks_count = self.index_file(file_path, chunker, batch_size)
                total_files += 1
                total_chunks += chunks_count
            except Exception as e:
                failed_files.append(f"{file_path}: {e}")

        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "failed_files": failed_files,
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
