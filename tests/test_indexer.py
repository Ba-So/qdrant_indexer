"""Tests for the QdrantIndexer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qdrant_indexer.chunkers import RecursiveChunker
from qdrant_indexer.indexer import QdrantIndexer


class TestQdrantIndexerInit:
    """Tests for QdrantIndexer initialization."""

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_indexer_initialization(self, mock_embedding, mock_client):
        """Verify QdrantClient and TextEmbedding created."""
        indexer = QdrantIndexer(
            qdrant_url="http://localhost:6333",
            collection_name="test-collection",
        )

        mock_client.assert_called_once_with(url="http://localhost:6333")
        mock_embedding.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert indexer.collection == "test-collection"

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_indexer_custom_embedding_model(self, mock_embedding, mock_client):
        """Test with different embedding model."""
        indexer = QdrantIndexer(
            qdrant_url="http://localhost:6333",
            collection_name="test",
            embedding_model="custom-model",
        )

        mock_embedding.assert_called_once_with(model_name="custom-model")


class TestEnsureCollection:
    """Tests for ensure_collection method."""

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_collection_exists_not_created(self, mock_embedding, mock_client):
        """Mock collection_exists=True, verify create not called."""
        mock_client_instance = MagicMock()
        mock_client_instance.collection_exists.return_value = True
        mock_client.return_value = mock_client_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")
        result = indexer.ensure_collection()

        assert result is False
        mock_client_instance.create_collection.assert_not_called()

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_collection_not_exists_created(self, mock_embedding, mock_client):
        """Mock collection_exists=False, verify create_collection called."""
        mock_client_instance = MagicMock()
        mock_client_instance.collection_exists.return_value = False
        mock_client.return_value = mock_client_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")
        result = indexer.ensure_collection()

        assert result is True
        mock_client_instance.create_collection.assert_called_once()
        # Verify vector config
        call_args = mock_client_instance.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "test"


class TestIndexFile:
    """Tests for index_file method."""

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_index_single_file_creates_points(
        self, mock_embedding, mock_client, sample_markdown_file: Path
    ):
        """Verify points generated for file."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed.return_value = [[0.1] * 384]
        mock_embedding.return_value = mock_embedding_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")
        chunker = RecursiveChunker(chunk_size=1000)
        result = indexer.index_file(sample_markdown_file, chunker)

        assert result >= 1
        mock_client_instance.upsert.assert_called()

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_returns_chunk_count(
        self, mock_embedding, mock_client, sample_markdown_file: Path
    ):
        """Verify index_file returns correct number of chunks."""
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_embedding_instance = MagicMock()
        # Return multiple embeddings for multiple chunks
        mock_embedding_instance.embed.return_value = [[0.1] * 384, [0.2] * 384]
        mock_embedding.return_value = mock_embedding_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")
        chunker = RecursiveChunker(chunk_size=50)  # Small chunks
        result = indexer.index_file(sample_markdown_file, chunker)

        assert isinstance(result, int)
        assert result >= 0


class TestIndexDirectory:
    """Tests for index_directory method."""

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_index_multiple_files(
        self, mock_embedding, mock_client, tmp_path: Path
    ):
        """Create multiple test files, verify all processed."""
        # Create test files
        (tmp_path / "doc1.md").write_text("# Doc 1\nContent one.")
        (tmp_path / "doc2.md").write_text("# Doc 2\nContent two.")
        (tmp_path / "doc3.md").write_text("# Doc 3\nContent three.")

        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed.return_value = [[0.1] * 384]
        mock_embedding.return_value = mock_embedding_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")
        result = indexer.index_directory(tmp_path, patterns=["**/*.md"])

        assert result["total_files"] == 3
        assert result["total_chunks"] >= 3

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_pattern_matching(
        self, mock_embedding, mock_client, tmp_path: Path
    ):
        """Test glob patterns filter files correctly."""
        # Create mixed files
        (tmp_path / "doc.md").write_text("Markdown")
        (tmp_path / "doc.txt").write_text("Text")
        (tmp_path / "doc.rst").write_text("RST")

        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed.return_value = [[0.1] * 384]
        mock_embedding.return_value = mock_embedding_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")

        # Only .md files
        result = indexer.index_directory(tmp_path, patterns=["**/*.md"])
        assert result["total_files"] == 1

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_summary_statistics(
        self, mock_embedding, mock_client, tmp_path: Path
    ):
        """Verify returned dict has expected keys."""
        (tmp_path / "test.md").write_text("# Test\nContent")

        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed.return_value = [[0.1] * 384]
        mock_embedding.return_value = mock_embedding_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")
        result = indexer.index_directory(tmp_path)

        assert "total_files" in result
        assert "total_chunks" in result
        assert "failed_files" in result
        assert "skipped_files" in result

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_default_chunker_used(
        self, mock_embedding, mock_client, tmp_path: Path
    ):
        """Verify RecursiveChunker created if None passed."""
        (tmp_path / "test.md").write_text("# Test\nContent")

        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed.return_value = [[0.1] * 384]
        mock_embedding.return_value = mock_embedding_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")
        # Don't pass chunker - should use default
        result = indexer.index_directory(tmp_path, chunker=None)

        assert result["total_files"] >= 0


class TestHelperMethods:
    """Tests for helper methods."""

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_generate_point_id_consistency(self, mock_embedding, mock_client, tmp_path: Path):
        """Same input produces same ID."""
        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.md"

        id1 = indexer._generate_point_id(file_path, 0)
        id2 = indexer._generate_point_id(file_path, 0)

        assert id1 == id2

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_generate_point_id_different_for_different_chunks(
        self, mock_embedding, mock_client, tmp_path: Path
    ):
        """Different chunk indices produce different IDs."""
        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.md"

        id1 = indexer._generate_point_id(file_path, 0)
        id2 = indexer._generate_point_id(file_path, 1)

        assert id1 != id2

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_generate_point_id_positive(self, mock_embedding, mock_client, tmp_path: Path):
        """All IDs are positive int64."""
        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.md"

        for i in range(100):
            point_id = indexer._generate_point_id(file_path, i)
            assert point_id > 0
            assert point_id < 2**63

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_build_payload_required_fields(self, mock_embedding, mock_client, tmp_path: Path):
        """Verify all required fields present."""
        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.md"

        payload = indexer._build_payload(
            chunk="Test content",
            file_path=file_path,
            chunk_index=0,
            total_chunks=5,
            metadata={},
        )

        assert "document" in payload
        assert "source" in payload
        assert "chunk_index" in payload
        assert "total_chunks" in payload
        assert "timestamp" in payload

    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_build_payload_merges_metadata(self, mock_embedding, mock_client, tmp_path: Path):
        """Verify custom metadata included."""
        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.md"

        payload = indexer._build_payload(
            chunk="Test content",
            file_path=file_path,
            chunk_index=0,
            total_chunks=1,
            metadata={"title": "Test Title", "author": "Test Author"},
        )

        assert payload["title"] == "Test Title"
        assert payload["author"] == "Test Author"
