"""Tests for the QdrantIndexer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qdrant_indexer.chunkers import RecursiveChunker
from qdrant_indexer.indexer import QdrantIndexer


class TestQdrantIndexerInit:
    """Tests for QdrantIndexer initialization."""

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_indexer_initialization(self, mock_embedding, mock_client, mock_model_info):
        """Verify QdrantClient and TextEmbedding created."""
        mock_model_info.return_value = {"dim": 384, "model": "sentence-transformers/all-MiniLM-L6-v2"}
        indexer = QdrantIndexer(
            qdrant_url="http://localhost:6333",
            collection_name="test-collection",
        )

        mock_client.assert_called_once_with(url="http://localhost:6333")
        assert indexer.collection == "test-collection"

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_indexer_custom_embedding_model(self, mock_embedding, mock_client, mock_model_info):
        """Test with different embedding model."""
        mock_model_info.return_value = {"dim": 768, "model": "custom-model"}
        indexer = QdrantIndexer(
            qdrant_url="http://localhost:6333",
            collection_name="test",
            embedding_model="custom-model",
        )

        mock_model_info.assert_called_with("custom-model")


class TestEnsureCollection:
    """Tests for ensure_collection method."""

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_collection_exists_not_created(self, mock_embedding, mock_client, mock_model_info):
        """Mock collection_exists=True, verify create not called."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        mock_client_instance = MagicMock()
        mock_client_instance.collection_exists.return_value = True
        mock_client.return_value = mock_client_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")
        result = indexer.ensure_collection()

        assert result is False
        mock_client_instance.create_collection.assert_not_called()

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_collection_not_exists_created(self, mock_embedding, mock_client, mock_model_info):
        """Mock collection_exists=False, verify create_collection called."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
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

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_index_single_file_creates_points(
        self, mock_embedding, mock_client, mock_model_info, sample_markdown_file: Path
    ):
        """Verify points generated for file."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed.return_value = [[0.1] * 384]
        mock_embedding.return_value = mock_embedding_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")
        chunker = RecursiveChunker(chunk_size=1000)
        chunk_count, point_ids, image_count, image_ids = indexer.index_file(sample_markdown_file, chunker)

        assert chunk_count >= 1
        assert isinstance(point_ids, list)
        assert len(point_ids) == chunk_count
        assert image_count == 0
        assert image_ids == []
        mock_client_instance.upsert.assert_called()

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_returns_chunk_count(
        self, mock_embedding, mock_client, mock_model_info, sample_markdown_file: Path
    ):
        """Verify index_file returns correct number of chunks."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_embedding_instance = MagicMock()
        # Return embeddings matching input count
        mock_embedding_instance.embed.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
        mock_embedding.return_value = mock_embedding_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")
        chunker = RecursiveChunker(chunk_size=50)  # Small chunks
        chunk_count, point_ids, image_count, image_ids = indexer.index_file(sample_markdown_file, chunker)

        assert isinstance(chunk_count, int)
        assert isinstance(point_ids, list)
        assert chunk_count >= 0
        assert len(point_ids) == chunk_count
        assert image_count == 0
        assert image_ids == []


class TestIndexDirectory:
    """Tests for index_directory method."""

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_index_multiple_files(
        self, mock_embedding, mock_client, mock_model_info, tmp_path: Path
    ):
        """Create multiple test files, verify all processed."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
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

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_pattern_matching(
        self, mock_embedding, mock_client, mock_model_info, tmp_path: Path
    ):
        """Test glob patterns filter files correctly."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
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

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_summary_statistics(
        self, mock_embedding, mock_client, mock_model_info, tmp_path: Path
    ):
        """Verify returned dict has expected keys."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
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

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_default_chunker_used(
        self, mock_embedding, mock_client, mock_model_info, tmp_path: Path
    ):
        """Verify RecursiveChunker created if None passed."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
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

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_generate_point_id_consistency(self, mock_embedding, mock_client, mock_model_info, tmp_path: Path):
        """Same input produces same ID."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.md"

        id1 = indexer._generate_point_id(file_path, 0)
        id2 = indexer._generate_point_id(file_path, 0)

        assert id1 == id2

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_generate_point_id_different_for_different_chunks(
        self, mock_embedding, mock_client, mock_model_info, tmp_path: Path
    ):
        """Different chunk indices produce different IDs."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.md"

        id1 = indexer._generate_point_id(file_path, 0)
        id2 = indexer._generate_point_id(file_path, 1)

        assert id1 != id2

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_generate_point_id_positive(self, mock_embedding, mock_client, mock_model_info, tmp_path: Path):
        """All IDs are positive int64."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.md"

        for i in range(100):
            point_id = indexer._generate_point_id(file_path, i)
            assert point_id > 0
            assert point_id < 2**63

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_build_payload_required_fields(self, mock_embedding, mock_client, mock_model_info, tmp_path: Path):
        """Verify all required fields present."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
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

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_build_payload_merges_metadata(self, mock_embedding, mock_client, mock_model_info, tmp_path: Path):
        """Verify custom metadata included."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
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


class TestCodeFileIndexing:
    """Tests for code file indexing with symbols."""

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_index_python_file_with_symbols(
        self, mock_embedding, mock_client, mock_model_info, tmp_path: Path
    ):
        """Verify Python code file with symbols is indexed correctly."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        # Create a Python file
        py_file = tmp_path / "test.py"
        py_file.write_text('def hello():\n    """Say hello."""\n    return "hello"')

        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed.return_value = [[0.1] * 384]
        mock_embedding.return_value = mock_embedding_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")
        chunker = RecursiveChunker()

        chunk_count, point_ids, image_count, image_ids = indexer.index_file(py_file, chunker)

        # Should have indexed at least one symbol
        assert chunk_count >= 1
        assert isinstance(point_ids, list)
        assert len(point_ids) == chunk_count
        assert image_count == 0  # Python files don't have images
        assert image_ids == []
        mock_client_instance.upsert.assert_called()

        # Check that points were created with code metadata
        call_args = mock_client_instance.upsert.call_args_list[0]
        points = call_args.kwargs["points"]
        assert len(points) > 0

        # Verify payload contains code-specific fields
        payload = points[0].payload
        assert "language" in payload
        assert "symbol_type" in payload
        assert "symbol_name" in payload
        assert payload["language"] == "python"

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_index_regular_file_uses_regular_indexer(
        self, mock_embedding, mock_client, mock_model_info, tmp_path: Path
    ):
        """Verify non-code file uses regular indexing path."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\nRegular markdown content.")

        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_embedding_instance = MagicMock()
        mock_embedding_instance.embed.return_value = [[0.1] * 384]
        mock_embedding.return_value = mock_embedding_instance

        indexer = QdrantIndexer("http://localhost:6333", "test")
        chunker = RecursiveChunker()
        chunk_count, point_ids, image_count, image_ids = indexer.index_file(md_file, chunker)

        assert chunk_count >= 1
        assert isinstance(point_ids, list)
        assert len(point_ids) == chunk_count
        assert image_count == 0  # Markdown files don't have images
        assert image_ids == []
        mock_client_instance.upsert.assert_called()

        # Check payload doesn't have code-specific fields
        call_args = mock_client_instance.upsert.call_args_list[0]
        points = call_args.kwargs["points"]
        payload = points[0].payload

        # Regular files shouldn't have symbol metadata
        assert "symbol_type" not in payload or payload.get("is_code") is not True

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_build_code_payload_has_all_fields(
        self, mock_embedding, mock_client, mock_model_info, tmp_path: Path
    ):
        """Verify _build_code_payload includes all code-specific fields."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        from qdrant_indexer.models import CodeSymbol

        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.py"

        symbol = CodeSymbol(
            name="test_func",
            qualified_name="test_func",
            symbol_type="function",
            content="def test_func(): pass",
            docstring="Test function",
            signature="()",
            line_start=1,
            line_end=1,
            parent=None,
            visibility=None,
            language="python",
        )

        payload = indexer._build_code_payload(
            chunk="function: test_func\n()\nTest function",
            symbol=symbol,
            file_path=file_path,
            chunk_index=0,
            total_chunks=1,
            metadata={"filename": "test.py", "is_code": True},
        )

        # Verify all code-specific fields
        assert payload["language"] == "python"
        assert payload["symbol_type"] == "function"
        assert payload["symbol_name"] == "test_func"
        assert payload["symbol_qualified_name"] == "test_func"
        assert payload["signature"] == "()"
        assert payload["docstring"] == "Test function"
        assert payload["line_start"] == 1
        assert payload["line_end"] == 1
        assert payload["parent_class"] == ""
        assert payload["visibility"] == ""
        assert payload["filename"] == "test.py"

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_build_code_payload_excludes_symbols_from_metadata(
        self, mock_embedding, mock_client, mock_model_info, tmp_path: Path
    ):
        """Verify _build_code_payload doesn't include large symbols list."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        from qdrant_indexer.models import CodeSymbol

        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.py"

        symbol = CodeSymbol(
            name="test_func",
            qualified_name="test_func",
            symbol_type="function",
            content="def test_func(): pass",
            docstring=None,
            signature="()",
            line_start=1,
            line_end=1,
            parent=None,
            visibility=None,
            language="python",
        )

        # Include symbols in metadata (simulating loaded document)
        metadata = {
            "filename": "test.py",
            "is_code": True,
            "symbols": [symbol, symbol, symbol],  # Large list
        }

        payload = indexer._build_code_payload(
            chunk="function: test_func",
            symbol=symbol,
            file_path=file_path,
            chunk_index=0,
            total_chunks=1,
            metadata=metadata,
        )

        # Verify symbols list is not in payload
        assert "symbols" not in payload
        # But other metadata should be included
        assert payload["filename"] == "test.py"
        assert payload["is_code"] is True

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_fallback_chunk_symbols(
        self, mock_embedding, mock_client, mock_model_info
    ):
        """Verify _fallback_chunk_symbols creates proper chunks."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        from qdrant_indexer.models import CodeSymbol

        indexer = QdrantIndexer("http://localhost:6333", "test")
        chunker = RecursiveChunker()

        symbols = [
            CodeSymbol(
                name="func1",
                qualified_name="func1",
                symbol_type="function",
                content="def func1(): pass",
                docstring="First function",
                signature="()",
                line_start=1,
                line_end=1,
                parent=None,
                visibility=None,
                language="python",
            ),
            CodeSymbol(
                name="func2",
                qualified_name="func2",
                symbol_type="function",
                content="def func2(): pass",
                docstring="Second function",
                signature="()",
                line_start=3,
                line_end=3,
                parent=None,
                visibility=None,
                language="python",
            ),
        ]

        chunks_with_symbols = indexer._fallback_chunk_symbols(symbols, chunker)

        # Should have at least 2 chunks (one per symbol)
        assert len(chunks_with_symbols) >= 2

        # Each item should be a tuple of (chunk_text, symbol)
        for chunk_text, symbol in chunks_with_symbols:
            assert isinstance(chunk_text, str)
            assert isinstance(symbol, CodeSymbol)
            # Chunk should contain symbol info
            assert symbol.name in chunk_text or symbol.symbol_type in chunk_text


class TestDeletionMethods:
    """Tests for file deletion methods."""

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_delete_points_by_ids_empty_list(self, mock_embedding, mock_client, mock_model_info):
        """Test delete_points_by_ids with empty list."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        indexer = QdrantIndexer("http://localhost:6333", "test-collection")

        # Should not call delete with empty list
        indexer.delete_points_by_ids([])
        mock_client_instance.delete.assert_not_called()

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_delete_points_by_ids(self, mock_embedding, mock_client, mock_model_info):
        """Test delete_points_by_ids with valid IDs."""
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        indexer = QdrantIndexer("http://localhost:6333", "test-collection")

        point_ids = [1, 2, 3, 4, 5]
        indexer.delete_points_by_ids(point_ids)

        mock_client_instance.delete.assert_called_once_with(
            collection_name="test-collection",
            points_selector=point_ids,
        )

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_delete_file_chunks_no_points(self, mock_embedding, mock_client, mock_model_info):
        """Test delete_file_chunks when no points exist for file."""
        from qdrant_client.models import Record

        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock scroll to return no points
        mock_client_instance.scroll.return_value = ([], None)

        indexer = QdrantIndexer("http://localhost:6333", "test-collection")

        file_path = Path("/test/doc.md")
        deleted_count = indexer.delete_file_chunks(file_path)

        assert deleted_count == 0
        mock_client_instance.delete.assert_not_called()

    @patch("qdrant_indexer.indexer.get_model_info")
    @patch("qdrant_indexer.indexer.QdrantClient")
    @patch("qdrant_indexer.indexer.TextEmbedding")
    def test_delete_file_chunks_with_points(self, mock_embedding, mock_client, mock_model_info):
        """Test delete_file_chunks when points exist for file."""
        from qdrant_client.models import Record

        mock_model_info.return_value = {"dim": 384, "model": "test-model"}
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        # Mock scroll to return 3 points
        mock_points = [
            Record(id=1, payload={}, vector=None),
            Record(id=2, payload={}, vector=None),
            Record(id=3, payload={}, vector=None),
        ]
        mock_client_instance.scroll.return_value = (mock_points, None)

        indexer = QdrantIndexer("http://localhost:6333", "test-collection")

        file_path = Path("/test/doc.md")
        deleted_count = indexer.delete_file_chunks(file_path)

        assert deleted_count == 3
        mock_client_instance.delete.assert_called_once_with(
            collection_name="test-collection",
            points_selector=[1, 2, 3],
        )
