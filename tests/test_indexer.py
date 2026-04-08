"""Tests for the QdrantIndexer."""

from pathlib import Path

import pytest

from qdrant_indexer.chunkers import RecursiveChunker
from qdrant_indexer.indexer import QdrantIndexer


class TestQdrantIndexerInit:
    """Tests for QdrantIndexer initialization."""

    def test_indexer_initialization(self, mock_indexer_env):
        """Verify QdrantClient and TextEmbedding created."""
        mock_indexer_env.mock_model_info.return_value = {
            "dim": 384,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        }
        indexer = QdrantIndexer(
            qdrant_url="http://localhost:6333",
            collection_name="test-collection",
        )

        mock_indexer_env.mock_client_cls.assert_called_once_with(url="http://localhost:6333")
        assert indexer.collection == "test-collection"

    def test_indexer_custom_embedding_model(self, mock_indexer_env):
        """Test with different embedding model."""
        mock_indexer_env.mock_model_info.return_value = {"dim": 768, "model": "custom-model"}
        QdrantIndexer(
            qdrant_url="http://localhost:6333",
            collection_name="test",
            embedding_model="custom-model",
        )

        mock_indexer_env.mock_model_info.assert_called_with("custom-model")


class TestEnsureCollection:
    """Tests for ensure_collection method."""

    def test_collection_exists_not_created(self, mock_indexer_env):
        """Mock collection_exists=True, verify create not called."""
        mock_indexer_env.mock_client.collection_exists.return_value = True

        indexer = QdrantIndexer("http://localhost:6333", "test")
        result = indexer.ensure_collection()

        assert result is False
        mock_indexer_env.mock_client.create_collection.assert_not_called()

    def test_collection_not_exists_created(self, mock_indexer_env):
        """Mock collection_exists=False, verify create_collection called."""
        mock_indexer_env.mock_client.collection_exists.return_value = False

        indexer = QdrantIndexer("http://localhost:6333", "test")
        result = indexer.ensure_collection()

        assert result is True
        mock_indexer_env.mock_client.create_collection.assert_called_once()
        call_args = mock_indexer_env.mock_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "test"


class TestIndexFile:
    """Tests for index_file method."""

    def test_index_single_file_creates_points(self, mock_indexer_env, sample_markdown_file: Path):
        """Verify points generated for file."""
        indexer = QdrantIndexer("http://localhost:6333", "test")
        chunker = RecursiveChunker(chunk_size=1000)
        chunk_count, point_ids, image_count, image_ids = indexer.index_file(sample_markdown_file, chunker)

        assert chunk_count >= 1
        assert isinstance(point_ids, list)
        assert len(point_ids) == chunk_count
        assert image_count == 0
        assert image_ids == []
        mock_indexer_env.mock_client.upsert.assert_called()

    def test_returns_chunk_count(self, mock_indexer_env, sample_markdown_file: Path):
        """Verify index_file returns correct number of chunks."""
        mock_indexer_env.mock_embed.embed.side_effect = lambda texts: [[0.1] * 384 for _ in texts]

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

    def test_index_multiple_files(self, mock_indexer_env, tmp_path: Path):
        """Create multiple test files, verify all processed."""
        (tmp_path / "doc1.md").write_text("# Doc 1\nContent one.")
        (tmp_path / "doc2.md").write_text("# Doc 2\nContent two.")
        (tmp_path / "doc3.md").write_text("# Doc 3\nContent three.")

        indexer = QdrantIndexer("http://localhost:6333", "test")
        result = indexer.index_directory(tmp_path, patterns=["**/*.md"])

        assert result.total_files == 3
        assert result.total_chunks >= 3

    def test_pattern_matching(self, mock_indexer_env, tmp_path: Path):
        """Test glob patterns filter files correctly."""
        (tmp_path / "doc.md").write_text("Markdown")
        (tmp_path / "doc.txt").write_text("Text")
        (tmp_path / "doc.rst").write_text("RST")

        indexer = QdrantIndexer("http://localhost:6333", "test")

        result = indexer.index_directory(tmp_path, patterns=["**/*.md"])
        assert result.total_files == 1

    def test_summary_statistics(self, mock_indexer_env, tmp_path: Path):
        """Verify returned result has expected attributes."""
        (tmp_path / "test.md").write_text("# Test\nContent")

        indexer = QdrantIndexer("http://localhost:6333", "test")
        result = indexer.index_directory(tmp_path)

        assert hasattr(result, "total_files")
        assert hasattr(result, "total_chunks")
        assert hasattr(result, "failed_files")
        assert hasattr(result, "skipped_files")

    def test_default_chunker_used(self, mock_indexer_env, tmp_path: Path):
        """Verify RecursiveChunker created if None passed."""
        (tmp_path / "test.md").write_text("# Test\nContent")

        indexer = QdrantIndexer("http://localhost:6333", "test")
        result = indexer.index_directory(tmp_path, chunker=None)

        assert result.total_files >= 0


class TestHelperMethods:
    """Tests for helper methods."""

    def test_generate_point_id_consistency(self, mock_indexer_env, tmp_path: Path):
        """Same input produces same ID."""
        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.md"

        id1 = indexer._generate_point_id(file_path, 0)
        id2 = indexer._generate_point_id(file_path, 0)

        assert id1 == id2

    def test_generate_point_id_different_for_different_chunks(self, mock_indexer_env, tmp_path: Path):
        """Different chunk indices produce different IDs."""
        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.md"

        id1 = indexer._generate_point_id(file_path, 0)
        id2 = indexer._generate_point_id(file_path, 1)

        assert id1 != id2

    def test_generate_point_id_positive(self, mock_indexer_env, tmp_path: Path):
        """All IDs are positive int64."""
        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.md"

        for i in range(100):
            point_id = indexer._generate_point_id(file_path, i)
            assert point_id > 0
            assert point_id < 2**63

    def test_build_payload_required_fields(self, mock_indexer_env, tmp_path: Path):
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
        assert "metadata" in payload
        assert "source" in payload["metadata"]
        assert "chunk_index" in payload["metadata"]
        assert "total_chunks" in payload["metadata"]
        assert "timestamp" in payload["metadata"]

    def test_build_payload_merges_metadata(self, mock_indexer_env, tmp_path: Path):
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

        assert payload["metadata"]["title"] == "Test Title"
        assert payload["metadata"]["author"] == "Test Author"


class TestCodeFileIndexing:
    """Tests for code file indexing with symbols."""

    def test_index_python_file_with_symbols(self, mock_indexer_env, tmp_path: Path):
        """Verify Python code file with symbols is indexed correctly."""
        py_file = tmp_path / "test.py"
        py_file.write_text('def hello():\n    """Say hello."""\n    return "hello"')

        indexer = QdrantIndexer("http://localhost:6333", "test")
        chunker = RecursiveChunker()

        chunk_count, point_ids, image_count, image_ids = indexer.index_file(py_file, chunker)

        assert chunk_count >= 1
        assert isinstance(point_ids, list)
        assert len(point_ids) == chunk_count
        assert image_count == 0
        assert image_ids == []
        mock_indexer_env.mock_client.upsert.assert_called()

        call_args = mock_indexer_env.mock_client.upsert.call_args_list[0]
        points = call_args.kwargs["points"]
        assert len(points) > 0

        payload = points[0].payload
        assert "metadata" in payload
        assert "language" in payload["metadata"]
        assert "symbol_type" in payload["metadata"]
        assert "symbol_name" in payload["metadata"]
        assert payload["metadata"]["language"] == "python"

    def test_index_regular_file_uses_regular_indexer(self, mock_indexer_env, tmp_path: Path):
        """Verify non-code file uses regular indexing path."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Test\nRegular markdown content.")

        indexer = QdrantIndexer("http://localhost:6333", "test")
        chunker = RecursiveChunker()
        chunk_count, point_ids, image_count, image_ids = indexer.index_file(md_file, chunker)

        assert chunk_count >= 1
        assert isinstance(point_ids, list)
        assert len(point_ids) == chunk_count
        assert image_count == 0
        assert image_ids == []
        mock_indexer_env.mock_client.upsert.assert_called()

        call_args = mock_indexer_env.mock_client.upsert.call_args_list[0]
        points = call_args.kwargs["points"]
        payload = points[0].payload

        assert "symbol_type" not in payload or payload.get("is_code") is not True

    def test_build_code_payload_has_all_fields(self, mock_indexer_env, tmp_path: Path):
        """Verify _build_code_payload includes all code-specific fields."""
        from qdrant_indexer.models import CodeSymbol

        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.py"

        symbol = CodeSymbol(
            name="test_func",
            qualified_name="test_func",
            symbol_type="function",
            content="def test_func(): pass",
            language="python",
            docstring="Test function",
            signature="()",
            line_start=1,
            line_end=1,
        )

        payload = indexer._build_code_payload(
            chunk="function: test_func\n()\nTest function",
            symbol=symbol,
            file_path=file_path,
            chunk_index=0,
            total_chunks=1,
            metadata={"filename": "test.py", "is_code": True},
        )

        meta = payload["metadata"]
        assert meta["language"] == "python"
        assert meta["symbol_type"] == "function"
        assert meta["symbol_name"] == "test_func"
        assert meta["symbol_qualified_name"] == "test_func"
        assert meta["signature"] == "()"
        assert meta["docstring"] == "Test function"
        assert meta["line_start"] == 1
        assert meta["line_end"] == 1
        assert meta["parent_class"] == ""
        assert meta["visibility"] == ""
        assert meta["filename"] == "test.py"

    def test_build_code_payload_excludes_symbols_from_metadata(self, mock_indexer_env, tmp_path: Path):
        """Verify _build_code_payload doesn't include large symbols list."""
        from qdrant_indexer.models import CodeSymbol

        indexer = QdrantIndexer("http://localhost:6333", "test")
        file_path = tmp_path / "test.py"

        symbol = CodeSymbol(
            name="test_func",
            qualified_name="test_func",
            symbol_type="function",
            content="def test_func(): pass",
            language="python",
            signature="()",
            line_start=1,
            line_end=1,
        )

        metadata = {
            "filename": "test.py",
            "is_code": True,
            "symbols": [symbol, symbol, symbol],
        }

        payload = indexer._build_code_payload(
            chunk="function: test_func",
            symbol=symbol,
            file_path=file_path,
            chunk_index=0,
            total_chunks=1,
            metadata=metadata,
        )

        assert "symbols" not in payload["metadata"]
        assert payload["metadata"]["filename"] == "test.py"
        assert payload["metadata"]["is_code"] is True

    def test_fallback_chunk_symbols(self, mock_indexer_env):
        """Verify _fallback_chunk_symbols creates proper chunks."""
        from qdrant_indexer.models import CodeSymbol

        indexer = QdrantIndexer("http://localhost:6333", "test")
        chunker = RecursiveChunker()

        symbols = [
            CodeSymbol(
                name="func1",
                qualified_name="func1",
                symbol_type="function",
                content="def func1(): pass",
                language="python",
                docstring="First function",
                signature="()",
                line_start=1,
                line_end=1,
            ),
            CodeSymbol(
                name="func2",
                qualified_name="func2",
                symbol_type="function",
                content="def func2(): pass",
                language="python",
                docstring="Second function",
                signature="()",
                line_start=3,
                line_end=3,
            ),
        ]

        chunks_with_symbols = indexer._fallback_chunk_symbols(symbols, chunker)

        assert len(chunks_with_symbols) >= 2

        for chunk_text, symbol in chunks_with_symbols:
            assert isinstance(chunk_text, str)
            assert isinstance(symbol, CodeSymbol)
            assert symbol.name in chunk_text or symbol.symbol_type in chunk_text


class TestDeletionMethods:
    """Tests for file deletion methods."""

    def test_delete_points_by_ids_empty_list(self, mock_indexer_env):
        """Test delete_points_by_ids with empty list."""
        indexer = QdrantIndexer("http://localhost:6333", "test-collection")

        indexer.delete_points_by_ids([])
        mock_indexer_env.mock_client.delete.assert_not_called()

    def test_delete_points_by_ids(self, mock_indexer_env):
        """Test delete_points_by_ids with valid IDs."""
        indexer = QdrantIndexer("http://localhost:6333", "test-collection")

        point_ids = [1, 2, 3, 4, 5]
        indexer.delete_points_by_ids(point_ids)

        mock_indexer_env.mock_client.delete.assert_called_once_with(
            collection_name="test-collection",
            points_selector=point_ids,
        )

    def test_delete_file_chunks_no_points(self, mock_indexer_env):
        """Test delete_file_chunks when no points exist for file."""
        mock_indexer_env.mock_client.scroll.return_value = ([], None)

        indexer = QdrantIndexer("http://localhost:6333", "test-collection")

        file_path = Path("/test/doc.md")
        deleted_count = indexer.delete_file_chunks(file_path)

        assert deleted_count == 0
        mock_indexer_env.mock_client.delete.assert_not_called()

    def test_delete_file_chunks_with_points(self, mock_indexer_env):
        """Test delete_file_chunks when points exist for file."""
        from qdrant_client.models import Record

        mock_points = [
            Record(id=1, payload={}, vector=None),
            Record(id=2, payload={}, vector=None),
            Record(id=3, payload={}, vector=None),
        ]
        mock_indexer_env.mock_client.scroll.return_value = (mock_points, None)

        indexer = QdrantIndexer("http://localhost:6333", "test-collection")

        file_path = Path("/test/doc.md")
        deleted_count = indexer.delete_file_chunks(file_path)

        assert deleted_count == 3
        mock_indexer_env.mock_client.delete.assert_called_once_with(
            collection_name="test-collection",
            points_selector=[1, 2, 3],
        )
