"""Tests for incremental sync functionality."""

import tempfile
from pathlib import Path

import pytest

from qdrant_indexer.chunkers import RecursiveChunker
from qdrant_indexer.indexer import QdrantIndexer

# Use localhost for integration tests
QDRANT_URL = "http://localhost:6333"


@pytest.fixture
def test_collection_name():
    """Generate unique collection name for each test."""
    import uuid
    return f"test-sync-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup_collection(test_collection_name):
    """Cleanup collection after test."""
    yield
    from qdrant_client import QdrantClient
    client = QdrantClient(url=QDRANT_URL)
    try:
        client.delete_collection(test_collection_name)
    except Exception:
        pass


class TestSyncDirectoryNewFiles:
    """Tests for sync_directory with new files."""

    def test_sync_new_file(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test syncing a new file."""
        # Create test file
        doc = tmp_path / "doc.md"
        doc.write_text("# Test Document\n\nThis is a test.")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        # First sync
        result = indexer.sync_directory(tmp_path)

        assert result.added == 1
        assert result.updated == 0
        assert result.deleted == 0
        assert result.unchanged == 0
        assert len(result.failed) == 0

        # Verify state file created
        state_file = tmp_path / ".qdrant-index-state.json"
        assert state_file.exists()

    def test_sync_multiple_new_files(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test syncing multiple new files."""
        # Create test files
        (tmp_path / "doc1.md").write_text("# Doc 1")
        (tmp_path / "doc2.md").write_text("# Doc 2")
        (tmp_path / "doc3.txt").write_text("Doc 3 content")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        result = indexer.sync_directory(tmp_path)

        assert result.added == 3
        assert result.updated == 0
        assert result.deleted == 0
        assert result.unchanged == 0

    def test_sync_no_changes(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test syncing when no files have changed."""
        doc = tmp_path / "doc.md"
        doc.write_text("# Test Document")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        # First sync
        result1 = indexer.sync_directory(tmp_path)
        assert result1.added == 1

        # Second sync (no changes)
        result2 = indexer.sync_directory(tmp_path)
        assert result2.added == 0
        assert result2.updated == 0
        assert result2.deleted == 0
        assert result2.unchanged == 1


class TestSyncDirectoryModifiedFiles:
    """Tests for sync_directory with modified files."""

    def test_sync_modified_file(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test syncing a modified file."""
        doc = tmp_path / "doc.md"
        doc.write_text("# Original Content")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        # First sync
        result1 = indexer.sync_directory(tmp_path)
        assert result1.added == 1

        # Modify file
        doc.write_text("# Modified Content\n\nThis has changed significantly.")

        # Second sync
        result2 = indexer.sync_directory(tmp_path)
        assert result2.added == 0
        assert result2.updated == 1
        assert result2.deleted == 0
        assert result2.unchanged == 0

    def test_sync_partially_modified(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test syncing when only some files are modified."""
        doc1 = tmp_path / "doc1.md"
        doc2 = tmp_path / "doc2.md"
        doc3 = tmp_path / "doc3.md"

        doc1.write_text("# Doc 1")
        doc2.write_text("# Doc 2")
        doc3.write_text("# Doc 3")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        # First sync
        result1 = indexer.sync_directory(tmp_path)
        assert result1.added == 3

        # Modify only doc2
        doc2.write_text("# Doc 2 - Modified\n\nNew content here.")

        # Second sync
        result2 = indexer.sync_directory(tmp_path)
        assert result2.added == 0
        assert result2.updated == 1
        assert result2.deleted == 0
        assert result2.unchanged == 2

    def test_sync_force_reindex(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test force flag re-indexes all files."""
        doc1 = tmp_path / "doc1.md"
        doc2 = tmp_path / "doc2.md"

        doc1.write_text("# Doc 1")
        doc2.write_text("# Doc 2")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        # First sync
        result1 = indexer.sync_directory(tmp_path)
        assert result1.added == 2

        # Force sync (no actual changes)
        result2 = indexer.sync_directory(tmp_path, force=True)
        assert result2.added == 0
        assert result2.updated == 2
        assert result2.deleted == 0
        assert result2.unchanged == 0


class TestSyncDirectoryDeletedFiles:
    """Tests for sync_directory with deleted files."""

    def test_sync_deleted_file(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test syncing when a file is deleted."""
        doc = tmp_path / "doc.md"
        doc.write_text("# Test Document")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        # First sync
        result1 = indexer.sync_directory(tmp_path)
        assert result1.added == 1

        # Delete file
        doc.unlink()

        # Second sync
        result2 = indexer.sync_directory(tmp_path)
        assert result2.added == 0
        assert result2.updated == 0
        assert result2.deleted == 1
        assert result2.unchanged == 0

    def test_sync_multiple_deleted_files(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test syncing when multiple files are deleted."""
        doc1 = tmp_path / "doc1.md"
        doc2 = tmp_path / "doc2.md"
        doc3 = tmp_path / "doc3.md"

        doc1.write_text("# Doc 1")
        doc2.write_text("# Doc 2")
        doc3.write_text("# Doc 3")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        # First sync
        result1 = indexer.sync_directory(tmp_path)
        assert result1.added == 3

        # Delete two files
        doc1.unlink()
        doc3.unlink()

        # Second sync
        result2 = indexer.sync_directory(tmp_path)
        assert result2.added == 0
        assert result2.updated == 0
        assert result2.deleted == 2
        assert result2.unchanged == 1

    def test_sync_mixed_operations(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test syncing with add, update, and delete operations."""
        doc1 = tmp_path / "doc1.md"
        doc2 = tmp_path / "doc2.md"

        doc1.write_text("# Doc 1")
        doc2.write_text("# Doc 2")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        # First sync
        result1 = indexer.sync_directory(tmp_path)
        assert result1.added == 2

        # Mixed operations:
        # - Delete doc1
        doc1.unlink()
        # - Modify doc2
        doc2.write_text("# Doc 2 - Modified")
        # - Add doc3
        doc3 = tmp_path / "doc3.md"
        doc3.write_text("# Doc 3")

        # Second sync
        result2 = indexer.sync_directory(tmp_path)
        assert result2.added == 1      # doc3
        assert result2.updated == 1    # doc2
        assert result2.deleted == 1    # doc1
        assert result2.unchanged == 0


class TestSyncDirectoryStateFile:
    """Tests for state file handling."""

    def test_custom_state_file_location(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test using custom state file location."""
        doc = tmp_path / "doc.md"
        doc.write_text("# Test Document")

        custom_state = tmp_path / "custom_state.json"

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        result = indexer.sync_directory(tmp_path, state_file=custom_state)
        assert result.added == 1

        # Verify custom state file created
        assert custom_state.exists()
        # Default state file should not exist
        assert not (tmp_path / ".qdrant-index-state.json").exists()

    def test_state_file_tracks_point_ids(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test that state file correctly tracks point IDs."""
        doc = tmp_path / "doc.md"
        doc.write_text("# Test Document\n\nSome content here.")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        result = indexer.sync_directory(tmp_path)
        assert result.added == 1

        # Load state and verify point IDs are stored
        from qdrant_indexer.state import IndexState
        state_file = tmp_path / ".qdrant-index-state.json"
        state = IndexState(state_file)
        state.load()

        file_state = state.get_file_state(doc)
        assert file_state is not None
        assert len(file_state.chunk_ids) > 0
        assert file_state.chunk_count == len(file_state.chunk_ids)

    def test_state_file_updated_on_modify(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test that state file is updated when file is modified."""
        doc = tmp_path / "doc.md"
        doc.write_text("# Original")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        # First sync
        indexer.sync_directory(tmp_path)

        from qdrant_indexer.state import IndexState
        state_file = tmp_path / ".qdrant-index-state.json"
        state1 = IndexState(state_file)
        state1.load()
        original_hash = state1.get_file_state(doc).content_hash

        # Modify file
        doc.write_text("# Modified Content")

        # Second sync
        indexer.sync_directory(tmp_path)

        state2 = IndexState(state_file)
        state2.load()
        new_hash = state2.get_file_state(doc).content_hash

        # Hash should have changed
        assert new_hash != original_hash


class TestSyncDirectoryPatterns:
    """Tests for file pattern matching in sync."""

    def test_sync_with_custom_patterns(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test syncing with custom file patterns."""
        (tmp_path / "doc.md").write_text("# Markdown")
        (tmp_path / "doc.txt").write_text("Text")
        (tmp_path / "doc.rst").write_text("ReStructured")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        # Only sync .md files
        result = indexer.sync_directory(tmp_path, patterns=["**/*.md"])
        assert result.added == 1

    def test_sync_with_exclude_patterns(self, tmp_path: Path, test_collection_name: str, cleanup_collection):
        """Test syncing with exclusion patterns."""
        (tmp_path / "doc.md").write_text("# Document")
        (tmp_path / "temp.md").write_text("# Temp")

        indexer = QdrantIndexer(QDRANT_URL, test_collection_name)
        indexer.ensure_collection()

        # Exclude files starting with 'temp'
        result = indexer.sync_directory(tmp_path, exclude_patterns=["temp*"])
        # Only doc.md should be added (temp.md excluded)
        assert result.added == 1
