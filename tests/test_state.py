"""Tests for state management."""

import json
import tempfile
from pathlib import Path

import pytest

from qdrant_indexer.models import IndexedFileState
from qdrant_indexer.state import IndexState, compute_file_hash


class TestIndexState:
    """Tests for IndexState class."""

    def test_init(self, tmp_path):
        """Test IndexState initialization."""
        state_file = tmp_path / "state.json"
        state = IndexState(state_file)
        assert state.state_file == state_file
        assert state.files == {}

    def test_load_empty_state(self, tmp_path):
        """Test loading when state file doesn't exist."""
        state_file = tmp_path / "state.json"
        state = IndexState(state_file)
        state.load()
        assert state.files == {}

    def test_save_and_load(self, tmp_path):
        """Test saving and loading state."""
        state_file = tmp_path / "state.json"
        state = IndexState(state_file)

        # Add file state
        file_state = IndexedFileState(
            path="/test/file.txt",
            content_hash="abc123",
            indexed_at="2024-01-01T00:00:00Z",
            chunk_count=3,
            chunk_ids=[1, 2, 3],
        )
        state.set_file_state(Path("/test/file.txt"), file_state)

        # Save state
        state.save()
        assert state_file.exists()

        # Load state in new instance
        new_state = IndexState(state_file)
        new_state.load()
        assert len(new_state.files) == 1
        loaded_state = new_state.get_file_state(Path("/test/file.txt"))
        assert loaded_state is not None
        assert loaded_state.path == "/test/file.txt"
        assert loaded_state.content_hash == "abc123"
        assert loaded_state.chunk_count == 3
        assert loaded_state.chunk_ids == [1, 2, 3]

    def test_json_format(self, tmp_path):
        """Test that JSON format is correct."""
        state_file = tmp_path / "state.json"
        state = IndexState(state_file)

        file_state = IndexedFileState(
            path="/test/file.txt",
            content_hash="abc123",
            indexed_at="2024-01-01T00:00:00Z",
            chunk_count=2,
            chunk_ids=[1, 2],
        )
        state.set_file_state(Path("/test/file.txt"), file_state)
        state.save()

        # Verify JSON structure
        with open(state_file, "r") as f:
            data = json.load(f)
        assert "files" in data
        assert "/test/file.txt" in data["files"]
        assert data["files"]["/test/file.txt"]["content_hash"] == "abc123"

    def test_get_file_state(self, tmp_path):
        """Test getting file state."""
        state_file = tmp_path / "state.json"
        state = IndexState(state_file)

        # Non-existent file
        assert state.get_file_state(Path("/nonexistent.txt")) is None

        # Existing file
        file_state = IndexedFileState(
            path="/test/file.txt",
            content_hash="abc123",
            indexed_at="2024-01-01T00:00:00Z",
            chunk_count=1,
            chunk_ids=[1],
        )
        state.set_file_state(Path("/test/file.txt"), file_state)
        retrieved = state.get_file_state(Path("/test/file.txt"))
        assert retrieved == file_state

    def test_set_file_state(self, tmp_path):
        """Test setting file state."""
        state_file = tmp_path / "state.json"
        state = IndexState(state_file)

        file_state = IndexedFileState(
            path="/test/file.txt",
            content_hash="abc123",
            indexed_at="2024-01-01T00:00:00Z",
            chunk_count=1,
            chunk_ids=[1],
        )
        state.set_file_state(Path("/test/file.txt"), file_state)
        assert len(state.files) == 1
        assert "/test/file.txt" in state.files

        # Update existing state
        new_state = IndexedFileState(
            path="/test/file.txt",
            content_hash="def456",
            indexed_at="2024-01-02T00:00:00Z",
            chunk_count=2,
            chunk_ids=[1, 2],
        )
        state.set_file_state(Path("/test/file.txt"), new_state)
        assert len(state.files) == 1
        assert state.files["/test/file.txt"].content_hash == "def456"

    def test_remove_file(self, tmp_path):
        """Test removing file from state."""
        state_file = tmp_path / "state.json"
        state = IndexState(state_file)

        file_state = IndexedFileState(
            path="/test/file.txt",
            content_hash="abc123",
            indexed_at="2024-01-01T00:00:00Z",
            chunk_count=1,
            chunk_ids=[1],
        )
        state.set_file_state(Path("/test/file.txt"), file_state)
        assert len(state.files) == 1

        state.remove_file(Path("/test/file.txt"))
        assert len(state.files) == 0
        assert state.get_file_state(Path("/test/file.txt")) is None

        # Removing non-existent file should not raise error
        state.remove_file(Path("/nonexistent.txt"))

    def test_get_all_paths(self, tmp_path):
        """Test getting all tracked file paths."""
        state_file = tmp_path / "state.json"
        state = IndexState(state_file)

        # Empty state
        assert state.get_all_paths() == set()

        # Add multiple files
        for i in range(3):
            file_state = IndexedFileState(
                path=f"/test/file{i}.txt",
                content_hash=f"hash{i}",
                indexed_at="2024-01-01T00:00:00Z",
                chunk_count=1,
                chunk_ids=[i],
            )
            state.set_file_state(Path(f"/test/file{i}.txt"), file_state)

        paths = state.get_all_paths()
        assert len(paths) == 3
        assert "/test/file0.txt" in paths
        assert "/test/file1.txt" in paths
        assert "/test/file2.txt" in paths

    def test_parent_directory_creation(self, tmp_path):
        """Test that parent directories are created on save."""
        nested_path = tmp_path / "nested" / "dir" / "state.json"
        state = IndexState(nested_path)

        file_state = IndexedFileState(
            path="/test/file.txt",
            content_hash="abc123",
            indexed_at="2024-01-01T00:00:00Z",
            chunk_count=1,
            chunk_ids=[1],
        )
        state.set_file_state(Path("/test/file.txt"), file_state)
        state.save()

        assert nested_path.exists()
        assert nested_path.parent.exists()


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_compute_file_hash(self, tmp_path):
        """Test computing hash of file content."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, world!"
        test_file.write_text(test_content)

        file_hash = compute_file_hash(test_file)
        assert isinstance(file_hash, str)
        assert len(file_hash) == 64  # SHA-256 produces 64 hex characters

    def test_compute_file_hash_consistency(self, tmp_path):
        """Test that same content produces same hash."""
        test_file = tmp_path / "test.txt"
        test_content = "Hello, world!"
        test_file.write_text(test_content)

        hash1 = compute_file_hash(test_file)
        hash2 = compute_file_hash(test_file)
        assert hash1 == hash2

    def test_compute_file_hash_different_content(self, tmp_path):
        """Test that different content produces different hash."""
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_text("Content A")
        file2.write_text("Content B")

        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)
        assert hash1 != hash2

    def test_compute_file_hash_binary(self, tmp_path):
        """Test hashing binary files."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03\xff")

        file_hash = compute_file_hash(test_file)
        assert isinstance(file_hash, str)
        assert len(file_hash) == 64

    def test_compute_file_hash_large_file(self, tmp_path):
        """Test hashing large files (tests chunked reading)."""
        test_file = tmp_path / "large.txt"
        # Create a file larger than the chunk size (8192 bytes)
        content = "x" * 10000
        test_file.write_text(content)

        file_hash = compute_file_hash(test_file)
        assert isinstance(file_hash, str)
        assert len(file_hash) == 64
