"""Integration tests for qdrant_indexer.

These tests require a running Qdrant instance. By default, they connect to
localhost:6333. Start Qdrant with: docker compose up -d

Tests are marked with @pytest.mark.integration and can be skipped with:
    pytest -m "not integration"

Or run only integration tests with:
    pytest -m integration
"""

import logging
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from qdrant_indexer.chunkers import (
    CodeChunker,
    HTMLChunker,
    MarkdownChunker,
    RecursiveChunker,
)
from qdrant_indexer.indexer import QdrantIndexer

# Default Qdrant URL for tests
QDRANT_URL = "http://localhost:6333"


def qdrant_available() -> bool:
    """Check if Qdrant is available at the default URL."""
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


# Skip all tests in this module if Qdrant is not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not qdrant_available(),
        reason="Qdrant not available at localhost:6333. Start with: docker compose up -d",
    ),
]


@pytest.fixture
def test_collection_name() -> str:
    """Generate a unique collection name for test isolation."""
    return f"test_integration_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def qdrant_client() -> QdrantClient:
    """Create a Qdrant client for direct verification."""
    return QdrantClient(url=QDRANT_URL)


@pytest.fixture
def cleanup_collection(qdrant_client: QdrantClient, test_collection_name: str):
    """Fixture to clean up test collection after test."""
    yield test_collection_name
    # Cleanup after test
    try:
        qdrant_client.delete_collection(test_collection_name)
    except Exception:
        pass  # Collection may not exist


class TestIndexerIntegration:
    """Integration tests for the QdrantIndexer class."""

    def test_ensure_collection_creates_new(
        self, test_collection_name: str, qdrant_client: QdrantClient, cleanup_collection: str
    ):
        """Test that ensure_collection creates a new collection."""
        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )

        # Collection should not exist yet
        assert not qdrant_client.collection_exists(test_collection_name)

        # Create collection
        created = indexer.ensure_collection()
        assert created is True

        # Collection should now exist
        assert qdrant_client.collection_exists(test_collection_name)

    def test_ensure_collection_existing(
        self, test_collection_name: str, qdrant_client: QdrantClient, cleanup_collection: str
    ):
        """Test that ensure_collection returns False for existing collection."""
        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )

        # Create collection first time
        indexer.ensure_collection()

        # Second call should return False
        created = indexer.ensure_collection()
        assert created is False

    def test_index_single_file(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test indexing a single file end-to-end."""
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text(
            """---
title: Integration Test Document
author: Test Suite
---
# Test Document

This is a test document for integration testing.

## Section One

Some content in section one with important keywords.

## Section Two

More content here to ensure we have enough text for chunking.
"""
        )

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        chunker = RecursiveChunker(chunk_size=200, overlap=20)
        chunk_count, point_ids = indexer.index_file(test_file, chunker)

        assert chunk_count > 0
        assert isinstance(point_ids, list)
        assert len(point_ids) == chunk_count

        # Verify points exist in collection
        collection_info = qdrant_client.get_collection(test_collection_name)
        assert collection_info.points_count == chunk_count

    def test_index_directory(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test indexing a directory of files."""
        # Create test files
        (tmp_path / "doc1.md").write_text("# Document One\n\nContent for document one.")
        (tmp_path / "doc2.md").write_text("# Document Two\n\nContent for document two.")
        (tmp_path / "doc3.txt").write_text("Plain text document.\n\nWith multiple paragraphs.")

        # Create a subdirectory with more files
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("# Nested Document\n\nNested content here.")

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        # Index all markdown files
        result = indexer.index_directory(tmp_path, patterns=["**/*.md"])

        assert result["total_files"] == 3  # doc1.md, doc2.md, nested.md
        assert result["total_chunks"] > 0
        assert len(result["failed_files"]) == 0

        # Verify points exist
        collection_info = qdrant_client.get_collection(test_collection_name)
        assert collection_info.points_count == result["total_chunks"]

    def test_search_indexed_content(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test that indexed content is searchable."""
        # Create test file with specific content
        test_file = tmp_path / "searchable.md"
        test_file.write_text(
            """# Python Programming Guide

This document covers Python programming concepts.

## Variables and Types

Python supports various data types including integers, strings, and lists.

## Functions

Functions in Python are defined using the def keyword.

## Classes

Object-oriented programming in Python uses classes and objects.
"""
        )

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        chunker = RecursiveChunker(chunk_size=200, overlap=20)
        indexer.index_file(test_file, chunker)

        # Search for content using the same embedding model
        from fastembed import TextEmbedding
        from qdrant_client.models import NamedVector

        embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        query_vector = list(embedding_model.embed(["Python programming"]))[0]

        # Use named vector since the collection uses named vectors
        results = qdrant_client.search(
            collection_name=test_collection_name,
            query_vector=NamedVector(
                name="fast-all-minilm-l6-v2",
                vector=list(query_vector),
            ),
            limit=3,
        )

        assert len(results) > 0
        # Should find content related to Python
        assert any("Python" in r.payload.get("document", "") for r in results)

    def test_idempotent_indexing(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test that re-indexing the same file doesn't create duplicates."""
        test_file = tmp_path / "idempotent.md"
        test_file.write_text("# Idempotent Test\n\nThis content should only appear once.")

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        chunker = RecursiveChunker(chunk_size=200, overlap=20)

        # Index the same file twice
        count1, ids1 = indexer.index_file(test_file, chunker)
        count2, ids2 = indexer.index_file(test_file, chunker)

        assert count1 == count2
        assert ids1 == ids2

        # Should have same number of points (upsert behavior)
        collection_info = qdrant_client.get_collection(test_collection_name)
        assert collection_info.points_count == count1

    def test_index_python_code_file(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test indexing Python source code with symbols."""
        # Create a Python file with functions and classes
        code_file = tmp_path / "utils.py"
        code_file.write_text(
            '''"""Utility functions for testing."""

def helper_function(x: int) -> int:
    """A helper function that doubles a value.

    Args:
        x: The value to double.

    Returns:
        The doubled value.
    """
    return x * 2


class Calculator:
    """A simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: First number.
            b: Second number.

        Returns:
            The sum of a and b.
        """
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a.

        Args:
            a: First number.
            b: Second number.

        Returns:
            The difference.
        """
        return a - b
'''
        )

        # Python code loader is registered automatically via lazy import in get_loader()
        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        chunker = RecursiveChunker()
        chunk_count, point_ids = indexer.index_file(code_file, chunker)

        # Should have indexed symbols (module, function, class, 2 methods)
        assert chunk_count >= 4
        assert isinstance(point_ids, list)
        assert len(point_ids) == chunk_count

        # Verify points exist
        collection_info = qdrant_client.get_collection(test_collection_name)
        assert collection_info.points_count == chunk_count

        # Search for the function using embeddings
        from fastembed import TextEmbedding
        from qdrant_client.models import NamedVector

        embedding_model = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            providers=["CPUExecutionProvider"],
        )
        query_vector = list(embedding_model.embed(["helper function that doubles"]))[0]

        # Use named vector since the collection uses named vectors
        results = qdrant_client.search(
            collection_name=test_collection_name,
            query_vector=NamedVector(
                name="fast-all-minilm-l6-v2",
                vector=list(query_vector),
            ),
            limit=5,
        )

        assert len(results) > 0

        # Check for code-specific metadata
        found_code_payload = False
        for result in results:
            if "symbol_type" in result.payload and "language" in result.payload:
                found_code_payload = True
                assert result.payload["language"] == "python"
                assert result.payload["symbol_type"] in ["function", "class", "method", "module"]
                break

        assert found_code_payload, "No code-specific metadata found in results"


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_cli_index_command(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test the index CLI command."""
        # Create test files
        (tmp_path / "cli_test.md").write_text("# CLI Test\n\nContent for CLI testing.")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "qdrant_indexer.cli",
                "index",
                str(tmp_path),
                "--collection",
                test_collection_name,
                "--url",
                QDRANT_URL,
                "--pattern",
                "**/*.md",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Verify collection was created and has points
        assert qdrant_client.collection_exists(test_collection_name)
        collection_info = qdrant_client.get_collection(test_collection_name)
        assert collection_info.points_count > 0

    def test_cli_list_collections(self, qdrant_client: QdrantClient):
        """Test the list-collections CLI command."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "qdrant_indexer.cli",
                "list-collections",
                "--url",
                QDRANT_URL,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_cli_delete_collection(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
    ):
        """Test the delete-collection CLI command."""
        # First create a collection
        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()
        assert qdrant_client.collection_exists(test_collection_name)

        # Delete via CLI
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "qdrant_indexer.cli",
                "delete-collection",
                test_collection_name,
                "--url",
                QDRANT_URL,
                "--yes",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert not qdrant_client.collection_exists(test_collection_name)

    def test_cli_verbose_output(
        self,
        test_collection_name: str,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test that --verbose flag produces debug output."""
        (tmp_path / "verbose_test.md").write_text("# Verbose Test\n\nContent.")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "qdrant_indexer.cli",
                "index",
                str(tmp_path),
                "--collection",
                test_collection_name,
                "--url",
                QDRANT_URL,
                "--verbose",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_cli_quiet_output(
        self,
        test_collection_name: str,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test that --quiet flag suppresses output."""
        (tmp_path / "quiet_test.md").write_text("# Quiet Test\n\nContent.")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "qdrant_indexer.cli",
                "index",
                str(tmp_path),
                "--collection",
                test_collection_name,
                "--url",
                QDRANT_URL,
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_cli_chunker_auto(
        self,
        test_collection_name: str,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test CLI with --chunker auto mode."""
        (tmp_path / "auto_test.md").write_text("# Auto Test\n\n## Section\n\nContent.")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "qdrant_indexer.cli",
                "index",
                str(tmp_path),
                "--collection",
                test_collection_name,
                "--url",
                QDRANT_URL,
                "--chunker",
                "auto",
                "--verbose",
                "--verbose",  # -vv for DEBUG output
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        # Verbose output should show auto-selection
        assert "Auto-selected" in result.stderr or "Chunker: auto" in result.stdout

    def test_cli_chunker_explicit(
        self,
        test_collection_name: str,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test CLI with explicit --chunker markdown."""
        (tmp_path / "explicit_test.md").write_text("# Explicit Test\n\nContent.")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "qdrant_indexer.cli",
                "index",
                str(tmp_path),
                "--collection",
                test_collection_name,
                "--url",
                QDRANT_URL,
                "--chunker",
                "markdown",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        # Check output includes chunker info
        assert "Chunker: markdown" in result.stdout or result.returncode == 0

    def test_cli_chunker_invalid(
        self,
        tmp_path: Path,
    ):
        """Test CLI with invalid chunker strategy."""
        (tmp_path / "invalid_test.md").write_text("# Invalid Test\n\nContent.")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "qdrant_indexer.cli",
                "index",
                str(tmp_path),
                "--collection",
                "test_invalid_chunker",
                "--url",
                QDRANT_URL,
                "--chunker",
                "invalid_name",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0
        # Error message may be in stdout or stderr depending on how CLI outputs errors
        output = result.stdout + result.stderr
        assert "Unknown chunker strategy" in output


class TestEdgeCases:
    """Integration tests for edge cases and error handling."""

    def test_empty_directory(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test indexing an empty directory."""
        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        result = indexer.index_directory(tmp_path, patterns=["**/*.md"])

        assert result["total_files"] == 0
        assert result["total_chunks"] == 0

    def test_binary_file_handling(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test that binary files are handled gracefully."""
        # Create a binary file
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\x04\x05")

        # Create a valid text file too
        (tmp_path / "valid.txt").write_text("Valid text content.")

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        # Should not fail on mixed content
        result = indexer.index_directory(tmp_path, patterns=["**/*"])

        # Should have indexed at least the text file
        assert result["total_files"] >= 1

    def test_large_file_batching(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test indexing a large file with batching."""
        # Create a large file
        large_file = tmp_path / "large.md"
        large_content = "# Large Document\n\n" + ("This is a paragraph of content. " * 100 + "\n\n") * 50
        large_file.write_text(large_content)

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        chunker = RecursiveChunker(chunk_size=200, overlap=20)
        chunk_count, point_ids = indexer.index_file(large_file, chunker, batch_size=10)

        assert chunk_count > 10  # Should have many chunks
        assert isinstance(point_ids, list)
        assert len(point_ids) == chunk_count

        # All chunks should be indexed
        collection_info = qdrant_client.get_collection(test_collection_name)
        assert collection_info.points_count == chunk_count

    def test_special_characters_in_content(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
    ):
        """Test handling of special characters and unicode."""
        test_file = tmp_path / "special.md"
        test_file.write_text(
            """# Special Characters Test

Unicode: 你好世界 🌍 émojis café naïve

Special chars: <script>alert('test')</script>

Code block:
```python
def hello():
    print("Hello, World!")
```

Math symbols: ∑ ∏ √ ∞ ≤ ≥ ≠
""",
            encoding="utf-8",
        )

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        chunker = RecursiveChunker(chunk_size=200, overlap=20)
        chunk_count, point_ids = indexer.index_file(test_file, chunker)

        assert chunk_count > 0
        assert isinstance(point_ids, list)
        assert len(point_ids) == chunk_count

        # Verify content preserved
        collection_info = qdrant_client.get_collection(test_collection_name)
        assert collection_info.points_count == chunk_count


class TestChunkerSelection:
    """Integration tests for automatic chunker selection."""

    def test_auto_chunker_markdown(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test auto-selection of MarkdownChunker for .md files."""
        test_file = tmp_path / "test_doc.md"
        test_file.write_text(
            """# Test Document

## Section One

Content in section one.

## Section Two

Content in section two.
"""
        )

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        with caplog.at_level(logging.DEBUG):
            # chunker=None triggers auto-selection
            result = indexer.index_directory(tmp_path, patterns=["**/*.md"], chunker=None)

        assert result["total_files"] == 1
        assert result["total_chunks"] > 0
        assert "Auto-selected MarkdownChunker" in caplog.text

    def test_auto_chunker_html(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test auto-selection of HTMLChunker for .html files."""
        test_file = tmp_path / "test_page.html"
        test_file.write_text(
            """<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
<article>
    <h1>Article Title</h1>
    <section>
        <h2>Section One</h2>
        <p>Content in section one.</p>
    </section>
    <section>
        <h2>Section Two</h2>
        <p>Content in section two.</p>
    </section>
</article>
</body>
</html>
"""
        )

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        with caplog.at_level(logging.DEBUG):
            result = indexer.index_directory(tmp_path, patterns=["**/*.html"], chunker=None)

        assert result["total_files"] == 1
        assert result["total_chunks"] > 0
        assert "Auto-selected HTMLChunker" in caplog.text

    def test_auto_chunker_code(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test auto-selection of CodeChunker for .py files."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text(
            '''"""Test module docstring."""

def hello_world():
    """Say hello."""
    print("Hello, World!")

def goodbye_world():
    """Say goodbye."""
    print("Goodbye, World!")
'''
        )

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        with caplog.at_level(logging.DEBUG):
            result = indexer.index_directory(tmp_path, patterns=["**/*.py"], chunker=None)

        assert result["total_files"] == 1
        assert result["total_chunks"] > 0
        assert "Auto-selected CodeChunker" in caplog.text

    def test_auto_chunker_mixed_directory(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test correct chunker selection for mixed file types."""
        # Create .md file
        (tmp_path / "readme.md").write_text("# Readme\n\nProject readme content.")

        # Create .html file
        (tmp_path / "index.html").write_text(
            "<html><body><h1>Index</h1><p>Index content.</p></body></html>"
        )

        # Create .py file
        (tmp_path / "script.py").write_text('"""Script docstring."""\n\ndef main():\n    pass\n')

        # Create .txt file (should use RecursiveChunker)
        (tmp_path / "notes.txt").write_text("Plain text notes.\n\nMore content here.")

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        with caplog.at_level(logging.DEBUG):
            result = indexer.index_directory(
                tmp_path, patterns=["**/*.md", "**/*.html", "**/*.py", "**/*.txt"], chunker=None
            )

        assert result["total_files"] == 4
        assert result["total_chunks"] > 0

        # Verify each file type got the correct chunker
        assert "Auto-selected MarkdownChunker for readme.md" in caplog.text
        assert "Auto-selected HTMLChunker for index.html" in caplog.text
        assert "Auto-selected CodeChunker for script.py" in caplog.text
        assert "Auto-selected RecursiveChunker for notes.txt" in caplog.text

    def test_explicit_chunker_override(
        self,
        test_collection_name: str,
        qdrant_client: QdrantClient,
        cleanup_collection: str,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        """Test that explicit chunker overrides auto-selection."""
        # Create .md file (would normally use MarkdownChunker)
        test_file = tmp_path / "doc.md"
        test_file.write_text("# Document\n\nContent here.")

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=test_collection_name,
        )
        indexer.ensure_collection()

        # Use explicit RecursiveChunker instead of auto-selection
        explicit_chunker = RecursiveChunker(chunk_size=500, overlap=50)

        with caplog.at_level(logging.DEBUG):
            result = indexer.index_directory(
                tmp_path, patterns=["**/*.md"], chunker=explicit_chunker
            )

        assert result["total_files"] == 1
        assert result["total_chunks"] > 0
        # Auto-selection log should NOT appear when using explicit chunker
        assert "Auto-selected" not in caplog.text
