"""Pytest fixtures shared across the entire test suite.

Only fixtures consumed by two or more test files live here.
Domain-specific fixtures are co-located with their test files.
"""

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared Qdrant connectivity constants and helpers
# ---------------------------------------------------------------------------

QDRANT_URL = "http://localhost:6333"


def qdrant_available() -> bool:
    """Return True if a Qdrant instance is reachable at QDRANT_URL."""
    try:
        from qdrant_client import QdrantClient

        QdrantClient(url=QDRANT_URL, timeout=2).get_collections()
        return True
    except Exception:
        return False


@pytest.fixture
def test_collection_name() -> str:
    """Generate a unique collection name for test isolation."""
    return f"test_integration_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def qdrant_client():
    """Return a QdrantClient connected to the local test instance."""
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL)


@pytest.fixture
def cleanup_collection(qdrant_client, test_collection_name: str):
    """Delete the test collection after each test, ignoring errors."""
    yield test_collection_name
    try:
        qdrant_client.delete_collection(test_collection_name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Mock fixture for QdrantIndexer unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_indexer_env():
    """Patch the three heavyweight dependencies of QdrantIndexer.

    Yields a SimpleNamespace with:
        - mock_model_info  : the patched get_model_info callable
        - mock_client_cls  : the patched QdrantClient class
        - mock_client      : the MagicMock instance returned by QdrantClient()
        - mock_embed_cls   : the patched TextEmbedding class
        - mock_embed       : the MagicMock instance returned by TextEmbedding()

    Default behaviours (tests may override before constructing QdrantIndexer):
        - mock_model_info  → {"dim": 384, "model": "test-model"}
        - mock_client.collection_exists → False
        - mock_embed.embed             → [[0.1] * 384]
    """
    with (
        patch("qdrant_indexer.indexer.get_model_info") as mock_model_info,
        patch("qdrant_indexer.indexer.QdrantClient") as mock_client_cls,
        patch("qdrant_indexer.indexer.TextEmbedding") as mock_embed_cls,
    ):
        mock_model_info.return_value = {"dim": 384, "model": "test-model"}

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client_cls.return_value = mock_client

        mock_embed = MagicMock()
        mock_embed.embed.return_value = [[0.1] * 384]
        mock_embed_cls.return_value = mock_embed

        yield SimpleNamespace(
            mock_model_info=mock_model_info,
            mock_client_cls=mock_client_cls,
            mock_client=mock_client,
            mock_embed_cls=mock_embed_cls,
            mock_embed=mock_embed,
        )


# ---------------------------------------------------------------------------
# Document fixtures shared by test_loaders.py and test_indexer.py
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_markdown_content() -> str:
    """Return markdown text with frontmatter."""
    return """---
title: Test Document
author: Test Author
tags:
  - test
  - sample
---
# Header

This is the content of the document.

## Subheader

More content here with **bold** and *italic* text.
"""


@pytest.fixture
def sample_markdown_file(tmp_path: Path, sample_markdown_content: str) -> Path:
    """Create a temporary markdown file."""
    file_path = tmp_path / "test.md"
    file_path.write_text(sample_markdown_content)
    return file_path


# ---------------------------------------------------------------------------
# HTML fixtures shared by test_loaders.py and test_chunkers.py
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_html_content() -> str:
    """Return sample HTML with title, meta tags, and scripts."""
    return """<!DOCTYPE html>
<html>
<head>
    <title>Test HTML Document</title>
    <meta name="description" content="A test HTML document for testing HTMLLoader">
    <meta name="keywords" content="test, html, loader">
    <meta name="author" content="Test Author">
    <script>console.log('should be removed');</script>
    <style>.test { color: red; }</style>
</head>
<body>
    <nav>Navigation menu</nav>
    <h1>Main Header</h1>
    <p>This is test content that should be extracted.</p>
    <script>alert('remove me');</script>
</body>
</html>"""


@pytest.fixture
def sample_html_file(tmp_path: Path, sample_html_content: str) -> Path:
    """Create a temporary HTML file."""
    file_path = tmp_path / "test.html"
    file_path.write_text(sample_html_content)
    return file_path


@pytest.fixture
def malformed_html_content() -> str:
    """Return malformed HTML for robustness testing."""
    return """<html>
<head><title>Malformed HTML</head>
<body>
<p>Unclosed paragraph
<div>Missing closing tags
<script>broken script
</body>"""


@pytest.fixture
def malformed_html_file(tmp_path: Path, malformed_html_content: str) -> Path:
    """Create a temporary malformed HTML file."""
    file_path = tmp_path / "malformed.html"
    file_path.write_text(malformed_html_content)
    return file_path
