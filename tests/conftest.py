"""Pytest fixtures for qdrant_indexer tests."""

import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from qdrant_indexer.models import Document

# ---------------------------------------------------------------------------
# Shared Qdrant connectivity constants and helpers (Issues #14)
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
# Mock fixture for QdrantIndexer unit tests (Issue #13)
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


@pytest.fixture
def sample_text_content() -> str:
    """Return plain text content."""
    return """This is a plain text file.

It has multiple paragraphs.

And some more content here for testing purposes.
"""


@pytest.fixture
def sample_text_file(tmp_path: Path, sample_text_content: str) -> Path:
    """Create a temporary text file."""
    file_path = tmp_path / "test.txt"
    file_path.write_text(sample_text_content)
    return file_path


@pytest.fixture
def sample_rst_content() -> str:
    """Return ReStructuredText content."""
    return """Test Document
=============

This is an RST document.

Section
-------

Some content in a section.
"""


@pytest.fixture
def sample_rst_file(tmp_path: Path, sample_rst_content: str) -> Path:
    """Create a temporary RST file."""
    file_path = tmp_path / "test.rst"
    file_path.write_text(sample_rst_content)
    return file_path


@pytest.fixture
def sample_long_text() -> str:
    """Return a long text string (~2000 chars) for chunking tests."""
    paragraph = "This is a test paragraph with some content. " * 10
    return (paragraph + "\n\n") * 5


@pytest.fixture
def sample_document(tmp_path: Path) -> Document:
    """Return a sample Document instance."""
    return Document(
        content="Sample document content for testing.",
        source_path=tmp_path / "sample.md",
        metadata={"title": "Sample", "author": "Tester"},
    )


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Return a mock QdrantClient."""
    mock = MagicMock()
    mock.collection_exists.return_value = False
    mock.create_collection.return_value = None
    mock.upsert.return_value = None
    return mock


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Return a mock TextEmbedding that returns fake vectors."""
    mock = MagicMock()
    # Return 384-dimensional vectors (matching all-MiniLM-L6-v2)
    mock.embed.return_value = [[0.1] * 384]
    return mock


@pytest.fixture
def sample_python_code() -> str:
    """Return sample Python code with function and class."""
    return '''"""A sample module."""

def greet(name: str) -> str:
    """Greet a person by name.

    Args:
        name: The person's name.

    Returns:
        A greeting message.
    """
    return f"Hello, {name}!"


class Greeter:
    """A class for greeting people."""

    def __init__(self, prefix: str = "Hello"):
        """Initialize the greeter.

        Args:
            prefix: The greeting prefix.
        """
        self.prefix = prefix

    def greet(self, name: str) -> str:
        """Greet a person.

        Args:
            name: The person's name.

        Returns:
            A greeting message.
        """
        return f"{self.prefix}, {name}!"
'''


@pytest.fixture
def sample_python_file(tmp_path: Path, sample_python_code: str) -> Path:
    """Create a temporary Python file."""
    file_path = tmp_path / "test.py"
    file_path.write_text(sample_python_code)
    return file_path


@pytest.fixture
def sample_php_code() -> str:
    """Return sample PHP code with class and methods."""
    return '''<?php
/**
 * A sample PHP file.
 */

/**
 * Greet a person by name.
 *
 * @param string $name The person's name
 * @return string A greeting message
 */
function greet($name) {
    return "Hello, $name!";
}

/**
 * A class for greeting people.
 */
class Greeter {
    /**
     * @var string Greeting prefix
     */
    private $prefix;

    /**
     * Initialize the greeter.
     *
     * @param string $prefix The greeting prefix
     */
    public function __construct($prefix = "Hello") {
        $this->prefix = $prefix;
    }

    /**
     * Greet a person.
     *
     * @param string $name The person's name
     * @return string A greeting message
     */
    public function greet($name) {
        return $this->prefix . ", " . $name . "!";
    }
}
'''


@pytest.fixture
def sample_php_file(tmp_path: Path, sample_php_code: str) -> Path:
    """Create a temporary PHP file."""
    file_path = tmp_path / "test.php"
    file_path.write_text(sample_php_code)
    return file_path


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


@pytest.fixture
def sample_rustdoc_content() -> str:
    """Return rustdoc-style HTML structure."""
    return """<!DOCTYPE html>
<html>
<head>
    <title>MyStruct in my_crate::module - Rust</title>
</head>
<body class="rustdoc struct">
    <div class="sidebar">Sidebar content</div>
    <div class="search-form">Search</div>
    <h1 class="fqn">my_crate::module::MyStruct</h1>
    <pre class="rust struct">pub struct MyStruct { /* fields */ }</pre>
    <div class="docblock">
        <p>Documentation for MyStruct.</p>
    </div>
    <footer class="rustdoc-footer">Footer</footer>
</body>
</html>"""


@pytest.fixture
def sample_rustdoc_file(tmp_path: Path, sample_rustdoc_content: str) -> Path:
    """Create a temporary rustdoc HTML file."""
    file_path = tmp_path / "rustdoc.html"
    file_path.write_text(sample_rustdoc_content)
    return file_path


@pytest.fixture
def sample_markdown_with_code_blocks() -> str:
    """Markdown with code blocks for chunking tests."""
    return '''# Main Title

Introduction paragraph.

## Code Examples

Here is some Python code:

```python
# This header inside code block should NOT be split
def example():
    """A function."""
    return 42
```

### More Details

Additional content after code block.

```javascript
// Another code block
const x = "## Not a header";
```

## Conclusion

Final section.
'''


@pytest.fixture
def sample_large_markdown() -> str:
    """Large markdown document (~3000 chars) for chunking tests."""
    section_content = "This is paragraph content. " * 50  # ~1350 chars per section
    return f'''# Document Title

{section_content}

## First Section

{section_content}

### Subsection A

Some shorter content here.

### Subsection B

More short content.

## Second Section

{section_content}
'''
