"""Pytest fixtures for qdrant_indexer tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from qdrant_indexer.models import Document


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
