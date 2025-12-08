"""Tests for document loaders."""

from pathlib import Path

import pytest

from qdrant_indexer.loaders import (
    LOADERS,
    DocumentLoader,
    MarkdownLoader,
    PDFLoader,
    ReStructuredTextLoader,
    TextLoader,
    get_loader,
)


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    def test_load_markdown_with_frontmatter(self, sample_markdown_file: Path):
        """Verify frontmatter is extracted to metadata."""
        loader = MarkdownLoader()
        doc = loader.load(sample_markdown_file)

        assert doc.metadata["title"] == "Test Document"
        assert doc.metadata["author"] == "Test Author"
        assert doc.metadata["tags"] == ["test", "sample"]

    def test_load_markdown_content_without_frontmatter(self, sample_markdown_file: Path):
        """Verify content does not include frontmatter."""
        loader = MarkdownLoader()
        doc = loader.load(sample_markdown_file)

        assert "---" not in doc.content
        assert "title: Test Document" not in doc.content
        assert "# Header" in doc.content

    def test_markdown_metadata_includes_filename(self, sample_markdown_file: Path):
        """Check basic metadata is included."""
        loader = MarkdownLoader()
        doc = loader.load(sample_markdown_file)

        assert doc.metadata["filename"] == "test.md"
        assert doc.metadata["extension"] == ".md"

    def test_markdown_source_path(self, sample_markdown_file: Path):
        """Verify source_path is set correctly."""
        loader = MarkdownLoader()
        doc = loader.load(sample_markdown_file)

        assert doc.source_path == sample_markdown_file

    def test_markdown_without_frontmatter(self, tmp_path: Path):
        """Test loading markdown without frontmatter."""
        file_path = tmp_path / "no_frontmatter.md"
        file_path.write_text("# Just a header\n\nSome content.")

        loader = MarkdownLoader()
        doc = loader.load(file_path)

        assert "# Just a header" in doc.content
        assert doc.metadata["filename"] == "no_frontmatter.md"


class TestTextLoader:
    """Tests for TextLoader."""

    def test_load_text_file(self, sample_text_file: Path, sample_text_content: str):
        """Verify content loaded correctly with UTF-8."""
        loader = TextLoader()
        doc = loader.load(sample_text_file)

        assert doc.content == sample_text_content

    def test_text_loader_metadata(self, sample_text_file: Path):
        """Check filename, size, mtime in metadata."""
        loader = TextLoader()
        doc = loader.load(sample_text_file)

        assert doc.metadata["filename"] == "test.txt"
        assert doc.metadata["extension"] == ".txt"
        assert "size" in doc.metadata
        assert "modified_time" in doc.metadata
        assert doc.metadata["size"] > 0

    def test_text_source_path(self, sample_text_file: Path):
        """Verify source_path is set correctly."""
        loader = TextLoader()
        doc = loader.load(sample_text_file)

        assert doc.source_path == sample_text_file


class TestReStructuredTextLoader:
    """Tests for ReStructuredTextLoader."""

    def test_load_rst_file(self, sample_rst_file: Path, sample_rst_content: str):
        """Verify RST content loaded."""
        loader = ReStructuredTextLoader()
        doc = loader.load(sample_rst_file)

        assert doc.content == sample_rst_content

    def test_rst_title_extraction(self, sample_rst_file: Path):
        """Test title extraction from RST heading."""
        loader = ReStructuredTextLoader()
        doc = loader.load(sample_rst_file)

        assert doc.metadata.get("title") == "Test Document"

    def test_rst_metadata(self, sample_rst_file: Path):
        """Check basic metadata."""
        loader = ReStructuredTextLoader()
        doc = loader.load(sample_rst_file)

        assert doc.metadata["filename"] == "test.rst"
        assert doc.metadata["extension"] == ".rst"


class TestLoaderFactory:
    """Tests for get_loader factory function."""

    def test_get_loader_for_markdown(self, tmp_path: Path):
        """Verify .md returns MarkdownLoader."""
        loader = get_loader(tmp_path / "test.md")
        assert isinstance(loader, MarkdownLoader)

    def test_get_loader_for_markdown_long_ext(self, tmp_path: Path):
        """Verify .markdown returns MarkdownLoader."""
        loader = get_loader(tmp_path / "test.markdown")
        assert isinstance(loader, MarkdownLoader)

    def test_get_loader_for_text(self, tmp_path: Path):
        """Verify .txt returns TextLoader."""
        loader = get_loader(tmp_path / "test.txt")
        assert isinstance(loader, TextLoader)

    def test_get_loader_for_pdf(self, tmp_path: Path):
        """Verify .pdf returns PDFLoader."""
        loader = get_loader(tmp_path / "test.pdf")
        assert isinstance(loader, PDFLoader)

    def test_get_loader_for_rst(self, tmp_path: Path):
        """Verify .rst returns ReStructuredTextLoader."""
        loader = get_loader(tmp_path / "test.rst")
        assert isinstance(loader, ReStructuredTextLoader)

    def test_get_loader_default_unknown_extension(self, tmp_path: Path):
        """Verify unknown extension returns TextLoader."""
        loader = get_loader(tmp_path / "test.unknown")
        assert isinstance(loader, TextLoader)

    def test_get_loader_case_insensitive(self, tmp_path: Path):
        """Verify extension matching is case insensitive."""
        loader = get_loader(tmp_path / "test.MD")
        assert isinstance(loader, MarkdownLoader)


class TestLoadersRegistry:
    """Tests for LOADERS registry."""

    def test_loaders_contains_expected_extensions(self):
        """Verify all expected extensions are registered."""
        assert ".md" in LOADERS
        assert ".markdown" in LOADERS
        assert ".txt" in LOADERS
        assert ".text" in LOADERS
        assert ".pdf" in LOADERS
        assert ".rst" in LOADERS

    def test_all_loaders_are_document_loaders(self):
        """Verify all registered loaders inherit from DocumentLoader."""
        for ext, loader_cls in LOADERS.items():
            assert issubclass(loader_cls, DocumentLoader), f"{ext} loader is not a DocumentLoader"
