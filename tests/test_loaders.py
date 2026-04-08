"""Tests for document loaders."""

from pathlib import Path

import pytest

from qdrant_indexer.loaders import (
    LOADERS,
    DocumentLoader,
    HTMLLoader,
    MarkdownLoader,
    PDFLoader,
    ReStructuredTextLoader,
    RustdocLoader,
    TextLoader,
    get_loader,
)

# ---------------------------------------------------------------------------
# Loader-specific fixtures (only used by this file)
# ---------------------------------------------------------------------------


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

    def test_get_loader_for_python(self, tmp_path: Path):
        """Verify .py returns PythonCodeLoader."""
        from qdrant_indexer.code_loaders import PythonCodeLoader

        loader = get_loader(tmp_path / "test.py")
        assert isinstance(loader, PythonCodeLoader)

    def test_get_loader_for_python_stub(self, tmp_path: Path):
        """Verify .pyi returns PythonCodeLoader."""
        from qdrant_indexer.code_loaders import PythonCodeLoader

        loader = get_loader(tmp_path / "test.pyi")
        assert isinstance(loader, PythonCodeLoader)

    def test_get_loader_for_php(self, tmp_path: Path):
        """Verify .php returns PHPCodeLoader."""
        from qdrant_indexer.code_loaders import PHPCodeLoader

        loader = get_loader(tmp_path / "test.php")
        assert isinstance(loader, PHPCodeLoader)

    def test_get_loader_for_php_variants(self, tmp_path: Path):
        """Verify PHP variant extensions return PHPCodeLoader."""
        from qdrant_indexer.code_loaders import PHPCodeLoader

        for ext in [".php3", ".php4", ".php5", ".phtml"]:
            loader = get_loader(tmp_path / f"test{ext}")
            assert isinstance(loader, PHPCodeLoader), f"Failed for extension {ext}"


class TestPDFLoaderHelpers:
    """Tests for PDFLoader helper methods."""

    def test_parse_pdf_date_standard_format(self):
        """Test parsing standard PDF date format."""
        loader = PDFLoader()
        # Standard PDF date: D:YYYYMMDDHHmmSS
        result = loader._parse_pdf_date("D:20231215143022")
        assert result == "2023-12-15"

    def test_parse_pdf_date_without_prefix(self):
        """Test parsing PDF date without D: prefix."""
        loader = PDFLoader()
        result = loader._parse_pdf_date("20231215143022")
        assert result == "2023-12-15"

    def test_parse_pdf_date_with_timezone(self):
        """Test parsing PDF date with timezone."""
        loader = PDFLoader()
        # With timezone: D:YYYYMMDDHHmmSS+HH'mm'
        result = loader._parse_pdf_date("D:20231215143022+01'00'")
        assert result == "2023-12-15"

    def test_parse_pdf_date_empty(self):
        """Test parsing empty date returns None."""
        loader = PDFLoader()
        assert loader._parse_pdf_date("") is None
        assert loader._parse_pdf_date(None) is None

    def test_parse_pdf_date_short_string(self):
        """Test parsing too-short date returns None."""
        loader = PDFLoader()
        assert loader._parse_pdf_date("2023") is None

    def test_extract_doi_standard_format(self):
        """Test extracting DOI in standard format."""
        loader = PDFLoader()
        text = "This paper has doi:10.1234/example.2023"
        result = loader._extract_doi(text)
        assert result == "10.1234/example.2023"

    def test_extract_doi_with_url(self):
        """Test extracting DOI from doi.org URL."""
        loader = PDFLoader()
        text = "Available at https://doi.org/10.5678/journal.abc.123"
        result = loader._extract_doi(text)
        assert result == "10.5678/journal.abc.123"

    def test_extract_doi_dx_url(self):
        """Test extracting DOI from dx.doi.org URL."""
        loader = PDFLoader()
        text = "Link: http://dx.doi.org/10.9999/paper.ref"
        result = loader._extract_doi(text)
        assert result == "10.9999/paper.ref"

    def test_extract_doi_uppercase(self):
        """Test extracting DOI with uppercase DOI prefix."""
        loader = PDFLoader()
        text = "DOI: 10.1234/UPPERCASE"
        result = loader._extract_doi(text)
        assert result == "10.1234/UPPERCASE"

    def test_extract_doi_strips_trailing_punctuation(self):
        """Test that trailing punctuation is stripped from DOI."""
        loader = PDFLoader()
        text = "See doi:10.1234/example, for more info."
        result = loader._extract_doi(text)
        assert result == "10.1234/example"

    def test_extract_doi_not_found(self):
        """Test returns None when no DOI present."""
        loader = PDFLoader()
        text = "This document has no DOI reference."
        result = loader._extract_doi(text)
        assert result is None


class TestPDFLoaderGarbledDetection:
    """Tests for PDFLoader garbled text detection and fallback."""

    def test_is_garbled_with_clean_text(self):
        """Clean text should not be detected as garbled."""
        loader = PDFLoader()
        assert loader._is_garbled("This is perfectly normal text.") is False

    def test_is_garbled_with_replacement_chars(self):
        """Text with high replacement char ratio should be detected as garbled."""
        loader = PDFLoader()
        garbled = "\ufffd" * 100 + "ok"
        assert loader._is_garbled(garbled) is True

    def test_is_garbled_with_few_replacement_chars(self):
        """Text with few replacement chars should not be detected as garbled."""
        loader = PDFLoader()
        mostly_ok = "a" * 100 + "\ufffd" * 5
        assert loader._is_garbled(mostly_ok) is False

    def test_is_garbled_empty_text(self):
        """Empty text should not be detected as garbled."""
        loader = PDFLoader()
        assert loader._is_garbled("") is False
        assert loader._is_garbled("   ") is False

    def test_is_garbled_threshold_boundary(self):
        """Test behavior at the threshold boundary."""
        loader = PDFLoader()
        # Exactly at 30%: 30 replacement chars + 70 normal = 30% ratio
        text = "\ufffd" * 30 + "a" * 70
        # 30/100 = 0.3, not > 0.3, so should be False
        assert loader._is_garbled(text) is False
        # Just over 30%
        text = "\ufffd" * 31 + "a" * 69
        assert loader._is_garbled(text) is True


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
        # Code loaders are in CODE_EXTENSIONS (lazy loading to avoid circular imports)
        from qdrant_indexer.loaders import CODE_EXTENSIONS
        assert ".py" in CODE_EXTENSIONS
        assert ".pyi" in CODE_EXTENSIONS
        assert ".php" in CODE_EXTENSIONS
        assert ".php3" in CODE_EXTENSIONS
        assert ".php4" in CODE_EXTENSIONS
        assert ".php5" in CODE_EXTENSIONS
        assert ".phtml" in CODE_EXTENSIONS

    def test_all_loaders_are_document_loaders(self):
        """Verify all registered loaders inherit from DocumentLoader."""
        for ext, loader_cls in LOADERS.items():
            assert issubclass(loader_cls, DocumentLoader), f"{ext} loader is not a DocumentLoader"


class TestHTMLLoader:
    """Tests for HTMLLoader."""

    def test_load_html_basic(self, sample_html_file: Path):
        """Verify basic HTML loading."""
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert doc.content
        assert doc.source_path == sample_html_file

    def test_html_title_extraction(self, sample_html_file: Path):
        """Verify title extracted from <title> tag."""
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert doc.metadata.get("title") == "Test HTML Document"

    def test_html_meta_extraction(self, sample_html_file: Path):
        """Verify meta tags extracted to metadata."""
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert doc.metadata.get("description") == "A test HTML document for testing HTMLLoader"
        assert doc.metadata.get("keywords") == "test, html, loader"
        assert doc.metadata.get("author") == "Test Author"

    def test_html_script_removal(self, sample_html_file: Path):
        """Verify script tags are removed from content."""
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert "console.log" not in doc.content
        assert "alert" not in doc.content

    def test_html_style_removal(self, sample_html_file: Path):
        """Verify style tags are removed from content."""
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert ".test { color: red; }" not in doc.content
        assert "color: red" not in doc.content

    def test_html_nav_removal(self, sample_html_file: Path):
        """Verify navigation elements are removed."""
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert "Navigation menu" not in doc.content

    def test_html_content_extracted(self, sample_html_file: Path):
        """Verify actual content is extracted."""
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert "Main Header" in doc.content
        assert "This is test content that should be extracted" in doc.content

    def test_malformed_html_handling(self, malformed_html_file: Path):
        """Verify malformed HTML is handled gracefully."""
        loader = HTMLLoader()
        doc = loader.load(malformed_html_file)
        assert doc.content  # Should not crash
        # Title may include extra content due to malformed HTML, but should contain expected text
        assert "Malformed HTML" in doc.metadata.get("title", "")

    def test_html_standard_metadata(self, sample_html_file: Path):
        """Verify standard metadata fields present."""
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert doc.metadata["filename"] == "test.html"
        assert doc.metadata["extension"] == ".html"
        assert "size" in doc.metadata
        assert "modified_time" in doc.metadata


class TestRustdocLoader:
    """Tests for RustdocLoader."""

    def test_rustdoc_doc_type(self, sample_rustdoc_file: Path):
        """Verify doc_type is set to 'rustdoc'."""
        loader = RustdocLoader()
        doc = loader.load(sample_rustdoc_file)
        assert doc.metadata["doc_type"] == "rustdoc"

    def test_rustdoc_module_path_extraction(self, sample_rustdoc_file: Path):
        """Verify module_path extracted from .fqn element."""
        loader = RustdocLoader()
        doc = loader.load(sample_rustdoc_file)
        assert doc.metadata.get("module_path") == "my_crate::module::MyStruct"

    def test_rustdoc_item_type_detection(self, sample_rustdoc_file: Path):
        """Verify item_type detected from body class."""
        loader = RustdocLoader()
        doc = loader.load(sample_rustdoc_file)
        assert doc.metadata.get("item_type") == "struct"

    def test_rustdoc_signature_extraction(self, sample_rustdoc_file: Path):
        """Verify signature extracted from .rust code block."""
        loader = RustdocLoader()
        doc = loader.load(sample_rustdoc_file)
        assert "MyStruct" in doc.metadata.get("signature", "")
        assert "pub struct" in doc.metadata.get("signature", "")

    def test_rustdoc_navigation_removal(self, sample_rustdoc_file: Path):
        """Verify rustdoc-specific elements removed."""
        loader = RustdocLoader()
        doc = loader.load(sample_rustdoc_file)
        assert "Sidebar content" not in doc.content
        assert "Footer" not in doc.content

    def test_rustdoc_content_extracted(self, sample_rustdoc_file: Path):
        """Verify documentation content is extracted."""
        loader = RustdocLoader()
        doc = loader.load(sample_rustdoc_file)
        assert "Documentation for MyStruct" in doc.content


class TestLoaderFactoryHTML:
    """Tests for HTML loader factory registration."""

    def test_get_loader_for_html(self, tmp_path: Path):
        """Verify .html returns HTMLLoader."""
        loader = get_loader(tmp_path / "test.html")
        assert isinstance(loader, HTMLLoader)

    def test_get_loader_for_htm(self, tmp_path: Path):
        """Verify .htm returns HTMLLoader."""
        loader = get_loader(tmp_path / "test.htm")
        assert isinstance(loader, HTMLLoader)


class TestHTMLDocAutoDetection:
    """Tests for automatic detection of specialized HTML doc formats."""

    def test_rustdoc_auto_detected_via_html_loader(self, sample_rustdoc_file: Path):
        """Verify rustdoc HTML is auto-detected when loaded via HTMLLoader."""
        loader = HTMLLoader()
        doc = loader.load(sample_rustdoc_file)
        # Should have rustdoc-specific metadata even though we used HTMLLoader
        assert doc.metadata.get("doc_type") == "rustdoc"
        assert doc.metadata.get("module_path") == "my_crate::module::MyStruct"
        assert doc.metadata.get("item_type") == "struct"

    def test_rustdoc_auto_detected_via_get_loader(self, sample_rustdoc_file: Path):
        """Verify rustdoc HTML is auto-detected when loaded via get_loader factory."""
        loader = get_loader(sample_rustdoc_file)
        doc = loader.load(sample_rustdoc_file)
        # Should have rustdoc-specific metadata
        assert doc.metadata.get("doc_type") == "rustdoc"
        assert doc.metadata.get("module_path") == "my_crate::module::MyStruct"

    def test_regular_html_no_doc_type(self, sample_html_file: Path):
        """Verify regular HTML does not get doc_type set."""
        loader = HTMLLoader()
        doc = loader.load(sample_html_file)
        assert "doc_type" not in doc.metadata


class TestPreferredChunker:
    """Tests for preferred_chunker class attribute."""

    def test_markdown_loader_preferred_chunker(self):
        assert MarkdownLoader.preferred_chunker == "markdown"

    def test_text_loader_preferred_chunker(self):
        assert TextLoader.preferred_chunker == "recursive"

    def test_pdf_loader_preferred_chunker(self):
        assert PDFLoader.preferred_chunker == "semantic"

    def test_rst_loader_preferred_chunker(self):
        assert ReStructuredTextLoader.preferred_chunker == "recursive"

    def test_html_loader_preferred_chunker(self):
        assert HTMLLoader.preferred_chunker == "html"

    def test_rustdoc_loader_preferred_chunker(self):
        assert RustdocLoader.preferred_chunker == "html"

    def test_code_loaders_preferred_chunker(self):
        from qdrant_indexer.code_loaders import (
            CodeLoader,
            PHPCodeLoader,
            PythonCodeLoader,
            RustCodeLoader,
        )

        assert CodeLoader.preferred_chunker == "code"
        assert PythonCodeLoader.preferred_chunker == "code"
        assert PHPCodeLoader.preferred_chunker == "code"
        assert RustCodeLoader.preferred_chunker == "code"

    def test_document_loader_default_preferred_chunker(self):
        assert DocumentLoader.preferred_chunker == "recursive"
