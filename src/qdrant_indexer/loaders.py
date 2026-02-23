"""Document loaders for different file formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from bs4 import BeautifulSoup
import frontmatter
import fitz  # pymupdf
import pymupdf.layout  # must be imported before pymupdf4llm to enable layout analysis
import pymupdf4llm

from qdrant_indexer.models import Document


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, path: Path) -> Document:
        """Load a document from the given path.

        Args:
            path: Path to the file to load.

        Returns:
            Document with content and metadata.
        """
        pass


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files with YAML frontmatter support."""

    def load(self, path: Path) -> Document:
        """Load a Markdown file, extracting frontmatter as metadata."""
        post = frontmatter.load(path)
        metadata = dict(post.metadata)
        metadata["filename"] = path.name
        metadata["extension"] = path.suffix

        return Document(
            content=post.content,
            source_path=path,
            metadata=metadata,
        )


class TextLoader(DocumentLoader):
    """Loader for plain text files."""

    def load(self, path: Path) -> Document:
        """Load a plain text file with basic metadata."""
        content = path.read_text(encoding="utf-8")
        stat = path.stat()

        return Document(
            content=content,
            source_path=path,
            metadata={
                "filename": path.name,
                "extension": path.suffix,
                "size": stat.st_size,
                "modified_time": stat.st_mtime,
            },
        )


class PDFLoader(DocumentLoader):
    """Loader for PDF files using pymupdf4llm for LLM-optimized extraction."""

    # Standard PDF metadata keys to extract
    PDF_METADATA_KEYS = [
        ("title", "title"),
        ("author", "author"),
        ("subject", "subject"),
        ("keywords", "keywords"),
        ("creator", "creator"),
        ("producer", "producer"),
    ]

    def load(self, path: Path) -> Document:
        """Load a PDF file, extracting text with proper table formatting.

        Uses pymupdf4llm to convert PDF content to Markdown format,
        which preserves table structure and document hierarchy.

        Extracts available PDF metadata (title, author, subject, keywords,
        creation date, etc.) when present.
        """
        md_text = pymupdf4llm.to_markdown(str(path))

        doc = fitz.open(path)
        page_count = len(doc)
        metadata = self._extract_metadata(doc, path)
        doc.close()

        metadata["page_count"] = page_count

        # Try to extract DOI from content if not in metadata
        if "doi" not in metadata:
            doi = self._extract_doi(md_text)
            if doi:
                metadata["doi"] = doi

        return Document(
            content=md_text,
            source_path=path,
            metadata=metadata,
        )

    def _extract_metadata(self, doc: fitz.Document, path: Path) -> dict:
        """Extract metadata from PDF document.

        Args:
            doc: Open PyMuPDF document.
            path: Path to the PDF file.

        Returns:
            Dictionary of metadata with only non-empty values.
        """
        metadata: dict = {
            "filename": path.name,
            "extension": path.suffix,
        }

        # Extract standard PDF metadata
        pdf_meta = doc.metadata
        if pdf_meta:
            for pdf_key, meta_key in self.PDF_METADATA_KEYS:
                value = pdf_meta.get(pdf_key, "")
                if value and value.strip():
                    metadata[meta_key] = value.strip()

            # Handle dates separately (may need parsing)
            for date_key in ("creationDate", "modDate"):
                date_value = pdf_meta.get(date_key, "")
                if date_value and date_value.strip():
                    parsed_date = self._parse_pdf_date(date_value)
                    if parsed_date:
                        metadata[date_key] = parsed_date

        return metadata

    def _parse_pdf_date(self, date_str: str) -> str | None:
        """Parse PDF date format to ISO format.

        PDF dates are typically in format: D:YYYYMMDDHHmmSS+HH'mm'

        Args:
            date_str: PDF date string.

        Returns:
            ISO formatted date string or None if parsing fails.
        """
        if not date_str:
            return None

        # Remove 'D:' prefix if present
        if date_str.startswith("D:"):
            date_str = date_str[2:]

        # Try to extract at least the date portion (YYYYMMDD)
        try:
            if len(date_str) >= 8:
                year = date_str[0:4]
                month = date_str[4:6]
                day = date_str[6:8]
                return f"{year}-{month}-{day}"
        except (ValueError, IndexError):
            pass

        return None

    def _extract_doi(self, text: str) -> str | None:
        """Extract DOI from document text.

        Looks for common DOI patterns in the document content.

        Args:
            text: Document text content.

        Returns:
            DOI string or None if not found.
        """
        import re

        # Common DOI patterns:
        # - doi:10.xxxx/xxxxx
        # - DOI: 10.xxxx/xxxxx
        # - https://doi.org/10.xxxx/xxxxx
        # - http://dx.doi.org/10.xxxx/xxxxx
        patterns = [
            r"(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,}/[^\s]+)",
            r"[Dd][Oo][Ii][:：]?\s*(10\.\d{4,}/[^\s]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                doi = match.group(1)
                # Clean up trailing punctuation
                doi = doi.rstrip(".,;:")
                return doi

        return None


class ReStructuredTextLoader(DocumentLoader):
    """Loader for ReStructuredText files."""

    def load(self, path: Path) -> Document:
        """Load an RST file with basic metadata."""
        content = path.read_text(encoding="utf-8")
        stat = path.stat()

        # Extract title from first underlined heading if present
        title = None
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if i > 0 and line and all(c in "=-~`" for c in line.strip()):
                title = lines[i - 1].strip()
                break

        metadata = {
            "filename": path.name,
            "extension": path.suffix,
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
        }
        if title:
            metadata["title"] = title

        return Document(
            content=content,
            source_path=path,
            metadata=metadata,
        )


class HTMLLoader(DocumentLoader):
    """Loader for HTML files with content cleaning and metadata extraction."""

    # Tags to remove for clean text extraction
    UNWANTED_TAGS = ["script", "style", "nav", "noscript", "iframe", "svg"]

    def load(self, path: Path) -> Document:
        """Load HTML file, extract metadata and clean text.

        Removes scripts, styles, navigation elements.
        Extracts metadata from <head> tags (title, description, keywords, author).
        Returns clean text content using BeautifulSoup's get_text().
        """
        content = path.read_text(encoding="utf-8")
        stat = path.stat()

        # Parse with lxml parser (fast and handles malformed HTML)
        soup = BeautifulSoup(content, "lxml")

        # Extract metadata from <head>
        metadata = self._extract_metadata(soup, path, stat)

        # Remove unwanted tags
        for tag in self.UNWANTED_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

        # Extract clean text
        text = soup.get_text(separator="\n", strip=True)

        return Document(
            content=text,
            source_path=path,
            metadata=metadata,
        )

    def _extract_metadata(self, soup: BeautifulSoup, path: Path, stat) -> dict:
        """Extract metadata from HTML <head> section."""
        metadata = {
            "filename": path.name,
            "extension": path.suffix,
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
        }

        # Extract title
        if soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()

        # Extract meta tags
        meta_mappings = [
            ("description", ["description", "og:description"]),
            ("keywords", ["keywords"]),
            ("author", ["author"]),
        ]

        for key, names in meta_mappings:
            for name in names:
                meta = soup.find("meta", attrs={"name": name}) or soup.find(
                    "meta", attrs={"property": name}
                )
                if meta and meta.get("content"):
                    metadata[key] = meta["content"].strip()
                    break

        return metadata


# Registry mapping file extensions to loader classes
# Code loaders are imported lazily in get_loader() to avoid circular imports
LOADERS: dict[str, type[DocumentLoader]] = {
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".txt": TextLoader,
    ".text": TextLoader,
    ".pdf": PDFLoader,
    ".rst": ReStructuredTextLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
}

# Code file extensions - loaded lazily
CODE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".php": "php",
    ".php3": "php",
    ".php4": "php",
    ".php5": "php",
    ".phtml": "php",
    ".rs": "rust",
}


def get_loader(file_path: Path) -> DocumentLoader:
    """Get the appropriate loader for a file based on its extension.

    Args:
        file_path: Path to the file.

    Returns:
        An instance of the appropriate DocumentLoader.
        Falls back to TextLoader for unknown extensions.
    """
    extension = file_path.suffix.lower()

    # Check standard loaders first
    if extension in LOADERS:
        return LOADERS[extension]()

    # Check code loaders (lazy import to avoid circular dependency)
    if extension in CODE_EXTENSIONS:
        from qdrant_indexer.code_loaders import (
            PHPCodeLoader,
            PythonCodeLoader,
            RustCodeLoader,
        )

        code_type = CODE_EXTENSIONS[extension]
        if code_type == "python":
            return PythonCodeLoader()
        elif code_type == "php":
            return PHPCodeLoader()
        elif code_type == "rust":
            return RustCodeLoader()

    # Default to text loader
    return TextLoader()
