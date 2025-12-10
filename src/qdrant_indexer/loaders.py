"""Document loaders for different file formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import frontmatter
import fitz  # pymupdf

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
    """Loader for PDF files using pymupdf."""

    def load(self, path: Path) -> Document:
        """Load a PDF file, extracting text from all pages."""
        doc = fitz.open(path)
        pages_text = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():
                pages_text.append(text)

        content = "\n\n".join(pages_text)
        page_count = len(doc)
        doc.close()

        return Document(
            content=content,
            source_path=path,
            metadata={
                "filename": path.name,
                "extension": path.suffix,
                "page_count": page_count,
            },
        )


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


# Registry mapping file extensions to loader classes
# Code loaders are imported lazily in get_loader() to avoid circular imports
LOADERS: dict[str, type[DocumentLoader]] = {
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".txt": TextLoader,
    ".text": TextLoader,
    ".pdf": PDFLoader,
    ".rst": ReStructuredTextLoader,
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
        from qdrant_indexer.code_loaders import PHPCodeLoader, PythonCodeLoader

        code_type = CODE_EXTENSIONS[extension]
        if code_type == "python":
            return PythonCodeLoader()
        elif code_type == "php":
            return PHPCodeLoader()

    # Default to text loader
    return TextLoader()
