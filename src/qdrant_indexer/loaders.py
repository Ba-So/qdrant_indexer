"""Document loaders for different file formats."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from bs4 import BeautifulSoup
import frontmatter
import fitz  # pymupdf
import pymupdf.layout  # must be imported before pymupdf4llm to enable layout analysis
import pymupdf4llm

from qdrant_indexer.models import Document, ExtractedImage


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    preferred_chunker: ClassVar[str] = "recursive"  # Default fallback

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

    preferred_chunker: ClassVar[str] = "markdown"

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

    preferred_chunker: ClassVar[str] = "recursive"

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

    preferred_chunker: ClassVar[str] = "semantic"

    # Standard PDF metadata keys to extract
    PDF_METADATA_KEYS = [
        ("title", "title"),
        ("author", "author"),
        ("subject", "subject"),
        ("keywords", "keywords"),
        ("creator", "creator"),
        ("producer", "producer"),
    ]

    def __init__(
        self,
        extract_images: bool = False,
        min_image_size: int = 100,
    ):
        """Initialize PDF loader.

        Args:
            extract_images: Whether to extract images from PDFs.
            min_image_size: Minimum image dimension (width or height) in pixels.
                           Images smaller than this are filtered out.
        """
        self.extract_images_enabled = extract_images
        self.min_image_size = min_image_size

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

    def extract_images(self, path: Path) -> list[ExtractedImage]:
        """Extract images from a PDF file.

        Uses PyMuPDF to extract embedded images, converts them to PNG format,
        and captures surrounding text and captions for context.

        Args:
            path: Path to the PDF file.

        Returns:
            List of ExtractedImage objects with image data and metadata.
        """
        import hashlib
        import io

        from PIL import Image

        images: list[ExtractedImage] = []
        seen_hashes: set[str] = set()

        doc = fitz.open(path)

        for page_num, page in enumerate(doc, start=1):
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]

                try:
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue

                    image_bytes = base_image["image"]
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Filter by minimum size
                    if width < self.min_image_size or height < self.min_image_size:
                        continue

                    # Convert to PNG for consistency
                    try:
                        img = Image.open(io.BytesIO(image_bytes))
                        png_buffer = io.BytesIO()
                        img.save(png_buffer, format="PNG")
                        png_data = png_buffer.getvalue()
                    except Exception:
                        # If PIL fails, skip this image
                        continue

                    # Compute MD5 hash for deduplication
                    image_hash = hashlib.md5(png_data).hexdigest()

                    # Skip duplicates
                    if image_hash in seen_hashes:
                        continue
                    seen_hashes.add(image_hash)

                    # Get image bounding box on the page
                    bbox = self._get_image_bbox(page, xref)
                    if bbox is None:
                        # Fallback: use full page dimensions
                        bbox = (0.0, 0.0, float(width), float(height))

                    # Extract surrounding text
                    surrounding_text = self._get_surrounding_text(page, bbox)

                    # Detect caption
                    caption = self._detect_caption(page, bbox)

                    images.append(
                        ExtractedImage(
                            image_data=png_data,
                            page_number=page_num,
                            bbox=bbox,
                            width=width,
                            height=height,
                            surrounding_text=surrounding_text,
                            caption=caption,
                            image_hash=image_hash,
                        )
                    )

                except Exception:
                    # Skip problematic images
                    continue

        doc.close()
        return images

    def _get_image_bbox(
        self, page: fitz.Page, xref: int
    ) -> tuple[float, float, float, float] | None:
        """Get the bounding box of an image on a page.

        Args:
            page: PyMuPDF page object.
            xref: Image xref ID.

        Returns:
            Bounding box as (x0, y0, x1, y1) or None if not found.
        """
        for img in page.get_images(full=True):
            if img[0] == xref:
                # Try to get the image rectangle from the page
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    return (rect.x0, rect.y0, rect.x1, rect.y1)
        return None

    def _get_surrounding_text(
        self, page: fitz.Page, bbox: tuple[float, float, float, float]
    ) -> str | None:
        """Extract text surrounding an image.

        Gets text from a region around the image bounding box to provide
        context for the image.

        Args:
            page: PyMuPDF page object.
            bbox: Image bounding box as (x0, y0, x1, y1).

        Returns:
            Surrounding text or None if not found.
        """
        x0, y0, x1, y1 = bbox
        page_rect = page.rect

        # Expand bbox to capture surrounding text (50 points margin)
        margin = 50
        expanded_rect = fitz.Rect(
            max(0, x0 - margin),
            max(0, y0 - margin),
            min(page_rect.width, x1 + margin),
            min(page_rect.height, y1 + margin),
        )

        # Get text blocks in the expanded region
        text_blocks = page.get_text("blocks", clip=expanded_rect)

        # Collect text from blocks (block format: (x0, y0, x1, y1, text, ...))
        texts = []
        for block in text_blocks:
            if len(block) >= 5 and isinstance(block[4], str):
                text = block[4].strip()
                if text:
                    texts.append(text)

        if texts:
            # Limit to a reasonable length
            combined = " ".join(texts)
            if len(combined) > 500:
                combined = combined[:500] + "..."
            return combined

        return None

    def _detect_caption(
        self, page: fitz.Page, bbox: tuple[float, float, float, float]
    ) -> str | None:
        """Detect caption for an image.

        Looks for text below the image that matches common caption patterns
        like "Figure 1:", "Fig. 2:", "Table 1:", etc.

        Args:
            page: PyMuPDF page object.
            bbox: Image bounding box as (x0, y0, x1, y1).

        Returns:
            Caption text or None if not found.
        """
        import re

        x0, y0, x1, y1 = bbox
        page_rect = page.rect

        # Look for caption below the image (within 100 points)
        caption_rect = fitz.Rect(
            max(0, x0 - 20),  # Slightly wider than image
            y1,  # Start at bottom of image
            min(page_rect.width, x1 + 20),
            min(page_rect.height, y1 + 100),  # Up to 100 points below
        )

        text_blocks = page.get_text("blocks", clip=caption_rect)

        # Caption patterns
        caption_pattern = re.compile(
            r"^(Figure|Fig\.?|Table|Plate|Image|Photo|Diagram|Chart|Graph|Illustration)\s*"
            r"(\d+(?:\.\d+)?)\s*[:\.\-—–]?\s*(.*)$",
            re.IGNORECASE,
        )

        for block in text_blocks:
            if len(block) >= 5 and isinstance(block[4], str):
                text = block[4].strip()
                if text:
                    match = caption_pattern.match(text)
                    if match:
                        # Return the full caption text
                        if len(text) > 300:
                            text = text[:300] + "..."
                        return text

        return None


class ReStructuredTextLoader(DocumentLoader):
    """Loader for ReStructuredText files."""

    preferred_chunker: ClassVar[str] = "recursive"

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
    """Loader for HTML files with content cleaning and metadata extraction.

    Automatically detects and delegates to specialized doc loaders (e.g., RustdocLoader)
    based on HTML content markers. See HTML_DOC_LOADERS registry.
    """

    preferred_chunker: ClassVar[str] = "html"

    # Tags to remove for clean text extraction
    UNWANTED_TAGS = ["script", "style", "nav", "noscript", "iframe", "svg"]

    def load(self, path: Path) -> Document:
        """Load HTML file, extract metadata and clean text.

        Automatically detects specialized documentation formats (rustdoc, etc.)
        and delegates to the appropriate loader.

        Removes scripts, styles, navigation elements.
        Extracts metadata from <head> tags (title, description, keywords, author).
        Returns clean text content using BeautifulSoup's get_text().
        """
        content = path.read_text(encoding="utf-8")
        stat = path.stat()

        # Parse with lxml parser (fast and handles malformed HTML)
        soup = BeautifulSoup(content, "lxml")

        # Check for specialized doc loaders (imported at end of module to avoid circular refs)
        for loader_cls in HTML_DOC_LOADERS:
            if loader_cls.can_handle(soup):
                return loader_cls()._load_from_soup(soup, path, stat)

        # Default HTML processing
        return self._load_from_soup(soup, path, stat)

    def _load_from_soup(self, soup: BeautifulSoup, path: Path, stat) -> Document:
        """Process parsed HTML soup into a Document.

        Args:
            soup: Parsed BeautifulSoup object.
            path: Path to the source file.
            stat: File stat result.

        Returns:
            Document with extracted content and metadata.
        """
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


class RustdocLoader(HTMLLoader):
    """Specialized loader for rustdoc-generated HTML documentation.

    Extends HTMLLoader with Rust-specific metadata extraction:
    - module_path from .fqn element
    - item_type from body class
    - signature from .rust code block
    - Additional filtering of .sidebar, .search-form, .rustdoc-footer
    """

    # Additional rustdoc-specific classes to remove
    RUSTDOC_UNWANTED_CLASSES = ["sidebar", "search-form", "rustdoc-footer"]

    @classmethod
    def can_handle(cls, soup: BeautifulSoup) -> bool:
        """Check if this HTML is rustdoc-generated.

        Detects rustdoc by looking for 'rustdoc' class on the body element.
        """
        body = soup.find("body")
        return body is not None and "rustdoc" in body.get("class", [])

    def _load_from_soup(self, soup: BeautifulSoup, path: Path, stat) -> Document:
        """Process rustdoc HTML soup into a Document with Rust-specific metadata."""
        # Extract base metadata
        metadata = self._extract_metadata(soup, path, stat)

        # Add rustdoc-specific metadata
        metadata["doc_type"] = "rustdoc"

        # Extract module path from .fqn element
        fqn = soup.find(class_="fqn")
        if fqn:
            metadata["module_path"] = fqn.get_text(strip=True)

        # Extract item type from body class (e.g., "struct", "fn", "trait")
        body = soup.find("body")
        if body and body.get("class"):
            classes = body["class"]
            for cls in classes:
                if cls in [
                    "struct",
                    "fn",
                    "trait",
                    "enum",
                    "mod",
                    "type",
                    "macro",
                    "constant",
                ]:
                    metadata["item_type"] = cls
                    break

        # Extract signature from .rust code block
        rust_code = soup.find("pre", class_="rust")
        if rust_code:
            metadata["signature"] = rust_code.get_text(strip=True)

        # Remove unwanted tags (parent class tags)
        for tag_name in self.UNWANTED_TAGS:
            for element in soup.find_all(tag_name):
                element.decompose()

        # Remove rustdoc-specific classes
        for class_name in self.RUSTDOC_UNWANTED_CLASSES:
            for element in soup.find_all(class_=class_name):
                element.decompose()

        # Extract clean text
        text = soup.get_text(separator="\n", strip=True)

        return Document(
            content=text,
            source_path=path,
            metadata=metadata,
        )


# Registry of specialized HTML doc loaders, checked in order by HTMLLoader.load()
# Each loader must implement can_handle(soup) classmethod and _load_from_soup() method
# To add a new doc type: create XxxdocLoader(HTMLLoader), implement can_handle(), add here
HTML_DOC_LOADERS: list[type[HTMLLoader]] = [
    RustdocLoader,
    # Add more specialized loaders here (e.g., JavadocLoader, DoxygenLoader, SphinxLoader)
]


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
