"""Document loaders for different file formats."""

import hashlib
import io
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from bs4 import BeautifulSoup
import frontmatter
import fitz  # pymupdf
import pymupdf.layout  # must be imported before pymupdf4llm to enable layout analysis
import pymupdf4llm

from qdrant_indexer.models import Document, ExtractedImage

logger = logging.getLogger(__name__)


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


class PDFImageExtractor:
    """Extracts and deduplicates images from PDF pages.

    Encapsulates all image-extraction logic so that PDFLoader stays focused
    on text and metadata concerns.  PDFLoader holds an instance of this class
    and delegates image-related calls to it.
    """

    # Minimum pixel dimension (width AND height) an image must have to be kept.
    MIN_IMAGE_SIZE: int = 100

    # Points added around an image bbox when harvesting surrounding text.
    SURROUNDING_TEXT_MARGIN: int = 50

    # Maximum characters kept from surrounding-text result (excess is truncated).
    SURROUNDING_TEXT_LIMIT: int = 500

    # Points below the image bottom edge that are searched for a caption.
    CAPTION_SEARCH_DISTANCE: int = 100

    # Horizontal padding added when building the caption search rectangle.
    CAPTION_HORIZONTAL_PADDING: int = 20

    # Maximum characters kept from a detected caption (excess is truncated).
    CAPTION_CHAR_LIMIT: int = 300

    def __init__(self, min_image_size: int = MIN_IMAGE_SIZE) -> None:
        """Initialize the extractor.

        Args:
            min_image_size: Minimum pixel dimension for images to be kept.
        """
        self.min_image_size = min_image_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_images(self, path: Path) -> list[ExtractedImage]:
        """Extract unique, size-filtered images from every page of a PDF.

        Opens the document, iterates pages, converts each qualifying raw image
        to PNG, deduplicates by MD5 hash, and attaches surrounding text and
        caption context before returning the collected results.

        Args:
            path: Path to the PDF file.

        Returns:
            List of ExtractedImage objects ordered by page then appearance.
        """
        from PIL import Image

        images: list[ExtractedImage] = []
        seen_hashes: set[str] = set()

        doc = fitz.open(path)
        try:
            for page_num, page in enumerate(doc, start=1):
                for img_info in page.get_images(full=True):
                    xref = img_info[0]
                    image = self._process_single_image(
                        doc, page, page_num, xref, seen_hashes, Image
                    )
                    if image is not None:
                        images.append(image)
        finally:
            doc.close()

        return images

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_single_image(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_num: int,
        xref: int,
        seen_hashes: set[str],
        Image,
    ) -> ExtractedImage | None:
        """Extract, convert, and annotate one image xref from a page.

        Returns None when the image should be skipped (too small, unreadable,
        or already seen).
        """
        try:
            base_image = doc.extract_image(xref)
            if not base_image:
                return None

            image_bytes = base_image["image"]
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            if width < self.min_image_size or height < self.min_image_size:
                return None

            png_data = self._to_png(image_bytes, Image)
            if png_data is None:
                return None

            image_hash = hashlib.md5(png_data).hexdigest()
            if image_hash in seen_hashes:
                return None
            seen_hashes.add(image_hash)

            bbox = self._get_image_bbox(page, xref)
            if bbox is None:
                bbox = (0.0, 0.0, float(width), float(height))

            return ExtractedImage(
                image_data=png_data,
                page_number=page_num,
                bbox=bbox,
                width=width,
                height=height,
                surrounding_text=self._get_surrounding_text(page, bbox),
                caption=self._detect_caption(page, bbox),
                image_hash=image_hash,
            )
        except Exception as e:
            logger.debug(f"Skipping image xref={xref} on page {page_num}: {e}")
            return None

    def _to_png(self, image_bytes: bytes, Image) -> bytes | None:
        """Convert raw image bytes to PNG format using Pillow.

        Returns None when Pillow cannot decode the image.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            logger.debug(f"Could not convert image bytes to PNG: {e}")
            return None

    def _get_image_bbox(
        self, page: fitz.Page, xref: int
    ) -> tuple[float, float, float, float] | None:
        """Return the bounding box of an image on a page, or None if absent.

        Args:
            page: PyMuPDF page object.
            xref: Image xref ID.

        Returns:
            Bounding box as (x0, y0, x1, y1) or None if not found.
        """
        for img in page.get_images(full=True):
            if img[0] == xref:
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    return (rect.x0, rect.y0, rect.x1, rect.y1)
        return None

    def _get_surrounding_text(
        self, page: fitz.Page, bbox: tuple[float, float, float, float]
    ) -> str | None:
        """Extract text from a margin around the image bbox.

        The expanded rectangle is clipped to the page boundaries so it never
        requests text outside the page.

        Args:
            page: PyMuPDF page object.
            bbox: Image bounding box as (x0, y0, x1, y1).

        Returns:
            Up to SURROUNDING_TEXT_LIMIT characters of nearby text, or None.
        """
        x0, y0, x1, y1 = bbox
        page_rect = page.rect
        margin = self.SURROUNDING_TEXT_MARGIN

        expanded_rect = fitz.Rect(
            max(0, x0 - margin),
            max(0, y0 - margin),
            min(page_rect.width, x1 + margin),
            min(page_rect.height, y1 + margin),
        )

        texts = [
            block[4].strip()
            for block in page.get_text("blocks", clip=expanded_rect)
            if len(block) >= 5 and isinstance(block[4], str) and block[4].strip()
        ]

        if not texts:
            return None

        combined = " ".join(texts)
        if len(combined) > self.SURROUNDING_TEXT_LIMIT:
            combined = combined[: self.SURROUNDING_TEXT_LIMIT] + "..."
        return combined

    def _detect_caption(
        self, page: fitz.Page, bbox: tuple[float, float, float, float]
    ) -> str | None:
        """Find a caption in the region directly below an image.

        Looks for text matching common figure/table caption patterns
        (e.g. "Figure 1:", "Fig. 2 -", "Table 3:") within
        CAPTION_SEARCH_DISTANCE points below the image bottom edge.

        Args:
            page: PyMuPDF page object.
            bbox: Image bounding box as (x0, y0, x1, y1).

        Returns:
            Caption text (truncated to CAPTION_CHAR_LIMIT) or None.
        """
        x0, y0, x1, y1 = bbox
        page_rect = page.rect
        pad = self.CAPTION_HORIZONTAL_PADDING

        caption_rect = fitz.Rect(
            max(0, x0 - pad),
            y1,
            min(page_rect.width, x1 + pad),
            min(page_rect.height, y1 + self.CAPTION_SEARCH_DISTANCE),
        )

        caption_pattern = re.compile(
            r"^(Figure|Fig\.?|Table|Plate|Image|Photo|Diagram|Chart|Graph|Illustration)\s*"
            r"(\d+(?:\.\d+)?)\s*[:\.\-—–]?\s*(.*)$",
            re.IGNORECASE,
        )

        for block in page.get_text("blocks", clip=caption_rect):
            if len(block) >= 5 and isinstance(block[4], str):
                text = block[4].strip()
                if text and caption_pattern.match(text):
                    if len(text) > self.CAPTION_CHAR_LIMIT:
                        text = text[: self.CAPTION_CHAR_LIMIT] + "..."
                    return text

        return None


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
        min_image_size: int = PDFImageExtractor.MIN_IMAGE_SIZE,
    ):
        """Initialize PDF loader.

        Args:
            extract_images: Whether to extract images from PDFs.
            min_image_size: Minimum image dimension (width or height) in pixels.
                           Images smaller than this are filtered out.
        """
        self.extract_images_enabled = extract_images
        self._image_extractor = PDFImageExtractor(min_image_size=min_image_size)

    @property
    def min_image_size(self) -> int:
        """Minimum image dimension in pixels (delegates to image extractor)."""
        return self._image_extractor.min_image_size

    # Threshold for replacement character ratio above which text is considered garbled
    GARBLED_THRESHOLD = 0.3

    def load(self, path: Path) -> Document:
        """Load a PDF file, extracting text with proper table formatting.

        Uses pymupdf4llm to convert PDF content to Markdown format,
        which preserves table structure and document hierarchy.
        Falls back to raw text extraction if pymupdf4llm produces garbled
        output (common with older PDFs using non-standard font encodings).

        Extracts available PDF metadata (title, author, subject, keywords,
        creation date, etc.) when present.
        """
        try:
            md_text = pymupdf4llm.to_markdown(str(path))
        except Exception as e:
            logger.warning(
                "pymupdf4llm failed, falling back to raw text extraction: %s (%s)",
                path.name,
                e,
            )
            md_text = self._extract_raw_text(path)

        if self._is_garbled(md_text):
            logger.warning(
                "PDF text extraction produced garbled output, "
                "falling back to raw text extraction: %s",
                path.name,
            )
            md_text = self._extract_raw_text(path)

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

    def _is_garbled(self, text: str) -> bool:
        """Check if extracted text is garbled (high ratio of replacement characters).

        Old PDFs with non-standard font encodings (e.g., AdvTimes) can cause
        pymupdf4llm to produce U+FFFD replacement characters instead of text.

        Args:
            text: Extracted text to check.

        Returns:
            True if the text appears garbled and should use fallback extraction.
        """
        if not text or len(text.strip()) == 0:
            return False
        replacement_count = text.count("\ufffd")
        return replacement_count / len(text) > self.GARBLED_THRESHOLD

    def _extract_raw_text(self, path: Path) -> str:
        """Extract text from PDF using raw PyMuPDF text extraction.

        This is a fallback for when pymupdf4llm produces garbled output.
        Uses fitz.Page.get_text() which handles old font encodings better.

        Args:
            path: Path to the PDF file.

        Returns:
            Concatenated text from all pages.
        """
        doc = fitz.open(path)
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n\n".join(pages)

    # ------------------------------------------------------------------
    # Image extraction — delegates to PDFImageExtractor
    # ------------------------------------------------------------------

    def extract_images(self, path: Path) -> list[ExtractedImage]:
        """Extract images from a PDF file.

        Delegates to PDFImageExtractor, which handles size filtering,
        PNG conversion, deduplication, and caption/context collection.

        Args:
            path: Path to the PDF file.

        Returns:
            List of ExtractedImage objects with image data and metadata.
        """
        return self._image_extractor.extract_images(path)



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


class EpubLoader(DocumentLoader):
    """Loader for EPUB e-book files.

    Extracts text from all document items in the EPUB, joining chapters with
    double newlines. Metadata is pulled from Dublin Core fields embedded in
    the EPUB container.

    Requires the ``ebooklib`` package (lazy-imported to avoid hard failure at
    startup when it is not installed).
    """

    preferred_chunker: ClassVar[str] = "recursive"

    def load(self, path: Path) -> Document:
        """Load an EPUB file, extract metadata and full text content.

        Args:
            path: Path to the ``.epub`` file.

        Returns:
            Document with joined chapter text and Dublin Core metadata.
        """
        import ebooklib
        import ebooklib.epub as epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(str(path))

        def _dc(field: str) -> list[str]:
            return [v for v, _ in book.get_metadata("DC", field)]

        title_vals = _dc("title")
        author_vals = _dc("creator")
        language_vals = _dc("language")
        publisher_vals = _dc("publisher")
        description_vals = _dc("description")
        date_vals = _dc("date")
        identifier_vals = _dc("identifier")

        metadata: dict[str, str] = {
            "filename": path.name,
            "extension": path.suffix,
        }
        if title_vals:
            metadata["title"] = title_vals[0]
        if author_vals:
            metadata["author"] = "; ".join(author_vals)
        if language_vals:
            metadata["language"] = language_vals[0]
        if publisher_vals:
            metadata["publisher"] = publisher_vals[0]
        if description_vals:
            metadata["description"] = description_vals[0]
        if date_vals:
            metadata["date"] = date_vals[0]
        if identifier_vals:
            metadata["identifier"] = identifier_vals[0]

        chapters: list[str] = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content()
            soup = BeautifulSoup(content, "lxml")
            text = soup.get_text(separator="\n", strip=True)
            if text.strip():
                chapters.append(text)

        return Document(
            content="\n\n".join(chapters),
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

    def _remove_unwanted_tags(self, soup: BeautifulSoup) -> None:
        """Remove all tags listed in UNWANTED_TAGS from the soup in-place.

        Args:
            soup: Parsed BeautifulSoup object to mutate.
        """
        for tag in self.UNWANTED_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

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
        self._remove_unwanted_tags(soup)

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

        # Remove unwanted tags (inherited from HTMLLoader)
        self._remove_unwanted_tags(soup)

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
    ".epub": EpubLoader,
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


_CODE_LOADERS: dict[str, type[DocumentLoader]] | None = None


def _get_code_loaders() -> dict[str, type[DocumentLoader]]:
    """Return the code-loader registry, building it once on first call.

    The import is deferred to avoid a circular dependency between loaders.py
    and code_loaders.py.
    """
    global _CODE_LOADERS
    if _CODE_LOADERS is None:
        from qdrant_indexer.code_loaders import (
            PHPCodeLoader,
            PythonCodeLoader,
            RustCodeLoader,
        )

        _CODE_LOADERS = {
            "python": PythonCodeLoader,
            "php": PHPCodeLoader,
            "rust": RustCodeLoader,
        }
    return _CODE_LOADERS


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
        code_type = CODE_EXTENSIONS[extension]
        loader_cls = _get_code_loaders().get(code_type)
        if loader_cls is not None:
            return loader_cls()

    # Default to text loader
    return TextLoader()
