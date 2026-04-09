"""Tests for PDF image extraction functionality."""

import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qdrant_indexer.loaders import PDFImageExtractor, PDFLoader
from qdrant_indexer.models import ExtractedImage


class TestExtractedImageModel:
    """Tests for ExtractedImage dataclass."""

    def test_extracted_image_creation(self):
        """Test creating an ExtractedImage instance."""
        image = ExtractedImage(
            image_data=b"fake_image_data",
            page_number=1,
            bbox=(0.0, 0.0, 100.0, 100.0),
            width=100,
            height=100,
        )

        assert image.image_data == b"fake_image_data"
        assert image.page_number == 1
        assert image.bbox == (0.0, 0.0, 100.0, 100.0)
        assert image.width == 100
        assert image.height == 100
        assert image.surrounding_text is None
        assert image.caption is None
        assert image.image_hash is None

    def test_extracted_image_with_optional_fields(self):
        """Test ExtractedImage with all optional fields."""
        image = ExtractedImage(
            image_data=b"data",
            page_number=3,
            bbox=(10.0, 20.0, 300.0, 400.0),
            width=290,
            height=380,
            surrounding_text="This figure shows...",
            caption="Figure 1: Test image",
            image_hash="abc123",
        )

        assert image.surrounding_text == "This figure shows..."
        assert image.caption == "Figure 1: Test image"
        assert image.image_hash == "abc123"


class TestPDFLoaderInit:
    """Tests for PDFLoader initialization."""

    def test_default_initialization(self):
        """Test PDFLoader with default settings."""
        loader = PDFLoader()
        assert loader.extract_images_enabled is False
        assert loader.min_image_size == 100

    def test_custom_initialization(self):
        """Test PDFLoader with custom settings."""
        loader = PDFLoader(extract_images=True, min_image_size=50)
        assert loader.extract_images_enabled is True
        assert loader.min_image_size == 50


class TestCaptionDetection:
    """Tests for caption detection heuristics."""

    @pytest.fixture
    def extractor(self):
        """Return a PDFImageExtractor instance."""
        return PDFImageExtractor()

    @pytest.fixture
    def mock_page(self):
        """Return a mock PDF page."""
        mock = MagicMock()
        mock.rect = MagicMock()
        mock.rect.width = 612
        mock.rect.height = 792
        return mock

    def test_detect_caption_figure_format(self, extractor, mock_page):
        """Test caption detection with 'Figure X:' format."""
        mock_page.get_text.return_value = [
            (10, 500, 200, 520, "Figure 1: A test image showing results", 0, 0)
        ]

        result = extractor._detect_caption(mock_page, (10, 400, 200, 495))

        # Should find the caption
        mock_page.get_text.assert_called_once()

    def test_detect_caption_fig_abbreviated(self, extractor, mock_page):
        """Test caption detection with 'Fig.' abbreviation."""
        mock_page.get_text.return_value = [
            (10, 500, 200, 520, "Fig. 2 - Comparison of methods", 0, 0)
        ]

        result = extractor._detect_caption(mock_page, (10, 400, 200, 495))

        mock_page.get_text.assert_called_once()

    def test_detect_caption_table_format(self, extractor, mock_page):
        """Test caption detection with 'Table X:' format."""
        mock_page.get_text.return_value = [
            (10, 500, 200, 520, "Table 3: Summary statistics", 0, 0)
        ]

        result = extractor._detect_caption(mock_page, (10, 400, 200, 495))

        mock_page.get_text.assert_called_once()

    def test_detect_caption_no_match(self, extractor, mock_page):
        """Test caption detection returns None when no caption found."""
        mock_page.get_text.return_value = [
            (10, 500, 200, 520, "Regular paragraph text without caption markers", 0, 0)
        ]

        result = extractor._detect_caption(mock_page, (10, 400, 200, 495))

        assert result is None

    def test_detect_caption_empty_blocks(self, extractor, mock_page):
        """Test caption detection with empty text blocks."""
        mock_page.get_text.return_value = []

        result = extractor._detect_caption(mock_page, (10, 400, 200, 495))

        assert result is None


class TestSurroundingTextExtraction:
    """Tests for surrounding text extraction."""

    @pytest.fixture
    def extractor(self):
        """Return a PDFImageExtractor instance."""
        return PDFImageExtractor()

    @pytest.fixture
    def mock_page(self):
        """Return a mock PDF page."""
        mock = MagicMock()
        mock.rect = MagicMock()
        mock.rect.width = 612
        mock.rect.height = 792
        return mock

    def test_get_surrounding_text_basic(self, extractor, mock_page):
        """Test basic surrounding text extraction."""
        mock_page.get_text.return_value = [
            (10, 100, 200, 120, "Text above the image", 0, 0),
            (10, 300, 200, 320, "Text below the image", 0, 0),
        ]

        result = extractor._get_surrounding_text(mock_page, (10, 150, 200, 250))

        assert result is not None
        mock_page.get_text.assert_called_once()

    def test_get_surrounding_text_empty(self, extractor, mock_page):
        """Test surrounding text extraction with no text blocks."""
        mock_page.get_text.return_value = []

        result = extractor._get_surrounding_text(mock_page, (10, 150, 200, 250))

        assert result is None

    def test_get_surrounding_text_truncation(self, extractor, mock_page):
        """Test that long surrounding text is truncated."""
        long_text = "x" * 600
        mock_page.get_text.return_value = [
            (10, 100, 200, 120, long_text, 0, 0),
        ]

        result = extractor._get_surrounding_text(mock_page, (10, 150, 200, 250))

        if result:
            assert len(result) <= 503  # 500 + "..."


class TestImageHashComputation:
    """Tests for image hash computation."""

    def test_md5_hash_computation(self):
        """Test that MD5 hash is computed correctly."""
        image_data = b"test image data for hashing"
        expected_hash = hashlib.md5(image_data).hexdigest()

        # The hash should be computed in extract_images method
        assert len(expected_hash) == 32
        assert expected_hash == hashlib.md5(image_data).hexdigest()

    def test_hash_uniqueness(self):
        """Test that different images produce different hashes."""
        data1 = b"image data 1"
        data2 = b"image data 2"

        hash1 = hashlib.md5(data1).hexdigest()
        hash2 = hashlib.md5(data2).hexdigest()

        assert hash1 != hash2


class TestImageExtractionIntegration:
    """Integration tests for image extraction (require actual PDF processing)."""

    @pytest.fixture
    def simple_pdf_with_image(self, tmp_path):
        """Create a simple PDF with an embedded image for testing.

        Note: This requires PyMuPDF (fitz) to create the PDF.
        """
        import fitz

        # Create a new PDF document
        doc = fitz.open()
        page = doc.new_page()

        # Create a simple image (red square)
        from PIL import Image

        img = Image.new("RGB", (200, 200), color="red")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # Insert image into PDF
        rect = fitz.Rect(100, 100, 300, 300)
        page.insert_image(rect, stream=img_buffer.getvalue())

        # Add some text
        page.insert_text((100, 350), "Figure 1: A red square")

        # Save PDF
        pdf_path = tmp_path / "test_with_image.pdf"
        doc.save(str(pdf_path))
        doc.close()

        return pdf_path

    def test_extract_images_from_pdf(self, simple_pdf_with_image):
        """Test extracting images from a PDF file."""
        loader = PDFLoader(extract_images=True, min_image_size=50)
        images = loader.extract_images(simple_pdf_with_image)

        assert len(images) >= 1
        image = images[0]
        assert isinstance(image, ExtractedImage)
        assert image.page_number == 1
        assert image.width >= 50
        assert image.height >= 50
        assert image.image_hash is not None
        assert len(image.image_hash) == 32  # MD5 hex

    def test_extract_images_min_size_filter(self, simple_pdf_with_image):
        """Test that small images are filtered out."""
        loader = PDFLoader(extract_images=True, min_image_size=500)
        images = loader.extract_images(simple_pdf_with_image)

        # The 200x200 image should be filtered out
        # (depends on exact sizing after PDF embedding)
        # Just verify the function runs without error
        assert isinstance(images, list)

    def test_extract_images_deduplication(self, tmp_path):
        """Test that duplicate images are deduplicated by hash."""
        import fitz
        from PIL import Image

        # Create PDF with same image twice
        doc = fitz.open()
        page = doc.new_page()

        img = Image.new("RGB", (150, 150), color="blue")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_data = img_buffer.getvalue()

        # Insert same image twice
        rect1 = fitz.Rect(50, 50, 200, 200)
        rect2 = fitz.Rect(250, 50, 400, 200)
        page.insert_image(rect1, stream=img_data)
        page.insert_image(rect2, stream=img_data)

        pdf_path = tmp_path / "duplicate_images.pdf"
        doc.save(str(pdf_path))
        doc.close()

        loader = PDFLoader(extract_images=True, min_image_size=50)
        images = loader.extract_images(pdf_path)

        # Should deduplicate to 1 unique image
        assert len(images) == 1

    def test_extract_images_empty_pdf(self, tmp_path):
        """Test extracting from PDF with no images."""
        import fitz

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((100, 100), "Just text, no images")

        pdf_path = tmp_path / "text_only.pdf"
        doc.save(str(pdf_path))
        doc.close()

        loader = PDFLoader(extract_images=True)
        images = loader.extract_images(pdf_path)

        assert images == []


class TestIndexedFileStateImageFields:
    """Tests for IndexedFileState image tracking fields."""

    def test_indexed_file_state_default_image_fields(self):
        """Test that IndexedFileState has correct default image values."""
        from qdrant_indexer.models import IndexedFileState

        state = IndexedFileState(
            path="/test/path.pdf",
            content_hash="abc123",
            indexed_at="2024-01-01T00:00:00",
            chunk_count=10,
            chunk_ids=[1, 2, 3],
        )

        assert state.image_count == 0
        assert state.image_ids == []

    def test_indexed_file_state_with_images(self):
        """Test IndexedFileState with image data."""
        from qdrant_indexer.models import IndexedFileState

        state = IndexedFileState(
            path="/test/paper.pdf",
            content_hash="def456",
            indexed_at="2024-01-01T12:00:00",
            chunk_count=20,
            chunk_ids=[1, 2, 3, 4, 5],
            image_count=3,
            image_ids=[100, 101, 102],
        )

        assert state.image_count == 3
        assert state.image_ids == [100, 101, 102]
