"""Stateless payload builder functions for Qdrant points."""

import hashlib
from datetime import datetime
from pathlib import Path

from qdrant_indexer.models import CodeSymbol, ExtractedImage


def generate_point_id(file_path: Path, chunk_index: int) -> int:
    """Generate a stable point ID from file path and chunk index.

    Args:
        file_path: Path to the source file.
        chunk_index: Index of the chunk within the file.

    Returns:
        Positive int64 ID.
    """
    key = f"{file_path.absolute()}-{chunk_index}"
    hash_obj = hashlib.sha256(key.encode())
    # Convert first 8 bytes to int64 and ensure positive
    return int.from_bytes(hash_obj.digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def base_metadata(
    file_path: Path,
    metadata: dict,
    chunk_index: int | None = None,
    total_chunks: int | None = None,
) -> dict:
    """Build shared metadata dict, always excluding 'symbols'.

    chunk_index and total_chunks are only included when provided; image
    payloads omit them to avoid polluting the image schema with text-chunk
    fields.
    """
    base: dict = {
        "source": str(file_path.absolute()),
        "timestamp": datetime.now().isoformat(),
    }
    if chunk_index is not None:
        base["chunk_index"] = chunk_index
    if total_chunks is not None:
        base["total_chunks"] = total_chunks
    for key, value in metadata.items():
        if key != "symbols":
            base[key] = value
    return base


def build_payload(
    chunk: str,
    file_path: Path,
    chunk_index: int,
    total_chunks: int,
    metadata: dict,
) -> dict:
    """Build the payload dict for a Qdrant point.

    Args:
        chunk: The text content of the chunk.
        file_path: Path to the source file.
        chunk_index: Index of this chunk.
        total_chunks: Total number of chunks from the source.
        metadata: Additional metadata from the document loader.

    Returns:
        Payload dict with all fields.
    """
    return {
        "document": chunk,  # Field name required by qdrant-mcp
        "metadata": base_metadata(
            file_path, metadata, chunk_index=chunk_index, total_chunks=total_chunks
        ),
    }


def build_code_payload(
    chunk: str,
    symbol: CodeSymbol,
    file_path: Path,
    chunk_index: int,
    total_chunks: int,
    metadata: dict,
) -> dict:
    """Build the payload dict for a code symbol point.

    Args:
        chunk: The text content of the chunk.
        symbol: The code symbol this chunk represents.
        file_path: Path to the source file.
        chunk_index: Index of this chunk.
        total_chunks: Total number of chunks from the source.
        metadata: Additional metadata from the document loader.

    Returns:
        Payload dict with all fields including code-specific metadata.
    """
    nested_metadata = base_metadata(
        file_path, metadata, chunk_index=chunk_index, total_chunks=total_chunks
    )
    # Code-specific metadata
    nested_metadata.update(
        {
            "language": symbol.language,
            "symbol_type": symbol.symbol_type,
            "symbol_name": symbol.name,
            "symbol_qualified_name": symbol.qualified_name,
            "signature": symbol.signature or "",
            "docstring": symbol.docstring or "",
            "line_start": symbol.line_start,
            "line_end": symbol.line_end,
            "parent_class": symbol.parent or "",
            "visibility": symbol.visibility or "",
        }
    )

    return {
        "document": chunk,  # Field name required by qdrant-mcp
        "metadata": nested_metadata,
    }


def generate_image_point_id(file_path: Path, image_index: int) -> int:
    """Generate a stable point ID for an image.

    Uses a different namespace than text chunks to avoid collisions.

    Args:
        file_path: Path to the source file.
        image_index: Index of the image within the file.

    Returns:
        Positive int64 ID.
    """
    key = f"image:{file_path.absolute()}-{image_index}"
    hash_obj = hashlib.sha256(key.encode())
    return int.from_bytes(hash_obj.digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def build_image_payload(
    image: ExtractedImage,
    file_path: Path,
    image_index: int,
    total_images: int,
    metadata: dict,
) -> dict:
    """Build the payload dict for an image point.

    Args:
        image: ExtractedImage object with image data and context.
        file_path: Path to the source file.
        image_index: Index of this image.
        total_images: Total number of images from the source.
        metadata: Additional metadata from the document loader.

    Returns:
        Payload dict with all fields including image-specific metadata.
    """
    # Build document text from caption and surrounding text
    doc_parts = []
    if image.caption:
        doc_parts.append(image.caption)
    if image.surrounding_text:
        doc_parts.append(image.surrounding_text)
    document_text = " ".join(doc_parts) if doc_parts else ""

    nested_metadata = base_metadata(file_path, metadata)
    # Image-specific metadata
    nested_metadata.update(
        {
            "content_type": "image",
            "image_index": image_index,
            "total_images": total_images,
            "page_number": image.page_number,
            "width": image.width,
            "height": image.height,
            "bbox": list(image.bbox),
            "caption": image.caption or "",
            "surrounding_text": image.surrounding_text or "",
            "image_hash": image.image_hash or "",
        }
    )

    return {
        "document": document_text,
        "metadata": nested_metadata,
    }
