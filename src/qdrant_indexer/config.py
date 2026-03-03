"""Configuration handling for Qdrant Indexer."""

import os

# Default embedding model name
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default batch size for embedding - smaller batches use less GPU memory
DEFAULT_EMBEDDING_BATCH_SIZE = 64

# Default number of workers for parallel processing (max 4, based on CPU count)
DEFAULT_WORKERS = min(4, (os.cpu_count() or 1))

# PDF extensions that need process‑based parallelism (PyMuPDF is not thread‑safe)
PDF_EXTENSIONS = {".pdf"}

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QdrantConfig:
    """Qdrant server configuration."""

    url: str = "http://localhost:6333"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: str | None = None


@dataclass
class ChunkingConfig:
    """Text chunking configuration."""

    strategy: str = "recursive"
    chunk_size: int = 1536
    chunk_overlap: int = 200


@dataclass
class IndexingConfig:
    """Indexing behavior configuration."""

    batch_size: int = 100
    pattern: str = "**/*.md"
    exclude_patterns: list[str] = field(default_factory=list)


@dataclass
class ImageEmbeddingConfig:
    """Image embedding configuration for CLIP-based image indexing."""

    enabled: bool = False
    vision_model: str = "Qdrant/clip-ViT-B-32-vision"
    min_image_size: int = 100


@dataclass
class Config:
    """Main configuration container."""

    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    image_embedding: ImageEmbeddingConfig = field(default_factory=ImageEmbeddingConfig)


DEFAULT_CONFIG_FILENAMES = ["config.toml", ".qdrant-indexer.toml"]


def find_config_file(start_path: Path | None = None) -> Path | None:
    """Search for config file in current directory and parents.

    Args:
        start_path: Directory to start searching from (defaults to cwd).

    Returns:
        Path to config file if found, None otherwise.
    """
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    while True:
        # Check each possible config filename
        for fn in DEFAULT_CONFIG_FILENAMES:
            config_path = current / fn
            if config_path.exists():
                return config_path
        parent = current.parent
        if parent == current:
            # Reached root
            break
        current = parent

    return None


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file and environment.

    Args:
        config_path: Explicit path to config file (optional).

    Returns:
        Config object with loaded settings.
    """
    config = Config()

    # Find and load config file
    if config_path is None:
        config_path = find_config_file()

    if config_path is not None and config_path.exists():
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        config = _parse_config(data)

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    return config


def _parse_config(data: dict[str, Any]) -> Config:
    """Parse TOML data into Config object."""
    config = Config()

    if "qdrant" in data:
        qdrant_data = data["qdrant"]
        config.qdrant = QdrantConfig(
            url=qdrant_data.get("url", config.qdrant.url),
        )

    if "embedding" in data:
        embed_data = data["embedding"]
        config.embedding = EmbeddingConfig(
            model=embed_data.get("model", config.embedding.model),
            cache_dir=embed_data.get("cache_dir", config.embedding.cache_dir),
        )

    if "chunking" in data:
        chunk_data = data["chunking"]
        config.chunking = ChunkingConfig(
            strategy=chunk_data.get("strategy", config.chunking.strategy),
            chunk_size=chunk_data.get("chunk_size", config.chunking.chunk_size),
            chunk_overlap=chunk_data.get(
                "chunk_overlap", config.chunking.chunk_overlap
            ),
        )

    if "indexing" in data:
        index_data = data["indexing"]
        config.indexing = IndexingConfig(
            batch_size=index_data.get("batch_size", config.indexing.batch_size),
            pattern=index_data.get("pattern", config.indexing.pattern),
            exclude_patterns=index_data.get(
                "exclude_patterns", config.indexing.exclude_patterns
            ),
        )

    if "image_embedding" in data:
        img_data = data["image_embedding"]
        config.image_embedding = ImageEmbeddingConfig(
            enabled=img_data.get("enabled", config.image_embedding.enabled),
            vision_model=img_data.get(
                "vision_model", config.image_embedding.vision_model
            ),
            min_image_size=img_data.get(
                "min_image_size", config.image_embedding.min_image_size
            ),
        )

    return config


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides to config."""
    # QDRANT_URL overrides config file
    if qdrant_url := os.environ.get("QDRANT_URL"):
        config.qdrant.url = qdrant_url

    # FASTEMBED_CACHE_PATH for embedding cache
    if cache_dir := os.environ.get("FASTEMBED_CACHE_PATH"):
        config.embedding.cache_dir = cache_dir

    return config


def merge_config(config: Config, **overrides: Any) -> Config:
    """Merge CLI arguments into config (CLI takes precedence).

    Args:
        config: Base configuration.
        **overrides: CLI argument overrides.

    Returns:
        New Config with overrides applied.
    """
    # Create copies of nested configs
    qdrant = QdrantConfig(url=config.qdrant.url)
    embedding = EmbeddingConfig(
        model=config.embedding.model,
        cache_dir=config.embedding.cache_dir,
    )
    chunking = ChunkingConfig(
        strategy=config.chunking.strategy,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )
    indexing = IndexingConfig(
        batch_size=config.indexing.batch_size,
        pattern=config.indexing.pattern,
        exclude_patterns=list(config.indexing.exclude_patterns),
    )
    image_embedding = ImageEmbeddingConfig(
        enabled=config.image_embedding.enabled,
        vision_model=config.image_embedding.vision_model,
        min_image_size=config.image_embedding.min_image_size,
    )

    # Apply overrides
    if "url" in overrides and overrides["url"] is not None:
        qdrant.url = overrides["url"]

    if "embedding_model" in overrides and overrides["embedding_model"] is not None:
        embedding.model = overrides["embedding_model"]

    if "chunk_size" in overrides and overrides["chunk_size"] is not None:
        chunking.chunk_size = overrides["chunk_size"]

    if "chunk_overlap" in overrides and overrides["chunk_overlap"] is not None:
        chunking.chunk_overlap = overrides["chunk_overlap"]

    if "batch_size" in overrides and overrides["batch_size"] is not None:
        indexing.batch_size = overrides["batch_size"]

    if "pattern" in overrides and overrides["pattern"] is not None:
        indexing.pattern = overrides["pattern"]

    # Image embedding overrides
    if "enable_images" in overrides and overrides["enable_images"] is not None:
        image_embedding.enabled = overrides["enable_images"]

    if "clip_model" in overrides and overrides["clip_model"] is not None:
        image_embedding.vision_model = overrides["clip_model"]

    if "min_image_size" in overrides and overrides["min_image_size"] is not None:
        image_embedding.min_image_size = overrides["min_image_size"]

    return Config(
        qdrant=qdrant,
        embedding=embedding,
        chunking=chunking,
        indexing=indexing,
        image_embedding=image_embedding,
    )
