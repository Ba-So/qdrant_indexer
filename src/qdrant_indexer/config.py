"""Configuration handling for Qdrant Indexer."""

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_indexer.filters import DEFAULT_INDEX_PATTERNS

# Default embedding model name
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default batch size for embedding - smaller batches use less GPU memory
DEFAULT_EMBEDDING_BATCH_SIZE = 64

# Default number of workers for parallel processing (max 4, based on CPU count)
DEFAULT_WORKERS = min(4, (os.cpu_count() or 1))

# Default CLIP vision model for image embeddings
DEFAULT_CLIP_VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"

# PDF extensions that need process‑based parallelism (PyMuPDF is not thread‑safe)
PDF_EXTENSIONS = {".pdf"}


@dataclass
class QdrantConfig:
    """Qdrant server configuration."""

    url: str = "http://localhost:6333"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model: str = DEFAULT_EMBEDDING_MODEL
    cache_dir: str = field(default_factory=lambda: str(default_model_cache_dir()))


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
    patterns: list[str] = field(default_factory=lambda: list(DEFAULT_INDEX_PATTERNS))
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

# XDG application subdirectory for the user-global config file.
XDG_APP_DIR = "qdrant-indexer"


def xdg_config_home() -> Path:
    """Return ``$XDG_CONFIG_HOME`` or the spec default ``~/.config``."""
    raw = os.environ.get("XDG_CONFIG_HOME")
    if raw:
        return Path(raw)
    return Path.home() / ".config"


def xdg_config_path() -> Path:
    """Return the canonical XDG-Home path for the user-global config file.

    ``$XDG_CONFIG_HOME/qdrant-indexer/config.toml`` (default
    ``~/.config/qdrant-indexer/config.toml``).
    """
    return xdg_config_home() / XDG_APP_DIR / DEFAULT_CONFIG_FILENAMES[0]


def xdg_cache_home() -> Path:
    """Return ``$XDG_CACHE_HOME`` or the spec default ``~/.cache``."""
    raw = os.environ.get("XDG_CACHE_HOME")
    if raw:
        return Path(raw)
    return Path.home() / ".cache"


def default_model_cache_dir() -> Path:
    """Return the default location for downloaded embedding model files.

    ``$XDG_CACHE_HOME/qdrant-indexer/models`` (default
    ``~/.cache/qdrant-indexer/models``). Cached model weights are regenerable,
    so they belong under the XDG cache hierarchy rather than config or data.
    """
    return xdg_cache_home() / XDG_APP_DIR / "models"

# Maps CLI kwarg names to dotted config paths ("section.field").
# This is the single authoritative record of how CLI arguments map to config fields.
_CLI_OVERRIDE_MAP: dict[str, str] = {
    "url": "qdrant.url",
    "embedding_model": "embedding.model",
    "chunk_size": "chunking.chunk_size",
    "chunk_overlap": "chunking.chunk_overlap",
    "batch_size": "indexing.batch_size",
    "pattern": "indexing.patterns",
    "enable_images": "image_embedding.enabled",
    "clip_model": "image_embedding.vision_model",
    "min_image_size": "image_embedding.min_image_size",
}


def find_config_file(start_path: Path | None = None) -> Path | None:
    """Search for a config file.

    Lookup order:

    1. The XDG-Home location returned by :func:`xdg_config_path`, which is
       the canonical user-global location.
    2. Walk up from ``start_path`` (default cwd) to filesystem root checking
       for any name in :data:`DEFAULT_CONFIG_FILENAMES`.

    Args:
        start_path: Directory to start the project-local search from.

    Returns:
        Path to the config file if found, otherwise ``None``.
    """
    xdg_path = xdg_config_path()
    if xdg_path.exists():
        return xdg_path

    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    while True:
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

    # Expand ~ in path-like fields so downstream consumers (FastEmbed) see
    # an absolute path instead of creating a literal "~" directory at cwd.
    if config.embedding.cache_dir:
        config.embedding.cache_dir = str(
            Path(config.embedding.cache_dir).expanduser()
        )

    return config


def _parse_nested(dc_instance: Any, section_data: dict[str, Any]) -> Any:
    """Return a new instance of dc_instance's type, with fields overridden by section_data.

    Only keys present in section_data that correspond to actual dataclass fields are
    applied; unknown keys are silently ignored so that future TOML additions don't
    crash older code.
    """
    known_fields = {f.name for f in dataclasses.fields(dc_instance)}
    kwargs = {k: v for k, v in section_data.items() if k in known_fields}
    return dataclasses.replace(dc_instance, **kwargs)


def _parse_config(data: dict[str, Any]) -> Config:
    """Parse TOML data into Config object.

    Each top-level key in data maps to the matching Config attribute by name.
    Fields within each section are applied generically via dataclasses.replace,
    so adding a new field to a nested dataclass requires no changes here.
    """
    config = Config()
    # Config attribute names match the TOML section names exactly.
    for config_field in dataclasses.fields(config):
        section_name = config_field.name
        if section_name in data:
            updated = _parse_nested(getattr(config, section_name), data[section_name])
            setattr(config, section_name, updated)
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


def _deep_copy_config(config: Config) -> Config:
    """Return a deep copy of config, duplicating each nested dataclass instance.

    list fields are shallow-copied so that mutations to list contents in the
    copy do not affect the original (and vice-versa).
    """
    section_copies: dict[str, Any] = {}
    for config_field in dataclasses.fields(config):
        section = getattr(config, config_field.name)
        field_kwargs: dict[str, Any] = {}
        for nested_field in dataclasses.fields(section):
            value = getattr(section, nested_field.name)
            field_kwargs[nested_field.name] = list(value) if isinstance(value, list) else value
        section_copies[config_field.name] = type(section)(**field_kwargs)
    return Config(**section_copies)


def _toml_value(value: Any) -> str:
    """Format a Python scalar/list as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def render_default_config() -> str:
    """Render a TOML document populated with current dataclass defaults.

    Fields whose default value is ``None`` are emitted as commented-out
    placeholders so the user sees the available keys without setting them.
    """
    config = Config()
    lines: list[str] = []
    for section_field in dataclasses.fields(config):
        section = getattr(config, section_field.name)
        lines.append(f"[{section_field.name}]")
        for f in dataclasses.fields(section):
            value = getattr(section, f.name)
            if value is None:
                lines.append(f"# {f.name} =")
            else:
                lines.append(f"{f.name} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def merge_config(config: Config, **overrides: Any) -> Config:
    """Merge CLI arguments into config (CLI takes precedence).

    The mapping from kwarg names to config fields is defined in _CLI_OVERRIDE_MAP.
    Only non-None override values are applied, preserving the loaded config value
    when a CLI flag was not supplied.

    Args:
        config: Base configuration.
        **overrides: CLI argument overrides.

    Returns:
        New Config with overrides applied.
    """
    result = _deep_copy_config(config)

    for kwarg_name, dotted_path in _CLI_OVERRIDE_MAP.items():
        value = overrides.get(kwarg_name)
        if value is None:
            continue
        section_name, field_name = dotted_path.split(".", 1)
        section = getattr(result, section_name)
        setattr(section, field_name, value)

    return result
