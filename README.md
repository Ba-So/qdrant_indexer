# Qdrant Indexer

Index documentation into [Qdrant](https://qdrant.tech/) vector database for semantic search.

## Features

- Index Markdown, Text, PDF, and ReStructuredText files
- Recursive text chunking with configurable size and overlap
- YAML frontmatter extraction for Markdown files
- Progress reporting with rich terminal output
- Configurable file exclusion patterns
- Nix flake for reproducible builds

## Installation

### Using Nix (recommended)

```bash
# Run directly
nix run github:user/qdrant-indexer -- --help

# Or enter development shell
nix develop
```

### Using uv

```bash
# Clone and install
git clone https://github.com/user/qdrant-indexer.git
cd qdrant-indexer
uv sync

# Run
uv run qdrant-indexer --help
```

### Using pip

```bash
pip install qdrant-indexer
```

## Quick Start

1. Start a Qdrant instance:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. Index your documentation:

```bash
qdrant-indexer index ./docs -c my-docs
```

3. Query using Qdrant client or compatible tools like qdrant-mcp.

## CLI Reference

### `index`

Index a directory into a Qdrant collection.

```bash
qdrant-indexer index PATH -c COLLECTION [OPTIONS]
```

**Arguments:**
- `PATH` - Directory to index (required)

**Options:**
- `-c, --collection` - Collection name (required)
- `-u, --url` - Qdrant server URL (default: `http://localhost:6333`)
- `-p, --pattern` - Glob pattern for files (default: `**/*.md`)
- `--chunk-size` - Chunk size in characters (default: 512)
- `--chunk-overlap` - Overlap between chunks (default: 50)
- `--batch-size` - Batch size for uploads (default: 100)
- `-e, --exclude` - Patterns to exclude (can be repeated)
- `--no-default-excludes` - Don't use default exclusion patterns
- `-v, --verbose` - Increase verbosity (-v for info, -vv for debug)
- `-q, --quiet` - Suppress non-error output

**Examples:**

```bash
# Index markdown files
qdrant-indexer index ./docs -c documentation

# Index multiple file types
qdrant-indexer index ./docs -c docs -p "**/*.{md,txt,rst}"

# Custom chunk size and exclusions
qdrant-indexer index ./src -c code-docs \
  --chunk-size 1024 \
  --chunk-overlap 100 \
  -e "test/**" \
  -e "*.test.md"

# Verbose output
qdrant-indexer index ./docs -c docs -v
```

### `list-collections`

List all Qdrant collections with their point counts.

```bash
qdrant-indexer list-collections [-u URL]
```

### `delete-collection`

Delete a Qdrant collection.

```bash
qdrant-indexer delete-collection COLLECTION [-u URL] [-y]
```

**Options:**
- `-y, --yes` - Skip confirmation prompt

### `show-excludes`

Show default exclusion patterns.

```bash
qdrant-indexer show-excludes
```

## Configuration File

Create `.qdrant-indexer.toml` in your project directory:

```toml
[qdrant]
url = "http://localhost:6333"

[embedding]
model = "sentence-transformers/all-MiniLM-L6-v2"

[chunking]
strategy = "recursive"
chunk_size = 512
chunk_overlap = 50

[indexing]
batch_size = 100
pattern = "**/*.md"
exclude_patterns = [
    "node_modules/**",
    "draft/**",
]
```

The config file is searched in the current directory and parent directories.

## Environment Variables

- `QDRANT_URL` - Override Qdrant server URL
- `FASTEMBED_CACHE_PATH` - Custom cache directory for embedding models

## Supported File Formats

| Format | Extensions | Features |
|--------|------------|----------|
| Markdown | `.md`, `.markdown` | YAML frontmatter extraction |
| Plain Text | `.txt`, `.text` | Basic metadata |
| PDF | `.pdf` | Text extraction from all pages |
| ReStructuredText | `.rst` | Title extraction from headings |

## Default Exclusion Patterns

The following patterns are excluded by default:

- `node_modules/**`
- `.git/**`
- `__pycache__/**`
- `*.pyc`
- `venv/**`, `.venv/**`
- `.tox/**`, `.nox/**`
- `.mypy_cache/**`, `.pytest_cache/**`, `.ruff_cache/**`
- `*.egg-info/**`
- `dist/**`, `build/**`
- `.direnv/**`

Use `--no-default-excludes` to disable these.

## Embedding Model

This tool uses [FastEmbed](https://github.com/qdrant/fastembed) with the `sentence-transformers/all-MiniLM-L6-v2` model by default:

- Vector dimensions: 384
- Distance metric: Cosine

This is compatible with [qdrant-mcp](https://github.com/qdrant/mcp-server-qdrant) when configured with the same embedding model.

## Development

### Running Tests

```bash
# Run unit tests only
uv run pytest -m "not integration"

# Run integration tests (requires Qdrant)
docker compose up -d
uv run pytest -m integration

# Run all tests
uv run pytest
```

### Test Coverage

- **Unit tests** (`tests/test_*.py` except integration): Test loaders, chunkers, and indexer with mocked dependencies
- **Integration tests** (`tests/test_integration.py`): Test full pipeline against real Qdrant instance

## Troubleshooting

### Connection refused to Qdrant

Ensure Qdrant is running:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### ONNX Runtime warning

The warning `Init provider bridge failed` from ONNX Runtime is non-critical and can be ignored. It indicates GPU acceleration is not available, falling back to CPU.

### Large files cause memory issues

Reduce `--batch-size` or process files in smaller batches:

```bash
qdrant-indexer index ./docs -c docs --batch-size 25
```

### Files not being indexed

Check if files match your pattern and aren't excluded:

```bash
# See what's excluded by default
qdrant-indexer show-excludes

# Use verbose mode to see processing details
qdrant-indexer index ./docs -c docs -vv
```

## License

MIT
