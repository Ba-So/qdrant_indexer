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
| Python | `.py`, `.pyi` | Symbol extraction (functions, classes, methods), docstring indexing, type annotations |
| PHP | `.php`, `.php3`, `.php4`, `.php5`, `.phtml` | Symbol extraction (functions, classes, methods, traits, interfaces), PHPDoc indexing |

## Performance

Performance characteristics for code indexing (measured on typical hardware):

| Metric | Python Code | PHP Code |
|--------|-------------|----------|
| Throughput | ~200-250 LOC/second | ~180-220 LOC/second |
| 10K LOC indexing time | ~40-50 seconds | ~45-55 seconds |
| Memory usage | ~500MB (embedding model) + ~50MB per 1000 files |

**Performance Tips:**

- **Batch size**: Default is 100 points per batch. Increase for better throughput on fast networks:
  ```bash
  qdrant-indexer index ./src -c code --batch-size 200
  ```

- **Chunk size**: Larger chunks reduce total chunks but may decrease search precision:
  ```bash
  qdrant-indexer index ./docs -c docs --chunk-size 1024
  ```

- **Parallel processing**: Process multiple directories separately for parallelization

- **Profiling**: Run performance tests to measure on your hardware:
  ```bash
  # Requires Qdrant running
  uv run pytest -m performance -v
  ```

## Code-Aware Indexing

Qdrant Indexer supports code-aware parsing for Python and PHP source files. Instead of treating code as plain text, it extracts structured information (functions, classes, methods, docstrings) for better semantic search over codebases.

### Features

- **Symbol-level indexing**: Each function, class, and method is indexed separately
- **Docstring extraction**: Python docstrings and PHPDoc comments are extracted and searchable
- **Rich metadata**: Store symbol type, qualified name, line numbers, visibility, and more
- **Signature preservation**: Function signatures and type annotations are preserved
- **Automatic detection**: Code files are automatically detected by extension

### Usage Examples

```bash
# Index Python codebase
qdrant-indexer index ./src -c my-code -p "**/*.py"

# Index PHP project
qdrant-indexer index ./app -c php-code -p "**/*.php"

# Index mixed codebase (Python, PHP, and docs)
qdrant-indexer index ./project -c all

# Index specific file patterns
qdrant-indexer index ./src -c api -p "**/*.{py,md}"
```

### Code Metadata

Code chunks include enhanced metadata for filtering and search:

```python
{
    "language": "python",              # "python" or "php"
    "symbol_type": "function",         # "function", "class", "method", "constant", "module"
    "symbol_name": "parse_data",       # Symbol name
    "symbol_qualified_name": "Parser.parse_data",  # Fully qualified name
    "signature": "(data: bytes) -> dict",  # Function signature with types
    "docstring": "Parse input data...",    # Extracted documentation
    "line_start": 45,                  # Start line number
    "line_end": 67,                    # End line number
    "parent_class": "Parser",          # Parent class name (for methods)
    "visibility": "public",            # PHP: public/private/protected, Python: None
}
```

### Querying Code

Use qdrant-client or qdrant-mcp to search indexed code:

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

# Search for functions related to parsing
results = client.search(
    collection_name="my-code",
    query_text="parse JSON data",
    query_filter={
        "must": [
            {"key": "language", "match": {"value": "python"}},
            {"key": "symbol_type", "match": {"value": "function"}}
        ]
    },
    limit=10
)

# Display results
for result in results:
    symbol = result.payload
    print(f"{symbol['symbol_qualified_name']} ({symbol['symbol_type']})")
    print(f"  {symbol['signature']}")
    print(f"  Lines {symbol['line_start']}-{symbol['line_end']}")
    if symbol.get('docstring'):
        print(f"  {symbol['docstring'][:100]}...")
```

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

# Run performance tests (requires Qdrant, may take 1-2 minutes)
docker compose up -d
uv run pytest -m performance

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

### Code files not parsing correctly

Ensure tree-sitter dependencies are installed:

```bash
uv sync
```

For syntax errors in source files, the indexer will log warnings and skip the file. Check logs with verbose flag:

```bash
qdrant-indexer index ./src -c code -vv
```

If code files are being treated as plain text, verify:
1. File extension is recognized (`.py`, `.pyi`, `.php`, etc.)
2. Code loader is registered for the extension
3. Check verbose output for loader selection

For parsing errors, ensure your source files are syntactically valid:

```bash
# Python syntax check
python -m py_compile yourfile.py

# PHP syntax check
php -l yourfile.php
```

## License

MIT
