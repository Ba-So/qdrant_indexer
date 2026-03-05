# Qdrant Indexer

CLI tool for indexing documentation and code into Qdrant for semantic search.

## Qdrant Server Setup

```bash
docker compose up -d
```

This starts Qdrant with REST API on `localhost:6333` and gRPC on `localhost:6334`.

## Installation

### Using uv (recommended)

```bash
uv pip install -e .
```

### Using Nix

```bash
# CPU
nix develop --impure

# GPU/CUDA (Linux only)
nix develop .#cuda --impure
```

## Usage

### Index files

```bash
# Index markdown/text/pdf files
qdrant-indexer index ./docs -c my-docs

# Index code
qdrant-indexer index ./src -c my-code -p "**/*.py" -p "**/*.rs"

# With GPU acceleration
qdrant-indexer index ./docs -c my-docs --gpu

# Full re-index (default is incremental)
qdrant-indexer index ./docs -c my-docs --full

# Specify chunking strategy (see Chunking Strategies section)
qdrant-indexer index ./docs -c my-docs --chunker markdown
```

### Other commands

```bash
# Check indexing status
qdrant-indexer status ./docs

# List collections
qdrant-indexer list-collections

# Delete collection
qdrant-indexer delete-collection my-docs

# List available embedding models
qdrant-indexer list-models
```

## Chunking Strategies

The `--chunker` option controls how documents are split into chunks for indexing:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `auto` | Automatically selects optimal chunker based on file type | General use (default) |
| `recursive` | General-purpose chunker using paragraph/line/sentence splitting | Plain text, mixed content |
| `fixed` | Simple fixed-size chunks without semantic awareness | When consistency matters |
| `markdown` | Markdown-aware chunking that splits on header boundaries | Markdown documentation |
| `html` | HTML-aware chunking that splits on semantic tags (article, section, etc.) | Web pages, HTML docs |
| `semantic` | Embedding-based chunking that splits at semantic boundaries | Highest quality retrieval |
| `code` | Code-aware chunking that respects symbol boundaries (functions, classes) | Source code files |

### Usage Examples

```bash
# Auto-select chunker based on file type (default)
qdrant-indexer index ./docs -c my-docs --chunker auto

# Use markdown-aware chunking for documentation
qdrant-indexer index ./docs -c my-docs --chunker markdown

# Use semantic chunking for better quality (slower)
qdrant-indexer index ./pdfs -c papers --chunker semantic

# Force recursive chunking for all files
qdrant-indexer index ./docs -c my-docs --chunker recursive

# Use code chunking for source files
qdrant-indexer index ./src -c my-code --chunker code
```

### Performance Notes

- `recursive`, `fixed`: Fastest (baseline)
- `markdown`, `html`, `code`: ~10% slower than recursive
- `semantic`: 10-20x slower (uses embedding model for boundary detection)

### Troubleshooting

- **Chunks too small/large**: Adjust `--chunk-size` and `--chunk-overlap` parameters
- **Poor retrieval quality**: Try `semantic` chunker or increase overlap
- **Slow indexing**: Use `recursive` or `fixed` instead of `semantic`
- **Code symbols split incorrectly**: Use `code` chunker for source files

## MCP Setup

### Claude

Add to your Claude MCP configuration (`~/.claude.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "qdrant-docs": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "mcp-server-qdrant",
        "--qdrant-url", "http://localhost:6333",
        "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2"
      ]
    }
  }
}
```

### GitHub Copilot

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "qdrant-docs": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "mcp-server-qdrant",
        "--qdrant-url", "http://localhost:6333",
        "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2"
      ]
    }
  }
}
```

For both setups, ensure you use the same embedding model that was used during indexing.
