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
