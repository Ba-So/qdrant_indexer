"""Command-line interface for Qdrant Indexer."""

import logging
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from qdrant_client import QdrantClient
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from qdrant_indexer.chunkers import RecursiveChunker
from qdrant_indexer.filters import DEFAULT_EXCLUDE_PATTERNS, filter_files
from qdrant_indexer.indexer import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_WORKERS,
    QdrantIndexer,
    is_cuda_available,
)

app = typer.Typer(help="Qdrant Indexer - Index documentation into Qdrant collections")
console = Console()

# Global verbosity level
_verbosity = 0
_quiet = False


def setup_logging(verbose: int, quiet: bool) -> None:
    """Configure logging based on verbosity level."""
    global _verbosity, _quiet
    _verbosity = verbose
    _quiet = quiet

    if quiet:
        level = logging.WARNING
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def display_error(message: str) -> None:
    """Display an error message with formatting."""
    console.print(f"[red]Error:[/red] {message}")


def display_success(message: str) -> None:
    """Display a success message with formatting."""
    if not _quiet:
        console.print(f"[green]✓[/green] {message}")


@app.command()
def index(
    path: Annotated[Path, typer.Argument(help="Directory to index")],
    collection: Annotated[str, typer.Option("--collection", "-c", help="Collection name")],
    url: Annotated[str, typer.Option("--url", "-u", help="Qdrant server URL")] = "http://localhost:6333",
    embedding_model: Annotated[str, typer.Option("--embedding-model", "-m", help="FastEmbed model for embeddings")] = DEFAULT_EMBEDDING_MODEL,
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="Chunk size in characters")] = 512,
    chunk_overlap: Annotated[int, typer.Option("--chunk-overlap", help="Overlap between chunks")] = 50,
    pattern: Annotated[list[str], typer.Option("--pattern", "-p", help="Glob patterns for files (can be repeated)")] = ["**/*.md", "**/*.txt", "**/*.pdf", "**/*.rst", "**/*.py", "**/*.php"],
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size for uploads")] = 100,
    exclude: Annotated[Optional[list[str]], typer.Option("--exclude", "-e", help="Patterns to exclude (can be repeated)")] = None,
    no_default_excludes: Annotated[bool, typer.Option("--no-default-excludes", help="Don't use default exclusion patterns")] = False,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Parallel workers for file loading")] = DEFAULT_WORKERS,
    gpu: Annotated[bool, typer.Option("--gpu", "--cuda", help="Enable GPU/CUDA acceleration for embeddings")] = False,
    embedding_batch_size: Annotated[int, typer.Option("--embedding-batch-size", help="Chunks to embed at once (lower = less GPU memory)")] = DEFAULT_EMBEDDING_BATCH_SIZE,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity (-v, -vv)")] = 0,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress non-error output")] = False,
) -> None:
    """Index a directory into a Qdrant collection.

    By default, indexes documentation (.md, .txt, .pdf, .rst) and code files (.py, .php).
    Code files are parsed to extract symbols (functions, classes, methods) for better
    semantic search.

    Examples:
        # Index documentation
        qdrant-indexer index ./docs -c my-docs

        # Index with GPU acceleration (requires CUDA build)
        qdrant-indexer index ./docs -c my-docs --gpu

        # Index Python codebase
        qdrant-indexer index ./src -c my-code -p "**/*.py"

        # Index with a different embedding model (e.g., for German docs)
        qdrant-indexer index ./docs -c german-docs -m jinaai/jina-embeddings-v2-base-de

        # Index code with Jina v3 (multilingual + code support)
        qdrant-indexer index ./src -c my-code -m jinaai/jina-embeddings-v3

        # Index mixed docs and code with GPU
        qdrant-indexer index ./project -c full-index --gpu
    """
    setup_logging(verbose, quiet)

    if not path.exists():
        display_error(f"Path does not exist: {path}")
        raise typer.Exit(1)

    if not path.is_dir():
        display_error(f"Path is not a directory: {path}")
        raise typer.Exit(1)

    start_time = time.time()

    try:
        # Check GPU availability if requested
        if gpu and not quiet:
            if is_cuda_available():
                console.print("[green]✓[/green] CUDA available - GPU acceleration enabled")
            else:
                console.print("[yellow]⚠[/yellow] CUDA not available - falling back to CPU")

        # Initialize indexer (shows spinner during model loading)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=quiet,
        ) as progress:
            init_task = progress.add_task(f"Loading embedding model ({embedding_model})...", total=None)

            indexer = QdrantIndexer(
                qdrant_url=url,
                collection_name=collection,
                embedding_model=embedding_model,
                use_cuda=gpu,
            )

            progress.update(init_task, description="Connecting to Qdrant...")

            created = indexer.ensure_collection()
            progress.remove_task(init_task)

        if not quiet:
            if created:
                console.print(f"Created collection: [cyan]{collection}[/cyan]")
            else:
                console.print(f"Using existing collection: [cyan]{collection}[/cyan]")

        chunker = RecursiveChunker(chunk_size=chunk_size, overlap=chunk_overlap)

        # Build exclude patterns list
        exclude_patterns = list(exclude) if exclude else []
        if not no_default_excludes:
            exclude_patterns.extend(DEFAULT_EXCLUDE_PATTERNS)

        if not quiet:
            console.print(f"Using [cyan]{workers}[/cyan] parallel workers")

        # Progress tracking state
        progress_state = {
            "phase": "discovery",
            "current": 0,
            "total": 0,
            "message": "",
        }

        # Progress bar with phases
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            task = progress.add_task("Discovering files...", total=None)

            def on_progress(event: str, current: int, total: int, message: str) -> None:
                """Handle progress updates from indexer."""
                if event == "discovery":
                    progress.update(task, description=f"Found {total} files", total=total, completed=0)
                elif event == "loading":
                    progress.update(task, description="Loading files...", total=total, completed=current)
                elif event == "file_loaded":
                    progress.update(task, description=f"Loading: {message.split(':')[0].replace('Loaded ', '')}", completed=current)
                elif event == "file_error":
                    progress.update(task, completed=current)
                elif event == "embedding":
                    if current == 0:
                        progress.update(task, description=f"Embedding {total} chunks...", total=total, completed=0)
                    else:
                        progress.update(task, description=f"Embedding ({current}/{total})", completed=current)
                elif event == "preparing":
                    if current == 0:
                        progress.update(task, description="Preparing points...", total=total, completed=0)
                    else:
                        progress.update(task, description=f"Preparing ({current}/{total})", completed=current)
                elif event == "uploading":
                    if current == 0:
                        progress.update(task, description="Uploading to Qdrant...", total=total, completed=0)
                    else:
                        progress.update(task, description=f"Uploading ({current}/{total})", completed=current)

            # Run parallel indexing
            result = indexer.index_directory(
                path=path,
                patterns=pattern,
                chunker=chunker,
                batch_size=batch_size,
                exclude_patterns=exclude_patterns if exclude_patterns else None,
                on_progress=on_progress,
                workers=workers,
                embedding_batch_size=embedding_batch_size,
            )

            progress.update(task, description="Complete", completed=result["total_chunks"])

        elapsed = time.time() - start_time

        # Display summary
        if not quiet:
            summary = Table.grid(padding=(0, 2))
            summary.add_column()
            summary.add_column()
            summary.add_row("Files indexed:", f"[cyan]{result['total_files']}[/cyan]")
            summary.add_row("Chunks created:", f"[cyan]{result['total_chunks']}[/cyan]")
            if result["skipped_files"]:
                summary.add_row("Files skipped:", f"[dim]{result['skipped_files']}[/dim]")
            summary.add_row("Workers:", f"[cyan]{workers}[/cyan]")
            if gpu:
                gpu_status = "[green]Yes[/green]" if indexer.use_cuda and is_cuda_available() else "[yellow]No (fallback)[/yellow]"
                summary.add_row("GPU:", gpu_status)
            summary.add_row("Time elapsed:", f"[cyan]{elapsed:.2f}s[/cyan]")

            if result["failed_files"]:
                summary.add_row("Failed files:", f"[red]{len(result['failed_files'])}[/red]")

            console.print(Panel(summary, title="Indexing Complete", border_style="green"))

            if result["failed_files"]:
                console.print("\n[yellow]Failed files:[/yellow]")
                for failed in result["failed_files"]:
                    console.print(f"  [red]•[/red] {failed}")

    except Exception as e:
        display_error(str(e))
        raise typer.Exit(1)


@app.command("list-collections")
def list_collections(
    url: Annotated[str, typer.Option("--url", "-u", help="Qdrant server URL")] = "http://localhost:6333",
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress non-error output")] = False,
) -> None:
    """List all Qdrant collections with their point counts."""
    global _quiet
    _quiet = quiet

    try:
        client = QdrantClient(url=url)
        collections = client.get_collections().collections

        if not collections:
            if not quiet:
                console.print("[yellow]No collections found.[/yellow]")
            return

        table = Table(title="Qdrant Collections")
        table.add_column("Collection Name", style="cyan")
        table.add_column("Points", justify="right")
        table.add_column("Vector Size", justify="right")

        for col in collections:
            info = client.get_collection(col.name)
            vector_size = "-"
            if info.config.params.vectors:
                vectors = info.config.params.vectors
                if hasattr(vectors, "size"):
                    # Unnamed vector config
                    vector_size = str(vectors.size)
                elif isinstance(vectors, dict):
                    # Named vectors - get sizes from all vector configs
                    sizes = [str(v.size) for v in vectors.values() if hasattr(v, "size")]
                    vector_size = ", ".join(sizes) if sizes else "-"

            table.add_row(
                col.name,
                str(info.points_count),
                vector_size,
            )

        console.print(table)

    except Exception as e:
        display_error(f"Failed to connect to Qdrant: {e}")
        raise typer.Exit(1)


@app.command("delete-collection")
def delete_collection(
    collection: Annotated[str, typer.Argument(help="Collection name to delete")],
    url: Annotated[str, typer.Option("--url", "-u", help="Qdrant server URL")] = "http://localhost:6333",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress non-error output")] = False,
) -> None:
    """Delete a Qdrant collection."""
    global _quiet
    _quiet = quiet

    try:
        client = QdrantClient(url=url)

        if not client.collection_exists(collection):
            display_error(f"Collection '{collection}' does not exist.")
            raise typer.Exit(1)

        if not yes:
            typer.confirm(
                f"Are you sure you want to delete collection '{collection}'?",
                abort=True,
            )

        client.delete_collection(collection)
        display_success(f"Deleted collection '{collection}'")

    except typer.Abort:
        if not quiet:
            console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        display_error(str(e))
        raise typer.Exit(1)


@app.command("show-excludes")
def show_excludes() -> None:
    """Show default exclusion patterns."""
    console.print("[bold]Default exclusion patterns:[/bold]\n")
    for pattern in DEFAULT_EXCLUDE_PATTERNS:
        console.print(f"  • {pattern}")
    console.print("\n[dim]Use --no-default-excludes to disable these.[/dim]")


@app.command("list-models")
def list_models(
    search: Annotated[Optional[str], typer.Argument(help="Filter models by name (e.g., 'jina', 'multilingual')")] = None,
) -> None:
    """List available FastEmbed embedding models.

    Examples:
        # List all models
        qdrant-indexer list-models

        # Find Jina models
        qdrant-indexer list-models jina

        # Find multilingual models
        qdrant-indexer list-models multilingual
    """
    from fastembed import TextEmbedding

    models = TextEmbedding.list_supported_models()

    # Filter if search term provided
    if search:
        search_lower = search.lower()
        models = [m for m in models if search_lower in m["model"].lower()]

    if not models:
        console.print(f"[yellow]No models found matching '{search}'[/yellow]")
        return

    table = Table(title="FastEmbed Embedding Models")
    table.add_column("Model", style="cyan")
    table.add_column("Dimensions", justify="right")
    table.add_column("Description", style="dim")

    for model in sorted(models, key=lambda m: m["model"]):
        desc = model.get("description", "")
        # Truncate long descriptions
        if len(desc) > 50:
            desc = desc[:47] + "..."
        table.add_row(
            model["model"],
            str(model.get("dim", "?")),
            desc,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(models)} models[/dim]")
    console.print(f"[dim]Default: {DEFAULT_EMBEDDING_MODEL}[/dim]")


if __name__ == "__main__":
    app()
