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
from qdrant_indexer.indexer import QdrantIndexer

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
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="Chunk size in characters")] = 512,
    chunk_overlap: Annotated[int, typer.Option("--chunk-overlap", help="Overlap between chunks")] = 50,
    pattern: Annotated[list[str], typer.Option("--pattern", "-p", help="Glob patterns for files (can be repeated)")] = ["**/*.md", "**/*.txt", "**/*.pdf", "**/*.rst", "**/*.py", "**/*.php"],
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size for uploads")] = 100,
    exclude: Annotated[Optional[list[str]], typer.Option("--exclude", "-e", help="Patterns to exclude (can be repeated)")] = None,
    no_default_excludes: Annotated[bool, typer.Option("--no-default-excludes", help="Don't use default exclusion patterns")] = False,
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

        # Index Python codebase
        qdrant-indexer index ./src -c my-code -p "**/*.py"

        # Index mixed docs and code
        qdrant-indexer index ./project -c full-index

        # Index only specific patterns
        qdrant-indexer index ./src -c api-docs -p "**/*.md" -p "**/*.py"
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
        # Initialize indexer (shows spinner during model loading)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=quiet,
        ) as progress:
            init_task = progress.add_task("Loading embedding model...", total=None)

            indexer = QdrantIndexer(
                qdrant_url=url,
                collection_name=collection,
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

        # Discover and filter files
        all_files = []
        seen = set()
        for p in pattern:
            for f in path.glob(p):
                if f.is_file() and f not in seen:
                    all_files.append(f)
                    seen.add(f)

        files, skipped = filter_files(
            all_files,
            path,
            exclude_patterns=exclude,
            use_defaults=not no_default_excludes,
        )

        if not files:
            if not quiet:
                patterns_str = ", ".join(pattern)
                console.print(f"[yellow]No files found matching patterns: {patterns_str}[/yellow]")
                if skipped:
                    console.print(f"[dim]({len(skipped)} files excluded by patterns)[/dim]")
            return

        if not quiet:
            msg = f"Found [cyan]{len(files)}[/cyan] files to index"
            if skipped:
                msg += f" [dim]({len(skipped)} excluded)[/dim]"
            console.print(msg)

        if verbose >= 2 and skipped:
            console.print("\n[dim]Excluded files:[/dim]")
            for f in skipped[:10]:  # Show first 10
                console.print(f"  [dim]• {f.relative_to(path)}[/dim]")
            if len(skipped) > 10:
                console.print(f"  [dim]... and {len(skipped) - 10} more[/dim]")
            console.print()

        # Progress bar for file indexing
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            disable=quiet,
        ) as progress:
            file_task = progress.add_task("Indexing files...", total=len(files))

            total_files = 0
            total_chunks = 0
            failed_files: list[str] = []

            for file_path in files:
                progress.update(file_task, description=f"[cyan]{file_path.name}[/cyan]")

                try:
                    chunks_count = indexer.index_file(file_path, chunker, batch_size)
                    total_files += 1
                    total_chunks += chunks_count

                    if verbose >= 1 and not quiet:
                        console.print(f"  [dim]{file_path.name}:[/dim] {chunks_count} chunks")

                except Exception as e:
                    failed_files.append(f"{file_path}: {e}")
                    if verbose >= 1:
                        console.print(f"  [red]Failed:[/red] {file_path.name}: {e}")

                progress.advance(file_task)

        elapsed = time.time() - start_time

        # Display summary
        if not quiet:
            summary = Table.grid(padding=(0, 2))
            summary.add_column()
            summary.add_column()
            summary.add_row("Files indexed:", f"[cyan]{total_files}[/cyan]")
            summary.add_row("Chunks created:", f"[cyan]{total_chunks}[/cyan]")
            if skipped:
                summary.add_row("Files skipped:", f"[dim]{len(skipped)}[/dim]")
            summary.add_row("Time elapsed:", f"[cyan]{elapsed:.2f}s[/cyan]")

            if failed_files:
                summary.add_row("Failed files:", f"[red]{len(failed_files)}[/red]")

            console.print(Panel(summary, title="Indexing Complete", border_style="green"))

            if failed_files:
                console.print("\n[yellow]Failed files:[/yellow]")
                for failed in failed_files:
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
                if hasattr(info.config.params.vectors, "size"):
                    vector_size = str(info.config.params.vectors.size)

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


if __name__ == "__main__":
    app()
