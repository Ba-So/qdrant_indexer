"""Command-line interface for Qdrant Indexer."""

import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from qdrant_client import QdrantClient
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from qdrant_indexer.chunkers import RecursiveChunker
from qdrant_indexer.indexer import QdrantIndexer

app = typer.Typer(help="Qdrant Indexer - Index documentation into Qdrant collections")
console = Console()


def display_error(message: str) -> None:
    """Display an error message with formatting."""
    console.print(f"[red]Error:[/red] {message}")


def display_success(message: str) -> None:
    """Display a success message with formatting."""
    console.print(f"[green]✓[/green] {message}")


@app.command()
def index(
    path: Annotated[Path, typer.Argument(help="Directory to index")],
    collection: Annotated[str, typer.Option("--collection", "-c", help="Collection name")],
    url: Annotated[str, typer.Option("--url", "-u", help="Qdrant server URL")] = "http://localhost:6333",
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="Chunk size in characters")] = 512,
    chunk_overlap: Annotated[int, typer.Option("--chunk-overlap", help="Overlap between chunks")] = 50,
    pattern: Annotated[str, typer.Option("--pattern", "-p", help="Glob pattern for files")] = "**/*.md",
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size for uploads")] = 100,
) -> None:
    """Index a directory into a Qdrant collection."""
    if not path.exists():
        display_error(f"Path does not exist: {path}")
        raise typer.Exit(1)

    if not path.is_dir():
        display_error(f"Path is not a directory: {path}")
        raise typer.Exit(1)

    start_time = time.time()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Initializing indexer...", total=None)

            indexer = QdrantIndexer(
                qdrant_url=url,
                collection_name=collection,
            )

            created = indexer.ensure_collection()
            if created:
                console.print(f"Created collection: [cyan]{collection}[/cyan]")
            else:
                console.print(f"Using existing collection: [cyan]{collection}[/cyan]")

            chunker = RecursiveChunker(chunk_size=chunk_size, overlap=chunk_overlap)

            progress.add_task(f"Indexing files matching {pattern}...", total=None)

            result = indexer.index_directory(
                path=path,
                pattern=pattern,
                chunker=chunker,
                batch_size=batch_size,
            )

        elapsed = time.time() - start_time

        # Display summary
        summary = Table.grid(padding=(0, 2))
        summary.add_column()
        summary.add_column()
        summary.add_row("Files indexed:", f"[cyan]{result['total_files']}[/cyan]")
        summary.add_row("Chunks created:", f"[cyan]{result['total_chunks']}[/cyan]")
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
) -> None:
    """List all Qdrant collections with their point counts."""
    try:
        client = QdrantClient(url=url)
        collections = client.get_collections().collections

        if not collections:
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
) -> None:
    """Delete a Qdrant collection."""
    try:
        client = QdrantClient(url=url)

        if not client.collection_exists(collection):
            display_error(f"Collection '{collection}' does not exist.")
            raise typer.Exit(1)

        if not yes:
            confirm = typer.confirm(
                f"Are you sure you want to delete collection '{collection}'?",
                abort=True,
            )

        client.delete_collection(collection)
        display_success(f"Deleted collection '{collection}'")

    except typer.Abort:
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        display_error(str(e))
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
