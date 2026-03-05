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
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from qdrant_indexer.chunkers import CHUNKERS, get_chunker
from qdrant_indexer.filters import DEFAULT_EXCLUDE_PATTERNS, filter_files
from qdrant_indexer.indexer import (
    DEFAULT_CLIP_VISION_MODEL,
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
    collection: Annotated[
        str, typer.Option("--collection", "-c", help="Collection name")
    ],
    url: Annotated[
        str, typer.Option("--url", "-u", help="Qdrant server URL")
    ] = "http://localhost:6333",
    embedding_model: Annotated[
        str,
        typer.Option("--embedding-model", "-m", help="FastEmbed model for embeddings"),
    ] = DEFAULT_EMBEDDING_MODEL,
    chunk_size: Annotated[
        int, typer.Option("--chunk-size", help="Chunk size in characters")
    ] = 1536,
    chunk_overlap: Annotated[
        int, typer.Option("--chunk-overlap", help="Overlap between chunks")
    ] = 200,
    chunker_strategy: Annotated[
        str,
        typer.Option(
            "--chunker",
            help="Chunking strategy: auto (file-based), recursive, fixed, markdown, html, semantic, code",
        ),
    ] = "auto",
    pattern: Annotated[
        list[str],
        typer.Option(
            "--pattern", "-p", help="Glob patterns for files (can be repeated)"
        ),
    ] = [
        "**/*.md",
        "**/*.txt",
        "**/*.pdf",
        "**/*.rst",
        "**/*.py",
        "**/*.php",
        "**/*.html",
        "**/*.htm",
    ],
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Batch size for uploads")
    ] = 100,
    exclude: Annotated[
        Optional[list[str]],
        typer.Option("--exclude", "-e", help="Patterns to exclude (can be repeated)"),
    ] = None,
    no_default_excludes: Annotated[
        bool,
        typer.Option(
            "--no-default-excludes", help="Don't use default exclusion patterns"
        ),
    ] = False,
    workers: Annotated[
        int, typer.Option("--workers", "-w", help="Parallel workers for file loading")
    ] = DEFAULT_WORKERS,
    gpu: Annotated[
        bool,
        typer.Option(
            "--gpu", "--cuda", help="Enable GPU/CUDA acceleration for embeddings"
        ),
    ] = False,
    embedding_batch_size: Annotated[
        int,
        typer.Option(
            "--embedding-batch-size",
            help="Chunks to embed at once (lower = less GPU memory)",
        ),
    ] = DEFAULT_EMBEDDING_BATCH_SIZE,
    incremental: Annotated[
        bool,
        typer.Option(
            "--incremental/--full",
            help="Incremental update (skip unchanged) vs full re-index",
        ),
    ] = True,
    state_file: Annotated[
        Optional[Path],
        typer.Option(
            "--state-file",
            help="Custom state file location (default: .qdrant-index-state.json in directory)",
        ),
    ] = None,
    images: Annotated[
        bool,
        typer.Option(
            "--images/--no-images",
            help="Extract and embed images from PDFs using CLIP",
        ),
    ] = False,
    clip_model: Annotated[
        str,
        typer.Option(
            "--clip-model",
            help="CLIP model for image embeddings",
        ),
    ] = DEFAULT_CLIP_VISION_MODEL,
    min_image_size: Annotated[
        int,
        typer.Option(
            "--min-image-size",
            help="Minimum image dimension in pixels",
        ),
    ] = 100,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose", "-v", count=True, help="Increase verbosity (-v, -vv)"
        ),
    ] = 0,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress non-error output")
    ] = False,
) -> None:
    """Index a directory into a Qdrant collection.

    By default, uses incremental indexing: only new and modified files are processed,
    and deleted files are removed from the database. Use --full to re-index everything.

    By default, indexes documentation (.md, .txt, .pdf, .rst) and code files (.py, .php).
    Code files are parsed to extract symbols (functions, classes, methods) for better
    semantic search.

    Examples:
        # Incremental index (default - only process changes)
        qdrant-indexer index ./docs -c my-docs

        # Full re-index (process all files)
        qdrant-indexer index ./docs -c my-docs --full

        # Index with GPU acceleration (requires CUDA build)
        qdrant-indexer index ./docs -c my-docs --gpu

        # Index Python codebase
        qdrant-indexer index ./src -c my-code -p "**/*.py"

        # Index with a different embedding model (e.g., for German docs)
        qdrant-indexer index ./docs -c german-docs -m jinaai/jina-embeddings-v2-base-de

        # Index code with Jina v3 (multilingual + code support)
        qdrant-indexer index ./src -c my-code -m jinaai/jina-embeddings-v3

        # Custom state file location
        qdrant-indexer index ./docs -c my-docs --state-file ./my-state.json

        # Extract and index images from PDFs with CLIP
        qdrant-indexer index ./papers -c research --images

        # Use a different CLIP model for images
        qdrant-indexer index ./papers -c research --images --clip-model Qdrant/resnet50-onnx
    """
    setup_logging(verbose, quiet)

    if not path.exists():
        display_error(f"Path does not exist: {path}")
        raise typer.Exit(1)

    if not path.is_dir():
        display_error(f"Path is not a directory: {path}")
        raise typer.Exit(1)

    # Validate chunker strategy early (before connecting to Qdrant)
    if chunker_strategy != "auto" and chunker_strategy not in CHUNKERS:
        valid_strategies = ", ".join(sorted(CHUNKERS.keys()))
        display_error(
            f"Unknown chunker strategy '{chunker_strategy}'. "
            f"Valid strategies: auto, {valid_strategies}"
        )
        raise typer.Exit(1)

    start_time = time.time()

    try:
        # Check GPU availability if requested
        if gpu and not quiet:
            if is_cuda_available():
                console.print(
                    "[green]✓[/green] CUDA available - GPU acceleration enabled"
                )
            else:
                console.print(
                    "[yellow]⚠[/yellow] CUDA not available - falling back to CPU"
                )

        # Initialize indexer (shows spinner during model loading)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            disable=quiet,
        ) as progress:
            init_task = progress.add_task(
                f"Loading embedding model ({embedding_model})...", total=None
            )

            indexer = QdrantIndexer(
                qdrant_url=url,
                collection_name=collection,
                embedding_model=embedding_model,
                use_cuda=gpu,
                enable_image_embeddings=images,
                clip_vision_model=clip_model,
                min_image_size=min_image_size,
            )

            progress.update(init_task, description="Connecting to Qdrant...")

            created = indexer.ensure_collection()
            progress.remove_task(init_task)

        if not quiet:
            if created:
                console.print(f"Created collection: [cyan]{collection}[/cyan]")
            else:
                console.print(f"Using existing collection: [cyan]{collection}[/cyan]")

        # Initialize chunker based on strategy
        if chunker_strategy == "auto":
            # Signal to use per-file chunker selection in indexer
            chunker = None
        else:
            # Strategy already validated above
            chunker = get_chunker(
                chunker_strategy,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
            )

        # Build exclude patterns list
        exclude_patterns = list(exclude) if exclude else []
        if not no_default_excludes:
            exclude_patterns.extend(DEFAULT_EXCLUDE_PATTERNS)

        if incremental:
            # Incremental mode: use sync_directory
            if not quiet:
                mode_msg = (
                    "[cyan]incremental[/cyan]"
                    if not state_file
                    else f"[cyan]incremental[/cyan] (state: {state_file})"
                )
                console.print(f"Mode: {mode_msg}")
                console.print(f"Chunker: [cyan]{chunker_strategy}[/cyan]")
                if images:
                    console.print(f"Image embeddings: [cyan]enabled[/cyan] ({clip_model})")

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

                def on_sync_progress(
                    event: str, current: int, total: int, message: str
                ) -> None:
                    """Handle progress updates from sync_directory."""
                    if event == "sync_discovery":
                        progress.update(
                            task,
                            description=f"Found {total} files",
                            total=total,
                            completed=0,
                        )
                    elif event == "sync_checking":
                        progress.update(
                            task, description=f"Checking: {message}", completed=current
                        )
                    elif event == "sync_indexing":
                        progress.update(
                            task,
                            description=f"Indexing: {message}",
                            total=total,
                            completed=current,
                        )
                    elif event == "sync_deleting":
                        progress.update(
                            task,
                            description=f"Removing: {message}",
                            total=total,
                            completed=current,
                        )

                result = indexer.sync_directory(
                    path=path,
                    patterns=pattern,
                    chunker=chunker,
                    batch_size=batch_size,
                    exclude_patterns=exclude_patterns if exclude_patterns else None,
                    state_file=state_file,
                    force=False,
                    on_progress=on_sync_progress,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap,
                )

                progress.update(task, description="Complete")
        else:
            # Full mode: use index_directory
            if not quiet:
                console.print(f"Mode: [cyan]full re-index[/cyan]")
                console.print(f"Chunker: [cyan]{chunker_strategy}[/cyan]")
                console.print(f"Using [cyan]{workers}[/cyan] parallel workers")
                if images:
                    console.print(f"Image embeddings: [cyan]enabled[/cyan] ({clip_model})")

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

                def on_progress(
                    event: str, current: int, total: int, message: str
                ) -> None:
                    """Handle progress updates from indexer."""
                    if event == "discovery":
                        progress.update(
                            task,
                            description=f"Found {total} files",
                            total=total,
                            completed=0,
                        )
                    elif event == "loading":
                        progress.update(
                            task,
                            description="Loading files...",
                            total=total,
                            completed=current,
                        )
                    elif event == "file_loaded":
                        progress.update(
                            task,
                            description=f"Loading: {message.split(':')[0].replace('Loaded ', '')}",
                            completed=current,
                        )
                    elif event == "file_error":
                        progress.update(task, completed=current)
                    elif event == "embedding":
                        if current == 0:
                            progress.update(
                                task,
                                description=f"Embedding {total} chunks...",
                                total=total,
                                completed=0,
                            )
                        else:
                            progress.update(
                                task,
                                description=f"Embedding ({current}/{total})",
                                completed=current,
                            )
                    elif event == "preparing":
                        if current == 0:
                            progress.update(
                                task,
                                description="Preparing points...",
                                total=total,
                                completed=0,
                            )
                        else:
                            progress.update(
                                task,
                                description=f"Preparing ({current}/{total})",
                                completed=current,
                            )
                    elif event == "uploading":
                        if current == 0:
                            progress.update(
                                task,
                                description="Uploading to Qdrant...",
                                total=total,
                                completed=0,
                            )
                        else:
                            progress.update(
                                task,
                                description=f"Uploading ({current}/{total})",
                                completed=current,
                            )

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

                progress.update(
                    task, description="Complete", completed=result["total_chunks"]
                )

        elapsed = time.time() - start_time

        # Display summary
        if not quiet:
            summary = Table.grid(padding=(0, 2))
            summary.add_column()
            summary.add_column()

            if incremental:
                # Display sync-specific summary
                summary.add_row("Files added:", f"[green]{result.added}[/green]")
                summary.add_row("Files updated:", f"[yellow]{result.updated}[/yellow]")
                summary.add_row("Files deleted:", f"[red]{result.deleted}[/red]")
                summary.add_row("Files unchanged:", f"[dim]{result.unchanged}[/dim]")
                if result.failed:
                    summary.add_row("Files failed:", f"[red]{len(result.failed)}[/red]")
            else:
                # Display full index summary
                summary.add_row(
                    "Files indexed:", f"[cyan]{result['total_files']}[/cyan]"
                )
                summary.add_row(
                    "Chunks created:", f"[cyan]{result['total_chunks']}[/cyan]"
                )
                if result["skipped_files"]:
                    summary.add_row(
                        "Files skipped:", f"[dim]{result['skipped_files']}[/dim]"
                    )
                summary.add_row("Workers:", f"[cyan]{workers}[/cyan]")
                if result["failed_files"]:
                    summary.add_row(
                        "Failed files:", f"[red]{len(result['failed_files'])}[/red]"
                    )

            if gpu:
                gpu_status = (
                    "[green]Yes[/green]"
                    if indexer.use_cuda and is_cuda_available()
                    else "[yellow]No (fallback)[/yellow]"
                )
                summary.add_row("GPU:", gpu_status)
            summary.add_row("Time elapsed:", f"[cyan]{elapsed:.2f}s[/cyan]")

            console.print(
                Panel(summary, title="Indexing Complete", border_style="green")
            )

            # Display failed files
            if incremental and result.failed:
                console.print("\n[yellow]Failed files:[/yellow]")
                for failed in result.failed:
                    console.print(f"  [red]•[/red] {failed}")
            elif not incremental and result["failed_files"]:
                console.print("\n[yellow]Failed files:[/yellow]")
                for failed in result["failed_files"]:
                    console.print(f"  [red]•[/red] {failed}")

    except Exception as e:
        display_error(str(e))
        raise typer.Exit(1)


@app.command()
def status(
    path: Annotated[Path, typer.Argument(help="Directory to check status for")],
    state_file: Annotated[
        Optional[Path], typer.Option("--state-file", help="Custom state file location")
    ] = None,
    pattern: Annotated[
        list[str], typer.Option("--pattern", "-p", help="Glob patterns for files")
    ] = [
        "**/*.md",
        "**/*.txt",
        "**/*.pdf",
        "**/*.rst",
        "**/*.py",
        "**/*.php",
        "**/*.html",
        "**/*.htm",
    ],
    exclude: Annotated[
        Optional[list[str]], typer.Option("--exclude", "-e", help="Patterns to exclude")
    ] = None,
    no_default_excludes: Annotated[bool, typer.Option("--no-default-excludes")] = False,
) -> None:
    """Show indexing status for a directory.

    Displays:
    - Total indexed files
    - Files pending update (modified)
    - Files pending addition (new)
    - Files pending deletion (removed from disk)
    - Last index timestamp

    Examples:
        # Check status of indexed directory
        qdrant-indexer status ./docs

        # Check status with custom state file
        qdrant-indexer status ./docs --state-file ./my-state.json
    """
    if not path.exists() or not path.is_dir():
        display_error(f"Path does not exist or is not a directory: {path}")
        raise typer.Exit(1)

    if state_file is None:
        state_file = path / ".qdrant-index-state.json"

    if not state_file.exists():
        console.print(f"[yellow]No state file found at {state_file}[/yellow]")
        console.print("Run 'qdrant-indexer index' with --incremental to create state.")
        raise typer.Exit(0)

    # Load state
    from qdrant_indexer.state import IndexState, compute_file_hash

    state = IndexState(state_file)
    state.load()

    # Discover current files
    all_files = []
    seen = set()
    for p in pattern:
        for f in path.glob(p):
            if f.is_file() and f not in seen:
                all_files.append(f)
                seen.add(f)

    # Build exclude patterns list
    exclude_patterns = list(exclude) if exclude else []
    if not no_default_excludes:
        exclude_patterns.extend(DEFAULT_EXCLUDE_PATTERNS)

    files, _ = filter_files(
        all_files,
        path,
        exclude_patterns if exclude_patterns else None,
        use_defaults=False,
    )

    # Analyze status
    current_paths = {str(f.absolute()) for f in files}
    tracked_paths = state.get_all_paths()

    new_files = []
    modified_files = []
    unchanged_files = []
    deleted_files = list(tracked_paths - current_paths)

    for file_path in files:
        file_state = state.get_file_state(file_path)

        if file_state is None:
            new_files.append(file_path)
        else:
            content_hash = compute_file_hash(file_path)
            if file_state.content_hash != content_hash:
                modified_files.append(file_path)
            else:
                unchanged_files.append(file_path)

    # Display results
    table = Table(title="Indexing Status")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")

    table.add_row("Indexed (up to date)", f"[green]{len(unchanged_files)}[/green]")
    table.add_row("Pending addition", f"[yellow]{len(new_files)}[/yellow]")
    table.add_row("Pending update", f"[yellow]{len(modified_files)}[/yellow]")
    table.add_row("Pending deletion", f"[red]{len(deleted_files)}[/red]")
    table.add_row("Total tracked", str(len(tracked_paths)))

    console.print(table)

    # Show most recent index time
    if state.files:
        latest_time = max(s.indexed_at for s in state.files.values())
        console.print(f"\nLast indexed: [cyan]{latest_time}[/cyan]")

    # Show pending files if any
    if new_files:
        console.print("\n[yellow]New files (first 5):[/yellow]")
        for f in new_files[:5]:
            console.print(f"  • {f.relative_to(path)}")
        if len(new_files) > 5:
            console.print(f"  ... and {len(new_files) - 5} more")

    if modified_files:
        console.print("\n[yellow]Modified files (first 5):[/yellow]")
        for f in modified_files[:5]:
            console.print(f"  • {f.relative_to(path)}")
        if len(modified_files) > 5:
            console.print(f"  ... and {len(modified_files) - 5} more")


@app.command()
def clean(
    path: Annotated[Path, typer.Argument(help="Directory to clean")],
    collection: Annotated[
        str, typer.Option("--collection", "-c", help="Collection name")
    ],
    url: Annotated[
        str, typer.Option("--url", "-u", help="Qdrant server URL")
    ] = "http://localhost:6333",
    state_file: Annotated[
        Optional[Path], typer.Option("--state-file", help="Custom state file location")
    ] = None,
    all: Annotated[
        bool, typer.Option("--all", help="Remove all entries for this path (reset)")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress non-error output")
    ] = False,
) -> None:
    """Clean database entries for a directory.

    By default, removes entries for deleted files (files tracked in state but no longer on disk).
    Use --all to remove ALL entries and reset the state file completely.

    Examples:
        # Remove entries for deleted files
        qdrant-indexer clean ./docs -c my-docs

        # Remove all entries and reset (complete cleanup)
        qdrant-indexer clean ./docs -c my-docs --all

        # Skip confirmation prompt
        qdrant-indexer clean ./docs -c my-docs -y
    """
    global _quiet
    _quiet = quiet

    if not path.exists() or not path.is_dir():
        display_error(f"Path does not exist or is not a directory: {path}")
        raise typer.Exit(1)

    if state_file is None:
        state_file = path / ".qdrant-index-state.json"

    if not state_file.exists():
        display_error(f"No state file found at {state_file}")
        raise typer.Exit(1)

    from qdrant_indexer.state import IndexState

    state = IndexState(state_file)
    state.load()

    indexer = QdrantIndexer(url, collection)

    try:
        if all:
            # Remove ALL entries
            total_files = len(state.files)
            if total_files == 0:
                if not quiet:
                    console.print("[yellow]No entries to remove[/yellow]")
                return

            if not yes:
                typer.confirm(
                    f"This will remove ALL {total_files} file entries from '{collection}'. Continue?",
                    abort=True,
                )

            total_deleted = 0
            failed = []
            for file_path_str, file_state in state.files.items():
                try:
                    indexer.delete_points_by_ids(file_state.chunk_ids)
                    total_deleted += len(file_state.chunk_ids)
                except Exception as e:
                    failed.append(f"{file_path_str}: {e}")
                    if not quiet:
                        console.print(
                            f"[red]Error removing {Path(file_path_str).name}: {e}[/red]"
                        )

            # Remove state file
            state_file.unlink()

            if not quiet:
                if failed:
                    console.print(
                        f"\n[yellow]Failed to remove {len(failed)} file(s)[/yellow]"
                    )
                display_success(
                    f"Removed {total_deleted} points and deleted state file"
                )

        else:
            # Remove only deleted files
            deleted_files = []
            for file_path_str in state.get_all_paths():
                if not Path(file_path_str).exists():
                    deleted_files.append(file_path_str)

            if not deleted_files:
                if not quiet:
                    console.print("[green]No deleted files to clean[/green]")
                return

            if not yes:
                typer.confirm(
                    f"Remove {len(deleted_files)} deleted file entries from '{collection}'?",
                    abort=True,
                )

            total_deleted = 0
            failed = []
            for file_path_str in deleted_files:
                file_state = state.files.get(file_path_str)
                if file_state:
                    try:
                        indexer.delete_points_by_ids(file_state.chunk_ids)
                        state.remove_file(Path(file_path_str))
                        total_deleted += len(file_state.chunk_ids)
                    except Exception as e:
                        failed.append(f"{file_path_str}: {e}")
                        if not quiet:
                            console.print(
                                f"[red]Error removing {Path(file_path_str).name}: {e}[/red]"
                            )

            state.save()

            if not quiet:
                if failed:
                    console.print(
                        f"\n[yellow]Failed to remove {len(failed)} file(s)[/yellow]"
                    )
                display_success(
                    f"Removed {total_deleted} points from {len(deleted_files)} deleted files"
                )

    except typer.Abort:
        if not quiet:
            console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        display_error(str(e))
        raise typer.Exit(1)


@app.command("list-collections")
def list_collections(
    url: Annotated[
        str, typer.Option("--url", "-u", help="Qdrant server URL")
    ] = "http://localhost:6333",
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress non-error output")
    ] = False,
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
                    sizes = [
                        str(v.size) for v in vectors.values() if hasattr(v, "size")
                    ]
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
    url: Annotated[
        str, typer.Option("--url", "-u", help="Qdrant server URL")
    ] = "http://localhost:6333",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress non-error output")
    ] = False,
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
    search: Annotated[
        Optional[str],
        typer.Argument(help="Filter models by name (e.g., 'jina', 'multilingual')"),
    ] = None,
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


@app.command("list-clip-models")
def list_clip_models(
    search: Annotated[
        Optional[str],
        typer.Argument(help="Filter models by name (e.g., 'clip', 'resnet')"),
    ] = None,
) -> None:
    """List available FastEmbed CLIP/image embedding models.

    Examples:
        # List all CLIP models
        qdrant-indexer list-clip-models

        # Find CLIP models
        qdrant-indexer list-clip-models clip

        # Find ResNet models
        qdrant-indexer list-clip-models resnet
    """
    from fastembed import ImageEmbedding

    models = ImageEmbedding.list_supported_models()

    # Filter if search term provided
    if search:
        search_lower = search.lower()
        models = [m for m in models if search_lower in m["model"].lower()]

    if not models:
        console.print(f"[yellow]No models found matching '{search}'[/yellow]")
        return

    table = Table(title="FastEmbed CLIP/Image Models")
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
    console.print(f"[dim]Default: {DEFAULT_CLIP_VISION_MODEL}[/dim]")


if __name__ == "__main__":
    app()
