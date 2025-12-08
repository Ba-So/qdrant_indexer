"""Command-line interface for Qdrant Indexer."""

import typer

app = typer.Typer(help="Qdrant Indexer - Index documentation into Qdrant collections")


@app.command()
def index():
    """Index a directory into a Qdrant collection."""
    typer.echo("Not implemented yet")


if __name__ == "__main__":
    app()
