"""Base class for code-aware document loaders."""

from abc import abstractmethod
from pathlib import Path

from qdrant_indexer.loaders import DocumentLoader
from qdrant_indexer.models import CodeSymbol, Document


class CodeLoader(DocumentLoader):
    """Abstract base class for code-aware document loaders.

    Code loaders extract structured symbols (functions, classes, methods) from
    source code files, storing them in the document metadata for code-aware
    chunking and indexing.

    Subclasses must implement:
        - extract_symbols: Parse source code and extract CodeSymbol objects
        - get_symbol_context: Format a symbol for embedding/search
    """

    @abstractmethod
    def extract_symbols(self, content: str, file_path: Path) -> list[CodeSymbol]:
        """Extract code symbols from source content.

        Args:
            content: Source code as string.
            file_path: Path to source file (for error reporting).

        Returns:
            List of extracted CodeSymbol objects.
        """
        pass

    @abstractmethod
    def get_symbol_context(self, symbol: CodeSymbol) -> str:
        """Get searchable context for a symbol.

        Formats the symbol information (name, signature, docstring) into a
        string suitable for embedding and semantic search.

        Args:
            symbol: The code symbol to format.

        Returns:
            Formatted context string for embedding.
        """
        pass

    def load(self, path: Path) -> Document:
        """Load source file and extract symbols.

        Reads the file content, parses it to extract code symbols, and
        returns a Document with the symbols stored in metadata.

        Args:
            path: Path to the source file.

        Returns:
            Document with content and symbol metadata.
        """
        content = path.read_text(encoding="utf-8")
        symbols = self.extract_symbols(content, path)
        stat = path.stat()

        return Document(
            content=content,
            source_path=path,
            metadata={
                "filename": path.name,
                "extension": path.suffix,
                "size": stat.st_size,
                "modified_time": stat.st_mtime,
                "symbols": symbols,
                "is_code": True,
            },
        )
