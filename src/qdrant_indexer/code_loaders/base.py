"""Base class for code-aware document loaders."""

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

from qdrant_indexer.loaders import DocumentLoader
from qdrant_indexer.models import CodeSymbol, Document


class CodeLoader(DocumentLoader):
    """Abstract base class for code-aware document loaders.

    Code loaders extract structured symbols (functions, classes, methods) from
    source code files, storing them in the document metadata for code-aware
    chunking and indexing.

    Subclasses must implement:
        - extract_symbols: Parse source code and extract CodeSymbol objects
    """

    preferred_chunker: ClassVar[str] = "code"

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

    def get_symbol_context(self, symbol: CodeSymbol) -> str:
        """Get searchable context for a symbol.

        Formats the symbol into a string suitable for embedding and semantic
        search.  A visibility prefix is included when the symbol carries an
        explicit, non-private visibility modifier (e.g. ``public`` in PHP,
        ``pub`` in Rust).  Python symbols never set ``visibility``, so they
        always use the bare ``type: name`` form.

        Args:
            symbol: The code symbol to format.

        Returns:
            Formatted context string for embedding.
        """
        parts = []

        if symbol.visibility:
            parts.append(
                f"{symbol.visibility} {symbol.symbol_type}: {symbol.qualified_name}"
            )
        else:
            parts.append(f"{symbol.symbol_type}: {symbol.qualified_name}")

        if symbol.signature:
            parts.append(symbol.signature)

        if symbol.docstring:
            parts.append(f"\n{symbol.docstring}")

        return "\n".join(parts)

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
