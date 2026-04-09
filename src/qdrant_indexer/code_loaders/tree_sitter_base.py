"""Shared base class for tree-sitter-based code loaders."""

from abc import abstractmethod
from pathlib import Path

from tree_sitter import Parser

from qdrant_indexer.models import CodeSymbol

from .base import CodeLoader


class TreeSitterCodeLoader(CodeLoader):
    """Intermediate base for code loaders that use tree-sitter parsing.

    Provides utilities shared by all tree-sitter loaders:
      - extract_symbols: template method that encodes, parses, and walks the AST
      - _walk_node: abstract hook that subclasses implement for language-specific traversal
      - _get_node_text: extract raw text for any AST node
      - _clean_block_comment: normalize /** ... */ and /*! ... */ block doc comments

    Subclasses are responsible for initialising ``self._parser`` in their
    ``__init__`` before calling any tree-sitter operations.
    """

    _parser: Parser

    def extract_symbols(self, content: str, file_path: Path) -> list[CodeSymbol]:
        """Template: encode, parse, walk. Subclasses implement _walk_node."""
        content_bytes = content.encode("utf-8")
        tree = self._parser.parse(content_bytes)
        symbols: list[CodeSymbol] = []
        self._walk_node(tree.root_node, content_bytes, symbols, None)
        return symbols

    @abstractmethod
    def _walk_node(
        self,
        node,
        content_bytes: bytes,
        symbols: list[CodeSymbol],
        parent_name: str | None,
    ) -> None:
        """Recursively walk AST nodes; subclasses control recursion strategy."""
        ...

    def _get_node_text(self, node, content_bytes: bytes) -> str:
        """Get text content of a tree-sitter node.

        Args:
            node: Tree-sitter node, or None.
            content_bytes: Source code as bytes.

        Returns:
            Decoded text slice, or empty string when node is None.
        """
        if node is None:
            return ""
        return content_bytes[node.start_byte : node.end_byte].decode("utf-8")

    def _clean_block_comment(self, comment: str) -> str:
        """Clean a block doc comment by removing delimiters and leading asterisks.

        Handles the following opening delimiters:
          ``/**``  — PHPDoc and Rust block doc comments
          ``/*!``  — Rust inner block doc comments

        After stripping the ``/**`` / ``/*!`` prefix and the closing ``*/``,
        each remaining line has its leading ``*`` (and surrounding whitespace)
        removed so the caller receives clean prose text.

        Args:
            comment: Raw block comment string including delimiters.

        Returns:
            Multi-line string with comment markers stripped.
        """
        # Strip opening delimiter (/** or /*!)
        if comment.startswith("/**") or comment.startswith("/*!"):
            comment = comment[3:]
        # Strip closing delimiter
        if comment.endswith("*/"):
            comment = comment[:-2]

        lines = comment.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            # Remove the leading asterisk that block-comment style conventionally adds
            if line.startswith("*"):
                line = line[1:].strip()
            if line:
                cleaned.append(line)
        return "\n".join(cleaned)
