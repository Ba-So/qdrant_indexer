"""PHP source code loader using tree-sitter parsing."""

import logging
from pathlib import Path

import tree_sitter_php
from tree_sitter import Language, Parser

from qdrant_indexer.models import CodeSymbol

from .tree_sitter_base import TreeSitterCodeLoader

logger = logging.getLogger(__name__)


class PHPCodeLoader(TreeSitterCodeLoader):
    """Loader for PHP source files using tree-sitter.

    Extracts functions, classes, methods, interfaces, traits, and constants
    from PHP source code. Also extracts PHPDoc comments.
    """

    def __init__(self) -> None:
        """Initialize the PHP parser with tree-sitter."""
        self._php_lang = Language(tree_sitter_php.language_php())
        self._parser = Parser(self._php_lang)

    def extract_symbols(self, content: str, file_path: Path) -> list[CodeSymbol]:
        """Extract symbols using tree-sitter PHP parser.

        Args:
            content: PHP source code as string.
            file_path: Path to source file for error reporting.

        Returns:
            List of extracted CodeSymbol objects.
        """
        content_bytes = content.encode("utf-8")
        tree = self._parser.parse(content_bytes)
        symbols: list[CodeSymbol] = []

        # Walk the tree to find symbols
        self._walk_node(tree.root_node, content_bytes, content, symbols, None)

        return symbols

    def _walk_node(
        self,
        node,
        content_bytes: bytes,
        content: str,
        symbols: list[CodeSymbol],
        parent_class: str | None,
    ) -> None:
        """Recursively walk AST nodes to extract symbols.

        Args:
            node: Tree-sitter node to process.
            content_bytes: Source code as bytes.
            content: Source code as string.
            symbols: List to append extracted symbols to.
            parent_class: Name of parent class/interface/trait if inside one.
        """
        if node.type == "function_definition":
            symbols.append(self._extract_function(node, content_bytes, content))

        elif node.type == "class_declaration":
            class_symbol = self._extract_class(node, content_bytes, content)
            symbols.append(class_symbol)
            # Process methods inside class
            decl_list = node.child_by_field_name("body")
            if decl_list:
                for child in decl_list.children:
                    if child.type == "method_declaration":
                        symbols.append(
                            self._extract_method(
                                child, content_bytes, content, class_symbol.name
                            )
                        )

        elif node.type == "interface_declaration":
            iface_symbol = self._extract_interface_trait(
                node, content_bytes, content, "interface"
            )
            symbols.append(iface_symbol)
            # Process method signatures inside interface
            decl_list = node.child_by_field_name("body")
            if decl_list:
                for child in decl_list.children:
                    if child.type == "method_declaration":
                        symbols.append(
                            self._extract_method(
                                child, content_bytes, content, iface_symbol.name
                            )
                        )

        elif node.type == "trait_declaration":
            trait_symbol = self._extract_interface_trait(
                node, content_bytes, content, "trait"
            )
            symbols.append(trait_symbol)
            # Process methods inside trait
            decl_list = node.child_by_field_name("body")
            if decl_list:
                for child in decl_list.children:
                    if child.type == "method_declaration":
                        symbols.append(
                            self._extract_method(
                                child, content_bytes, content, trait_symbol.name
                            )
                        )

        elif node.type == "const_declaration" and parent_class is None:
            # Module-level constants only
            symbols.append(self._extract_constant(node, content_bytes, content))

        else:
            # Recurse into children (but not into class/interface/trait bodies
            # which we handle specially above)
            if node.type not in (
                "class_declaration",
                "interface_declaration",
                "trait_declaration",
            ):
                for child in node.children:
                    self._walk_node(
                        child, content_bytes, content, symbols, parent_class
                    )

    def _extract_phpdoc(self, node, content_bytes: bytes) -> str | None:
        """Extract PHPDoc comment above a node.

        Args:
            node: The node to find PHPDoc for.
            content_bytes: Source code as bytes.

        Returns:
            Cleaned PHPDoc content or None if not found.
        """
        prev = node.prev_sibling
        while prev:
            if prev.type == "comment":
                comment_text = content_bytes[prev.start_byte : prev.end_byte].decode(
                    "utf-8"
                )
                if comment_text.startswith("/**"):
                    return self._clean_block_comment(comment_text)
            elif prev.type not in ("text", "php_tag"):
                # Stop if we hit something that's not whitespace/comment
                break
            prev = prev.prev_sibling
        return None

    def _extract_function(
        self, node, content_bytes: bytes, content: str
    ) -> CodeSymbol:
        """Extract function symbol.

        Args:
            node: Function definition node.
            content_bytes: Source code as bytes.
            content: Source code as string.

        Returns:
            CodeSymbol representing the function.
        """
        name_node = node.child_by_field_name("name")
        name = self._get_node_text(name_node, content_bytes) or "unknown"

        source = self._get_node_text(node, content_bytes)
        phpdoc = self._extract_phpdoc(node, content_bytes)

        # Extract parameters
        params_node = node.child_by_field_name("parameters")
        params = self._get_node_text(params_node, content_bytes) or "()"

        # Extract return type
        return_type_node = node.child_by_field_name("return_type")
        signature = params
        if return_type_node:
            return_type = self._get_node_text(return_type_node, content_bytes)
            signature += f": {return_type}"

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type="function",
            content=source,
            language="php",
            docstring=phpdoc,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
        )

    def _extract_class(self, node, content_bytes: bytes, content: str) -> CodeSymbol:
        """Extract class symbol.

        Args:
            node: Class declaration node.
            content_bytes: Source code as bytes.
            content: Source code as string.

        Returns:
            CodeSymbol representing the class.
        """
        name_node = node.child_by_field_name("name")
        name = self._get_node_text(name_node, content_bytes) or "unknown"

        source = self._get_node_text(node, content_bytes)
        phpdoc = self._extract_phpdoc(node, content_bytes)

        # Extract extends/implements
        extends = []
        implements = []
        for child in node.children:
            if child.type == "base_clause":
                extends.append(self._get_node_text(child, content_bytes))
            elif child.type == "class_interface_clause":
                implements.append(self._get_node_text(child, content_bytes))

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type="class",
            content=source,
            language="php",
            docstring=phpdoc,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            metadata={"extends": extends, "implements": implements},
        )

    def _extract_method(
        self, node, content_bytes: bytes, content: str, parent_class: str
    ) -> CodeSymbol:
        """Extract method symbol.

        Args:
            node: Method declaration node.
            content_bytes: Source code as bytes.
            content: Source code as string.
            parent_class: Name of containing class/interface/trait.

        Returns:
            CodeSymbol representing the method.
        """
        name_node = node.child_by_field_name("name")
        name = self._get_node_text(name_node, content_bytes) or "unknown"

        source = self._get_node_text(node, content_bytes)
        phpdoc = self._extract_phpdoc(node, content_bytes)

        # Extract visibility
        visibility = "public"  # PHP default
        for child in node.children:
            if child.type in ("public", "private", "protected"):
                visibility = child.type
                break
            elif child.type == "visibility_modifier":
                visibility = self._get_node_text(child, content_bytes)
                break

        # Extract parameters
        params_node = node.child_by_field_name("parameters")
        params = self._get_node_text(params_node, content_bytes) or "()"

        # Extract return type
        return_type_node = node.child_by_field_name("return_type")
        signature = params
        if return_type_node:
            return_type = self._get_node_text(return_type_node, content_bytes)
            signature += f": {return_type}"

        qualified_name = f"{parent_class}.{name}"

        return CodeSymbol(
            name=name,
            qualified_name=qualified_name,
            symbol_type="method",
            content=source,
            language="php",
            docstring=phpdoc,
            signature=signature,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            parent=parent_class,
            visibility=visibility,
        )

    def _extract_constant(
        self, node, content_bytes: bytes, content: str
    ) -> CodeSymbol:
        """Extract constant symbol.

        Args:
            node: Const declaration node.
            content_bytes: Source code as bytes.
            content: Source code as string.

        Returns:
            CodeSymbol representing the constant.
        """
        source = self._get_node_text(node, content_bytes)

        # Extract name from const_element
        name = "CONSTANT"
        for child in node.children:
            if child.type == "const_element":
                # Name is a direct child with type "name"
                for subchild in child.children:
                    if subchild.type == "name":
                        name = self._get_node_text(subchild, content_bytes)
                        break
                break

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type="constant",
            content=source,
            language="php",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
        )

    def _extract_interface_trait(
        self, node, content_bytes: bytes, content: str, symbol_type: str
    ) -> CodeSymbol:
        """Extract interface or trait symbol.

        Args:
            node: Interface or trait declaration node.
            content_bytes: Source code as bytes.
            content: Source code as string.
            symbol_type: Either 'interface' or 'trait'.

        Returns:
            CodeSymbol representing the interface/trait.
        """
        name_node = node.child_by_field_name("name")
        name = self._get_node_text(name_node, content_bytes) or "unknown"

        source = self._get_node_text(node, content_bytes)
        phpdoc = self._extract_phpdoc(node, content_bytes)

        return CodeSymbol(
            name=name,
            qualified_name=name,
            symbol_type=symbol_type,
            content=source,
            language="php",
            docstring=phpdoc,
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
        )

