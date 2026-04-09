"""PHP source code loader using tree-sitter parsing."""

import logging
from collections.abc import Callable

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

    def _walk_node(
        self,
        node,
        content_bytes: bytes,
        symbols: list[CodeSymbol],
        parent_class: str | None,
    ) -> None:
        """Recursively walk AST nodes to extract symbols.

        Args:
            node: Tree-sitter node to process.
            content_bytes: Source code as bytes.
            symbols: List to append extracted symbols to.
            parent_class: Name of parent class/interface/trait if inside one.
        """
        if node.type == "function_definition":
            symbols.append(self._extract_function(node, content_bytes))

        elif node.type == "class_declaration":
            symbols.extend(
                self._extract_container_with_methods(
                    node, content_bytes, self._extract_class
                )
            )

        elif node.type == "interface_declaration":
            symbols.extend(
                self._extract_container_with_methods(
                    node,
                    content_bytes,
                    lambda n, cb: self._extract_interface_trait(n, cb, "interface"),
                )
            )

        elif node.type == "trait_declaration":
            symbols.extend(
                self._extract_container_with_methods(
                    node,
                    content_bytes,
                    lambda n, cb: self._extract_interface_trait(n, cb, "trait"),
                )
            )

        elif node.type == "const_declaration" and parent_class is None:
            # Module-level constants only
            symbols.append(self._extract_constant(node, content_bytes))

        else:
            for child in node.children:
                self._walk_node(child, content_bytes, symbols, parent_class)

    def _extract_container_with_methods(
        self,
        node,
        content_bytes: bytes,
        extract_container: Callable,
    ) -> list[CodeSymbol]:
        """Extract a container symbol (class/interface/trait) and its methods.

        Args:
            node: The container AST node.
            content_bytes: Source code as bytes.
            extract_container: Callable(node, content_bytes) -> CodeSymbol that
                produces the container symbol.

        Returns:
            List with the container symbol followed by all method symbols found
            in its body.
        """
        container = extract_container(node, content_bytes)
        result: list[CodeSymbol] = [container]

        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                if child.type == "method_declaration":
                    result.append(
                        self._extract_method(child, content_bytes, container.name)
                    )

        return result

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

    def _extract_function(self, node, content_bytes: bytes) -> CodeSymbol:
        """Extract function symbol.

        Args:
            node: Function definition node.
            content_bytes: Source code as bytes.

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

    def _extract_class(self, node, content_bytes: bytes) -> CodeSymbol:
        """Extract class symbol.

        Args:
            node: Class declaration node.
            content_bytes: Source code as bytes.

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
        self, node, content_bytes: bytes, parent_class: str
    ) -> CodeSymbol:
        """Extract method symbol.

        Args:
            node: Method declaration node.
            content_bytes: Source code as bytes.
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

    def _extract_constant(self, node, content_bytes: bytes) -> CodeSymbol:
        """Extract constant symbol.

        Args:
            node: Const declaration node.
            content_bytes: Source code as bytes.

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
        self, node, content_bytes: bytes, symbol_type: str
    ) -> CodeSymbol:
        """Extract interface or trait symbol.

        Args:
            node: Interface or trait declaration node.
            content_bytes: Source code as bytes.
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

