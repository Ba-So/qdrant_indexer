"""Python source code loader using AST parsing."""

import ast
import logging
from pathlib import Path

from qdrant_indexer.models import CodeSymbol

from .base import CodeLoader

logger = logging.getLogger(__name__)


class PythonCodeLoader(CodeLoader):
    """Loader for Python source files using AST parsing.

    Extracts functions, classes, methods, module docstrings, and constants
    from Python source code using the built-in ast module.
    """

    def extract_symbols(self, content: str, file_path: Path) -> list[CodeSymbol]:
        """Extract symbols using Python's ast module.

        Args:
            content: Python source code as string.
            file_path: Path to source file for error reporting.

        Returns:
            List of extracted CodeSymbol objects.
        """
        symbols: list[CodeSymbol] = []

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []

        # Extract module-level docstring
        module_doc = ast.get_docstring(tree)
        if module_doc:
            symbols.append(
                CodeSymbol(
                    name="__module__",
                    qualified_name=f"{file_path.stem}.__module__",
                    symbol_type="module",
                    content=module_doc,
                    language="python",
                    docstring=module_doc,
                    line_start=1,
                    line_end=1,
                )
            )

        # Track which functions are methods (to avoid duplicates)
        method_nodes: set[int] = set()

        # First pass: identify all methods inside classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_nodes.add(id(item))

        # Second pass: extract all symbols
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip if this is a method (will be handled with class)
                if id(node) not in method_nodes:
                    symbols.append(self._extract_function(node, content, None))

            elif isinstance(node, ast.ClassDef):
                symbols.append(self._extract_class(node, content))
                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(self._extract_function(item, content, node.name))

            elif isinstance(node, ast.Assign):
                # Extract module-level constants (NAME = value pattern)
                # Only at module level (parent is Module)
                if hasattr(node, "parent") or self._is_module_level(node, tree):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            symbols.append(self._extract_constant(node, target, content))

        return symbols

    def _is_module_level(self, node: ast.AST, tree: ast.Module) -> bool:
        """Check if a node is at module level."""
        return node in tree.body

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        content: str,
        parent_class: str | None,
    ) -> CodeSymbol:
        """Extract function or method symbol.

        Args:
            node: AST function definition node.
            content: Full source code.
            parent_class: Parent class name if this is a method.

        Returns:
            CodeSymbol representing the function/method.
        """
        source_lines = content.split("\n")
        end_lineno = node.end_lineno or node.lineno
        func_source = "\n".join(source_lines[node.lineno - 1 : end_lineno])
        docstring = ast.get_docstring(node)

        # Build signature
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        sig = f"({', '.join(args)})"
        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"

        qualified_name = f"{parent_class}.{node.name}" if parent_class else node.name
        symbol_type = "method" if parent_class else "function"

        # Check for async
        is_async = isinstance(node, ast.AsyncFunctionDef)

        return CodeSymbol(
            name=node.name,
            qualified_name=qualified_name,
            symbol_type=symbol_type,
            content=func_source,
            language="python",
            docstring=docstring,
            signature=sig,
            line_start=node.lineno,
            line_end=end_lineno,
            parent=parent_class,
            metadata={
                "decorators": [ast.unparse(d) for d in node.decorator_list],
                "is_async": is_async,
            },
        )

    def _extract_class(self, node: ast.ClassDef, content: str) -> CodeSymbol:
        """Extract class symbol.

        Args:
            node: AST class definition node.
            content: Full source code.

        Returns:
            CodeSymbol representing the class.
        """
        source_lines = content.split("\n")
        end_lineno = node.end_lineno or node.lineno
        class_source = "\n".join(source_lines[node.lineno - 1 : end_lineno])
        docstring = ast.get_docstring(node)

        bases = [ast.unparse(base) for base in node.bases]
        sig = f"({', '.join(bases)})" if bases else None

        return CodeSymbol(
            name=node.name,
            qualified_name=node.name,
            symbol_type="class",
            content=class_source,
            language="python",
            docstring=docstring,
            signature=sig,
            line_start=node.lineno,
            line_end=end_lineno,
            metadata={
                "bases": bases,
                "decorators": [ast.unparse(d) for d in node.decorator_list],
            },
        )

    def _extract_constant(
        self, node: ast.Assign, target: ast.Name, content: str
    ) -> CodeSymbol:
        """Extract constant assignment.

        Args:
            node: AST assignment node.
            target: The name target being assigned.
            content: Full source code.

        Returns:
            CodeSymbol representing the constant.
        """
        source_lines = content.split("\n")
        end_lineno = node.end_lineno or node.lineno
        const_source = "\n".join(source_lines[node.lineno - 1 : end_lineno])

        return CodeSymbol(
            name=target.id,
            qualified_name=target.id,
            symbol_type="constant",
            content=const_source,
            language="python",
            line_start=node.lineno,
            line_end=end_lineno,
        )

