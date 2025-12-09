"""Tests for code-aware document loaders."""

from pathlib import Path

import pytest

from qdrant_indexer.code_loaders import CodeLoader, PHPCodeLoader, PythonCodeLoader
from qdrant_indexer.models import CodeSymbol


class TestCodeLoaderBase:
    """Tests for CodeLoader abstract base class."""

    def test_code_loader_is_abstract(self):
        """Verify CodeLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CodeLoader()

    def test_code_loader_requires_extract_symbols(self):
        """Verify subclass must implement extract_symbols."""

        class IncompleteLoader(CodeLoader):
            def get_symbol_context(self, symbol: CodeSymbol) -> str:
                return ""

        with pytest.raises(TypeError):
            IncompleteLoader()

    def test_code_loader_requires_get_symbol_context(self):
        """Verify subclass must implement get_symbol_context."""

        class IncompleteLoader(CodeLoader):
            def extract_symbols(self, content: str, file_path: Path) -> list[CodeSymbol]:
                return []

        with pytest.raises(TypeError):
            IncompleteLoader()


class MinimalCodeLoader(CodeLoader):
    """Minimal concrete implementation for testing."""

    def extract_symbols(self, content: str, file_path: Path) -> list[CodeSymbol]:
        """Extract a dummy symbol."""
        return [
            CodeSymbol(
                name="test_func",
                qualified_name="test_func",
                symbol_type="function",
                content="def test_func(): pass",
                docstring="Test function",
                signature="()",
                line_start=1,
                line_end=1,
                parent=None,
                visibility=None,
                language="python",
            )
        ]

    def get_symbol_context(self, symbol: CodeSymbol) -> str:
        """Get symbol context."""
        return f"{symbol.name}{symbol.signature}: {symbol.docstring}"


class TestMinimalCodeLoader:
    """Tests for minimal concrete CodeLoader implementation."""

    def test_load_returns_document(self, tmp_path: Path):
        """Verify load() returns Document with correct structure."""
        loader = MinimalCodeLoader()
        test_file = tmp_path / "test.py"
        test_file.write_text("def test_func(): pass")

        doc = loader.load(test_file)

        assert doc.content == "def test_func(): pass"
        assert doc.source_path == test_file
        assert doc.metadata["filename"] == "test.py"
        assert doc.metadata["extension"] == ".py"

    def test_load_includes_symbols_metadata(self, tmp_path: Path):
        """Verify symbols are stored in metadata."""
        loader = MinimalCodeLoader()
        test_file = tmp_path / "test.py"
        test_file.write_text("def test_func(): pass")

        doc = loader.load(test_file)

        assert "symbols" in doc.metadata
        assert doc.metadata["is_code"] is True
        assert isinstance(doc.metadata["symbols"], list)
        assert len(doc.metadata["symbols"]) == 1
        assert doc.metadata["symbols"][0].name == "test_func"

    def test_load_includes_file_stats(self, tmp_path: Path):
        """Verify file stats are included in metadata."""
        loader = MinimalCodeLoader()
        test_file = tmp_path / "test.py"
        test_file.write_text("def test_func(): pass")

        doc = loader.load(test_file)

        assert "size" in doc.metadata
        assert "modified_time" in doc.metadata
        assert doc.metadata["size"] > 0


class TestPythonCodeLoader:
    """Tests for PythonCodeLoader."""

    def test_load_python_file(self, tmp_path: Path):
        """Verify Python file is loaded correctly."""
        loader = PythonCodeLoader()
        test_file = tmp_path / "test.py"
        test_file.write_text('"""Module docstring."""\n\ndef hello():\n    """Say hello."""\n    return "hello"')

        doc = loader.load(test_file)

        assert doc.content == '"""Module docstring."""\n\ndef hello():\n    """Say hello."""\n    return "hello"'
        assert doc.metadata["is_code"] is True
        assert doc.metadata["extension"] == ".py"

    def test_extract_python_function(self, tmp_path: Path):
        """Verify Python function is extracted."""
        loader = PythonCodeLoader()
        test_file = tmp_path / "test.py"
        test_file.write_text('def greet(name):\n    """Greet a person."""\n    return f"Hello, {name}"')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the function
        func_symbols = [s for s in symbols if s.symbol_type == "function"]
        assert len(func_symbols) >= 1

        greet_func = next((s for s in func_symbols if s.name == "greet"), None)
        assert greet_func is not None
        assert greet_func.symbol_type == "function"
        assert greet_func.docstring == "Greet a person."
        assert greet_func.language == "python"

    def test_extract_python_class(self, tmp_path: Path):
        """Verify Python class is extracted."""
        loader = PythonCodeLoader()
        test_file = tmp_path / "test.py"
        test_file.write_text('class MyClass:\n    """A test class."""\n    pass')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the class
        class_symbols = [s for s in symbols if s.symbol_type == "class"]
        assert len(class_symbols) == 1
        assert class_symbols[0].name == "MyClass"
        assert class_symbols[0].docstring == "A test class."

    def test_get_symbol_context(self):
        """Verify get_symbol_context formats symbols correctly."""
        loader = PythonCodeLoader()
        symbol = CodeSymbol(
            name="test_func",
            qualified_name="test_func",
            symbol_type="function",
            content="def test_func(x): pass",
            docstring="Test function.",
            signature="(x)",
            line_start=1,
            line_end=1,
            parent=None,
            visibility=None,
            language="python",
        )

        context = loader.get_symbol_context(symbol)

        assert "test_func" in context
        assert "(x)" in context
        assert "Test function." in context

    def test_handles_syntax_errors(self, tmp_path: Path):
        """Verify syntax errors are handled gracefully."""
        loader = PythonCodeLoader()
        test_file = tmp_path / "bad.py"
        test_file.write_text("def broken(:\n    pass")

        doc = loader.load(test_file)

        # Should return document with empty symbols list
        assert doc.metadata["symbols"] == []


class TestPHPCodeLoader:
    """Tests for PHPCodeLoader."""

    def test_load_php_file(self, tmp_path: Path):
        """Verify PHP file is loaded correctly."""
        loader = PHPCodeLoader()
        test_file = tmp_path / "test.php"
        test_file.write_text('<?php\nfunction hello() {\n    return "hello";\n}')

        doc = loader.load(test_file)

        assert '<?php' in doc.content
        assert doc.metadata["is_code"] is True
        assert doc.metadata["extension"] == ".php"

    def test_extract_php_function(self, tmp_path: Path):
        """Verify PHP function is extracted."""
        loader = PHPCodeLoader()
        test_file = tmp_path / "test.php"
        test_file.write_text('<?php\n/**\n * Greet a person.\n */\nfunction greet($name) {\n    return "Hello, " . $name;\n}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the function
        func_symbols = [s for s in symbols if s.symbol_type == "function"]
        assert len(func_symbols) >= 1

        greet_func = next((s for s in func_symbols if s.name == "greet"), None)
        assert greet_func is not None
        assert greet_func.symbol_type == "function"
        assert greet_func.language == "php"

    def test_extract_php_class(self, tmp_path: Path):
        """Verify PHP class is extracted."""
        loader = PHPCodeLoader()
        test_file = tmp_path / "test.php"
        test_file.write_text('<?php\n/**\n * A test class.\n */\nclass MyClass {\n}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the class
        class_symbols = [s for s in symbols if s.symbol_type == "class"]
        assert len(class_symbols) >= 1
        assert any(s.name == "MyClass" for s in class_symbols)

    def test_extract_php_method_with_visibility(self, tmp_path: Path):
        """Verify PHP method visibility is extracted."""
        loader = PHPCodeLoader()
        test_file = tmp_path / "test.php"
        test_file.write_text('<?php\nclass Test {\n    public function publicMethod() {}\n    private function privateMethod() {}\n}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract methods with visibility
        method_symbols = [s for s in symbols if s.symbol_type == "method"]
        assert len(method_symbols) >= 2

        public_method = next((s for s in method_symbols if s.name == "publicMethod"), None)
        private_method = next((s for s in method_symbols if s.name == "privateMethod"), None)

        if public_method:
            assert public_method.visibility == "public"
            assert public_method.parent == "Test"

        if private_method:
            assert private_method.visibility == "private"
            assert private_method.parent == "Test"

    def test_get_symbol_context(self):
        """Verify get_symbol_context formats PHP symbols correctly."""
        loader = PHPCodeLoader()
        symbol = CodeSymbol(
            name="testFunc",
            qualified_name="testFunc",
            symbol_type="function",
            content="function testFunc($x) {}",
            docstring="Test function.",
            signature="($x)",
            line_start=1,
            line_end=1,
            parent=None,
            visibility="public",
            language="php",
        )

        context = loader.get_symbol_context(symbol)

        assert "testFunc" in context
        assert "($x)" in context or "Test function." in context


class TestCodeLoaderIntegration:
    """Integration tests for code_loaders package."""

    def test_import_all_from_package(self):
        """Verify all exports can be imported from package."""
        from qdrant_indexer.code_loaders import CodeLoader, PHPCodeLoader, PythonCodeLoader

        assert CodeLoader is not None
        assert PythonCodeLoader is not None
        assert PHPCodeLoader is not None

    def test_python_and_php_loaders_coexist(self, tmp_path: Path):
        """Verify both loaders can be used simultaneously."""
        py_loader = PythonCodeLoader()
        php_loader = PHPCodeLoader()

        py_file = tmp_path / "test.py"
        py_file.write_text("def test(): pass")

        php_file = tmp_path / "test.php"
        php_file.write_text("<?php\nfunction test() {}")

        py_doc = py_loader.load(py_file)
        php_doc = php_loader.load(php_file)

        assert py_doc.metadata["is_code"] is True
        assert php_doc.metadata["is_code"] is True
        assert py_doc.metadata["extension"] == ".py"
        assert php_doc.metadata["extension"] == ".php"
