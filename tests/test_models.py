"""Tests for data models."""

import pytest

from qdrant_indexer.models import CodeSymbol, Chunk, Document


class TestCodeSymbol:
    """Tests for CodeSymbol model."""

    def test_python_function_symbol(self):
        """Verify Python function symbol with all fields."""
        symbol = CodeSymbol(
            name="parse_segment",
            qualified_name="parse_segment",
            symbol_type="function",
            content="def parse_segment(data: bytes) -> Segment:\n    return Segment(data)",
            language="python",
            docstring="Parse a segment from raw bytes.",
            signature="(data: bytes) -> Segment",
            line_start=10,
            line_end=15,
            metadata={"decorators": ["@staticmethod"]},
        )

        assert symbol.name == "parse_segment"
        assert symbol.qualified_name == "parse_segment"
        assert symbol.symbol_type == "function"
        assert "def parse_segment" in symbol.content
        assert symbol.docstring == "Parse a segment from raw bytes."
        assert symbol.signature == "(data: bytes) -> Segment"
        assert symbol.line_start == 10
        assert symbol.line_end == 15
        assert symbol.parent is None
        assert symbol.visibility is None
        assert symbol.language == "python"
        assert symbol.metadata["decorators"] == ["@staticmethod"]

    def test_php_method_symbol(self):
        """Verify PHP method symbol with visibility and parent class."""
        symbol = CodeSymbol(
            name="processData",
            qualified_name="MyClass.processData",
            symbol_type="method",
            content="public function processData($input) {\n    return $input;\n}",
            docstring="Process the input data.",
            signature="($input)",
            line_start=25,
            line_end=30,
            parent="MyClass",
            visibility="public",
            language="php",
        )

        assert symbol.name == "processData"
        assert symbol.qualified_name == "MyClass.processData"
        assert symbol.symbol_type == "method"
        assert "public function processData" in symbol.content
        assert symbol.docstring == "Process the input data."
        assert symbol.signature == "($input)"
        assert symbol.line_start == 25
        assert symbol.line_end == 30
        assert symbol.parent == "MyClass"
        assert symbol.visibility == "public"
        assert symbol.language == "php"

    def test_class_symbol(self):
        """Verify class type symbol."""
        symbol = CodeSymbol(
            name="MyClass",
            qualified_name="MyClass",
            symbol_type="class",
            content="class MyClass:\n    pass",
            language="python",
            docstring="A sample class.",
            line_start=1,
            line_end=2,
            metadata={"base_classes": ["BaseClass"]},
        )

        assert symbol.name == "MyClass"
        assert symbol.qualified_name == "MyClass"
        assert symbol.symbol_type == "class"
        assert symbol.signature is None
        assert symbol.parent is None
        assert symbol.metadata["base_classes"] == ["BaseClass"]

    def test_metadata_default_factory(self):
        """Verify metadata defaults to empty dict and is mutable per instance."""
        symbol1 = CodeSymbol(
            name="func1",
            qualified_name="func1",
            symbol_type="function",
            content="def func1(): pass",
            language="python",
            signature="()",
            line_start=1,
            line_end=1,
        )
        symbol2 = CodeSymbol(
            name="func2",
            qualified_name="func2",
            symbol_type="function",
            content="def func2(): pass",
            language="python",
            signature="()",
            line_start=2,
            line_end=2,
        )

        # Verify metadata defaults to empty dict
        assert symbol1.metadata == {}
        assert symbol2.metadata == {}

        # Verify they are separate instances (not shared)
        symbol1.metadata["test"] = "value1"
        assert symbol1.metadata["test"] == "value1"
        assert "test" not in symbol2.metadata

    def test_python_constant_symbol(self):
        """Verify Python constant symbol."""
        symbol = CodeSymbol(
            name="MAX_SIZE",
            qualified_name="MAX_SIZE",
            symbol_type="constant",
            content="MAX_SIZE = 1024",
            language="python",
            docstring="Maximum size in bytes.",
            line_start=5,
            line_end=5,
        )

        assert symbol.name == "MAX_SIZE"
        assert symbol.symbol_type == "constant"
        assert symbol.signature is None
        assert symbol.language == "python"

    def test_php_private_method(self):
        """Verify PHP private method with visibility."""
        symbol = CodeSymbol(
            name="helperMethod",
            qualified_name="Service.helperMethod",
            symbol_type="method",
            content="private function helperMethod() {}",
            language="php",
            signature="()",
            line_start=100,
            line_end=102,
            parent="Service",
            visibility="private",
        )

        assert symbol.visibility == "private"
        assert symbol.parent == "Service"
        assert symbol.language == "php"


class TestModelIntegration:
    """Tests for integration between different models."""

    def test_import_all_models(self):
        """Verify all models can be imported together."""
        from qdrant_indexer.models import Chunk, CodeSymbol, Document

        # Instantiate each to verify no conflicts
        doc = Document(content="test", source_path=None, metadata={})
        chunk = Chunk(text="test", index=0, total_chunks=1, metadata={})
        symbol = CodeSymbol(
            name="test",
            qualified_name="test",
            symbol_type="function",
            content="def test(): pass",
            language="python",
            signature="()",
            line_start=1,
            line_end=1,
        )

        assert isinstance(doc, Document)
        assert isinstance(chunk, Chunk)
        assert isinstance(symbol, CodeSymbol)
