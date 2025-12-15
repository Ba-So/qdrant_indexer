"""Tests for code-aware document loaders."""

from pathlib import Path

import pytest

from qdrant_indexer.code_loaders import (
    CodeLoader,
    PHPCodeLoader,
    PythonCodeLoader,
    RustCodeLoader,
)
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


class TestRustCodeLoader:
    """Tests for RustCodeLoader."""

    def test_load_rust_file(self, tmp_path: Path):
        """Verify Rust file is loaded correctly."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('fn main() {\n    println!("Hello");\n}')

        doc = loader.load(test_file)

        assert 'fn main()' in doc.content
        assert doc.metadata["is_code"] is True
        assert doc.metadata["extension"] == ".rs"

    def test_extract_rust_function(self, tmp_path: Path):
        """Verify Rust function is extracted."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('/// Greet a person.\npub fn greet(name: &str) -> String {\n    format!("Hello, {}", name)\n}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the function
        func_symbols = [s for s in symbols if s.symbol_type == "function"]
        assert len(func_symbols) >= 1

        greet_func = next((s for s in func_symbols if s.name == "greet"), None)
        assert greet_func is not None
        assert greet_func.symbol_type == "function"
        assert greet_func.docstring == "Greet a person."
        assert greet_func.language == "rust"
        assert greet_func.visibility == "pub"

    def test_extract_rust_struct(self, tmp_path: Path):
        """Verify Rust struct is extracted."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('/// A test struct.\n#[derive(Debug, Clone)]\npub struct MyStruct {\n    pub field: i32,\n}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the struct
        struct_symbols = [s for s in symbols if s.symbol_type == "struct"]
        assert len(struct_symbols) == 1
        assert struct_symbols[0].name == "MyStruct"
        assert struct_symbols[0].docstring == "A test struct."
        assert struct_symbols[0].visibility == "pub"
        assert "Debug" in struct_symbols[0].metadata["derives"]
        assert "Clone" in struct_symbols[0].metadata["derives"]

    def test_extract_rust_enum(self, tmp_path: Path):
        """Verify Rust enum is extracted."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('/// Status enum.\npub enum Status {\n    Ok,\n    Error(String),\n}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the enum
        enum_symbols = [s for s in symbols if s.symbol_type == "enum"]
        assert len(enum_symbols) == 1
        assert enum_symbols[0].name == "Status"
        assert enum_symbols[0].docstring == "Status enum."
        assert "Ok" in enum_symbols[0].metadata["variants"]
        assert "Error" in enum_symbols[0].metadata["variants"]

    def test_extract_rust_trait(self, tmp_path: Path):
        """Verify Rust trait is extracted."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('/// A handler trait.\npub trait Handler {\n    fn handle(&self);\n}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the trait
        trait_symbols = [s for s in symbols if s.symbol_type == "trait"]
        assert len(trait_symbols) == 1
        assert trait_symbols[0].name == "Handler"
        assert trait_symbols[0].docstring == "A handler trait."

        # Should also extract the trait method
        method_symbols = [s for s in symbols if s.symbol_type == "method"]
        assert len(method_symbols) >= 1
        handle_method = next((s for s in method_symbols if s.name == "handle"), None)
        assert handle_method is not None
        assert handle_method.parent == "Handler"

    def test_extract_rust_impl(self, tmp_path: Path):
        """Verify Rust impl block is extracted."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('struct Config {}\n\nimpl Config {\n    /// Create new config.\n    pub fn new() -> Self {\n        Config {}\n    }\n}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the impl block
        impl_symbols = [s for s in symbols if s.symbol_type == "impl"]
        assert len(impl_symbols) == 1
        assert impl_symbols[0].name == "Config"

        # Should also extract the impl method
        method_symbols = [s for s in symbols if s.symbol_type == "method"]
        assert len(method_symbols) >= 1
        new_method = next((s for s in method_symbols if s.name == "new"), None)
        assert new_method is not None
        assert new_method.parent == "Config"
        assert new_method.docstring == "Create new config."

    def test_extract_rust_impl_trait_for_type(self, tmp_path: Path):
        """Verify impl Trait for Type is extracted."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('trait Display {}\nstruct Point {}\n\nimpl Display for Point {}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the impl block with trait info
        impl_symbols = [s for s in symbols if s.symbol_type == "impl"]
        trait_impl = next((s for s in impl_symbols if "Display" in s.name), None)
        assert trait_impl is not None
        assert trait_impl.metadata["trait"] == "Display"
        assert trait_impl.metadata["self_type"] == "Point"

    def test_extract_rust_const(self, tmp_path: Path):
        """Verify Rust const is extracted."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('/// Maximum value.\npub const MAX_VALUE: usize = 100;')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the const
        const_symbols = [s for s in symbols if s.symbol_type == "constant"]
        assert len(const_symbols) == 1
        assert const_symbols[0].name == "MAX_VALUE"
        assert const_symbols[0].docstring == "Maximum value."
        assert const_symbols[0].metadata["const_type"] == "usize"

    def test_extract_rust_static(self, tmp_path: Path):
        """Verify Rust static is extracted."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('/// Global counter.\npub static mut COUNTER: i32 = 0;')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the static
        static_symbols = [s for s in symbols if s.symbol_type == "static"]
        assert len(static_symbols) == 1
        assert static_symbols[0].name == "COUNTER"
        assert static_symbols[0].docstring == "Global counter."
        assert static_symbols[0].metadata["is_mutable"] is True

    def test_extract_rust_type_alias(self, tmp_path: Path):
        """Verify Rust type alias is extracted."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('/// Result alias.\npub type Result<T> = std::result::Result<T, Error>;')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the type alias
        type_symbols = [s for s in symbols if s.symbol_type == "type_alias"]
        assert len(type_symbols) == 1
        assert type_symbols[0].name == "Result"

    def test_extract_rust_macro(self, tmp_path: Path):
        """Verify Rust macro_rules is extracted."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('/// A logging macro.\n#[macro_export]\nmacro_rules! log {\n    ($msg:expr) => { println!("{}", $msg) };\n}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        # Should extract the macro
        macro_symbols = [s for s in symbols if s.symbol_type == "macro"]
        assert len(macro_symbols) == 1
        assert macro_symbols[0].name == "log"
        assert macro_symbols[0].docstring == "A logging macro."
        assert macro_symbols[0].metadata["is_exported"] is True

    def test_extract_rust_async_function(self, tmp_path: Path):
        """Verify async function metadata is captured."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('pub async fn fetch(url: &str) -> String {\n    String::new()\n}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        func_symbols = [s for s in symbols if s.symbol_type == "function"]
        assert len(func_symbols) >= 1
        fetch_func = next((s for s in func_symbols if s.name == "fetch"), None)
        assert fetch_func is not None
        assert fetch_func.metadata["is_async"] is True

    def test_extract_rust_unsafe_function(self, tmp_path: Path):
        """Verify unsafe function metadata is captured."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('pub unsafe fn dangerous() {}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        func_symbols = [s for s in symbols if s.symbol_type == "function"]
        assert len(func_symbols) >= 1
        dangerous_func = next((s for s in func_symbols if s.name == "dangerous"), None)
        assert dangerous_func is not None
        assert dangerous_func.metadata["is_unsafe"] is True

    def test_extract_rust_generics(self, tmp_path: Path):
        """Verify generic parameters are captured."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('pub fn process<T: Clone>(data: T) -> T {\n    data\n}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        func_symbols = [s for s in symbols if s.symbol_type == "function"]
        process_func = next((s for s in func_symbols if s.name == "process"), None)
        assert process_func is not None
        assert "T: Clone" in process_func.metadata["generics"] or "T" in str(process_func.metadata["generics"])

    def test_extract_rust_lifetimes(self, tmp_path: Path):
        """Verify lifetime parameters are captured."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text("pub fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {\n    x\n}")

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        func_symbols = [s for s in symbols if s.symbol_type == "function"]
        longest_func = next((s for s in func_symbols if s.name == "longest"), None)
        assert longest_func is not None
        assert "'a" in longest_func.metadata["lifetimes"]

    def test_get_symbol_context(self):
        """Verify get_symbol_context formats Rust symbols correctly."""
        loader = RustCodeLoader()
        symbol = CodeSymbol(
            name="process",
            qualified_name="process",
            symbol_type="function",
            content="pub fn process(x: i32) -> i32 { x }",
            docstring="Process a value.",
            signature="pub fn process(x: i32) -> i32",
            line_start=1,
            line_end=1,
            parent=None,
            visibility="pub",
            language="rust",
        )

        context = loader.get_symbol_context(symbol)

        assert "process" in context
        assert "Process a value." in context

    def test_extract_block_doc_comment(self, tmp_path: Path):
        """Verify block doc comments are extracted."""
        loader = RustCodeLoader()
        test_file = tmp_path / "test.rs"
        test_file.write_text('/**\n * Block doc comment.\n */\npub fn documented() {}')

        doc = loader.load(test_file)
        symbols = doc.metadata["symbols"]

        func_symbols = [s for s in symbols if s.symbol_type == "function"]
        documented_func = next((s for s in func_symbols if s.name == "documented"), None)
        assert documented_func is not None
        assert "Block doc comment" in documented_func.docstring


class TestCodeLoaderIntegration:
    """Integration tests for code_loaders package."""

    def test_import_all_from_package(self):
        """Verify all exports can be imported from package."""
        from qdrant_indexer.code_loaders import (
            CodeLoader,
            PHPCodeLoader,
            PythonCodeLoader,
            RustCodeLoader,
        )

        assert CodeLoader is not None
        assert PythonCodeLoader is not None
        assert PHPCodeLoader is not None
        assert RustCodeLoader is not None

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

    def test_all_loaders_coexist(self, tmp_path: Path):
        """Verify all loaders can be used simultaneously."""
        py_loader = PythonCodeLoader()
        php_loader = PHPCodeLoader()
        rust_loader = RustCodeLoader()

        py_file = tmp_path / "test.py"
        py_file.write_text("def test(): pass")

        php_file = tmp_path / "test.php"
        php_file.write_text("<?php\nfunction test() {}")

        rust_file = tmp_path / "test.rs"
        rust_file.write_text("fn test() {}")

        py_doc = py_loader.load(py_file)
        php_doc = php_loader.load(php_file)
        rust_doc = rust_loader.load(rust_file)

        assert py_doc.metadata["is_code"] is True
        assert php_doc.metadata["is_code"] is True
        assert rust_doc.metadata["is_code"] is True
        assert py_doc.metadata["extension"] == ".py"
        assert php_doc.metadata["extension"] == ".php"
        assert rust_doc.metadata["extension"] == ".rs"
