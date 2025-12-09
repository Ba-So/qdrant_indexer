"""Profile indexing performance to identify bottlenecks.

Run with: python tests/profile_indexing.py
"""

import cProfile
import pstats
import tempfile
from pathlib import Path

from qdrant_indexer.chunkers import RecursiveChunker
from qdrant_indexer.code_loaders import PythonCodeLoader
from qdrant_indexer.loaders import LOADERS
from qdrant_indexer.models import CodeSymbol

# Register code loader
LOADERS[".py"] = PythonCodeLoader


def generate_test_code(num_lines: int = 500) -> str:
    """Generate Python code for profiling."""
    lines = ['"""Test module."""', '']
    num_funcs = num_lines // 20

    for i in range(num_funcs):
        lines.append(f'def func_{i}(x, y):')
        lines.append(f'    """Function {i}."""')
        lines.append('    return x + y')
        lines.append('')

    return '\n'.join(lines)


def profile_symbol_extraction():
    """Profile the symbol extraction process."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test files
        for i in range(10):
            file = tmp_path / f"module_{i}.py"
            file.write_text(generate_test_code(500))

        # Profile loading and extraction
        profiler = cProfile.Profile()
        profiler.enable()

        loader = PythonCodeLoader()
        for file in tmp_path.glob("*.py"):
            doc = loader.load(file)
            symbols = doc.metadata.get('symbols', [])

        profiler.disable()

        # Print results
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        print("\n=== Symbol Extraction Profile ===")
        stats.print_stats(20)


def profile_chunking():
    """Profile the chunking process."""
    # Generate sample symbols
    symbols = []
    for i in range(100):
        symbols.append(
            CodeSymbol(
                name=f"func_{i}",
                qualified_name=f"func_{i}",
                symbol_type="function",
                content=f"def func_{i}(): pass",
                docstring=f"Function {i} docstring with some text.",
                signature="()",
                line_start=i * 10,
                line_end=i * 10 + 5,
                parent=None,
                visibility=None,
                language="python",
            )
        )

    profiler = cProfile.Profile()
    profiler.enable()

    chunker = RecursiveChunker(chunk_size=512, overlap=50)
    for symbol in symbols:
        text = f"{symbol.symbol_type}: {symbol.qualified_name}\n{symbol.docstring or ''}"
        chunks = chunker.chunk(text)

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    print("\n=== Chunking Profile ===")
    stats.print_stats(20)


if __name__ == "__main__":
    print("Profiling indexing operations...")
    print("\nThis will identify performance bottlenecks in:")
    print("  1. Symbol extraction from source code")
    print("  2. Text chunking")
    print("\n")

    profile_symbol_extraction()
    profile_chunking()

    print("\n=== Optimization Recommendations ===")
    print("1. Symbol Extraction:")
    print("   - Cache tree-sitter parser instances")
    print("   - Batch process multiple files in parallel")
    print("2. Chunking:")
    print("   - Current chunker is already efficient")
    print("   - Consider symbol-aware chunking for better boundaries")
    print("3. Embedding & Upload:")
    print("   - Already batched in indexer.py")
    print("   - Batch size can be tuned (default: 100)")
