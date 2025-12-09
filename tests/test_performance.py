"""Performance tests for code indexing.

These tests verify that indexing performance meets success criteria.
Run with: pytest -m performance

Note: These tests require a running Qdrant instance and may take 30-60 seconds each.
"""

import time
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from qdrant_indexer.chunkers import RecursiveChunker
from qdrant_indexer.indexer import QdrantIndexer

# Default Qdrant URL for tests
QDRANT_URL = "http://localhost:6333"


def qdrant_available() -> bool:
    """Check if Qdrant is available at the default URL."""
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=2)
        client.get_collections()
        return True
    except Exception:
        return False


# Skip all tests in this module if Qdrant is not available
pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(
        not qdrant_available(),
        reason="Qdrant not available at localhost:6333. Start with: docker compose up -d",
    ),
]


@pytest.fixture
def perf_test_collection() -> str:
    """Generate a unique collection name for performance test isolation."""
    return f"perf_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def qdrant_client() -> QdrantClient:
    """Create a Qdrant client for cleanup."""
    return QdrantClient(url=QDRANT_URL)


@pytest.fixture
def cleanup_perf_collection(qdrant_client: QdrantClient, perf_test_collection: str):
    """Fixture to clean up test collection after test."""
    yield perf_test_collection
    # Cleanup after test
    try:
        qdrant_client.delete_collection(perf_test_collection)
    except Exception:
        pass  # Collection may not exist


def generate_python_file(num_lines: int) -> str:
    """Generate syntactically valid Python code with specified number of lines.

    Args:
        num_lines: Target number of lines to generate.

    Returns:
        Python source code string.
    """
    code = ['"""Test module for performance testing."""', '']

    # Calculate number of functions to generate (~25 lines per function)
    num_functions = max(1, num_lines // 25)

    for i in range(num_functions):
        code.append(f'def function_{i}(x: int, y: int) -> int:')
        code.append(f'    """Function {i} for testing.')
        code.append('    ')
        code.append('    Performs a computation on two integers.')
        code.append('    ')
        code.append('    Args:')
        code.append('        x: First integer.')
        code.append('        y: Second integer.')
        code.append('    ')
        code.append('    Returns:')
        code.append('        The computed result.')
        code.append('    """')
        code.append('    # Initialize result')
        code.append('    result = 0')
        code.append('    ')
        code.append('    # Perform computation')
        code.append('    result = x + y')
        code.append('    result = result * 2')
        code.append('    ')
        code.append('    # Return result')
        code.append('    return result')
        code.append('')

    # Add a class with methods if there's room
    if num_lines > 50:
        code.append('')
        code.append('class TestClass:')
        code.append('    """A test class for performance testing."""')
        code.append('    ')
        code.append('    def __init__(self, value: int):')
        code.append('        """Initialize with a value."""')
        code.append('        self.value = value')
        code.append('    ')
        code.append('    def process(self, data: int) -> int:')
        code.append('        """Process data with the stored value."""')
        code.append('        return self.value + data')

    return '\n'.join(code)


def generate_php_file(num_lines: int) -> str:
    """Generate syntactically valid PHP code with specified number of lines.

    Args:
        num_lines: Target number of lines to generate.

    Returns:
        PHP source code string.
    """
    code = ['<?php', '/**', ' * Test module for performance testing.', ' */', '']

    # Calculate number of functions to generate
    num_functions = max(1, num_lines // 25)

    for i in range(num_functions):
        code.append('/**')
        code.append(f' * Function {i} for testing.')
        code.append(' * ')
        code.append(' * Performs a computation on two integers.')
        code.append(' * ')
        code.append(' * @param int $x First integer')
        code.append(' * @param int $y Second integer')
        code.append(' * @return int The computed result')
        code.append(' */')
        code.append(f'function function_{i}($x, $y) {{')
        code.append('    // Initialize result')
        code.append('    $result = 0;')
        code.append('    ')
        code.append('    // Perform computation')
        code.append('    $result = $x + $y;')
        code.append('    $result = $result * 2;')
        code.append('    ')
        code.append('    // Return result')
        code.append('    return $result;')
        code.append('}')
        code.append('')

    # Add a class with methods if there's room
    if num_lines > 50:
        code.append('/**')
        code.append(' * A test class for performance testing.')
        code.append(' */')
        code.append('class TestClass {')
        code.append('    /**')
        code.append('     * @var int Stored value')
        code.append('     */')
        code.append('    private $value;')
        code.append('    ')
        code.append('    /**')
        code.append('     * Initialize with a value.')
        code.append('     * ')
        code.append('     * @param int $value Initial value')
        code.append('     */')
        code.append('    public function __construct($value) {')
        code.append('        $this->value = $value;')
        code.append('    }')
        code.append('    ')
        code.append('    /**')
        code.append('     * Process data with the stored value.')
        code.append('     * ')
        code.append('     * @param int $data Data to process')
        code.append('     * @return int Processed result')
        code.append('     */')
        code.append('    public function process($data) {')
        code.append('        return $this->value + $data;')
        code.append('    }')
        code.append('}')

    return '\n'.join(code)


class TestCodeIndexingPerformance:
    """Performance tests for code indexing."""

    def test_index_10k_loc_python(
        self,
        tmp_path: Path,
        perf_test_collection: str,
        cleanup_perf_collection: str,
    ):
        """Test indexing 10,000 lines of Python code in under 60 seconds.

        Success criteria: Index 10K LOC in < 60 seconds.
        """
        # Generate test Python files totaling ~10K LOC
        code_dir = tmp_path / "large_project"
        code_dir.mkdir()

        # Create 20 files with 500 lines each
        for i in range(20):
            code = generate_python_file(500)
            (code_dir / f"module_{i}.py").write_text(code)

        # Register Python code loader
        from qdrant_indexer.code_loaders import PythonCodeLoader
        from qdrant_indexer.loaders import LOADERS
        LOADERS[".py"] = PythonCodeLoader

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=perf_test_collection,
        )
        indexer.ensure_collection()

        chunker = RecursiveChunker(chunk_size=512, overlap=50)

        start = time.time()
        total_chunks = 0
        total_files = 0

        for file in code_dir.glob("*.py"):
            chunks = indexer.index_file(file, chunker)
            total_chunks += chunks
            total_files += 1

        elapsed = time.time() - start

        # Calculate throughput
        total_loc = 20 * 500  # 10,000 lines
        loc_per_second = total_loc / elapsed if elapsed > 0 else 0

        print(f"\nPerformance Results (Python):")
        print(f"  Files indexed: {total_files}")
        print(f"  Total LOC: {total_loc}")
        print(f"  Chunks created: {total_chunks}")
        print(f"  Time elapsed: {elapsed:.2f}s")
        print(f"  Throughput: {loc_per_second:.1f} LOC/second")

        assert elapsed < 60, f"Indexing took {elapsed:.2f}s, expected <60s"
        assert total_chunks > 0, "No chunks were created"
        assert total_files == 20, f"Expected 20 files, indexed {total_files}"

    def test_index_10k_loc_php(
        self,
        tmp_path: Path,
        perf_test_collection: str,
        cleanup_perf_collection: str,
    ):
        """Test indexing 10,000 lines of PHP code in under 60 seconds.

        Success criteria: Index 10K LOC in < 60 seconds.
        """
        # Generate test PHP files totaling ~10K LOC
        code_dir = tmp_path / "large_project"
        code_dir.mkdir()

        # Create 20 files with 500 lines each
        for i in range(20):
            code = generate_php_file(500)
            (code_dir / f"module_{i}.php").write_text(code)

        # Register PHP code loader
        from qdrant_indexer.code_loaders import PHPCodeLoader
        from qdrant_indexer.loaders import LOADERS
        LOADERS[".php"] = PHPCodeLoader

        indexer = QdrantIndexer(
            qdrant_url=QDRANT_URL,
            collection_name=perf_test_collection,
        )
        indexer.ensure_collection()

        chunker = RecursiveChunker(chunk_size=512, overlap=50)

        start = time.time()
        total_chunks = 0
        total_files = 0

        for file in code_dir.glob("*.php"):
            chunks = indexer.index_file(file, chunker)
            total_chunks += chunks
            total_files += 1

        elapsed = time.time() - start

        # Calculate throughput
        total_loc = 20 * 500  # 10,000 lines
        loc_per_second = total_loc / elapsed if elapsed > 0 else 0

        print(f"\nPerformance Results (PHP):")
        print(f"  Files indexed: {total_files}")
        print(f"  Total LOC: {total_loc}")
        print(f"  Chunks created: {total_chunks}")
        print(f"  Time elapsed: {elapsed:.2f}s")
        print(f"  Throughput: {loc_per_second:.1f} LOC/second")

        assert elapsed < 60, f"Indexing took {elapsed:.2f}s, expected <60s"
        assert total_chunks > 0, "No chunks were created"
        assert total_files == 20, f"Expected 20 files, indexed {total_files}"
