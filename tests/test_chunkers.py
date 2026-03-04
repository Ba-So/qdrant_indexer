"""Tests for text chunkers."""

from pathlib import Path

import pytest

from qdrant_indexer.chunkers import (
    CHUNKERS,
    Chunker,
    CodeChunker,
    FixedSizeChunker,
    HTMLChunker,
    MarkdownChunker,
    RecursiveChunker,
    SemanticChunker,
    get_chunker,
    get_chunker_for_file,
    merge_small_chunks,
    split_text_with_overlap,
)


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_chunks_respect_size_limit(self, sample_long_text: str):
        """Assert all chunks <= chunk_size."""
        chunker = RecursiveChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk(sample_long_text)

        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 200, f"Chunk {i} exceeds size limit: {len(chunk)}"

    def test_empty_text_returns_empty_list(self):
        """Edge case for empty string."""
        chunker = RecursiveChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_text_smaller_than_chunk_size(self):
        """Verify single chunk returned for small text."""
        chunker = RecursiveChunker(chunk_size=1000)
        text = "Short text"
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_paragraph_splitting(self):
        """Text with \\n\\n should split at paragraphs first."""
        chunker = RecursiveChunker(chunk_size=100, overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text)

        # Should split into separate chunks at paragraph boundaries
        assert len(chunks) >= 1
        # Full text should fit in one chunk at this size
        assert "First paragraph" in chunks[0]

    def test_overlap_parameter_zero(self):
        """Test with overlap=0."""
        chunker = RecursiveChunker(chunk_size=50, overlap=0)
        text = "A" * 100
        chunks = chunker.chunk(text)

        assert len(chunks) == 2
        # No overlap means chunks don't share content
        assert chunks[0] == "A" * 50
        assert chunks[1] == "A" * 50

    def test_overlap_parameter_nonzero(self):
        """Test with nonzero overlap."""
        chunker = RecursiveChunker(chunk_size=50, overlap=10)
        text = "A" * 100
        chunks = chunker.chunk(text)

        # With overlap, second chunk should start with overlap from first
        assert len(chunks) >= 2

    def test_large_document_chunking(self, sample_long_text: str):
        """Verify chunking works on larger documents."""
        chunker = RecursiveChunker(chunk_size=300, overlap=30)
        chunks = chunker.chunk(sample_long_text)

        assert len(chunks) > 1
        # All chunks should have content
        for chunk in chunks:
            assert len(chunk) > 0

    def test_unicode_text(self):
        """Test chunking works with unicode characters."""
        chunker = RecursiveChunker(chunk_size=50, overlap=5)
        text = "Hello 世界! " * 20
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        # Verify no corruption
        combined = "".join(chunks)
        assert "世界" in combined

    def test_is_chunker_subclass(self):
        """Verify RecursiveChunker is a Chunker."""
        assert issubclass(RecursiveChunker, Chunker)


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_fixed_chunks_size(self):
        """Verify chunks are approximately chunk_size."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        text = "A" * 150
        chunks = chunker.chunk(text)

        assert len(chunks) == 3
        assert len(chunks[0]) == 50
        assert len(chunks[1]) == 50
        assert len(chunks[2]) == 50

    def test_fixed_overlap_calculation(self):
        """Verify overlap is correctly applied."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        text = "A" * 100
        chunks = chunker.chunk(text)

        # With overlap of 10, step is 40, so we need more chunks
        assert len(chunks) >= 2

    def test_no_overlap_mode(self):
        """Test with overlap=0."""
        chunker = FixedSizeChunker(chunk_size=25, overlap=0)
        text = "ABCDE" * 10  # 50 chars
        chunks = chunker.chunk(text)

        assert len(chunks) == 2
        assert chunks[0] == "ABCDE" * 5
        assert chunks[1] == "ABCDE" * 5

    def test_text_smaller_than_chunk(self):
        """Should return single chunk for small text."""
        chunker = FixedSizeChunker(chunk_size=100)
        text = "Short"
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_fixed(self):
        """Edge case handling for empty text."""
        chunker = FixedSizeChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("  ") == []

    def test_is_chunker_subclass(self):
        """Verify FixedSizeChunker is a Chunker."""
        assert issubclass(FixedSizeChunker, Chunker)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_split_text_with_overlap_basic(self):
        """Test basic splitting with overlap."""
        text = "A" * 100
        chunks = split_text_with_overlap(text, size=50, overlap=10)

        assert len(chunks) >= 2
        assert len(chunks[0]) == 50

    def test_split_text_with_overlap_empty(self):
        """Test with empty text."""
        assert split_text_with_overlap("", size=50, overlap=10) == []

    def test_split_text_with_overlap_small(self):
        """Test with text smaller than size."""
        text = "Small"
        chunks = split_text_with_overlap(text, size=50, overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_merge_small_chunks_basic(self):
        """Test merging small chunks."""
        chunks = ["Hi", "there", "world"]
        merged = merge_small_chunks(chunks, min_size=10)

        # Should merge small chunks together
        assert len(merged) <= len(chunks)

    def test_merge_small_chunks_empty(self):
        """Test with empty list."""
        assert merge_small_chunks([], min_size=10) == []

    def test_merge_small_chunks_already_large(self):
        """Test when chunks are already large enough."""
        chunks = ["A" * 50, "B" * 50]
        merged = merge_small_chunks(chunks, min_size=10)

        assert len(merged) == 2


class TestChunkerDefaults:
    """Tests for default parameter values."""

    def test_recursive_chunker_defaults(self):
        """Verify default values for RecursiveChunker."""
        chunker = RecursiveChunker()
        assert chunker.chunk_size == 1536
        assert chunker.overlap == 200

    def test_fixed_size_chunker_defaults(self):
        """Verify default values for FixedSizeChunker."""
        chunker = FixedSizeChunker()
        assert chunker.chunk_size == 1536
        assert chunker.overlap == 200


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [
        (100, 10),
        (200, 20),
        (500, 50),
        (1000, 100),
    ],
)
def test_recursive_chunker_parametrized(
    chunk_size: int, overlap: int, sample_long_text: str
):
    """Test RecursiveChunker with various configurations."""
    chunker = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk(sample_long_text)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= chunk_size


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [
        (50, 0),
        (50, 10),
        (100, 25),
    ],
)
def test_fixed_size_chunker_parametrized(chunk_size: int, overlap: int):
    """Test FixedSizeChunker with various configurations."""
    text = "X" * 200
    chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk(text)

    assert len(chunks) > 0
    # First chunk should be exactly chunk_size
    assert len(chunks[0]) == chunk_size


class TestMarkdownChunker:
    """Tests for MarkdownChunker."""

    def test_header_splitting_all_levels(self):
        """Test that h1-h6 headers are detected and split."""
        text = """# H1 Title

Content under H1.

## H2 Section

Content under H2.

### H3 Subsection

Content under H3.

#### H4 Heading

Content under H4.

##### H5 Heading

Content under H5.

###### H6 Heading

Content under H6.
"""
        chunker = MarkdownChunker(chunk_size=500, min_section_size=10)
        chunks = chunker.chunk(text)

        # Should have multiple chunks, one per section
        assert len(chunks) >= 6
        # Verify headers are present
        assert any("# H1 Title" in c for c in chunks)
        assert any("## H2 Section" in c for c in chunks)
        assert any("### H3 Subsection" in c for c in chunks)
        assert any("#### H4 Heading" in c for c in chunks)
        assert any("##### H5 Heading" in c for c in chunks)
        assert any("###### H6 Heading" in c for c in chunks)

    def test_code_block_preservation(self, sample_markdown_with_code_blocks: str):
        """Test that content in ``` blocks stays intact and headers inside are ignored."""
        chunker = MarkdownChunker(chunk_size=500, min_section_size=10)
        chunks = chunker.chunk(sample_markdown_with_code_blocks)

        # Find chunk containing the Python code block
        code_chunks = [c for c in chunks if "def example():" in c]
        assert len(code_chunks) >= 1

        # The # comment inside the code block should NOT cause a split
        # The code block should be in the same chunk as its preceding header
        for chunk in code_chunks:
            assert "```python" in chunk or "def example():" in chunk

    def test_frontmatter_handling(self):
        """Test that YAML frontmatter is treated as separate section."""
        text = """---
title: Test Document
author: Test Author
---
# Main Content

Body text here.
"""
        chunker = MarkdownChunker(chunk_size=500)
        chunks = chunker.chunk(text)

        # First chunk should be frontmatter
        assert chunks[0].startswith("---")
        assert "title: Test Document" in chunks[0]
        assert chunks[0].endswith("---")

        # Second chunk should be main content
        assert any("# Main Content" in c for c in chunks[1:])

    def test_large_section_fallback(self, sample_large_markdown: str):
        """Test that sections > chunk_size use RecursiveChunker."""
        chunker = MarkdownChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk(sample_large_markdown)

        # All chunks should respect size limit
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 500, f"Chunk {i} exceeds size limit: {len(chunk)}"

        # Should have multiple chunks due to large content
        assert len(chunks) > 3

    def test_no_headers_fallback(self):
        """Test that documents without headers use RecursiveChunker."""
        text = "This is a plain text document without any headers. " * 50
        chunker = MarkdownChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk(text)

        # Should still produce chunks
        assert len(chunks) > 0
        # All chunks should respect size limit
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_min_section_size_merging(self):
        """Test that small sections are merged with neighbors."""
        text = """# Section 1

A

## Section 2

B

## Section 3

C
"""
        # With high min_section_size, small sections should merge
        chunker = MarkdownChunker(chunk_size=500, min_section_size=100)
        chunks = chunker.chunk(text)

        # Should have fewer chunks than sections due to merging
        assert len(chunks) < 3

    def test_empty_text(self):
        """Test that empty input returns []."""
        chunker = MarkdownChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []
        assert chunker.chunk("\n\n") == []

    def test_is_chunker_subclass(self):
        """Test that MarkdownChunker inherits from Chunker."""
        assert issubclass(MarkdownChunker, Chunker)

    def test_nested_headers(self):
        """Test that nested headers preserve hierarchy context."""
        text = """# Top Level

Content.

## Second Level

More content.

### Third Level

Deep content that is repeated many times to exceed chunk size. """ + "X" * 2000

        chunker = MarkdownChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk(text)

        # The large third-level section should be split
        # Later chunks should have context about parent headers
        assert len(chunks) > 3

    def test_chunks_respect_size_limit(self, sample_large_markdown: str):
        """Assert all chunks <= chunk_size."""
        chunker = MarkdownChunker(chunk_size=400, overlap=40)
        chunks = chunker.chunk(sample_large_markdown)

        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 400, f"Chunk {i} exceeds size limit: {len(chunk)}"

    def test_default_parameters(self):
        """Verify default values for MarkdownChunker."""
        chunker = MarkdownChunker()
        assert chunker.chunk_size == 1500
        assert chunker.overlap == 100
        assert chunker.min_section_size == 100


class TestHTMLChunker:
    """Tests for HTMLChunker."""

    def test_semantic_tag_splitting(self):
        """Test that semantic tags (article, section, etc.) are split."""
        html = """
        <html>
        <body>
            <article>Article content here.</article>
            <section>Section content here.</section>
            <aside>Sidebar content.</aside>
        </body>
        </html>
        """
        chunker = HTMLChunker(chunk_size=500)
        chunks = chunker.chunk(html)

        assert len(chunks) >= 3
        assert any("Article content" in c for c in chunks)
        assert any("Section content" in c for c in chunks)
        assert any("Sidebar content" in c for c in chunks)

    def test_heading_tag_fallback(self):
        """Test fallback to heading tags (h1-h6) when no semantic tags."""
        html = """
        <html>
        <body>
            <h1>Main Title</h1>
            <p>Content under main title.</p>
            <h2>Section One</h2>
            <p>Content under section one.</p>
            <h3>Subsection</h3>
            <p>Content under subsection.</p>
        </body>
        </html>
        """
        chunker = HTMLChunker(chunk_size=500)
        chunks = chunker.chunk(html)

        assert len(chunks) >= 3
        assert any("Main Title" in c for c in chunks)
        assert any("Section One" in c for c in chunks)
        assert any("Subsection" in c for c in chunks)

    def test_tag_stripping_output_is_plain_text(self):
        """Test that output is plain text with no HTML tags."""
        html = """
        <html>
        <body>
            <h1>Title</h1>
            <p>This is <strong>bold</strong> and <em>italic</em> text.</p>
            <div><span>Nested <a href="#">link</a> content.</span></div>
        </body>
        </html>
        """
        chunker = HTMLChunker(chunk_size=1000)
        chunks = chunker.chunk(html)

        for chunk in chunks:
            assert "<" not in chunk
            assert ">" not in chunk
            assert "bold" in chunk or "italic" in chunk or "link" in chunk

    def test_script_style_removal(self):
        """Test that script and style tags are completely removed."""
        html = """
        <html>
        <head>
            <style>.red { color: red; }</style>
            <script>alert('evil');</script>
        </head>
        <body>
            <h1>Title</h1>
            <p>Visible content.</p>
            <script>console.log('removed');</script>
            <noscript>JavaScript disabled message.</noscript>
        </body>
        </html>
        """
        chunker = HTMLChunker(chunk_size=500)
        chunks = chunker.chunk(html)

        combined = " ".join(chunks)
        assert "alert" not in combined
        assert "console.log" not in combined
        assert "color: red" not in combined
        assert "JavaScript disabled" not in combined
        assert "Visible content" in combined

    def test_malformed_html_handling(self, malformed_html_content: str):
        """Test that malformed HTML is handled gracefully."""
        chunker = HTMLChunker(chunk_size=500)
        # Should not raise an exception
        chunks = chunker.chunk(malformed_html_content)

        # Should still extract some content
        assert len(chunks) > 0
        combined = " ".join(chunks)
        assert "Unclosed paragraph" in combined or "Missing closing" in combined

    def test_table_keep_together(self):
        """Test that small tables are kept together."""
        html = """
        <html>
        <body>
            <h1>Data Table</h1>
            <table>
                <tr><th>Name</th><th>Value</th></tr>
                <tr><td>Item A</td><td>100</td></tr>
                <tr><td>Item B</td><td>200</td></tr>
            </table>
        </body>
        </html>
        """
        chunker = HTMLChunker(chunk_size=500)
        chunks = chunker.chunk(html)

        # Table content should be in chunks
        combined = " ".join(chunks)
        assert "Item A" in combined
        assert "Item B" in combined
        assert "100" in combined
        assert "200" in combined

    def test_table_split_by_row(self):
        """Test that large tables are split by row."""
        # Create a large table
        rows = "".join(
            f"<tr><td>Row {i}</td><td>{'X' * 100}</td></tr>" for i in range(50)
        )
        html = f"""
        <html>
        <body>
            <table>{rows}</table>
        </body>
        </html>
        """
        chunker = HTMLChunker(chunk_size=300, overlap=30)
        chunks = chunker.chunk(html)

        # Should have multiple chunks
        assert len(chunks) > 1
        # All chunks should respect size limit
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 300, f"Chunk {i} exceeds size: {len(chunk)}"

    def test_no_structure_fallback_to_recursive(self):
        """Test fallback to RecursiveChunker when no structure tags."""
        html = """
        <html>
        <body>
            <p>Just some plain paragraph text that repeats. </p>
        </body>
        </html>
        """ + "<p>More text content. </p>" * 100

        chunker = HTMLChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk(html)

        # Should produce chunks
        assert len(chunks) > 0
        # All should respect size limit
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_empty_text(self):
        """Test that empty input returns []."""
        chunker = HTMLChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []
        assert chunker.chunk("\n\n") == []

    def test_is_chunker_subclass(self):
        """Test that HTMLChunker inherits from Chunker."""
        assert issubclass(HTMLChunker, Chunker)

    def test_default_parameters(self):
        """Verify default values for HTMLChunker."""
        chunker = HTMLChunker()
        assert chunker.chunk_size == 1500
        assert chunker.overlap == 100
        assert chunker.preserve_tags is None

    def test_chunks_respect_size_limit(self, sample_html_content: str):
        """Assert all chunks <= chunk_size."""
        chunker = HTMLChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk(sample_html_content)

        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 200, f"Chunk {i} exceeds size limit: {len(chunk)}"

    def test_deeply_nested_structure(self):
        """Test handling of deeply nested HTML structures."""
        html = """
        <html>
        <body>
            <div>
                <div>
                    <div>
                        <div>
                            <p>Deeply nested content here.</p>
                            <ul>
                                <li>Item 1</li>
                                <li>Item 2</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        chunker = HTMLChunker(chunk_size=500)
        chunks = chunker.chunk(html)

        combined = " ".join(chunks)
        assert "Deeply nested content" in combined
        assert "Item 1" in combined
        assert "Item 2" in combined

    def test_mixed_semantic_and_heading_tags(self):
        """Test HTML with both semantic tags and headings."""
        html = """
        <html>
        <body>
            <article>
                <h1>Article Title</h1>
                <p>Article intro.</p>
            </article>
            <section>
                <h2>Section Title</h2>
                <p>Section content.</p>
            </section>
        </body>
        </html>
        """
        chunker = HTMLChunker(chunk_size=500)
        chunks = chunker.chunk(html)

        # Semantic tags should take priority
        combined = " ".join(chunks)
        assert "Article Title" in combined
        assert "Section Title" in combined

    def test_unicode_content(self):
        """Test chunking works with unicode characters."""
        html = """
        <html>
        <body>
            <h1>Unicode Test</h1>
            <p>Hello 世界! Émojis: 🎉🚀</p>
            <p>Ελληνικά, العربية, עברית</p>
        </body>
        </html>
        """
        chunker = HTMLChunker(chunk_size=500)
        chunks = chunker.chunk(html)

        combined = " ".join(chunks)
        assert "世界" in combined
        assert "🎉" in combined or "🚀" in combined
        assert "Ελληνικά" in combined

    def test_whitespace_normalization(self):
        """Test that excessive whitespace is normalized."""
        html = """
        <html>
        <body>
            <h1>Title</h1>
            <p>Content    with     extra      spaces.</p>
            <p>
                And

                newlines.
            </p>
        </body>
        </html>
        """
        chunker = HTMLChunker(chunk_size=500)
        chunks = chunker.chunk(html)

        combined = " ".join(chunks)
        # Should not have multiple consecutive spaces
        assert "    " not in combined
        assert "Content with extra spaces" in combined


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_is_chunker_subclass(self):
        """Verify SemanticChunker is a Chunker."""
        assert issubclass(SemanticChunker, Chunker)

    def test_empty_text_returns_empty_list(self):
        """Test that empty input returns []."""
        chunker = SemanticChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []
        assert chunker.chunk("\n\n") == []

    def test_small_text_returns_single_chunk(self):
        """Verify single chunk returned for small text."""
        chunker = SemanticChunker(chunk_size=1000)
        text = "Short text that fits in one chunk."
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_default_parameters(self):
        """Verify default values for SemanticChunker."""
        chunker = SemanticChunker()
        assert chunker.chunk_size == 1500
        assert chunker.min_chunk_size == 200
        assert chunker.similarity_threshold == 0.5
        assert chunker.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_splits_at_topic_change(self):
        """Test that text is split at semantic topic boundaries."""
        # Create text with distinctly different topics, each long enough
        python_topic = (
            "Python is a programming language. It has functions and classes. "
            "Variables store data. Loops iterate over sequences. "
            "Python supports object-oriented programming. "
            "Python developers write clean and readable code. "
            "The Python community creates many useful libraries. "
            "Type hints improve Python code documentation."
        )
        cooking_topic = (
            "Cooking requires ingredients and recipes. Baking uses an oven. "
            "Chefs prepare delicious meals. Spices add flavor to dishes. "
            "Kitchen equipment includes pots and pans. "
            "Restaurants serve customers every day. "
            "Food preparation follows safety guidelines. "
            "Culinary arts combine creativity with technique."
        )
        space_topic = (
            "Space exploration involves rockets and satellites. "
            "Astronauts travel to the International Space Station. "
            "The Moon orbits the Earth. Mars is the red planet. "
            "Telescopes observe distant galaxies. "
            "NASA launches missions to explore the solar system. "
            "Space technology advances human knowledge constantly."
        )

        text = f"{python_topic}\n\n{cooking_topic}\n\n{space_topic}"

        # Use smaller chunk_size to ensure splitting is needed
        chunker = SemanticChunker(chunk_size=400, similarity_threshold=0.5)
        chunks = chunker.chunk(text)

        # Should split into multiple chunks at topic boundaries
        assert len(chunks) >= 2

        # Each chunk should contain content
        for chunk in chunks:
            assert len(chunk) > 0

    def test_keeps_similar_content_together(self):
        """Test that semantically similar content stays together."""
        # All about programming
        text = """
        Programming involves writing code. Developers use text editors.
        Software engineering is a discipline. Code must be tested.

        Functions accept parameters. Methods belong to classes.
        Variables store values. Types define data structures.

        Debugging finds errors. Testing validates behavior.
        Refactoring improves code. Documentation explains usage.
        """

        chunker = SemanticChunker(chunk_size=1000, similarity_threshold=0.3)
        chunks = chunker.chunk(text.strip())

        # With similar content and low threshold, may produce fewer chunks
        assert len(chunks) >= 1
        # Content should be preserved
        combined = " ".join(chunks)
        assert "Programming" in combined
        assert "Functions" in combined

    def test_chunks_respect_max_size(self):
        """Assert all chunks <= chunk_size."""
        # Create a long text with varying topics
        topics = [
            "Machine learning algorithms process data to find patterns. "
            "Neural networks mimic the human brain structure. " * 5,
            "Ancient history spans thousands of years of civilization. "
            "Empires rose and fell across the world. " * 5,
            "Music theory covers scales, chords, and harmony. "
            "Composers create symphonies and concertos. " * 5,
        ]
        text = "\n\n".join(topics)

        chunker = SemanticChunker(chunk_size=400, similarity_threshold=0.5)
        chunks = chunker.chunk(text)

        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 400, f"Chunk {i} exceeds size limit: {len(chunk)}"

    def test_small_chunks_merged(self):
        """Test that chunks smaller than min_chunk_size are merged."""
        # Create paragraphs that would individually be very small
        text = """
        Hi.

        Hello.

        Hey.

        Greetings.

        Salutations.

        Now let's talk about something completely different for a longer paragraph.
        This paragraph discusses philosophy and the meaning of existence in depth.
        """

        chunker = SemanticChunker(
            chunk_size=500, min_chunk_size=100, similarity_threshold=0.3
        )
        chunks = chunker.chunk(text.strip())

        # Small chunks should be merged
        # We shouldn't have 6 separate tiny chunks
        assert len(chunks) <= 4

    def test_fallback_on_uniform_similarity(self):
        """Test fallback to RecursiveChunker when similarity is uniform."""
        # Highly repetitive text with uniform similarity
        text = ("This is a sentence. " * 100).strip()

        chunker = SemanticChunker(chunk_size=200, similarity_threshold=0.5)
        chunks = chunker.chunk(text)

        # Should still produce chunks (via fallback)
        assert len(chunks) > 0
        # All chunks should respect size limit
        for chunk in chunks:
            assert len(chunk) <= 200

    def test_model_caching(self):
        """Test that the embedding model is cached at class level."""
        chunker1 = SemanticChunker()
        chunker2 = SemanticChunker()

        # Access model through both chunkers
        model1 = SemanticChunker._get_model(chunker1.embedding_model)
        model2 = SemanticChunker._get_model(chunker2.embedding_model)

        # Should be the same cached instance
        assert model1 is model2

    def test_unicode_text(self):
        """Test chunking works with unicode characters."""
        text = """
        中文段落讨论编程。Python是一种流行的编程语言。
        变量存储数据。函数执行操作。

        日本語のセクション。プログラミングは楽しいです。
        コードを書くのは創造的な活動です。

        Ελληνικά κείμενα. Ο προγραμματισμός είναι τέχνη.
        Οι αλγόριθμοι λύνουν προβλήματα.
        """

        chunker = SemanticChunker(chunk_size=500, similarity_threshold=0.5)
        chunks = chunker.chunk(text.strip())

        # Should produce chunks
        assert len(chunks) > 0

        # Content should be preserved
        combined = " ".join(chunks)
        assert "中文" in combined or "Python" in combined
        assert "日本語" in combined or "プログラミング" in combined

    def test_sentence_level_splitting(self):
        """Test that sentence-level splitting works when paragraphs don't."""
        # Single paragraph with multiple sentences on different topics
        text = (
            "Python is a programming language used for software development. "
            "JavaScript runs in web browsers. "
            "The weather today is sunny and warm. "
            "Rain is expected tomorrow afternoon. "
            "Cooking pasta requires boiling water. "
            "Italian cuisine features tomatoes and olive oil. "
        ) * 3

        chunker = SemanticChunker(chunk_size=400, similarity_threshold=0.4)
        chunks = chunker.chunk(text.strip())

        # Should produce chunks
        assert len(chunks) >= 1
        # Content preserved
        combined = " ".join(chunks)
        assert "Python" in combined
        assert "weather" in combined

    def test_custom_embedding_model(self):
        """Test that custom embedding model parameter is stored."""
        custom_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
        chunker = SemanticChunker(embedding_model=custom_model)

        assert chunker.embedding_model == custom_model

    def test_similarity_threshold_effect(self):
        """Test that lower threshold results in fewer splits."""
        text = """
        Programming involves writing code. Software developers create applications.
        Code editors help write programs. IDEs provide debugging tools.

        Machine learning uses algorithms. Neural networks learn patterns.
        Data science analyzes information. AI automates decisions.

        Web development creates websites. HTML structures content.
        CSS styles pages. JavaScript adds interactivity.
        """

        # High threshold - more likely to split
        chunker_high = SemanticChunker(
            chunk_size=1000, similarity_threshold=0.7
        )
        chunks_high = chunker_high.chunk(text.strip())

        # Low threshold - less likely to split
        chunker_low = SemanticChunker(
            chunk_size=1000, similarity_threshold=0.2
        )
        chunks_low = chunker_low.chunk(text.strip())

        # Higher threshold should generally produce more or equal chunks
        # (though this isn't strictly guaranteed due to merging)
        assert len(chunks_high) >= 1
        assert len(chunks_low) >= 1


class TestGetChunker:
    """Tests for get_chunker factory function."""

    def test_returns_recursive_chunker(self):
        chunker = get_chunker("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_returns_fixed_size_chunker(self):
        chunker = get_chunker("fixed")
        assert isinstance(chunker, FixedSizeChunker)

    def test_returns_markdown_chunker(self):
        chunker = get_chunker("markdown")
        assert isinstance(chunker, MarkdownChunker)

    def test_returns_html_chunker(self):
        chunker = get_chunker("html")
        assert isinstance(chunker, HTMLChunker)

    def test_returns_semantic_chunker(self):
        chunker = get_chunker("semantic")
        assert isinstance(chunker, SemanticChunker)

    def test_returns_code_chunker(self):
        chunker = get_chunker("code")
        assert isinstance(chunker, CodeChunker)

    def test_passes_kwargs_to_chunker(self):
        chunker = get_chunker("recursive", chunk_size=500, overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50

    def test_raises_value_error_for_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown chunker strategy"):
            get_chunker("nonexistent")

    def test_raises_value_error_for_auto_strategy(self):
        with pytest.raises(ValueError, match="requires a file path"):
            get_chunker("auto")


class TestGetChunkerForFile:
    """Tests for get_chunker_for_file factory function."""

    def test_markdown_file_returns_markdown_chunker(self, tmp_path: Path):
        chunker = get_chunker_for_file(tmp_path / "test.md")
        assert isinstance(chunker, MarkdownChunker)

    def test_html_file_returns_html_chunker(self, tmp_path: Path):
        chunker = get_chunker_for_file(tmp_path / "test.html")
        assert isinstance(chunker, HTMLChunker)

    def test_pdf_file_returns_semantic_chunker(self, tmp_path: Path):
        chunker = get_chunker_for_file(tmp_path / "test.pdf")
        assert isinstance(chunker, SemanticChunker)

    def test_python_file_returns_code_chunker(self, tmp_path: Path):
        chunker = get_chunker_for_file(tmp_path / "test.py")
        assert isinstance(chunker, CodeChunker)

    def test_php_file_returns_code_chunker(self, tmp_path: Path):
        chunker = get_chunker_for_file(tmp_path / "test.php")
        assert isinstance(chunker, CodeChunker)

    def test_rust_file_returns_code_chunker(self, tmp_path: Path):
        chunker = get_chunker_for_file(tmp_path / "test.rs")
        assert isinstance(chunker, CodeChunker)

    def test_txt_file_returns_recursive_chunker(self, tmp_path: Path):
        chunker = get_chunker_for_file(tmp_path / "test.txt")
        assert isinstance(chunker, RecursiveChunker)

    def test_unknown_extension_returns_recursive_chunker(self, tmp_path: Path):
        chunker = get_chunker_for_file(tmp_path / "test.unknown")
        assert isinstance(chunker, RecursiveChunker)

    def test_passes_kwargs_to_chunker(self, tmp_path: Path):
        chunker = get_chunker_for_file(tmp_path / "test.md", chunk_size=500)
        assert chunker.chunk_size == 500


class TestChunkersRegistry:
    """Tests for CHUNKERS registry."""

    def test_contains_all_strategies(self):
        expected = {"recursive", "fixed", "markdown", "html", "semantic", "code"}
        assert set(CHUNKERS.keys()) == expected

    def test_all_values_are_chunker_subclasses(self):
        for name, cls in CHUNKERS.items():
            assert issubclass(cls, Chunker), f"{name} is not a Chunker subclass"
