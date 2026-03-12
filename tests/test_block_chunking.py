"""Tests for block-aware markdown chunking."""

from Data.preprocessing import Block, pack_chunks, parse_blocks


class TestParseBlocks:
    """Tests for the markdown block parser."""

    def test_heading_block(self):
        text = "## Section Title\n\nSome paragraph text here."
        blocks = parse_blocks(text)
        assert blocks[0].type == "heading"
        assert blocks[0].text == "## Section Title"
        assert blocks[0].heading_context == "Section Title"

    def test_paragraph_block(self):
        text = "This is a paragraph.\nIt continues on the next line."
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "paragraph"
        assert "This is a paragraph." in blocks[0].text

    def test_table_block(self):
        text = "| Header 1 | Header 2 |\n|----------|----------|\n| Cell 1 | Cell 2 |"
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "table"
        assert "Header 1" in blocks[0].text
        assert "Cell 2" in blocks[0].text

    def test_list_block_bullet(self):
        text = "- Item one\n- Item two\n- Item three"
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "list"

    def test_list_block_numbered(self):
        text = "1. First item\n2. Second item\n3. Third item"
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "list"

    def test_list_block_dotted_numbering(self):
        text = "1.1.1 Recommendation about GFR.\n1.1.2 Another recommendation.\n1.1.3 Third recommendation."
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "list"

    def test_list_with_continuation(self):
        text = "- Item one\n  continuation of item one\n- Item two"
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "list"
        assert "continuation" in blocks[0].text

    def test_blank_lines_separate_blocks(self):
        text = "Paragraph one.\n\nParagraph two."
        blocks = parse_blocks(text)
        assert len(blocks) == 2
        assert blocks[0].type == "paragraph"
        assert blocks[1].type == "paragraph"

    def test_heading_context_propagates(self):
        text = "## Overview\n\nFirst paragraph.\n\nSecond paragraph."
        blocks = parse_blocks(text)
        assert blocks[0].heading_context == "Overview"
        assert blocks[1].heading_context == "Overview"
        assert blocks[2].heading_context == "Overview"

    def test_heading_context_updates(self):
        text = "## Section A\n\nParagraph A.\n\n## Section B\n\nParagraph B."
        blocks = parse_blocks(text)
        assert blocks[1].heading_context == "Section A"
        assert blocks[3].heading_context == "Section B"

    def test_heading_context_none_before_first_heading(self):
        text = "Orphan paragraph.\n\n## First Heading\n\nContent."
        blocks = parse_blocks(text)
        assert blocks[0].heading_context is None

    def test_mixed_content(self):
        text = (
            "## Overview\n\n"
            "Some intro text.\n\n"
            "| Col A | Col B |\n|-------|-------|\n| 1 | 2 |\n\n"
            "- Bullet one\n- Bullet two\n\n"
            "Closing paragraph."
        )
        blocks = parse_blocks(text)
        types = [b.type for b in blocks]
        assert types == ["heading", "paragraph", "table", "list", "paragraph"]

    def test_empty_text(self):
        blocks = parse_blocks("")
        assert blocks == []

    def test_asterisk_list(self):
        text = "* Item one\n* Item two"
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "list"

    def test_heading_strips_hashes_for_context(self):
        text = "### Sub-sub heading"
        blocks = parse_blocks(text)
        assert blocks[0].heading_context == "Sub-sub heading"


class TestPackChunks:
    """Tests for the greedy block packer."""

    def _make_block(self, type_="paragraph", text="X" * 400, heading_context="Section"):
        """Helper to create a Block with controllable size."""
        return Block(type=type_, text=text, heading_context=heading_context)

    def test_single_small_block(self):
        blocks = [self._make_block(text="Short text.")]
        chunks = pack_chunks(blocks, chunk_size=800)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Short text."

    def test_packing_multiple_small_blocks(self):
        # Each block ~25 tokens (100 chars / 4). 4 blocks = 100 tokens < 800.
        blocks = [self._make_block(text="A" * 100) for _ in range(4)]
        chunks = pack_chunks(blocks, chunk_size=800)
        assert len(chunks) == 1

    def test_split_when_exceeding_limit(self):
        # Each block ~200 tokens (800 chars). 2 blocks = 400 tokens < 800. 3 blocks = 600 > 500.
        blocks = [self._make_block(text="A" * 800) for _ in range(3)]
        chunks = pack_chunks(blocks, chunk_size=500)
        assert len(chunks) >= 2

    def test_oversized_single_block_not_split(self):
        big_block = self._make_block(text="A" * 8000)  # 2000 tokens, way over 800
        chunks = pack_chunks([big_block], chunk_size=800)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "A" * 8000

    def test_heading_sticks_to_next_block(self):
        heading = self._make_block(type_="heading", text="## Title")
        para = self._make_block(text="Content here.")
        # Make a preceding block that fills up close to the limit
        filler = self._make_block(text="F" * 3000)  # 750 tokens
        chunks = pack_chunks([filler, heading, para], chunk_size=800)
        # Heading and para should be in the same chunk, not heading alone
        for chunk in chunks:
            lines = chunk["text"].split("\n\n")
            if "## Title" in lines:
                assert "Content here." in chunk["text"]

    def test_overlap_content_carried_forward(self):
        b1 = self._make_block(text="B" * 3200)  # 800 tokens, fills chunk alone
        b2 = self._make_block(text="Overlap block")  # small, under 150 token cap
        b3 = self._make_block(text="C" * 3200)  # 800 tokens, fills next chunk
        chunks = pack_chunks([b1, b2, b3], chunk_size=800, chunk_overlap=1)
        # b2 should appear in its chunk AND as overlap in the next chunk
        assert len(chunks) >= 2
        chunks_with_overlap = [c for c in chunks if "Overlap block" in c["text"]]
        assert len(chunks_with_overlap) >= 2, "Overlap block should appear in two consecutive chunks"

    def test_overlap_respects_token_cap(self):
        big = self._make_block(text="B" * 3200)  # 800 tokens — over 150 token cap
        after = self._make_block(text="After content.")
        chunks = pack_chunks([big, after], chunk_size=800, chunk_overlap=1)
        # big exceeds 150 token overlap cap, so should NOT be carried as overlap
        assert len(chunks) == 2
        assert "B" * 3200 not in chunks[1]["text"]

    def test_section_metadata_from_first_block(self):
        blocks = [
            self._make_block(text="Content.", heading_context="Intro"),
            self._make_block(text="More.", heading_context="Methods"),
        ]
        chunks = pack_chunks(blocks, chunk_size=800)
        if len(chunks) == 1:
            assert chunks[0]["section"] == "Intro"

    def test_empty_blocks(self):
        chunks = pack_chunks([], chunk_size=800)
        assert chunks == []
