"""Tests for block-aware markdown chunking."""

from Data.preprocessing import Block, parse_blocks


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
