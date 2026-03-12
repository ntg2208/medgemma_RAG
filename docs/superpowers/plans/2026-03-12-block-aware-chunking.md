# Block-Aware Markdown Chunking Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the PDF-based chunking pipeline with a block-aware markdown chunker that reads from pre-processed `Data/processed_with_sections/` and preserves paragraphs, lists, and tables as atomic units.

**Architecture:** A block parser reads markdown into typed blocks (heading, table, list, paragraph). A greedy packer accumulates blocks into chunks up to the token limit without splitting any block. Headings stick to the next content block. Overlap is block-based with a token cap.

**Tech Stack:** Python, LangChain `Document`, `NamedTuple`, `pytest`

**Spec:** `docs/superpowers/specs/2026-03-12-block-aware-chunking-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `Data/preprocessing.py` | Rewrite | Block parser + greedy packer + `DocumentPreprocessor` |
| `Data/export_chunks.py` | Modify | Drop `ckd_stages`, `page_number`, `document_type` from chunk metadata |
| `config.py` | Modify | Add `PROCESSED_WITH_SECTIONS_DIR`, remove `MIN_CHUNK_SIZE`, update `CHUNK_OVERLAP` |
| `tests/test_block_chunking.py` | Create | Tests for block parser, greedy packer, integration |
| `test.py` | Modify | Remove `MIN_CHUNK_SIZE` import |

---

## Chunk 1: Config and Block Parser Tests + Implementation

### Task 1: Update config.py

**Files:**
- Modify: `config.py:38-43`
- Modify: `test.py:32` (remove `MIN_CHUNK_SIZE` import)

- [ ] **Step 1: Update config.py**

Add `PROCESSED_WITH_SECTIONS_DIR`, remove `MIN_CHUNK_SIZE`, update `CHUNK_OVERLAP`:

```python
# In config.py, Chunking Configuration section:

# Directory Paths section - add after VECTORSTORE_DIR:
PROCESSED_WITH_SECTIONS_DIR = DATA_DIR / "processed_with_sections"

# Chunking Configuration section - replace:
CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 1  # number of trailing blocks to repeat (capped at 150 tokens)
# Remove MIN_CHUNK_SIZE entirely
```

- [ ] **Step 2: Fix test.py import**

In `test.py:32`, remove `MIN_CHUNK_SIZE` from the import line.

- [ ] **Step 3: Verify no other imports break**

Run: `uv run python -c "from config import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSED_WITH_SECTIONS_DIR; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add config.py test.py
git commit -m "chore: update chunking config for block-aware chunking"
```

### Task 2: Write block parser tests

**Files:**
- Create: `tests/test_block_chunking.py`

- [ ] **Step 1: Write block type detection tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_block_chunking.py::TestParseBlocks -v`
Expected: FAIL — `parse_blocks` and `Block` do not exist yet.

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_block_chunking.py
git commit -m "test: add block parser tests (red)"
```

### Task 3: Implement block parser

**Files:**
- Modify: `Data/preprocessing.py`

- [ ] **Step 1: Rewrite the entire `Data/preprocessing.py` file**

Replace the **entire file** with the new implementation. This avoids a broken intermediate state where old code references removed imports. The file includes `parse_blocks` now; `pack_chunks` and `DocumentPreprocessor` will be added in Tasks 5 and 6. For now, include minimal stubs for those so the module is importable:

```python
"""
Document preprocessing pipeline for CKD RAG System.

Reads pre-processed markdown files from Data/processed_with_sections/
and chunks them using a block-aware algorithm that preserves paragraphs,
lists, and tables as atomic units.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import NamedTuple, Optional

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PROCESSED_WITH_SECTIONS_DIR,
)

logger = logging.getLogger(__name__)


class Block(NamedTuple):
    """An atomic unit of markdown content."""
    type: str  # "heading", "table", "list", "paragraph"
    text: str
    heading_context: Optional[str]


# Regex for dotted numbering: 1. , 1.1 , 1.1.1 , etc.
_LIST_PATTERN = re.compile(r"^(\d[\d.]*\.\s+|- |\* )")
_HEADING_PATTERN = re.compile(r"^#+\s+")
_TABLE_PATTERN = re.compile(r"^\|")


def _classify_line(line: str) -> str:
    """Classify a single line by its markdown type."""
    if _HEADING_PATTERN.match(line):
        return "heading"
    if _TABLE_PATTERN.match(line):
        return "table"
    if _LIST_PATTERN.match(line):
        return "list"
    return "other"


def parse_blocks(text: str) -> list[Block]:
    """Parse markdown text into a list of typed blocks.

    Blocks are atomic units: headings, tables, lists, or paragraphs.
    Blank lines separate blocks. Contiguous lines of the same type
    (table, list) are grouped. Headings are always single-line blocks.

    Args:
        text: Markdown text content.

    Returns:
        List of Block namedtuples.
    """
    if not text.strip():
        return []

    lines = text.split("\n")
    blocks: list[Block] = []
    current_heading: Optional[str] = None

    # Accumulator for the current block being built
    acc_lines: list[str] = []
    acc_type: Optional[str] = None

    def flush():
        """Emit the accumulated lines as a block."""
        nonlocal acc_lines, acc_type
        if not acc_lines:
            return
        block_text = "\n".join(acc_lines)
        block_type = acc_type if acc_type in ("table", "list") else "paragraph"
        blocks.append(Block(type=block_type, text=block_text, heading_context=current_heading))
        acc_lines = []
        acc_type = None

    for line in lines:
        stripped = line.strip()

        # Blank line = block separator
        if not stripped:
            flush()
            continue

        line_type = _classify_line(stripped)

        if line_type == "heading":
            flush()
            # Extract heading text (strip # prefix)
            heading_text = _HEADING_PATTERN.sub("", stripped).strip()
            current_heading = heading_text
            blocks.append(Block(type="heading", text=stripped, heading_context=current_heading))
            continue

        if line_type == "table":
            if acc_type == "table":
                acc_lines.append(stripped)
            else:
                flush()
                acc_type = "table"
                acc_lines = [stripped]
            continue

        if line_type == "list":
            if acc_type == "list":
                acc_lines.append(line)  # preserve indentation
            else:
                flush()
                acc_type = "list"
                acc_lines = [line]
            continue

        # "other" line — could be a list continuation or a paragraph
        if acc_type == "list" and (line.startswith("  ") or line.startswith("\t")):
            # Continuation of list item (indented)
            acc_lines.append(line)
            continue

        if acc_type == "paragraph" or acc_type is None:
            if acc_type is None:
                acc_type = "paragraph"
            acc_lines.append(stripped)
            continue

        # Line doesn't match current block type — flush and start new paragraph
        flush()
        acc_type = "paragraph"
        acc_lines = [stripped]

    flush()
    return blocks


# ---- Stubs (replaced in Tasks 5 and 6) ----

def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def pack_chunks(blocks, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, overlap_token_cap=150):
    raise NotImplementedError("Implemented in Task 5")


class DocumentPreprocessor:
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_directory(self, input_dir=None):
        raise NotImplementedError("Implemented in Task 6")

    def process_document(self, doc_dir):
        raise NotImplementedError("Implemented in Task 6")

    def get_document_stats(self, documents):
        raise NotImplementedError("Implemented in Task 6")


def preprocess_documents(input_dir=None, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    raise NotImplementedError("Implemented in Task 6")


if __name__ == "__main__":
    pass
```

- [ ] **Step 2: Run parser tests**

Run: `uv run pytest tests/test_block_chunking.py::TestParseBlocks -v`
Expected: All 15 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add Data/preprocessing.py
git commit -m "feat: implement block parser for markdown chunking"
```

---

## Chunk 2: Greedy Packer Tests + Implementation

### Task 4: Write greedy packer tests

**Files:**
- Modify: `tests/test_block_chunking.py`

- [ ] **Step 1: Add packer tests**

Append to `tests/test_block_chunking.py`:

```python
from Data.preprocessing import pack_chunks, Block


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
```

- [ ] **Step 2: Run to verify tests fail**

Run: `uv run pytest tests/test_block_chunking.py::TestPackChunks -v`
Expected: FAIL — `pack_chunks` not defined.

- [ ] **Step 3: Commit**

```bash
git add tests/test_block_chunking.py
git commit -m "test: add greedy packer tests (red)"
```

### Task 5: Implement greedy packer

**Files:**
- Modify: `Data/preprocessing.py`

- [ ] **Step 1: Add pack_chunks function after parse_blocks**

```python
def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def pack_chunks(
    blocks: list[Block],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    overlap_token_cap: int = 150,
) -> list[dict]:
    """Pack blocks into chunks without splitting any block.

    Args:
        blocks: List of Block namedtuples from parse_blocks.
        chunk_size: Maximum tokens per chunk.
        chunk_overlap: Number of trailing blocks to carry as overlap.
        overlap_token_cap: Max tokens for overlap blocks.

    Returns:
        List of dicts with keys: text, section, block_types.
    """
    if not blocks:
        return []

    chunks: list[dict] = []
    acc: list[Block] = []
    acc_tokens: int = 0

    def emit():
        """Emit accumulated blocks as a chunk."""
        nonlocal acc, acc_tokens
        if not acc:
            return
        text = "\n\n".join(b.text for b in acc)
        section = acc[0].heading_context
        chunks.append({"text": text, "section": section})
        acc = []
        acc_tokens = 0

    def get_overlap_blocks(prev_acc: list[Block]) -> list[Block]:
        """Get trailing blocks from previous chunk for overlap, respecting token cap."""
        overlap: list[Block] = []
        overlap_tokens = 0
        for block in reversed(prev_acc):
            # Don't carry a heading as sole overlap
            if block.type == "heading" and not overlap:
                continue
            btokens = _estimate_tokens(block.text)
            if overlap_tokens + btokens > overlap_token_cap:
                break
            overlap.insert(0, block)
            overlap_tokens += btokens
            if len(overlap) >= chunk_overlap:
                break
        return overlap

    i = 0
    while i < len(blocks):
        block = blocks[i]
        block_tokens = _estimate_tokens(block.text)

        # Heading rule: peek ahead and bind heading to next content block
        if block.type == "heading" and i + 1 < len(blocks):
            next_block = blocks[i + 1]
            combined_tokens = block_tokens + _estimate_tokens(next_block.text)

            if acc_tokens + combined_tokens > chunk_size and acc:
                prev_acc = list(acc)
                emit()
                acc = get_overlap_blocks(prev_acc)
                acc_tokens = sum(_estimate_tokens(b.text) for b in acc)

            acc.append(block)
            acc.append(next_block)
            acc_tokens += combined_tokens
            i += 2
            continue

        # Would adding this block exceed the limit?
        if acc_tokens + block_tokens > chunk_size and acc:
            prev_acc = list(acc)
            emit()
            acc = get_overlap_blocks(prev_acc)
            acc_tokens = sum(_estimate_tokens(b.text) for b in acc)

        acc.append(block)
        acc_tokens += block_tokens
        i += 1

    emit()
    return chunks
```

- [ ] **Step 2: Run packer tests**

Run: `uv run pytest tests/test_block_chunking.py::TestPackChunks -v`
Expected: All 9 tests PASS.

- [ ] **Step 3: Run all chunking tests together**

Run: `uv run pytest tests/test_block_chunking.py -v`
Expected: All 24 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add Data/preprocessing.py
git commit -m "feat: implement greedy block packer"
```

---

## Chunk 3: DocumentPreprocessor Rewrite + Export Update

### Task 6: Rewrite DocumentPreprocessor class

**Files:**
- Modify: `Data/preprocessing.py`

- [ ] **Step 1: Replace the DocumentPreprocessor class body**

Remove all old methods (`extract_text_from_pdf`, `clean_text`, `extract_ckd_stages`, `detect_section`, `classify_document_type`, `process_pdf`). Remove `CKD_STAGE_PATTERNS`, `NOISE_PATTERNS`, `DocumentMetadata` dataclass. Remove old imports (`fitz`, `dataclass`, `field`, `RecursiveCharacterTextSplitter`, `DOCUMENTS_DIR`, `PROCESSED_DIR`).

New class:

```python
class DocumentPreprocessor:
    """Preprocessor that reads pre-processed markdown and chunks using block-aware algorithm."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_document(self, doc_dir: Path) -> list[Document]:
        """Process a single document directory into LangChain Documents.

        Args:
            doc_dir: Path to document directory containing main_text.md and metadata.json.

        Returns:
            List of LangChain Document objects with metadata.
        """
        main_text_path = doc_dir / "main_text.md"
        metadata_path = doc_dir / "metadata.json"

        if not main_text_path.exists():
            logger.warning(f"No main_text.md in {doc_dir.name}")
            return []

        text = main_text_path.read_text(encoding="utf-8")
        if not text.strip():
            logger.warning(f"Empty main_text.md in {doc_dir.name}")
            return []

        # Load metadata
        doc_metadata = {}
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                doc_metadata = json.load(f)

        title = doc_metadata.get("title", doc_dir.name)
        source = doc_metadata.get("source_file", doc_dir.name)

        # Parse and pack
        blocks = parse_blocks(text)
        packed = pack_chunks(blocks, self.chunk_size, self.chunk_overlap)

        # Build LangChain Documents
        documents = []
        for i, chunk in enumerate(packed):
            metadata = {
                "source": source,
                "title": title,
                "section": chunk["section"],
                "chunk_id": i,
                "total_chunks": len(packed),
            }
            documents.append(Document(page_content=chunk["text"], metadata=metadata))

        logger.info(f"Created {len(documents)} chunks from {doc_dir.name}")
        return documents

    def process_directory(self, input_dir: Optional[Path] = None) -> list[Document]:
        """Process all document directories.

        Args:
            input_dir: Directory containing document subdirectories.
                       Default: PROCESSED_WITH_SECTIONS_DIR.

        Returns:
            Combined list of all Document objects.
        """
        input_dir = Path(input_dir) if input_dir else PROCESSED_WITH_SECTIONS_DIR

        all_documents = []
        doc_dirs = sorted([
            d for d in input_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not doc_dirs:
            logger.warning(f"No document directories found in {input_dir}")
            return []

        logger.info(f"Found {len(doc_dirs)} document directories to process")

        for doc_dir in doc_dirs:
            try:
                documents = self.process_document(doc_dir)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to process {doc_dir.name}: {e}")
                continue

        logger.info(f"Total documents created: {len(all_documents)}")
        return all_documents

    def get_document_stats(self, documents: list[Document]) -> dict:
        """Get statistics about processed documents."""
        if not documents:
            return {"total_documents": 0}

        sources = set(d.metadata.get("source", "unknown") for d in documents)
        total_chars = sum(len(d.page_content) for d in documents)

        return {
            "total_documents": len(documents),
            "unique_sources": len(sources),
            "sources": sorted(sources),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // len(documents) if documents else 0,
        }


def preprocess_documents(
    input_dir: Optional[Path] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Convenience function to preprocess all documents in a directory."""
    preprocessor = DocumentPreprocessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return preprocessor.process_directory(input_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocessor = DocumentPreprocessor()
    documents = preprocessor.process_directory()
    if documents:
        stats = preprocessor.get_document_stats(documents)
        print("\nDocument Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
```

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from Data.preprocessing import DocumentPreprocessor, preprocess_documents, parse_blocks, pack_chunks, Block; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add Data/preprocessing.py
git commit -m "feat: rewrite DocumentPreprocessor to read from processed markdown"
```

### Task 7: Update export_chunks.py

**Files:**
- Modify: `Data/export_chunks.py:56-72`

- [ ] **Step 1: Update prepare_chunk_data to drop removed fields**

Replace the `chunk_data` dict in `prepare_chunk_data` (lines 56-70):

```python
    chunk_data = {
        "content": document.page_content,
        "metadata": {
            "source": document.metadata.get("source"),
            "title": document.metadata.get("title"),
            "chunk_id": document.metadata.get("chunk_id"),
            "total_chunks": document.metadata.get("total_chunks"),
            "section": document.metadata.get("section"),
        },
        "statistics": calculate_chunk_stats(document.page_content),
        "processed_at": export_timestamp,
    }
```

Removed: `page_number`, `ckd_stages`, `document_type`.

- [ ] **Step 2: Verify export_chunks imports still work**

Run: `uv run python -c "from Data.export_chunks import export_to_json, prepare_chunk_data; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add Data/export_chunks.py
git commit -m "chore: remove ckd_stages and page_number from chunk export metadata"
```

---

## Chunk 4: Integration Test

### Task 8: Write and run integration test

**Files:**
- Modify: `tests/test_block_chunking.py`

- [ ] **Step 1: Add integration test**

Append to `tests/test_block_chunking.py`:

```python
from pathlib import Path
from Data.preprocessing import DocumentPreprocessor

PROCESSED_DIR = Path(__file__).parent.parent / "Data" / "processed_with_sections"


class TestIntegration:
    """Integration tests using real document data."""

    def test_process_single_document(self):
        """Process one real document directory end-to-end."""
        doc_dir = PROCESSED_DIR / "Diet_and_Haemodialysis"
        if not doc_dir.exists():
            import pytest
            pytest.skip("Test data not available")

        preprocessor = DocumentPreprocessor()
        docs = preprocessor.process_document(doc_dir)

        assert len(docs) > 0
        for doc in docs:
            assert doc.page_content.strip() != ""
            assert "source" in doc.metadata
            assert "title" in doc.metadata
            assert "chunk_id" in doc.metadata
            assert "total_chunks" in doc.metadata
            assert "section" in doc.metadata
            # ckd_stages and page_number should NOT be present
            assert "ckd_stages" not in doc.metadata
            assert "page_number" not in doc.metadata

    def test_process_directory(self):
        """Process all documents and verify stats."""
        if not PROCESSED_DIR.exists():
            import pytest
            pytest.skip("Test data not available")

        preprocessor = DocumentPreprocessor()
        docs = preprocessor.process_directory()

        assert len(docs) > 0
        stats = preprocessor.get_document_stats(docs)
        assert stats["unique_sources"] > 1
        assert stats["avg_chunk_size"] > 0

    def test_tables_not_split(self):
        """Verify that markdown tables remain intact within chunks."""
        doc_dir = PROCESSED_DIR / "Chronic_Kidney_Disease_Assessment_and_Management_Guidelines"
        if not doc_dir.exists():
            import pytest
            pytest.skip("Test data not available")

        preprocessor = DocumentPreprocessor()
        docs = preprocessor.process_document(doc_dir)

        for doc in docs:
            content = doc.page_content
            # If chunk contains a table start, it should contain matching rows
            lines = content.split("\n")
            in_table = False
            for idx, line in enumerate(lines):
                if line.strip().startswith("|"):
                    in_table = True
                elif in_table and line.strip() == "":
                    in_table = False  # table ended normally at blank line
                elif in_table and not line.strip().startswith("|"):
                    # Non-table line while we think we're in a table
                    # Check if any table lines appear after this point
                    remaining = lines[idx + 1:]
                    table_continues = any(l.strip().startswith("|") for l in remaining)
                    assert not table_continues, f"Table appears split in chunk {doc.metadata['chunk_id']}"
                    in_table = False

    def test_no_chunk_is_only_heading(self):
        """No chunk should consist of just a heading with no content."""
        if not PROCESSED_DIR.exists():
            import pytest
            pytest.skip("Test data not available")

        preprocessor = DocumentPreprocessor()
        docs = preprocessor.process_directory()

        for doc in docs:
            lines = [l for l in doc.page_content.split("\n") if l.strip()]
            # A chunk should never be just heading lines
            all_headings = all(l.strip().startswith("#") for l in lines)
            if all_headings and len(lines) <= 2:
                assert False, (
                    f"Chunk {doc.metadata['chunk_id']} in {doc.metadata['source']} "
                    f"is only headings: {doc.page_content[:100]}"
                )
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/test_block_chunking.py::TestIntegration -v`
Expected: All 4 tests PASS (or skip if data not present).

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass (existing + new).

- [ ] **Step 4: Commit**

```bash
git add tests/test_block_chunking.py
git commit -m "test: add integration tests for block-aware chunking"
```

---

## Known Deferred Work

These items are out of scope for this plan but should be addressed in a follow-up:

**Downstream consumers of removed metadata fields:**
The following files reference `ckd_stages`, `page_number`, or `document_type` in chunk metadata. After this change, `.get()` calls will return `None`/`[]` (no runtime errors), but filtering methods like `search_by_ckd_stage()` and `search_by_document_type()` in `vectorstore.py` will return empty results for newly-indexed documents.

- `1_Retrieval_Augmented_Generation/vectorstore.py` — `search_by_ckd_stage`, `search_by_document_type` methods
- `1_Retrieval_Augmented_Generation/retriever.py` — `document_type` and `ckd_stages` filter params
- `1_Retrieval_Augmented_Generation/chain.py:232` — references `page_number`
- `2_Agentic_RAG/nodes.py:208` — references `page_number`
- `3_MultiAgent_RAG/agents/rag_agent.py:111-130` — references `page_number` and `ckd_stages`
- `tests/conftest.py:35-37` — mock `MockDocument` includes removed fields (cosmetic, no test failures)

**`test_docling_preprocessing.py`:** This root-level file imports the old `DocumentPreprocessor` API and will break after the rewrite. It is not in `tests/` so `uv run pytest` won't pick it up. It should be updated or removed in a follow-up.
