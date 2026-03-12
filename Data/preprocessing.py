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


# Regex for list items: 1. , 1.1 , 1.1.1 , bullets, asterisks
_LIST_PATTERN = re.compile(r"^(\d+(?:\.\d+)*\.?\s+|- |\* )")
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
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def pack_chunks(
    blocks: list[Block],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    overlap_token_cap: int = 150,
) -> list[dict]:
    """Pack blocks into chunks without splitting any block."""
    raise NotImplementedError("Implemented in Task 5")


class DocumentPreprocessor:
    """Preprocessor that reads pre-processed markdown and chunks using block-aware algorithm."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_directory(self, input_dir: Optional[Path] = None) -> list[Document]:
        raise NotImplementedError("Implemented in Task 6")

    def process_document(self, doc_dir: Path) -> list[Document]:
        raise NotImplementedError("Implemented in Task 6")

    def get_document_stats(self, documents: list[Document]) -> dict:
        raise NotImplementedError("Implemented in Task 6")


def preprocess_documents(
    input_dir: Optional[Path] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Convenience function to preprocess all documents in a directory."""
    raise NotImplementedError("Implemented in Task 6")


if __name__ == "__main__":
    pass
