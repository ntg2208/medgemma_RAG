"""
Build a section tree from markdown headings in clinical guideline documents.

Parses the implicit hierarchy encoded in heading numbering patterns
(e.g., "## 3.4.1. ESA dosing") and constructs a tree structure that
maps the document's logical organization.

Usage:
    uv run python Data/tree_builder.py                  # build all
    uv run python Data/tree_builder.py --doc KDIGO_2025  # single doc (substring match)
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent / "processed_with_sections"

# ---------------------------------------------------------------------------
# Heading classification patterns (priority order — first match wins)
# ---------------------------------------------------------------------------

# CHAPTER 3. USE OF ESAs... → depth 0
_CHAPTER_RE = re.compile(
    r"^CHAPTER\s+(\d+)", re.IGNORECASE
)

# Section 1: Community → depth 0
_SECTION_RE = re.compile(
    r"^Section\s+(\d+)", re.IGNORECASE
)

# Guideline 1.2.1 - ... → depth from dot count + 1
_GUIDELINE_RE = re.compile(
    r"^Guideline\s+(\d+(?:\.\d+)*)", re.IGNORECASE
)

# Practice Point 2.10: ... → depth from dot count + 1
_PRACTICE_PT_RE = re.compile(
    r"^Practice\s+Point\s+(\d+(?:\.\d+)*)", re.IGNORECASE
)

# Recommendation 3.2.1: ... → depth from dot count + 1
_RECOMMENDATION_RE = re.compile(
    r"^Recommendation\s+(\d+(?:\.\d+)*)", re.IGNORECASE
)

# 3.4.1. ESA dosing → depth = number of dots
_DOTTED_NUM_RE = re.compile(
    r"^(\d+(?:\.\d+)+)\b"
)

# Single top-level number: 1. SCOPE, 6. DETAILS → depth 0
_TOP_NUM_RE = re.compile(
    r"^(\d+)\.\s"
)

# (iii) Potassium, (iv) Phosphate → depth = parent + 1
_ROMAN_RE = re.compile(
    r"^\(([ivxlcIVXLC]+)\)"
)

# (B) Transplantation → depth = parent + 1
_LETTER_RE = re.compile(
    r"^\(([A-Z])\)"
)

# Table 5 | ..., Figure 3 ... → depth = parent + 1
_TABLE_FIG_RE = re.compile(
    r"^(?:Table|Figure)\s+\d+", re.IGNORECASE
)

# Step 1, Step 2 → depth 1
_STEP_RE = re.compile(
    r"^Step\s+(\d+)", re.IGNORECASE
)

# Appendix A, APPENDIX B → depth 0
_APPENDIX_RE = re.compile(
    r"^APPENDIX\s+([A-Z])", re.IGNORECASE
)


def _count_dots(numbering: str) -> int:
    """Count the number of dot separators in a numbering string."""
    return numbering.count(".")


def classify_heading(heading_text: str, parent_depth: int = -1) -> tuple[int, Optional[str]]:
    """Classify a heading and return (depth, numbering_or_None).

    Args:
        heading_text: The heading text with ## prefix stripped.
        parent_depth: Depth of the current parent node (-1 for root).

    Returns:
        Tuple of (depth, numbering). Numbering is the extracted section
        number (e.g., "3.4.1") or None for unnumbered headings.
    """
    text = heading_text.strip()

    # CHAPTER N → depth 0
    m = _CHAPTER_RE.match(text)
    if m:
        return 0, m.group(1)

    # APPENDIX A → depth 0
    m = _APPENDIX_RE.match(text)
    if m:
        return 0, f"Appendix {m.group(1)}"

    # Section N → depth 0
    m = _SECTION_RE.match(text)
    if m:
        return 0, m.group(1)

    # Guideline N.N.N → depth from dot count + 1
    m = _GUIDELINE_RE.match(text)
    if m:
        num = m.group(1)
        return _count_dots(num) + 1, num

    # Practice Point N.N.N → depth from dot count + 1
    m = _PRACTICE_PT_RE.match(text)
    if m:
        num = m.group(1)
        return _count_dots(num) + 1, num

    # Recommendation N.N.N → depth from dot count + 1
    m = _RECOMMENDATION_RE.match(text)
    if m:
        num = m.group(1)
        return _count_dots(num) + 1, num

    # Dotted number: 3.4.1 → depth = dot count
    m = _DOTTED_NUM_RE.match(text)
    if m:
        num = m.group(1).rstrip(".")
        return _count_dots(num), num

    # Single top-level number: "1. SCOPE" → depth 0
    m = _TOP_NUM_RE.match(text)
    if m:
        return 0, m.group(1)

    # (iii), (iv) roman → parent + 1
    m = _ROMAN_RE.match(text)
    if m:
        return max(parent_depth + 1, 1), None

    # (B) letter → parent + 1
    m = _LETTER_RE.match(text)
    if m:
        return max(parent_depth + 1, 1), None

    # Table / Figure → parent + 1
    m = _TABLE_FIG_RE.match(text)
    if m:
        return max(parent_depth + 1, 1), None

    # Step N → depth 1
    m = _STEP_RE.match(text)
    if m:
        return 1, None

    # Unnumbered heading → depth 0 (flat sibling)
    return 0, None


# ---------------------------------------------------------------------------
# Tree data structure
# ---------------------------------------------------------------------------

@dataclass
class SectionNode:
    """A node in the document section tree."""
    heading: str                    # Raw heading text (without ## prefix)
    depth: int                      # 0 = chapter/top-level, 1 = section, 2 = subsection, ...
    numbering: Optional[str]        # Extracted number like "3.4.1" or None
    line_start: int                 # 0-based line number where heading appears
    line_end: int = 0               # 0-based line number where section ends (exclusive)
    content: str = ""               # Text content under this heading (before children)
    children: list["SectionNode"] = field(default_factory=list)
    ancestor_path: list[str] = field(default_factory=list)  # Heading texts of ancestors
    content_token_estimate: int = 0

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        d = {
            "heading": self.heading,
            "depth": self.depth,
            "numbering": self.numbering,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content_preview": self.content[:200] if self.content else "",
            "content_token_estimate": self.content_token_estimate,
            "ancestor_path": self.ancestor_path,
        }
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    def flatten(self) -> list["SectionNode"]:
        """Return this node and all descendants as a flat list."""
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Heading extraction
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$")


def extract_headings(lines: list[str]) -> list[tuple[int, str]]:
    """Extract (line_number, heading_text) pairs from markdown lines.

    Args:
        lines: List of lines from main_text.md.

    Returns:
        List of (0-based line index, heading text without ## prefix).
    """
    headings = []
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line.strip())
        if m:
            headings.append((i, m.group(1).strip()))
    return headings


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

def build_tree(lines: list[str]) -> list[SectionNode]:
    """Build a section tree from markdown lines.

    Uses a stack-based algorithm: for each heading, pop the stack until
    we find the parent (a node with depth < current heading's depth),
    then add the new node as a child.

    Args:
        lines: Lines from a main_text.md file.

    Returns:
        List of root-level SectionNode objects.
    """
    headings = extract_headings(lines)
    if not headings:
        return []

    total_lines = len(lines)
    root_children: list[SectionNode] = []

    # Stack entries: (depth, node)
    # Sentinel root at depth -1
    stack: list[tuple[int, Optional[SectionNode]]] = [(-1, None)]

    nodes_in_order: list[SectionNode] = []

    for idx, (line_num, heading_text) in enumerate(headings):
        parent_depth = stack[-1][0]
        depth, numbering = classify_heading(heading_text, parent_depth)

        node = SectionNode(
            heading=heading_text,
            depth=depth,
            numbering=numbering,
            line_start=line_num,
        )
        nodes_in_order.append(node)

        # Pop stack until we find the parent
        while stack[-1][0] >= depth:
            stack.pop()

        parent_node = stack[-1][1]
        if parent_node is not None:
            parent_node.children.append(node)
            node.ancestor_path = parent_node.ancestor_path + [parent_node.heading]
        else:
            root_children.append(node)
            node.ancestor_path = []

        stack.append((depth, node))

    # Assign line_end: initially to next heading, then propagate children's end to parents
    for i, node in enumerate(nodes_in_order):
        if i + 1 < len(nodes_in_order):
            node.line_end = nodes_in_order[i + 1].line_start
        else:
            node.line_end = total_lines

    # Propagate: parent line_end = max of its own line_end and all children's line_end
    # Process in reverse document order so children are finalized before parents
    def _propagate_line_end(node: SectionNode):
        for child in node.children:
            _propagate_line_end(child)
        if node.children:
            node.line_end = max(node.line_end, max(c.line_end for c in node.children))

    for root in root_children:
        _propagate_line_end(root)

    # Assign content for each node (text between heading and first child)
    for node in nodes_in_order:
        content_end = node.children[0].line_start if node.children else node.line_end
        content_lines = []
        for ln in range(node.line_start + 1, content_end):
            content_lines.append(lines[ln])

        node.content = "\n".join(content_lines).strip()
        node.content_token_estimate = _estimate_tokens(node.content)

    return root_children


# ---------------------------------------------------------------------------
# Build and save tree for a document
# ---------------------------------------------------------------------------

def build_document_tree(doc_dir: Path) -> Optional[dict]:
    """Build and save the section tree for a single document directory.

    Args:
        doc_dir: Path to document directory containing main_text.md.

    Returns:
        Tree dictionary, or None if no headings found.
    """
    main_text = doc_dir / "main_text.md"
    if not main_text.exists():
        logger.warning(f"No main_text.md in {doc_dir.name}")
        return None

    lines = main_text.read_text(encoding="utf-8").splitlines()
    if not lines:
        logger.warning(f"Empty main_text.md in {doc_dir.name}")
        return None

    roots = build_tree(lines)
    if not roots:
        logger.warning(f"No headings found in {doc_dir.name}")
        return None

    # Count total sections
    total = sum(len(r.flatten()) for r in roots)

    tree_data = {
        "doc_name": doc_dir.name,
        "total_sections": total,
        "total_lines": len(lines),
        "tree": [r.to_dict() for r in roots],
    }

    # Save alongside the document
    output_path = doc_dir / "section_tree.json"
    output_path.write_text(
        json.dumps(tree_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"  {doc_dir.name}: {total} sections, max depth {_max_depth(roots)}")

    return tree_data


def _max_depth(nodes: list[SectionNode]) -> int:
    """Find maximum depth in a tree."""
    if not nodes:
        return -1
    return max(
        max(node.depth, _max_depth(node.children))
        for node in nodes
    )


def build_all_trees(input_dir: Optional[Path] = None) -> list[dict]:
    """Build section trees for all documents.

    Args:
        input_dir: Directory containing document subdirectories.

    Returns:
        List of tree dictionaries.
    """
    input_dir = input_dir or PROCESSED_DIR

    doc_dirs = sorted([
        d for d in input_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    logger.info(f"Building trees for {len(doc_dirs)} documents...")

    trees = []
    for doc_dir in doc_dirs:
        tree = build_document_tree(doc_dir)
        if tree:
            trees.append(tree)

    logger.info(f"Built {len(trees)} trees")
    return trees


# ---------------------------------------------------------------------------
# Lookup helpers (used by preprocessing and retriever)
# ---------------------------------------------------------------------------

def load_tree(doc_dir: Path) -> Optional[list[SectionNode]]:
    """Load a section tree from JSON and reconstruct SectionNode objects.

    Args:
        doc_dir: Path to document directory.

    Returns:
        List of root SectionNode objects, or None if no tree file exists.
    """
    tree_path = doc_dir / "section_tree.json"
    if not tree_path.exists():
        return None

    data = json.loads(tree_path.read_text(encoding="utf-8"))
    return [_node_from_dict(d) for d in data.get("tree", [])]


def _node_from_dict(d: dict) -> SectionNode:
    """Reconstruct a SectionNode from a dictionary."""
    children = [_node_from_dict(c) for c in d.get("children", [])]
    return SectionNode(
        heading=d["heading"],
        depth=d["depth"],
        numbering=d.get("numbering"),
        line_start=d["line_start"],
        line_end=d["line_end"],
        content=d.get("content_preview", ""),
        children=children,
        ancestor_path=d.get("ancestor_path", []),
        content_token_estimate=d.get("content_token_estimate", 0),
    )


def find_section_for_line(roots: list[SectionNode], line_num: int) -> Optional[SectionNode]:
    """Find the most specific (deepest) section containing a given line number.

    Args:
        roots: Root nodes of the section tree.
        line_num: 0-based line number.

    Returns:
        The deepest SectionNode containing the line, or None.
    """
    best = None
    for root in roots:
        node = _find_deepest(root, line_num)
        if node is not None:
            if best is None or node.depth > best.depth:
                best = node
    return best


def _find_deepest(node: SectionNode, line_num: int) -> Optional[SectionNode]:
    """Recursively find the deepest node containing line_num."""
    if not (node.line_start <= line_num < node.line_end):
        return None

    # Check children first (they are more specific)
    for child in node.children:
        result = _find_deepest(child, line_num)
        if result is not None:
            return result

    return node


def get_section_path(node: SectionNode) -> str:
    """Get pipe-delimited ancestor path for a section node.

    Returns:
        e.g., "CHAPTER 3. USE OF ESAs|3.4. ESA dosing|3.4.1. ESA dosing"
    """
    parts = list(node.ancestor_path) + [node.heading]
    return "|".join(parts)


def get_section_numbering(node: SectionNode) -> str:
    """Get the section numbering, falling back to heading text.

    Returns:
        The numbering string (e.g., "3.4.1") or the heading text for
        unnumbered sections.
    """
    return node.numbering if node.numbering else node.heading


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build section trees from document headings")
    parser.add_argument("--doc", type=str, help="Substring match for a single document name")
    args = parser.parse_args()

    if args.doc:
        doc_dirs = [
            d for d in PROCESSED_DIR.iterdir()
            if d.is_dir() and args.doc.lower() in d.name.lower()
        ]
        if not doc_dirs:
            logger.error(f"No document matching '{args.doc}'")
            sys.exit(1)
        for doc_dir in doc_dirs:
            tree = build_document_tree(doc_dir)
            if tree:
                print(json.dumps(tree, indent=2)[:3000])
    else:
        trees = build_all_trees()
        print(f"\nSummary: {len(trees)} trees built")
        for t in trees:
            print(f"  {t['doc_name']}: {t['total_sections']} sections")


if __name__ == "__main__":
    main()
