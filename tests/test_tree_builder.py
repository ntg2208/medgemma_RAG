"""Tests for the tree builder module."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from Data.tree_builder import (
    SectionNode,
    build_tree,
    classify_heading,
    extract_headings,
    find_section_for_line,
    get_section_numbering,
    get_section_path,
)


# ============================================================================
# classify_heading tests
# ============================================================================


class TestClassifyHeading:
    """Test heading classification into depth and numbering."""

    def test_chapter_heading(self):
        depth, num = classify_heading("CHAPTER 3. USE OF ESAs")
        assert depth == 0
        assert num == "3"

    def test_chapter_case_insensitive(self):
        depth, num = classify_heading("Chapter 1. Diagnosis")
        assert depth == 0
        assert num == "1"

    def test_appendix(self):
        depth, num = classify_heading("APPENDIX A. POPULATION-BASED ALGORITHMS")
        assert depth == 0
        assert num == "Appendix A"

    def test_section_heading(self):
        depth, num = classify_heading("Section 1: Community")
        assert depth == 0
        assert num == "1"

    def test_dotted_two_levels(self):
        depth, num = classify_heading("3.1. Treatment initiation")
        assert depth == 1
        assert num == "3.1"

    def test_dotted_three_levels(self):
        depth, num = classify_heading("3.4.1. ESA dosing")
        assert depth == 2
        assert num == "3.4.1"

    def test_guideline_prefix(self):
        depth, num = classify_heading("Guideline 1.2.1 - Monitoring after episode")
        assert depth >= 2
        assert num == "1.2.1"

    def test_practice_point(self):
        depth, num = classify_heading("Practice Point 2.10: something")
        assert depth >= 1
        assert num == "2.10"

    def test_recommendation(self):
        depth, num = classify_heading("Recommendation 3.2.1: something")
        assert depth >= 2
        assert num == "3.2.1"

    def test_top_level_number(self):
        depth, num = classify_heading("1. SCOPE")
        assert depth == 0
        assert num == "1"

    def test_roman_numeral(self):
        depth, num = classify_heading("(iii) Potassium", parent_depth=0)
        assert depth == 1
        assert num is None

    def test_letter_sub_item(self):
        depth, num = classify_heading("(B) Transplantation", parent_depth=0)
        assert depth == 1
        assert num is None

    def test_table_heading(self):
        depth, num = classify_heading("Table 5 | Circumstances warranting...", parent_depth=1)
        assert depth == 2
        assert num is None

    def test_unnumbered_heading(self):
        depth, num = classify_heading("Protein")
        assert depth == 0
        assert num is None

    def test_step_heading(self):
        depth, num = classify_heading("Step 1")
        assert depth == 1
        assert num is None


# ============================================================================
# extract_headings tests
# ============================================================================


class TestExtractHeadings:
    """Test heading extraction from markdown lines."""

    def test_basic_extraction(self):
        lines = [
            "## Introduction",
            "Some text",
            "## Methods",
            "More text",
        ]
        headings = extract_headings(lines)
        assert len(headings) == 2
        assert headings[0] == (0, "Introduction")
        assert headings[1] == (2, "Methods")

    def test_level_one_heading(self):
        lines = ["# Title", "## Section"]
        headings = extract_headings(lines)
        assert len(headings) == 2

    def test_no_headings(self):
        lines = ["Just text", "More text"]
        headings = extract_headings(lines)
        assert len(headings) == 0

    def test_ignores_deeper_headings(self):
        """### and deeper are not extracted (documents only use ## )."""
        lines = ["## Valid", "### Not extracted", "#### Also not"]
        headings = extract_headings(lines)
        assert len(headings) == 1


# ============================================================================
# build_tree tests
# ============================================================================


class TestBuildTree:
    """Test tree construction from markdown."""

    def test_flat_document(self):
        """Document with only unnumbered headings should be flat."""
        lines = [
            "## Protein",
            "Some content about protein",
            "",
            "## Potassium",
            "Some content about potassium",
        ]
        roots = build_tree(lines)
        assert len(roots) == 2
        assert roots[0].heading == "Protein"
        assert roots[1].heading == "Potassium"
        assert roots[0].children == []

    def test_nested_hierarchy(self):
        """Dotted numbering should create parent-child relationships."""
        lines = [
            "## CHAPTER 3. USE OF ESAs",
            "Chapter intro",
            "",
            "## 3.1. Treatment initiation",
            "Treatment content",
            "",
            "## 3.2. ESA initiation",
            "ESA content",
            "",
            "## 3.4. ESA dosing, frequency",
            "",
            "## 3.4.1. ESA dosing",
            "Dosing content",
            "",
            "## 3.4.2. ESA route",
            "Route content",
        ]
        roots = build_tree(lines)
        assert len(roots) == 1  # Single chapter as root

        chapter = roots[0]
        assert chapter.heading == "CHAPTER 3. USE OF ESAs"
        assert chapter.depth == 0
        assert len(chapter.children) == 3  # 3.1, 3.2, 3.4

        section_34 = chapter.children[2]
        assert section_34.heading == "3.4. ESA dosing, frequency"
        assert len(section_34.children) == 2  # 3.4.1, 3.4.2

        subsection = section_34.children[0]
        assert subsection.heading == "3.4.1. ESA dosing"
        assert subsection.depth == 2
        assert subsection.ancestor_path == [
            "CHAPTER 3. USE OF ESAs",
            "3.4. ESA dosing, frequency",
        ]

    def test_line_ranges(self):
        """Nodes should have correct line_start and line_end."""
        lines = [
            "## A",     # line 0
            "content",  # line 1
            "",         # line 2
            "## B",     # line 3
            "content",  # line 4
        ]
        roots = build_tree(lines)
        assert roots[0].line_start == 0
        assert roots[0].line_end == 3
        assert roots[1].line_start == 3
        assert roots[1].line_end == 5

    def test_content_extraction(self):
        """Node content should contain text between heading and first child."""
        lines = [
            "## CHAPTER 1. Intro",
            "Chapter intro text",
            "More intro",
            "",
            "## 1.1. Details",
            "Detail text",
        ]
        roots = build_tree(lines)
        chapter = roots[0]
        assert "Chapter intro text" in chapter.content
        assert "More intro" in chapter.content
        # Child content should NOT be in parent content
        assert "Detail text" not in chapter.content

    def test_empty_lines(self):
        roots = build_tree([])
        assert roots == []

    def test_flatten(self):
        """flatten() should return all nodes in the tree."""
        lines = [
            "## CHAPTER 1. Root",
            "## 1.1. Child",
            "## 1.1.1. Grandchild",
        ]
        roots = build_tree(lines)
        flat = roots[0].flatten()
        assert len(flat) == 3


# ============================================================================
# find_section_for_line tests
# ============================================================================


class TestFindSectionForLine:
    """Test finding the most specific section for a line number."""

    def test_finds_leaf_section(self):
        lines = [
            "## CHAPTER 1. Root",    # 0
            "root content",          # 1
            "## 1.1. Child",         # 2
            "child content line 1",  # 3
            "child content line 2",  # 4
        ]
        roots = build_tree(lines)
        node = find_section_for_line(roots, 3)
        assert node is not None
        assert node.heading == "1.1. Child"

    def test_finds_parent_when_before_child(self):
        lines = [
            "## CHAPTER 1. Root",  # 0
            "root content",        # 1
            "## 1.1. Child",       # 2
            "child content",       # 3
        ]
        roots = build_tree(lines)
        node = find_section_for_line(roots, 1)
        assert node is not None
        assert node.heading == "CHAPTER 1. Root"

    def test_returns_none_for_out_of_range(self):
        lines = ["## A", "content"]
        roots = build_tree(lines)
        node = find_section_for_line(roots, 100)
        assert node is None


# ============================================================================
# Helper function tests
# ============================================================================


class TestHelpers:
    """Test get_section_path and get_section_numbering."""

    def test_section_path(self):
        node = SectionNode(
            heading="3.4.1. ESA dosing",
            depth=2,
            numbering="3.4.1",
            line_start=0,
            ancestor_path=["CHAPTER 3", "3.4. ESA dosing, frequency"],
        )
        path = get_section_path(node)
        assert path == "CHAPTER 3|3.4. ESA dosing, frequency|3.4.1. ESA dosing"

    def test_section_numbering_with_number(self):
        node = SectionNode(heading="3.4.1. ESA dosing", depth=2, numbering="3.4.1", line_start=0)
        assert get_section_numbering(node) == "3.4.1"

    def test_section_numbering_fallback(self):
        node = SectionNode(heading="Protein", depth=0, numbering=None, line_start=0)
        assert get_section_numbering(node) == "Protein"

    def test_to_dict_roundtrip(self):
        """to_dict should produce valid JSON."""
        node = SectionNode(
            heading="Test",
            depth=0,
            numbering="1",
            line_start=0,
            line_end=10,
            content="Some content",
            ancestor_path=["Parent"],
        )
        d = node.to_dict()
        assert d["heading"] == "Test"
        assert d["numbering"] == "1"
        assert d["ancestor_path"] == ["Parent"]
        # Should be JSON serializable
        json.dumps(d)
