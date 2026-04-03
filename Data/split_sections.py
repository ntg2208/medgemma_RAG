"""
Split cleaned clinical guideline documents into 3 files per document:
- main_text.md: clinical guidelines, recommendations, appendices
- metadata.json: title, source, front matter text, processing stats
- references.md: all reference/bibliography sections and inline citation blocks

Usage:
    uv run python Data/split_sections.py
    uv run python Data/split_sections.py --no-llm  # fallback heuristics only
"""

import json
import logging
import re
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CLEANED_DIR = Path(__file__).parent / "cleaned_documents"
OUTPUT_DIR = Path(__file__).parent / "processed_with_sections"

VALID_TYPES = {"front_matter", "main_content", "references", "end_matter"}


# ---------------------------------------------------------------------------
# Heading extraction
# ---------------------------------------------------------------------------

_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+")


def extract_headings(lines: list[str]) -> list[dict]:
    """Extract all markdown headings (any level) with 1-based line numbers."""
    headings = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _HEADING_LINE_RE.match(stripped):
            headings.append({"line": i + 1, "text": stripped})
    return headings


# ---------------------------------------------------------------------------
# LLM classification
# ---------------------------------------------------------------------------

def build_classification_prompt(title: str, headings: list[dict]) -> str:
    heading_lines = "\n".join(
        f"Line {h['line']}: {h['text']}" for h in headings
    )
    return f"""You are analyzing a clinical guideline document structure.

Document title: {title}

Here are the section headings with their line numbers:
{heading_lines}

Classify each heading as ONE of:
- front_matter (authors, dates, version, methods, acknowledgements, table of contents, disclaimers, conflicts of interest, grading systems)
- main_content (clinical guidelines, recommendations, chapters, summaries, appendices, definitions, background, introduction, overview, audit measures, rationale)
- references (bibliography sections, numbered citation lists)
- end_matter (abbreviation lists at the very end, "finding more information", "update information", copyright notices)

Respond with ONLY a JSON array (no other text):
[{{"line": <line_number>, "type": "<classification>"}}, ...]

Include every heading listed above."""


def classify_with_llm(llm, title: str, headings: list[dict]) -> list[dict] | None:
    """Call MedGemma to classify headings. Returns None on failure."""
    if not headings:
        return []

    prompt = build_classification_prompt(title, headings)
    try:
        response = llm.generate(prompt)
        return parse_llm_response(response, headings)
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None


def parse_llm_response(response: str, headings: list[dict]) -> list[dict] | None:
    """Extract JSON array from LLM response. Returns None if unparseable."""
    match = re.search(r"\[[\s\S]*\]", response)
    if not match:
        return None

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None

    valid_lines = {h["line"] for h in headings}
    result = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        line = item.get("line")
        type_ = item.get("type", "").strip()
        if line in valid_lines and type_ in VALID_TYPES:
            result.append({"line": line, "type": type_})

    # Fill in any missing headings as main_content
    classified_lines = {r["line"] for r in result}
    for h in headings:
        if h["line"] not in classified_lines:
            result.append({"line": h["line"], "type": "main_content"})

    return result if result else None


# ---------------------------------------------------------------------------
# Heuristic boundary detection
# ---------------------------------------------------------------------------

# References section heading — broad matching for clinical documents
# Matches: ## References, ## 7. References, ## References (Non-dialysis CKD),
# ## REFERENCES, ## Reference, ## References:, ## Useful websites and reading,
# ## Acknowledgments, ## Acknowledgements, ## Further reading, ## Bibliography
REFERENCES_KEYWORD = re.compile(
    r"^#{1,6}\s*(?:\d+[\.\)]\s*)?"  # any heading level + optional numbered prefix
    r"(?:"
    r"references?(?:\s*:|\s*\(.*\))?"  # References, Reference, References:, References (...)
    r"|useful\s+websites"               # Useful websites and reading
    r"|further\s+reading"               # Further reading
    r"|bibliography"                     # Bibliography
    r"|patient\s+information\s+and\s+resources"  # Patient Information and Resources
    r")\s*.*$",                          # allow trailing text
    re.IGNORECASE,
)

# Acknowledgments — separate pattern so we can detect but not mix with main refs
_ACKNOWLEDGMENTS_KEYWORD = re.compile(
    r"^#{1,6}\s*(?:\d+[\.\)]\s*)?acknowledge?ments?\s*$",
    re.IGNORECASE,
)

# Headings that signal "end of front matter — main content starts at NEXT heading"
_FRONT_MATTER_BOUNDARY = re.compile(
    r"^#{1,6}\s*(contents?|table\s+of\s+contents?|guideline\s+clinical\s+content)",
    re.IGNORECASE,
)

# Headings that are definitively the LAST front-matter section
# (when no TOC is present, main content starts at the next heading after the last of these)
_LAST_FRONT_MATTER = re.compile(
    r"^#{1,6}\s*(conflicts?\s+of\s+interest|acknowledgements?|"
    r"method\s+used|endorsements?|grading\s+the\s+evidence|"
    r"notice\s*$|work\s+group|public\s+review)",
    re.IGNORECASE,
)

# End matter patterns (only checked near end of document)
_END_MATTER = re.compile(
    r"^#{1,6}\s*(finding\s+more\s+information|update\s+information|"
    r"abbreviations?\s+and\s+acronyms?\s*$|abbreviations?\s*$)",
    re.IGNORECASE,
)


def find_main_content_start(headings: list[dict]) -> int:
    """Return the 1-based line number where main content begins."""
    # 1. Explicit TOC / guideline-clinical-content boundary
    for i, h in enumerate(headings):
        if _FRONT_MATTER_BOUNDARY.match(h["text"]):
            return headings[i + 1]["line"] if i + 1 < len(headings) else h["line"]

    # 2. No TOC — find the LAST known front-matter heading; main starts at the next heading
    last_fm_idx = -1
    for i, h in enumerate(headings):
        if _LAST_FRONT_MATTER.match(h["text"]):
            last_fm_idx = i

    if last_fm_idx >= 0 and last_fm_idx + 1 < len(headings):
        return headings[last_fm_idx + 1]["line"]

    # 3. Entire document is main content
    return 1


def find_end_matter_start(headings: list[dict], total_lines: int) -> int | None:
    """Return the 1-based line number where end matter begins, or None."""
    for h in headings:
        position = h["line"] / max(total_lines, 1)
        if position >= 0.7 and _END_MATTER.match(h["text"]):
            return h["line"]
    return None


def classify_with_heuristics(headings: list[dict], total_lines: int) -> list[dict]:
    """Boundary-based heuristic classification."""
    if not headings:
        return []

    main_start = find_main_content_start(headings)
    end_matter_start = find_end_matter_start(headings, total_lines)

    result = []
    for h in headings:
        line = h["line"]
        text = h["text"]

        if REFERENCES_KEYWORD.match(text) or _ACKNOWLEDGMENTS_KEYWORD.match(text):
            type_ = "references"
        elif end_matter_start is not None and line >= end_matter_start:
            type_ = "end_matter"
        elif line < main_start:
            type_ = "front_matter"
        else:
            type_ = "main_content"

        result.append({"line": line, "type": type_})

    return result


# ---------------------------------------------------------------------------
# Document splitting
# ---------------------------------------------------------------------------

def get_section_ranges(
    headings: list[dict], classifications: list[dict], total_lines: int
) -> dict:
    """
    Compute 0-based line ranges for each section (end exclusive).
    Content before the first heading is assigned to front_matter.
    """
    type_map = {c["line"]: c["type"] for c in classifications}

    ranges: dict[str, list[tuple[int, int]]] = {
        "front_matter": [],
        "main_content": [],
        "references": [],
        "end_matter": [],
    }

    # Lines before the first heading go to front_matter
    if headings and headings[0]["line"] > 1:
        ranges["front_matter"].append((0, headings[0]["line"] - 1))

    for i, h in enumerate(headings):
        start = h["line"] - 1  # 0-based
        end = headings[i + 1]["line"] - 1 if i + 1 < len(headings) else total_lines
        type_ = type_map.get(h["line"], "main_content")
        ranges[type_].append((start, end))

    return ranges


def extract_lines(lines: list[str], ranges: list[tuple[int, int]]) -> list[str]:
    result = []
    for start, end in ranges:
        result.extend(lines[start:end])
    return result


# ---------------------------------------------------------------------------
# Inline reference block stripping
# ---------------------------------------------------------------------------

def strip_reference_blocks(lines: list[str]) -> tuple[list[str], list[str]]:
    """
    Remove numbered citation blocks under ## References headings.

    A reference block is a ## References heading followed (possibly after blank
    lines) by numbered citations like '1. Author...' until the next ## heading.

    Returns (cleaned_lines, stripped_lines).
    """
    cleaned = []
    stripped = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped_line = line.strip()

        if REFERENCES_KEYWORD.match(stripped_line) or _ACKNOWLEDGMENTS_KEYWORD.match(stripped_line):
            # Strip the entire section until next heading
            stripped.append(line)
            i += 1
            while i < len(lines):
                curr = lines[i].strip()
                if _HEADING_LINE_RE.match(curr):
                    break
                stripped.append(lines[i])
                i += 1
        else:
            cleaned.append(line)
            i += 1

    return cleaned, stripped


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_existing_metadata(json_path: Path) -> dict:
    if json_path.exists():
        try:
            with json_path.open() as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def extract_title_from_lines(lines: list[str]) -> str:
    for line in lines:
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped
    return "Unknown"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_document(md_path: Path, llm=None) -> dict:
    """Process a single markdown document. Returns a report dict."""
    doc_name = md_path.stem
    out_dir = OUTPUT_DIR / doc_name
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = md_path.read_text(encoding="utf-8").splitlines(keepends=True)
    total_lines = len(lines)

    json_path = md_path.with_suffix(".json")
    existing_meta = load_existing_metadata(json_path)
    title = existing_meta.get("title") or extract_title_from_lines(lines)

    headings = extract_headings(lines)

    method_used = "heuristics"
    classifications = None

    if llm and headings:
        logger.info(f"  LLM classifying {len(headings)} headings...")
        classifications = classify_with_llm(llm, title, headings)
        if classifications:
            method_used = "llm"
        else:
            logger.warning("  LLM failed, falling back to heuristics")

    if not classifications:
        classifications = classify_with_heuristics(headings, total_lines)

    ranges = get_section_ranges(headings, classifications, total_lines)

    front_lines = extract_lines(lines, ranges["front_matter"])
    main_lines = extract_lines(lines, ranges["main_content"])
    ref_section_lines = extract_lines(lines, ranges["references"])
    end_lines = extract_lines(lines, ranges["end_matter"])

    # Strip any reference blocks that slipped into main content
    main_lines_clean, inline_ref_lines = strip_reference_blocks(main_lines)

    # Combine all reference-like content (inline refs + classified ref sections + end matter)
    all_ref_lines = inline_ref_lines + ref_section_lines + end_lines

    # --- Output exactly 3 files ---

    # 1. main_text.md
    (out_dir / "main_text.md").write_text("".join(main_lines_clean), encoding="utf-8")

    # 2. references.md (always created, even if empty)
    (out_dir / "references.md").write_text(
        "".join(all_ref_lines) if all_ref_lines else "No references found.\n",
        encoding="utf-8",
    )

    # 3. metadata.json (includes front matter text and processing stats)
    meta_out = {
        "title": title,
        "source_file": md_path.name,
        **{k: v for k, v in existing_meta.items() if k not in ("title",)},
        "front_matter": "".join(front_lines).strip() if front_lines else "",
        "processing": {
            "method": method_used,
            "total_lines": total_lines,
            "main_text_lines": len(main_lines_clean),
            "references_lines": len(all_ref_lines),
            "front_matter_lines": len(front_lines),
        },
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Clean up old files from previous runs
    for old_file in ("main_content.md", "front_matter.md", "split_report.json"):
        old_path = out_dir / old_file
        if old_path.exists():
            old_path.unlink()

    report = {
        "filename": md_path.name,
        "title": title,
        "method": method_used,
        "total_lines": total_lines,
        "main_text_lines": len(main_lines_clean),
        "references_lines": len(all_ref_lines),
        "front_matter_lines": len(front_lines),
    }

    return report


def generate_review_summary(reports: list[dict]) -> str:
    out = [
        "# Document Split Review Summary\n\n",
        "| Document | Method | Total | Main | Refs | Front |\n",
        "|----------|--------|-------|------|------|-------|\n",
    ]
    for r in sorted(reports, key=lambda x: x["filename"]):
        name = r["filename"].replace(".md", "")[:55]
        out.append(
            f"| {name} | {r['method']} | {r['total_lines']} "
            f"| {r['main_text_lines']} | {r['references_lines']} "
            f"| {r['front_matter_lines']} |\n"
        )

    seen: dict[int, list[str]] = {}
    for r in reports:
        seen.setdefault(r["total_lines"], []).append(r["filename"])

    dupes = [(cnt, names) for cnt, names in seen.items() if len(names) > 1]
    if dupes:
        out.append("\n## Potential Duplicates (same line count)\n\n")
        for count, names in dupes:
            out.append(f"- {count} lines: {', '.join(names)}\n")

    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description="Split cleaned documents into sections")
    parser.add_argument("--no-llm", action="store_true", help="Use heuristics only")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    md_files = sorted(CLEANED_DIR.glob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files in {CLEANED_DIR}")

    llm = None
    if not args.no_llm:
        logger.info("Loading MedGemma LLM...")
        try:
            import importlib
            chain_mod = importlib.import_module("simple_rag.chain")
            llm = chain_mod.MedGemmaLLM(max_new_tokens=1024, temperature=0.1)
            logger.info("MedGemma loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load MedGemma ({e}), using heuristics for all docs")

    reports = []
    for i, md_path in enumerate(md_files, 1):
        logger.info(f"[{i}/{len(md_files)}] Processing: {md_path.name}")
        try:
            report = process_document(md_path, llm=llm)
            reports.append(report)
            logger.info(
                f"  -> main: {report['main_text_lines']}, "
                f"refs: {report['references_lines']}, "
                f"front: {report['front_matter_lines']}, "
                f"method: {report['method']}"
            )
        except Exception as e:
            logger.error(f"  ERROR processing {md_path.name}: {e}")
            import traceback
            traceback.print_exc()

    summary = generate_review_summary(reports)
    (OUTPUT_DIR / "review_summary.md").write_text(summary, encoding="utf-8")
    logger.info(f"\nDone. Output in: {OUTPUT_DIR}")
    logger.info(f"Review summary: {OUTPUT_DIR / 'review_summary.md'}")


if __name__ == "__main__":
    main()
