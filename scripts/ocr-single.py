#!/usr/bin/env python3
"""Process a single PDF with PaddleOCR-VL."""
import argparse
import json
import sys
from pathlib import Path

try:
    from paddleocr import PaddleOCRVL
except ImportError:
    print(
        'Error: paddleocr not installed. Run:\n'
        '  pip install "paddleocr[doc-parser]"\n'
        'Also ensure PaddlePaddle >= 3.2.1 is installed.'
    )
    sys.exit(1)


def process_pdf(input_path: str, output_dir: str, device: str = None) -> dict:
    """Process PDF and extract structured content using PaddleOCR-VL.

    Args:
        input_path: Path to input PDF
        output_dir: Directory for output files
        device: Device for inference (e.g. "gpu:0", "cpu"). None = auto-detect.

    Returns:
        dict with paths to generated files
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Processing: {input_path}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device or 'auto'}")
    print("")

    # Configure pipeline
    kwargs = {}
    if device:
        kwargs["device"] = device

    pipeline = PaddleOCRVL(**kwargs)

    print("Converting document...")
    page_results = list(pipeline.predict(str(input_path)))

    # Restructure multi-page results:
    #  - merge_tables: merge tables split across pages
    #  - relevel_titles: reconstruct multi-level heading hierarchy
    #  - concatenate_pages: merge all pages into a single result
    merged = pipeline.restructure_pages(
        page_results,
        merge_tables=True,
        relevel_titles=True,
        concatenate_pages=True,
    )

    stem = input_path.stem
    outputs = {}

    # Collect markdown and JSON from merged results
    md_parts = []
    json_parts = []
    text_parts = []

    for res in merged:
        # Markdown output (preserves headings, tables, lists)
        md_data = res.markdown
        if isinstance(md_data, dict):
            md_parts.append(md_data.get("markdown_texts", str(md_data)))
        else:
            md_parts.append(str(md_data))

        # JSON output (structured data)
        json_data = res.json
        if isinstance(json_data, dict):
            json_parts.append(json_data)
        elif isinstance(json_data, str):
            try:
                json_parts.append(json.loads(json_data))
            except json.JSONDecodeError:
                json_parts.append({"raw": json_data})
        else:
            json_parts.append({"raw": str(json_data)})

    # Build combined markdown
    full_md = "\n\n".join(md_parts)

    # Build combined JSON with title extraction
    title = _extract_title(full_md, stem)
    combined_json = {
        "title": title,
        "source_file": input_path.name,
        "pages": json_parts,
    }

    # Build plain text (strip markdown formatting)
    full_text = _markdown_to_plain_text(full_md)

    # Write outputs
    md_path = output_dir / f"{stem}.md"
    md_path.write_text(full_md, encoding="utf-8")
    outputs["markdown"] = str(md_path)
    print(f"  Markdown: {md_path}")

    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(combined_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    outputs["json"] = str(json_path)
    print(f"  JSON:     {json_path}")

    txt_path = output_dir / f"{stem}.txt"
    txt_path.write_text(full_text, encoding="utf-8")
    outputs["text"] = str(txt_path)
    print(f"  Text:     {txt_path}")

    print(f"\nSuccessfully processed {input_path.name}")
    return outputs


def _extract_title(md_text: str, fallback: str) -> str:
    """Extract title from first heading in markdown, or use fallback."""
    for line in md_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.lstrip("#").strip()
    return fallback


def _markdown_to_plain_text(md_text: str) -> str:
    """Simple markdown-to-plain-text conversion."""
    lines = []
    for line in md_text.splitlines():
        # Strip heading markers
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        # Strip table separators
        if stripped and all(c in "-| " for c in stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Process PDF with PaddleOCR-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF
  python ocr-single.py document.pdf -o ./output

  # Process on CPU
  python ocr-single.py document.pdf -o ./output --device cpu

  # Process on specific GPU
  python ocr-single.py document.pdf -o ./output --device gpu:0
        """,
    )
    parser.add_argument("input", help="Input PDF path")
    parser.add_argument(
        "-o",
        "--output",
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference: gpu, gpu:0, cpu (default: auto-detect)",
    )

    args = parser.parse_args()

    try:
        process_pdf(args.input, args.output, device=args.device)
        return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
