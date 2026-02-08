#!/usr/bin/env python3
"""Process a single PDF with Docling OCR."""
import argparse
import json
import sys
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
except ImportError:
    print("Error: docling not installed. Run: pip install docling")
    sys.exit(1)


def process_pdf(input_path: str, output_dir: str, use_ocr: bool = True) -> dict:
    """Process PDF and extract structured content.

    Args:
        input_path: Path to input PDF
        output_dir: Directory for output files
        use_ocr: Whether to use OCR for scanned PDFs

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
    print(f"OCR: {'enabled' if use_ocr else 'disabled'}")
    print("")

    # Configure pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = use_ocr
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    # Convert
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        pdf_pipeline_options=pipeline_options,
    )

    print("Converting document...")
    result = converter.convert(str(input_path))

    # Export formats
    stem = input_path.stem
    outputs = {}

    # Markdown (good for RAG chunking)
    md_path = output_dir / f"{stem}.md"
    md_path.write_text(result.document.export_to_markdown())
    outputs['markdown'] = str(md_path)
    print(f"✓ Markdown: {md_path}")

    # JSON (structured data)
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(result.document.export_to_dict(), indent=2))
    outputs['json'] = str(json_path)
    print(f"✓ JSON:     {json_path}")

    # Plain text
    txt_path = output_dir / f"{stem}.txt"
    txt_path.write_text(result.document.export_to_text())
    outputs['text'] = str(txt_path)
    print(f"✓ Text:     {txt_path}")

    print(f"\n✓ Successfully processed {input_path.name}")
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Process PDF with Docling OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with OCR
  python ocr-single.py document.pdf -o ./output

  # Process without OCR
  python ocr-single.py document.pdf -o ./output --no-ocr

  # Process to specific directory
  python ocr-single.py /path/to/doc.pdf -o /path/to/output
        """
    )
    parser.add_argument("input", help="Input PDF path")
    parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR for scanned PDFs"
    )

    args = parser.parse_args()

    try:
        process_pdf(args.input, args.output, use_ocr=not args.no_ocr)
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
