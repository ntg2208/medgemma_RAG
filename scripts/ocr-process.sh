#!/bin/bash
# Process PDFs with Docling OCR
# Usage: ./ocr-process.sh [input_dir] [output_dir]

set -e

INPUT_DIR=${1:-/data/medgemma_RAG/Data/documents}
OUTPUT_DIR=${2:-/data/medgemma_RAG/Data/processed_ocr}

mkdir -p "$OUTPUT_DIR"

echo "═══════════════════════════════════════"
echo "Docling PDF OCR Processing"
echo "═══════════════════════════════════════"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if directory exists
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory does not exist: $INPUT_DIR"
  exit 1
fi

# Activate environment
if [ -f "/data/medgemma_RAG/.venv/bin/activate" ]; then
  source /data/medgemma_RAG/.venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Warning: Virtual environment not found. Using system Python."
fi

# Check if docling is installed
if ! python -c "import docling" 2>/dev/null; then
  echo "Error: docling not installed. Run: pip install docling"
  exit 1
fi

# Count PDFs
pdf_count=$(find "$INPUT_DIR" -maxdepth 1 -name "*.pdf" -type f | wc -l)
echo "Found $pdf_count PDF files"
echo ""

# Process each PDF
counter=0
for pdf in "$INPUT_DIR"/*.pdf; do
  if [ -f "$pdf" ]; then
    counter=$((counter + 1))
    filename=$(basename "$pdf" .pdf)
    echo "[$counter/$pdf_count] Processing: $filename"

    python << EOF
from docling.document_converter import DocumentConverter
from pathlib import Path
import json

try:
    converter = DocumentConverter()
    result = converter.convert("$pdf")

    # Export as markdown
    md_path = Path("$OUTPUT_DIR") / "${filename}.md"
    md_path.write_text(result.document.export_to_markdown())

    # Export as JSON (structured)
    json_path = Path("$OUTPUT_DIR") / "${filename}.json"
    json_path.write_text(json.dumps(result.document.export_to_dict(), indent=2))

    print(f"  ✓ {md_path.name}")
    print(f"  ✓ {json_path.name}")
except Exception as e:
    print(f"  ✗ Error: {e}")
EOF
    echo ""
  fi
done

echo "═══════════════════════════════════════"
echo "✓ OCR processing complete!"
echo "Output files in: $OUTPUT_DIR"
echo "═══════════════════════════════════════"
