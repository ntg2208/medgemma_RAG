#!/bin/bash
# Process PDFs with PaddleOCR-VL
# Usage: ./ocr-process.sh [input_dir] [output_dir]

set -e

INPUT_DIR=${1:-~/medgemma_RAG/Data/documents}
OUTPUT_DIR=${2:-~/medgemma_RAG/Data/processed_ocr}

mkdir -p "$OUTPUT_DIR"

echo "═══════════════════════════════════════"
echo "PaddleOCR-VL PDF Processing"
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
if [ -f "~/medgemma_RAG/.venv/bin/activate" ]; then
  source ~/medgemma_RAG/.venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "Warning: Virtual environment not found. Using system Python."
fi

# Check if paddleocr is installed
if ! python -c "from paddleocr import PaddleOCRVL" 2>/dev/null; then
  echo "Error: paddleocr not installed. Run: pip install \"paddleocr[doc-parser]\""
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
import json
from pathlib import Path
from paddleocr import PaddleOCRVL

try:
    pipeline = PaddleOCRVL()
    page_results = list(pipeline.predict("$pdf"))

    # Restructure: merge tables across pages, fix heading levels, concatenate
    merged = pipeline.restructure_pages(
        page_results,
        merge_tables=True,
        relevel_titles=True,
        concatenate_pages=True,
    )

    out_dir = Path("$OUTPUT_DIR")

    md_parts = []
    json_parts = []
    for res in merged:
        md_data = res.markdown
        if isinstance(md_data, dict):
            md_parts.append(md_data.get("markdown_texts", str(md_data)))
        else:
            md_parts.append(str(md_data))

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

    full_md = "\n\n".join(md_parts)

    # Extract title from first heading
    title = "${filename}"
    for line in full_md.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped.lstrip("#").strip()
            break

    # Write markdown
    md_path = out_dir / "${filename}.md"
    md_path.write_text(full_md, encoding="utf-8")
    print(f"    {md_path.name}")

    # Write JSON
    json_path = out_dir / "${filename}.json"
    json_path.write_text(
        json.dumps({"title": title, "source_file": "$(basename "$pdf")", "pages": json_parts}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"    {json_path.name}")

except Exception as e:
    print(f"    Error: {e}")
EOF
    echo ""
  fi
done

echo "═══════════════════════════════════════"
echo "PaddleOCR-VL processing complete!"
echo "Output files in: $OUTPUT_DIR"
echo "═══════════════════════════════════════"
