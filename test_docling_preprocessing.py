#!/usr/bin/env python3
"""
Test script for Docling-based preprocessing.

Tests the updated preprocessing.py with Docling integration.
"""

import logging
from pathlib import Path

from Data.preprocessing import DocumentPreprocessor, preprocess_documents

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_single_pdf():
    """Test processing a single PDF file."""
    logger.info("=" * 60)
    logger.info("Test 1: Single PDF Processing with Docling")
    logger.info("=" * 60)

    # Find first PDF in documents directory
    docs_dir = Path("Data/documents")
    pdf_files = list(docs_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error("No PDF files found in Data/documents/")
        return False

    test_pdf = pdf_files[0]
    logger.info(f"Testing with: {test_pdf.name}")

    # Create preprocessor with Docling
    preprocessor = DocumentPreprocessor(
        chunk_size=512,  # Smaller for testing
        chunk_overlap=50,
        use_ocr=True,
        export_markdown=True,
    )

    # Process single PDF
    output_dir = Path("Data/test_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        documents = preprocessor.process_pdf(test_pdf, output_dir=output_dir)

        logger.info(f"\n✓ Successfully processed {test_pdf.name}")
        logger.info(f"  - Created {len(documents)} chunks")

        if documents:
            logger.info(f"  - First chunk length: {len(documents[0].page_content)} chars")
            logger.info(f"  - Metadata keys: {list(documents[0].metadata.keys())}")

        # Check if markdown was exported
        md_path = output_dir / f"{test_pdf.stem}.md"
        if md_path.exists():
            logger.info(f"  - Markdown exported: {md_path.name} ({md_path.stat().st_size} bytes)")
        else:
            logger.warning(f"  - Markdown NOT found at {md_path}")

        return True

    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_batch_processing():
    """Test batch processing with markdown export."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Batch Processing with Markdown Export")
    logger.info("=" * 60)

    output_dir = Path("Data/test_output")

    try:
        documents = preprocess_documents(
            input_dir=Path("Data/documents"),
            chunk_size=512,
            chunk_overlap=50,
            use_ocr=True,
            export_markdown=True,
        )

        logger.info(f"\n✓ Batch processing complete")
        logger.info(f"  - Total documents: {len(documents)}")

        # Check unique sources
        sources = set(d.metadata.get("source") for d in documents)
        logger.info(f"  - Unique sources: {len(sources)}")

        # Check markdown files
        md_files = list(output_dir.glob("*.md"))
        logger.info(f"  - Markdown files created: {len(md_files)}")
        for md_file in md_files[:3]:  # Show first 3
            logger.info(f"    • {md_file.name}")

        return True

    except Exception as e:
        logger.error(f"✗ Batch processing failed: {e}", exc_info=True)
        return False


def test_comparison():
    """Compare Docling output quality."""
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Quality Check")
    logger.info("=" * 60)

    output_dir = Path("Data/test_output")
    md_files = list(output_dir.glob("*.md"))

    if not md_files:
        logger.warning("No markdown files to check")
        return True

    # Check first markdown file
    md_file = md_files[0]
    content = md_file.read_text(encoding='utf-8')

    logger.info(f"Checking {md_file.name}:")
    logger.info(f"  - File size: {len(content)} chars")
    logger.info(f"  - Lines: {len(content.splitlines())}")

    # Check for markdown structures
    has_headers = '##' in content or '###' in content
    has_tables = '|' in content
    has_lists = any(line.strip().startswith(('- ', '* ', '1. ')) for line in content.splitlines())

    logger.info(f"  - Has headers: {has_headers}")
    logger.info(f"  - Has tables: {has_tables}")
    logger.info(f"  - Has lists: {has_lists}")

    # Show preview
    preview_lines = content.splitlines()[:10]
    logger.info("\nPreview (first 10 lines):")
    for line in preview_lines:
        logger.info(f"  {line[:100]}")

    return True


def main():
    """Run all tests."""
    logger.info("\n" + "=" * 60)
    logger.info("Docling Preprocessing Test Suite")
    logger.info("=" * 60 + "\n")

    results = []

    # Test 1: Single PDF
    results.append(("Single PDF", test_single_pdf()))

    # Test 2: Batch processing
    # Uncomment if you want to test batch processing (slower)
    # results.append(("Batch Processing", test_batch_processing()))

    # Test 3: Quality check
    results.append(("Quality Check", test_comparison()))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status} - {test_name}")

    all_passed = all(result[1] for result in results)
    logger.info("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
