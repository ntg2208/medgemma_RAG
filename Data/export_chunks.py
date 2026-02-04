"""
Export processed document chunks to JSON files.

Creates one JSON file per source PDF with all chunks, metadata, and statistics.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR
from Data.preprocessing import DocumentPreprocessor

logger = logging.getLogger(__name__)


def calculate_chunk_stats(text: str) -> dict:
    """
    Calculate statistics for a text chunk.

    Args:
        text: The chunk text

    Returns:
        Dictionary with character count, word count, and token estimate
    """
    char_count = len(text)
    word_count = len(text.split())
    # Rough token estimate: ~4 chars per token
    token_estimate = char_count // 4

    return {
        "character_count": char_count,
        "word_count": word_count,
        "estimated_tokens": token_estimate,
    }


def prepare_chunk_data(document: Document, export_timestamp: str) -> dict:
    """
    Prepare a Document object for JSON export.

    Args:
        document: LangChain Document object
        export_timestamp: ISO format timestamp of export

    Returns:
        Dictionary ready for JSON serialization
    """
    chunk_data = {
        "content": document.page_content,
        "metadata": {
            "source": document.metadata.get("source"),
            "title": document.metadata.get("title"),
            "chunk_id": document.metadata.get("chunk_id"),
            "total_chunks": document.metadata.get("total_chunks"),
            "page_number": document.metadata.get("page_number"),
            "section": document.metadata.get("section"),
            "ckd_stages": document.metadata.get("ckd_stages", []),
            "document_type": document.metadata.get("document_type"),
        },
        "statistics": calculate_chunk_stats(document.page_content),
        "processed_at": export_timestamp,
    }

    return chunk_data


def export_to_json(
    documents: list[Document],
    output_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Export processed documents to JSON files (one per source PDF).

    Args:
        documents: List of processed Document objects
        output_dir: Directory to save JSON files (default: PROCESSED_DIR)

    Returns:
        Dictionary mapping source filenames to output file paths
    """
    output_dir = Path(output_dir) if output_dir else PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not documents:
        logger.warning("No documents to export")
        return {}

    # Group documents by source
    docs_by_source = {}
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(doc)

    logger.info(f"Exporting chunks from {len(docs_by_source)} source documents")

    export_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    exported_files = {}

    for source, source_docs in docs_by_source.items():
        # Create clean filename from source
        base_name = Path(source).stem  # Remove .pdf extension
        output_filename = f"{base_name}_chunks.json"
        output_path = output_dir / output_filename

        # Prepare export data
        export_data = {
            "export_metadata": {
                "source_file": source,
                "export_timestamp": export_timestamp,
                "total_chunks": len(source_docs),
                "document_title": source_docs[0].metadata.get("title") if source_docs else None,
            },
            "chunks": [
                prepare_chunk_data(doc, export_timestamp)
                for doc in sorted(source_docs, key=lambda d: d.metadata.get("chunk_id", 0))
            ],
        }

        # Write to JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(source_docs)} chunks to {output_path.name}")
        exported_files[source] = output_path

    logger.info(f"Successfully exported all chunks to {output_dir}")
    return exported_files


def main():
    """
    Main entry point for the export script.

    Processes all PDFs in the documents directory and exports chunks to JSON.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("Starting document chunk export...")

    # Process documents
    preprocessor = DocumentPreprocessor()
    documents = preprocessor.process_directory()

    if not documents:
        logger.error("No documents were processed. Check if PDFs exist in documents directory.")
        return

    # Export to JSON
    exported_files = export_to_json(documents)

    # Print summary
    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"\nTotal source documents: {len(exported_files)}")
    print(f"Total chunks exported: {len(documents)}")
    print(f"\nOutput directory: {PROCESSED_DIR}")
    print("\nExported files:")
    for source, path in exported_files.items():
        print(f"  - {path.name}")
    print("\n" + "="*60)

    # Show sample chunk structure
    if documents:
        print("\nSample chunk structure:")
        sample_chunk = prepare_chunk_data(documents[0], datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        # Show structure without full content
        sample_structure = {
            "content": f"{sample_chunk['content'][:100]}..." if len(sample_chunk['content']) > 100 else sample_chunk['content'],
            "metadata": sample_chunk['metadata'],
            "statistics": sample_chunk['statistics'],
            "processed_at": sample_chunk['processed_at'],
        }
        print(json.dumps(sample_structure, indent=2))


if __name__ == "__main__":
    main()
