"""
Build section trees and create the section heading ChromaDB collection.

This script:
1. Builds section trees for all documents (Data/tree_builder.py)
2. Creates a small ChromaDB collection with one entry per section heading
   for use by the TreeRetriever's section-routing phase.

Usage:
    uv run python scripts/build_section_index.py
    uv run python scripts/build_section_index.py --rebuild  # force rebuild trees
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from config import (
    CHROMA_PERSIST_DIRECTORY,
    SECTION_COLLECTION_NAME,
    get_embeddings,
)
from Data.tree_builder import build_all_trees, load_tree, get_section_path, get_section_numbering

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed_with_sections"


def collect_section_documents() -> list[Document]:
    """Walk all document trees and create one Document per section node.

    Each document contains:
    - page_content: heading text + first 200 chars of section content
    - metadata: doc_name, source, section_numbering, depth, line_start, line_end, section_path
    """
    docs = []

    doc_dirs = sorted([
        d for d in PROCESSED_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    for doc_dir in doc_dirs:
        roots = load_tree(doc_dir)
        if not roots:
            continue

        # Load metadata for source file name
        meta_path = doc_dir / "metadata.json"
        source = doc_dir.name
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            source = meta.get("source_file", doc_dir.name)

        for root in roots:
            for node in root.flatten():
                # Build page content: heading + content preview
                content_preview = node.content[:200] if node.content else ""
                page_content = node.heading
                if content_preview:
                    page_content = f"{node.heading}\n\n{content_preview}"

                section_numbering = get_section_numbering(node)
                section_path = get_section_path(node)

                docs.append(Document(
                    page_content=page_content,
                    metadata={
                        "doc_name": doc_dir.name,
                        "source": source,
                        "section_numbering": section_numbering,
                        "section_path": section_path,
                        "depth": node.depth,
                        "line_start": node.line_start,
                        "line_end": node.line_end,
                    },
                ))

    return docs


def build_section_collection(docs: list[Document]):
    """Create/rebuild the section heading ChromaDB collection."""
    import chromadb
    from chromadb.config import Settings
    from langchain_chroma import Chroma

    embeddings = get_embeddings()

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIRECTORY,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

    # Delete existing collection if present
    try:
        client.delete_collection(SECTION_COLLECTION_NAME)
        logger.info(f"Deleted existing '{SECTION_COLLECTION_NAME}' collection")
    except Exception:
        pass

    # Create new collection via LangChain wrapper
    vectorstore = Chroma(
        client=client,
        collection_name=SECTION_COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Generate unique IDs
    ids = [
        f"{doc.metadata['doc_name']}_{doc.metadata['section_numbering']}_{i}"
        for i, doc in enumerate(docs)
    ]

    # Add in batches
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vectorstore.add_documents(batch_docs, ids=batch_ids)
        logger.info(f"  Added batch {i // batch_size + 1}: {len(batch_docs)} sections")

    logger.info(f"Section collection created: {len(docs)} entries in '{SECTION_COLLECTION_NAME}'")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Build section heading index for tree retrieval")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild all section trees")
    args = parser.parse_args()

    # Step 1: Build trees
    existing_trees = list(PROCESSED_DIR.glob("*/section_tree.json"))
    if args.rebuild or not existing_trees:
        logger.info("Building section trees...")
        build_all_trees()
    else:
        logger.info(f"Found {len(existing_trees)} existing trees (use --rebuild to regenerate)")

    # Step 2: Collect section documents
    logger.info("Collecting section headings from trees...")
    docs = collect_section_documents()
    logger.info(f"Collected {len(docs)} section headings")

    if not docs:
        logger.error("No sections found. Run tree builder first.")
        sys.exit(1)

    # Step 3: Build ChromaDB collection
    logger.info("Building section heading collection...")
    build_section_collection(docs)

    logger.info("Done!")


if __name__ == "__main__":
    main()
