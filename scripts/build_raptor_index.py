#!/usr/bin/env python
"""Build the RAPTOR index from existing processed chunks.

Usage:
    uv run python scripts/build_raptor_index.py
    uv run python scripts/build_raptor_index.py --max-depth 2
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_chunks(processed_dir: Path):
    """Load all chunk JSON files into LangChain Documents."""
    from langchain_core.documents import Document

    docs = []
    for f in sorted(processed_dir.glob("*_chunks.json")):
        data = json.loads(f.read_text())
        meta = data.get("export_metadata", {})
        title = meta.get("document_title", f.stem)
        source = meta.get("source_file", f.name)
        for i, chunk in enumerate(data.get("chunks", [])):
            chunk_meta = chunk.get("metadata", {})
            chunk_meta.update({"source": source, "document_title": title, "chunk_id": i})
            docs.append(Document(page_content=chunk["content"], metadata=chunk_meta))
    return docs


def main():
    parser = argparse.ArgumentParser(description="Build RAPTOR index")
    parser.add_argument("--max-depth", type=int, default=3, help="Max tree depth")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    processed_dir = PROJECT_ROOT / "Data" / "processed"
    logger.info(f"Loading chunks from {processed_dir}...")
    chunks = load_chunks(processed_dir)
    logger.info(f"Loaded {len(chunks)} chunks")

    if not chunks:
        logger.error("No chunks found. Run the data pipeline first.")
        sys.exit(1)

    from config import get_llm, get_embeddings
    from simple_rag.raptor_builder import RaptorBuilder
    from simple_rag.vectorstore import CKDVectorStore
    from config import RAPTOR_COLLECTION_NAME

    logger.info("Loading embedding model...")
    embeddings = get_embeddings()

    logger.info("Loading LLM...")
    llm = get_llm()

    builder = RaptorBuilder(
        embedding_function=embeddings,
        llm=llm,
        max_depth=args.max_depth,
    )

    logger.info("Building RAPTOR tree...")
    tree = builder.build(chunks)
    logger.info(f"Tree built: {len(tree.nodes)} nodes, depth={tree.depth}")

    logger.info(f"Indexing into ChromaDB collection '{RAPTOR_COLLECTION_NAME}'...")
    store = CKDVectorStore(
        embedding_function=embeddings,
        collection_name=RAPTOR_COLLECTION_NAME,
    )
    store.delete_collection()
    docs = tree.all_documents()
    store.add_documents(docs)
    logger.info(f"Indexed {len(docs)} nodes. Done.")


if __name__ == "__main__":
    main()
