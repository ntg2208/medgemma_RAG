#!/usr/bin/env python
"""Build the Contextual RAG index from existing processed chunks.

Usage:
    uv run python scripts/build_contextual_index.py
"""

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


def load_document_texts(sections_dir: Path) -> dict[str, str]:
    """Load full document texts from processed_with_sections/*/main_text.md."""
    texts = {}
    for doc_dir in sorted(sections_dir.iterdir()):
        if not doc_dir.is_dir() or doc_dir.name.startswith("."):
            continue
        main_text = doc_dir / "main_text.md"
        metadata_file = doc_dir / "metadata.json"
        if main_text.exists() and metadata_file.exists():
            meta = json.loads(metadata_file.read_text())
            source = meta.get("source_file", "")
            if source:
                texts[source] = main_text.read_text()
    return texts


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    processed_dir = PROJECT_ROOT / "Data" / "processed"
    sections_dir = PROJECT_ROOT / "Data" / "processed_with_sections"

    logger.info(f"Loading chunks from {processed_dir}...")
    chunks = load_chunks(processed_dir)
    logger.info(f"Loaded {len(chunks)} chunks")

    if not chunks:
        logger.error("No chunks found. Run the data pipeline first.")
        sys.exit(1)

    logger.info(f"Loading document texts from {sections_dir}...")
    doc_texts = load_document_texts(sections_dir)
    logger.info(f"Loaded {len(doc_texts)} document texts")

    from config import get_llm, get_embeddings, CONTEXTUAL_COLLECTION_NAME
    from simple_rag.contextual_builder import ContextualBuilder
    from simple_rag.vectorstore import CKDVectorStore

    logger.info("Loading embedding model...")
    embeddings = get_embeddings()

    logger.info("Loading LLM...")
    llm = get_llm()

    builder = ContextualBuilder(embedding_function=embeddings, llm=llm)
    logger.info("Generating context for all chunks (this may take a while)...")
    ctx_chunks = builder.build(chunks, doc_texts)

    logger.info(f"Indexing into ChromaDB collection '{CONTEXTUAL_COLLECTION_NAME}'...")
    store = CKDVectorStore(
        embedding_function=embeddings,
        collection_name=CONTEXTUAL_COLLECTION_NAME,
    )
    store.delete_collection()
    store.add_documents(ctx_chunks)
    logger.info(f"Indexed {len(ctx_chunks)} contextualized chunks. Done.")


if __name__ == "__main__":
    main()
