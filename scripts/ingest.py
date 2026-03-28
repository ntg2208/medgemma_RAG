"""Ingest all processed chunks into ChromaDB vectorstore."""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from config import get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"


def load_chunks(processed_dir: Path) -> list[Document]:
    """Load all *_chunks.json files into LangChain Documents."""
    docs = []
    chunk_files = sorted(processed_dir.glob("*_chunks.json"))
    if not chunk_files:
        logger.error(f"No *_chunks.json files found in {processed_dir}")
        return docs

    for f in chunk_files:
        data = json.loads(f.read_text())
        meta = data.get("export_metadata", {})
        title = meta.get("document_title", f.stem)
        source = meta.get("source_file", f.name)

        chunks = data.get("chunks", [])
        for i, chunk in enumerate(chunks):
            chunk_meta = chunk.get("metadata", {})
            chunk_meta.update({
                "source": source,
                "document_title": title,
                "chunk_id": i,
            })
            docs.append(Document(
                page_content=chunk["content"],
                metadata=chunk_meta,
            ))
        logger.info(f"  {f.name}: {len(chunks)} chunks")

    return docs


def main():
    # Load chunks
    logger.info(f"Loading chunks from {PROCESSED_DIR}")
    docs = load_chunks(PROCESSED_DIR)
    if not docs:
        sys.exit(1)
    logger.info(f"Total: {len(docs)} chunks from {len(list(PROCESSED_DIR.glob('*_chunks.json')))} files")

    # Init embeddings + vectorstore
    logger.info("Loading embedding model...")
    embeddings = get_embeddings()

    # Import vectorstore (handles numeric prefix)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rag_pkg.vectorstore",
        PROJECT_ROOT / "1_Retrieval_Augmented_Generation" / "vectorstore.py",
    )
    vs_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vs_module)

    # Reset and re-ingest
    store = vs_module.CKDVectorStore(embeddings)
    stats = store.get_collection_stats()
    if stats["document_count"] > 0:
        logger.info(f"Existing collection has {stats['document_count']} docs — resetting...")
        store.reset()

    logger.info(f"Embedding and storing {len(docs)} chunks...")
    ids = store.add_documents(docs)
    logger.info(f"Done! Stored {len(ids)} documents in ChromaDB")

    # Verify
    stats = store.get_collection_stats()
    logger.info(f"Final stats: {stats}")


if __name__ == "__main__":
    main()
