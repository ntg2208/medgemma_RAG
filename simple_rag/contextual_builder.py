"""
Contextual RAG builder for the CKD RAG System.

Implements Anthropic's Contextual Retrieval technique:
for each chunk, generates a short LLM context paragraph that explains
where the chunk sits in the document, then prepends it before embedding.

Also builds a BM25 index over the contextualized text for hybrid retrieval.
"""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CONTEXTUAL_COLLECTION_NAME,
    CONTEXTUAL_BM25_PATH,
)

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """<document>
{document_text}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


class ContextualBuilder:
    """Builds contextualized chunk index for Contextual RAG."""

    def __init__(
        self,
        embedding_function: Embeddings,
        llm: Any,
        collection_name: str = CONTEXTUAL_COLLECTION_NAME,
        bm25_path: str = CONTEXTUAL_BM25_PATH,
    ):
        self.embedding_function = embedding_function
        self.llm = llm
        self.collection_name = collection_name
        self.bm25_path = bm25_path

    def generate_context(self, document_text: str, chunk: Document) -> str:
        """Generate contextual description for a single chunk."""
        prompt = CONTEXT_PROMPT.format(
            document_text=document_text,
            chunk_content=chunk.page_content,
        )
        return self.llm.generate(prompt)

    def contextualize_chunk(self, document_text: str, chunk: Document) -> Document:
        """Generate context and prepend it to the chunk."""
        context = self.generate_context(document_text, chunk)
        new_chunk = deepcopy(chunk)
        new_chunk.page_content = f"{context}\n\n{chunk.page_content}"
        new_chunk.metadata["contextual_context"] = context
        return new_chunk

    def build(self, chunks: list[Document], document_texts: dict[str, str]) -> list[Document]:
        """Contextualize all chunks and build BM25 index."""
        contextualized = []
        bm25_corpus = []
        bm25_chunk_ids = []

        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "")
            doc_text = document_texts.get(source, "")

            if not doc_text:
                logger.warning(f"No document text for source '{source}', skipping context generation for chunk {i}")
                ctx_chunk = deepcopy(chunk)
                ctx_chunk.metadata["contextual_context"] = ""
            else:
                ctx_chunk = self.contextualize_chunk(doc_text, chunk)

            contextualized.append(ctx_chunk)
            bm25_corpus.append(ctx_chunk.page_content)
            bm25_chunk_ids.append(f"{source}_{chunk.metadata.get('chunk_id', i)}")

            if (i + 1) % 50 == 0:
                logger.info(f"Contextualized {i + 1}/{len(chunks)} chunks")

        save_bm25_data(bm25_corpus, bm25_chunk_ids, self.bm25_path)
        logger.info(f"Contextual build complete: {len(contextualized)} chunks, BM25 saved to {self.bm25_path}")
        return contextualized


def save_bm25_data(corpus: list[str], chunk_ids: list[str], path: str) -> None:
    """Save BM25 corpus and chunk IDs to JSON for persistence."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {"corpus": corpus, "chunk_ids": chunk_ids}
    Path(path).write_text(json.dumps(data))
    logger.info(f"BM25 data saved: {len(corpus)} entries → {path}")


def load_bm25_data(path: str) -> tuple[list[str], list[str]]:
    """Load BM25 corpus and chunk IDs from JSON."""
    p = Path(path)
    if not p.exists():
        return [], []
    data = json.loads(p.read_text())
    return data.get("corpus", []), data.get("chunk_ids", [])
