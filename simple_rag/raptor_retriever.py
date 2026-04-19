"""
RAPTOR collapsed retriever for the CKD RAG System.

Query-time component: performs flat top-k similarity search over all
RAPTOR tree layers (leaves + summaries) in a single ChromaDB collection.
"""

import logging
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIRECTORY,
    RAPTOR_COLLECTION_NAME,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


class RaptorRetriever(BaseRetriever):
    """Collapsed retrieval over all RAPTOR tree layers.

    Searches leaves and summaries in a single flat index.
    Higher-level summaries match broad thematic queries;
    leaf nodes match specific detail queries.
    """

    vectorstore: Any  # CKDVectorStore or Chroma-backed store
    embedding_function: Any
    k: int = TOP_K_RESULTS
    score_threshold: float = SIMILARITY_THRESHOLD

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Flat top-k similarity search over all RAPTOR layers.

        Args:
            query: User query.
            run_manager: Callback manager.

        Returns:
            Top-k documents above score threshold, from any tree layer.
        """
        results = self.vectorstore.search_with_scores(
            query=query,
            k=self.k,
        )

        docs = [
            doc for doc, score in results
            if score >= self.score_threshold
        ]

        # Enforce k limit (the vectorstore may return more if k is not honoured)
        docs = docs[: self.k]

        logger.info(
            f"RAPTOR retrieval: {len(docs)}/{len(results)} docs "
            f"(threshold={self.score_threshold})"
        )
        return docs


def create_raptor_retriever(
    embedding_function: Embeddings,
    collection_name: str = RAPTOR_COLLECTION_NAME,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    k: int = TOP_K_RESULTS,
    score_threshold: float = SIMILARITY_THRESHOLD,
) -> RaptorRetriever:
    """Factory function to create a RAPTOR retriever.

    Creates a CKDVectorStore-compatible wrapper around the RAPTOR
    ChromaDB collection.

    Args:
        embedding_function: Embeddings model.
        collection_name: ChromaDB collection name for RAPTOR nodes.
        persist_directory: ChromaDB persistence directory.
        k: Number of results to return.
        score_threshold: Minimum similarity score.

    Returns:
        Configured RaptorRetriever.
    """
    from simple_rag.vectorstore import CKDVectorStore

    store = CKDVectorStore(
        embedding_function=embedding_function,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    return RaptorRetriever(
        vectorstore=store,
        embedding_function=embedding_function,
        k=k,
        score_threshold=score_threshold,
    )
