"""
Contextual RAG hybrid retriever for the CKD RAG System.

Combines semantic search over contextualized embeddings with BM25
keyword search using reciprocal rank fusion (RRF).
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
    CONTEXTUAL_COLLECTION_NAME,
    CONTEXTUAL_BM25_PATH,
    CONTEXTUAL_SEMANTIC_WEIGHT,
    CONTEXTUAL_BM25_WEIGHT,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

RRF_K = 60

# Module-level import so the name is patchable in tests
from simple_rag.vectorstore import CKDVectorStore  # noqa: E402


class ContextualRetriever(BaseRetriever):
    """Hybrid semantic + BM25 retrieval over contextualized chunks."""

    vectorstore: Any
    embedding_function: Any
    bm25_index: Any = None  # BM25Okapi or None
    bm25_corpus_ids: list = []
    bm25_doc_map: dict = {}  # chunk_id -> Document
    k: int = TOP_K_RESULTS
    semantic_weight: float = CONTEXTUAL_SEMANTIC_WEIGHT
    bm25_weight: float = CONTEXTUAL_BM25_WEIGHT
    score_threshold: float = SIMILARITY_THRESHOLD

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """Hybrid retrieval with reciprocal rank fusion."""
        fetch_k = self.k * 3
        semantic_results = self.vectorstore.search_with_scores(query=query, k=fetch_k)

        semantic_results = [
            (doc, score) for doc, score in semantic_results
            if score >= self.score_threshold
        ]

        if not semantic_results:
            return []

        if self.bm25_index is None:
            return [doc for doc, _ in semantic_results[:self.k]]

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        import numpy as np
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]

        # Reciprocal rank fusion
        doc_rrf: dict[str, tuple[Document, float]] = {}

        for rank, (doc, score) in enumerate(semantic_results, start=1):
            doc_key = self._doc_key(doc)
            rrf_score = self.semantic_weight / (rank + RRF_K)
            if doc_key in doc_rrf:
                existing_doc, existing_score = doc_rrf[doc_key]
                doc_rrf[doc_key] = (existing_doc, existing_score + rrf_score)
            else:
                doc_rrf[doc_key] = (doc, rrf_score)

        for rank, idx in enumerate(top_bm25_indices, start=1):
            if idx >= len(self.bm25_corpus_ids):
                continue
            chunk_id = self.bm25_corpus_ids[idx]
            if chunk_id not in self.bm25_doc_map:
                continue
            doc = self.bm25_doc_map[chunk_id]
            doc_key = self._doc_key(doc)
            rrf_score = self.bm25_weight / (rank + RRF_K)
            if doc_key in doc_rrf:
                existing_doc, existing_score = doc_rrf[doc_key]
                doc_rrf[doc_key] = (existing_doc, existing_score + rrf_score)
            else:
                doc_rrf[doc_key] = (doc, rrf_score)

        sorted_results = sorted(doc_rrf.values(), key=lambda x: x[1], reverse=True)
        docs = [doc for doc, _ in sorted_results[:self.k]]
        logger.info(f"Contextual retrieval: {len(semantic_results)} semantic + BM25 → {len(docs)} fused results")
        return docs

    @staticmethod
    def _doc_key(doc: Document) -> str:
        source = doc.metadata.get("source", "")
        chunk_id = doc.metadata.get("chunk_id", "")
        return f"{source}_{chunk_id}"


def create_contextual_retriever(
    embedding_function: Embeddings,
    collection_name: str = CONTEXTUAL_COLLECTION_NAME,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    bm25_path: str = CONTEXTUAL_BM25_PATH,
    k: int = TOP_K_RESULTS,
    score_threshold: float = SIMILARITY_THRESHOLD,
) -> ContextualRetriever:
    """Factory function to create a Contextual retriever."""
    from simple_rag.contextual_builder import load_bm25_data

    store = CKDVectorStore(
        embedding_function=embedding_function,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    bm25_index = None
    bm25_corpus_ids = []
    bm25_doc_map = {}

    corpus, chunk_ids = load_bm25_data(bm25_path)
    if corpus:
        from rank_bm25 import BM25Okapi
        tokenized = [text.lower().split() for text in corpus]
        bm25_index = BM25Okapi(tokenized)
        bm25_corpus_ids = chunk_ids
        logger.info(f"Loaded BM25 index: {len(corpus)} entries from {bm25_path}")
    else:
        logger.warning(f"No BM25 data at {bm25_path}, using semantic-only retrieval")

    return ContextualRetriever(
        vectorstore=store,
        embedding_function=embedding_function,
        bm25_index=bm25_index,
        bm25_corpus_ids=bm25_corpus_ids,
        bm25_doc_map=bm25_doc_map,
        k=k,
        score_threshold=score_threshold,
    )
