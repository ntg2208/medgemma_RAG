"""
Tree-based retriever for the CKD RAG System.

Implements a section-route-then-chunk retrieval strategy:
1. Find relevant sections via heading similarity (small collection)
2. Expand to include parent sections for context
3. Retrieve chunks from those sections only (main collection)
4. Re-rank and prepend hierarchical context

This provides better precision than flat similarity search by using
document structure to narrow the search space.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_PERSIST_DIRECTORY,
    SECTION_COLLECTION_NAME,
    SECTION_K,
    CHUNKS_PER_SECTION,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "Data" / "processed_with_sections"


class TreeRetriever(BaseRetriever):
    """
    Tree-based retriever that routes queries through document structure.

    Uses a two-phase approach:
    1. Section routing: finds relevant sections via heading similarity
    2. Chunk retrieval: fetches chunks only from matched sections

    This narrows retrieval to structurally relevant content, avoiding
    noise from tangential mentions of query terms.

    Example:
        >>> retriever = TreeRetriever(
        ...     vectorstore=ckd_vectorstore,
        ...     embedding_function=embeddings,
        ... )
        >>> docs = retriever.invoke("ESA dosing for hemodialysis")
    """

    vectorstore: Any  # CKDVectorStore
    embedding_function: Any  # Embeddings
    k: int = TOP_K_RESULTS
    section_k: int = SECTION_K
    chunks_per_section: int = CHUNKS_PER_SECTION
    score_threshold: float = SIMILARITY_THRESHOLD
    prepend_context: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_section_collection(self) -> Chroma:
        """Get the section heading ChromaDB collection.

        Reuses the ChromaDB client from the main vectorstore to avoid
        conflicting PersistentClient instances for the same directory.
        """
        # CKDVectorStore exposes _client; fall back to creating one if needed
        client = getattr(self.vectorstore, "_client", None)
        if client is None:
            client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIRECTORY,
                settings=Settings(anonymized_telemetry=False, allow_reset=False),
            )
        return Chroma(
            client=client,
            collection_name=SECTION_COLLECTION_NAME,
            embedding_function=self.embedding_function,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Retrieve documents using tree-based section routing.

        Phase 1: Route to relevant sections via heading similarity
        Phase 2: Expand to parent sections for context
        Phase 3: Retrieve chunks from matched sections
        Phase 4: Re-rank and prepend hierarchical context

        Args:
            query: User query
            run_manager: Callback manager

        Returns:
            List of relevant documents with section context
        """
        # Phase 1: Find relevant sections
        section_collection = self._get_section_collection()
        section_hits = section_collection.similarity_search_with_relevance_scores(
            query,
            k=self.section_k,
        )

        if not section_hits:
            logger.warning("No section matches found, falling back to flat retrieval")
            return self._flat_fallback(query)

        logger.debug(
            f"Section routing: {len(section_hits)} hits, "
            f"top={section_hits[0][0].metadata.get('section_numbering', '?')} "
            f"score={section_hits[0][1]:.3f}"
        )

        # Phase 2: Collect target sections (including parents)
        target_filters = set()
        for hit_doc, score in section_hits:
            doc_name = hit_doc.metadata.get("doc_name", "")
            source = hit_doc.metadata.get("source", "")
            section_path = hit_doc.metadata.get("section_path", "")

            # Add matched section
            section_num = hit_doc.metadata.get("section_numbering", "")
            if source and section_num:
                target_filters.add((source, section_num))

            # Add parent sections from the path
            if section_path:
                path_parts = section_path.split("|")
                # Load tree to get parent numberings
                tree_nodes = _load_tree_cached(doc_name)
                if tree_nodes:
                    for part in path_parts[:-1]:  # all ancestors
                        for node in tree_nodes:
                            if node.heading == part:
                                parent_num = node.numbering if node.numbering else node.heading
                                target_filters.add((source, parent_num))

        if not target_filters:
            logger.warning("No target sections resolved, falling back to flat retrieval")
            return self._flat_fallback(query)

        # Phase 3: Retrieve chunks from target sections
        all_chunks: list[tuple[Document, float]] = []

        for source, section_num in target_filters:
            try:
                filter_dict = {
                    "$and": [
                        {"source": {"$eq": source}},
                        {"section_numbering": {"$eq": section_num}},
                    ]
                }
                chunks = self.vectorstore.search_with_scores(
                    query=query,
                    k=self.chunks_per_section,
                    filter_dict=filter_dict,
                )
                all_chunks.extend(chunks)
            except Exception as e:
                logger.debug(f"Filter failed for {source}/{section_num}: {e}")
                continue

        if not all_chunks:
            logger.warning("No chunks found in target sections, falling back to flat retrieval")
            return self._flat_fallback(query)

        # Phase 4: Re-rank by relevance score, deduplicate, take top-k
        # Deduplicate by (source, chunk_id)
        seen = set()
        unique_chunks = []
        for doc, score in all_chunks:
            key = (doc.metadata.get("source", ""), doc.metadata.get("chunk_id", ""))
            if key not in seen:
                seen.add(key)
                unique_chunks.append((doc, score))

        # Sort by score descending
        unique_chunks.sort(key=lambda x: x[1], reverse=True)

        # Apply threshold and limit
        results = []
        for doc, score in unique_chunks[:self.k]:
            if score < self.score_threshold:
                continue

            # Prepend hierarchical context
            if self.prepend_context:
                section_path = doc.metadata.get("section_path", "")
                if section_path:
                    context_header = " > ".join(section_path.split("|"))
                    doc.page_content = f"[{context_header}]\n\n{doc.page_content}"

            results.append(doc)

        logger.info(
            f"Tree retrieval: {len(target_filters)} sections -> "
            f"{len(all_chunks)} chunks -> {len(results)} results"
        )

        return results

    def _flat_fallback(self, query: str) -> list[Document]:
        """Fall back to flat similarity search when tree routing fails."""
        results = self.vectorstore.search_with_scores(
            query=query,
            k=self.k,
        )
        return [
            doc for doc, score in results
            if score >= self.score_threshold
        ]

    def with_config(
        self,
        k: Optional[int] = None,
        **kwargs,
    ) -> "TreeRetriever":
        """Create a new retriever with updated configuration."""
        return TreeRetriever(
            vectorstore=self.vectorstore,
            embedding_function=self.embedding_function,
            k=k if k is not None else self.k,
            section_k=self.section_k,
            chunks_per_section=self.chunks_per_section,
            score_threshold=self.score_threshold,
            prepend_context=self.prepend_context,
        )


def _load_tree_cached(doc_name: str) -> list:
    """Load and cache a document's section tree.

    Uses a module-level cache to avoid re-reading JSON on every retrieval.
    """
    return _tree_cache.get(doc_name, [])


# Module-level tree cache, populated on first use
_tree_cache: dict[str, list] = {}


def _populate_tree_cache():
    """Populate the tree cache from all document directories."""
    global _tree_cache
    if _tree_cache:
        return

    from Data.tree_builder import load_tree

    for doc_dir in PROCESSED_DIR.iterdir():
        if not doc_dir.is_dir() or doc_dir.name.startswith("."):
            continue
        roots = load_tree(doc_dir)
        if roots:
            # Flatten all nodes for easy lookup
            all_nodes = []
            for root in roots:
                all_nodes.extend(root.flatten())
            _tree_cache[doc_dir.name] = all_nodes

    logger.info(f"Loaded tree cache: {len(_tree_cache)} documents")


# Populate cache on module import
try:
    _populate_tree_cache()
except Exception as e:
    logger.debug(f"Tree cache not populated (expected if trees not built yet): {e}")
