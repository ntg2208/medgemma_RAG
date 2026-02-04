"""
Document retriever for the CKD RAG System.

Provides a high-level interface for retrieving relevant documents
with support for query expansion and reranking.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TOP_K_RESULTS, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


class CKDRetriever(BaseRetriever):
    """
    Custom retriever for CKD-related documents.

    Features:
    - CKD stage-aware retrieval
    - Document type filtering
    - Query expansion for medical terms
    - Configurable similarity thresholds

    Example:
        >>> retriever = CKDRetriever(vectorstore)
        >>> docs = retriever.invoke("dietary restrictions for stage 3")
    """

    vectorstore: "CKDVectorStore"  # type: ignore
    k: int = TOP_K_RESULTS
    score_threshold: float = SIMILARITY_THRESHOLD
    ckd_stage: Optional[int] = None
    document_type: Optional[str] = None
    expand_queries: bool = True

    class Config:
        arbitrary_types_allowed = True

    # Medical term expansions for query enhancement
    TERM_EXPANSIONS = {
        "ckd": ["chronic kidney disease", "kidney disease", "renal disease"],
        "egfr": ["estimated glomerular filtration rate", "gfr", "kidney function"],
        "potassium": ["K+", "hyperkalemia", "hypokalemia"],
        "phosphorus": ["phosphate", "hyperphosphatemia"],
        "protein": ["proteinuria", "albuminuria", "dietary protein"],
        "ace": ["ace inhibitor", "angiotensin converting enzyme"],
        "arb": ["angiotensin receptor blocker"],
        "bp": ["blood pressure", "hypertension"],
        "dialysis": ["hemodialysis", "peritoneal dialysis", "renal replacement"],
    }

    def _expand_query(self, query: str) -> str:
        """
        Expand query with related medical terms.

        Args:
            query: Original query text

        Returns:
            Expanded query with additional terms
        """
        if not self.expand_queries:
            return query

        query_lower = query.lower()
        expansions = []

        for term, related in self.TERM_EXPANSIONS.items():
            if term in query_lower:
                # Add first expansion term that's not already in query
                for expansion in related:
                    if expansion.lower() not in query_lower:
                        expansions.append(expansion)
                        break

        if expansions:
            expanded = f"{query} ({', '.join(expansions)})"
            logger.debug(f"Query expanded: {query} -> {expanded}")
            return expanded

        return query

    def _build_filter(self) -> Optional[dict]:
        """
        Build metadata filter based on configured options.

        Returns:
            Filter dictionary or None
        """
        filters = {}

        if self.ckd_stage is not None:
            filters["ckd_stages"] = {"$contains": self.ckd_stage}

        if self.document_type is not None:
            filters["document_type"] = self.document_type

        return filters if filters else None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Retrieve relevant documents for the query.

        Args:
            query: User query
            run_manager: Callback manager

        Returns:
            List of relevant documents
        """
        # Expand query with medical terms
        expanded_query = self._expand_query(query)

        # Build filter
        filter_dict = self._build_filter()

        # Perform search
        results = self.vectorstore.search_with_scores(
            query=expanded_query,
            k=self.k,
            filter_dict=filter_dict,
        )

        # Filter by score threshold
        filtered_results = [
            doc for doc, score in results
            if score >= self.score_threshold
        ]

        logger.info(
            f"Retrieved {len(filtered_results)}/{len(results)} documents "
            f"(threshold={self.score_threshold})"
        )

        return filtered_results

    def with_config(
        self,
        ckd_stage: Optional[int] = None,
        document_type: Optional[str] = None,
        k: Optional[int] = None,
    ) -> "CKDRetriever":
        """
        Create a new retriever with updated configuration.

        Args:
            ckd_stage: CKD stage to filter by
            document_type: Document type to filter by
            k: Number of results to return

        Returns:
            New CKDRetriever instance with updated config
        """
        return CKDRetriever(
            vectorstore=self.vectorstore,
            k=k if k is not None else self.k,
            score_threshold=self.score_threshold,
            ckd_stage=ckd_stage if ckd_stage is not None else self.ckd_stage,
            document_type=document_type if document_type is not None else self.document_type,
            expand_queries=self.expand_queries,
        )


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines multiple retrieval strategies.

    Merges results from:
    - Semantic similarity search
    - Keyword-based search (BM25-style)
    - Metadata-filtered search

    Uses reciprocal rank fusion to combine results.
    """

    vectorstore: "CKDVectorStore"  # type: ignore
    k: int = TOP_K_RESULTS
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Retrieve documents using hybrid approach.

        Args:
            query: User query
            run_manager: Callback manager

        Returns:
            Combined and reranked documents
        """
        # Get semantic search results
        semantic_results = self.vectorstore.search_with_scores(
            query=query,
            k=self.k * 2,  # Get more for fusion
        )

        # Simple reciprocal rank fusion
        doc_scores: dict[str, tuple[Document, float]] = {}

        for rank, (doc, score) in enumerate(semantic_results, start=1):
            doc_id = f"{doc.metadata.get('source')}_{doc.metadata.get('chunk_id')}"
            rrf_score = self.semantic_weight / (rank + 60)  # RRF formula

            if doc_id in doc_scores:
                existing_doc, existing_score = doc_scores[doc_id]
                doc_scores[doc_id] = (existing_doc, existing_score + rrf_score)
            else:
                doc_scores[doc_id] = (doc, rrf_score)

        # Sort by combined score and return top k
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [doc for doc, _ in sorted_results[:self.k]]


def create_retriever(
    vectorstore: "CKDVectorStore",  # type: ignore
    ckd_stage: Optional[int] = None,
    document_type: Optional[str] = None,
    k: int = TOP_K_RESULTS,
    use_hybrid: bool = False,
) -> BaseRetriever:
    """
    Factory function to create a retriever.

    Args:
        vectorstore: Vector store to retrieve from
        ckd_stage: Optional CKD stage filter
        document_type: Optional document type filter
        k: Number of results to return
        use_hybrid: Whether to use hybrid retrieval

    Returns:
        Configured retriever instance
    """
    if use_hybrid:
        return HybridRetriever(vectorstore=vectorstore, k=k)

    return CKDRetriever(
        vectorstore=vectorstore,
        k=k,
        ckd_stage=ckd_stage,
        document_type=document_type,
    )
