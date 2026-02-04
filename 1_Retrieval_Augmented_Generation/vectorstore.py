"""
ChromaDB vector store for the CKD RAG System.

Handles document storage, indexing, and similarity search
with metadata filtering support.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


class CKDVectorStore:
    """
    Vector store for CKD guidelines and resources.

    Uses ChromaDB for persistent storage with support for:
    - Metadata filtering (by CKD stage, document type, source)
    - Similarity search with score thresholds
    - Batch document insertion

    Example:
        >>> from embeddings import EmbeddingGemmaWrapper
        >>> embeddings = EmbeddingGemmaWrapper()
        >>> store = CKDVectorStore(embeddings)
        >>> store.add_documents(documents)
        >>> results = store.search("dietary restrictions for stage 3")
    """

    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    ):
        """
        Initialize the vector store.

        Args:
            embedding_function: LangChain-compatible embedding function
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
        """
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Ensure persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Initialize LangChain Chroma wrapper
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )

        logger.info(
            f"Initialized CKDVectorStore: collection='{collection_name}', "
            f"persist_dir='{persist_directory}'"
        )

    @property
    def vectorstore(self) -> Chroma:
        """Get the underlying LangChain Chroma instance."""
        return self._vectorstore

    def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 100,
    ) -> list[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents to add per batch

        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents to add")
            return []

        all_ids = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Generate unique IDs based on content hash and metadata
            ids = [
                f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', i+j)}"
                for j, doc in enumerate(batch)
            ]

            # Add to vector store
            self._vectorstore.add_documents(batch, ids=ids)
            all_ids.extend(ids)

            logger.debug(f"Added batch {i//batch_size + 1}: {len(batch)} documents")

        logger.info(f"Added {len(all_ids)} documents to vector store")
        return all_ids

    def search(
        self,
        query: str,
        k: int = TOP_K_RESULTS,
        filter_dict: Optional[dict] = None,
        score_threshold: Optional[float] = None,
    ) -> list[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: Metadata filter (e.g., {"document_type": "guideline"})
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of matching Document objects
        """
        # Use similarity search with scores to enable thresholding
        results_with_scores = self._vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=filter_dict,
        )

        # Apply score threshold if specified
        if score_threshold is not None:
            results_with_scores = [
                (doc, score) for doc, score in results_with_scores
                if score >= score_threshold
            ]

        # Extract just the documents
        documents = [doc for doc, _ in results_with_scores]

        logger.debug(f"Search returned {len(documents)} results for: {query[:50]}...")
        return documents

    def search_with_scores(
        self,
        query: str,
        k: int = TOP_K_RESULTS,
        filter_dict: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.

        Args:
            query: Search query text
            k: Number of results to return
            filter_dict: Metadata filter

        Returns:
            List of (Document, score) tuples
        """
        return self._vectorstore.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=filter_dict,
        )

    def search_by_ckd_stage(
        self,
        query: str,
        ckd_stage: int,
        k: int = TOP_K_RESULTS,
    ) -> list[Document]:
        """
        Search for documents relevant to a specific CKD stage.

        Args:
            query: Search query text
            ckd_stage: CKD stage (1-5)
            k: Number of results to return

        Returns:
            List of matching Document objects
        """
        if not 1 <= ckd_stage <= 5:
            raise ValueError(f"Invalid CKD stage: {ckd_stage}. Must be 1-5.")

        # ChromaDB filter for array contains
        # Note: This checks if ckd_stages array contains the stage
        filter_dict = {"ckd_stages": {"$contains": ckd_stage}}

        return self.search(query, k=k, filter_dict=filter_dict)

    def search_by_document_type(
        self,
        query: str,
        doc_type: str,
        k: int = TOP_K_RESULTS,
    ) -> list[Document]:
        """
        Search within a specific document type.

        Args:
            query: Search query text
            doc_type: Document type ("guideline", "dietary", "clinical")
            k: Number of results to return

        Returns:
            List of matching Document objects
        """
        filter_dict = {"document_type": doc_type}
        return self.search(query, k=k, filter_dict=filter_dict)

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the current collection.

        Returns:
            Dictionary with collection statistics
        """
        collection = self._client.get_collection(self.collection_name)
        count = collection.count()

        # Get sample of metadata to understand content
        if count > 0:
            sample = collection.peek(min(count, 10))
            sources = set()
            doc_types = set()

            if sample.get("metadatas"):
                for meta in sample["metadatas"]:
                    if meta:
                        sources.add(meta.get("source", "unknown"))
                        doc_types.add(meta.get("document_type", "unknown"))

            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "sample_sources": list(sources),
                "sample_doc_types": list(doc_types),
            }

        return {
            "collection_name": self.collection_name,
            "document_count": 0,
        }

    def delete_collection(self):
        """Delete the entire collection."""
        self._client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")

        # Reinitialize empty collection
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def reset(self):
        """Reset the vector store (delete all documents)."""
        self.delete_collection()
        logger.info("Vector store reset complete")

    def as_retriever(
        self,
        search_type: str = "similarity",
        k: int = TOP_K_RESULTS,
        score_threshold: Optional[float] = SIMILARITY_THRESHOLD,
        filter_dict: Optional[dict] = None,
    ):
        """
        Get a LangChain retriever interface.

        Args:
            search_type: Type of search ("similarity" or "mmr")
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            filter_dict: Metadata filter

        Returns:
            LangChain Retriever object
        """
        search_kwargs = {"k": k}

        if filter_dict:
            search_kwargs["filter"] = filter_dict

        if score_threshold and search_type == "similarity":
            search_type = "similarity_score_threshold"
            search_kwargs["score_threshold"] = score_threshold

        return self._vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )


def create_vectorstore(
    embedding_function: Embeddings,
    documents: Optional[list[Document]] = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> CKDVectorStore:
    """
    Factory function to create and optionally populate a vector store.

    Args:
        embedding_function: Embedding function to use
        documents: Optional documents to add immediately
        collection_name: Name for the collection

    Returns:
        Initialized CKDVectorStore
    """
    store = CKDVectorStore(
        embedding_function=embedding_function,
        collection_name=collection_name,
    )

    if documents:
        store.add_documents(documents)

    return store


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    from embeddings import EmbeddingGemmaWrapper

    print("Testing CKDVectorStore...")

    # Initialize embeddings and store
    embeddings = EmbeddingGemmaWrapper(dimension=768)
    store = CKDVectorStore(embeddings)

    # Create sample documents
    sample_docs = [
        Document(
            page_content="Patients with CKD stage 3 should limit potassium intake to 2000-3000mg per day.",
            metadata={"source": "nice_guidelines.pdf", "chunk_id": 0, "document_type": "guideline", "ckd_stages": [3]},
        ),
        Document(
            page_content="ACE inhibitors are recommended for CKD patients with proteinuria.",
            metadata={"source": "nice_guidelines.pdf", "chunk_id": 1, "document_type": "guideline", "ckd_stages": [1, 2, 3, 4, 5]},
        ),
        Document(
            page_content="High potassium foods include bananas, oranges, and potatoes.",
            metadata={"source": "kidneycareuk_diet.pdf", "chunk_id": 0, "document_type": "dietary", "ckd_stages": [3, 4, 5]},
        ),
    ]

    # Add documents
    store.add_documents(sample_docs)

    # Test search
    results = store.search("potassium dietary restrictions")
    print(f"\nSearch results: {len(results)}")
    for doc in results:
        print(f"  - {doc.page_content[:80]}...")

    # Get stats
    stats = store.get_collection_stats()
    print(f"\nCollection stats: {stats}")
