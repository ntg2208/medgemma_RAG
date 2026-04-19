"""Tests for RAPTOR collapsed retriever."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


class TestRaptorRetriever:
    """Test collapsed retrieval over RAPTOR collection."""

    def _make_retriever(self, search_results):
        """Helper to create a RaptorRetriever with mocked vectorstore."""
        from simple_rag.raptor_retriever import RaptorRetriever

        mock_vs = MagicMock()
        mock_vs.search_with_scores = MagicMock(return_value=search_results)

        mock_embed = MagicMock()

        return RaptorRetriever(
            vectorstore=mock_vs,
            embedding_function=mock_embed,
            k=3,
            score_threshold=0.3,
        )

    def test_returns_documents_above_threshold(self):
        results = [
            (Document(page_content="doc1", metadata={"raptor_layer": 0}), 0.9),
            (Document(page_content="doc2", metadata={"raptor_layer": 1}), 0.5),
            (Document(page_content="doc3", metadata={"raptor_layer": 0}), 0.1),
        ]
        retriever = self._make_retriever(results)
        docs = retriever.invoke("test query")
        assert len(docs) == 2  # doc3 below threshold
        assert docs[0].page_content == "doc1"

    def test_respects_k_limit(self):
        results = [
            (Document(page_content=f"doc{i}", metadata={}), 0.9 - i * 0.1)
            for i in range(5)
        ]
        retriever = self._make_retriever(results)
        docs = retriever.invoke("test query")
        assert len(docs) <= 3  # k=3

    def test_empty_collection_returns_empty(self):
        retriever = self._make_retriever([])
        docs = retriever.invoke("test query")
        assert docs == []

    def test_metadata_preserved(self):
        results = [
            (Document(
                page_content="summary text",
                metadata={"raptor_layer": 2, "raptor_node_id": "summary_L2_C0"},
            ), 0.8),
        ]
        retriever = self._make_retriever(results)
        docs = retriever.invoke("test")
        assert docs[0].metadata["raptor_layer"] == 2


class TestRaptorRetrieverFromCollection:
    """Test factory function that creates retriever from ChromaDB."""

    def test_create_raptor_retriever(self):
        from simple_rag.raptor_retriever import create_raptor_retriever

        mock_embed = MagicMock()
        # Patch chromadb and Chroma inside simple_rag.vectorstore, which is
        # where CKDVectorStore actually calls PersistentClient and Chroma.
        with patch("simple_rag.vectorstore.chromadb") as mock_chroma, \
             patch("simple_rag.vectorstore.Chroma") as mock_chroma_lc:
            mock_client = MagicMock()
            mock_chroma.PersistentClient.return_value = mock_client

            retriever = create_raptor_retriever(embedding_function=mock_embed)

        assert retriever.k == 5  # default from config
