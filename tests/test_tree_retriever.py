"""Tests for the tree-based retriever module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document


# ============================================================================
# Fixtures
# ============================================================================


def _make_doc(content, source="test.pdf", chunk_id=0, section_path="", section_numbering=""):
    """Helper to create a Document with metadata."""
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "chunk_id": chunk_id,
            "section": "Test Section",
            "section_path": section_path,
            "section_numbering": section_numbering,
            "doc_name": "test_doc",
        },
    )


def _make_section_doc(heading, source="test.pdf", doc_name="test_doc",
                       section_numbering="1.1", section_path="Chapter 1|1.1. Details",
                       depth=1):
    """Helper to create a section heading Document."""
    return Document(
        page_content=heading,
        metadata={
            "source": source,
            "doc_name": doc_name,
            "section_numbering": section_numbering,
            "section_path": section_path,
            "depth": depth,
            "line_start": 0,
            "line_end": 10,
        },
    )


@pytest.fixture
def mock_vectorstore():
    """Mock CKDVectorStore."""
    mock = MagicMock()
    mock.search_with_scores = MagicMock(return_value=[
        (_make_doc("Chunk about ESA dosing", section_path="Chapter 3|3.4. ESA dosing",
                   section_numbering="3.4"), 0.85),
        (_make_doc("Another chunk about ESA", chunk_id=1, section_path="Chapter 3|3.4. ESA dosing",
                   section_numbering="3.4"), 0.75),
    ])
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock embedding function."""
    mock = MagicMock()
    mock.embed_query = MagicMock(return_value=[0.1] * 768)
    mock.embed_documents = MagicMock(return_value=[[0.1] * 768])
    return mock


# ============================================================================
# TreeRetriever tests
# ============================================================================


class TestTreeRetriever:
    """Test the TreeRetriever class."""

    def test_instantiation(self, mock_vectorstore, mock_embeddings):
        """TreeRetriever should instantiate with required parameters."""
        from importlib import import_module
        tree_mod = import_module("simple_rag.tree_retriever")

        retriever = tree_mod.TreeRetriever(
            vectorstore=mock_vectorstore,
            embedding_function=mock_embeddings,
        )
        assert retriever.k == 5  # TOP_K_RESULTS default
        assert retriever.section_k == 8  # SECTION_K default
        assert retriever.chunks_per_section == 3  # CHUNKS_PER_SECTION default

    def test_flat_fallback(self, mock_vectorstore, mock_embeddings):
        """Should fall back to flat retrieval when section routing fails."""
        from importlib import import_module
        tree_mod = import_module("simple_rag.tree_retriever")

        retriever = tree_mod.TreeRetriever(
            vectorstore=mock_vectorstore,
            embedding_function=mock_embeddings,
        )
        results = retriever._flat_fallback("test query")
        assert len(results) == 2
        mock_vectorstore.search_with_scores.assert_called_once()

    def test_flat_fallback_respects_threshold(self, mock_vectorstore, mock_embeddings):
        """Flat fallback should filter by score threshold."""
        from importlib import import_module
        tree_mod = import_module("simple_rag.tree_retriever")

        mock_vectorstore.search_with_scores.return_value = [
            (_make_doc("Good chunk"), 0.8),
            (_make_doc("Bad chunk", chunk_id=1), 0.1),
        ]

        retriever = tree_mod.TreeRetriever(
            vectorstore=mock_vectorstore,
            embedding_function=mock_embeddings,
            score_threshold=0.5,
        )
        results = retriever._flat_fallback("test query")
        assert len(results) == 1

    def test_with_config(self, mock_vectorstore, mock_embeddings):
        """with_config should create a new retriever with updated k."""
        from importlib import import_module
        tree_mod = import_module("simple_rag.tree_retriever")

        retriever = tree_mod.TreeRetriever(
            vectorstore=mock_vectorstore,
            embedding_function=mock_embeddings,
            k=5,
        )
        new_retriever = retriever.with_config(k=10)
        assert new_retriever.k == 10
        assert new_retriever.vectorstore is mock_vectorstore

    def test_invoke_with_section_hits(self, mock_vectorstore, mock_embeddings):
        """Full retrieval should work when section collection returns hits."""
        from importlib import import_module
        tree_mod = import_module("simple_rag.tree_retriever")

        # Mock section collection
        mock_collection = MagicMock()
        mock_collection.similarity_search_with_relevance_scores.return_value = [
            (_make_section_doc("3.4. ESA dosing", section_numbering="3.4",
                               section_path="Chapter 3|3.4. ESA dosing"), 0.9),
        ]

        retriever = tree_mod.TreeRetriever(
            vectorstore=mock_vectorstore,
            embedding_function=mock_embeddings,
        )
        retriever._get_section_collection = MagicMock(return_value=mock_collection)
        results = retriever.invoke("ESA dosing for hemodialysis")

        assert len(results) > 0
        mock_vectorstore.search_with_scores.assert_called()

    def test_prepend_context(self, mock_vectorstore, mock_embeddings):
        """Retrieved chunks should have hierarchical context prepended."""
        from importlib import import_module
        tree_mod = import_module("simple_rag.tree_retriever")

        mock_collection = MagicMock()
        mock_collection.similarity_search_with_relevance_scores.return_value = [
            (_make_section_doc("3.4. ESA dosing", section_numbering="3.4",
                               section_path="Chapter 3|3.4. ESA dosing"), 0.9),
        ]

        retriever = tree_mod.TreeRetriever(
            vectorstore=mock_vectorstore,
            embedding_function=mock_embeddings,
            prepend_context=True,
        )
        retriever._get_section_collection = MagicMock(return_value=mock_collection)
        results = retriever.invoke("ESA dosing")

        assert len(results) > 0
        assert "[Chapter 3 > 3.4. ESA dosing]" in results[0].page_content

    def test_no_section_hits_falls_back(self, mock_vectorstore, mock_embeddings):
        """Should fall back to flat retrieval when no section hits."""
        from importlib import import_module
        tree_mod = import_module("simple_rag.tree_retriever")

        mock_collection = MagicMock()
        mock_collection.similarity_search_with_relevance_scores.return_value = []

        retriever = tree_mod.TreeRetriever(
            vectorstore=mock_vectorstore,
            embedding_function=mock_embeddings,
        )
        retriever._get_section_collection = MagicMock(return_value=mock_collection)
        results = retriever.invoke("test query")

        assert len(results) > 0


# ============================================================================
# Factory integration test
# ============================================================================


class TestCreateRetrieverFactory:
    """Test the create_retriever factory with use_tree option."""

    def test_create_retriever_with_tree(self, mock_vectorstore, mock_embeddings):
        """create_retriever(use_tree=True) should return TreeRetriever."""
        from importlib import import_module
        retriever_mod = import_module("simple_rag.retriever")
        tree_mod = import_module("simple_rag.tree_retriever")

        retriever = retriever_mod.create_retriever(
            vectorstore=mock_vectorstore,
            use_tree=True,
            embedding_function=mock_embeddings,
        )
        assert isinstance(retriever, tree_mod.TreeRetriever)

    def test_create_retriever_tree_requires_embeddings(self, mock_vectorstore):
        """create_retriever(use_tree=True) without embeddings should raise."""
        from importlib import import_module
        retriever_mod = import_module("simple_rag.retriever")

        with pytest.raises(ValueError, match="embedding_function"):
            retriever_mod.create_retriever(
                vectorstore=mock_vectorstore,
                use_tree=True,
            )

    def test_create_retriever_default_still_works(self, mock_vectorstore):
        """Default create_retriever() should still return CKDRetriever."""
        from importlib import import_module
        retriever_mod = import_module("simple_rag.retriever")

        retriever = retriever_mod.create_retriever(vectorstore=mock_vectorstore)
        assert isinstance(retriever, retriever_mod.CKDRetriever)
