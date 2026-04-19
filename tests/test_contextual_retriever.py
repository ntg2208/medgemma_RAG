"""Tests for Contextual RAG hybrid retriever."""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


class TestContextualRetriever:
    """Test hybrid semantic + BM25 retrieval."""

    def _make_retriever(self, semantic_results, bm25_corpus=None, bm25_ids=None):
        from simple_rag.contextual_retriever import ContextualRetriever

        mock_vs = MagicMock()
        mock_vs.search_with_scores = MagicMock(return_value=semantic_results)
        mock_embed = MagicMock()

        bm25_index = None
        bm25_doc_map = {}
        if bm25_corpus:
            from rank_bm25 import BM25Okapi
            tokenized = [doc.lower().split() for doc in bm25_corpus]
            bm25_index = BM25Okapi(tokenized)
            for i, (doc, _) in enumerate(semantic_results):
                chunk_id = bm25_ids[i] if bm25_ids and i < len(bm25_ids) else str(i)
                bm25_doc_map[chunk_id] = doc

        return ContextualRetriever(
            vectorstore=mock_vs,
            embedding_function=mock_embed,
            bm25_index=bm25_index,
            bm25_corpus_ids=bm25_ids or [],
            bm25_doc_map=bm25_doc_map,
            k=3,
            score_threshold=0.2,
        )

    def test_semantic_only_when_no_bm25(self):
        results = [
            (Document(page_content="doc1", metadata={"source": "a.pdf"}), 0.9),
            (Document(page_content="doc2", metadata={"source": "b.pdf"}), 0.5),
        ]
        retriever = self._make_retriever(results)
        docs = retriever.invoke("test query")
        assert len(docs) == 2

    def test_hybrid_merges_results(self):
        semantic_results = [
            (Document(page_content="CKD potassium limits", metadata={"source": "a", "chunk_id": 0}), 0.9),
            (Document(page_content="CKD sodium limits", metadata={"source": "a", "chunk_id": 1}), 0.7),
            (Document(page_content="Exercise guidelines", metadata={"source": "b", "chunk_id": 0}), 0.5),
        ]
        bm25_corpus = ["CKD potassium limits", "CKD sodium limits", "Exercise guidelines"]
        bm25_ids = ["a_0", "a_1", "b_0"]

        retriever = self._make_retriever(semantic_results, bm25_corpus, bm25_ids)
        docs = retriever.invoke("potassium restriction CKD")
        assert len(docs) <= 3
        assert len(docs) >= 1

    def test_empty_results(self):
        retriever = self._make_retriever([])
        docs = retriever.invoke("test")
        assert docs == []

    def test_respects_k_limit(self):
        results = [
            (Document(page_content=f"doc{i}", metadata={"source": "a", "chunk_id": i}), 0.9)
            for i in range(10)
        ]
        bm25_corpus = [f"doc{i}" for i in range(10)]
        bm25_ids = [f"a_{i}" for i in range(10)]
        retriever = self._make_retriever(results, bm25_corpus, bm25_ids)
        docs = retriever.invoke("test")
        assert len(docs) <= 3


class TestCreateContextualRetriever:
    """Test factory function."""

    def test_create_without_bm25_file(self, tmp_path):
        from simple_rag.contextual_retriever import create_contextual_retriever

        mock_embed = MagicMock()

        with patch("simple_rag.contextual_retriever.CKDVectorStore") as mock_store_cls:
            mock_store = MagicMock()
            mock_store_cls.return_value = mock_store

            retriever = create_contextual_retriever(
                embedding_function=mock_embed,
                bm25_path=str(tmp_path / "nonexistent.json"),
            )

        assert retriever.bm25_index is None
