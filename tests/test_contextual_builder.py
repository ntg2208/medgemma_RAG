"""Tests for Contextual RAG builder — context generation and indexing."""

import json
import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from langchain_core.documents import Document


class TestGenerateContext:
    """Test LLM-based context generation for individual chunks."""

    def test_generate_context_calls_llm(self):
        from simple_rag.contextual_builder import ContextualBuilder

        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(
            return_value="This chunk discusses potassium limits for CKD stage 3."
        )
        mock_embed = MagicMock()

        builder = ContextualBuilder(embedding_function=mock_embed, llm=mock_llm)

        chunk = Document(
            page_content="Limit potassium to 2000-3000mg per day.",
            metadata={"source": "nice.pdf", "chunk_id": 0},
        )

        context = builder.generate_context(
            document_text="# NICE Guidelines\n\n## Dietary\n\nLimit potassium...",
            chunk=chunk,
        )

        assert isinstance(context, str)
        assert len(context) > 0
        mock_llm.generate.assert_called_once()
        prompt_arg = mock_llm.generate.call_args[0][0]
        assert "NICE Guidelines" in prompt_arg
        assert "Limit potassium" in prompt_arg

    def test_generate_context_returns_llm_output(self):
        from simple_rag.contextual_builder import ContextualBuilder

        expected = "This chunk is from the dietary section of NICE NG203."
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value=expected)
        mock_embed = MagicMock()

        builder = ContextualBuilder(embedding_function=mock_embed, llm=mock_llm)
        chunk = Document(page_content="Some text.", metadata={})
        result = builder.generate_context("Full doc text.", chunk)
        assert result == expected


class TestContextualize:
    """Test contextualizing a chunk (prepending context to text)."""

    def test_contextualize_chunk(self):
        from simple_rag.contextual_builder import ContextualBuilder

        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="Context about CKD stage 3.")
        mock_embed = MagicMock()

        builder = ContextualBuilder(embedding_function=mock_embed, llm=mock_llm)

        chunk = Document(
            page_content="Limit potassium intake.",
            metadata={"source": "nice.pdf", "chunk_id": 0},
        )

        ctx_chunk = builder.contextualize_chunk("Full document.", chunk)

        assert ctx_chunk.page_content.startswith("Context about CKD stage 3.")
        assert "Limit potassium intake." in ctx_chunk.page_content
        assert ctx_chunk.metadata["contextual_context"] == "Context about CKD stage 3."
        assert ctx_chunk.metadata["source"] == "nice.pdf"


class TestBM25Persistence:
    """Test BM25 index save/load."""

    def test_save_and_load_bm25(self, tmp_path):
        from simple_rag.contextual_builder import save_bm25_data, load_bm25_data

        corpus = [
            "CKD stage 3 potassium dietary limits",
            "Hemodialysis sodium restriction guidelines",
            "ACE inhibitor dosing for proteinuria",
        ]
        chunk_ids = ["c0", "c1", "c2"]
        path = str(tmp_path / "bm25.json")

        save_bm25_data(corpus, chunk_ids, path)
        loaded_corpus, loaded_ids = load_bm25_data(path)

        assert loaded_corpus == corpus
        assert loaded_ids == chunk_ids

    def test_load_missing_file_returns_empty(self, tmp_path):
        from simple_rag.contextual_builder import load_bm25_data

        corpus, ids = load_bm25_data(str(tmp_path / "nonexistent.json"))
        assert corpus == []
        assert ids == []
