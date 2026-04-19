"""Tests for retriever comparison evaluation helpers."""

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLoadQueries:
    """Test query dataset loading."""

    def test_load_queries_file_exists(self):
        path = Path(__file__).parent.parent / "eval" / "test_queries.json"
        assert path.exists(), f"test_queries.json not found at {path}"

    def test_load_queries_valid_json(self):
        path = Path(__file__).parent.parent / "eval" / "test_queries.json"
        data = json.loads(path.read_text())
        assert "queries" in data
        assert len(data["queries"]) >= 5

    def test_query_schema(self):
        path = Path(__file__).parent.parent / "eval" / "test_queries.json"
        data = json.loads(path.read_text())
        for q in data["queries"]:
            assert "id" in q, f"Query missing 'id': {q}"
            assert "query" in q, f"Query missing 'query': {q}"
            assert "category" in q, f"Query missing 'category': {q}"
            assert "reference" in q, f"Query missing 'reference' key: {q}"


class TestAggregateScores:
    """Test score aggregation logic."""

    def test_aggregate_empty(self):
        from eval.run_retriever_comparison import aggregate_scores
        result = aggregate_scores([])
        assert result == {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "average": 0.0,
        }

    def test_aggregate_single(self):
        from eval.run_retriever_comparison import aggregate_scores
        from importlib import import_module
        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        scores = [ragas_mod.RAGASScores(0.8, 0.9, 0.7, 0.6)]
        result = aggregate_scores(scores)
        assert result["faithfulness"] == pytest.approx(0.8)
        assert result["answer_relevancy"] == pytest.approx(0.9)
        assert result["context_precision"] == pytest.approx(0.7)
        assert result["context_recall"] == pytest.approx(0.6)
        assert result["average"] == pytest.approx(0.75)

    def test_aggregate_multiple(self):
        from eval.run_retriever_comparison import aggregate_scores
        from importlib import import_module
        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        scores = [
            ragas_mod.RAGASScores(1.0, 1.0, 1.0, 1.0),
            ragas_mod.RAGASScores(0.0, 0.0, 0.0, 0.0),
        ]
        result = aggregate_scores(scores)
        assert result["faithfulness"] == pytest.approx(0.5)
        assert result["average"] == pytest.approx(0.5)


class TestFormatContext:
    """Test context formatting from retrieved documents."""

    def test_format_contexts_extracts_page_content(self):
        from eval.run_retriever_comparison import extract_contexts
        from unittest.mock import MagicMock

        doc1 = MagicMock()
        doc1.page_content = "Context about potassium limits."
        doc2 = MagicMock()
        doc2.page_content = "Context about sodium intake."

        result = extract_contexts([doc1, doc2])
        assert result == [
            "Context about potassium limits.",
            "Context about sodium intake.",
        ]

    def test_format_contexts_empty(self):
        from eval.run_retriever_comparison import extract_contexts
        assert extract_contexts([]) == []
