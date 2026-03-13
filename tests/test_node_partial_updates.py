"""Tests for node partial update pattern (LangGraph best practice)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_nodes(pii_handler=None, retriever=None, llm=None):
    """Create RAGNodes with mocks."""
    import importlib
    nodes_module = importlib.import_module("2_Agentic_RAG.nodes")

    if pii_handler is None:
        pii_handler = MagicMock()
        pii_handler.anonymize.return_value = MagicMock(
            anonymized_text="test query",
            pii_found=False,
            placeholder_map={},
            entities_detected=[],
        )
    if retriever is None:
        retriever = MagicMock()
        retriever.invoke.return_value = []
    if llm is None:
        llm = MagicMock()
        llm.generate.return_value = "test response"

    return nodes_module.RAGNodes(pii_handler=pii_handler, retriever=retriever, llm=llm)


class TestPartialUpdates:
    """All nodes should return partial state updates, not the full state."""

    def test_pii_check_returns_partial(self):
        nodes = _make_nodes()
        state = {"original_query": "What is CKD?", "processing_steps": []}
        result = nodes.pii_check(state)

        # Should NOT contain original_query (that's input, not an update)
        assert "original_query" not in result
        assert "processing_steps" in result
        assert "anonymized_query" in result

    def test_analyze_query_returns_partial(self):
        nodes = _make_nodes()
        state = {"anonymized_query": "what is kidney disease", "processing_steps": []}
        result = nodes.analyze_query(state)

        assert "original_query" not in result
        assert "query_intent" in result
        assert "processing_steps" in result

    def test_retrieve_documents_returns_partial(self):
        nodes = _make_nodes()
        state = {"anonymized_query": "CKD diet", "processing_steps": []}
        result = nodes.retrieve_documents(state)

        assert "anonymized_query" not in result
        assert "processing_steps" in result
        assert "retrieved_documents" in result

    def test_generate_response_returns_partial(self):
        nodes = _make_nodes()
        state = {
            "anonymized_query": "CKD diet",
            "context": "Some context",
            "processing_steps": [],
        }
        result = nodes.generate_response(state)

        assert "processing_steps" in result
        assert "final_response" in result

    def test_generate_direct_returns_partial(self):
        nodes = _make_nodes()
        state = {"anonymized_query": "What is CKD?", "processing_steps": []}
        result = nodes.generate_direct_response(state)

        assert "processing_steps" in result
        assert "final_response" in result

    def test_generate_clarification_returns_partial(self):
        nodes = _make_nodes()
        state = {"processing_steps": []}
        result = nodes.generate_clarification(state)

        assert "processing_steps" in result
        assert "final_response" in result
        assert len(result) == 2  # Only these two keys

    def test_generate_out_of_scope_returns_partial(self):
        nodes = _make_nodes()
        state = {"processing_steps": []}
        result = nodes.generate_out_of_scope(state)

        assert "processing_steps" in result
        assert "final_response" in result
        assert len(result) == 2

    def test_evaluate_response_returns_partial(self):
        nodes = _make_nodes()
        state = {
            "original_query": "test",
            "final_response": "response",
            "processing_steps": [],
        }
        result = nodes.evaluate_response(state)

        assert "processing_steps" in result
        # No evaluator configured, so just processing_steps
        assert len(result) == 1

    def test_processing_steps_uses_reducer_format(self):
        """Processing steps should be a list (for Annotated[list, add] reducer)."""
        nodes = _make_nodes()
        state = {"original_query": "What is CKD?", "processing_steps": []}
        result = nodes.pii_check(state)

        # Should return a list (reducer will concatenate)
        assert isinstance(result["processing_steps"], list)
        assert result["processing_steps"] == ["pii_check"]
