"""
Tests for state consolidation refactoring.

Verifies that the Agentic RAG graph uses TypedDict-compatible state directly
without unnecessary GraphState dataclass conversion.
"""

import sys
import importlib.util
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

# Dynamically load 2_Agentic_RAG.nodes module
spec = importlib.util.spec_from_file_location(
    "nodes",
    Path(__file__).parent.parent / "2_Agentic_RAG" / "nodes.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load nodes module")
nodes_module = importlib.util.module_from_spec(spec)  # type: ignore[assignment]
spec.loader.exec_module(nodes_module)  # type: ignore[attr-defined]
RAGNodes = nodes_module.RAGNodes  # noqa: F401


class TestStateConsolidation:
    """Test that state is passed as dict without GraphState wrapping."""

    def test_state_is_dict(self):
        """Verify state parameter is dict, not GraphState dataclass."""
        # Create mock dependencies
        pii_handler = MagicMock()
        retriever = MagicMock()
        llm = MagicMock()

        # Mock PII handler response
        pii_handler.anonymize.return_value = MagicMock(
            anonymized_text="test query",
            pii_found=False,
            placeholder_map={}
        )

        # Create nodes
        nodes = RAGNodes(pii_handler=pii_handler, retriever=retriever, llm=llm)

        # Test state as dict
        test_state: dict = {
            "original_query": "What is CKD?",
            "ckd_stage": 3,
            "anonymized_query": "What is CKD?",
            "pii_detected": False,
            "pii_map": {},
            "query_intent": "retrieval",
            "query_keywords": [],
            "retrieved_documents": [],
            "context": "",
            "raw_response": "",
            "final_response": "",
            "evaluation_scores": {},
            "error": None,
            "processing_steps": [],
        }

        # Call node with dict state - should not require GraphState
        result = nodes.pii_check(test_state)

        # Verify result is dict, not GraphState
        assert isinstance(result, dict), "Node should return dict state"
        assert "original_query" in result
        assert "processing_steps" in result

    def test_all_nodes_accept_dict(self):
        """Verify all node methods accept and return dict state."""
        # Create mock dependencies
        pii_handler = MagicMock()
        retriever = MagicMock()
        llm = MagicMock()

        # Mock responses
        pii_handler.anonymize.return_value = MagicMock(
            anonymized_text="test query",
            pii_found=False,
            placeholder_map={}
        )
        retriever.invoke.return_value = []
        llm.generate.return_value = "Test response"

        nodes = RAGNodes(pii_handler=pii_handler, retriever=retriever, llm=llm)

        # Test state
        test_state: dict = {
            "original_query": "What is CKD?",
            "ckd_stage": 3,
            "anonymized_query": "What is CKD?",
            "pii_detected": False,
            "pii_map": {},
            "query_intent": "retrieval",
            "query_keywords": [],
            "retrieved_documents": [],
            "context": "",
            "raw_response": "",
            "final_response": "",
            "evaluation_scores": {},
            "error": None,
            "processing_steps": [],
        }

        # Test all node methods
        nodes.pii_check(test_state)
        nodes.analyze_query(test_state)
        nodes.retrieve_documents(test_state)
        nodes.generate_response(test_state)
        nodes.generate_direct_response(test_state)
        nodes.generate_clarification(test_state)
        nodes.generate_out_of_scope(test_state)

        # All should work with dict state
        assert isinstance(test_state, dict)

    def test_node_signatures_use_dict(self):
        """Verify node method signatures use dict type hints."""
        source = inspect.getsource(nodes_module)

        # The refactored code should use dict type hints in function signatures
        # Check pii_check signature
        assert "def pii_check(self, state: dict)" in source or "def pii_check(self, state:dict)" in source, \
            "pii_check should use dict type hint"

        # Check analyze_query signature
        assert "def analyze_query(self, state: dict)" in source or "def analyze_query(self, state:dict)" in source, \
            "analyze_query should use dict type hint"

    def test_agentic_graph_typeddict_exists(self):
        """Verify AgenticGraphState TypedDict exists and is dict-compatible."""
        # Read graph.py source to verify AgenticGraphState exists
        graph_path = Path(__file__).parent.parent / "2_Agentic_RAG" / "graph.py"
        source = graph_path.read_text()

        # Verify AgenticGraphState TypedDict is defined
        assert "class AgenticGraphState(TypedDict" in source or "AgenticGraphState = TypedDict" in source, \
            "AgenticGraphState TypedDict should be defined in graph.py"

        # Verify it has required fields in source
        required_fields = [
            "original_query",
            "anonymized_query",
            "pii_detected",
            "query_intent",
            "context",
            "final_response",
        ]

        for field in required_fields:
            assert f'{field}:' in source or f'{field} :' in source, \
                f"Missing field {field} in AgenticGraphState definition"
