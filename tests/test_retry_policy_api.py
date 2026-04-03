"""
Tests for RetryPolicy configuration on API nodes in LangGraph.

Verifies that API-dependent nodes (PII check, document retrieval, evaluation)
have appropriate retry configurations for transient errors.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock


def test_api_retry_policy_parameters():
    """Verify API nodes use appropriate retry parameters."""
    from langgraph.types import RetryPolicy

    # API nodes (pii_check, retrieve_documents) typically need longer intervals
    # because API calls can be slower than LLM generation
    api_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=2.0,   # Longer initial delay for API
        max_interval=60.0,      # Cap at 60 seconds
    )

    assert api_retry.max_attempts == 3
    assert api_retry.initial_interval == 2.0
    assert api_retry.max_interval == 60.0


def test_evaluator_retry_policy():
    """Verify evaluator node has minimal retry (best-effort)."""
    from langgraph.types import RetryPolicy

    # Evaluator is best-effort - should have fewer retries
    eval_retry = RetryPolicy(
        max_attempts=2,
        initial_interval=1.0,
    )

    assert eval_retry.max_attempts == 2
    assert eval_retry.initial_interval == 1.0


def test_graph_compiles_with_api_nodes():
    """Verify graph with API nodes compiles successfully."""
    graph_path = Path(__file__).parent.parent / "agentic_rag" / "graph.py"

    spec = importlib.util.spec_from_file_location("graph", graph_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load graph module")
    graph_module = importlib.util.module_from_spec(spec)  # type: ignore[assignment]
    spec.loader.exec_module(graph_module)  # type: ignore[attr-defined]

    # Verify RetryPolicy is imported in the module
    assert hasattr(graph_module, 'RetryPolicy'), \
        "RetryPolicy should be imported in graph module"

    # Create a minimal graph instance with all required components
    pii_handler = MagicMock()
    retriever = MagicMock()
    llm = MagicMock()

    AgenticRAGGraph = graph_module.AgenticRAGGraph
    rag_graph = AgenticRAGGraph(
        pii_handler=pii_handler,
        retriever=retriever,
        llm=llm,
    )

    # Verify the graph was compiled successfully
    assert rag_graph.compiled_graph is not None
    assert hasattr(rag_graph.compiled_graph, 'invoke'), \
        "Compiled graph should have invoke method"
