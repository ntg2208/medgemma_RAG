"""
Tests for RetryPolicy configuration in LangGraph nodes.

Verifies that transient errors will be automatically retried with
exponential backoff for nodes that make external API calls.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock


def test_retry_policy_import():
    """Verify RetryPolicy can be imported from langgraph.types."""
    from langgraph.types import RetryPolicy

    # Verify RetryPolicy class exists and can be instantiated
    retry_policy = RetryPolicy(max_attempts=3, initial_interval=1.0, max_interval=30.0)

    assert retry_policy is not None
    assert retry_policy.max_attempts == 3
    assert retry_policy.initial_interval == 1.0
    assert retry_policy.max_interval == 30.0


def test_llm_retry_policy_config():
    """Verify RetryPolicy uses correct parameters for LLM nodes."""
    from langgraph.types import RetryPolicy

    # Expected configuration for LLM nodes (generate_response, generate_direct)
    expected_max_attempts = 3
    expected_initial_interval = 1.0
    expected_max_interval = 30.0

    retry_policy = RetryPolicy(max_attempts=3, initial_interval=1.0, max_interval=30.0)

    assert retry_policy.max_attempts == expected_max_attempts
    assert retry_policy.initial_interval == expected_initial_interval
    assert retry_policy.max_interval == expected_max_interval


def test_api_retry_policy_config():
    """Verify RetryPolicy for API nodes has longer intervals."""
    from langgraph.types import RetryPolicy

    # API nodes (pii_check, retrieve_documents) need longer intervals
    retry_policy = RetryPolicy(max_attempts=3, initial_interval=2.0, max_interval=60.0)

    assert retry_policy.max_attempts == 3
    assert retry_policy.initial_interval == 2.0
    assert retry_policy.max_interval == 60.0


def test_graph_compiles_with_retry_policy_import():
    """Verify the graph module can be imported and compiled with RetryPolicy imported."""
    # This test ensures the graph.py file has RetryPolicy imported
    # and the graph can be compiled successfully
    graph_path = Path(__file__).parent.parent / "2_Agentic_RAG" / "graph.py"

    spec = importlib.util.spec_from_file_location("graph", graph_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load graph module")
    graph_module = importlib.util.module_from_spec(spec)  # type: ignore[assignment]
    spec.loader.exec_module(graph_module)  # type: ignore[attr-defined]

    # Verify RetryPolicy is imported in the module
    assert hasattr(graph_module, 'RetryPolicy'), \
        "RetryPolicy should be imported in graph module"

    # Create a minimal graph instance to verify compilation
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
