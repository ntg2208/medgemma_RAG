"""
Tests for query routing functionality.

Verifies the get_route function correctly routes queries based on intent.
Uses dict state (after state consolidation refactoring).
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from typing import Literal

# Dynamically load 2_Agentic_RAG.nodes module (contains get_route)
spec = importlib.util.spec_from_file_location(
    "nodes",
    Path(__file__).parent.parent / "2_Agentic_RAG" / "nodes.py"
)
nodes_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes_module)
get_route = nodes_module.get_route
QueryIntent = nodes_module.QueryIntent


class TestRouting:
    """Test query routing logic with dict state."""

    def test_route_retrieval_intent(self):
        """Test routing for retrieval intent."""
        state = {"query_intent": QueryIntent.RETRIEVAL}
        result = get_route(state)

        assert result == "retrieval"

    def test_route_direct_intent(self):
        """Test routing for direct intent."""
        state = {"query_intent": QueryIntent.DIRECT}
        result = get_route(state)

        assert result == "direct"

    def test_route_clarification_intent(self):
        """Test routing for clarification intent."""
        state = {"query_intent": QueryIntent.CLARIFICATION}
        result = get_route(state)

        assert result == "clarification"

    def test_route_out_of_scope_intent(self):
        """Test routing for out-of-scope intent."""
        state = {"query_intent": QueryIntent.OUT_OF_SCOPE}
        result = get_route(state)

        assert result == "out_of_scope"


class TestRouteQueryDictCompatibility:
    """Test that routing works with dict state."""

    def test_dict_state_routing(self):
        """Verify get_route accepts dict state."""
        # Test with dict-like state (after refactoring)
        states = [
            ({"query_intent": "retrieval"}, "retrieval"),
            ({"query_intent": "direct"}, "direct"),
            ({"query_intent": "clarification"}, "clarification"),
            ({"query_intent": "out_of_scope"}, "out_of_scope"),
        ]

        for state, expected in states:
            result = get_route(state)
            assert result == expected, f"Expected {expected} for {state}, got {result}"

    def test_route_dict_with_string_intent(self):
        """Test routing with string intent (not QueryIntent enum)."""
        # After refactoring, QueryIntent is just string constants
        state = {"query_intent": "retrieval"}
        result = get_route(state)
        assert result == "retrieval"
