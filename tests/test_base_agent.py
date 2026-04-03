"""Tests for BaseAgent abstract base class."""

from abc import ABC

import importlib.util
from pathlib import Path


def test_base_agent_is_abstract():
    """Verify BaseAgent is an abstract class."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    base_path = agents_path / "base.py"

    spec = importlib.util.spec_from_file_location("base", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load base module")
    base_module = importlib.util.module_from_spec(spec)  # type: ignore[assignment]
    spec.loader.exec_module(base_module)  # type: ignore[attr-defined]

    BaseAgent = base_module.BaseAgent
    assert issubclass(BaseAgent, ABC)


def test_agent_response_has_default_fields():
    """Verify AgentResponse has sensible defaults."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    base_path = agents_path / "base.py"

    spec = importlib.util.spec_from_file_location("base", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load base module")
    base_module = importlib.util.module_from_spec(spec)  # type: ignore[assignment]
    spec.loader.exec_module(base_module)  # type: ignore[attr-defined]

    AgentResponse = base_module.AgentResponse
    response = AgentResponse(answer="test")
    assert response.answer == "test"
    assert response.confidence == 0.0
    assert len(response.disclaimer) > 0
