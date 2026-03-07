"""Tests for LifestyleAgent extending BaseAgent."""

import importlib.util
from pathlib import Path


def test_lifestyle_agent_extends_base_agent():
    """Verify LifestyleAgent extends BaseAgent."""
    agents_path = Path(__file__).parent.parent / "3_MultiAgent_RAG" / "agents"
    diet_path = agents_path / "diet_agent.py"

    # Clean start - remove from sys.modules
    import sys
    sys.modules.pop("3_MultiAgent_RAG.agents.base", None)
    sys.modules.pop("3_MultiAgent_RAG.agents.diet_agent", None)
    sys.modules.pop("3_MultiAgent_RAG.agents.lifestyle_agent", None)

    # Load diet_agent first (to prime base module)
    diet_spec = importlib.util.spec_from_file_location("3_MultiAgent_RAG.agents.diet_agent", diet_path)
    diet_module = importlib.util.module_from_spec(diet_spec)  # type: ignore[assignment]
    sys.modules["3_MultiAgent_RAG.agents.diet_agent"] = diet_module
    diet_spec.loader.exec_module(diet_module)  # type: ignore[attr-defined]

    # Now load lifestyle_agent
    lifestyle_path = agents_path / "lifestyle_agent.py"
    lifestyle_spec = importlib.util.spec_from_file_location("3_MultiAgent_RAG.agents.lifestyle_agent", lifestyle_path)
    if lifestyle_spec is None or lifestyle_spec.loader is None:
        raise RuntimeError("Failed to load lifestyle_agent module")
    lifestyle_module = importlib.util.module_from_spec(lifestyle_spec)  # type: ignore[assignment]
    sys.modules["3_MultiAgent_RAG.agents.lifestyle_agent"] = lifestyle_module
    lifestyle_spec.loader.exec_module(lifestyle_module)  # type: ignore[attr-defined]
    LifestyleAgent = lifestyle_module.LifestyleAgent
    BaseAgent = lifestyle_module.BaseAgent

    assert issubclass(LifestyleAgent, BaseAgent), "LifestyleAgent must extend BaseAgent"


def test_lifestyle_agent_has_can_handle():
    """Verify LifestyleAgent implements can_handle method."""
    agents_path = Path(__file__).parent.parent / "3_MultiAgent_RAG" / "agents"
    lifestyle_path = agents_path / "lifestyle_agent.py"

    # Clean start
    import sys
    sys.modules.pop("3_MultiAgent_RAG.agents.base", None)
    sys.modules.pop("3_MultiAgent_RAG.agents.diet_agent", None)
    sys.modules.pop("3_MultiAgent_RAG.agents.lifestyle_agent", None)

    # Load lifestyle_agent
    lifestyle_spec = importlib.util.spec_from_file_location("3_MultiAgent_RAG.agents.lifestyle_agent", lifestyle_path)
    lifestyle_module = importlib.util.module_from_spec(lifestyle_spec)  # type: ignore[assignment]
    sys.modules["3_MultiAgent_RAG.agents.lifestyle_agent"] = lifestyle_module
    lifestyle_spec.loader.exec_module(lifestyle_module)  # type: ignore[attr-defined]
    LifestyleAgent = lifestyle_module.LifestyleAgent

    agent = LifestyleAgent()
    assert hasattr(agent, "can_handle"), "LifestyleAgent must have can_handle method"
    assert callable(agent.can_handle), "can_handle must be callable"

    # Test can_handle with lifestyle-related queries
    assert agent.can_handle("How much exercise do I need?") is True
    assert agent.can_handle("What about smoking?") is True
    assert agent.can_handle("How much potassium?") is False  # DietAgent handles this


def test_lifestyle_agent_answer_returns_agent_response():
    """Verify answer() returns AgentResponse with correct fields."""
    agents_path = Path(__file__).parent.parent / "3_MultiAgent_RAG" / "agents"
    lifestyle_path = agents_path / "lifestyle_agent.py"

    # Clean start
    import sys
    sys.modules.pop("3_MultiAgent_RAG.agents.base", None)
    sys.modules.pop("3_MultiAgent_RAG.agents.diet_agent", None)
    sys.modules.pop("3_MultiAgent_RAG.agents.lifestyle_agent", None)

    # Load lifestyle_agent
    lifestyle_spec = importlib.util.spec_from_file_location("3_MultiAgent_RAG.agents.lifestyle_agent", lifestyle_path)
    lifestyle_module = importlib.util.module_from_spec(lifestyle_spec)  # type: ignore[assignment]
    sys.modules["3_MultiAgent_RAG.agents.lifestyle_agent"] = lifestyle_module
    lifestyle_spec.loader.exec_module(lifestyle_module)  # type: ignore[attr-defined]
    LifestyleAgent = lifestyle_module.LifestyleAgent
    AgentResponse = lifestyle_module.AgentResponse

    agent = LifestyleAgent()
    response = agent.answer("How much exercise do I need?")

    assert isinstance(response, AgentResponse), "answer() must return AgentResponse"
    assert isinstance(response.answer, str), "answer must be a string"
    assert isinstance(response.confidence, float), "confidence must be a float"
    assert 0.0 <= response.confidence <= 1.0, "confidence must be between 0 and 1"
    assert len(response.disclaimer) > 0, "disclaimer must not be empty"


def test_lifestyle_agent_guidance_preserved():
    """Verify get_guidance method is preserved for direct access."""
    agents_path = Path(__file__).parent.parent / "3_MultiAgent_RAG" / "agents"
    lifestyle_path = agents_path / "lifestyle_agent.py"

    # Clean start
    import sys
    sys.modules.pop("3_MultiAgent_RAG.agents.base", None)
    sys.modules.pop("3_MultiAgent_RAG.agents.diet_agent", None)
    sys.modules.pop("3_MultiAgent_RAG.agents.lifestyle_agent", None)

    # Load lifestyle_agent
    lifestyle_spec = importlib.util.spec_from_file_location("3_MultiAgent_RAG.agents.lifestyle_agent", lifestyle_path)
    lifestyle_module = importlib.util.module_from_spec(lifestyle_spec)  # type: ignore[assignment]
    sys.modules["3_MultiAgent_RAG.agents.lifestyle_agent"] = lifestyle_module
    lifestyle_spec.loader.exec_module(lifestyle_module)  # type: ignore[attr-defined]
    LifestyleAgent = lifestyle_module.LifestyleAgent

    agent = LifestyleAgent()
    rec = agent.get_guidance(lifestyle_module.LifestyleCategory.EXERCISE, ckd_stage=3)

    assert rec is not None, "get_guidance must return a recommendation"
    assert hasattr(rec, "category"), "Recommendation must have category"
    assert hasattr(rec, "title"), "Recommendation must have title"
    assert hasattr(rec, "guidance"), "Recommendation must have guidance"
    assert hasattr(rec, "tips"), "Recommendation must have tips"
