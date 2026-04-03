"""Tests for DietAgent extending BaseAgent."""

import importlib.util
from pathlib import Path


def test_diet_agent_extends_base_agent():
    """Verify DietAgent extends BaseAgent."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    diet_path = agents_path / "diet_agent.py"

    # Clean start - remove from sys.modules to simulate fresh import
    import sys
    sys.modules.pop("multi_agent_rag.agents.base", None)
    sys.modules.pop("multi_agent_rag.agents.diet_agent", None)

    # Load diet_agent which should also load base
    diet_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.diet_agent", diet_path)
    if diet_spec is None or diet_spec.loader is None:
        raise RuntimeError("Failed to load diet_agent module")
    diet_module = importlib.util.module_from_spec(diet_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.diet_agent"] = diet_module
    diet_spec.loader.exec_module(diet_module)  # type: ignore[attr-defined]
    DietAgent = diet_module.DietAgent

    # Now load base separately and compare
    base_module = sys.modules.get("multi_agent_rag.agents.base")
    if base_module is None:
        raise RuntimeError("base module should be loaded")
    BaseAgent = base_module.BaseAgent

    assert issubclass(DietAgent, BaseAgent), "DietAgent must extend BaseAgent"


def test_diet_agent_has_can_handle():
    """Verify DietAgent implements can_handle method."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    diet_path = agents_path / "diet_agent.py"

    # Clean start - remove from sys.modules
    import sys
    sys.modules.pop("multi_agent_rag.agents.base", None)
    sys.modules.pop("multi_agent_rag.agents.diet_agent", None)

    # Load diet_agent which should also load base
    diet_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.diet_agent", diet_path)
    if diet_spec is None or diet_spec.loader is None:
        raise RuntimeError("Failed to load diet_agent module")
    diet_module = importlib.util.module_from_spec(diet_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.diet_agent"] = diet_module
    diet_spec.loader.exec_module(diet_module)  # type: ignore[attr-defined]
    DietAgent = diet_module.DietAgent

    agent = DietAgent()
    assert hasattr(agent, "can_handle"), "DietAgent must have can_handle method"
    assert callable(agent.can_handle), "can_handle must be callable"

    # Test can_handle with diet-related queries
    assert agent.can_handle("What should I eat?") is True
    assert agent.can_handle("How much potassium?") is True
    assert agent.can_handle("What is my blood pressure?") is False


def test_diet_agent_answer_returns_agent_response():
    """Verify answer() returns AgentResponse with correct fields."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    diet_path = agents_path / "diet_agent.py"

    # Clean start - remove from sys.modules
    import sys
    sys.modules.pop("multi_agent_rag.agents.base", None)
    sys.modules.pop("multi_agent_rag.agents.diet_agent", None)

    # Load diet_agent which should also load base
    diet_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.diet_agent", diet_path)
    if diet_spec is None or diet_spec.loader is None:
        raise RuntimeError("Failed to load diet_agent module")
    diet_module = importlib.util.module_from_spec(diet_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.diet_agent"] = diet_module
    diet_spec.loader.exec_module(diet_module)  # type: ignore[attr-defined]
    DietAgent = diet_module.DietAgent
    AgentResponse = diet_module.AgentResponse

    agent = DietAgent()
    response = agent.answer("What should I eat?", ckd_stage=3)

    assert isinstance(response, AgentResponse), "answer() must return AgentResponse"
    assert isinstance(response.answer, str), "answer must be a string"
    assert isinstance(response.confidence, float), "confidence must be a float"
    assert 0.0 <= response.confidence <= 1.0, "confidence must be between 0 and 1"
    assert len(response.disclaimer) > 0, "disclaimer must not be empty"


def test_diet_agent_calculate_preserved():
    """Verify calculate() method is preserved for direct access."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    diet_path = agents_path / "diet_agent.py"

    # Clean start - remove from sys.modules
    import sys
    sys.modules.pop("multi_agent_rag.agents.base", None)
    sys.modules.pop("multi_agent_rag.agents.diet_agent", None)

    # Load diet_agent which should also load base
    diet_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.diet_agent", diet_path)
    if diet_spec is None or diet_spec.loader is None:
        raise RuntimeError("Failed to load diet_agent module")
    diet_module = importlib.util.module_from_spec(diet_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.diet_agent"] = diet_module
    diet_spec.loader.exec_module(diet_module)  # type: ignore[attr-defined]
    DietAgent = diet_module.DietAgent

    agent = DietAgent()
    response = agent.calculate(ckd_stage=3, weight_kg=70)

    assert response is not None, "calculate() must return a response"
    assert hasattr(response, "recommendations"), "Response must have recommendations"
    assert hasattr(response, "summary"), "Response must have summary"
    assert hasattr(response, "ckd_stage"), "Response must have ckd_stage"
    assert hasattr(response, "confidence"), "Response must have confidence"
