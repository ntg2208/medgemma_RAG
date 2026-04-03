"""Tests for MedicationAgent extending BaseAgent."""

import importlib.util
from pathlib import Path


def test_medication_agent_extends_base_agent():
    """Verify MedicationAgent extends BaseAgent."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    diet_path = agents_path / "diet_agent.py"

    # Clean start - remove from sys.modules
    import sys
    sys.modules.pop("multi_agent_rag.agents.base", None)
    sys.modules.pop("multi_agent_rag.agents.diet_agent", None)
    sys.modules.pop("multi_agent_rag.agents.medication_agent", None)

    # Load diet_agent first (to prime base module)
    diet_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.diet_agent", diet_path)
    diet_module = importlib.util.module_from_spec(diet_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.diet_agent"] = diet_module
    diet_spec.loader.exec_module(diet_module)  # type: ignore[attr-defined]

    # Now load medication_agent
    medication_path = agents_path / "medication_agent.py"
    medication_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.medication_agent", medication_path)
    if medication_spec is None or medication_spec.loader is None:
        raise RuntimeError("Failed to load medication_agent module")
    medication_module = importlib.util.module_from_spec(medication_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.medication_agent"] = medication_module
    medication_spec.loader.exec_module(medication_module)  # type: ignore[attr-defined]
    MedicationAgent = medication_module.MedicationAgent
    BaseAgent = medication_module.BaseAgent

    assert issubclass(MedicationAgent, BaseAgent), "MedicationAgent must extend BaseAgent"


def test_medication_agent_has_can_handle():
    """Verify MedicationAgent implements can_handle method."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    medication_path = agents_path / "medication_agent.py"

    # Clean start
    import sys
    sys.modules.pop("multi_agent_rag.agents.base", None)
    sys.modules.pop("multi_agent_rag.agents.diet_agent", None)
    sys.modules.pop("multi_agent_rag.agents.medication_agent", None)

    # Load medication_agent
    medication_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.medication_agent", medication_path)
    medication_module = importlib.util.module_from_spec(medication_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.medication_agent"] = medication_module
    medication_spec.loader.exec_module(medication_module)  # type: ignore[attr-defined]
    MedicationAgent = medication_module.MedicationAgent

    agent = MedicationAgent()
    assert hasattr(agent, "can_handle"), "MedicationAgent must have can_handle method"
    assert callable(agent.can_handle), "can_handle must be callable"

    # Test can_handle with medication-related queries
    assert agent.can_handle("Is ibuprofen safe?") is True
    assert agent.can_handle("What painkillers can I take?") is True
    assert agent.can_handle("How much protein?") is False  # DietAgent handles this


def test_medication_agent_answer_returns_agent_response():
    """Verify answer() returns AgentResponse with correct fields."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    medication_path = agents_path / "medication_agent.py"

    # Clean start
    import sys
    sys.modules.pop("multi_agent_rag.agents.base", None)
    sys.modules.pop("multi_agent_rag.agents.diet_agent", None)
    sys.modules.pop("multi_agent_rag.agents.medication_agent", None)

    # Load medication_agent
    medication_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.medication_agent", medication_path)
    medication_module = importlib.util.module_from_spec(medication_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.medication_agent"] = medication_module
    medication_spec.loader.exec_module(medication_module)  # type: ignore[attr-defined]
    MedicationAgent = medication_module.MedicationAgent
    AgentResponse = medication_module.AgentResponse

    agent = MedicationAgent()
    response = agent.answer("Is ibuprofen safe for CKD?")

    assert isinstance(response, AgentResponse), "answer() must return AgentResponse"
    assert isinstance(response.answer, str), "answer must be a string"
    assert isinstance(response.confidence, float), "confidence must be a float"
    assert 0.0 <= response.confidence <= 1.0, "confidence must be between 0 and 1"
    assert len(response.disclaimer) > 0, "disclaimer must not be empty"


def test_medication_agent_check_preserved():
    """Verify check() method is preserved for direct access."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    medication_path = agents_path / "medication_agent.py"

    # Clean start
    import sys
    sys.modules.pop("multi_agent_rag.agents.base", None)
    sys.modules.pop("multi_agent_rag.agents.diet_agent", None)
    sys.modules.pop("multi_agent_rag.agents.medication_agent", None)

    # Load medication_agent
    medication_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.medication_agent", medication_path)
    medication_module = importlib.util.module_from_spec(medication_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.medication_agent"] = medication_module
    medication_spec.loader.exec_module(medication_module)  # type: ignore[attr-defined]
    MedicationAgent = medication_module.MedicationAgent

    agent = MedicationAgent()
    response = agent.check("ibuprofen", ckd_stage=3)

    assert response is not None, "check() must return a response"
    assert hasattr(response, "medications_analyzed"), "Response must have medications_analyzed"
    assert hasattr(response, "general_guidance"), "Response must have general_guidance"
    assert hasattr(response, "warnings"), "Response must have warnings"
    assert hasattr(response, "confidence"), "Response must have confidence"
