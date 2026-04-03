"""Tests for RAGAgent extending BaseAgent."""

import importlib.util
from pathlib import Path


def test_rag_agent_extends_base_agent():
    """Verify RAGAgent extends BaseAgent."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    diet_path = agents_path / "diet_agent.py"

    # Clean start - remove from sys.modules
    import sys
    sys.modules.pop("multi_agent_rag.agents.base", None)
    sys.modules.pop("multi_agent_rag.agents.diet_agent", None)
    sys.modules.pop("multi_agent_rag.agents.rag_agent", None)

    # Load diet_agent first (to prime base module)
    diet_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.diet_agent", diet_path)
    diet_module = importlib.util.module_from_spec(diet_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.diet_agent"] = diet_module
    diet_spec.loader.exec_module(diet_module)  # type: ignore[attr-defined]

    # Now load rag_agent
    rag_path = agents_path / "rag_agent.py"
    rag_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.rag_agent", rag_path)
    if rag_spec is None or rag_spec.loader is None:
        raise RuntimeError("Failed to load rag_agent module")
    rag_module = importlib.util.module_from_spec(rag_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.rag_agent"] = rag_module
    rag_spec.loader.exec_module(rag_module)  # type: ignore[attr-defined]
    RAGAgent = rag_module.RAGAgent
    BaseAgent = rag_module.BaseAgent

    assert issubclass(RAGAgent, BaseAgent), "RAGAgent must extend BaseAgent"


def test_rag_agent_has_can_handle():
    """Verify RAGAgent implements can_handle method."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    rag_path = agents_path / "rag_agent.py"

    # Clean start
    import sys
    sys.modules.pop("multi_agent_rag.agents.base", None)
    sys.modules.pop("multi_agent_rag.agents.diet_agent", None)
    sys.modules.pop("multi_agent_rag.agents.rag_agent", None)

    # Load rag_agent
    rag_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.rag_agent", rag_path)
    rag_module = importlib.util.module_from_spec(rag_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.rag_agent"] = rag_module
    rag_spec.loader.exec_module(rag_module)  # type: ignore[attr-defined]
    RAGAgent = rag_module.RAGAgent

    # Create mock retriever and llm
    class MockRetriever:
        def with_config(self, **_kwargs):
            return self
        def invoke(self, _query):
            return []

    class MockLLM:
        def generate(self, _prompt):
            return "Test response"

    agent = RAGAgent(retriever=MockRetriever(), llm=MockLLM())
    assert hasattr(agent, "can_handle"), "RAGAgent must have can_handle method"
    assert callable(agent.can_handle), "can_handle must be callable"

    # Test can_handle with CKD-related queries
    assert agent.can_handle("What is CKD?") is True
    assert agent.can_handle("What are the guidelines?") is True
    assert agent.can_handle("What should I eat?") is False  # DietAgent handles this


def test_rag_agent_answer_returns_agent_response():
    """Verify answer() returns AgentResponse with correct fields."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    rag_path = agents_path / "rag_agent.py"

    # Clean start
    import sys
    sys.modules.pop("multi_agent_rag.agents.base", None)
    sys.modules.pop("multi_agent_rag.agents.diet_agent", None)
    sys.modules.pop("multi_agent_rag.agents.rag_agent", None)

    # Load rag_agent
    rag_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.rag_agent", rag_path)
    rag_module = importlib.util.module_from_spec(rag_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.rag_agent"] = rag_module
    rag_spec.loader.exec_module(rag_module)  # type: ignore[attr-defined]
    RAGAgent = rag_module.RAGAgent
    AgentResponse = rag_module.AgentResponse

    # Create mock retriever and llm
    class MockRetriever:
        def with_config(self, **_kwargs):
            return self
        def invoke(self, _query):
            return []

    class MockLLM:
        def generate(self, _prompt):
            return "Test response"

    agent = RAGAgent(retriever=MockRetriever(), llm=MockLLM())
    response = agent.answer("What is CKD?")

    assert isinstance(response, AgentResponse), "answer() must return AgentResponse"
    assert isinstance(response.answer, str), "answer must be a string"
    assert isinstance(response.confidence, float), "confidence must be a float"
    assert 0.0 <= response.confidence <= 1.0, "confidence must be between 0 and 1"
    assert len(response.disclaimer) > 0, "disclaimer must not be empty"


def test_rag_agent_answer_with_retrieval():
    """Verify answer() works with mocked retrieval."""
    agents_path = Path(__file__).parent.parent / "multi_agent_rag" / "agents"
    rag_path = agents_path / "rag_agent.py"

    # Clean start
    import sys
    sys.modules.pop("multi_agent_rag.agents.base", None)
    sys.modules.pop("multi_agent_rag.agents.diet_agent", None)
    sys.modules.pop("multi_agent_rag.agents.rag_agent", None)

    # Load rag_agent
    rag_spec = importlib.util.spec_from_file_location("multi_agent_rag.agents.rag_agent", rag_path)
    rag_module = importlib.util.module_from_spec(rag_spec)  # type: ignore[assignment]
    sys.modules["multi_agent_rag.agents.rag_agent"] = rag_module
    rag_spec.loader.exec_module(rag_module)  # type: ignore[attr-defined]
    RAGAgent = rag_module.RAGAgent
    AgentResponse = rag_module.AgentResponse

    # Create mock document
    class MockDoc:
        def __init__(self):
            self.page_content = "CKD is chronic kidney disease."
            self.metadata = {"source": "NICE Guidelines", "page_number": 1, "section": "Introduction"}

    class MockRetriever:
        def with_config(self, **kwargs):
            return self
        def invoke(self, query):
            return [MockDoc()]

    class MockLLM:
        def generate(self, prompt):
            return "Chronic Kidney Disease (CKD) is a condition where kidneys gradually lose function over time."

    agent = RAGAgent(retriever=MockRetriever(), llm=MockLLM())
    response = agent.answer("What is CKD?")

    assert isinstance(response, AgentResponse)
    assert "CKD" in response.answer or "kidney" in response.answer.lower()
    assert response.confidence > 0.5
