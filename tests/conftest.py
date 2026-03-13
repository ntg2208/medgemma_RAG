"""Pytest configuration and shared fixtures for the MedGemma RAG test suite."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Mock Components
# ============================================================================

@pytest.fixture
def mock_llm():
    """Mock LLM for testing agent responses."""
    mock = MagicMock()
    mock.generate = MagicMock(return_value="Test response from LLM")
    return mock


@pytest.fixture
def mock_retriever():
    """Mock document retriever for testing."""
    mock = MagicMock()

    class MockDocument:
        def __init__(self):
            self.page_content = "Test document content"
            self.metadata = {
                "source": "Test Source",
                "section": "Introduction",
                "chunk_id": 0,
                "total_chunks": 1,
            }

    mock.invoke = MagicMock(return_value=[MockDocument()])
    mock.with_config = MagicMock(return_value=mock)
    return mock


@pytest.fixture
def mock_pii_handler():
    """Mock PII handler for testing."""
    mock = MagicMock()
    mock.anonymize = MagicMock(return_value="Anonymized query")
    mock.detect = MagicMock(return_value={"pii_detected": False, "pii_map": {}})
    return mock


# ============================================================================
# Agent Fixtures
# ============================================================================

@pytest.fixture
def diet_agent(mock_llm):
    """Create a DietAgent instance for testing."""
    import importlib.util
    from pathlib import Path

    # Clean sys.modules for fresh load
    sys.modules.pop("3_MultiAgent_RAG.agents.base", None)
    sys.modules.pop("3_MultiAgent_RAG.agents.diet_agent", None)

    agents_path = Path(__file__).parent.parent / "3_MultiAgent_RAG" / "agents"
    diet_path = agents_path / "diet_agent.py"

    diet_spec = importlib.util.spec_from_file_location(
        "3_MultiAgent_RAG.agents.diet_agent", diet_path
    )
    diet_module = importlib.util.module_from_spec(diet_spec)  # type: ignore[assignment]
    sys.modules["3_MultiAgent_RAG.agents.diet_agent"] = diet_module
    diet_spec.loader.exec_module(diet_module)  # type: ignore[attr-defined]

    return diet_module.DietAgent(mock_llm)


@pytest.fixture
def lifestyle_agent(mock_llm):
    """Create a LifestyleAgent instance for testing."""
    import importlib.util
    from pathlib import Path

    # Clean sys.modules
    sys.modules.pop("3_MultiAgent_RAG.agents.base", None)
    sys.modules.pop("3_MultiAgent_RAG.agents.lifestyle_agent", None)

    agents_path = Path(__file__).parent.parent / "3_MultiAgent_RAG" / "agents"
    lifestyle_path = agents_path / "lifestyle_agent.py"

    lifestyle_spec = importlib.util.spec_from_file_location(
        "3_MultiAgent_RAG.agents.lifestyle_agent", lifestyle_path
    )
    lifestyle_module = importlib.util.module_from_spec(lifestyle_spec)  # type: ignore[assignment]
    sys.modules["3_MultiAgent_RAG.agents.lifestyle_agent"] = lifestyle_module
    lifestyle_spec.loader.exec_module(lifestyle_module)  # type: ignore[attr-defined]

    return lifestyle_module.LifestyleAgent(mock_llm)


@pytest.fixture
def medication_agent(mock_llm):
    """Create a MedicationAgent instance for testing."""
    import importlib.util
    from pathlib import Path

    # Clean sys.modules
    sys.modules.pop("3_MultiAgent_RAG.agents.base", None)
    sys.modules.pop("3_MultiAgent_RAG.agents.medication_agent", None)

    agents_path = Path(__file__).parent.parent / "3_MultiAgent_RAG" / "agents"
    medication_path = agents_path / "medication_agent.py"

    medication_spec = importlib.util.spec_from_file_location(
        "3_MultiAgent_RAG.agents.medication_agent", medication_path
    )
    medication_module = importlib.util.module_from_spec(medication_spec)  # type: ignore[assignment]
    sys.modules["3_MultiAgent_RAG.agents.medication_agent"] = medication_module
    medication_spec.loader.exec_module(medication_module)  # type: ignore[attr-defined]

    return medication_module.MedicationAgent(mock_llm)


@pytest.fixture
def rag_agent(mock_retriever, mock_llm):
    """Create a RAGAgent instance for testing."""
    import importlib.util
    from pathlib import Path

    # Clean sys.modules
    sys.modules.pop("3_MultiAgent_RAG.agents.base", None)
    sys.modules.pop("3_MultiAgent_RAG.agents.rag_agent", None)

    agents_path = Path(__file__).parent.parent / "3_MultiAgent_RAG" / "agents"
    rag_path = agents_path / "rag_agent.py"

    rag_spec = importlib.util.spec_from_file_location(
        "3_MultiAgent_RAG.agents.rag_agent", rag_path
    )
    rag_module = importlib.util.module_from_spec(rag_spec)  # type: ignore[assignment]
    sys.modules["3_MultiAgent_RAG.agents.rag_agent"] = rag_module
    rag_spec.loader.exec_module(rag_module)  # type: ignore[attr-defined]

    return rag_module.RAGAgent(
        retriever=mock_retriever,
        llm=mock_llm,
    )


# ============================================================================
# Config Fixtures
# ============================================================================

@pytest.fixture
def test_ckd_stage():
    """Provide a valid CKD stage for testing."""
    return 3


@pytest.fixture
def test_weight_kg():
    """Provide a test weight in kg."""
    return 70.0


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest options."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Mark slow tests
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
