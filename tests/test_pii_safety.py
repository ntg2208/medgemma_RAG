"""Tests for PII safety - ensuring PII failures never pass unredacted text."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPIINodeSafety:
    """Verify PII check node fails safely."""

    def _get_nodes(self, pii_handler):
        """Create RAGNodes with given PII handler."""
        import importlib
        nodes_module = importlib.import_module("agentic_rag.nodes")
        return nodes_module.RAGNodes(
            pii_handler=pii_handler,
            retriever=MagicMock(),
            llm=MagicMock(),
        )

    def test_pii_failure_raises_exception(self):
        """When PII check fails, the node should raise (not pass through unredacted text)."""
        pii_handler = MagicMock()
        pii_handler.anonymize.side_effect = RuntimeError("PII service down")

        nodes = self._get_nodes(pii_handler)
        state = {"original_query": "My NHS number is 123 456 7890", "processing_steps": []}

        with pytest.raises(RuntimeError, match="PII service down"):
            nodes.pii_check(state)

    def test_pii_success_returns_anonymized(self):
        """When PII check succeeds, anonymized text is returned."""
        pii_handler = MagicMock()
        pii_handler.anonymize.return_value = MagicMock(
            anonymized_text="My NHS number is [REDACTED]",
            pii_found=True,
            placeholder_map={"[REDACTED]": "123 456 7890"},
            entities_detected=["NHS_NUMBER"],
        )

        nodes = self._get_nodes(pii_handler)
        state = {"original_query": "My NHS number is 123 456 7890", "processing_steps": []}

        result = nodes.pii_check(state)

        assert result["anonymized_query"] == "My NHS number is [REDACTED]"
        assert result["pii_detected"] is True
        assert "original_query" not in result  # Partial update, not full state

    def test_pii_no_pii_detected(self):
        """When no PII is found, the original query passes through."""
        pii_handler = MagicMock()
        pii_handler.anonymize.return_value = MagicMock(
            anonymized_text="What is CKD stage 3?",
            pii_found=False,
            placeholder_map={},
            entities_detected=[],
        )

        nodes = self._get_nodes(pii_handler)
        state = {"original_query": "What is CKD stage 3?", "processing_steps": []}

        result = nodes.pii_check(state)
        assert result["anonymized_query"] == "What is CKD stage 3?"
        assert result["pii_detected"] is False


class TestOrchestratorPIISafety:
    """Verify orchestrator blocks queries when PII check fails."""

    def test_orchestrator_blocks_on_pii_failure(self):
        """Orchestrator should return safe error when PII handler fails."""
        import importlib

        # Import orchestrator
        orch_module = importlib.import_module("multi_agent_rag.orchestrator")

        pii_handler = MagicMock()
        pii_handler.anonymize.side_effect = RuntimeError("PII service down")

        orchestrator = orch_module.MultiAgentOrchestrator(
            retriever=MagicMock(),
            llm=MagicMock(),
            pii_handler=pii_handler,
        )

        result = orchestrator.process("My NHS number is 123 456 7890")

        assert "privacy safety check failure" in result.answer.lower()
        assert result.confidence == 0.0
        assert result.agents_used == []
