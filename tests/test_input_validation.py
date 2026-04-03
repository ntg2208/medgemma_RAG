"""Tests for input validation at system boundaries."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGraphInputValidation:
    """Test AgenticRAGGraph input validation."""

    def _make_graph(self):
        import importlib
        graph_module = importlib.import_module("agentic_rag.graph")

        pii_handler = MagicMock()
        pii_handler.anonymize.return_value = MagicMock(
            anonymized_text="test", pii_found=False,
            placeholder_map={}, entities_detected=[],
        )

        return graph_module.AgenticRAGGraph(
            pii_handler=pii_handler,
            retriever=MagicMock(),
            llm=MagicMock(generate=MagicMock(return_value="response")),
        )

    def test_empty_query_raises(self):
        graph = self._make_graph()
        with pytest.raises(ValueError, match="non-empty"):
            graph.invoke("")

    def test_whitespace_query_raises(self):
        graph = self._make_graph()
        with pytest.raises(ValueError, match="non-empty"):
            graph.invoke("   ")

    def test_invalid_ckd_stage_zero(self):
        graph = self._make_graph()
        with pytest.raises(ValueError, match="Invalid CKD stage"):
            graph.invoke("test query", ckd_stage=0)

    def test_invalid_ckd_stage_six(self):
        graph = self._make_graph()
        with pytest.raises(ValueError, match="Invalid CKD stage"):
            graph.invoke("test query", ckd_stage=6)

    def test_valid_ckd_stages(self):
        """CKD stages 1-5 should be accepted."""
        graph = self._make_graph()
        for stage in range(1, 6):
            # Should not raise - the graph will process (may error at node level, that's ok)
            try:
                graph.invoke("What is CKD?", ckd_stage=stage)
            except ValueError:
                pytest.fail(f"CKD stage {stage} should be valid")
            except Exception:
                pass  # Other errors from mock setup are fine

    def test_none_ckd_stage_accepted(self):
        graph = self._make_graph()
        try:
            graph.invoke("What is CKD?", ckd_stage=None)
        except ValueError:
            pytest.fail("None ckd_stage should be accepted")
        except Exception:
            pass


class TestOrchestratorInputValidation:
    """Test MultiAgentOrchestrator input validation."""

    def _make_orchestrator(self):
        import importlib
        orch_module = importlib.import_module("multi_agent_rag.orchestrator")
        return orch_module.MultiAgentOrchestrator(
            retriever=MagicMock(),
            llm=MagicMock(generate=MagicMock(return_value="response")),
        )

    def test_empty_query_raises(self):
        orch = self._make_orchestrator()
        with pytest.raises(ValueError, match="non-empty"):
            orch.process("")

    def test_invalid_ckd_stage_raises(self):
        orch = self._make_orchestrator()
        with pytest.raises(ValueError, match="Invalid CKD stage"):
            orch.process("test", ckd_stage=0)

    def test_invalid_ckd_stage_high_raises(self):
        orch = self._make_orchestrator()
        with pytest.raises(ValueError, match="Invalid CKD stage"):
            orch.process("test", ckd_stage=6)
