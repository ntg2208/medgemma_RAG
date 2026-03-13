"""Tests for the MultiAgentOrchestrator."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
orch_module = importlib.import_module("3_MultiAgent_RAG.orchestrator")
MultiAgentOrchestrator = orch_module.MultiAgentOrchestrator
AgentType = orch_module.AgentType


class TestOrchestratorRouting:
    """Test query routing to appropriate agents."""

    def _make_orchestrator(self):
        return MultiAgentOrchestrator(
            retriever=MagicMock(),
            llm=MagicMock(generate=MagicMock(return_value="response")),
        )

    def test_diet_query_routes_to_diet(self):
        orch = self._make_orchestrator()
        decision = orch.route("What foods should I avoid with kidney disease?")
        assert decision.primary_agent == AgentType.DIET

    def test_medication_query_routes_to_medication(self):
        orch = self._make_orchestrator()
        decision = orch.route("Is ibuprofen safe for kidney patients?")
        assert decision.primary_agent == AgentType.MEDICATION

    def test_lifestyle_query_routes_to_lifestyle(self):
        orch = self._make_orchestrator()
        decision = orch.route("What exercise is good for blood pressure?")
        assert decision.primary_agent == AgentType.LIFESTYLE

    def test_general_query_routes_to_rag(self):
        orch = self._make_orchestrator()
        decision = orch.route("What is CKD?")
        assert decision.primary_agent == AgentType.RAG

    def test_no_match_defaults_to_rag(self):
        orch = self._make_orchestrator()
        decision = orch.route("Hello there")
        assert decision.primary_agent == AgentType.RAG

    def test_routing_has_confidence(self):
        orch = self._make_orchestrator()
        decision = orch.route("What diet for stage 3 kidney disease?")
        assert 0.0 <= decision.confidence <= 1.0

    def test_routing_has_reasoning(self):
        orch = self._make_orchestrator()
        decision = orch.route("What diet for CKD?")
        assert len(decision.reasoning) > 0


class TestOrchestratorProcess:
    """Test full query processing."""

    def _make_orchestrator(self):
        return MultiAgentOrchestrator(
            retriever=MagicMock(invoke=MagicMock(return_value=[])),
            llm=MagicMock(generate=MagicMock(return_value="test response")),
        )

    def test_process_returns_orchestrator_response(self):
        orch = self._make_orchestrator()
        result = orch.process("What potassium foods to avoid with CKD?")
        assert hasattr(result, "answer")
        assert hasattr(result, "agents_used")
        assert hasattr(result, "confidence")
        assert hasattr(result, "disclaimer")

    def test_process_populates_agents_used(self):
        orch = self._make_orchestrator()
        result = orch.process("What potassium foods to limit in stage 3 kidney disease?")
        assert len(result.agents_used) > 0

    def test_process_with_ckd_stage(self):
        orch = self._make_orchestrator()
        result = orch.process("What potassium foods to avoid?", ckd_stage=3)
        assert result.ckd_stage == 3

    def test_process_answer_not_empty(self):
        orch = self._make_orchestrator()
        result = orch.process("What are dietary restrictions for kidney disease?")
        assert len(result.answer) > 0

    def test_process_includes_disclaimer(self):
        orch = self._make_orchestrator()
        result = orch.process("What foods to avoid with CKD?")
        assert "medical advice" in result.disclaimer.lower() or "educational" in result.disclaimer.lower()
