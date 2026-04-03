"""
Tests for RAGAS evaluation module.

These tests verify data structures, sample creation, and evaluator initialization
WITHOUT running actual RAGAS evaluation (which requires a judge LLM and is expensive).
"""

import pytest
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRAGASScores:
    """Test the RAGASScores dataclass."""

    def test_scores_creation(self):
        from importlib import import_module

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        scores = ragas_mod.RAGASScores(
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7,
            context_recall=0.6,
        )
        assert scores.faithfulness == 0.8
        assert scores.answer_relevancy == 0.9
        assert scores.context_precision == 0.7
        assert scores.context_recall == 0.6

    def test_scores_average(self):
        from importlib import import_module

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        scores = ragas_mod.RAGASScores(
            faithfulness=0.8,
            answer_relevancy=0.8,
            context_precision=0.8,
            context_recall=0.8,
        )
        assert scores.average == pytest.approx(0.8)

    def test_scores_average_mixed(self):
        from importlib import import_module

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        scores = ragas_mod.RAGASScores(
            faithfulness=1.0,
            answer_relevancy=0.5,
            context_precision=0.75,
            context_recall=0.25,
        )
        assert scores.average == pytest.approx(0.625)

    def test_scores_to_dict(self):
        from importlib import import_module

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        scores = ragas_mod.RAGASScores(
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7,
            context_recall=0.6,
        )
        d = scores.to_dict()
        assert d == {
            "faithfulness": 0.8,
            "answer_relevancy": 0.9,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }

    def test_zero_scores(self):
        from importlib import import_module

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        scores = ragas_mod.RAGASScores(0.0, 0.0, 0.0, 0.0)
        assert scores.average == 0.0


class TestBuildSample:
    """Test SingleTurnSample creation."""

    def test_basic_sample(self):
        from importlib import import_module

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        sample = ragas_mod.build_sample(
            query="What is CKD?",
            response="CKD is chronic kidney disease.",
            contexts=["Chronic kidney disease (CKD) is a condition..."],
        )
        assert sample.user_input == "What is CKD?"
        assert sample.response == "CKD is chronic kidney disease."
        assert sample.retrieved_contexts == [
            "Chronic kidney disease (CKD) is a condition..."
        ]
        assert sample.reference is None

    def test_sample_with_reference(self):
        from importlib import import_module

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        sample = ragas_mod.build_sample(
            query="What is CKD?",
            response="CKD is chronic kidney disease.",
            contexts=["CKD is a condition..."],
            reference="Chronic kidney disease is a progressive loss of kidney function.",
        )
        assert sample.reference == "Chronic kidney disease is a progressive loss of kidney function."

    def test_sample_multiple_contexts(self):
        from importlib import import_module

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        contexts = [
            "Context 1: Potassium limits for CKD...",
            "Context 2: Dietary guidance for stage 3...",
            "Context 3: NICE NG203 recommendations...",
        ]
        sample = ragas_mod.build_sample(
            query="Potassium limits?",
            response="Limit to 2000-3000mg/day.",
            contexts=contexts,
        )
        assert len(sample.retrieved_contexts) == 3

    def test_sample_is_valid_single_turn_sample(self):
        """Verify the sample is a proper RAGAS SingleTurnSample instance."""
        from importlib import import_module
        from ragas.dataset_schema import SingleTurnSample

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        sample = ragas_mod.build_sample(
            query="Test?",
            response="Test answer.",
            contexts=["Test context."],
        )
        assert isinstance(sample, SingleTurnSample)


class TestEvaluationDataset:
    """Test that samples can be assembled into an EvaluationDataset."""

    def test_dataset_creation(self):
        from importlib import import_module
        from ragas.dataset_schema import EvaluationDataset

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")

        samples = [
            ragas_mod.build_sample(
                query=f"Question {i}?",
                response=f"Answer {i}.",
                contexts=[f"Context {i}."],
            )
            for i in range(3)
        ]
        dataset = EvaluationDataset(samples=samples)
        assert len(dataset) == 3


class TestRAGASEvaluatorInit:
    """Test evaluator initialization (no actual evaluation)."""

    def test_default_metrics(self):
        from importlib import import_module

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        evaluator = ragas_mod.RAGASEvaluator()
        assert len(evaluator.metrics) == 4

    def test_custom_metrics(self):
        from importlib import import_module
        from ragas.metrics import Faithfulness

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        evaluator = ragas_mod.RAGASEvaluator(metrics=[Faithfulness()])
        assert len(evaluator.metrics) == 1

    def test_metrics_description(self):
        from importlib import import_module

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        desc = ragas_mod.RAGASEvaluator.get_metrics_description()
        assert "faithfulness" in desc
        assert "answer_relevancy" in desc
        assert "context_precision" in desc
        assert "context_recall" in desc

    def test_factory_function(self):
        from importlib import import_module

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")
        evaluator = ragas_mod.create_evaluator()
        assert isinstance(evaluator, ragas_mod.RAGASEvaluator)

    def test_evaluate_returns_zero_scores_on_failure(self):
        """When judge LLM is unavailable, evaluate should return zero scores."""
        from importlib import import_module

        ragas_mod = import_module("agentic_rag.evaluation.ragas_eval")

        # Create evaluator without valid API key — will fail on evaluate
        evaluator = ragas_mod.RAGASEvaluator()

        # Mock _ensure_llm to raise so we test the error path
        evaluator._ensure_llm = MagicMock(side_effect=ValueError("No API key"))

        scores = evaluator.evaluate(
            query="Test?",
            response="Test.",
            contexts=["Context."],
        )
        assert scores.faithfulness == 0.0
        assert scores.average == 0.0
