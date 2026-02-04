"""
RAGAS evaluation for the CKD RAG System.

Implements standard RAG evaluation metrics:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Does it address the question?
- Context Precision: Are retrieved docs relevant?
- Context Recall: Are all relevant docs retrieved?
"""

import logging
from typing import Optional
from dataclasses import dataclass

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class RAGASScores:
    """Container for RAGAS evaluation scores."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }

    @property
    def average(self) -> float:
        """Calculate average score."""
        return (
            self.faithfulness +
            self.answer_relevancy +
            self.context_precision +
            self.context_recall
        ) / 4


class RAGASEvaluator:
    """
    RAGAS-based evaluation for RAG responses.

    Evaluates response quality across multiple dimensions
    using the RAGAS framework.

    Example:
        >>> evaluator = RAGASEvaluator()
        >>> scores = evaluator.evaluate(
        ...     query="What are potassium restrictions?",
        ...     response="Limit potassium to 2000-3000mg daily.",
        ...     contexts=["CKD stage 3 patients should limit potassium..."],
        ... )
        >>> print(scores.faithfulness)
    """

    def __init__(
        self,
        metrics: Optional[list] = None,
        llm: Optional[any] = None,
        embeddings: Optional[any] = None,
    ):
        """
        Initialize the RAGAS evaluator.

        Args:
            metrics: List of RAGAS metrics to use (default: all four)
            llm: LLM for evaluation (uses RAGAS default if None)
            embeddings: Embeddings for evaluation (uses RAGAS default if None)
        """
        self.metrics = metrics or [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        self.llm = llm
        self.embeddings = embeddings

        logger.info(f"RAGASEvaluator initialized with {len(self.metrics)} metrics")

    def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> RAGASScores:
        """
        Evaluate a single RAG response.

        Args:
            query: User question
            response: Generated answer
            contexts: Retrieved context passages
            ground_truth: Optional ground truth answer (for recall)

        Returns:
            RAGASScores with evaluation results
        """
        # Prepare dataset for RAGAS
        data = {
            "question": [query],
            "answer": [response],
            "contexts": [contexts],
        }

        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)

        try:
            # Run evaluation
            eval_kwargs = {"dataset": dataset, "metrics": self.metrics}

            if self.llm:
                eval_kwargs["llm"] = self.llm
            if self.embeddings:
                eval_kwargs["embeddings"] = self.embeddings

            results = evaluate(**eval_kwargs)

            # Extract scores
            scores = RAGASScores(
                faithfulness=results.get("faithfulness", 0.0),
                answer_relevancy=results.get("answer_relevancy", 0.0),
                context_precision=results.get("context_precision", 0.0),
                context_recall=results.get("context_recall", 0.0) if ground_truth else 0.0,
            )

            logger.info(f"RAGAS evaluation complete: avg={scores.average:.3f}")
            return scores

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            # Return zero scores on failure
            return RAGASScores(
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_precision=0.0,
                context_recall=0.0,
            )

    def evaluate_batch(
        self,
        queries: list[str],
        responses: list[str],
        contexts_list: list[list[str]],
        ground_truths: Optional[list[str]] = None,
    ) -> list[RAGASScores]:
        """
        Evaluate multiple RAG responses.

        Args:
            queries: List of user questions
            responses: List of generated answers
            contexts_list: List of context lists
            ground_truths: Optional list of ground truth answers

        Returns:
            List of RAGASScores
        """
        # Prepare batch dataset
        data = {
            "question": queries,
            "answer": responses,
            "contexts": contexts_list,
        }

        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        try:
            eval_kwargs = {"dataset": dataset, "metrics": self.metrics}

            if self.llm:
                eval_kwargs["llm"] = self.llm
            if self.embeddings:
                eval_kwargs["embeddings"] = self.embeddings

            results = evaluate(**eval_kwargs)

            # Convert to list of scores
            scores_list = []
            result_df = results.to_pandas()

            for _, row in result_df.iterrows():
                scores_list.append(RAGASScores(
                    faithfulness=row.get("faithfulness", 0.0),
                    answer_relevancy=row.get("answer_relevancy", 0.0),
                    context_precision=row.get("context_precision", 0.0),
                    context_recall=row.get("context_recall", 0.0) if ground_truths else 0.0,
                ))

            return scores_list

        except Exception as e:
            logger.error(f"Batch RAGAS evaluation failed: {e}")
            return [RAGASScores(0.0, 0.0, 0.0, 0.0) for _ in queries]

    def get_metrics_description(self) -> dict[str, str]:
        """Get descriptions of the evaluation metrics."""
        return {
            "faithfulness": (
                "Measures how factually accurate the answer is based on the context. "
                "Higher scores mean the answer makes claims supported by the retrieved documents."
            ),
            "answer_relevancy": (
                "Measures how relevant the answer is to the question. "
                "Higher scores mean the answer directly addresses what was asked."
            ),
            "context_precision": (
                "Measures whether the retrieved documents are relevant to the question. "
                "Higher scores mean the retrieval found appropriate documents."
            ),
            "context_recall": (
                "Measures whether all relevant information was retrieved. "
                "Requires ground truth answers for comparison."
            ),
        }


def create_evaluator(
    llm: Optional[any] = None,
    embeddings: Optional[any] = None,
) -> RAGASEvaluator:
    """
    Factory function to create a RAGAS evaluator.

    Args:
        llm: Optional LLM for evaluation
        embeddings: Optional embeddings for evaluation

    Returns:
        Configured RAGASEvaluator
    """
    return RAGASEvaluator(llm=llm, embeddings=embeddings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("RAGAS Evaluator module loaded.")
    print("\nMetrics available:")

    evaluator = RAGASEvaluator()
    for metric, desc in evaluator.get_metrics_description().items():
        print(f"\n{metric}:")
        print(f"  {desc}")
