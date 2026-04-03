"""
RAGAS evaluation for the CKD RAG System.

Updated for RAGAS v0.4.x API using SingleTurnSample and EvaluationDataset.

Implements standard RAG evaluation metrics:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Does it address the question?
- Context Precision: Are retrieved docs relevant?
- Context Recall: Are all relevant docs retrieved?

Requires a judge LLM (any OpenAI-compatible API: Gemini, OpenRouter, OpenAI).
Configure via environment variables:
    RAGAS_JUDGE_MODEL, RAGAS_JUDGE_API_KEY, RAGAS_JUDGE_BASE_URL
"""

import logging
from dataclasses import dataclass
from typing import Optional

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    ResponseRelevancy,
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class RAGASScores:
    """Container for RAGAS evaluation scores."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    def to_dict(self) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }

    @property
    def average(self) -> float:
        return (
            self.faithfulness
            + self.answer_relevancy
            + self.context_precision
            + self.context_recall
        ) / 4


def _get_judge_llm():
    """Create a LangChain ChatOpenAI for RAGAS judge duties.

    Uses config values: RAGAS_JUDGE_MODEL, RAGAS_JUDGE_API_KEY, RAGAS_JUDGE_BASE_URL.
    Works with OpenRouter, Gemini, OpenAI, or any OpenAI-compatible endpoint.
    """
    from langchain_openai import ChatOpenAI
    from config import RAGAS_JUDGE_MODEL, RAGAS_JUDGE_API_KEY, RAGAS_JUDGE_BASE_URL

    if not RAGAS_JUDGE_API_KEY:
        raise ValueError(
            "RAGAS_JUDGE_API_KEY is required. Set it in .env or environment. "
            "Get a free key at https://openrouter.ai/ or use your Gemini/OpenAI key."
        )

    return ChatOpenAI(
        model=RAGAS_JUDGE_MODEL,
        api_key=RAGAS_JUDGE_API_KEY,
        base_url=RAGAS_JUDGE_BASE_URL,
        temperature=0,
    )


def _get_judge_embeddings():
    """Create embeddings for RAGAS answer relevancy metric.

    Uses OpenAI-compatible embedding endpoint from the same provider.
    """
    from langchain_openai import OpenAIEmbeddings
    from config import RAGAS_EMBEDDINGS_MODEL, RAGAS_JUDGE_API_KEY, RAGAS_JUDGE_BASE_URL

    return OpenAIEmbeddings(
        model=RAGAS_EMBEDDINGS_MODEL,
        api_key=RAGAS_JUDGE_API_KEY,
        base_url=RAGAS_JUDGE_BASE_URL,
    )


def build_sample(
    query: str,
    response: str,
    contexts: list[str],
    reference: Optional[str] = None,
) -> SingleTurnSample:
    """Build a RAGAS SingleTurnSample from RAG pipeline outputs.

    Args:
        query: User question.
        response: Generated answer.
        contexts: Retrieved context passages.
        reference: Optional ground truth answer (needed for context_recall).

    Returns:
        SingleTurnSample ready for evaluation.
    """
    kwargs = {
        "user_input": query,
        "response": response,
        "retrieved_contexts": contexts,
    }
    if reference is not None:
        kwargs["reference"] = reference
    return SingleTurnSample(**kwargs)


class RAGASEvaluator:
    """
    RAGAS-based evaluation for RAG responses (v0.4.x API).

    Uses a judge LLM (Gemini/OpenRouter/OpenAI) to score responses on
    faithfulness, relevancy, context precision, and context recall.

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
        llm=None,
        embeddings=None,
        metrics: Optional[list] = None,
    ):
        """Initialize the RAGAS evaluator.

        Args:
            llm: LangChain LLM for judge duties (auto-created from config if None).
            embeddings: LangChain embeddings for relevancy metric (auto-created if None).
            metrics: Custom list of RAGAS metric instances. Defaults to the standard four.
        """
        self._llm_raw = llm
        self._embeddings_raw = embeddings

        # Metrics — instantiate fresh so each evaluator gets its own config
        self.metrics = metrics or [
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
            LLMContextRecall(),
        ]

        logger.info(
            f"RAGASEvaluator initialized with {len(self.metrics)} metrics"
        )

    def _ensure_llm(self):
        """Lazy-init judge LLM and wrap for RAGAS."""
        if self._llm_raw is None:
            self._llm_raw = _get_judge_llm()
        return LangchainLLMWrapper(self._llm_raw)

    def _ensure_embeddings(self):
        """Lazy-init embeddings and wrap for RAGAS."""
        if self._embeddings_raw is None:
            self._embeddings_raw = _get_judge_embeddings()
        return LangchainEmbeddingsWrapper(self._embeddings_raw)

    def evaluate(
        self,
        query: str,
        response: str,
        contexts: list[str],
        reference: Optional[str] = None,
    ) -> RAGASScores:
        """Evaluate a single RAG response.

        Args:
            query: User question.
            response: Generated answer.
            contexts: Retrieved context passages.
            reference: Optional ground truth answer (for context_recall).

        Returns:
            RAGASScores with evaluation results.
        """
        sample = build_sample(query, response, contexts, reference)
        dataset = EvaluationDataset(samples=[sample])

        try:
            results = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self._ensure_llm(),
                embeddings=self._ensure_embeddings(),
                show_progress=False,
            )

            df = results.to_pandas()
            row = df.iloc[0]

            scores = RAGASScores(
                faithfulness=float(row.get("faithfulness", 0.0)),
                answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                context_precision=float(
                    row.get("llm_context_precision_without_reference", 0.0)
                ),
                context_recall=float(
                    row.get("context_recall", 0.0) if reference else 0.0
                ),
            )

            logger.info(f"RAGAS evaluation complete: avg={scores.average:.3f}")
            return scores

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
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
        references: Optional[list[str]] = None,
    ) -> list[RAGASScores]:
        """Evaluate multiple RAG responses in a single batch.

        Args:
            queries: List of user questions.
            responses: List of generated answers.
            contexts_list: List of context lists.
            references: Optional list of ground truth answers.

        Returns:
            List of RAGASScores (one per query).
        """
        samples = []
        for i, (q, r, c) in enumerate(zip(queries, responses, contexts_list)):
            ref = references[i] if references else None
            samples.append(build_sample(q, r, c, ref))

        dataset = EvaluationDataset(samples=samples)

        try:
            results = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self._ensure_llm(),
                embeddings=self._ensure_embeddings(),
                show_progress=True,
            )

            df = results.to_pandas()
            scores_list = []
            for _, row in df.iterrows():
                scores_list.append(
                    RAGASScores(
                        faithfulness=float(row.get("faithfulness", 0.0)),
                        answer_relevancy=float(
                            row.get("answer_relevancy", 0.0)
                        ),
                        context_precision=float(
                            row.get(
                                "llm_context_precision_without_reference", 0.0
                            )
                        ),
                        context_recall=float(
                            row.get("context_recall", 0.0)
                            if references
                            else 0.0
                        ),
                    )
                )
            return scores_list

        except Exception as e:
            logger.error(f"Batch RAGAS evaluation failed: {e}")
            return [RAGASScores(0.0, 0.0, 0.0, 0.0) for _ in queries]

    @staticmethod
    def get_metrics_description() -> dict[str, str]:
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
    llm=None,
    embeddings=None,
) -> RAGASEvaluator:
    """Factory function to create a RAGAS evaluator.

    Args:
        llm: Optional LangChain LLM for judge duties.
        embeddings: Optional LangChain embeddings.

    Returns:
        Configured RAGASEvaluator.
    """
    return RAGASEvaluator(llm=llm, embeddings=embeddings)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("RAGAS Evaluator module (v0.4.x)")
    print("\nMetrics available:")

    evaluator = RAGASEvaluator()
    for metric, desc in evaluator.get_metrics_description().items():
        print(f"\n{metric}:")
        print(f"  {desc}")
