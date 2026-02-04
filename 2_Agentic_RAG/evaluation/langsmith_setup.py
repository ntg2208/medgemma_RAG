"""
LangSmith integration for the CKD RAG System.

Provides tracing, logging, and debugging capabilities
for the RAG pipeline.
"""

import logging
import os
from typing import Optional, Any
from functools import wraps
from contextlib import contextmanager

from langsmith import Client, traceable
from langsmith.run_trees import RunTree

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_TRACING,
)

logger = logging.getLogger(__name__)


def setup_langsmith(
    api_key: Optional[str] = None,
    project_name: Optional[str] = None,
    tracing_enabled: Optional[bool] = None,
) -> bool:
    """
    Configure LangSmith tracing.

    Sets environment variables needed for LangSmith integration
    with LangChain and LangGraph.

    Args:
        api_key: LangSmith API key (uses config default if None)
        project_name: Project name for traces (uses config default if None)
        tracing_enabled: Enable/disable tracing (uses config default if None)

    Returns:
        True if LangSmith is configured and enabled
    """
    api_key = api_key or LANGSMITH_API_KEY
    project_name = project_name or LANGSMITH_PROJECT
    tracing_enabled = tracing_enabled if tracing_enabled is not None else LANGSMITH_TRACING

    if not tracing_enabled:
        logger.info("LangSmith tracing is disabled")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    if not api_key:
        logger.warning("LangSmith API key not provided. Tracing disabled.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    # Set environment variables for LangChain integration
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    logger.info(f"LangSmith tracing enabled for project: {project_name}")
    return True


class LangSmithTracer:
    """
    LangSmith tracer for custom tracing and evaluation.

    Provides decorators and context managers for tracing
    specific operations in the RAG pipeline.

    Example:
        >>> tracer = LangSmithTracer()
        >>> @tracer.trace("retrieval")
        ... def retrieve_documents(query):
        ...     return docs
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: Optional[str] = None,
    ):
        """
        Initialize the tracer.

        Args:
            api_key: LangSmith API key
            project_name: Project name for organization
        """
        self.api_key = api_key or LANGSMITH_API_KEY
        self.project_name = project_name or LANGSMITH_PROJECT
        self.enabled = bool(self.api_key)

        if self.enabled:
            self.client = Client(api_key=self.api_key)
            logger.info("LangSmithTracer initialized")
        else:
            self.client = None
            logger.info("LangSmithTracer disabled (no API key)")

    def trace(
        self,
        name: str,
        run_type: str = "chain",
        metadata: Optional[dict] = None,
    ):
        """
        Decorator to trace a function.

        Args:
            name: Name for the trace
            run_type: Type of run (chain, llm, retriever, tool)
            metadata: Additional metadata to log

        Returns:
            Decorated function
        """
        def decorator(func):
            if not self.enabled:
                return func

            @wraps(func)
            @traceable(name=name, run_type=run_type, metadata=metadata or {})
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @contextmanager
    def trace_context(
        self,
        name: str,
        run_type: str = "chain",
        inputs: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Context manager for tracing a block of code.

        Args:
            name: Name for the trace
            run_type: Type of run
            inputs: Input data to log
            metadata: Additional metadata

        Yields:
            RunTree object for adding outputs
        """
        if not self.enabled:
            yield None
            return

        run_tree = RunTree(
            name=name,
            run_type=run_type,
            inputs=inputs or {},
            extra={"metadata": metadata or {}},
            project_name=self.project_name,
        )

        try:
            run_tree.post()
            yield run_tree
            run_tree.end()
            run_tree.patch()
        except Exception as e:
            run_tree.end(error=str(e))
            run_tree.patch()
            raise

    def log_feedback(
        self,
        run_id: str,
        key: str,
        score: float,
        comment: Optional[str] = None,
    ):
        """
        Log feedback for a run.

        Args:
            run_id: ID of the run to provide feedback for
            key: Feedback key (e.g., "correctness", "helpfulness")
            score: Score value (typically 0-1)
            comment: Optional comment
        """
        if not self.enabled:
            return

        try:
            self.client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment,
            )
            logger.debug(f"Logged feedback for run {run_id}: {key}={score}")
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")

    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Create a dataset for evaluation.

        Args:
            name: Dataset name
            description: Dataset description

        Returns:
            Dataset object or None if disabled
        """
        if not self.enabled:
            return None

        try:
            dataset = self.client.create_dataset(
                dataset_name=name,
                description=description or f"Evaluation dataset for {self.project_name}",
            )
            logger.info(f"Created dataset: {name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return None

    def add_examples(
        self,
        dataset_name: str,
        examples: list[dict],
    ):
        """
        Add examples to a dataset.

        Args:
            dataset_name: Name of the dataset
            examples: List of example dicts with 'inputs' and 'outputs'
        """
        if not self.enabled:
            return

        try:
            for example in examples:
                self.client.create_example(
                    inputs=example.get("inputs", {}),
                    outputs=example.get("outputs"),
                    dataset_name=dataset_name,
                )
            logger.info(f"Added {len(examples)} examples to {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to add examples: {e}")

    def get_runs(
        self,
        limit: int = 100,
        filter_dict: Optional[dict] = None,
    ) -> list:
        """
        Get recent runs from the project.

        Args:
            limit: Maximum number of runs to return
            filter_dict: Optional filters

        Returns:
            List of run objects
        """
        if not self.enabled:
            return []

        try:
            runs = list(self.client.list_runs(
                project_name=self.project_name,
                limit=limit,
                filter=filter_dict,
            ))
            return runs
        except Exception as e:
            logger.error(f"Failed to get runs: {e}")
            return []


# Convenience decorators
def trace_rag_query(func):
    """Decorator to trace RAG query processing."""
    @wraps(func)
    @traceable(name="rag_query", run_type="chain")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def trace_retrieval(func):
    """Decorator to trace document retrieval."""
    @wraps(func)
    @traceable(name="retrieval", run_type="retriever")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def trace_generation(func):
    """Decorator to trace LLM generation."""
    @wraps(func)
    @traceable(name="generation", run_type="llm")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def create_tracer(
    api_key: Optional[str] = None,
    project_name: Optional[str] = None,
) -> LangSmithTracer:
    """
    Factory function to create a LangSmith tracer.

    Args:
        api_key: LangSmith API key
        project_name: Project name

    Returns:
        Configured LangSmithTracer
    """
    return LangSmithTracer(api_key=api_key, project_name=project_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("LangSmith Setup Module")
    print("-" * 40)

    # Check configuration
    configured = setup_langsmith()
    print(f"LangSmith enabled: {configured}")

    if configured:
        tracer = LangSmithTracer()
        print(f"Project: {tracer.project_name}")
    else:
        print("Set LANGSMITH_API_KEY in .env to enable tracing")
