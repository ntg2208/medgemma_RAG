"""
LangGraph workflow for the Agentic RAG system.

Defines the stateful workflow that orchestrates:
- PII detection
- Query analysis
- Conditional routing
- Document retrieval
- Response generation
- Evaluation
"""

import importlib
import importlib.util
import logging
from typing import Any, Optional, TypedDict, Annotated, Literal
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from nodes module - use importlib for numeric module names
_nodes_spec = importlib.import_module("2_Agentic_RAG.nodes")  # type: ignore[call-overload]
RAGNodes = _nodes_spec.RAGNodes
QueryIntent = _nodes_spec.QueryIntent
get_route = _nodes_spec.get_route

logger = logging.getLogger(__name__)


class AgenticGraphState(TypedDict, total=False):
    """
    TypedDict version of GraphState for LangGraph.

    LangGraph requires TypedDict for state schema.
    """
    # Input
    original_query: str
    ckd_stage: int | None

    # PII handling
    anonymized_query: str
    pii_detected: bool
    pii_map: dict[str, str]

    # Query analysis
    query_intent: str  # Stored as string for serialization
    query_keywords: list[str]

    # Retrieval
    retrieved_documents: list[Any]
    context: str

    # Generation
    raw_response: str
    final_response: str

    # Evaluation
    evaluation_scores: dict[str, float]

    # Metadata
    error: str | None
    processing_steps: Annotated[list[str], add]


class AgenticRAGGraph:
    """
    LangGraph-based Agentic RAG workflow.

    Implements a stateful, conditional workflow:

    ```
    START → PII Check → Query Analysis → [Router]
                                            ├─ Retrieval → Generate → Evaluate → END
                                            ├─ Direct → Generate Direct → END
                                            ├─ Clarification → Generate Clarification → END
                                            └─ Out of Scope → Generate OOS → END
    ```

    Example:
        >>> graph = AgenticRAGGraph(pii_handler, retriever, llm)
        >>> result = graph.invoke("What are the dietary restrictions for CKD stage 3?")
        >>> print(result["final_response"])
    """

    def __init__(
        self,
        pii_handler: Any,
        retriever: Any,
        llm: Any,
        evaluator: Optional[Any] = None,
        enable_memory: bool = False,
    ):
        """
        Initialize the Agentic RAG graph.

        Args:
            pii_handler: PIIHandler instance
            retriever: Document retriever
            llm: Language model for generation
            evaluator: Optional RAGAS evaluator
            enable_memory: Enable conversation memory
        """
        self.nodes = RAGNodes(
            pii_handler=pii_handler,
            retriever=retriever,
            llm=llm,
            evaluator=evaluator,
        )

        self.enable_memory = enable_memory
        self.memory = MemorySaver() if enable_memory else None

        # Build the graph
        self.graph = self._build_graph()
        self.compiled_graph = self._compile_graph()

        logger.info("AgenticRAGGraph initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Create graph with state schema
        graph = StateGraph(AgenticGraphState)

        # Add nodes (state is passed as dict directly)

        # API nodes - add RetryPolicy for transient error handling
        api_retry = RetryPolicy(max_attempts=3, initial_interval=2.0, max_interval=60.0)
        graph.add_node("pii_check", self.nodes.pii_check, retry_policy=api_retry)
        graph.add_node("analyze_query", self.nodes.analyze_query)
        graph.add_node("retrieve_documents", self.nodes.retrieve_documents, retry_policy=api_retry)

        # LLM generation nodes - add RetryPolicy for transient error handling
        llm_retry = RetryPolicy(max_attempts=3, initial_interval=1.0, max_interval=30.0)
        graph.add_node("generate_response", self.nodes.generate_response, retry_policy=llm_retry)
        graph.add_node("generate_direct", self.nodes.generate_direct_response, retry_policy=llm_retry)

        graph.add_node("generate_clarification", self.nodes.generate_clarification)
        graph.add_node("generate_out_of_scope", self.nodes.generate_out_of_scope)

        # Evaluator node - minimal retry (best-effort evaluation)
        eval_retry = RetryPolicy(max_attempts=2, initial_interval=1.0)
        graph.add_node("evaluate", self.nodes.evaluate_response, retry_policy=eval_retry)

        # Set entry point
        graph.set_entry_point("pii_check")

        # Add edges
        graph.add_edge("pii_check", "analyze_query")

        # Conditional routing based on query intent
        def route_query(state: dict) -> Literal["retrieval", "direct", "clarification", "out_of_scope"]:
            """Route to appropriate node based on query intent."""
            intent = state.get("query_intent", QueryIntent.RETRIEVAL)
            if intent == QueryIntent.RETRIEVAL:
                return "retrieval"
            elif intent == QueryIntent.DIRECT:
                return "direct"
            elif intent == QueryIntent.CLARIFICATION:
                return "clarification"
            else:
                return "out_of_scope"

        graph.add_conditional_edges(
            "analyze_query",
            route_query,
            {
                "retrieval": "retrieve_documents",
                "direct": "generate_direct",
                "clarification": "generate_clarification",
                "out_of_scope": "generate_out_of_scope",
            }
        )

        # Retrieval path
        graph.add_edge("retrieve_documents", "generate_response")
        graph.add_edge("generate_response", "evaluate")
        graph.add_edge("evaluate", END)

        # Direct paths to END
        graph.add_edge("generate_direct", END)
        graph.add_edge("generate_clarification", END)
        graph.add_edge("generate_out_of_scope", END)

        return graph

    def _compile_graph(self):
        """Compile the graph for execution."""
        if self.memory:
            return self.graph.compile(checkpointer=self.memory)
        return self.graph.compile()

    def invoke(
        self,
        query: str,
        ckd_stage: Optional[int] = None,
        thread_id: Optional[str] = None,
    ) -> dict:
        """
        Process a query through the RAG workflow.

        Args:
            query: User question
            ckd_stage: Optional CKD stage for filtering
            thread_id: Optional thread ID for memory

        Returns:
            Final state dict with response and metadata
        """
        initial_state = {
            "original_query": query,
            "ckd_stage": ckd_stage,
            "processing_steps": [],
        }

        config = {}
        if thread_id and self.memory:
            config["configurable"] = {"thread_id": thread_id}

        result = self.compiled_graph.invoke(initial_state, config)

        logger.info(f"Workflow completed. Steps: {result.get('processing_steps', [])}")
        return result

    def stream(
        self,
        query: str,
        ckd_stage: Optional[int] = None,
    ):
        """
        Stream the workflow execution.

        Yields state updates as each node completes.

        Args:
            query: User question
            ckd_stage: Optional CKD stage

        Yields:
            State updates from each node
        """
        initial_state = {
            "original_query": query,
            "ckd_stage": ckd_stage,
            "processing_steps": [],
        }

        for state_update in self.compiled_graph.stream(initial_state):
            yield state_update

    def get_graph_diagram(self) -> str:
        """
        Get a Mermaid diagram of the graph.

        Returns:
            Mermaid diagram string
        """
        try:
            return self.compiled_graph.get_graph().draw_mermaid()
        except Exception:
            # Fallback ASCII representation
            return """
            START
              │
              ▼
           PII Check
              │
              ▼
         Query Analysis
              │
         ┌────┼────┬────────────┐
         │    │    │            │
         ▼    ▼    ▼            ▼
      Retrieve Direct Clarify  OOS
         │    │    │            │
         ▼    │    │            │
      Generate │    │            │
         │    │    │            │
         ▼    │    │            │
      Evaluate│    │            │
         │    │    │            │
         └────┴────┴────────────┘
              │
              ▼
             END
            """


def create_agentic_rag(
    pii_handler: Any,
    retriever: Any,
    llm: Any,
    evaluator: Optional[Any] = None,
) -> AgenticRAGGraph:
    """
    Factory function to create an Agentic RAG graph.

    Args:
        pii_handler: PIIHandler instance
        retriever: Document retriever
        llm: Language model
        evaluator: Optional evaluator

    Returns:
        Configured AgenticRAGGraph
    """
    return AgenticRAGGraph(
        pii_handler=pii_handler,
        retriever=retriever,
        llm=llm,
        evaluator=evaluator,
    )


if __name__ == "__main__":
    # Example usage (requires actual components)
    logging.basicConfig(level=logging.INFO)

    print("AgenticRAGGraph module loaded.")
    print("To use, initialize with pii_handler, retriever, and llm components.")
