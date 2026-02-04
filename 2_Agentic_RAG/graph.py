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

import logging
from typing import Any, Optional, TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .nodes import RAGNodes, GraphState, QueryIntent, get_route

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

    def _state_to_graph_state(self, state: dict) -> GraphState:
        """Convert TypedDict state to GraphState dataclass."""
        gs = GraphState(original_query=state.get("original_query", ""))

        # Copy all fields
        for key in ["ckd_stage", "anonymized_query", "pii_detected",
                    "context", "raw_response", "final_response", "error"]:
            if key in state and state[key] is not None:
                setattr(gs, key, state[key])

        # Handle dict/list fields
        if "pii_map" in state:
            gs.pii_map = state["pii_map"] or {}
        if "query_keywords" in state:
            gs.query_keywords = state["query_keywords"] or []
        if "retrieved_documents" in state:
            gs.retrieved_documents = state["retrieved_documents"] or []
        if "evaluation_scores" in state:
            gs.evaluation_scores = state["evaluation_scores"] or {}
        if "processing_steps" in state:
            gs.processing_steps = state["processing_steps"] or []

        # Handle enum
        if "query_intent" in state and state["query_intent"]:
            try:
                gs.query_intent = QueryIntent(state["query_intent"])
            except ValueError:
                gs.query_intent = QueryIntent.RETRIEVAL

        return gs

    def _graph_state_to_dict(self, gs: GraphState) -> dict:
        """Convert GraphState dataclass to dict for LangGraph."""
        return {
            "original_query": gs.original_query,
            "ckd_stage": gs.ckd_stage,
            "anonymized_query": gs.anonymized_query,
            "pii_detected": gs.pii_detected,
            "pii_map": gs.pii_map,
            "query_intent": gs.query_intent.value if isinstance(gs.query_intent, QueryIntent) else gs.query_intent,
            "query_keywords": gs.query_keywords,
            "retrieved_documents": gs.retrieved_documents,
            "context": gs.context,
            "raw_response": gs.raw_response,
            "final_response": gs.final_response,
            "evaluation_scores": gs.evaluation_scores,
            "error": gs.error,
            "processing_steps": gs.processing_steps,
        }

    def _wrap_node(self, node_func):
        """Wrap a node function to handle state conversion."""
        def wrapped(state: dict) -> dict:
            gs = self._state_to_graph_state(state)
            result_gs = node_func(gs)
            return self._graph_state_to_dict(result_gs)
        return wrapped

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        # Create graph with state schema
        graph = StateGraph(AgenticGraphState)

        # Add nodes
        graph.add_node("pii_check", self._wrap_node(self.nodes.pii_check))
        graph.add_node("analyze_query", self._wrap_node(self.nodes.analyze_query))
        graph.add_node("retrieve_documents", self._wrap_node(self.nodes.retrieve_documents))
        graph.add_node("generate_response", self._wrap_node(self.nodes.generate_response))
        graph.add_node("generate_direct", self._wrap_node(self.nodes.generate_direct_response))
        graph.add_node("generate_clarification", self._wrap_node(self.nodes.generate_clarification))
        graph.add_node("generate_out_of_scope", self._wrap_node(self.nodes.generate_out_of_scope))
        graph.add_node("evaluate", self._wrap_node(self.nodes.evaluate_response))

        # Set entry point
        graph.set_entry_point("pii_check")

        # Add edges
        graph.add_edge("pii_check", "analyze_query")

        # Conditional routing based on query intent
        def route_query(state: dict) -> str:
            intent = state.get("query_intent", "retrieval")
            if intent == "retrieval":
                return "retrieval"
            elif intent == "direct":
                return "direct"
            elif intent == "clarification":
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
