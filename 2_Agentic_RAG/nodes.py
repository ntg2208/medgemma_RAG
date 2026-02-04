"""
Node functions for the Agentic RAG LangGraph workflow.

Each node represents a step in the RAG pipeline:
- PII detection and redaction
- Query analysis and classification
- Document retrieval
- Response generation
- Evaluation
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAG_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Classification of user query intent."""
    RETRIEVAL = "retrieval"        # Needs document lookup
    DIRECT = "direct"              # Can be answered directly
    CLARIFICATION = "clarification"  # Needs more info from user
    OUT_OF_SCOPE = "out_of_scope"  # Outside CKD domain


@dataclass
class GraphState:
    """
    State object passed through the LangGraph workflow.

    Accumulates information as it passes through nodes.
    """
    # Input
    original_query: str
    ckd_stage: Optional[int] = None

    # PII handling
    anonymized_query: str = ""
    pii_detected: bool = False
    pii_map: dict[str, str] = field(default_factory=dict)

    # Query analysis
    query_intent: QueryIntent = QueryIntent.RETRIEVAL
    query_keywords: list[str] = field(default_factory=list)

    # Retrieval
    retrieved_documents: list[Any] = field(default_factory=list)
    context: str = ""

    # Generation
    raw_response: str = ""
    final_response: str = ""

    # Evaluation
    evaluation_scores: dict[str, float] = field(default_factory=dict)

    # Metadata
    error: Optional[str] = None
    processing_steps: list[str] = field(default_factory=list)


class RAGNodes:
    """
    Collection of node functions for the Agentic RAG workflow.

    Each method is a node that takes GraphState and returns updated state.

    Example:
        >>> nodes = RAGNodes(pii_handler, retriever, llm)
        >>> state = GraphState(original_query="What is CKD stage 3?")
        >>> state = nodes.pii_check(state)
        >>> state = nodes.analyze_query(state)
    """

    # Keywords indicating different query intents
    CLARIFICATION_TRIGGERS = [
        "what do you mean",
        "can you explain",
        "i don't understand",
        "more details",
    ]

    DIRECT_ANSWER_TOPICS = [
        "what is ckd",
        "what does ckd stand for",
        "what is egfr",
        "what is chronic kidney disease",
    ]

    CKD_KEYWORDS = [
        "kidney", "renal", "ckd", "egfr", "dialysis", "nephro",
        "potassium", "phosphorus", "creatinine", "proteinuria",
        "stage 1", "stage 2", "stage 3", "stage 4", "stage 5",
    ]

    def __init__(
        self,
        pii_handler: Any,
        retriever: Any,
        llm: Any,
        evaluator: Optional[Any] = None,
    ):
        """
        Initialize RAG nodes.

        Args:
            pii_handler: PIIHandler instance
            retriever: Document retriever
            llm: Language model for generation
            evaluator: Optional evaluator for scoring
        """
        self.pii_handler = pii_handler
        self.retriever = retriever
        self.llm = llm
        self.evaluator = evaluator

    def pii_check(self, state: GraphState) -> GraphState:
        """
        Node: Check for and redact PII in the query.

        Args:
            state: Current graph state

        Returns:
            Updated state with PII handled
        """
        logger.debug("Node: pii_check")
        state.processing_steps.append("pii_check")

        try:
            result = self.pii_handler.anonymize(state.original_query)

            state.anonymized_query = result.anonymized_text
            state.pii_detected = result.pii_found
            state.pii_map = result.placeholder_map

            if result.pii_found:
                logger.info(f"PII detected and redacted: {len(result.entities_detected)} entities")

        except Exception as e:
            logger.error(f"PII check failed: {e}")
            state.anonymized_query = state.original_query
            state.error = f"PII check error: {str(e)}"

        return state

    def analyze_query(self, state: GraphState) -> GraphState:
        """
        Node: Analyze query intent and extract keywords.

        Args:
            state: Current graph state

        Returns:
            Updated state with query analysis
        """
        logger.debug("Node: analyze_query")
        state.processing_steps.append("analyze_query")

        query = state.anonymized_query.lower()

        # Check for clarification requests
        if any(trigger in query for trigger in self.CLARIFICATION_TRIGGERS):
            state.query_intent = QueryIntent.CLARIFICATION
            return state

        # Check for direct answer topics
        if any(topic in query for topic in self.DIRECT_ANSWER_TOPICS):
            state.query_intent = QueryIntent.DIRECT
            return state

        # Check if query is CKD-related
        if not any(keyword in query for keyword in self.CKD_KEYWORDS):
            state.query_intent = QueryIntent.OUT_OF_SCOPE
            return state

        # Default: needs retrieval
        state.query_intent = QueryIntent.RETRIEVAL

        # Extract keywords for retrieval
        keywords = [kw for kw in self.CKD_KEYWORDS if kw in query]
        state.query_keywords = keywords

        return state

    def retrieve_documents(self, state: GraphState) -> GraphState:
        """
        Node: Retrieve relevant documents.

        Args:
            state: Current graph state

        Returns:
            Updated state with retrieved documents
        """
        logger.debug("Node: retrieve_documents")
        state.processing_steps.append("retrieve_documents")

        try:
            # Use CKD stage filter if available
            if state.ckd_stage and hasattr(self.retriever, 'with_config'):
                retriever = self.retriever.with_config(ckd_stage=state.ckd_stage)
            else:
                retriever = self.retriever

            docs = retriever.invoke(state.anonymized_query)
            state.retrieved_documents = docs

            # Format context
            state.context = self._format_context(docs)

            logger.info(f"Retrieved {len(docs)} documents")

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state.error = f"Retrieval error: {str(e)}"

        return state

    def _format_context(self, docs: list) -> str:
        """Format retrieved documents into context string."""
        if not docs:
            return "No relevant documents found."

        parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page_number", "?")
            parts.append(f"[{i}] (Source: {source}, Page {page})\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    def generate_response(self, state: GraphState) -> GraphState:
        """
        Node: Generate response using LLM.

        Args:
            state: Current graph state

        Returns:
            Updated state with generated response
        """
        logger.debug("Node: generate_response")
        state.processing_steps.append("generate_response")

        try:
            # Build prompt
            prompt = f"""{RAG_SYSTEM_PROMPT}

{RAG_PROMPT_TEMPLATE.format(
    context=state.context,
    question=state.anonymized_query
)}"""

            # Generate response
            state.raw_response = self.llm.generate(prompt)

            # Restore PII if needed
            if state.pii_detected and state.pii_map:
                state.final_response = self.pii_handler.restore_in_response(
                    state.raw_response,
                    state.pii_map,
                )
            else:
                state.final_response = state.raw_response

            logger.info("Response generated successfully")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            state.error = f"Generation error: {str(e)}"
            state.final_response = "I apologize, but I encountered an error generating a response. Please try again."

        return state

    def generate_direct_response(self, state: GraphState) -> GraphState:
        """
        Node: Generate a direct response without retrieval.

        For simple definitional questions.

        Args:
            state: Current graph state

        Returns:
            Updated state with direct response
        """
        logger.debug("Node: generate_direct_response")
        state.processing_steps.append("generate_direct_response")

        try:
            prompt = f"""{RAG_SYSTEM_PROMPT}

Answer the following question about chronic kidney disease directly and concisely.

Question: {state.anonymized_query}

Answer:"""

            state.raw_response = self.llm.generate(prompt)
            state.final_response = state.raw_response

        except Exception as e:
            logger.error(f"Direct generation failed: {e}")
            state.error = f"Generation error: {str(e)}"

        return state

    def generate_clarification(self, state: GraphState) -> GraphState:
        """
        Node: Generate a clarification request.

        Args:
            state: Current graph state

        Returns:
            Updated state with clarification response
        """
        logger.debug("Node: generate_clarification")
        state.processing_steps.append("generate_clarification")

        state.final_response = (
            "I'd be happy to help clarify. Could you please provide more details "
            "about what specific aspect of CKD management you'd like to understand better? "
            "For example:\n"
            "- Dietary restrictions for a specific CKD stage\n"
            "- Medication considerations\n"
            "- Lifestyle recommendations\n"
            "- Understanding test results (eGFR, creatinine)"
        )

        return state

    def generate_out_of_scope(self, state: GraphState) -> GraphState:
        """
        Node: Handle out-of-scope queries.

        Args:
            state: Current graph state

        Returns:
            Updated state with out-of-scope response
        """
        logger.debug("Node: generate_out_of_scope")
        state.processing_steps.append("generate_out_of_scope")

        state.final_response = (
            "I'm specialized in Chronic Kidney Disease (CKD) management based on "
            "NICE guidelines and KidneyCareUK resources. Your question appears to be "
            "outside my area of expertise.\n\n"
            "I can help with questions about:\n"
            "- CKD stages and progression\n"
            "- Dietary recommendations (potassium, phosphorus, sodium, protein)\n"
            "- Medication considerations for kidney patients\n"
            "- Lifestyle and exercise guidance\n"
            "- Understanding kidney function tests\n\n"
            "Please rephrase your question if it relates to CKD management."
        )

        return state

    def evaluate_response(self, state: GraphState) -> GraphState:
        """
        Node: Evaluate the response quality.

        Args:
            state: Current graph state

        Returns:
            Updated state with evaluation scores
        """
        logger.debug("Node: evaluate_response")
        state.processing_steps.append("evaluate_response")

        if self.evaluator is None:
            logger.debug("No evaluator configured, skipping evaluation")
            return state

        try:
            scores = self.evaluator.evaluate(
                query=state.original_query,
                response=state.final_response,
                contexts=[doc.page_content for doc in state.retrieved_documents],
            )
            state.evaluation_scores = scores
            logger.info(f"Evaluation scores: {scores}")

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Don't set error - evaluation failure shouldn't block response

        return state


def get_route(state: GraphState) -> str:
    """
    Router function for the LangGraph workflow.

    Determines which path to take based on query intent.

    Args:
        state: Current graph state

    Returns:
        Route name (matches graph edge names)
    """
    intent = state.query_intent

    if intent == QueryIntent.RETRIEVAL:
        return "retrieval"
    elif intent == QueryIntent.DIRECT:
        return "direct"
    elif intent == QueryIntent.CLARIFICATION:
        return "clarification"
    else:
        return "out_of_scope"
