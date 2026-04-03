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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAG_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, build_system_prompt, load_patient_context

_SYSTEM_PROMPT = build_system_prompt(load_patient_context())

logger = logging.getLogger(__name__)


class QueryIntent:
    """Classification of user query intent (string constants)."""
    RETRIEVAL = "retrieval"        # Needs document lookup
    DIRECT = "direct"              # Can be answered directly
    CLARIFICATION = "clarification"  # Needs more info from user
    OUT_OF_SCOPE = "out_of_scope"  # Outside CKD domain


class RAGNodes:
    """
    Collection of node functions for the Agentic RAG workflow.

    Each node takes dict state and returns updated dict state.
    State is passed as dict (no GraphState dataclass wrapper).

    Example:
        >>> nodes = RAGNodes(pii_handler, retriever, llm)
        >>> state = {"original_query": "What is CKD stage 3?"}
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

    def pii_check(self, state: dict) -> dict:
        """
        Node: Check for and redact PII in the query.

        Args:
            state: Current graph state as dict

        Returns:
            Updated state dict with PII handled
        """
        logger.debug("Node: pii_check")

        try:
            result = self.pii_handler.anonymize(state["original_query"])

            updates = {
                "processing_steps": ["pii_check"],
                "anonymized_query": result.anonymized_text,
                "pii_detected": result.pii_found,
                "pii_map": result.placeholder_map,
            }

            if result.pii_found:
                logger.info(f"PII detected and redacted: {len(result.entities_detected)} entities")

            return updates

        except Exception as e:
            logger.error(f"PII check failed: {e}")
            # SAFETY: Never pass unredacted text through the pipeline.
            # Let RetryPolicy handle transient failures; fatal ones stop the graph.
            raise

    def analyze_query(self, state: dict) -> dict:
        """
        Node: Analyze query intent and extract keywords.

        Args:
            state: Current graph state as dict

        Returns:
            Updated state dict with query analysis
        """
        logger.debug("Node: analyze_query")

        query = state.get("anonymized_query", state.get("original_query", "")).lower()

        # Check for clarification requests
        if any(trigger in query for trigger in self.CLARIFICATION_TRIGGERS):
            return {
                "processing_steps": ["analyze_query"],
                "query_intent": QueryIntent.CLARIFICATION,
            }

        # Check for direct answer topics
        if any(topic in query for topic in self.DIRECT_ANSWER_TOPICS):
            return {
                "processing_steps": ["analyze_query"],
                "query_intent": QueryIntent.DIRECT,
            }

        # Check if query is CKD-related
        if not any(keyword in query for keyword in self.CKD_KEYWORDS):
            return {
                "processing_steps": ["analyze_query"],
                "query_intent": QueryIntent.OUT_OF_SCOPE,
            }

        # Default: needs retrieval
        keywords = [kw for kw in self.CKD_KEYWORDS if kw in query]
        return {
            "processing_steps": ["analyze_query"],
            "query_intent": QueryIntent.RETRIEVAL,
            "query_keywords": keywords,
        }

    def retrieve_documents(self, state: dict) -> dict:
        """
        Node: Retrieve relevant documents.

        Args:
            state: Current graph state as dict

        Returns:
            Updated state dict with retrieved documents
        """
        logger.debug("Node: retrieve_documents")

        try:
            # Metadata filtering for ckd_stage is no longer supported
            retriever = self.retriever

            docs = retriever.invoke(state["anonymized_query"])
            logger.info(f"Retrieved {len(docs)} documents")

            return {
                "processing_steps": ["retrieve_documents"],
                "retrieved_documents": docs,
                "context": self._format_context(docs),
            }

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "processing_steps": ["retrieve_documents"],
                "error": f"Retrieval error: {str(e)}",
            }

    def _format_context(self, docs: list) -> str:
        """Format retrieved documents into context string."""
        if not docs:
            return "No relevant documents found."

        parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            section = doc.metadata.get("section", "")
            header = f"[{i}] (Source: {source})"
            if section:
                header = f"[{i}] (Source: {source}, Section: {section})"
            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    def generate_response(self, state: dict) -> dict:
        """
        Node: Generate response using LLM.

        Args:
            state: Current graph state as dict

        Returns:
            Updated state dict with generated response
        """
        logger.debug("Node: generate_response")

        try:
            # Build prompt
            prompt = f"""{_SYSTEM_PROMPT}

{RAG_PROMPT_TEMPLATE.format(
    context=state.get("context", ""),
    question=state["anonymized_query"]
)}"""

            # Generate response
            raw_response = self.llm.generate(prompt)

            # Restore PII if needed
            if state.get("pii_detected") and state.get("pii_map"):
                final_response = self.pii_handler.restore_in_response(
                    raw_response,
                    state["pii_map"],
                )
            else:
                final_response = raw_response

            logger.info("Response generated successfully")

            return {
                "processing_steps": ["generate_response"],
                "raw_response": raw_response,
                "final_response": final_response,
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "processing_steps": ["generate_response"],
                "error": f"Generation error: {str(e)}",
                "final_response": "I apologize, but I encountered an error generating a response. Please try again.",
            }

    def generate_direct_response(self, state: dict) -> dict:
        """
        Node: Generate a direct response without retrieval.

        For simple definitional questions.

        Args:
            state: Current graph state as dict

        Returns:
            Updated state dict with direct response
        """
        logger.debug("Node: generate_direct_response")

        try:
            prompt = f"""{_SYSTEM_PROMPT}

Answer the following question about chronic kidney disease directly and concisely.

Question: {state["anonymized_query"]}

Answer:"""

            raw_response = self.llm.generate(prompt)

            return {
                "processing_steps": ["generate_direct_response"],
                "raw_response": raw_response,
                "final_response": raw_response,
            }

        except Exception as e:
            logger.error(f"Direct generation failed: {e}")
            return {
                "processing_steps": ["generate_direct_response"],
                "error": f"Generation error: {str(e)}",
            }

    def generate_clarification(self, state: dict) -> dict:
        """
        Node: Generate a clarification request.

        Args:
            state: Current graph state as dict

        Returns:
            Updated state dict with clarification response
        """
        logger.debug("Node: generate_clarification")

        return {
            "processing_steps": ["generate_clarification"],
            "final_response": (
                "I'd be happy to help clarify. Could you please provide more details "
                "about what specific aspect of CKD management you'd like to understand better? "
                "For example:\n"
                "- Dietary restrictions for a specific CKD stage\n"
                "- Medication considerations\n"
                "- Lifestyle recommendations\n"
                "- Understanding test results (eGFR, creatinine)"
            ),
        }

    def generate_out_of_scope(self, state: dict) -> dict:
        """
        Node: Handle out-of-scope queries.

        Args:
            state: Current graph state as dict

        Returns:
            Updated state dict with out-of-scope response
        """
        logger.debug("Node: generate_out_of_scope")

        return {
            "processing_steps": ["generate_out_of_scope"],
            "final_response": (
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
            ),
        }

    def evaluate_response(self, state: dict) -> dict:
        """
        Node: Evaluate the response quality.

        Args:
            state: Current graph state as dict

        Returns:
            Updated state dict with evaluation scores
        """
        logger.debug("Node: evaluate_response")

        if self.evaluator is None:
            logger.debug("No evaluator configured, skipping evaluation")
            return {"processing_steps": ["evaluate_response"]}

        try:
            scores = self.evaluator.evaluate(
                query=state["original_query"],
                response=state["final_response"],
                contexts=[doc.page_content for doc in state.get("retrieved_documents", [])],
            )
            logger.info(f"Evaluation scores: {scores}")

            return {
                "processing_steps": ["evaluate_response"],
                "evaluation_scores": scores,
            }

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Don't set error - evaluation failure shouldn't block response
            return {"processing_steps": ["evaluate_response"]}


def get_route(state: dict) -> str:
    """
    Router function for the LangGraph workflow.

    Determines which path to take based on query intent.

    Args:
        state: Current graph state as dict

    Returns:
        Route name (matches graph edge names)
    """
    intent = state.get("query_intent", QueryIntent.RETRIEVAL)

    if intent == QueryIntent.RETRIEVAL:
        return "retrieval"
    elif intent == QueryIntent.DIRECT:
        return "direct"
    elif intent == QueryIntent.CLARIFICATION:
        return "clarification"
    else:
        return "out_of_scope"
