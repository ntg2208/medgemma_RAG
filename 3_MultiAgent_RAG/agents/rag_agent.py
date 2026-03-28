"""
RAG Agent for the Multi-Agent CKD System.

Handles general knowledge retrieval from NICE guidelines
and KidneyCareUK documents.
"""

import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAG_SYSTEM_PROMPT

# Import BaseAgent and AgentResponse using importlib (needed for numeric module names)
# Check sys.modules first to reuse existing base module
agents_path = Path(__file__).parent
base_module = sys.modules.get("3_MultiAgent_RAG.agents.base")

if base_module is None:
    # First load - create and store in sys.modules
    base_spec = importlib.util.spec_from_file_location(
        "3_MultiAgent_RAG.agents.base", agents_path / "base.py"
    )
    base_module = importlib.util.module_from_spec(base_spec)  # type: ignore[assignment]
    sys.modules["3_MultiAgent_RAG.agents.base"] = base_module
    base_spec.loader.exec_module(base_module)  # type: ignore[attr-defined]

BaseAgent = base_module.BaseAgent
AgentResponse = base_module.AgentResponse

logger = logging.getLogger(__name__)


@dataclass
class RAGAgentResponse:
    """Response from the RAG Agent."""
    answer: str
    sources: list[dict]
    confidence: float
    agent_name: str = "RAG Agent"
    disclaimer: str = field(default_factory=base_module._get_default_disclaimer)


class RAGAgent(BaseAgent):
    """
    Knowledge retrieval agent for CKD information.

    Handles questions that require looking up information from
    the NICE guidelines and KidneyCareUK documents.

    Capabilities:
    - General CKD information retrieval
    - Guideline-based recommendations
    - Source citation

    Example:
        >>> agent = RAGAgent(retriever, llm)
        >>> response = agent.answer("What is the target blood pressure for CKD?")
        >>> print(response.answer)
    """

    AGENT_PROMPT = """You are a medical knowledge assistant specializing in Chronic Kidney Disease.
Your role is to provide accurate information from NICE guidelines and KidneyCareUK resources.

GUIDELINES:
1. Base your answers strictly on the provided context
2. Always cite your sources with [Source: document, section/page]
3. If the context doesn't contain the answer, say so clearly
4. Use clear, patient-friendly language

CONTEXT:
{context}

QUESTION: {question}

Provide a comprehensive answer with citations:"""

    def __init__(
        self,
        retriever: Any,
        llm: Any,
    ):
        """
        Initialize the RAG Agent.

        Args:
            retriever: Document retriever
            llm: Language model for generation
        """
        self.retriever = retriever
        self.llm = llm

        logger.info("RAGAgent initialized")

    def _format_context(self, documents: list) -> str:
        """Format retrieved documents into context."""
        if not documents:
            return "No relevant documents found."

        parts = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown")
            section = doc.metadata.get("section", "")

            header = f"[{i}] Source: {source}"
            if section:
                header += f", Section: {section}"

            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, documents: list) -> list[dict]:
        """Extract source information from documents."""
        sources = []
        for doc in documents:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "section": doc.metadata.get("section"),
            })
        return sources

    def can_handle(self, query: str) -> bool:
        """
        Check if this agent can handle the query.

        The RAG agent is the general fallback and can handle
        most CKD-related queries.

        Args:
            query: User question

        Returns:
            True if agent can handle query
        """
        # RAG agent handles general knowledge queries
        ckd_keywords = [
            "ckd", "kidney", "renal", "egfr", "creatinine",
            "guideline", "nice", "recommendation", "stage",
            "what is", "what are", "how", "why", "when",
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in ckd_keywords)

    def answer(
        self,
        query: str,
        **kwargs,
    ) -> AgentResponse:
        """
        Answer a query using RAG.

        Args:
            query: User question
            **kwargs: Additional parameters (e.g., weight_kg)

        Returns:
            AgentResponse with answer
        """
        # Metadata filtering for ckd_stage is no longer supported
        retriever = self.retriever

        # Retrieve documents
        try:
            documents = retriever.invoke(query)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return AgentResponse(
                answer="I encountered an error retrieving information. Please try again.",
                confidence=0.0,
            )

        if not documents:
            return AgentResponse(
                answer="I couldn't find relevant information in the guidelines for your question. "
                       "Please try rephrasing or ask about a different aspect of CKD management.",
                confidence=0.2,
            )

        # Format context and generate response
        context = self._format_context(documents)
        prompt = self.AGENT_PROMPT.format(context=context, question=query)

        try:
            answer = self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return AgentResponse(
                answer="I encountered an error generating a response. Please try again.",
                confidence=0.1,
            )

        # Calculate confidence based on retrieval quality
        confidence = min(0.9, 0.5 + (len(documents) * 0.1))

        return AgentResponse(
            answer=answer,
            confidence=confidence,
        )


def create_rag_agent(
    retriever: Any,
    llm: Any,
) -> RAGAgent:
    """Factory function to create a RAG agent."""
    return RAGAgent(retriever=retriever, llm=llm)
