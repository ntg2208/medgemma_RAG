"""
RAG Agent for the Multi-Agent CKD System.

Handles general knowledge retrieval from NICE guidelines
and KidneyCareUK documents.
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class RAGAgentResponse:
    """Response from the RAG Agent."""
    answer: str
    sources: list[dict]
    confidence: float
    agent_name: str = "RAG Agent"


class RAGAgent:
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
5. Include relevant CKD stage information when applicable

CONTEXT:
{context}

QUESTION: {question}

Provide a comprehensive answer with citations:"""

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        ckd_stage: Optional[int] = None,
    ):
        """
        Initialize the RAG Agent.

        Args:
            retriever: Document retriever
            llm: Language model for generation
            ckd_stage: Optional default CKD stage for filtering
        """
        self.retriever = retriever
        self.llm = llm
        self.ckd_stage = ckd_stage

        logger.info("RAGAgent initialized")

    def _format_context(self, documents: list) -> str:
        """Format retrieved documents into context."""
        if not documents:
            return "No relevant documents found."

        parts = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page_number", "?")
            section = doc.metadata.get("section", "")

            header = f"[{i}] Source: {source}, Page {page}"
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
                "page": doc.metadata.get("page_number"),
                "section": doc.metadata.get("section"),
                "ckd_stages": doc.metadata.get("ckd_stages", []),
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
        ckd_stage: Optional[int] = None,
    ) -> RAGAgentResponse:
        """
        Answer a query using RAG.

        Args:
            query: User question
            ckd_stage: Optional CKD stage for filtering

        Returns:
            RAGAgentResponse with answer and sources
        """
        stage = ckd_stage or self.ckd_stage

        # Configure retriever for CKD stage if available
        if stage and hasattr(self.retriever, 'with_config'):
            retriever = self.retriever.with_config(ckd_stage=stage)
        else:
            retriever = self.retriever

        # Retrieve documents
        try:
            documents = retriever.invoke(query)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return RAGAgentResponse(
                answer="I encountered an error retrieving information. Please try again.",
                sources=[],
                confidence=0.0,
            )

        if not documents:
            return RAGAgentResponse(
                answer="I couldn't find relevant information in the guidelines for your question. "
                       "Please try rephrasing or ask about a different aspect of CKD management.",
                sources=[],
                confidence=0.2,
            )

        # Format context and generate response
        context = self._format_context(documents)
        prompt = self.AGENT_PROMPT.format(context=context, question=query)

        try:
            answer = self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return RAGAgentResponse(
                answer="I encountered an error generating a response. Please try again.",
                sources=self._extract_sources(documents),
                confidence=0.1,
            )

        # Calculate confidence based on retrieval quality
        confidence = min(0.9, 0.5 + (len(documents) * 0.1))

        return RAGAgentResponse(
            answer=answer,
            sources=self._extract_sources(documents),
            confidence=confidence,
        )


def create_rag_agent(
    retriever: Any,
    llm: Any,
    ckd_stage: Optional[int] = None,
) -> RAGAgent:
    """Factory function to create a RAG agent."""
    return RAGAgent(retriever=retriever, llm=llm, ckd_stage=ckd_stage)
