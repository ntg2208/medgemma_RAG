"""Base agent class for the Multi-Agent CKD System."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


def _get_default_disclaimer() -> str:
    """Get the default medical disclaimer."""
    return (
        "This information is for educational purposes only and should not replace "
        "professional medical advice. Always consult your healthcare provider for "
        "personalized recommendations."
    )


@dataclass
class AgentResponse:
    """Base response from any agent.

    Attributes:
        answer: The agent's response text
        confidence: Confidence score (0.0 to 1.0)
        disclaimer: Medical disclaimer to include
    """
    answer: str
    confidence: float = 0.0
    disclaimer: str = field(default_factory=_get_default_disclaimer)


class BaseAgent(ABC):
    """Abstract base class for all agents.

    All agents in the multi-agent system must implement:
    - can_handle(): Check if this agent can handle a query
    - answer(): Process the query and return response

    Example:
        >>> class DietAgent(BaseAgent):
        ...     def can_handle(self, query: str) -> bool:
        ...         return "diet" in query.lower()
        ...
        ...     def answer(self, query: str) -> AgentResponse:
        ...         return AgentResponse(answer="Diet advice here")
    """

    def __init__(self, llm: Optional[object] = None):
        """Initialize agent.

        Args:
            llm: Optional language model for generating responses
        """
        self.llm = llm

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """Check if this agent can handle the query.

        Args:
            query: User question

        Returns:
            True if this agent should handle the query
        """
        pass

    @abstractmethod
    def answer(self, query: str, **kwargs) -> AgentResponse:
        """Answer a query.

        Args:
            query: User question
            **kwargs: Additional parameters (e.g., ckd_stage, weight_kg)

        Returns:
            AgentResponse with answer and metadata
        """
        pass
