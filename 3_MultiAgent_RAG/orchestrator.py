"""
Multi-Agent Orchestrator for the CKD RAG System.

Routes queries to appropriate specialized agents and
synthesizes responses from multiple agents when needed.
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .agents.rag_agent import RAGAgent, RAGAgentResponse
from .agents.diet_agent import DietAgent, DietAgentResponse
from .agents.medication_agent import MedicationAgent, MedicationAgentResponse
from .agents.lifestyle_agent import LifestyleAgent, LifestyleAgentResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of specialized agents."""
    RAG = "rag"
    DIET = "diet"
    MEDICATION = "medication"
    LIFESTYLE = "lifestyle"
    MULTI = "multi"  # Multiple agents needed


@dataclass
class RoutingDecision:
    """Decision about which agent(s) to use."""
    primary_agent: AgentType
    secondary_agents: list[AgentType] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class OrchestratorResponse:
    """Response from the orchestrator."""
    answer: str
    agents_used: list[str]
    routing_decision: RoutingDecision
    individual_responses: dict[str, Any] = field(default_factory=dict)
    ckd_stage: Optional[int] = None
    confidence: float = 0.0
    disclaimer: str = (
        "This information is for educational purposes only and should not replace "
        "professional medical advice. Always consult your healthcare provider for "
        "personalized recommendations."
    )


class MultiAgentOrchestrator:
    """
    Orchestrator for the multi-agent CKD system.

    Routes queries to specialized agents based on intent:
    - Diet questions → Diet Agent
    - Medication questions → Medication Agent
    - Lifestyle questions → Lifestyle Agent
    - General CKD info → RAG Agent
    - Complex questions → Multiple agents

    Example:
        >>> orchestrator = MultiAgentOrchestrator(retriever, llm)
        >>> response = orchestrator.process("What foods should I avoid with stage 3 CKD?")
        >>> print(response.answer)
        >>> print(f"Handled by: {response.agents_used}")
    """

    # Keywords for routing decisions
    ROUTING_KEYWORDS = {
        AgentType.DIET: [
            "diet", "food", "eat", "nutrition", "potassium", "phosphorus",
            "sodium", "salt", "protein", "meal", "calorie", "nutrient",
            "fruit", "vegetable", "meat", "dairy", "limit", "avoid eating",
            "how much", "daily intake", "restrict",
        ],
        AgentType.MEDICATION: [
            "medication", "medicine", "drug", "pill", "tablet", "prescription",
            "safe to take", "painkiller", "antibiotic", "ibuprofen", "nsaid",
            "aspirin", "paracetamol", "dose", "side effect", "interaction",
            "over-the-counter", "otc", "pharmacy",
        ],
        AgentType.LIFESTYLE: [
            "exercise", "activity", "walk", "swim", "gym", "workout",
            "blood pressure", "bp", "smoking", "quit", "cigarette",
            "stress", "anxiety", "sleep", "weight", "lifestyle",
            "hydration", "water intake", "fluid",
        ],
        AgentType.RAG: [
            "guideline", "nice", "recommendation", "what is", "explain",
            "stage", "egfr", "creatinine", "kidney function", "ckd",
            "progression", "treatment", "referral", "specialist",
            "dialysis", "transplant",
        ],
    }

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        pii_handler: Optional[Any] = None,
    ):
        """
        Initialize the orchestrator with all agents.

        Args:
            retriever: Document retriever for RAG agent
            llm: Language model for generation
            pii_handler: Optional PII handler for query anonymization
        """
        # Initialize agents
        self.rag_agent = RAGAgent(retriever=retriever, llm=llm)
        self.diet_agent = DietAgent(llm=llm)
        self.medication_agent = MedicationAgent(llm=llm)
        self.lifestyle_agent = LifestyleAgent(llm=llm)

        self.llm = llm
        self.pii_handler = pii_handler

        logger.info("MultiAgentOrchestrator initialized with all agents")

    def _calculate_keyword_scores(self, query: str) -> dict[AgentType, float]:
        """Calculate keyword match scores for each agent type."""
        query_lower = query.lower()
        scores = {}

        for agent_type, keywords in self.ROUTING_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            # Normalize by number of keywords
            scores[agent_type] = matches / len(keywords) if keywords else 0

        return scores

    def route(self, query: str) -> RoutingDecision:
        """
        Determine which agent(s) should handle the query.

        Args:
            query: User question

        Returns:
            RoutingDecision with primary and secondary agents
        """
        scores = self._calculate_keyword_scores(query)

        # Sort by score
        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Get top scoring agents
        if not sorted_agents or sorted_agents[0][1] == 0:
            # No clear match, default to RAG
            return RoutingDecision(
                primary_agent=AgentType.RAG,
                confidence=0.5,
                reasoning="No specific keywords matched. Using general knowledge retrieval.",
            )

        primary_agent, primary_score = sorted_agents[0]
        secondary_agents = []

        # Check if multiple agents are needed
        threshold = 0.3 * primary_score  # Secondary if score is at least 30% of primary
        for agent_type, score in sorted_agents[1:]:
            if score >= threshold and score > 0:
                secondary_agents.append(agent_type)

        # Determine if this is a multi-agent query
        if secondary_agents:
            return RoutingDecision(
                primary_agent=AgentType.MULTI,
                secondary_agents=[primary_agent] + secondary_agents,
                confidence=primary_score,
                reasoning=f"Query spans multiple domains: {[a.value for a in [primary_agent] + secondary_agents]}",
            )

        return RoutingDecision(
            primary_agent=primary_agent,
            confidence=min(1.0, primary_score * 10),  # Scale up for display
            reasoning=f"Query best matches {primary_agent.value} agent based on keywords.",
        )

    def _get_agent(self, agent_type: AgentType):
        """Get the agent instance for a type."""
        agents = {
            AgentType.RAG: self.rag_agent,
            AgentType.DIET: self.diet_agent,
            AgentType.MEDICATION: self.medication_agent,
            AgentType.LIFESTYLE: self.lifestyle_agent,
        }
        return agents.get(agent_type)

    def _call_agent(
        self,
        agent_type: AgentType,
        query: str,
        ckd_stage: Optional[int] = None,
        weight_kg: Optional[float] = None,
    ) -> Any:
        """Call a specific agent with the query."""
        agent = self._get_agent(agent_type)

        if agent is None:
            return None

        try:
            if agent_type == AgentType.DIET:
                return agent.answer(query, ckd_stage=ckd_stage, weight_kg=weight_kg)
            elif agent_type == AgentType.MEDICATION:
                return agent.answer(query, ckd_stage=ckd_stage)
            elif agent_type == AgentType.LIFESTYLE:
                return agent.answer(query, ckd_stage=ckd_stage)
            else:  # RAG
                return agent.answer(query, ckd_stage=ckd_stage)
        except Exception as e:
            logger.error(f"Agent {agent_type.value} failed: {e}")
            return None

    def _synthesize_responses(
        self,
        query: str,
        responses: dict[AgentType, Any],
        ckd_stage: Optional[int] = None,
    ) -> str:
        """Synthesize multiple agent responses into one coherent answer."""
        parts = []

        for agent_type, response in responses.items():
            if response is None:
                continue

            section_title = {
                AgentType.RAG: "General Information",
                AgentType.DIET: "Dietary Recommendations",
                AgentType.MEDICATION: "Medication Guidance",
                AgentType.LIFESTYLE: "Lifestyle Recommendations",
            }.get(agent_type, agent_type.value.title())

            parts.append(f"## {section_title}\n")

            # Extract answer from different response types
            if isinstance(response, RAGAgentResponse):
                parts.append(response.answer)
            elif isinstance(response, DietAgentResponse):
                parts.append(response.summary)
                for rec in response.recommendations[:2]:  # Limit to 2
                    parts.append(f"\n**{rec.nutrient}:** {rec.daily_limit} {rec.unit}")
            elif isinstance(response, MedicationAgentResponse):
                parts.append(response.general_guidance)
            elif isinstance(response, LifestyleAgentResponse):
                parts.append(response.summary)

            parts.append("\n")

        if not parts:
            return "I couldn't generate a comprehensive response. Please try rephrasing your question."

        return "\n".join(parts)

    def _format_single_response(self, response: Any) -> str:
        """Format a single agent response."""
        if isinstance(response, RAGAgentResponse):
            return response.answer
        elif isinstance(response, DietAgentResponse):
            parts = [response.summary, "\n"]
            for rec in response.recommendations:
                parts.append(f"\n**{rec.nutrient}:** {rec.daily_limit} {rec.unit}")
                parts.append(f"\n{rec.guidance}")
                if rec.foods_to_limit:
                    parts.append(f"\n*Limit:* {', '.join(rec.foods_to_limit[:5])}")
                if rec.foods_to_prefer:
                    parts.append(f"\n*Prefer:* {', '.join(rec.foods_to_prefer[:5])}")
            parts.append(f"\n\n_{response.disclaimer}_")
            return "\n".join(parts)
        elif isinstance(response, MedicationAgentResponse):
            parts = [response.general_guidance]
            if response.warnings:
                parts.append("\n**Warnings:**")
                for warning in response.warnings:
                    parts.append(f"- {warning}")
            parts.append(f"\n\n_{response.disclaimer}_")
            return "\n".join(parts)
        elif isinstance(response, LifestyleAgentResponse):
            parts = [response.summary, "\n"]
            for rec in response.recommendations:
                parts.append(f"\n**{rec.title}**")
                parts.append(rec.guidance[:300] + "..." if len(rec.guidance) > 300 else rec.guidance)
                if rec.tips:
                    parts.append("\n*Tips:*")
                    for tip in rec.tips[:3]:
                        parts.append(f"- {tip}")
            parts.append(f"\n\n_{response.disclaimer}_")
            return "\n".join(parts)
        else:
            return str(response)

    def process(
        self,
        query: str,
        ckd_stage: Optional[int] = None,
        weight_kg: Optional[float] = None,
    ) -> OrchestratorResponse:
        """
        Process a query through the multi-agent system.

        Args:
            query: User question
            ckd_stage: Patient's CKD stage (1-5)
            weight_kg: Patient's weight in kg (for protein calculations)

        Returns:
            OrchestratorResponse with synthesized answer
        """
        # Handle PII if handler available
        processed_query = query
        if self.pii_handler:
            try:
                result = self.pii_handler.anonymize(query)
                processed_query = result.anonymized_text
            except Exception as e:
                logger.warning(f"PII handling failed: {e}")

        # Route the query
        routing = self.route(processed_query)
        logger.info(f"Routing decision: {routing.primary_agent.value} (confidence: {routing.confidence:.2f})")

        individual_responses = {}
        agents_used = []

        if routing.primary_agent == AgentType.MULTI:
            # Call multiple agents
            for agent_type in routing.secondary_agents:
                response = self._call_agent(
                    agent_type, processed_query, ckd_stage, weight_kg
                )
                if response:
                    individual_responses[agent_type] = response
                    agents_used.append(agent_type.value)

            # Synthesize responses
            answer = self._synthesize_responses(
                processed_query, individual_responses, ckd_stage
            )
            confidence = routing.confidence * 0.9  # Slight penalty for multi-agent
        else:
            # Call single agent
            response = self._call_agent(
                routing.primary_agent, processed_query, ckd_stage, weight_kg
            )

            if response:
                individual_responses[routing.primary_agent] = response
                agents_used.append(routing.primary_agent.value)
                answer = self._format_single_response(response)
                confidence = getattr(response, 'confidence', routing.confidence)
            else:
                answer = "I encountered an error processing your question. Please try again."
                confidence = 0.1

        return OrchestratorResponse(
            answer=answer,
            agents_used=agents_used,
            routing_decision=routing,
            individual_responses={k.value: v for k, v in individual_responses.items()},
            ckd_stage=ckd_stage,
            confidence=confidence,
        )

    def get_agent_capabilities(self) -> dict[str, str]:
        """Get descriptions of each agent's capabilities."""
        return {
            "RAG Agent": (
                "Retrieves information from NICE guidelines and KidneyCareUK documents. "
                "Handles general CKD questions, guideline recommendations, and clinical information."
            ),
            "Diet Agent": (
                "Provides personalized dietary recommendations including daily limits for "
                "potassium, phosphorus, sodium, and protein based on CKD stage and weight."
            ),
            "Medication Agent": (
                "Checks medication safety for CKD patients. Identifies nephrotoxic drugs, "
                "provides dose adjustment guidance, and suggests alternatives."
            ),
            "Lifestyle Agent": (
                "Offers guidance on exercise, blood pressure monitoring, hydration, "
                "smoking cessation, stress management, and sleep hygiene for CKD patients."
            ),
        }


def create_orchestrator(
    retriever: Any,
    llm: Any,
    pii_handler: Optional[Any] = None,
) -> MultiAgentOrchestrator:
    """
    Factory function to create a multi-agent orchestrator.

    Args:
        retriever: Document retriever
        llm: Language model
        pii_handler: Optional PII handler

    Returns:
        Configured MultiAgentOrchestrator
    """
    return MultiAgentOrchestrator(
        retriever=retriever,
        llm=llm,
        pii_handler=pii_handler,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("MultiAgentOrchestrator module loaded.")
    print("\nAgent Capabilities:")

    # Show routing keywords
    for agent_type, keywords in MultiAgentOrchestrator.ROUTING_KEYWORDS.items():
        print(f"\n{agent_type.value}: {', '.join(keywords[:5])}...")
