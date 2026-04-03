"""
Level 3: Multi-Agent RAG System

This module implements a multi-agent architecture:
- Orchestrator for query routing
- Specialized agents for different domains:
  - RAG Agent: Knowledge retrieval
  - Diet Agent: Dietary calculations
  - Medication Agent: Drug safety checks
  - Lifestyle Agent: Lifestyle guidance
"""

from .orchestrator import MultiAgentOrchestrator

__all__ = [
    "MultiAgentOrchestrator",
]
