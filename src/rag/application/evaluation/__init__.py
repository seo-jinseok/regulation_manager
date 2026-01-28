"""
Application layer for RAG quality evaluation.

Contains use cases and orchestration logic following Clean Architecture principles.
This layer coordinates domain objects and infrastructure implementations.
"""

from .evaluation_service import EvaluationService

__all__ = [
    "EvaluationService",
]
