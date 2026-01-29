"""
Infrastructure storage layer for RAG evaluation results.

This module provides persistent storage for evaluation results,
enabling historical tracking and trend analysis.
"""

from .evaluation_store import EvaluationStatistics, EvaluationStore

__all__ = [
    "EvaluationStore",
    "EvaluationStatistics",
]
