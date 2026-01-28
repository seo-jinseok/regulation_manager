"""
Domain layer for RAG quality evaluation.

Contains core evaluation entities and value objects following Clean Architecture principles.
This layer has no external dependencies and defines the evaluation domain model.
"""

from .models import (
    EvaluationFramework,
    EvaluationResult,
    EvaluationThresholds,
    ImprovementSuggestion,
    MetricScore,
    PersonaProfile,
    QualityIssue,
    TestCase,
)
from .personas import PERSONAS, PersonaManager
from .quality_analyzer import QualityAnalyzer
from .quality_evaluator import RAGQualityEvaluator
from .synthetic_data import SyntheticDataGenerator

__all__ = [
    # Models
    "EvaluationFramework",
    "EvaluationResult",
    "EvaluationThresholds",
    "ImprovementSuggestion",
    "MetricScore",
    "PersonaProfile",
    "QualityIssue",
    "TestCase",
    # Core components
    "RAGQualityEvaluator",
    "PersonaManager",
    "QualityAnalyzer",
    "SyntheticDataGenerator",
    # Constants
    "PERSONAS",
]
