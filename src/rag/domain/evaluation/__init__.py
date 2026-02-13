"""
Domain layer for RAG quality evaluation.

Contains core evaluation entities and value objects following Clean Architecture principles.
This layer has no external dependencies and defines the evaluation domain model.
"""

from .custom_judge import (
    CustomEvaluationResult,
    CustomJudgeConfig,
    CustomLLMJudge,
)
from .llm_judge import (
    EvaluationBatch,
    EvaluationSummary,
    JudgeResult,
    LLMJudge,
    QualityLevel,
)
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
from .parallel_evaluator import (
    ParallelPersonaEvaluator,
    PersonaEvaluationResult,
    PersonaQuery,
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
    # Custom LLM-as-Judge
    "CustomLLMJudge",
    "CustomJudgeConfig",
    "CustomEvaluationResult",
    # New LLM-as-Judge (rag-quality-local skill)
    "LLMJudge",
    "JudgeResult",
    "EvaluationBatch",
    "EvaluationSummary",
    "QualityLevel",
    # Parallel Persona Evaluation
    "ParallelPersonaEvaluator",
    "PersonaEvaluationResult",
    "PersonaQuery",
    # Constants
    "PERSONAS",
]
