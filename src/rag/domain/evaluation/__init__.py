"""
Domain layer for RAG quality evaluation.

Contains core evaluation entities and value objects following Clean Architecture principles.
This layer has no external dependencies and defines the evaluation domain model.
"""

from .batch_executor import (
    BatchEvaluationExecutor,
    BatchResult,
    CacheEntry,
    CostEstimator,
    EvaluationCache,
    RateLimitConfig,
    RateLimiter,
    RateLimitStats,
)
from .custom_judge import (
    CustomEvaluationResult,
    CustomJudgeConfig,
    CustomLLMJudge,
)
from .failure_classifier import (
    FailureClassifier,
    FailureSummary,
    FailureType,
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
    PersonaMetrics,
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
from .recommendation_engine import (
    Priority,
    Recommendation,
    RecommendationEngine,
)
from .regulation_query_generator import (
    QueryTemplate,
    RegulationArticle,
    RegulationQueryGenerator,
)
from .spec_generator import (
    SPECDocument,
    SPECGenerator,
)
from .synthetic_data import SyntheticDataGenerator

__all__ = [
    # Models
    "EvaluationFramework",
    "EvaluationResult",
    "EvaluationThresholds",
    "ImprovementSuggestion",
    "MetricScore",
    "PersonaMetrics",
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
    # Batch Execution (SPEC-RAG-EVAL-001 P0)
    "BatchEvaluationExecutor",
    "BatchResult",
    "CostEstimator",
    "EvaluationCache",
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitStats",
    "CacheEntry",
    # Regulation Query Generation (SPEC-RAG-EVAL-001 P0)
    "RegulationQueryGenerator",
    "RegulationArticle",
    "QueryTemplate",
    # Failure Classification (SPEC-RAG-EVAL-001 P1)
    "FailureClassifier",
    "FailureType",
    "FailureSummary",
    # Recommendation Engine (SPEC-RAG-EVAL-001 P1)
    "RecommendationEngine",
    "Recommendation",
    "Priority",
    # SPEC Generation (SPEC-RAG-EVAL-001 P1)
    "SPECGenerator",
    "SPECDocument",
    # Constants
    "PERSONAS",
]
