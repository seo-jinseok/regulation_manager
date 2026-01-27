"""
Domain LLM components.

Provides circuit breaker pattern for LLM provider resilience
and ambiguity classifier for query disambiguation.
"""

from .ambiguity_classifier import (
    AmbiguityClassifier,
    AmbiguityClassifierConfig,
    AmbiguityFactors,
    AmbiguityLevel,
    ClassificationResult,
    DisambiguationDialog,
    DisambiguationOption,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerOpenError,
    CircuitState,
)

__all__ = [
    "AmbiguityClassifier",
    "AmbiguityClassifierConfig",
    "AmbiguityFactors",
    "AmbiguityLevel",
    "ClassificationResult",
    "DisambiguationDialog",
    "DisambiguationOption",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "CircuitBreakerOpenError",
    "CircuitState",
]
