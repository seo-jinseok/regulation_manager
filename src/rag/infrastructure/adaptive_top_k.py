"""
Adaptive Top-K Selector for Regulation RAG System.

Implements SPEC-RAG-SEARCH-001 TAG-003: Adaptive Top-K Selection.

Classifies queries into complexity levels and dynamically adjusts Top-K:
- SIMPLE: Single entity, direct question → Top-5
- MEDIUM: 2-3 entities, compound query → Top-10
- COMPLEX: Multiple entities, nested conditions → Top-15
- MULTI_PART: Multiple questions → Top-20

Features:
- Complexity scoring based on entity count, query length, question marks, conjunctions
- Latency guardrails (reduce Top-K if >500ms)
- Fallback to Top-10 if classification fails
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """
    Query complexity levels for adaptive Top-K selection.

    Part of SPEC-RAG-SEARCH-001 TAG-003: Adaptive Top-K.

    Levels:
        SIMPLE: Single keyword or regulation name (Top-5)
        MEDIUM: Natural questions with 1-2 concepts (Top-10)
        COMPLEX: Procedures, requirements, multiple conditions (Top-15)
        MULTI_PART: Multiple distinct questions/conjunctions (Top-20)
    """

    SIMPLE = "simple"  # Single keyword, regulation name
    MEDIUM = "medium"  # Natural questions
    COMPLEX = "complex"  # Multiple conditions
    MULTI_PART = "multi_part"  # Multiple queries


@dataclass
class TopKConfig:
    """
    Top-K configuration for each complexity level.

    Part of SPEC-RAG-SEARCH-001 TAG-003: Adaptive Top-K.

    Attributes:
        simple: Top-K for simple queries (default: 5)
        medium: Top-K for medium queries (default: 10)
        complex: Top-K for complex queries (default: 15)
        multi_part: Top-K for multi-part queries (default: 20)
        max_limit: Absolute maximum Top-K to prevent performance issues (default: 25)
        latency_threshold_ms: Latency threshold in ms for Top-K reduction (default: 500)
    """

    simple: int = 5
    medium: int = 10
    complex: int = 15
    multi_part: int = 20
    max_limit: int = 25
    latency_threshold_ms: int = 500

    def __post_init__(self):
        """Validate configuration values."""
        if self.simple < 1:
            raise ValueError(f"simple Top-K must be >= 1, got {self.simple}")
        if self.simple > self.medium:
            raise ValueError(
                f"simple Top-K must be <= medium, got {self.simple} > {self.medium}"
            )
        if self.medium > self.complex:
            raise ValueError(
                f"medium Top-K must be <= complex, got {self.medium} > {self.complex}"
            )
        if self.complex > self.multi_part:
            raise ValueError(
                f"complex Top-K must be <= multi_part, got {self.complex} > {self.multi_part}"
            )
        if self.multi_part > self.max_limit:
            raise ValueError(
                f"multi_part Top-K must be <= max_limit, got {self.multi_part} > {self.max_limit}"
            )


@dataclass
class ComplexityAnalysisResult:
    """
    Result of query complexity analysis.

    Attributes:
        complexity: The classified complexity level
        score: Numerical complexity score (0-100)
        top_k: Recommended Top-K value
        factors: Dict of factors that influenced the classification
        processing_time_ms: Time taken to analyze (for latency monitoring)
    """

    complexity: QueryComplexity
    score: float
    top_k: int
    factors: Dict[str, float]
    processing_time_ms: float


class AdaptiveTopKSelector:
    """
    Adaptive Top-K selection based on query complexity.

    Part of SPEC-RAG-SEARCH-001 TAG-003: Adaptive Top-K.

    This class analyzes query complexity and selects appropriate Top-K values
    to balance recall and performance.

    Complexity factors (REQ-AT-001 ~ REQ-AT-011):
    - Entity count: More entities → higher Top-K
    - Query length: Longer queries → higher Top-K
    - Question marks: Natural language questions → higher Top-K
    - Conjunctions: Multiple parts → higher Top-K
    - Special keywords: Procedures, requirements → higher Top-K

    Example:
        selector = AdaptiveTopKSelector()
        result = selector.select_top_k("장학금 신청 방법")
        # Returns: Top-10 (MEDIUM complexity)

        result = selector.select_top_k("장학금 신청 방법과 자격 요건")
        # Returns: Top-15 (COMPLEX - multiple concepts)

        result = selector.select_top_k("교원인사규정")
        # Returns: Top-5 (SIMPLE - single regulation name)
    """

    # Patterns for multi-part query detection (REQ-AT-005)
    MULTI_PART_INDICATORS = [
        r"\s+그리고\s+",
        r"\s+또는\s+",
        r"\s+및\s+",
        r",\s*",
        r"、",
        r";\s*",
    ]

    # Complex query indicators (REQ-AT-004)
    COMPLEX_INDICATORS = [
        "방법",
        "절차",
        "신청",
        "자격",
        "요건",
        "조건",
        "기준",
        "제한",
        "혜택",
        "지급",
        "지원",
        "발급",
        "제출",
        "서류",
        "구비서류",
        "심사",
        "승인",
    ]

    # Simple query indicators (REQ-AT-002)
    SIMPLE_PATTERNS = [
        re.compile(r"^[가-힣a-zA-Z0-9\s]+$"),  # No special characters
        re.compile(r"규정$|학칙$|내규$|세칙$|지침$"),  # Regulation suffixes
    ]

    # Question markers (REQ-AT-003)
    QUESTION_MARKERS = [
        "?",
        "어떻게",
        "무엇",
        "왜",
        "언제",
        "어디",
        "누가",
        "어떤",
        "할까",
        "인가",
        "방법",  # Also a complex indicator, context matters
        "절차",  # Also a complex indicator
    ]

    def __init__(self, config: Optional[TopKConfig] = None):
        """
        Initialize adaptive Top-K selector.

        Args:
            config: Top-K configuration (uses default if not provided)
        """
        self._config = config or TopKConfig()
        self._latency_history: List[float] = []

        # Compile multi-part patterns for efficiency
        self._multi_part_patterns = [
            re.compile(pattern) for pattern in self.MULTI_PART_INDICATORS
        ]

    def select_top_k(self, query: str, latency_guardrails: bool = True) -> int:
        """
        Select Top-K based on query complexity.

        This is the main entry point for Top-K selection.
        It analyzes the query and returns the recommended Top-K value.

        Args:
            query: The search query text
            latency_guardrails: Whether to reduce Top-K if latency is high (default: True)

        Returns:
            Recommended Top-K value

        Example:
            selector = AdaptiveTopKSelector()
            top_k = selector.select_top_k("장학금 신청 방법")
            # Returns: 10 (MEDIUM complexity)
        """
        if not query or not query.strip():
            return self._config.medium

        analysis = self.analyze_complexity(query)

        # Apply latency guardrails if enabled
        if latency_guardrails and self._latency_history:
            avg_latency = sum(self._latency_history) / len(self._latency_history)
            if avg_latency > self._config.latency_threshold_ms:
                # Reduce Top-K if latency is too high
                reduced_top_k = max(analysis.top_k // 2, self._config.simple)
                logger.warning(
                    f"High latency detected ({avg_latency:.0f}ms), "
                    f"reducing Top-K from {analysis.top_k} to {reduced_top_k}"
                )
                return reduced_top_k

        return analysis.top_k

    def analyze_complexity(self, query: str) -> ComplexityAnalysisResult:
        """
        Analyze query complexity and return detailed result.

        This method performs comprehensive complexity analysis and returns
        detailed factors that influenced the classification.

        Args:
            query: The search query text

        Returns:
            ComplexityAnalysisResult with classification details
        """
        start_time = time.time()

        # Calculate complexity factors
        factors = self._calculate_complexity_factors(query)

        # Calculate total complexity score
        score = self._calculate_complexity_score(factors)

        # Classify complexity
        complexity = self._classify_complexity(score, factors, query)

        # Select Top-K based on complexity
        top_k = self._get_top_k_for_complexity(complexity)

        processing_time_ms = (time.time() - start_time) * 1000

        return ComplexityAnalysisResult(
            complexity=complexity,
            score=score,
            top_k=top_k,
            factors=factors,
            processing_time_ms=processing_time_ms,
        )

    def _calculate_complexity_factors(self, query: str) -> Dict[str, float]:
        """
        Calculate individual complexity factors.

        Factors include:
        - entity_count: Number of entities found
        - query_length: Normalized query length (0-1)
        - question_marks: Number of question markers
        - conjunctions: Number of conjunction indicators
        - complex_keywords: Number of complex keywords
        - has_multi_part: Whether query has multiple parts
        """
        factors = {
            "entity_count": 0.0,
            "query_length": 0.0,
            "question_marks": 0.0,
            "conjunctions": 0.0,
            "complex_keywords": 0.0,
            "has_multi_part": 0.0,
        }

        # Entity count (approximate based on word count)
        words = query.split()
        factors["entity_count"] = min(len(words) / 5.0, 1.0)  # Normalize to 0-1

        # Query length (normalize: 0 chars = 0, 50+ chars = 1)
        factors["query_length"] = min(len(query) / 50.0, 1.0)

        # Question markers
        for marker in self.QUESTION_MARKERS:
            if marker in query:
                factors["question_marks"] += 1
        factors["question_marks"] = min(factors["question_marks"] / 2.0, 1.0)

        # Conjunctions (multi-part detection)
        for pattern in self._multi_part_patterns:
            if pattern.search(query):
                factors["conjunctions"] += 1
        factors["conjunctions"] = min(factors["conjunctions"], 1.0)

        # Complex keywords
        for keyword in self.COMPLEX_INDICATORS:
            if keyword in query:
                factors["complex_keywords"] += 1
        factors["complex_keywords"] = min(factors["complex_keywords"] / 3.0, 1.0)

        # Multi-part indicator
        factors["has_multi_part"] = 1.0 if factors["conjunctions"] > 0 else 0.0

        return factors

    def _calculate_complexity_score(self, factors: Dict[str, float]) -> float:
        """
        Calculate overall complexity score from factors.

        Score calculation:
        - Base score: 0-100
        - Each factor contributes weighted score
        - Multi-part queries get bonus
        """
        weights = {
            "entity_count": 15.0,
            "query_length": 10.0,
            "question_marks": 15.0,
            "conjunctions": 25.0,
            "complex_keywords": 25.0,
            "has_multi_part": 10.0,
        }

        score = sum(factors[key] * weights[key] for key in weights)
        return min(score, 100.0)

    def _classify_complexity(
        self, score: float, factors: Dict[str, float], query: str
    ) -> QueryComplexity:
        """
        Classify query complexity based on score and factors.

        Classification rules (REQ-AT-001 ~ REQ-AT-005):
        - SIMPLE: score < 30, regulation name, single keyword
        - MEDIUM: score 30-50, natural questions
        - COMPLEX: score 50-70, procedures, requirements
        - MULTI_PART: score > 70 or has conjunctions
        """
        # Check for multi-part first (highest priority)
        if factors["has_multi_part"] > 0 or score > 70:
            return QueryComplexity.MULTI_PART

        # Check for very long queries (high entity count)
        if factors["entity_count"] >= 0.8:
            return QueryComplexity.COMPLEX

        # Check for natural questions (question markers should prioritize MEDIUM over COMPLEX)
        if factors["question_marks"] > 0:
            # Natural question with complex keywords: check score
            if score < 50:
                return QueryComplexity.MEDIUM
            # High score natural question: could be complex
            return QueryComplexity.COMPLEX

        # Check for complex queries (procedures, requirements)
        if score > 50 or factors["complex_keywords"] > 0:
            return QueryComplexity.COMPLEX

        # Check for medium complexity
        if score > 30:
            return QueryComplexity.MEDIUM

        # Check for simple queries (regulation names)
        if self._is_simple_query(query):
            return QueryComplexity.SIMPLE

        # Default to medium
        return QueryComplexity.MEDIUM

    def _is_simple_query(self, query: str) -> bool:
        """
        Check if query is simple (single keyword or regulation name).

        Simple queries (REQ-AT-002):
        - Single word or short phrase (≤ 2 words)
        - Regulation name only (e.g., "교원인사규정")
        - No question markers or complex keywords
        """
        words = query.split()

        # Single word or short phrase
        if len(words) <= 2:
            # Check for complex indicators
            has_complex = any(keyword in query for keyword in self.COMPLEX_INDICATORS)
            has_question = any(marker in query for marker in self.QUESTION_MARKERS)
            if not has_complex and not has_question:
                return True

        # Regulation name only
        if any(pattern.search(query) for pattern in self.SIMPLE_PATTERNS):
            # Check if it's just a regulation name without other content
            has_question = any(marker in query for marker in self.QUESTION_MARKERS)
            if not has_question:
                return True

        return False

    def _get_top_k_for_complexity(self, complexity: QueryComplexity) -> int:
        """
        Get Top-K value for complexity level.

        Maps complexity to Top-K (REQ-AT-002 ~ REQ-AT-005):
        - SIMPLE: 5
        - MEDIUM: 10
        - COMPLEX: 15
        - MULTI_PART: 20
        """
        mapping = {
            QueryComplexity.SIMPLE: self._config.simple,
            QueryComplexity.MEDIUM: self._config.medium,
            QueryComplexity.COMPLEX: self._config.complex,
            QueryComplexity.MULTI_PART: self._config.multi_part,
        }
        return mapping.get(complexity, self._config.medium)

    def record_latency(self, latency_ms: float):
        """
        Record search latency for adaptive behavior.

        Maintains a history of recent latencies for guardrail decisions.

        Args:
            latency_ms: Search latency in milliseconds
        """
        self._latency_history.append(latency_ms)
        # Keep only last 10 measurements
        if len(self._latency_history) > 10:
            self._latency_history.pop(0)

    def clear_latency_history(self):
        """Clear latency history."""
        self._latency_history.clear()

    def get_average_latency(self) -> Optional[float]:
        """
        Get average latency from history.

        Returns:
            Average latency in ms, or None if no history
        """
        if not self._latency_history:
            return None
        return sum(self._latency_history) / len(self._latency_history)
