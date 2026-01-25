"""
Evaluation Constants for RAG Quality Assessment.

Defines scoring thresholds, auto-fail conditions, and detection patterns.
Centralizes magic numbers to improve maintainability.
"""

import re
from typing import List


class ScoringThresholds:
    """
    Quality scoring thresholds and limits.

    All scores are normalized to 0.0-1.0 range unless specified.
    """

    # Score boundaries
    MIN_SCORE = 0.0
    MAX_SCORE = 1.0
    MAX_PRACTICALITY_SCORE = 0.5
    MAX_ACTIONABILITY_SCORE = 0.5

    # Total score boundaries (sum of all dimensions)
    MIN_TOTAL_SCORE = 0.0
    MAX_TOTAL_SCORE = 5.0  # 1.0 * 4 + 0.5 * 2
    PASS_THRESHOLD = 4.0

    # Minimum answer length
    MIN_ANSWER_LENGTH = 10
    TARGET_ANSWER_LENGTH = 200

    # Weight factors
    KEYWORD_OVERLAP_RELEVANCE_WEIGHT = 0.8
    BASE_RELEVANCE_SCORE = 0.2
    NO_CITATION_SCORE = 0.3
    NO_PRACTICAL_INFO_SCORE = 0.2
    WITH_PRACTICAL_INFO_SCORE = 0.5
    NO_ACTION_VERB_SCORE = 0.2
    WITH_ACTION_VERB_SCORE = 0.5


class AutoFailPatterns:
    """
    Patterns for detecting automatic failure conditions.

    Answers matching these patterns receive automatic failing scores.
    """

    # Generalization phrases indicate vague, non-specific answers
    GENERALIZATION_PATTERNS: List[str] = [
        r"대학마다\s*다를\s*수",
        r"각\s*대학의\s*상황에\s*따라",
        r"일반적으로",
        r"보통은",
        r"대체로",
    ]

    # Source citation patterns for regulation references
    CITATION_PATTERNS: List[str] = [
        r"제\d+[조항]",
        r"\d+[조항]\s*.*규정",
        r"[가-힣]+규정",
        r"[가-힣]+학칙",
    ]

    # Practical information patterns (deadlines, requirements, contacts)
    PRACTICAL_INFO_PATTERNS: List[str] = [
        r"\d+[년월일시점분]\s*이내",
        r"\d+회\s*이상",
        r"\d+[학점점]",
        r"\d+\.\d+\s*이상",
        r"[가-힣]+\s*부서",
        r"[가-힣]+\s*담당자",
    ]

    # Action verbs indicating actionable advice
    ACTION_VERBS: List[str] = [
        "신청",
        "제출",
        "방문",
        "연락",
        "확인",
        "준비",
    ]

    @staticmethod
    def matches_any_pattern(text: str, patterns: List[str]) -> bool:
        """
        Check if text matches any of the given regex patterns.

        Args:
            text: Text to check
            patterns: List of regex patterns

        Returns:
            True if any pattern matches
        """
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def is_generalization(answer: str) -> bool:
        """Check if answer contains generalization phrases."""
        return AutoFailPatterns.matches_any_pattern(
            answer, AutoFailPatterns.GENERALIZATION_PATTERNS
        )

    @staticmethod
    def has_citation(answer: str) -> bool:
        """Check if answer contains source citations."""
        return AutoFailPatterns.matches_any_pattern(
            answer, AutoFailPatterns.CITATION_PATTERNS
        )

    @staticmethod
    def has_practical_info(answer: str) -> bool:
        """Check if answer contains practical information."""
        return AutoFailPatterns.matches_any_pattern(
            answer, AutoFailPatterns.PRACTICAL_INFO_PATTERNS
        )

    @staticmethod
    def has_action_verbs(answer: str) -> bool:
        """Check if answer contains action verbs."""
        return any(verb in answer for verb in AutoFailPatterns.ACTION_VERBS)


class QualityDimensions:
    """
    Quality dimension identifiers for evaluation.

    Maps to the 6 quality dimensions:
    - Accuracy (1.0): Correctness of factual information
    - Completeness (1.0): Coverage of question aspects
    - Relevance (1.0): Alignment with user intent
    - Source Citation (1.0): Proper regulation references
    - Practicality (0.5): Deadlines, requirements, contact info
    - Actionability (0.5): Clear next steps for user
    """

    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    SOURCE_CITATION = "source_citation"
    PRACTICALITY = "practicality"
    ACTIONABILITY = "actionability"

    # Maximum scores per dimension
    MAX_SCORES = {
        ACCURACY: 1.0,
        COMPLETENESS: 1.0,
        RELEVANCE: 1.0,
        SOURCE_CITATION: 1.0,
        PRACTICALITY: 0.5,
        ACTIONABILITY: 0.5,
    }
