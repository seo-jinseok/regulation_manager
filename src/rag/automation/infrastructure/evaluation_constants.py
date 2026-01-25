"""
Evaluation Constants for RAG Quality Assessment.

Defines scoring thresholds, auto-fail conditions, and detection patterns.
Centralizes magic numbers to improve maintainability.

Enhanced with more comprehensive patterns for Korean university regulations.
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
    Enhanced with comprehensive Korean university regulation patterns.
    """

    # Generalization phrases indicate vague, non-specific answers
    GENERALIZATION_PATTERNS: List[str] = [
        r"대학마다\s*다를\s*수",
        r"각\s*대학의\s*상황에\s*따라",
        r"일반적으로",
        r"보통은",
        r"대체로",
        r"기관에\s*따라\s*다르",
        r"상황에\s*따라\s*다르",
        r"학교\s*규정에\s*맞게",
        r"관련\s*부서에\s*문의",
        r"확인이\s*필요",
        r"해당\s*규정.*?참고",  # More flexible: allows particles
        r"각\s*기관\s*차이",
        r"대학\s*상황에\s*따라",
    ]

    # Vague answer patterns (no specific regulation cited)
    VAGUE_ANSWER_PATTERNS: List[str] = [
        r"^(네|예|아니요)\.?$",  # Just yes/no
        r"규정에\s*딸?르?",  # Matches "따르" with conjugation
        r"정해진\s*바가\s*있",
        r"별도로\s*정해",
        r"관련\s*규정\s*참조",
        r"규정\s*준수",
    ]

    # Source citation patterns for regulation references
    CITATION_PATTERNS: List[str] = [
        r"제\d+\s*[조항]",  # 제N조/제N항
        r"\d+\s*[조항]",  # N조/N항
        r"\d+조\s*\d+항",  # N조 M항
        r"\d+조\s*제\d+항",  # N조 제M항
        r"\w*규정",  # Any word ending with 규정
        r"\w*학칙",  # Any word ending with 학칙
        r"\w*시행세칙",  # Any word ending with 시행세칙
    ]

    # Practical information patterns (deadlines, requirements, contacts)
    PRACTICAL_INFO_PATTERNS: List[str] = [
        r"\d+\s*년\s*이내",  # N년 이내
        r"\d+\s*개?월\s*이내",  # N월/N개월 이내
        r"\d+\s*일\s*이내",  # N일 이내
        r"\d+\s*시간\s*이내",  # N시간 이내
        r"\d+\s*분\s*이내",  # N분 이내
        r"\d+\s*주일?\s*이내",  # N주/N주일 이내
        r"\d+\s*년\s*전",  # N년 전
        r"\d+\s*개?월\s*전",  # N월/N개월 전
        r"\d+\s*일\s*전",  # N일 전
        r"\d+\s*시간\s*전",  # N시간 전
        r"\d+\s*분\s*전",  # N분 전
        r"\d+\s*주일?\s*전",  # N주/N주일 전
        r"\d+회\s*이상",  # N회 이상
        r"\d+\s*[학점점]",  # N학점/N점
        r"\d+\.\d+\s*이상",  # N.N 이상
        r"\w+\s*부서",  # Any word + 부서
        r"\w+\s*담당자",  # Any word + 담당자
        r"학부사무실",
        r"교학과",
        r"학적과",
        r"장학팀",
        r"학사지원팀",
        r"교무처",
        r"학생처",
    ]

    # Action verbs indicating actionable advice
    ACTION_VERBS: List[str] = [
        "신청",
        "제출",
        "방문",
        "연락",
        "확인",
        "준비",
        "등록",
        "완료",
        "발급",
        "심사",
        "신고",
        "접수",
        "승인",
        "변경",
        "취소",
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
    def is_vague_answer(answer: str) -> bool:
        """Check if answer is vague without specific information."""
        return AutoFailPatterns.matches_any_pattern(
            answer, AutoFailPatterns.VAGUE_ANSWER_PATTERNS
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
