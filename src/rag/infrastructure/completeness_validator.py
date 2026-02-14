"""
Completeness Validator for period-related RAG responses.

Validates that responses to period-related queries include either:
1. Specific date/period information, OR
2. Alternative guidance (calendar reference, department contact, etc.)
"""

import re
from dataclasses import dataclass, field
from typing import List, Pattern

from src.rag.infrastructure.period_keyword_detector import PeriodKeywordDetector

# Score constants for clarity and maintainability
SCORE_BOTH_ELEMENTS = 1.0
SCORE_SPECIFIC_ONLY = 0.85
SCORE_GUIDANCE_ONLY = 0.7
SCORE_NEITHER = 0.3
SCORE_NON_PERIOD_QUERY = 0.7


@dataclass
class CompletenessResult:
    """
    Result of completeness validation for period-related queries.

    Attributes:
        is_complete: True if response meets completeness criteria
        has_specific_info: True if dates/periods found in response
        has_alternative_guidance: True if guidance message found in response
        missing_elements: List of what's missing from the response
        score: Completeness score (0.0 to 1.0)
    """

    is_complete: bool
    has_specific_info: bool
    has_alternative_guidance: bool
    missing_elements: List[str] = field(default_factory=list)
    score: float = 0.0


class CompletenessValidator:
    """
    Validates completeness of responses to period-related queries.

    For period-related queries, a complete response should include either:
    - Specific date/period information (dates, ranges, durations)
    - Alternative guidance (calendar reference, department contact, etc.)

    Score calculation:
    - 1.0: Both specific info and alternative guidance
    - 0.85: Specific info only
    - 0.7: Alternative guidance only
    - 0.3: Neither present
    """

    # Date patterns for specific information detection (compiled for performance)
    _DATE_PATTERNS: List[Pattern] = [
        # Korean date formats
        re.compile(r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일"),  # 2024년 3월 1일
        re.compile(r"\d{1,2}월\s*\d{1,2}일"),  # 3월 1일
        re.compile(r"\d{1,2}일"),  # 1일, 15일 (standalone day)
        # Numeric date formats
        re.compile(r"\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}"),  # 2024.03.01, 2024-03-01
        re.compile(r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4}"),  # 03.01.2024
    ]

    # Period/duration patterns (compiled for performance)
    _PERIOD_PATTERNS: List[Pattern] = [
        re.compile(r"\d+\s*주"),  # 2주
        re.compile(r"\d+\s*개월"),  # 3개월
        re.compile(r"\d+\s*일"),  # 7일 (when used as duration context)
        re.compile(r"\d+\s*시간"),  # 24시간
        re.compile(r"이내"),  # within
        re.compile(r"동안"),  # during
        re.compile(r"부터.*까지"),  # from ... to
        re.compile(r"\d{1,2}월\s*\d{1,2}일.*\d{1,2}월\s*\d{1,2}일"),  # date range
    ]

    # Alternative guidance patterns in Korean (compiled for performance)
    _ALTERNATIVE_GUIDANCE_PATTERNS: List[Pattern] = [
        re.compile(r"학사일정.*확인"),  # 학사일정을 확인
        re.compile(r"담당\s*부서.*문의"),  # 담당 부서에 문의
        re.compile(r"규정.*구체적\s*기한.*명시.*않"),  # 규정에 구체적 기한이 명시되어 있지 않습니다
        re.compile(r"학교\s*홈페이지.*참고"),  # 학교 홈페이지 참고
    ]

    def __init__(self):
        """Initialize CompletenessValidator with PeriodKeywordDetector."""
        self.period_detector = PeriodKeywordDetector()

    def check_specific_info_present(self, response: str) -> bool:
        """
        Check if response contains specific date/period information.

        Args:
            response: The response text to check.

        Returns:
            True if dates/periods found, False otherwise.
        """
        if not response or not response.strip():
            return False

        # Check for date patterns (compiled regex for performance)
        for pattern in self._DATE_PATTERNS:
            if pattern.search(response):
                return True

        # Check for period/duration patterns (compiled regex for performance)
        for pattern in self._PERIOD_PATTERNS:
            if pattern.search(response):
                return True

        return False

    def check_alternative_guidance_present(self, response: str) -> bool:
        """
        Check if response contains alternative guidance message.

        Args:
            response: The response text to check.

        Returns:
            True if guidance message found, False otherwise.
        """
        if not response or not response.strip():
            return False

        # Check for alternative guidance patterns (compiled regex for performance)
        for pattern in self._ALTERNATIVE_GUIDANCE_PATTERNS:
            if pattern.search(response):
                return True

        return False

    def validate_period_response(self, query: str, response: str) -> CompletenessResult:
        """
        Validate completeness of a response to a period-related query.

        Args:
            query: The user's query text.
            response: The generated response text.

        Returns:
            CompletenessResult with validation details and score.
        """
        # Check if query is period-related
        is_period_query = self.period_detector.is_period_related(query) if query else False

        # If not a period query, default to complete with base score
        if not is_period_query:
            return CompletenessResult(
                is_complete=True,
                has_specific_info=False,
                has_alternative_guidance=False,
                missing_elements=[],
                score=SCORE_NON_PERIOD_QUERY,
            )

        # Check response content
        has_specific_info = self.check_specific_info_present(response)
        has_alternative_guidance = self.check_alternative_guidance_present(response)

        # Determine completeness and score
        missing_elements = self._determine_missing_elements(
            has_specific_info, has_alternative_guidance
        )
        score = self._calculate_score(has_specific_info, has_alternative_guidance)

        # Response is complete if it has either specific info OR alternative guidance
        is_complete = has_specific_info or has_alternative_guidance

        return CompletenessResult(
            is_complete=is_complete,
            has_specific_info=has_specific_info,
            has_alternative_guidance=has_alternative_guidance,
            missing_elements=missing_elements,
            score=score,
        )

    def _determine_missing_elements(
        self, has_specific_info: bool, has_alternative_guidance: bool
    ) -> List[str]:
        """
        Determine which elements are missing from the response.

        Args:
            has_specific_info: Whether specific info is present
            has_alternative_guidance: Whether alternative guidance is present

        Returns:
            List of missing element names.
        """
        missing = []
        if not has_specific_info:
            missing.append("specific_info")
        if not has_alternative_guidance:
            missing.append("alternative_guidance")
        return missing

    def _calculate_score(self, has_specific_info: bool, has_alternative_guidance: bool) -> float:
        """
        Calculate completeness score based on present elements.

        Args:
            has_specific_info: Whether specific info is present
            has_alternative_guidance: Whether alternative guidance is present

        Returns:
            Completeness score (0.0 to 1.0).
        """
        if has_specific_info and has_alternative_guidance:
            return SCORE_BOTH_ELEMENTS
        elif has_specific_info:
            return SCORE_SPECIFIC_ONLY
        elif has_alternative_guidance:
            return SCORE_GUIDANCE_ONLY
        else:
            return SCORE_NEITHER
