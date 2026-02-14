"""
Unit tests for CompletenessValidator.

TDD approach: These tests define the expected behavior of completeness validation
for period-related queries.
"""

import pytest

from src.rag.infrastructure.completeness_validator import (
    CompletenessValidator,
    CompletenessResult,
)


class TestCompletenessResult:
    """Tests for CompletenessResult dataclass."""

    def test_completeness_result_creation(self):
        """Create CompletenessResult with all fields."""
        result = CompletenessResult(
            is_complete=True,
            has_specific_info=True,
            has_alternative_guidance=False,
            missing_elements=[],
            score=0.85,
        )
        assert result.is_complete is True
        assert result.has_specific_info is True
        assert result.has_alternative_guidance is False
        assert result.missing_elements == []
        assert result.score == 0.85

    def test_completeness_result_with_missing_elements(self):
        """Create result with missing elements list."""
        result = CompletenessResult(
            is_complete=False,
            has_specific_info=False,
            has_alternative_guidance=False,
            missing_elements=["specific_info", "alternative_guidance"],
            score=0.3,
        )
        assert "specific_info" in result.missing_elements
        assert "alternative_guidance" in result.missing_elements


class TestCompletenessValidator:
    """Tests for CompletenessValidator class."""

    @pytest.fixture
    def validator(self) -> CompletenessValidator:
        """Create CompletenessValidator instance."""
        return CompletenessValidator()

    # ==================== check_specific_info_present Tests ====================

    def test_check_specific_info_present_with_date(self, validator):
        """Return True when response contains date information."""
        # Korean date patterns
        assert validator.check_specific_info_present("2024년 3월 1일부터 시작합니다") is True
        assert validator.check_specific_info_present("기간은 1월 15일부터 2월 28일까지입니다") is True
        assert validator.check_specific_info_present("마감일은 12월 31일입니다") is True

    def test_check_specific_info_present_with_period(self, validator):
        """Return True when response contains period information."""
        # Period patterns
        assert validator.check_specific_info_present("신청 기간은 2주입니다") is True
        assert validator.check_specific_info_present("7일 이내에 제출해야 합니다") is True
        assert validator.check_specific_info_present("3개월 동안 유효합니다") is True

    def test_check_specific_info_present_with_date_range(self, validator):
        """Return True when response contains date range."""
        assert validator.check_specific_info_present("3월 1일부터 3월 15일까지") is True
        assert validator.check_specific_info_present("2024.03.01 ~ 2024.03.15") is True
        assert validator.check_specific_info_present("2024-03-01 ~ 2024-03-15") is True

    def test_check_specific_info_present_without_specific_info(self, validator):
        """Return False when response lacks specific dates/periods."""
        assert validator.check_specific_info_present("학사일정을 확인해 주세요") is False
        assert validator.check_specific_info_present("담당 부서에 문의하세요") is False
        assert validator.check_specific_info_present("규정에 구체적 기한이 명시되어 있지 않습니다") is False

    def test_check_specific_info_present_empty_string(self, validator):
        """Handle empty string input."""
        assert validator.check_specific_info_present("") is False

    # ==================== check_alternative_guidance_present Tests ====================

    def test_check_alternative_guidance_with_calendar_reference(self, validator):
        """Return True when response mentions checking academic calendar."""
        # "학사일정을 확인" pattern
        assert validator.check_alternative_guidance_present("학사일정을 확인해 주시기 바랍니다") is True
        assert validator.check_alternative_guidance_present("학사일정을 확인하세요") is True
        assert validator.check_alternative_guidance_present("학사일정을 확인하시기 바랍니다") is True

    def test_check_alternative_guidance_with_department_contact(self, validator):
        """Return True when response mentions contacting department."""
        # "담당 부서에 문의" pattern
        assert validator.check_alternative_guidance_present("담당 부서에 문의하세요") is True
        assert validator.check_alternative_guidance_present("담당 부서에 문의하시기 바랍니다") is True

    def test_check_alternative_guidance_with_no_specific_deadline(self, validator):
        """Return True when response mentions no specific deadline in regulation."""
        # "규정에 구체적 기한이 명시되어 있지 않습니다" pattern
        assert (
            validator.check_alternative_guidance_present("규정에 구체적 기한이 명시되어 있지 않습니다")
            is True
        )

    def test_check_alternative_guidance_with_website_reference(self, validator):
        """Return True when response mentions checking website."""
        # "학교 홈페이지 참고" pattern
        assert validator.check_alternative_guidance_present("학교 홈페이지 참고하세요") is True
        assert validator.check_alternative_guidance_present("학교 홈페이지를 참고해 주시기 바랍니다") is True

    def test_check_alternative_guidance_without_guidance(self, validator):
        """Return False when response lacks alternative guidance."""
        assert validator.check_alternative_guidance_present("2024년 3월 1일입니다") is False
        assert validator.check_alternative_guidance_present("수강신청은 온라인으로 합니다") is False

    def test_check_alternative_guidance_empty_string(self, validator):
        """Handle empty string input."""
        assert validator.check_alternative_guidance_present("") is False

    # ==================== validate_period_response Tests ====================

    def test_validate_period_response_with_specific_info_only(self, validator):
        """Return complete result when response has specific info only."""
        query = "수강신청 기간이 언제인가요?"
        response = "수강신청 기간은 2024년 2월 20일부터 2월 27일까지입니다."

        result = validator.validate_period_response(query, response)

        assert result.is_complete is True
        assert result.has_specific_info is True
        assert result.has_alternative_guidance is False
        assert result.score == 0.85

    def test_validate_period_response_with_alternative_guidance_only(self, validator):
        """Return complete result when response has alternative guidance only."""
        query = "수강신청 기간이 언제인가요?"
        response = "규정에 구체적 기한이 명시되어 있지 않습니다. 학사일정을 확인해 주시기 바랍니다."

        result = validator.validate_period_response(query, response)

        assert result.is_complete is True
        assert result.has_specific_info is False
        assert result.has_alternative_guidance is True
        assert result.score == 0.7

    def test_validate_period_response_with_both(self, validator):
        """Return complete result with perfect score when both present."""
        query = "수강신청 기간이 언제인가요?"
        response = (
            "수강신청 기간은 2024년 2월 20일부터 2월 27일까지입니다. "
            "자세한 내용은 학사일정을 확인해 주시기 바랍니다."
        )

        result = validator.validate_period_response(query, response)

        assert result.is_complete is True
        assert result.has_specific_info is True
        assert result.has_alternative_guidance is True
        assert result.score == 1.0

    def test_validate_period_response_with_neither(self, validator):
        """Return incomplete result when neither present."""
        query = "수강신청 기간이 언제인가요?"
        response = "수강신청은 매 학기 시작 전에 진행됩니다."

        result = validator.validate_period_response(query, response)

        assert result.is_complete is False
        assert result.has_specific_info is False
        assert result.has_alternative_guidance is False
        assert result.score == 0.3
        assert "specific_info" in result.missing_elements
        assert "alternative_guidance" in result.missing_elements

    def test_validate_period_response_non_period_query(self, validator):
        """Return complete for non-period queries (default behavior)."""
        query = "교원 승진 규정에 대해 알려주세요"
        response = "교원 승진 규정은 승진 심사 위원회에서 진행됩니다."

        result = validator.validate_period_response(query, response)

        # Non-period queries should pass by default
        assert result.is_complete is True
        assert result.score >= 0.7


class TestCompletenessValidatorEdgeCases:
    """Tests for edge cases in completeness validation."""

    @pytest.fixture
    def validator(self) -> CompletenessValidator:
        """Create CompletenessValidator instance."""
        return CompletenessValidator()

    def test_empty_query(self, validator):
        """Handle empty query string."""
        result = validator.validate_period_response("", "2024년 3월 1일입니다")
        assert result.is_complete is True  # Empty query defaults to complete

    def test_empty_response(self, validator):
        """Handle empty response string."""
        result = validator.validate_period_response("수강신청 기간", "")
        assert result.is_complete is False
        assert result.score == 0.3

    def test_both_empty(self, validator):
        """Handle both empty strings."""
        result = validator.validate_period_response("", "")
        assert result.is_complete is True  # No period query, defaults to complete

    def test_whitespace_only_response(self, validator):
        """Handle whitespace-only response."""
        result = validator.validate_period_response("수강신청 기간", "   \n\t  ")
        assert result.is_complete is False

    def test_mixed_language_response(self, validator):
        """Handle response with mixed Korean and English."""
        response = "Period is 2024-03-01 to 2024-03-15. 기간은 3월 1일부터입니다."
        result = validator.validate_period_response("수강신청 기간", response)
        assert result.has_specific_info is True

    def test_long_response(self, validator):
        """Handle long response text."""
        long_response = (
            "이것은 매우 긴 응답입니다. " * 50
            + "2024년 3월 1일부터 3월 15일까지입니다. "
            + "이것은 매우 긴 응답입니다. " * 50
        )
        result = validator.validate_period_response("수강신청 기간", long_response)
        assert result.has_specific_info is True

    def test_multiple_dates_in_response(self, validator):
        """Handle response with multiple dates."""
        response = "신청 시작은 2월 20일, 마감은 2월 27일, 발표는 3월 1일입니다."
        result = validator.validate_period_response("수강신청 기간", response)
        assert result.has_specific_info is True


class TestCompletenessValidatorScoring:
    """Tests for completeness score calculation."""

    @pytest.fixture
    def validator(self) -> CompletenessValidator:
        """Create CompletenessValidator instance."""
        return CompletenessValidator()

    def test_score_1_0_with_both_elements(self, validator):
        """Score 1.0 when both specific info and alternative guidance present."""
        response = "2024년 3월 1일부터입니다. 학사일정을 확인해 주시기 바랍니다."
        result = validator.validate_period_response("수강신청 기간", response)
        assert result.score == 1.0

    def test_score_0_85_with_specific_info_only(self, validator):
        """Score 0.85 when only specific info present."""
        response = "2024년 3월 1일부터 3월 15일까지입니다."
        result = validator.validate_period_response("수강신청 기간", response)
        assert result.score == 0.85

    def test_score_0_7_with_alternative_guidance_only(self, validator):
        """Score 0.7 when only alternative guidance present."""
        response = "규정에 구체적 기한이 명시되어 있지 않습니다. 학사일정을 확인해 주시기 바랍니다."
        result = validator.validate_period_response("수강신청 기간", response)
        assert result.score == 0.7

    def test_score_0_3_with_neither_element(self, validator):
        """Score 0.3 when neither specific info nor alternative guidance present."""
        response = "수강신청은 매 학기 진행됩니다."
        result = validator.validate_period_response("수강신청 기간", response)
        assert result.score == 0.3


class TestCompletenessValidatorKoreanPatterns:
    """Tests for Korean-specific patterns in completeness validation."""

    @pytest.fixture
    def validator(self) -> CompletenessValidator:
        """Create CompletenessValidator instance."""
        return CompletenessValidator()

    def test_korean_date_formats(self, validator):
        """Recognize various Korean date formats."""
        date_formats = [
            "2024년 3월 1일",
            "3월 1일",
            "3월 1일부터",
            "1일까지",
            "2024.03.01",
            "2024-03-01",
        ]
        for date_text in date_formats:
            assert validator.check_specific_info_present(date_text) is True, (
                f"Failed for: {date_text}"
            )

    def test_korean_period_expressions(self, validator):
        """Recognize Korean period expressions."""
        period_expressions = [
            "2주 이내",
            "3개월간",
            "7일 동안",
            "10일 내에",
        ]
        for period_text in period_expressions:
            assert validator.check_specific_info_present(period_text) is True, (
                f"Failed for: {period_text}"
            )

    def test_korean_alternative_guidance_variations(self, validator):
        """Recognize various Korean alternative guidance expressions."""
        guidance_patterns = [
            "학사일정을 확인해 주시기 바랍니다",
            "학사일정을 확인하시기 바랍니다",
            "학사일정을 확인하세요",
            "담당 부서에 문의하시기 바랍니다",
            "담당 부서에 문의하세요",
            "규정에 구체적 기한이 명시되어 있지 않습니다",
            "학교 홈페이지를 참고하세요",
            "학교 홈페이지 참고하시기 바랍니다",
        ]
        for guidance in guidance_patterns:
            assert validator.check_alternative_guidance_present(guidance) is True, (
                f"Failed for: {guidance}"
            )

    def test_honorific_variations(self, validator):
        """Handle Korean honorific variations (-요, -시기 바랍니다, etc.)."""
        # All these should be recognized as alternative guidance
        variations = [
            "학사일정을 확인하세요",
            "학사일정을 확인해요",
            "학사일정을 확인하시기 바랍니다",
            "학사일정을 확인해 주세요",
        ]
        for variation in variations:
            result = validator.check_alternative_guidance_present(variation)
            assert result is True, f"Failed for: {variation}"
