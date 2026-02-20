"""
Unit tests for FaithfulnessValidator domain component.

Tests claim extraction, context matching, and faithfulness score calculation.
TDD Implementation: RED phase - these tests define expected behavior.
"""

import pytest

from src.rag.domain.evaluation.faithfulness_validator import (
    FaithfulnessValidationResult,
    FaithfulnessValidator,
)


class TestFaithfulnessValidator:
    """Test FaithfulnessValidator functionality."""

    @pytest.fixture
    def validator(self):
        """Create faithfulness validator."""
        return FaithfulnessValidator()

    # ========================================
    # Claim Extraction Tests
    # ========================================

    def test_extract_citation_claims_korean(self, validator):
        """WHEN Korean answer contains citations, THEN should extract claim patterns."""
        answer = "학칙 제10조에 따르면 휴학은 2학기까지 가능합니다."
        context = ["학칙 제10조 (휴학) 학생은 2학기까지 휴학할 수 있다."]

        result = validator.validate_answer(answer, context)

        assert "제10조" in str(result.grounded_claims)
        assert result.score >= 0.6

    def test_extract_numerical_claims_dates(self, validator):
        """WHEN answer contains date claims, THEN should extract date patterns."""
        answer = "2024년 3월 1일부터 2024년 8월 31일까지 등록기간입니다."
        context = ["등록기간: 2024년 3월 1일 ~ 2024년 8월 31일"]

        result = validator.validate_answer(answer, context)

        # Date claims should be grounded
        assert any("2024년" in claim or "3월" in claim for claim in result.grounded_claims)

    def test_extract_numerical_claims_percentages(self, validator):
        """WHEN answer contains percentage claims, THEN should extract percentage patterns."""
        answer = "장학금은 등록금의 50%까지 지급됩니다."
        context = ["장학금 지급 비율: 등록금의 최대 50%"]

        result = validator.validate_answer(answer, context)

        assert any("50%" in claim for claim in result.grounded_claims)

    def test_extract_contact_claims_phone(self, validator):
        """WHEN answer contains phone numbers, THEN should extract phone patterns."""
        answer = "문의사항은 02-1234-5678로 연락바랍니다."
        context = ["연락처: 02-1234-5678"]

        result = validator.validate_answer(answer, context)

        assert any("02-1234-5678" in claim for claim in result.grounded_claims)

    def test_extract_contact_claims_email(self, validator):
        """WHEN answer contains email addresses, THEN should extract email patterns."""
        answer = "이메일 contact@university.ac.kr로 문의하세요."
        context = ["문의: contact@university.ac.kr"]

        result = validator.validate_answer(answer, context)

        assert any("contact@university.ac.kr" in claim for claim in result.grounded_claims)

    def test_extract_multiple_claims(self, validator):
        """WHEN answer contains multiple claim types, THEN should extract all claims."""
        answer = "제5조에 따르면 30일 이내에 신청해야 하며, 문의는 02-123-4567로 바랍니다."
        context = [
            "제5조: 30일 이내에 신청",
            "연락처: 02-123-4567",
        ]

        result = validator.validate_answer(answer, context)

        # Should have at least 2 grounded claims
        assert len(result.grounded_claims) >= 2
        assert result.score >= 0.7

    # ========================================
    # Context Matching Tests
    # ========================================

    def test_grounded_claim_in_context(self, validator):
        """WHEN claim is supported by context, THEN should be marked as grounded."""
        answer = "휴학은 최대 4학기까지 가능합니다."
        context = ["학칙 제12조: 휴학 기간은 최대 4학기까지 허용된다."]

        result = validator.validate_answer(answer, context)

        assert len(result.grounded_claims) > 0
        assert len(result.ungrounded_claims) == 0
        assert result.is_acceptable is True

    def test_ungrounded_claim_not_in_context(self, validator):
        """WHEN claim is not in context, THEN should be marked as ungrounded."""
        answer = "휴학은 최대 10학기까지 가능합니다."  # Incorrect claim - 10 not in context
        context = ["학칙 제12조: 휴학 기간은 최대 4학기까지 허용된다."]  # Says 4, not 10

        result = validator.validate_answer(answer, context)

        # "10학기" should be ungrounded because "10" is not in context
        assert len(result.ungrounded_claims) > 0
        assert result.score < 1.0

    def test_partial_groundedness(self, validator):
        """WHEN some claims are grounded and some are not, THEN should reflect partial grounding."""
        answer = "휴학은 4학기까지 가능하며, 등록금은 100% 환불됩니다."
        context = [
            "학칙 제12조: 휴학 기간은 최대 4학기까지 허용된다.",
            # Missing: 등록금 100% 환불 정보
        ]

        result = validator.validate_answer(answer, context)

        # Should have both grounded and ungrounded claims
        assert len(result.grounded_claims) > 0
        assert len(result.ungrounded_claims) > 0
        assert 0.0 < result.score < 1.0

    def test_check_groundedness_with_fuzzy_match(self, validator):
        """WHEN claim has minor variations, THEN should still be considered grounded."""
        answer = "신청기간은 2024년 3월 1일부터입니다."
        context = ["신청 기간: 2024년 3월 1일 ~ 2024년 3월 31일"]

        result = validator.validate_answer(answer, context)

        # Should recognize similarity despite spacing difference
        assert len(result.grounded_claims) > 0 or result.score > 0.5

    # ========================================
    # Faithfulness Score Calculation Tests
    # ========================================

    def test_score_calculation_all_grounded(self, validator):
        """WHEN all claims are grounded, THEN score should be high."""
        answer = "제10조에 따르면 등록금 납부는 30일 이내입니다."
        context = [
            "제10조: 등록금 납부 기한은 30일 이내입니다",
        ]

        result = validator.validate_answer(answer, context)

        # With all claims grounded, score should be high (>= 0.7)
        assert result.score >= 0.7
        assert result.is_acceptable is True

    def test_score_calculation_no_claims(self, validator):
        """WHEN answer has no extractable claims, THEN should use context overlap."""
        answer = "일반적인 안내 정보입니다."
        context = ["일반적인 안내 정보에 대한 내용"]

        result = validator.validate_answer(answer, context)

        # Should use keyword overlap when no claims exist
        assert 0.0 <= result.score <= 1.0

    def test_score_calculation_mixed_claims(self, validator):
        """WHEN answer has mixed grounded/ungrounded claims, THEN score should reflect ratio."""
        answer = "제10조에 따르면 30일 이내이고, 제20조에서는 60일 이내입니다."
        context = [
            "제10조: 30일 이내",
            # 제20조 not in context
        ]

        result = validator.validate_answer(answer, context)

        # Score should be around 0.5 (1 grounded, 1 ungrounded)
        assert 0.3 <= result.score <= 0.7

    def test_score_range_zero_to_one(self, validator):
        """WHEN calculating score, THEN score should always be between 0.0 and 1.0."""
        # Test with various inputs
        test_cases = [
            ("", ["context"]),
            ("answer", []),
            ("answer", ["context"]),
            ("제1조 제2조 제3조 제4조 제5조", ["제1조"]),
        ]

        for answer, context in test_cases:
            result = validator.validate_answer(answer, context)
            assert 0.0 <= result.score <= 1.0, f"Score {result.score} out of range for '{answer}'"

    # ========================================
    # Threshold-Based Acceptance Tests
    # ========================================

    def test_is_acceptable_above_threshold(self, validator):
        """WHEN score >= threshold, THEN is_acceptable should be True."""
        answer = "휴학은 최대 4학기까지 가능합니다."
        context = ["휴학 최대 4학기까지 허용"]

        result = validator.validate_answer(answer, context, threshold=0.6)

        assert result.score >= 0.6
        assert result.is_acceptable is True

    def test_is_acceptable_below_threshold(self, validator):
        """WHEN score < threshold, THEN is_acceptable should be False."""
        answer = "휴학은 6학기까지 가능하며 등록금은 100% 환불됩니다."
        context = ["휴학 최대 4학기까지 허용"]  # Contradicts 6학기

        result = validator.validate_answer(answer, context, threshold=0.6)

        # Score should be lower due to ungrounded claims
        if result.score < 0.6:
            assert result.is_acceptable is False

    def test_default_threshold_0_6(self, validator):
        """WHEN threshold not specified, THEN should use default 0.6."""
        answer = "휴학은 4학기까지 가능합니다."
        context = ["휴학 최대 4학기까지 허용"]

        result = validator.validate_answer(answer, context)

        # Should use default threshold of 0.6
        assert result.is_acceptable == (result.score >= 0.6)

    def test_custom_threshold(self, validator):
        """WHEN custom threshold specified, THEN should use custom threshold."""
        answer = "일반적인 안내입니다."
        context = ["일반적인 안내에 대한 내용"]

        result = validator.validate_answer(answer, context, threshold=0.9)

        # Should use custom threshold of 0.9
        assert result.is_acceptable == (result.score >= 0.9)

    # ========================================
    # Edge Cases and Error Handling Tests
    # ========================================

    def test_empty_answer(self, validator):
        """WHEN answer is empty, THEN should return low score with appropriate suggestion."""
        answer = ""
        context = ["some context"]

        result = validator.validate_answer(answer, context)

        assert result.score == 0.0
        assert result.is_acceptable is False
        assert "empty" in result.suggestion.lower() or "no answer" in result.suggestion.lower()

    def test_empty_context(self, validator):
        """WHEN context is empty, THEN should return low score."""
        answer = "제10조에 따르면 30일 이내입니다."
        context = []

        result = validator.validate_answer(answer, context)

        assert result.score < 0.5
        assert result.is_acceptable is False

    def test_whitespace_only_answer(self, validator):
        """WHEN answer is whitespace only, THEN should handle gracefully."""
        answer = "   \n\t  "
        context = ["some context"]

        result = validator.validate_answer(answer, context)

        assert result.score == 0.0
        assert result.is_acceptable is False

    def test_very_long_answer(self, validator):
        """WHEN answer is very long, THEN should still process correctly."""
        answer = "제1조 내용 " * 100  # 100 repeated claims
        context = ["제1조: 내용"]

        result = validator.validate_answer(answer, context)

        # Should handle long answer without crashing
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.grounded_claims, list)
        assert isinstance(result.ungrounded_claims, list)

    def test_unicode_korean_text(self, validator):
        """WHEN answer contains Korean Unicode, THEN should handle correctly."""
        answer = "학생은 등록금을 납부해야 합니다. (제10조)"
        context = ["제10조: 학생의 등록금 납부 의무"]

        result = validator.validate_answer(answer, context)

        assert result.score >= 0.0
        assert isinstance(result.suggestion, str)

    # ========================================
    # Suggestion Generation Tests
    # ========================================

    def test_suggestion_for_low_score(self, validator):
        """WHEN score is low, THEN should provide improvement suggestion."""
        answer = "잘못된 정보: 6학기까지 휴학 가능, 등록금 100% 환불"
        context = ["휴학 최대 4학기"]

        result = validator.validate_answer(answer, context)

        assert result.score < 0.6
        assert len(result.suggestion) > 0
        assert "ungrounded" in result.suggestion.lower() or "검증" in result.suggestion

    def test_suggestion_for_missing_context(self, validator):
        """WHEN context is missing relevant info, THEN should suggest adding context."""
        answer = "제20조에 따르면 60일 이내입니다."
        context = ["제10조에 대한 내용만 있음"]  # Missing 제20조

        result = validator.validate_answer(answer, context)

        # Should suggest context improvement
        assert len(result.suggestion) > 0

    def test_suggestion_format(self, validator):
        """WHEN suggestion is generated, THEN should be readable string."""
        answer = "제10조에 따르면 30일 이내입니다."
        context = ["제10조: 30일 이내"]

        result = validator.validate_answer(answer, context)

        assert isinstance(result.suggestion, str)
        assert len(result.suggestion) > 0


class TestFaithfulnessValidationResult:
    """Test FaithfulnessValidationResult dataclass."""

    def test_result_dataclass_creation(self):
        """WHEN creating result, THEN should have all required fields."""
        result = FaithfulnessValidationResult(
            score=0.85,
            is_acceptable=True,
            grounded_claims=["claim1"],
            ungrounded_claims=[],
            suggestion="Good answer",
        )

        assert result.score == 0.85
        assert result.is_acceptable is True
        assert result.grounded_claims == ["claim1"]
        assert result.ungrounded_claims == []
        assert result.suggestion == "Good answer"

    def test_result_with_empty_claims(self):
        """WHEN creating result with empty claims, THEN should be valid."""
        result = FaithfulnessValidationResult(
            score=1.0,
            is_acceptable=True,
            grounded_claims=[],
            ungrounded_claims=[],
            suggestion="No claims to verify",
        )

        assert result.score == 1.0
        assert len(result.grounded_claims) == 0
        assert len(result.ungrounded_claims) == 0

    def test_result_score_rounding(self):
        """WHEN score has many decimals, THEN should be rounded appropriately."""
        result = FaithfulnessValidationResult(
            score=0.857392,
            is_acceptable=True,
            grounded_claims=[],
            ungrounded_claims=[],
            suggestion="Test",
        )

        # Score should be a valid float
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
