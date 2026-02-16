"""
Unit tests for Faithfulness calculation (SPEC-RAG-QUALITY-004).

Tests for faithfulness score calculation and blocking behavior.
Validates that answers with low faithfulness (< 0.3) are blocked
and safe responses are generated.
"""

import pytest

from src.rag.application.hallucination_filter import (
    FAITHFULNESS_BLOCK_THRESHOLD,
    FaithfulnessResult,
    FilterMode,
    HallucinationFilter,
)


class TestFaithfulnessThreshold:
    """Test faithfulness threshold constant."""

    def test_threshold_is_0_3(self):
        """THEN faithfulness block threshold should be 0.3."""
        assert FAITHFULNESS_BLOCK_THRESHOLD == 0.3


class TestFaithfulnessResult:
    """Test FaithfulnessResult dataclass."""

    def test_faithfulness_result_creation(self):
        """THEN FaithfulnessResult should store all fields."""
        result = FaithfulnessResult(
            score=0.5,
            should_block=False,
            reason="Faithfulness acceptable",
            verified_claims=5,
            total_claims=5,
            context_overlap_ratio=0.8,
        )
        assert result.score == 0.5
        assert result.should_block is False
        assert result.reason == "Faithfulness acceptable"
        assert result.verified_claims == 5
        assert result.total_claims == 5
        assert result.context_overlap_ratio == 0.8

    def test_faithfulness_result_blocking(self):
        """THEN FaithfulnessResult should indicate blocking when score < 0.3."""
        result = FaithfulnessResult(
            score=0.2,
            should_block=True,
            reason="Low faithfulness",
            verified_claims=1,
            total_claims=5,
            context_overlap_ratio=0.1,
        )
        assert result.score == 0.2
        assert result.should_block is True


class TestCalculateFaithfulness:
    """Test faithfulness calculation method."""

    def test_empty_response_returns_zero_faithfulness(self):
        """
        WHEN response is empty
        THEN faithfulness should be 0.0 and should_block should be True
        """
        filter_service = HallucinationFilter()
        result = filter_service.calculate_faithfulness("", ["context"])

        assert result.score == 0.0
        assert result.should_block is True
        assert "Empty response" in result.reason

    def test_empty_context_returns_zero_faithfulness(self):
        """
        WHEN context is empty
        THEN faithfulness should be 0.0 and should_block should be True
        """
        filter_service = HallucinationFilter()
        result = filter_service.calculate_faithfulness("response", [])

        assert result.score == 0.0
        assert result.should_block is True
        assert "No context available" in result.reason

    def test_high_faithfulness_with_verified_claims(self):
        """
        WHEN all claims in response are verified by context
        THEN faithfulness should be high (> 0.5) and should_block should be False
        """
        filter_service = HallucinationFilter()
        response = "학적팀(02-1234-5678)에 문의하면 학칙 제10조에 따라 처리됩니다."
        context = [
            "담당 부서: 학적팀",
            "전화: 02-1234-5678",
            "학칙 제10조: 등록에 관한 규정",
        ]

        result = filter_service.calculate_faithfulness(response, context)

        assert result.score > 0.5
        assert result.should_block is False

    def test_low_faithfulness_with_unverified_claims(self):
        """
        WHEN response contains claims not in context
        THEN faithfulness should be low
        """
        filter_service = HallucinationFilter()
        response = "학술연구지원팀(02-9999-9999)에 문의하면 규정 제99조에 따라 처리됩니다."
        context = [
            "다른 부서 정보만 있습니다.",
        ]

        result = filter_service.calculate_faithfulness(response, context)

        # Low faithfulness because claims are not verified
        assert result.score < 0.5
        assert result.total_claims > 0
        assert result.verified_claims < result.total_claims

    def test_faithfulness_blocks_below_threshold(self):
        """
        WHEN faithfulness score < 0.3
        THEN should_block should be True
        """
        filter_service = HallucinationFilter()
        # Response with multiple unverified claims
        response = "문의: 02-9999-9999 (학술연구지원팀) 규정 제88조 제99조"
        context = ["완전히 다른 내용입니다."]

        result = filter_service.calculate_faithfulness(response, context)

        assert result.score < FAITHFULNESS_BLOCK_THRESHOLD
        assert result.should_block is True

    def test_faithfulness_with_context_overlap(self):
        """
        WHEN response shares keywords with context
        THEN context_overlap_ratio should be higher
        """
        filter_service = HallucinationFilter()
        response = "휴학은 학기 개시 1개월 전까지 신청해야 합니다."
        context = ["학칙 제40조: 휴학은 학기 개시 1개월 전까지 신청해야 한다."]

        result = filter_service.calculate_faithfulness(response, context)

        # High overlap because same keywords
        assert result.context_overlap_ratio > 0.3

    def test_faithfulness_without_claims(self):
        """
        WHEN response has no verifiable claims (no phone, dept, citation)
        THEN faithfulness should rely on context overlap
        """
        filter_service = HallucinationFilter()
        response = "이 규정은 대학의 학사 운영에 관한 사항을 규정합니다."
        context = ["학칙은 대학의 학사 운영에 관한 기본적인 사항을 규정한다."]

        result = filter_service.calculate_faithfulness(response, context)

        # No claims = neutral claim score, rely on overlap
        assert result.total_claims == 0
        assert result.verified_claims == 0

    def test_faithfulness_mixed_claims(self):
        """
        WHEN response has mix of verified and unverified claims
        THEN faithfulness should reflect partial verification
        """
        filter_service = HallucinationFilter()
        response = "학적팀(02-1234-5678)과 장학팀(02-9999-9999)에 문의하세요."
        context = [
            "담당 부서: 학적팀",
            "전화: 02-1234-5678",
        ]

        result = filter_service.calculate_faithfulness(response, context)

        # Partial verification: 학적팀 verified, 장학팀 not verified
        assert result.total_claims >= 2
        assert result.verified_claims >= 1
        assert result.verified_claims < result.total_claims


class TestExtractKeywords:
    """Test keyword extraction for faithfulness calculation."""

    def test_extract_korean_keywords(self):
        """
        WHEN text contains Korean words
        THEN should extract meaningful keywords
        """
        filter_service = HallucinationFilter()
        text = "학칙 제10조에 따라 휴학을 신청할 수 있습니다."
        keywords = filter_service._extract_keywords(text)

        assert "학칙" in keywords
        assert "제10조" in keywords

    def test_extract_article_numbers(self):
        """
        WHEN text contains article numbers
        THEN should extract them as keywords
        """
        filter_service = HallucinationFilter()
        text = "제15조 제20조 제3조"
        keywords = filter_service._extract_keywords(text)

        assert "제15조" in keywords
        assert "제20조" in keywords
        assert "제3조" in keywords

    def test_extract_period_patterns(self):
        """
        WHEN text contains period patterns (days, months)
        THEN should extract them as keywords
        """
        filter_service = HallucinationFilter()
        text = "30일 이내에 6개월 동안 50% 감면"
        keywords = filter_service._extract_keywords(text)

        assert "30일" in keywords
        assert "6개월" in keywords
        assert "50%" in keywords

    def test_filter_stop_words(self):
        """
        WHEN text contains Korean stop words
        THEN should filter them out
        """
        filter_service = HallucinationFilter()
        text = "합니다 바랍니다 있습니다"
        keywords = filter_service._extract_keywords(text)

        # Stop words should be filtered
        assert "합니다" not in keywords
        assert "바랍니다" not in keywords


class TestFaithfulnessIntegration:
    """Test faithfulness integration with filter_response."""

    def test_filter_response_does_not_modify_behavior(self):
        """
        WHEN filter_response is called
        THEN existing behavior should remain unchanged
        """
        filter_service = HallucinationFilter(mode=FilterMode.SANITIZE)
        response = "문의: 02-9999-9999 (학술연구지원팀)"
        context = ["다른 정보만 있습니다."]

        result = filter_service.filter_response(response, context)

        # Existing sanitization behavior should work
        assert result.is_modified is True
        assert "02-9999-9999" not in result.sanitized_response

    def test_calculate_faithfulness_separate_from_filter(self):
        """
        WHEN calculate_faithfulness is called
        THEN it should return separate result from filter_response
        """
        filter_service = HallucinationFilter()
        response = "학적팀에 문의하세요."
        context = ["담당 부서: 학적팀"]

        filter_result = filter_service.filter_response(response, context)
        faithfulness_result = filter_service.calculate_faithfulness(response, context)

        # These are separate operations
        assert hasattr(filter_result, "sanitized_response")
        assert hasattr(faithfulness_result, "score")


class TestEdgeCases:
    """Test edge cases for faithfulness calculation."""

    def test_whitespace_only_response(self):
        """
        WHEN response contains only whitespace
        THEN faithfulness should be 0.0
        """
        filter_service = HallucinationFilter()
        result = filter_service.calculate_faithfulness("   \n\t  ", ["context"])

        assert result.score == 0.0
        assert result.should_block is True

    def test_context_with_only_whitespace(self):
        """
        WHEN context contains only whitespace
        THEN faithfulness should be 0.0
        """
        filter_service = HallucinationFilter()
        result = filter_service.calculate_faithfulness("response", ["   ", "\n", "\t"])

        assert result.score == 0.0
        assert result.should_block is True

    def test_very_long_response(self):
        """
        WHEN response is very long
        THEN faithfulness should still calculate correctly
        """
        filter_service = HallucinationFilter()
        # Long response with verified claims
        response = "학적팀(02-1234-5678)에 문의하면 " * 100
        context = ["담당 부서: 학적팀", "전화: 02-1234-5678"]

        result = filter_service.calculate_faithfulness(response, context)

        # Should handle long response
        assert result.score >= 0.0
        assert result.score <= 1.0

    def test_unicode_handling(self):
        """
        WHEN response contains unicode characters
        THEN faithfulness should handle correctly
        """
        filter_service = HallucinationFilter()
        response = "학적팀📧test@example.com에 문의하세요."
        context = ["이메일: test@example.com", "부서: 학적팀"]

        result = filter_service.calculate_faithfulness(response, context)

        # Should handle unicode
        assert result.score >= 0.0
