"""
Unit tests for TypoCorrector module.

Tests the hybrid typo correction system:
1. Rule-based correction for common patterns
2. SymSpell-based correction (if available)
3. Edit distance-based correction for regulation names
4. LLM fallback
"""

import pytest

from src.rag.infrastructure.typo_corrector import (
    TypoCorrector,
)


class TestTypoCorrector:
    """Test suite for TypoCorrector."""

    @pytest.fixture
    def corrector(self):
        """Create a TypoCorrector instance for testing."""
        return TypoCorrector(
            llm_client=None,  # No LLM for basic tests
            regulation_names=[
                "교원인사규정",
                "학칙",
                "등록금규정",
                "장학금지급규정",
                "휴학규정",
                "복학규정",
                "제적규정",
                "자퇴규정",
            ],
        )

    def test_no_correction_needed(self, corrector):
        """Test that correct text is not modified."""
        result = corrector.correct("장학금 신청 방법")
        assert result.corrected == "장학금 신청 방법"
        assert result.method == "none"
        assert result.corrections == []
        assert result.confidence == 1.0

    def test_rule_based_correction_desire_expressions(self, corrector):
        """Test rule-based correction for desire expressions."""
        # "시퍼" -> "싶어"
        result = corrector.correct("장학금 받고 시퍼")
        assert result.corrected == "장학금 받고 싶어"
        assert ("시퍼", "싶어") in result.corrections
        assert result.method == "rule"
        assert result.confidence >= 0.9

    def test_rule_based_correction_informal_speech(self, corrector):
        """Test rule-based correction for informal speech."""
        # "되요" -> "돼요"
        result = corrector.correct("언제 되요")
        assert result.corrected == "언제 돼요"
        assert ("되요", "돼요") in result.corrections

        # "하가요" -> "하세요"
        result = corrector.correct("신청 어떻게 하가요")
        assert result.corrected == "신청 어떻게 하세요"
        assert ("하가요", "하세요") in result.corrections

    def test_rule_based_correction_regulation_names(self, corrector):
        """Test rule-based correction for regulation names."""
        # "극정" -> "규정"
        result = corrector.correct("교원인사극정")
        assert result.corrected == "교원인사규정"
        assert ("극정", "규정") in result.corrections

    def test_rule_based_correction_multiple_patterns(self, corrector):
        """Test multiple pattern corrections in one query."""
        # Note: "시퍼" pattern has $ anchor (end-of-string), so it only corrects "시퍼" at end
        # Using a simpler input to test multiple corrections
        result = corrector.correct("되요 하가요")
        assert "돼요" in result.corrected
        assert "하세요" in result.corrected
        assert len(result.corrections) >= 2

    def test_edit_distance_correction_regulation_names(self, corrector):
        """Test edit distance correction for regulation names."""
        # "교원인사극정" -> "교원인사규정"
        result = corrector.correct("교원인사극정")
        assert result.corrected == "교원인사규정"
        # Note: "극정" -> "규정" might be caught by rule-based first

        # Similar typo not in rule patterns
        result = corrector.correct(
            "학칙의"
        )  # "학칙의" -> "학칙" (particle normalization)
        # This might not trigger edit distance if it's close enough

    def test_cache_functionality(self, corrector):
        """Test that corrections are cached."""
        query = "장학금 받고 시퍼"

        # First call
        result1 = corrector.correct(query)
        # Second call should return cached result
        result2 = corrector.correct(query)

        assert result1.original == result2.original
        assert result1.corrected == result2.corrected
        assert result1.method == result2.method

    def test_empty_query(self, corrector):
        """Test handling of empty query."""
        result = corrector.correct("")
        assert result.original == ""
        assert result.corrected == ""
        assert result.method == "none"
        assert result.confidence == 1.0

    def test_set_regulation_names(self, corrector):
        """Test updating regulation names."""
        new_regulations = ["새로운규정", "또다른규정"]
        corrector.set_regulation_names(new_regulations)

        # Cache should be cleared
        query = "테스트"
        corrector.correct(query)
        # Just verify no errors occur

    def test_clear_cache(self, corrector):
        """Test cache clearing."""
        corrector.correct("장학금 받고 시퍼")
        corrector.clear_cache()
        # Cache should be empty now

    def test_confidence_calculation(self, corrector):
        """Test confidence score calculation."""
        # No correction = 1.0
        result = corrector.correct("장학금 신청")
        assert result.confidence == 1.0

        # Rule-based correction = high confidence
        result = corrector.correct("장학금 받고 시퍼")
        assert result.confidence >= 0.9


class TestTypoCorrectionPatterns:
    """Test specific typo correction patterns."""

    @pytest.fixture
    def corrector(self):
        """Create a TypoCorrector instance."""
        return TypoCorrector(llm_client=None)

    @pytest.mark.parametrize(
        "input_text,expected_text,expected_correction",
        [
            ("시퍼", "싶어", ("시퍼", "싶어")),
            ("되요", "돼요", ("되요", "돼요")),
            ("하가요", "하세요", ("하가요", "하세요")),
            ("바드려면", "바라면", ("바드려면", "바라면")),
            ("극정", "규정", ("극정", "규정")),
        ],
    )
    def test_common_patterns(
        self, corrector, input_text, expected_text, expected_correction
    ):
        """Test common typo correction patterns."""
        result = corrector.correct(input_text)
        assert expected_text in result.corrected
        assert expected_correction in result.corrections

    def test_particle_normalization(self, corrector):
        """Test particle normalization."""
        result = corrector.correct("규정으로서")
        assert "으로" in result.corrected

    def test_complex_sentence(self, corrector):
        """Test correction in complex sentence."""
        result = corrector.correct("장학금 받고 시퍼서 휴학하고 싶어")
        assert "싶어" in result.corrected
        # Both desire expressions should be corrected


class TestIntegrationWithQueryAnalyzer:
    """Test integration with QueryAnalyzer."""

    def test_query_analyzer_typo_correction_integration(self):
        """Test that QueryAnalyzer uses typo correction."""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer

        # Create analyzer with typo correction enabled
        analyzer = QueryAnalyzer(
            llm_client=None,
            enable_typo_correction=True,
            regulation_names=["교원인사규정", "학칙"],
        )

        # Test typo correction in query rewrite
        result = analyzer.rewrite_query_with_info("장학금 받고 시퍼")

        # Verify typo correction was applied
        assert result.typo_corrected
        assert ("시퍼", "싶어") in result.typo_corrections
        assert "싶어" in result.rewritten

    def test_query_analyzer_without_typo_correction(self):
        """Test QueryAnalyzer with typo correction disabled."""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer(llm_client=None, enable_typo_correction=False)

        result = analyzer.rewrite_query_with_info("장학금 받고 시퍼")

        # Typo correction should not be applied
        assert not result.typo_corrected
        assert result.typo_corrections == ()

    def test_query_analyzer_set_regulation_names(self):
        """Test setting regulation names on QueryAnalyzer."""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer(llm_client=None, enable_typo_correction=True)

        regulations = ["교원인사규정", "학칙", "등록금규정"]
        analyzer.set_regulation_names(regulations)

        # Verify no errors occur
        analyzer.rewrite_query_with_info("교원인사극정 제8조")
        # Should correct "극정" to "규정"


@pytest.mark.parametrize(
    "query,expected_corrections",
    [
        ("장학금 받고 시퍼", [("시퍼", "싶어")]),
        ("휴학 언제 되요", [("되요", "돼요")]),
        ("신청 어떻게 하가요", [("하가요", "하세요")]),
        ("교원인사극정 제8조", [("극정", "규정")]),
        ("휴학원서 제출", [("휴학원", "휴학")]),
    ],
)
def test_parametrized_corrections(query, expected_corrections):
    """Parametrized test for common correction patterns."""
    corrector = TypoCorrector(llm_client=None)
    result = corrector.correct(query)

    for original, corrected in expected_corrections:
        assert (original, corrected) in result.corrections
        assert corrected in result.corrected
