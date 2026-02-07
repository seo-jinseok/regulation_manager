"""
Unit tests for Adaptive Top-K Selector.

Implements SPEC-RAG-SEARCH-001 TAG-003: Adaptive Top-K Selection.

Tests cover:
- Query complexity classification (4 types)
- Top-K selection for each complexity level
- Edge cases (empty queries, very long queries)
- Latency guardrails
- Configuration validation
"""

import pytest

from src.rag.infrastructure.adaptive_top_k import (
    AdaptiveTopKSelector,
    QueryComplexity,
    TopKConfig,
)


class TestTopKConfig:
    """Test TopKConfig validation and defaults."""

    def test_default_config_values(self):
        """Test that default configuration values are correct."""
        config = TopKConfig()
        assert config.simple == 5
        assert config.medium == 10
        assert config.complex == 15
        assert config.multi_part == 20
        assert config.max_limit == 25
        assert config.latency_threshold_ms == 500

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = TopKConfig(
            simple=3,
            medium=8,
            complex=12,
            multi_part=18,
            max_limit=20,
            latency_threshold_ms=300,
        )
        assert config.simple == 3
        assert config.medium == 8
        assert config.complex == 12
        assert config.multi_part == 18
        assert config.max_limit == 20
        assert config.latency_threshold_ms == 300

    def test_config_validation_simple_too_small(self):
        """Test that simple Top-K must be >= 1."""
        with pytest.raises(ValueError, match="simple Top-K must be >= 1"):
            TopKConfig(simple=0)

    def test_config_validation_simple_greater_than_medium(self):
        """Test that simple Top-K must be <= medium."""
        with pytest.raises(ValueError, match="simple Top-K must be <= medium"):
            TopKConfig(simple=15, medium=10)

    def test_config_validation_medium_greater_than_complex(self):
        """Test that medium Top-K must be <= complex."""
        with pytest.raises(ValueError, match="medium Top-K must be <= complex"):
            TopKConfig(simple=5, medium=15, complex=10)

    def test_config_validation_complex_greater_than_multi_part(self):
        """Test that complex Top-K must be <= multi_part."""
        with pytest.raises(ValueError, match="complex Top-K must be <= multi_part"):
            TopKConfig(simple=5, medium=10, complex=20, multi_part=15)

    def test_config_validation_multi_part_greater_than_max_limit(self):
        """Test that multi_part Top-K must be <= max_limit."""
        with pytest.raises(ValueError, match="multi_part Top-K must be <= max_limit"):
            TopKConfig(simple=5, medium=10, complex=15, multi_part=30, max_limit=25)


class TestQueryComplexityClassification:
    """Test query complexity classification (REQ-AT-001 ~ REQ-AT-005)."""

    def test_simple_classification_single_word(self):
        """Test SIMPLE classification for single word."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("휴학")
        assert result.complexity == QueryComplexity.SIMPLE

    def test_simple_classification_regulation_name(self):
        """Test SIMPLE classification for regulation name (REQ-AT-002)."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("교원인사규정")
        assert result.complexity == QueryComplexity.SIMPLE

    def test_simple_classification_short_phrase(self):
        """Test SIMPLE classification for short phrase."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("장학금")
        assert result.complexity == QueryComplexity.SIMPLE

    def test_medium_classification_natural_question(self):
        """Test MEDIUM classification for natural question (REQ-AT-003)."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("장학금을 어떻게 신청하나요?")
        assert result.complexity == QueryComplexity.MEDIUM

    def test_medium_classification_how_to_question(self):
        """Test MEDIUM classification for 'how-to' questions."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("휴학 신청 방법")
        # This could be MEDIUM or COMPLEX depending on implementation
        assert result.complexity in [QueryComplexity.MEDIUM, QueryComplexity.COMPLEX]

    def test_complex_classification_procedure_query(self):
        """Test COMPLEX classification for procedure query (REQ-AT-004)."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("장학금 신청 절차와 구비서류")
        assert result.complexity == QueryComplexity.COMPLEX

    def test_complex_classification_requirement_query(self):
        """Test COMPLEX classification for requirement query."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("연구년 자격 요건과 충족 조건")
        assert result.complexity == QueryComplexity.COMPLEX

    def test_complex_classification_benefit_query(self):
        """Test COMPLEX classification for benefit query."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("조교 혜택과 지급 급여")
        assert result.complexity == QueryComplexity.COMPLEX

    def test_multi_part_classification_with_and(self):
        """Test MULTI_PART classification with 'and' conjunction (REQ-AT-005)."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("장학금 신청 방법 그리고 자격 요건")
        assert result.complexity == QueryComplexity.MULTI_PART

    def test_multi_part_classification_with_or(self):
        """Test MULTI_PART classification with 'or' conjunction."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("휴학 절차 또는 복학 방법")
        assert result.complexity == QueryComplexity.MULTI_PART

    def test_multi_part_classification_with_comma(self):
        """Test MULTI_PART classification with comma separator."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("등록금, 장학금, 학점")
        assert result.complexity == QueryComplexity.MULTI_PART


class TestTopKSelection:
    """Test Top-K selection for each complexity level."""

    def test_select_top_k_simple_query(self):
        """Test Top-5 for simple queries (REQ-AT-002)."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("교원인사규정")
        assert top_k == 5

    def test_select_top_k_medium_query(self):
        """Test Top-10 for medium queries (REQ-AT-003)."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("장학금을 어떻게 신청하나요?")
        assert top_k == 10

    def test_select_top_k_complex_query(self):
        """Test Top-15 for complex queries (REQ-AT-004)."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("장학금 신청 절차와 구비서류")
        assert top_k == 15

    def test_select_top_k_multi_part_query(self):
        """Test Top-20 for multi-part queries (REQ-AT-005)."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("장학금 신청 방법 그리고 자격 요건")
        assert top_k == 20

    def test_select_top_k_empty_query(self):
        """Test Top-10 for empty query (fallback)."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("")
        assert top_k == 10

    def test_select_top_k_whitespace_only(self):
        """Test Top-10 for whitespace-only query (fallback)."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("   ")
        assert top_k == 10


class TestComplexityAnalysisResult:
    """Test ComplexityAnalysisResult structure and properties."""

    def test_analysis_result_structure(self):
        """Test that analysis result contains all required fields."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("장학금 신청")

        assert hasattr(result, "complexity")
        assert hasattr(result, "score")
        assert hasattr(result, "top_k")
        assert hasattr(result, "factors")
        assert hasattr(result, "processing_time_ms")

    def test_analysis_result_factors(self):
        """Test that analysis result contains all complexity factors."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("장학금 신청 방법")

        expected_factors = [
            "entity_count",
            "query_length",
            "question_marks",
            "conjunctions",
            "complex_keywords",
            "has_multi_part",
        ]
        for factor in expected_factors:
            assert factor in result.factors

    def test_analysis_result_score_range(self):
        """Test that complexity score is in valid range (0-100)."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("장학금 신청 방법")
        assert 0 <= result.score <= 100

    def test_analysis_result_processing_time_positive(self):
        """Test that processing time is positive."""
        selector = AdaptiveTopKSelector()
        result = selector.analyze_complexity("장학금 신청 방법")
        assert result.processing_time_ms >= 0


class TestLatencyGuardrails:
    """Test latency guardrail behavior (REQ-AT-009)."""

    def test_record_latency(self):
        """Test recording latency values."""
        selector = AdaptiveTopKSelector()
        selector.record_latency(100.0)
        selector.record_latency(200.0)
        selector.record_latency(150.0)

        assert selector.get_average_latency() == pytest.approx(150.0)

    def test_get_average_latency_no_history(self):
        """Test average latency with no history."""
        selector = AdaptiveTopKSelector()
        assert selector.get_average_latency() is None

    def test_clear_latency_history(self):
        """Test clearing latency history."""
        selector = AdaptiveTopKSelector()
        selector.record_latency(100.0)
        selector.clear_latency_history()
        assert selector.get_average_latency() is None

    def test_latency_history_max_10_entries(self):
        """Test that latency history keeps only last 10 entries."""
        selector = AdaptiveTopKSelector()
        for i in range(15):
            selector.record_latency(float(i))

        # Should have only last 10
        assert selector.get_average_latency() == pytest.approx(9.5)  # Average of 5-14

    def test_latency_guardrail_reduces_top_k(self):
        """Test that high latency reduces Top-K (REQ-AT-009)."""
        selector = AdaptiveTopKSelector()
        # Record high latencies
        for _ in range(5):
            selector.record_latency(600.0)  # Above threshold of 500ms

        top_k = selector.select_top_k("장학금 신청 방법", latency_guardrails=True)
        # Should reduce from 15 to at least 5
        assert top_k < 15

    def test_latency_guardrail_disabled(self):
        """Test that disabling guardrails prevents Top-K reduction."""
        selector = AdaptiveTopKSelector()
        # Record high latencies
        for _ in range(5):
            selector.record_latency(600.0)

        # Use a query that will clearly be COMPLEX without question markers
        top_k = selector.select_top_k(
            "장학금 신청 절차와 자격 요건", latency_guardrails=False
        )
        # Should not reduce with guardrails disabled
        assert top_k == 15


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_long_query(self):
        """Test behavior with very long query."""
        selector = AdaptiveTopKSelector()
        long_query = " ".join(["장학금"] * 100)
        result = selector.analyze_complexity(long_query)

        # Should handle gracefully
        assert result.complexity in [
            QueryComplexity.COMPLEX,
            QueryComplexity.MULTI_PART,
        ]
        assert result.top_k <= 25  # Max limit

    def test_query_with_special_characters(self):
        """Test query with special characters."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("장학금! 신청? 방법...")
        assert isinstance(top_k, int)
        assert 1 <= top_k <= 25

    def test_query_with_numbers(self):
        """Test query with numbers."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("제15조 장학금 지급")
        assert isinstance(top_k, int)
        assert 1 <= top_k <= 25

    def test_query_with_mixed_english_korean(self):
        """Test query with mixed English and Korean."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("GPA 장학금 신청 방법")
        assert isinstance(top_k, int)
        assert 1 <= top_k <= 25

    def test_unicode_normalization(self):
        """Test that unicode characters are handled correctly."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("장학금 ㅏㅏㅏ 신청")  # With Korean vowels
        assert isinstance(top_k, int)
        assert 1 <= top_k <= 25


class TestComplexityFactors:
    """Test individual complexity factor calculations."""

    def test_entity_count_factor(self):
        """Test entity count factor calculation."""
        selector = AdaptiveTopKSelector()

        # Single word
        result1 = selector.analyze_complexity("장학금")
        assert result1.factors["entity_count"] < 0.5

        # Multiple words
        result2 = selector.analyze_complexity(" ".join(["장학금"] * 10))
        assert result2.factors["entity_count"] >= 0.5

    def test_query_length_factor(self):
        """Test query length factor calculation."""
        selector = AdaptiveTopKSelector()

        # Short query
        result1 = selector.analyze_complexity("휴학")
        assert result1.factors["query_length"] < 0.5

        # Long query (uses actual length: 24 chars / 50 = 0.48)
        result2 = selector.analyze_complexity(
            "장학금 신청 방법과 자격 요건 및 구비 서류"
        )
        # The query is 24 characters, which gives 24/50 = 0.48
        assert 0.4 <= result2.factors["query_length"] < 0.5

    def test_question_marks_factor(self):
        """Test question marks factor calculation."""
        selector = AdaptiveTopKSelector()

        # No question marks
        result1 = selector.analyze_complexity("장학금 신청")
        assert result1.factors["question_marks"] == 0

        # With question marker
        result2 = selector.analyze_complexity("장학금을 어떻게 신청하나요?")
        assert result2.factors["question_marks"] > 0

    def test_conjunctions_factor(self):
        """Test conjunctions factor calculation."""
        selector = AdaptiveTopKSelector()

        # No conjunctions
        result1 = selector.analyze_complexity("장학금 신청 방법")
        assert result1.factors["conjunctions"] == 0

        # With conjunction
        result2 = selector.analyze_complexity("장학금 신청 방법 그리고 자격 요건")
        assert result2.factors["conjunctions"] > 0

    def test_complex_keywords_factor(self):
        """Test complex keywords factor calculation."""
        selector = AdaptiveTopKSelector()

        # No complex keywords
        result1 = selector.analyze_complexity("장학금")
        assert result1.factors["complex_keywords"] == 0

        # With complex keywords
        result2 = selector.analyze_complexity("장학금 신청 절차 자격 요건")
        assert result2.factors["complex_keywords"] > 0


class TestConfigurationScenarios:
    """Test various configuration scenarios."""

    def test_custom_config_affects_selection(self):
        """Test that custom configuration affects Top-K selection."""
        config = TopKConfig(simple=3, medium=6, complex=9, multi_part=12, max_limit=15)
        selector = AdaptiveTopKSelector(config)

        # Simple query
        top_k1 = selector.select_top_k("휴학")
        assert top_k1 == 3

        # Medium query
        top_k2 = selector.select_top_k("장학금을 어떻게 신청하나요?")
        assert top_k2 == 6

    def test_fallback_to_medium_on_classification_failure(self):
        """Test fallback to medium Top-K if classification fails (REQ-AT-010)."""
        selector = AdaptiveTopKSelector()
        # Use a query that doesn't fit any category clearly
        # The fallback to medium happens for queries without clear classification
        # We can test this by verifying the default behavior
        top_k = selector.select_top_k("")
        # Empty query falls back to medium
        assert top_k == 10


class TestSpecCompliance:
    """Test SPEC-RAG-SEARCH-001 TAG-003 compliance."""

    def test_req_at_001_dynamic_adjustment(self):
        """Test REQ-AT-001: System adjusts Top-K based on query complexity."""
        selector = AdaptiveTopKSelector()

        simple_top_k = selector.select_top_k("휴학")
        medium_top_k = selector.select_top_k("휴학을 어떻게 하나요?")
        complex_top_k = selector.select_top_k("휴학 신청 절차와 구비서류")
        multi_part_top_k = selector.select_top_k("휴학 절차 그리고 복학 방법")

        # Should be different for different complexities
        assert simple_top_k < medium_top_k <= complex_top_k <= multi_part_top_k

    def test_req_at_002_simple_top_5(self):
        """Test REQ-AT-002: Simple queries use Top-5."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("교원인사규정")
        assert top_k == 5

    def test_req_at_003_medium_top_10(self):
        """Test REQ-AT-003: Medium queries use Top-10."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("장학금을 어떻게 신청하나요?")
        assert top_k == 10

    def test_req_at_004_complex_top_15(self):
        """Test REQ-AT-004: Complex queries use Top-15."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("장학금 신청 절차와 구비서류")
        assert top_k == 15

    def test_req_at_005_multi_part_top_20(self):
        """Test REQ-AT-005: Multi-part queries use Top-20."""
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("장학금 신청 방법 그리고 자격 요건")
        assert top_k == 20

    def test_req_at_009_latency_guardrail(self):
        """Test REQ-AT-009: System reduces Top-K if response time > 500ms."""
        selector = AdaptiveTopKSelector()

        # Record latencies above threshold
        for _ in range(5):
            selector.record_latency(600.0)

        top_k_with_guardrails = selector.select_top_k(
            "장학금 신청 절차와 구비서류", latency_guardrails=True
        )
        top_k_without_guardrails = selector.select_top_k(
            "장학금 신청 절차와 구비서류", latency_guardrails=False
        )

        # With guardrails should be lower
        assert top_k_with_guardrails < top_k_without_guardrails

    def test_req_at_010_fallback_on_classification_failure(self):
        """Test REQ-AT-010: System falls back to Top-10 if classification fails."""
        selector = AdaptiveTopKSelector()
        # Empty query should fallback to medium
        top_k = selector.select_top_k("")
        assert top_k == 10

    def test_req_at_011_return_available_if_less_than_top_k(self):
        """Test REQ-AT-011: System returns available results if < Top-K."""
        # This is tested at integration level, here we just verify
        # that the selector provides appropriate Top-K values
        selector = AdaptiveTopKSelector()
        top_k = selector.select_top_k("복학 절차")
        assert top_k > 0
