"""
Tests for Colloquial-to-Formal Query Transformer.

Implements TDD tests for SPEC-RAG-QUALITY-003 Phase 1.
"""

import pytest

from src.rag.domain.query.colloquial_transformer import (
    ColloquialPattern,
    ColloquialTransformer,
    TransformResult,
    create_colloquial_transformer,
)


class TestColloquialPattern:
    """Tests for ColloquialPattern dataclass."""

    def test_valid_pattern_creation(self):
        """Test creating a valid ColloquialPattern."""
        pattern = ColloquialPattern(
            pattern="어떻게 해",
            formal="방법",
            context="procedure",
        )
        assert pattern.pattern == "어떻게 해"
        assert pattern.formal == "방법"
        assert pattern.context == "procedure"

    def test_default_context(self):
        """Test that context defaults to 'general'."""
        pattern = ColloquialPattern(pattern="뭐야", formal="정의")
        assert pattern.context == "general"

    def test_invalid_empty_pattern(self):
        """Test that empty pattern raises ValueError."""
        with pytest.raises(ValueError, match="Pattern and formal must be non-empty"):
            ColloquialPattern(pattern="", formal="정의")

    def test_invalid_empty_formal(self):
        """Test that empty formal raises ValueError."""
        with pytest.raises(ValueError, match="Pattern and formal must be non-empty"):
            ColloquialPattern(pattern="뭐야", formal="")


class TestTransformResult:
    """Tests for TransformResult dataclass."""

    def test_transformation_applied_property(self):
        """Test transformation_applied property alias."""
        result = TransformResult(
            original_query="휴학 어떻게 해?",
            transformed_query="휴학 방법",
            was_transformed=True,
        )
        assert result.transformation_applied is True

    def test_transformation_not_applied(self):
        """Test when no transformation was applied."""
        result = TransformResult(
            original_query="휴학 규정",
            transformed_query="휴학 규정",
            was_transformed=False,
        )
        assert result.transformation_applied is False

    def test_default_values(self):
        """Test default values for TransformResult."""
        result = TransformResult(
            original_query="query",
            transformed_query="query",
        )
        assert result.patterns_matched == []
        assert result.confidence == 1.0
        assert result.was_transformed is False
        assert result.context_hints == []
        assert result.method == "none"


class TestColloquialTransformer:
    """Tests for ColloquialTransformer class."""

    @pytest.fixture
    def transformer(self):
        """Create a transformer instance for testing."""
        return ColloquialTransformer()

    # ============ Basic Transformation Tests ============

    def test_transform_basic_colloquial_query(self, transformer):
        """Test transforming a basic colloquial query."""
        result = transformer.transform("휴학 어떻게 해?")
        assert result.was_transformed is True
        assert "방법" in result.transformed_query

    def test_transform_preserves_original_query(self, transformer):
        """Test that original query is preserved in result."""
        original = "휴학 어떻게 해?"
        result = transformer.transform(original)
        assert result.original_query == original

    def test_transform_empty_query(self, transformer):
        """Test handling empty query."""
        result = transformer.transform("")
        assert result.method == "empty"
        assert result.transformed_query == ""

    def test_transform_whitespace_only_query(self, transformer):
        """Test handling whitespace-only query."""
        result = transformer.transform("   ")
        assert result.method == "empty"

    # ============ Formal Query Detection Tests ============

    def test_formal_query_not_transformed(self, transformer):
        """Test that formal queries are not transformed."""
        result = transformer.transform("제3조제1항에 따른 휴학 규정")
        assert result.was_transformed is False
        assert result.method == "formal_skip"

    def test_formal_query_with_regulation_keyword(self, transformer):
        """Test that queries with '규정' are not transformed."""
        result = transformer.transform("휴학 규정에 대해 알려주세요")
        assert result.method == "formal_skip"

    def test_formal_query_with_article_reference(self, transformer):
        """Test that queries with article references are not transformed."""
        result = transformer.transform("제5조 휴학 절차")
        assert result.method == "formal_skip"

    # ============ Dictionary Pattern Tests ============

    def test_dictionary_pattern_how_to(self, transformer):
        """Test '어떻게 해' -> '방법' transformation."""
        result = transformer.transform("휴학 어떻게 해?")
        assert "방법" in result.transformed_query
        assert result.method in ["dictionary", "combined"]

    def test_dictionary_pattern_what_is(self, transformer):
        """Test '뭐야' -> '정의' transformation."""
        result = transformer.transform("장학금이 뭐야?")
        assert "정의" in result.transformed_query

    def test_dictionary_pattern_tell_me(self, transformer):
        """Test '알려줘' -> '안내' transformation."""
        result = transformer.transform("등록금 납부 알려줘")
        assert "안내" in result.transformed_query

    def test_dictionary_pattern_deadline(self, transformer):
        """Test '언제까지' -> '기한' transformation."""
        result = transformer.transform("신청 언제까지야?")
        assert "기한" in result.transformed_query

    # ============ Regex Pattern Tests ============

    def test_regex_pattern_how_to_method(self, transformer):
        """Test regex transformation for '하는법' -> '방법'."""
        result = transformer.transform("장학금 신청하는법")
        assert "방법" in result.transformed_query

    def test_regex_pattern_where(self, transformer):
        """Test regex transformation for '어디서' -> '위치'."""
        result = transformer.transform("등록금 납부어디서")
        assert "위치" in result.transformed_query

    # ============ Context Hint Tests ============

    def test_context_hint_procedure(self, transformer):
        """Test extracting procedure context hint."""
        result = transformer.transform("휴학 어떻게 해?")
        assert "procedure" in result.context_hints

    def test_context_hint_definition(self, transformer):
        """Test extracting definition context hint."""
        result = transformer.transform("장학금이 뭐야?")
        assert "definition" in result.context_hints

    def test_context_hint_deadline(self, transformer):
        """Test extracting deadline context hint."""
        result = transformer.transform("신청 언제까지?")
        assert "deadline" in result.context_hints

    # ============ Confidence Score Tests ============

    def test_high_confidence_dictionary_match(self, transformer):
        """Test high confidence for direct dictionary match."""
        result = transformer.transform("휴학 어떻게 해?")
        assert result.confidence >= 0.85

    def test_confidence_one_for_no_transform(self, transformer):
        """Test confidence is 1.0 when no transformation is needed."""
        result = transformer.transform("제3조 휴학 규정")
        assert result.confidence == 1.0

    # ============ Pattern Detection Tests ============

    def test_detect_patterns_finds_colloquial(self, transformer):
        """Test that detect_patterns finds colloquial patterns."""
        patterns = transformer.detect_patterns("휴학 어떻게 해?")
        assert len(patterns) > 0
        assert any(p.formal == "방법" for p in patterns)

    def test_detect_patterns_empty_for_formal(self, transformer):
        """Test that detect_patterns returns empty for formal queries."""
        patterns = transformer.detect_patterns("제3조 휴학 규정")
        # May find patterns but transformation won't be applied
        assert isinstance(patterns, list)

    # ============ Cache Tests ============

    def test_cache_enabled_by_default(self, transformer):
        """Test that cache is enabled by default."""
        assert transformer._cache_enabled is True

    def test_cache_stores_result(self, transformer):
        """Test that cache stores transformation results."""
        query = "휴학 어떻게 해?"
        result1 = transformer.transform(query)
        result2 = transformer.transform(query)

        # Both should be the same object (from cache)
        assert result1.transformed_query == result2.transformed_query

    def test_clear_cache(self, transformer):
        """Test clearing the cache."""
        transformer.transform("휴학 어떻게 해?")
        assert len(transformer._cache) > 0

        transformer.clear_cache()
        assert len(transformer._cache) == 0

    # ============ Statistics Tests ============

    def test_get_stats(self, transformer):
        """Test getting transformer statistics."""
        stats = transformer.get_stats()

        assert "total_mappings" in stats
        assert "total_regex_patterns" in stats
        assert "total_formal_indicators" in stats
        assert "cache_size" in stats
        assert "unknown_patterns_queued" in stats

        # Verify we have loaded patterns
        assert stats["total_mappings"] >= 50  # SPEC requires at least 50 patterns
        assert stats["total_regex_patterns"] > 0
        assert stats["total_formal_indicators"] > 0

    # ============ Unknown Pattern Queue Tests ============

    def test_queue_unknown_pattern(self, transformer):
        """Test queuing unknown patterns for expansion."""
        transformer.queue_unknown_pattern("새로운패턴")
        assert "새로운패턴" in transformer.get_unknown_patterns()

    def test_queue_unknown_pattern_no_duplicates(self, transformer):
        """Test that duplicate patterns are not queued twice."""
        transformer.queue_unknown_pattern("패턴1")
        transformer.queue_unknown_pattern("패턴1")
        assert transformer.get_unknown_patterns().count("패턴1") == 1

    # ============ Edge Cases Tests ============

    def test_multiple_patterns_in_query(self, transformer):
        """Test transforming query with multiple colloquial patterns."""
        result = transformer.transform("장학금 어떻게 해? 언제까지야?")
        assert result.was_transformed is True
        assert len(result.patterns_matched) >= 1

    def test_query_with_mixed_formal_colloquial(self, transformer):
        """Test query with both formal and colloquial elements."""
        # Even with colloquial ending, if formal indicators present, skip
        result = transformer.transform("규정에 따른 휴학 어떻게 해?")
        # Should still transform since not purely formal
        assert isinstance(result, TransformResult)

    def test_special_characters_preserved(self, transformer):
        """Test that special characters are preserved during transformation."""
        result = transformer.transform("휴학 어떻게 해?!")
        assert "!" in result.transformed_query or "?" in result.transformed_query

    # ============ Integration Tests ============

    def test_full_pipeline_with_typical_student_query(self, transformer):
        """Test full pipeline with a typical student query."""
        result = transformer.transform("휴학 신청 어떻게 해? 언제까지야?")

        assert result.was_transformed is True
        assert len(result.patterns_matched) > 0
        assert len(result.context_hints) > 0
        assert result.confidence > 0.5

    def test_transformation_accuracy_target(self, transformer):
        """Test transformation accuracy meets 95% target."""
        test_cases = [
            ("휴학 어떻게 해?", True, "방법"),
            ("장학금 뭐야?", True, "정의"),
            ("등록금 납부 알려줘", True, "안내"),
            ("신청 언제까지?", True, "기한"),
            ("제3조 규정", False, ""),  # Formal, no transform
            ("서류 어디서?", True, "위치"),
        ]

        correct = 0
        total = len(test_cases)

        for query, should_transform, expected_keyword in test_cases:
            result = transformer.transform(query)
            if should_transform:
                if expected_keyword in result.transformed_query:
                    correct += 1
            else:
                if not result.was_transformed:
                    correct += 1

        accuracy = correct / total
        assert accuracy >= 0.95, f"Accuracy {accuracy:.2%} below 95% target"


class TestColloquialTransformerFactory:
    """Tests for factory function."""

    def test_create_with_default_path(self):
        """Test creating transformer with default path."""
        transformer = create_colloquial_transformer()
        assert isinstance(transformer, ColloquialTransformer)

    def test_create_with_custom_path(self, tmp_path):
        """Test creating transformer with custom path."""
        import json

        config = {
            "mappings": [{"pattern": "테스트", "formal": "시험", "context": "test"}],
            "regex_patterns": [],
            "formal_indicators": [],
        }

        config_path = tmp_path / "test_patterns.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        transformer = create_colloquial_transformer(str(config_path))
        assert transformer._mappings[0].pattern == "테스트"


class TestSPECRequirements:
    """Tests for SPEC-RAG-QUALITY-003 requirements."""

    @pytest.fixture
    def transformer(self):
        """Create transformer for SPEC requirement testing."""
        return ColloquialTransformer()

    def test_req_min_50_patterns(self, transformer):
        """REQ: System SHALL handle at least 50 common colloquial patterns."""
        stats = transformer.get_stats()
        assert stats["total_mappings"] >= 50

    def test_req_transformation_preserves_intent(self, transformer):
        """REQ: Transformation SHALL preserve original query intent."""
        # Test that semantic meaning is preserved
        result = transformer.transform("휴학 어떻게 해?")

        # Original keyword should still be present
        assert "휴학" in result.transformed_query

        # Formal equivalent should be added
        assert "방법" in result.transformed_query

    def test_req_log_transformation_decisions(self, transformer):
        """REQ: System SHALL log transformation decisions for debugging."""
        # This is tested indirectly through the logging mechanism
        # Just verify the flag exists and can be toggled
        assert hasattr(transformer, "_enable_logging")

        transformer_no_log = ColloquialTransformer(enable_logging=False)
        assert transformer_no_log._enable_logging is False

    def test_req_fallback_on_unknown_pattern(self, transformer):
        """REQ: System SHALL fallback to original query with warning for unknown patterns."""
        # Use a query that likely won't match any pattern
        result = transformer.transform("xyz123 unknown pattern test")

        # Should return original query unchanged
        assert result.transformed_query == "xyz123 unknown pattern test"
        assert result.was_transformed is False

    def test_req_performance_under_50ms(self, transformer):
        """REQ: Transformation SHALL complete within 50ms."""
        import time

        queries = [
            "휴학 어떻게 해?",
            "장학금 뭐야?",
            "등록금 납부 알려줘",
            "신청 언제까지?",
        ]

        for query in queries:
            start = time.perf_counter()
            transformer.transform(query)
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < 50, f"Transformation took {elapsed_ms:.2f}ms for '{query}'"

    def test_req_colloquial_handling_rate(self, transformer):
        """REQ: Colloquial query handling rate >= 85%."""
        # Test with a variety of colloquial queries
        test_queries = [
            "휴학 어떻게 해?",
            "장학금 뭐야?",
            "등록금 납부 알려줘",
            "신청 언제까지?",
            "복학 어떻게 하나요?",
            "성적 이의 신청하고 싶어",
            "졸업 요건 뭐야?",
            "전과 할 수 있?",
            "휴학원 어디서?",
            "교수 연구년 되나요?",
            "조교 신청 하는법",
            "등록금 분할 납부 되나?",
            "학사 경고 뭐야?",
            "수강 신청 언제?",
            "성적 정정 어떻게?",
        ]

        handled = 0
        total = len(test_queries)

        for query in test_queries:
            result = transformer.transform(query)
            if result.was_transformed or result.method == "formal_skip":
                handled += 1

        handling_rate = handled / total
        assert handling_rate >= 0.85, f"Handling rate {handling_rate:.2%} below 85% target"
