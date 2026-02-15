"""
Tests for Hybrid Weight Optimizer (Phase 5).

SPEC-RAG-QUALITY-003 Phase 5: Hybrid Weight Optimization

Tests cover:
- Formal/informal query detection
- Dynamic weight calculation
- Integration with ColloquialTransformer
- Statistics tracking
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.rag.application.hybrid_weight_optimizer import (
    HybridWeightOptimizer,
    WeightDecision,
    create_hybrid_weight_optimizer,
)


class TestWeightDecision:
    """Tests for WeightDecision dataclass."""

    def test_weight_sum_is_one(self):
        """Test that weight sum equals 1.0."""
        decision = WeightDecision(
            bm25_weight=0.3,
            vector_weight=0.7,
            is_colloquial=True,
            formality_score=0.3,
        )

        assert decision.total_weight == 1.0

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        decision = WeightDecision(
            bm25_weight=0.5,
            vector_weight=0.5,
            is_colloquial=False,
            formality_score=0.8,
            detected_patterns=["pattern1"],
            reasoning="test reasoning",
        )

        result = decision.to_dict()

        assert result["bm25_weight"] == 0.5
        assert result["vector_weight"] == 0.5
        assert result["is_colloquial"] is False
        assert result["formality_score"] == 0.8
        assert "pattern1" in result["detected_patterns"]
        assert result["reasoning"] == "test reasoning"


class TestHybridWeightOptimizer:
    """Tests for HybridWeightOptimizer functionality."""

    def test_initialization_default_weights(self):
        """Test initialization with default weights."""
        optimizer = HybridWeightOptimizer()

        assert optimizer._formal_weights == (0.5, 0.5)
        assert optimizer._colloquial_weights == (0.3, 0.7)

    def test_initialization_custom_weights(self):
        """Test initialization with custom weights."""
        optimizer = HybridWeightOptimizer(
            formal_weights=(0.6, 0.4),
            colloquial_weights=(0.2, 0.8),
        )

        assert optimizer._formal_weights == (0.6, 0.4)
        assert optimizer._colloquial_weights == (0.2, 0.8)

    def test_invalid_weights_raises_error(self):
        """Test that invalid weights raise ValueError."""
        with pytest.raises(ValueError):
            HybridWeightOptimizer(formal_weights=(0.5, 0.6))  # Sum > 1

        with pytest.raises(ValueError):
            HybridWeightOptimizer(colloquial_weights=(1.5, -0.5))  # Invalid range

    def test_formal_query_detection(self):
        """Test detection of formal queries."""
        optimizer = HybridWeightOptimizer()

        # Formal query with regulation terms
        decision = optimizer.optimize("학칙 제15조 휴학신청방법에 대해 안내")

        assert decision.is_colloquial is False
        assert decision.formality_score >= 0.5
        assert decision.bm25_weight == 0.5
        assert decision.vector_weight == 0.5

    def test_colloquial_query_detection(self):
        """Test detection of colloquial queries."""
        optimizer = HybridWeightOptimizer()

        # Colloquial query
        decision = optimizer.optimize("휴학 어떻게 해?")

        assert decision.is_colloquial is True
        assert decision.formality_score < 0.5
        assert decision.bm25_weight == 0.3
        assert decision.vector_weight == 0.7

    def test_manual_weight_override(self):
        """Test manual weight override."""
        optimizer = HybridWeightOptimizer()

        decision = optimizer.optimize(
            "any query",
            override_weights=(0.4, 0.6),
        )

        assert decision.bm25_weight == 0.4
        assert decision.vector_weight == 0.6
        assert "override" in decision.reasoning.lower()

    def test_get_weights_convenience_method(self):
        """Test get_weights convenience method."""
        optimizer = HybridWeightOptimizer()

        bm25_w, vector_w = optimizer.get_weights("휴학 어떻게 해?")

        assert bm25_w + vector_w == 1.0
        assert 0.0 <= bm25_w <= 1.0
        assert 0.0 <= vector_w <= 1.0

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        optimizer = HybridWeightOptimizer()

        # Process some queries
        optimizer.optimize("학칙 제15조")  # Formal
        optimizer.optimize("휴학 어떻게 해?")  # Colloquial
        optimizer.optimize("규정 신청 방법")  # Formal

        stats = optimizer.get_stats()

        assert stats["total_queries"] == 3
        assert stats["formal_queries"] >= 1
        assert stats["colloquial_queries"] >= 1

    def test_reset_statistics(self):
        """Test statistics reset."""
        optimizer = HybridWeightOptimizer()

        optimizer.optimize("query")
        optimizer.reset_stats()

        stats = optimizer.get_stats()
        assert stats["total_queries"] == 0

    def test_set_formal_weights(self):
        """Test updating formal weights."""
        optimizer = HybridWeightOptimizer()

        optimizer.set_formal_weights(0.6, 0.4)

        assert optimizer._formal_weights == (0.6, 0.4)

    def test_set_colloquial_weights(self):
        """Test updating colloquial weights."""
        optimizer = HybridWeightOptimizer()

        optimizer.set_colloquial_weights(0.25, 0.75)

        assert optimizer._colloquial_weights == (0.25, 0.75)

    def test_invalid_weight_update_raises_error(self):
        """Test that invalid weight updates raise error."""
        optimizer = HybridWeightOptimizer()

        with pytest.raises(ValueError):
            optimizer.set_formal_weights(0.8, 0.5)

        with pytest.raises(ValueError):
            optimizer.set_colloquial_weights(-0.1, 1.1)


class TestRuleBasedFormalityDetection:
    """Tests for rule-based formality detection (fallback mode)."""

    def test_colloquial_pattern_detection(self):
        """Test detection of colloquial patterns."""
        optimizer = HybridWeightOptimizer()

        # Test colloquial patterns that are definitely in our pattern list
        colloquial_queries = [
            "휴학 어떻게 해?",  # "어떻게 해" pattern
            "이거 뭐야?",  # "뭐야" pattern
            "알려줘",  # "알려줘" pattern
        ]

        for query in colloquial_queries:
            is_colloquial, _, patterns = optimizer._rule_based_formality_detection(query)
            assert is_colloquial is True, f"Query '{query}' should be colloquial (patterns: {patterns})"

    def test_formal_indicator_detection(self):
        """Test detection of formal indicators."""
        optimizer = HybridWeightOptimizer()

        # Query with multiple formal indicators
        is_colloquial, formality_score, _ = optimizer._rule_based_formality_detection(
            "학칙 제15조 휴학신청방법 규정"
        )

        assert formality_score >= 0.5
        # Should be formal due to multiple formal indicators

    def test_mixed_query_handling(self):
        """Test handling of mixed formal/colloquial queries."""
        optimizer = HybridWeightOptimizer()

        # Query with both colloquial ending and formal terms
        decision = optimizer.optimize("휴학신청방법 알려줘")

        # Should have reasonable weights
        assert 0.0 <= decision.bm25_weight <= 1.0
        assert 0.0 <= decision.vector_weight <= 1.0


class TestColloquialTransformerIntegration:
    """Tests for ColloquialTransformer integration."""

    def test_integration_when_available(self):
        """Test integration when ColloquialTransformer is available."""
        mock_transformer = Mock()
        mock_result = Mock()
        mock_result.was_transformed = True
        mock_result.confidence = 0.9
        mock_result.patterns_matched = ["어떻게 해"]
        mock_transformer.transform.return_value = mock_result

        optimizer = HybridWeightOptimizer(
            colloquial_transformer=mock_transformer,
        )

        decision = optimizer.optimize("휴학 어떻게 해?")

        assert decision.is_colloquial is True
        assert "어떻게 해" in decision.detected_patterns

    def test_fallback_when_transformer_unavailable(self):
        """Test fallback when ColloquialTransformer is unavailable."""
        optimizer = HybridWeightOptimizer()
        optimizer._transformer_available = False
        optimizer._transformer = None

        # Should still work with rule-based detection
        decision = optimizer.optimize("휴학 어떻게 해?")

        assert decision.is_colloquial is True


class TestHybridWeightOptimizerLogging:
    """Tests for logging functionality."""

    def test_logging_enabled(self, caplog):
        """Test that logging works when enabled."""
        import logging

        with caplog.at_level(logging.INFO):
            optimizer = HybridWeightOptimizer(enable_logging=True)
            optimizer.optimize("휴학 어떻게 해?")

        assert any("Weight optimization" in record.message for record in caplog.records)

    def test_logging_disabled(self, caplog):
        """Test that logging is suppressed when disabled."""
        import logging

        with caplog.at_level(logging.INFO):
            optimizer = HybridWeightOptimizer(enable_logging=False)
            optimizer.optimize("휴학 어떻게 해?")

        # Should not have weight optimization logs
        weight_logs = [r for r in caplog.records if "Weight optimization" in r.message]
        assert len(weight_logs) == 0


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_hybrid_weight_optimizer(self):
        """Test factory function creates optimizer correctly."""
        optimizer = create_hybrid_weight_optimizer(enable_logging=False)

        assert isinstance(optimizer, HybridWeightOptimizer)
        assert optimizer._enable_logging is False

    def test_create_with_transformer(self):
        """Test factory function with transformer."""
        mock_transformer = Mock()

        optimizer = create_hybrid_weight_optimizer(
            colloquial_transformer=mock_transformer,
        )

        assert optimizer._transformer == mock_transformer


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_query(self):
        """Test handling of empty query."""
        optimizer = HybridWeightOptimizer()

        decision = optimizer.optimize("")

        assert decision is not None
        assert decision.bm25_weight + decision.vector_weight == 1.0

    def test_very_long_query(self):
        """Test handling of very long query."""
        optimizer = HybridWeightOptimizer()

        long_query = "학칙 제15조 " * 100

        decision = optimizer.optimize(long_query)

        assert decision is not None

    def test_query_with_special_characters(self):
        """Test handling of special characters."""
        optimizer = HybridWeightOptimizer()

        decision = optimizer.optimize("휴학 신청!@#$%방법?")

        assert decision is not None

    def test_query_with_numbers(self):
        """Test handling of queries with numbers."""
        optimizer = HybridWeightOptimizer()

        decision = optimizer.optimize("제15조 2024년 휴학신청")

        assert decision is not None


class TestPerformance:
    """Tests for performance requirements."""

    def test_optimization_speed(self):
        """Test that optimization completes quickly."""
        import time

        optimizer = HybridWeightOptimizer()

        queries = [
            "휴학 신청 방법",
            "학칙 제15조",
            "어떻게 해?",
            "규정 안내",
        ] * 10  # 40 queries

        start = time.perf_counter()
        for query in queries:
            optimizer.optimize(query)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should process all queries quickly (under 30ms each on average)
        avg_time_ms = elapsed_ms / len(queries)
        assert avg_time_ms < HybridWeightOptimizer.MAX_PROCESSING_TIME_MS


class TestWeightDecisionReasoning:
    """Tests for weight decision reasoning."""

    def test_colloquial_reasoning_includes_patterns(self):
        """Test that colloquial reasoning includes detected patterns."""
        optimizer = HybridWeightOptimizer()

        decision = optimizer.optimize("휴학 어떻게 해?")

        if decision.detected_patterns:
            assert any(
                pattern in decision.reasoning for pattern in decision.detected_patterns
            )

    def test_formal_reasoning_explains_balance(self):
        """Test that formal reasoning explains balanced weights."""
        optimizer = HybridWeightOptimizer()

        decision = optimizer.optimize("학칙 제15조 휴학신청방법")

        assert "balanced" in decision.reasoning.lower() or "formal" in decision.reasoning.lower()
