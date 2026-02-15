"""
Tests for LLM-as-Judge Evaluation System (Phase 4 Enhancement).

SPEC-RAG-QUALITY-003 Phase 4: LLM-as-Judge Integration

Tests cover:
- Judgment caching
- Graceful degradation
- Multi-provider support
- Cache statistics
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from src.rag.domain.evaluation.llm_judge import (
    LLMJudge,
    JudgeResult,
    JudgmentCache,
    EvaluationBatch,
    EvaluationSummary,
    QualityLevel,
)


class TestJudgmentCache:
    """Tests for JudgmentCache functionality."""

    def test_cache_initialization(self):
        """Test cache initializes with correct settings."""
        cache = JudgmentCache(max_size=100, ttl_seconds=600)

        stats = cache.get_stats()
        assert stats["max_size"] == 100
        assert stats["ttl_seconds"] == 600
        assert stats["cache_size"] == 0
        assert stats["hit_rate"] == 0.0

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        cache = JudgmentCache(max_size=10, ttl_seconds=3600)

        # Create a mock result
        result = JudgeResult(
            query="test query",
            answer="test answer",
            sources=[{"content": "source", "score": 0.9}],
            accuracy=0.9,
            completeness=0.8,
            citations=0.7,
            context_relevance=0.85,
            overall_score=0.82,
            passed=True,
        )

        # Set in cache
        cache.set(result, "test query", "test answer", [{"content": "source", "score": 0.9}])

        # Get from cache
        cached = cache.get("test query", "test answer", [{"content": "source", "score": 0.9}])

        assert cached is not None
        assert cached.query == "test query"
        assert cached.accuracy == 0.9

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = JudgmentCache()

        cached = cache.get("nonexistent", "answer", [])
        assert cached is None

        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_cache_hit_rate_calculation(self):
        """Test hit rate calculation."""
        cache = JudgmentCache()

        result = JudgeResult(
            query="q",
            answer="a",
            sources=[],
            accuracy=0.8,
            completeness=0.8,
            citations=0.8,
            context_relevance=0.8,
            overall_score=0.8,
            passed=True,
        )

        # Set one item
        cache.set(result, "q1", "a1", [])

        # One hit
        cache.get("q1", "a1", [])
        # One miss
        cache.get("q2", "a2", [])

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        cache = JudgmentCache(max_size=10, ttl_seconds=0)  # 0 seconds TTL

        result = JudgeResult(
            query="q",
            answer="a",
            sources=[],
            accuracy=0.8,
            completeness=0.8,
            citations=0.8,
            context_relevance=0.8,
            overall_score=0.8,
            passed=True,
        )

        cache.set(result, "q", "a", [])

        # Should be expired immediately
        cached = cache.get("q", "a", [])
        assert cached is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = JudgmentCache(max_size=2, ttl_seconds=3600)

        result = JudgeResult(
            query="q",
            answer="a",
            sources=[],
            accuracy=0.8,
            completeness=0.8,
            citations=0.8,
            context_relevance=0.8,
            overall_score=0.8,
            passed=True,
        )

        # Fill cache
        cache.set(result, "q1", "a1", [])
        cache.set(result, "q2", "a2", [])

        # Add third item - should evict first
        cache.set(result, "q3", "a3", [])

        # First item should be evicted
        cached = cache.get("q1", "a1", [])
        assert cached is None

        # Second and third should exist
        assert cache.get("q2", "a2", []) is not None
        assert cache.get("q3", "a3", []) is not None

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = JudgmentCache()

        result = JudgeResult(
            query="q",
            answer="a",
            sources=[],
            accuracy=0.8,
            completeness=0.8,
            citations=0.8,
            context_relevance=0.8,
            overall_score=0.8,
            passed=True,
        )

        cache.set(result, "q1", "a1", [])
        cache.set(result, "q2", "a2", [])

        cache.clear()

        stats = cache.get_stats()
        assert stats["cache_size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestLLMJudgePhase4:
    """Tests for LLMJudge Phase 4 enhancements."""

    def test_initialization_with_cache(self):
        """Test LLMJudge initializes with cache."""
        judge = LLMJudge(enable_cache=True, cache_max_size=100, cache_ttl_seconds=600)

        assert judge._enable_cache is True
        assert judge._cache is not None

    def test_initialization_without_cache(self):
        """Test LLMJudge can be initialized without cache."""
        judge = LLMJudge(enable_cache=False)

        assert judge._enable_cache is False
        assert judge._cache is None

    def test_cache_hit_on_repeated_evaluation(self):
        """Test that cache is used for repeated evaluations."""
        mock_client = Mock()
        mock_client.generate.return_value = json.dumps({
            "accuracy": 0.9,
            "completeness": 0.8,
            "citations": 0.7,
            "context_relevance": 0.85,
            "accuracy_reasoning": "test",
            "issues": [],
            "strengths": [],
        })

        with patch(
            "src.rag.domain.evaluation.llm_judge.EVALUATION_PROMPTS_AVAILABLE", True
        ):
            with patch(
                "src.rag.domain.evaluation.llm_judge.EvaluationPrompts"
            ) as mock_prompts:
                mock_prompts.format_accuracy_prompt.return_value = ("sys", "user")

                judge = LLMJudge(llm_client=mock_client, enable_cache=True)

                # First evaluation - should call LLM
                result1 = judge.evaluate_with_llm(
                    "query", "answer", [{"content": "src", "score": 0.9}]
                )

                # Second evaluation with same params - should use cache
                result2 = judge.evaluate_with_llm(
                    "query", "answer", [{"content": "src", "score": 0.9}]
                )

                # LLM should only be called once
                assert mock_client.generate.call_count == 1

                # Results should be identical
                assert result1.accuracy == result2.accuracy

    def test_fallback_to_rule_based_on_llm_failure(self):
        """Test graceful fallback to rule-based evaluation."""
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("LLM unavailable")

        judge = LLMJudge(llm_client=mock_client)

        result = judge.evaluate_with_llm("query", "answer", [])

        # Should still return a result
        assert result is not None
        assert isinstance(result, JudgeResult)

    def test_stats_tracking(self):
        """Test that statistics are tracked correctly."""
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("LLM unavailable")

        judge = LLMJudge(llm_client=mock_client)

        # Trigger fallback evaluation
        judge.evaluate_with_llm("q1", "a1", [])

        stats = judge.get_stats()
        assert "fallback_evaluations" in stats
        assert stats["fallback_evaluations"] == 1

    def test_is_llm_available(self):
        """Test LLM availability check."""
        mock_client = Mock()
        judge = LLMJudge(llm_client=mock_client)

        # Should return boolean
        assert isinstance(judge.is_llm_available(), bool)


class TestLLMJudgeCharacterization:
    """Characterization tests to preserve existing behavior."""

    def test_evaluate_returns_judge_result(self):
        """Test that evaluate returns a JudgeResult."""
        judge = LLMJudge()

        result = judge.evaluate(
            "휴학 신청 방법",
            "휴학은 학기 시작 전에 신청할 수 있습니다.",
            [{"content": "source", "score": 0.9}],
        )

        assert isinstance(result, JudgeResult)
        assert result.query == "휴학 신청 방법"
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.completeness <= 1.0
        assert 0.0 <= result.citations <= 1.0
        assert 0.0 <= result.context_relevance <= 1.0

    def test_hallucination_detection(self):
        """Test that hallucinations are detected."""
        judge = LLMJudge()

        # Answer with wrong university name
        result = judge.evaluate(
            "test query",
            "서울대에 문의하세요.",  # Wrong university
            [],
        )

        assert result.accuracy == 0.0

    def test_empty_answer_handling(self):
        """Test handling of empty answers."""
        judge = LLMJudge()

        result = judge.evaluate("query", "", [])

        assert result.accuracy == 0.0

    def test_pass_fail_determination(self):
        """Test pass/fail determination logic."""
        judge = LLMJudge()

        # High quality answer should pass
        high_quality = judge.evaluate(
            "휴학 신청 방법",
            "학칙 제15조에 따라 휴학은 학기 시작 14일 전까지 신청 가능합니다. "
            "휴학신청서를 소속 학과에 제출하시기 바랍니다.",
            [{"content": "source", "score": 0.95}],
        )

        # Check thresholds are applied
        if high_quality.overall_score >= 0.80:
            assert high_quality.passed is True


class TestEvaluationBatch:
    """Tests for EvaluationBatch functionality."""

    def test_batch_summary_calculation(self):
        """Test batch summary statistics."""
        batch = EvaluationBatch()

        # Add multiple results
        for i in range(5):
            result = JudgeResult(
                query=f"q{i}",
                answer=f"a{i}",
                sources=[],
                accuracy=0.8,
                completeness=0.8,
                citations=0.8,
                context_relevance=0.8,
                overall_score=0.8,
                passed=i < 3,  # 3 pass, 2 fail
            )
            batch.add_result(result)

        summary = batch.get_summary()

        assert summary.total_queries == 5
        assert summary.passed == 3
        assert summary.failed == 2
        assert summary.pass_rate == 0.6

    def test_empty_batch_summary(self):
        """Test summary for empty batch."""
        batch = EvaluationBatch()

        summary = batch.get_summary()

        assert summary.total_queries == 0
        assert summary.passed == 0


class TestJudgeResult:
    """Tests for JudgeResult dataclass."""

    def test_auto_generated_evaluation_id(self):
        """Test that evaluation_id is auto-generated."""
        result = JudgeResult(
            query="q",
            answer="a",
            sources=[],
            accuracy=0.8,
            completeness=0.8,
            citations=0.8,
            context_relevance=0.8,
            overall_score=0.8,
            passed=True,
        )

        assert result.evaluation_id != ""
        assert result.evaluation_id.startswith("eval_")

    def test_auto_generated_timestamp(self):
        """Test that timestamp is auto-generated."""
        result = JudgeResult(
            query="q",
            answer="a",
            sources=[],
            accuracy=0.8,
            completeness=0.8,
            citations=0.8,
            context_relevance=0.8,
            overall_score=0.8,
            passed=True,
        )

        assert result.timestamp != ""


class TestSemanticFallback:
    """Tests for semantic evaluator fallback (Phase 4)."""

    def test_semantic_fallback_enhances_accuracy(self):
        """Test that semantic fallback enhances accuracy calculation."""
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("LLM unavailable")

        # Create mock semantic evaluator
        mock_semantic = Mock()
        mock_result = Mock()
        mock_result.similarity_score = 0.85
        mock_semantic.evaluate_with_query.return_value = mock_result

        judge = LLMJudge(
            llm_client=mock_client,
            semantic_evaluator=mock_semantic,
        )

        result = judge._evaluate_with_fallback(
            "query",
            "answer with expected content",
            [],
            ["expected content"]
        )

        # Should have called semantic evaluator
        mock_semantic.evaluate_with_query.assert_called_once()

        # Result should be a JudgeResult
        assert isinstance(result, JudgeResult)
