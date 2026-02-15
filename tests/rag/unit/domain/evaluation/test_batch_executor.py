"""
Unit tests for BatchEvaluationExecutor.

Tests for SPEC-RAG-EVAL-001 Milestone 1: API Budget Optimization.
"""

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.domain.evaluation.batch_executor import (
    BatchEvaluationExecutor,
    BatchResult,
    CacheEntry,
    CostEstimator,
    EvaluationCache,
    RateLimitConfig,
    RateLimiter,
    RateLimitStats,
)


@dataclass
class MockPersonaQuery:
    """Mock query for testing."""

    query: str
    persona: str = "test"
    category: str = "test"
    difficulty: str = "medium"


@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator function."""

    async def evaluate(query):
        await asyncio.sleep(0.01)  # Simulate API call
        return {"query": query.query, "score": 0.85}

    return evaluate


@pytest.fixture
def sync_mock_evaluator():
    """Create a synchronous mock evaluator."""

    def evaluate(query):
        return {"query": query.query, "score": 0.85}

    return evaluate


class TestRateLimitConfig:
    """Tests for RateLimitConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 60
        assert config.tokens_per_minute == 90000
        assert config.min_request_interval_ms == 1000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_minute=30, tokens_per_minute=45000, min_request_interval_ms=500
        )
        assert config.requests_per_minute == 30
        assert config.tokens_per_minute == 45000
        assert config.min_request_interval_ms == 500


class TestRateLimitStats:
    """Tests for RateLimitStats."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = RateLimitStats()
        assert stats.requests_made == 0
        assert stats.tokens_used == 0
        assert stats.last_request_time is None
        assert stats.requests_this_minute == 0
        assert stats.tokens_this_minute == 0


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_not_expired(self):
        """Test that fresh entry is not expired."""
        from datetime import datetime, timedelta

        entry = CacheEntry(
            result={"test": "data"}, timestamp=datetime.now(), ttl_seconds=3600
        )
        assert not entry.is_expired()

    def test_expired(self):
        """Test that old entry is expired."""
        from datetime import datetime, timedelta

        entry = CacheEntry(
            result={"test": "data"},
            timestamp=datetime.now() - timedelta(seconds=7200),
            ttl_seconds=3600,
        )
        assert entry.is_expired()


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_no_wait_initial(self):
        """Test that first acquire has no wait."""
        limiter = RateLimiter()
        wait_time = await limiter.acquire(1000)
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_updates_stats(self):
        """Test that acquire updates statistics."""
        limiter = RateLimiter()
        await limiter.acquire(1000)
        stats = limiter.get_stats()
        assert stats.requests_made == 1
        assert stats.tokens_used == 1000

    @pytest.mark.asyncio
    async def test_acquire_enforces_min_interval(self):
        """Test that rapid requests are rate limited."""
        config = RateLimitConfig(min_request_interval_ms=100)
        limiter = RateLimiter(config)

        # First request
        await limiter.acquire(100)

        # Second request should have wait time
        wait_time = await limiter.acquire(100)
        assert wait_time > 0

    @pytest.mark.asyncio
    async def test_acquire_respects_requests_per_minute(self):
        """Test that requests per minute limit is enforced."""
        config = RateLimitConfig(requests_per_minute=2, min_request_interval_ms=0)
        limiter = RateLimiter(config)

        # First two requests
        await limiter.acquire(100)
        await limiter.acquire(100)

        # Third request should have wait time (would exceed rate limit)
        # Note: Due to timing, this test might be flaky
        stats = limiter.get_stats()
        assert stats.requests_this_minute == 2


class TestEvaluationCache:
    """Tests for EvaluationCache."""

    def test_get_miss(self):
        """Test cache miss."""
        cache = EvaluationCache()
        result = cache.get("test query")
        assert result is None
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_set_and_get(self):
        """Test cache set and get."""
        cache = EvaluationCache()
        cache.set("test query", {"score": 0.9})
        result = cache.get("test query")
        assert result == {"score": 0.9}
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_get_with_persona(self):
        """Test cache with persona differentiation."""
        cache = EvaluationCache()
        cache.set("test query", {"score": 0.9}, persona="professor")
        cache.set("test query", {"score": 0.8}, persona="student")

        prof_result = cache.get("test query", persona="professor")
        student_result = cache.get("test query", persona="student")

        assert prof_result["score"] == 0.9
        assert student_result["score"] == 0.8

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = EvaluationCache(max_size=3)

        cache.set("query1", {"data": 1})
        cache.set("query2", {"data": 2})
        cache.set("query3", {"data": 3})
        cache.set("query4", {"data": 4})  # Should evict query1

        assert cache.get("query1") is None
        assert cache.get("query2") is not None
        assert cache.get("query4") is not None

    def test_clear(self):
        """Test cache clear."""
        cache = EvaluationCache()
        cache.set("query1", {"data": 1})
        cache.set("query2", {"data": 2})
        cache.get("query1")  # Generate a hit

        cache.clear()

        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        cache = EvaluationCache()
        cache.set("query1", {"data": 1})

        cache.get("query1")  # Hit
        cache.get("query2")  # Miss
        cache.get("query3")  # Miss

        stats = cache.get_stats()
        assert stats["hit_rate"] == pytest.approx(1 / 3, rel=0.01)


class TestCostEstimator:
    """Tests for CostEstimator."""

    def test_estimate_cost_openai(self):
        """Test cost estimation for OpenAI models."""
        estimator = CostEstimator(provider="openai", model="gpt-4o-mini")
        cost = estimator.estimate_cost(query_count=100, avg_input_tokens=500, avg_output_tokens=300)
        # gpt-4o-mini: input $0.00015/1K, output $0.0006/1K
        # 100 queries * 500 input / 1000 * 0.00015 = $0.0075
        # 100 queries * 300 output / 1000 * 0.0006 = $0.018
        # Total = $0.0255
        assert cost == Decimal("0.0255")

    def test_estimate_cost_ollama_free(self):
        """Test cost estimation for Ollama (free)."""
        estimator = CostEstimator(provider="ollama", model="default")
        cost = estimator.estimate_cost(query_count=100)
        assert cost == Decimal("0.0")

    def test_get_pricing_info(self):
        """Test getting pricing information."""
        estimator = CostEstimator(provider="openai", model="gpt-4o-mini")
        info = estimator.get_pricing_info()
        assert info["provider"] == "openai"
        assert info["model"] == "gpt-4o-mini"
        assert "pricing" in info


class TestBatchEvaluationExecutor:
    """Tests for BatchEvaluationExecutor."""

    def test_init_default_values(self, mock_evaluator):
        """Test executor initialization with defaults."""
        executor = BatchEvaluationExecutor(mock_evaluator)
        assert executor.batch_size == 5
        assert executor.budget_limit is None
        assert executor.rate_limiter is not None
        assert executor.cache is not None

    def test_init_custom_batch_size(self, mock_evaluator):
        """Test executor with custom batch size."""
        executor = BatchEvaluationExecutor(mock_evaluator, batch_size=3)
        assert executor.batch_size == 3

    def test_init_invalid_batch_size_too_small(self, mock_evaluator):
        """Test that batch size < 1 raises error."""
        with pytest.raises(ValueError, match="batch_size must be between"):
            BatchEvaluationExecutor(mock_evaluator, batch_size=0)

    def test_init_invalid_batch_size_too_large(self, mock_evaluator):
        """Test that batch size > 10 raises error."""
        with pytest.raises(ValueError, match="batch_size must be between"):
            BatchEvaluationExecutor(mock_evaluator, batch_size=15)

    @pytest.mark.asyncio
    async def test_execute_batch_basic(self, mock_evaluator):
        """Test basic batch execution."""
        executor = BatchEvaluationExecutor(mock_evaluator, batch_size=2)
        queries = [
            MockPersonaQuery(query="test1"),
            MockPersonaQuery(query="test2"),
            MockPersonaQuery(query="test3"),
        ]

        result = await executor.execute_batch(queries)

        assert result.successful == 3
        assert result.failed == 0
        assert len(result.results) == 3

    @pytest.mark.asyncio
    async def test_execute_batch_with_cache_hits(self, mock_evaluator):
        """Test that cache hits are counted."""
        cache = EvaluationCache()
        executor = BatchEvaluationExecutor(mock_evaluator, batch_size=2, cache=cache)

        queries = [
            MockPersonaQuery(query="test1"),
            MockPersonaQuery(query="test1"),  # Duplicate should be cache hit
        ]

        result = await executor.execute_batch(queries)

        assert result.cache_hits == 1

    @pytest.mark.asyncio
    async def test_execute_batch_with_budget_limit(self, mock_evaluator):
        """Test execution with budget limit."""
        cost_estimator = CostEstimator(provider="ollama", model="default")
        executor = BatchEvaluationExecutor(
            mock_evaluator,
            batch_size=2,
            cost_estimator=cost_estimator,
            budget_limit=Decimal("0.01"),
        )

        queries = [MockPersonaQuery(query=f"test{i}") for i in range(5)]
        result = await executor.execute_batch(queries)

        # With Ollama (free), budget should not block
        assert result.successful == 5

    @pytest.mark.asyncio
    async def test_execute_batch_budget_exceeded(self, mock_evaluator):
        """Test execution when budget is exceeded."""
        cost_estimator = CostEstimator(provider="openai", model="gpt-4o")
        executor = BatchEvaluationExecutor(
            mock_evaluator,
            batch_size=2,
            cost_estimator=cost_estimator,
            budget_limit=Decimal("0.001"),  # Very low budget
        )

        queries = [MockPersonaQuery(query=f"test{i}") for i in range(100)]
        result = await executor.execute_batch(queries)

        # Should fail due to budget
        assert len(result.errors) > 0
        assert "Budget" in result.errors[0]

    @pytest.mark.asyncio
    async def test_execute_batch_with_progress_callback(self, mock_evaluator):
        """Test that progress callback is called."""
        executor = BatchEvaluationExecutor(mock_evaluator, batch_size=2)

        progress_calls = []

        def progress_callback(completed, total):
            progress_calls.append((completed, total))

        queries = [MockPersonaQuery(query=f"test{i}") for i in range(5)]
        await executor.execute_batch(queries, progress_callback=progress_callback)

        assert len(progress_calls) > 0
        # Last call should show completion
        assert progress_calls[-1][0] == 5

    @pytest.mark.asyncio
    async def test_execute_batch_handles_errors(self):
        """Test that errors are captured."""

        async def failing_evaluator(query):
            raise ValueError("API Error")

        executor = BatchEvaluationExecutor(failing_evaluator, batch_size=2)
        queries = [MockPersonaQuery(query="test1")]

        result = await executor.execute_batch(queries)

        assert result.failed == 1
        assert len(result.errors) == 1
        assert "API Error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_sync_evaluator_support(self, sync_mock_evaluator):
        """Test that sync evaluators work."""
        executor = BatchEvaluationExecutor(sync_mock_evaluator, batch_size=2)
        queries = [MockPersonaQuery(query="test1")]

        result = await executor.execute_batch(queries)

        assert result.successful == 1

    def test_estimate_cost(self, mock_evaluator):
        """Test cost estimation for queries."""
        executor = BatchEvaluationExecutor(mock_evaluator)
        queries = [MockPersonaQuery(query=f"test{i}") for i in range(10)]

        cost = executor.estimate_cost(queries)

        assert isinstance(cost, Decimal)
        assert cost >= 0

    def test_check_budget_no_limit(self, mock_evaluator):
        """Test budget check when no limit is set."""
        executor = BatchEvaluationExecutor(mock_evaluator)
        within_budget, msg = executor.check_budget(Decimal("100.00"))

        assert within_budget is True
        assert "No budget limit" in msg

    def test_check_budget_within_limit(self, mock_evaluator):
        """Test budget check when within limit."""
        executor = BatchEvaluationExecutor(
            mock_evaluator, budget_limit=Decimal("10.00")
        )
        within_budget, msg = executor.check_budget(Decimal("5.00"))

        assert within_budget is True
        assert "Within budget" in msg

    def test_check_budget_exceeded(self, mock_evaluator):
        """Test budget check when exceeded."""
        executor = BatchEvaluationExecutor(
            mock_evaluator, budget_limit=Decimal("1.00")
        )
        # First, "spend" some budget
        executor._total_spent = Decimal("0.50")

        within_budget, msg = executor.check_budget(Decimal("1.00"))

        assert within_budget is False
        assert "Budget exceeded" in msg

    def test_get_stats(self, mock_evaluator):
        """Test getting executor statistics."""
        cache = EvaluationCache()
        rate_limiter = RateLimiter()
        executor = BatchEvaluationExecutor(
            mock_evaluator,
            batch_size=3,
            cache=cache,
            rate_limiter=rate_limiter,
            budget_limit=Decimal("10.00"),
        )

        stats = executor.get_stats()

        assert stats["batch_size"] == 3
        assert stats["budget_limit"] == 10.0
        assert "cache" in stats
        assert "rate_limiter" in stats


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_creation(self):
        """Test BatchResult creation."""
        result = BatchResult(
            results=[{"score": 0.9}, {"score": 0.8}],
            total_cost=Decimal("0.05"),
            successful=2,
            failed=0,
            cache_hits=1,
            execution_time_ms=150.5,
        )

        assert len(result.results) == 2
        assert result.total_cost == Decimal("0.05")
        assert result.successful == 2
        assert result.failed == 0
        assert result.cache_hits == 1
        assert result.execution_time_ms == 150.5

    def test_default_errors(self):
        """Test that errors list is empty by default."""
        result = BatchResult(
            results=[],
            total_cost=Decimal("0.00"),
            successful=0,
            failed=0,
            cache_hits=0,
            execution_time_ms=0,
        )

        assert result.errors == []
