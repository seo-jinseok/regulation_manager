"""
Batch Evaluation Executor for RAG Quality Assessment.

Implements efficient parallel evaluation with API budget awareness,
rate limiting, and result caching as specified in SPEC-RAG-EVAL-001.

Features:
- Configurable batch sizes (1-10 queries per batch)
- Rate limiting per API provider
- LRU caching for repeated query evaluations
- Cost estimation before execution
- Graceful degradation on budget limits
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    tokens_per_minute: int = 90000
    min_request_interval_ms: int = 1000  # Minimum time between requests


@dataclass
class RateLimitStats:
    """Statistics for rate limiting."""

    requests_made: int = 0
    tokens_used: int = 0
    last_request_time: Optional[float] = None
    current_minute_start: Optional[float] = None
    requests_this_minute: int = 0
    tokens_this_minute: int = 0


@dataclass
class CacheEntry:
    """Entry in the evaluation cache."""

    result: Any
    timestamp: datetime
    ttl_seconds: int = 86400  # 24 hours default

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() - self.timestamp > timedelta(seconds=self.ttl_seconds)


@dataclass
class BatchResult:
    """Result of a batch evaluation."""

    results: List[Any]
    total_cost: Decimal
    successful: int
    failed: int
    cache_hits: int
    execution_time_ms: float
    errors: List[str] = field(default_factory=list)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Implements per-provider rate limiting with configurable thresholds.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize the rate limiter.

        Args:
            config: Rate limit configuration (uses defaults if not provided)
        """
        self.config = config or RateLimitConfig()
        self.stats = RateLimitStats()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 1000) -> float:
        """
        Acquire permission to make an API call.

        Implements token bucket algorithm with smooth limiting.

        Args:
            estimated_tokens: Estimated tokens for this request

        Returns:
            Time to wait in seconds (0 if no wait needed)
        """
        async with self._lock:
            now = time.time()

            # Initialize minute tracking
            if self.stats.current_minute_start is None:
                self.stats.current_minute_start = now

            # Reset counters if new minute
            if now - self.stats.current_minute_start >= 60:
                self.stats.current_minute_start = now
                self.stats.requests_this_minute = 0
                self.stats.tokens_this_minute = 0

            wait_time = 0.0

            # Check requests per minute limit
            if self.stats.requests_this_minute >= self.config.requests_per_minute:
                wait_time = max(wait_time, 60 - (now - self.stats.current_minute_start))

            # Check tokens per minute limit
            if (
                self.stats.tokens_this_minute + estimated_tokens
                > self.config.tokens_per_minute
            ):
                wait_time = max(wait_time, 60 - (now - self.stats.current_minute_start))

            # Check minimum interval between requests
            if self.stats.last_request_time is not None:
                time_since_last = (now - self.stats.last_request_time) * 1000
                if time_since_last < self.config.min_request_interval_ms:
                    wait_time = max(
                        wait_time,
                        (self.config.min_request_interval_ms - time_since_last) / 1000,
                    )

            # Update stats after acquiring
            self.stats.last_request_time = now
            self.stats.requests_made += 1
            self.stats.requests_this_minute += 1
            self.stats.tokens_used += estimated_tokens
            self.stats.tokens_this_minute += estimated_tokens

            return wait_time

    def get_stats(self) -> RateLimitStats:
        """Get current rate limit statistics."""
        return self.stats


class EvaluationCache:
    """
    LRU cache for evaluation results.

    Caches repeated query evaluations with TTL-based expiration.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 86400):
        """
        Initialize the evaluation cache.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._hits = 0
        self._misses = 0

    def _generate_key(
        self,
        query: str,
        persona: Optional[str] = None,
        context_hash: Optional[str] = None,
    ) -> str:
        """Generate cache key from query and optional parameters."""
        key_parts = [query]
        if persona:
            key_parts.append(f"persona:{persona}")
        if context_hash:
            key_parts.append(f"context:{context_hash}")

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def get(
        self,
        query: str,
        persona: Optional[str] = None,
        context_hash: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Get cached result if available and not expired.

        Args:
            query: Query string
            persona: Optional persona identifier
            context_hash: Optional hash of context documents

        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(query, persona, context_hash)
        entry = self._cache.get(key)

        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired():
            # Remove expired entry
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self._misses += 1
            return None

        # Update access order for LRU
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        self._hits += 1

        return entry.result

    def set(
        self,
        query: str,
        result: Any,
        persona: Optional[str] = None,
        context_hash: Optional[str] = None,
    ) -> None:
        """
        Cache a result.

        Args:
            query: Query string
            result: Result to cache
            persona: Optional persona identifier
            context_hash: Optional hash of context documents
        """
        key = self._generate_key(query, persona, context_hash)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)

        self._cache[key] = CacheEntry(
            result=result, timestamp=datetime.now(), ttl_seconds=self.ttl_seconds
        )
        self._access_order.append(key)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses)
            if (self._hits + self._misses) > 0
            else 0,
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0


class CostEstimator:
    """
    Estimates API costs for evaluation runs.

    Supports multiple providers with configurable pricing.
    """

    # Pricing per 1K tokens (as of 2025)
    PRICING = {
        "openai": {
            "gpt-4o": {"input": 0.0025, "output": 0.01},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        },
        "gemini": {
            "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
            "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        },
        "ollama": {  # Local, no cost
            "default": {"input": 0.0, "output": 0.0}
        },
    }

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        """
        Initialize the cost estimator.

        Args:
            provider: API provider name
            model: Model name
        """
        self.provider = provider
        self.model = model

    def estimate_cost(
        self,
        query_count: int,
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 300,
    ) -> Decimal:
        """
        Estimate total cost for evaluation run.

        Args:
            query_count: Number of queries to evaluate
            avg_input_tokens: Average input tokens per query
            avg_output_tokens: Average output tokens per query

        Returns:
            Estimated cost in USD
        """
        provider_pricing = self.PRICING.get(self.provider, {})
        model_pricing = provider_pricing.get(self.model, {"input": 0.0, "output": 0.0})

        input_cost = (
            Decimal(str(model_pricing["input"]))
            * avg_input_tokens
            * query_count
            / 1000
        )
        output_cost = (
            Decimal(str(model_pricing["output"]))
            * avg_output_tokens
            * query_count
            / 1000
        )

        return input_cost + output_cost

    def get_pricing_info(self) -> Dict[str, Any]:
        """Get pricing information for current provider/model."""
        provider_pricing = self.PRICING.get(self.provider, {})
        return {
            "provider": self.provider,
            "model": self.model,
            "pricing": provider_pricing.get(self.model, {"input": 0.0, "output": 0.0}),
        }


class BatchEvaluationExecutor:
    """
    Executes evaluations in batches with rate limiting and caching.

    Features:
    - Configurable batch sizes (1-10 queries per batch)
    - Rate limiting per API provider
    - LRU caching for repeated evaluations
    - Cost estimation before execution
    - Budget limits with graceful degradation
    """

    # Constraints from SPEC-RAG-EVAL-001
    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 10
    DEFAULT_BATCH_SIZE = 5

    def __init__(
        self,
        evaluator: Callable,
        batch_size: int = DEFAULT_BATCH_SIZE,
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[EvaluationCache] = None,
        cost_estimator: Optional[CostEstimator] = None,
        budget_limit: Optional[Decimal] = None,
    ):
        """
        Initialize the batch executor.

        Args:
            evaluator: Callable that evaluates a single query
            batch_size: Number of queries per batch (1-10)
            rate_limiter: Optional rate limiter instance
            cache: Optional evaluation cache instance
            cost_estimator: Optional cost estimator instance
            budget_limit: Optional budget limit in USD
        """
        if not self.MIN_BATCH_SIZE <= batch_size <= self.MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size must be between {self.MIN_BATCH_SIZE} and "
                f"{self.MAX_BATCH_SIZE}, got {batch_size}"
            )

        self.evaluator = evaluator
        self.batch_size = batch_size
        self.rate_limiter = rate_limiter or RateLimiter()
        self.cache = cache or EvaluationCache()
        self.cost_estimator = cost_estimator or CostEstimator()
        self.budget_limit = budget_limit

        self._total_spent = Decimal("0.00")

    def estimate_cost(self, queries: List[Any]) -> Decimal:
        """
        Estimate total cost for evaluating a list of queries.

        Args:
            queries: List of queries to evaluate

        Returns:
            Estimated cost in USD
        """
        return self.cost_estimator.estimate_cost(len(queries))

    def check_budget(self, estimated_cost: Decimal) -> Tuple[bool, str]:
        """
        Check if estimated cost is within budget.

        Args:
            estimated_cost: Estimated cost in USD

        Returns:
            Tuple of (is_within_budget, message)
        """
        if self.budget_limit is None:
            return True, "No budget limit set"

        if self._total_spent + estimated_cost > self.budget_limit:
            remaining = self.budget_limit - self._total_spent
            return False, f"Budget exceeded. Remaining: ${remaining:.2f}"

        return True, f"Within budget. Remaining after: ${self.budget_limit - self._total_spent - estimated_cost:.2f}"

    async def execute_batch(
        self,
        queries: List[Any],
        persona: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """
        Execute evaluations in batches.

        Args:
            queries: List of PersonaQuery objects to evaluate
            persona: Optional persona identifier for caching
            progress_callback: Optional callback(completed, total) for progress

        Returns:
            BatchResult with all evaluation results
        """
        start_time = time.time()
        results = []
        errors = []
        successful = 0
        failed = 0
        cache_hits = 0
        total_cost = Decimal("0.00")

        # Estimate cost and check budget
        estimated_cost = self.estimate_cost(queries)
        within_budget, budget_msg = self.check_budget(estimated_cost)

        if not within_budget:
            logger.warning(f"Budget check failed: {budget_msg}")
            return BatchResult(
                results=[],
                total_cost=Decimal("0.00"),
                successful=0,
                failed=len(queries),
                cache_hits=0,
                execution_time_ms=0,
                errors=[budget_msg],
            )

        # Process queries in batches
        total_queries = len(queries)
        completed = 0

        for batch_start in range(0, total_queries, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_queries)
            batch = queries[batch_start:batch_end]

            # Evaluate each query in the batch
            batch_results = await self._evaluate_batch(
                batch, persona, cache_hits, total_cost, errors
            )

            for result in batch_results:
                if result is not None:
                    if isinstance(result, tuple) and len(result) == 2:
                        # Tuple indicates (result, was_cache_hit)
                        results.append(result[0])
                        if result[1]:
                            cache_hits += 1
                        successful += 1
                    else:
                        results.append(result)
                        successful += 1
                else:
                    failed += 1

            completed = batch_end
            if progress_callback:
                progress_callback(completed, total_queries)

        execution_time_ms = (time.time() - start_time) * 1000

        # Update total spent
        self._total_spent += total_cost

        return BatchResult(
            results=results,
            total_cost=total_cost,
            successful=successful,
            failed=failed,
            cache_hits=cache_hits,
            execution_time_ms=execution_time_ms,
            errors=errors,
        )

    async def _evaluate_batch(
        self,
        batch: List[Any],
        persona: Optional[str],
        cache_hits: int,
        total_cost: Decimal,
        errors: List[str],
    ) -> List[Optional[Tuple[Any, bool]]]:
        """
        Evaluate a single batch of queries.

        Args:
            batch: List of queries in this batch
            persona: Optional persona identifier
            cache_hits: Running count of cache hits
            total_cost: Running total cost
            errors: List to append errors to

        Returns:
            List of (result, was_cache_hit) tuples or None for failures
        """
        results = []

        for query_obj in batch:
            try:
                # Extract query text
                query_text = (
                    query_obj.query
                    if hasattr(query_obj, "query")
                    else str(query_obj)
                )

                # Check cache first
                cached_result = self.cache.get(query_text, persona)
                if cached_result is not None:
                    results.append((cached_result, True))
                    logger.debug(f"Cache hit for query: {query_text[:50]}...")
                    continue

                # Apply rate limiting
                wait_time = await self.rate_limiter.acquire()
                if wait_time > 0:
                    logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)

                # Evaluate
                result = await self._evaluate_single(query_obj)

                # Cache the result
                self.cache.set(query_text, result, persona)
                results.append((result, False))

            except Exception as e:
                error_msg = f"Error evaluating query: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                results.append(None)

        return results

    async def _evaluate_single(self, query_obj: Any) -> Any:
        """
        Evaluate a single query.

        Handles both sync and async evaluators.

        Args:
            query_obj: Query object to evaluate

        Returns:
            Evaluation result
        """
        if asyncio.iscoroutinefunction(self.evaluator):
            return await self.evaluator(query_obj)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.evaluator, query_obj)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get executor statistics.

        Returns:
            Dictionary with rate limiter, cache, and budget stats
        """
        return {
            "batch_size": self.batch_size,
            "total_spent": float(self._total_spent),
            "budget_limit": float(self.budget_limit) if self.budget_limit else None,
            "rate_limiter": {
                "requests_made": self.rate_limiter.stats.requests_made,
                "tokens_used": self.rate_limiter.stats.tokens_used,
            },
            "cache": self.cache.get_stats(),
        }
