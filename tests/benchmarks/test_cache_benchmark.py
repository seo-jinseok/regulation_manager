"""
Performance Benchmarks for LLM Cache.

Measures:
- Cache hit/miss rates
- Hash computation performance
- Cache expiration overhead
- Memory usage patterns
"""

import time

import pytest

# Check if pytest-benchmark is available
HAS_BENCHMARK = False
try:
    __import__("pytest_benchmark")
    HAS_BENCHMARK = True
except ImportError:
    pass


# Provide a dummy benchmark fixture when pytest-benchmark is not available
if not HAS_BENCHMARK:

    @pytest.fixture
    def benchmark(request, *args, **kwargs):  # noqa: ARG001
        """Dummy benchmark fixture when pytest-benchmark is not available."""

        class DummyBenchmark:
            def __call__(self, func, *args, **kwargs):
                # Run function and return result
                return func(*args, **kwargs)

        return DummyBenchmark()


@pytest.mark.benchmark
class TestCacheEffectiveness:
    """Benchmark cache effectiveness metrics."""

    def test_cache_hit_rate_after_warmup(self, tmp_path):
        """
        Benchmark cache hit rate after warmup.

        Simulates repeated queries to measure cache effectiveness.
        """
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path / "cache"), ttl_days=30)

        # Warmup: cache some responses
        queries = [
            ("system_prompt", f"query_{i}", "gpt-4o-mini", f"response_{i}")
            for i in range(10)
        ]

        for sp, q, m, r in queries:
            cache.set(sp, q, m, r)

        # Measure hit rate
        hits = 0
        total = 20

        for i in range(total):
            # 50% should be cached
            if i < 10:
                result = cache.get("system_prompt", f"query_{i}", "gpt-4o-mini")
                if result is not None:
                    hits += 1
            else:
                result = cache.get("system_prompt", f"new_query_{i}", "gpt-4o-mini")

        hit_rate = hits / total
        # Should have at least 45% hit rate (allowing for some variance)
        assert hit_rate >= 0.45, f"Hit rate {hit_rate:.2%} too low, expected >= 45%"

    def test_cache_miss_latency(self, tmp_path, benchmark=None):
        """Benchmark cache miss latency."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path / "cache"), ttl_days=30)

        def cache_miss():
            return cache.get("nonexistent_prompt", "nonexistent_query", "model")

        if HAS_BENCHMARK and benchmark is not None:
            result = benchmark(cache_miss)
        else:
            result = cache_miss()
        assert result is None

    def test_cache_hit_latency(self, tmp_path, benchmark=None):
        """Benchmark cache hit latency."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path / "cache"), ttl_days=30)
        cache.set("test_prompt", "test_query", "test_model", "test_response")

        def cache_hit():
            return cache.get("test_prompt", "test_query", "test_model")

        if HAS_BENCHMARK and benchmark is not None:
            result = benchmark(cache_hit)
        else:
            result = cache_hit()
        assert result == "test_response"

    def test_cache_write_latency(self, tmp_path, benchmark=None):
        """Benchmark cache write latency."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path / "cache"), ttl_days=30)

        def cache_write():
            cache.set(f"prompt_{time.time()}", "query", "model", "response")

        if HAS_BENCHMARK and benchmark is not None:
            benchmark(cache_write)
        else:
            cache_write()

    def test_cache_cleanup_performance(self, tmp_path, benchmark=None):
        """Benchmark cache cleanup of expired entries."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path / "cache"), ttl_days=0)

        # Add some entries
        for i in range(100):
            cache.set(f"prompt_{i}", f"query_{i}", "model", f"response_{i}")

        # Manually expire them
        for key in cache._index:
            cache._index[key]["timestamp"] = time.time() - 3600

        def cleanup():
            return cache.clear_expired()

        if HAS_BENCHMARK and benchmark is not None:
            removed = benchmark(cleanup)
        else:
            removed = cleanup()
        assert removed > 0


@pytest.mark.benchmark
class TestCacheScalability:
    """Test cache performance at different scales."""

    def test_small_cache_performance(self, tmp_path):
        """Test cache with small number of entries (100)."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(
            cache_dir=str(tmp_path / "cache_small"), ttl_days=30, max_entries=100
        )

        # Add entries
        for i in range(100):
            cache.set(f"prompt_{i}", f"query_{i}", "model", f"response_{i}")

        # Measure read performance
        start = time.time()
        for i in range(100):
            cache.get(f"prompt_{i}", f"query_{i}", "model")
        read_time = time.time() - start

        # Should complete quickly
        assert read_time < 1.0, f"Small cache read took {read_time:.3f}s"

    def test_medium_cache_performance(self, tmp_path):
        """Test cache with medium number of entries (1000)."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(
            cache_dir=str(tmp_path / "cache_medium"), ttl_days=30, max_entries=1000
        )

        # Add entries
        for i in range(500):  # Add 500 to test performance
            cache.set(f"prompt_{i}", f"query_{i}", "model", f"response_{i}")

        # Measure read performance
        start = time.time()
        for i in range(500):
            cache.get(f"prompt_{i}", f"query_{i}", "model")
        read_time = time.time() - start

        # Should still be reasonable
        assert read_time < 2.0, f"Medium cache read took {read_time:.3f}s"

    def test_cache_overflow_handling(self, tmp_path):
        """Test cache handling when exceeding max_entries."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(
            cache_dir=str(tmp_path / "cache_overflow"), ttl_days=30, max_entries=10
        )

        # Add more entries than max
        for i in range(20):
            cache.set(f"prompt_{i}", f"query_{i}", "model", f"response_{i}")

        # Should not exceed max significantly
        stats = cache.stats()
        assert stats["total_entries"] <= 15  # Allow some buffer


@pytest.mark.benchmark
class TestP2Optimizations:
    """Verify P2 performance improvements in caching."""

    def test_cache_effectiveness_after_p2(self, tmp_path):
        """
        Test that P2 caching improvements are effective.

        P2 introduced various caching optimizations - this verifies
        they're working as expected.
        """
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path / "cache_p2"), ttl_days=30)

        # Simulate typical usage pattern
        unique_queries = 20
        repeated_queries = 80

        # Unique queries
        for i in range(unique_queries):
            cache.set("system", f"unique_query_{i}", "model", f"response_{i}")

        # Repeated queries (should hit cache)
        hits = 0
        for i in range(repeated_queries):
            query_idx = i % unique_queries
            result = cache.get("system", f"unique_query_{query_idx}", "model")
            if result is not None:
                hits += 1

        hit_rate = hits / repeated_queries

        # Should have high hit rate after warmup
        assert hit_rate > 0.8, f"P2 cache hit rate {hit_rate:.2%} too low"

    def test_hash_consistency(self, tmp_path):
        """Test that hash computation is consistent (important for caching)."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path / "cache_hash"), ttl_days=30)

        # Same inputs should produce same hash
        hash1 = cache._compute_hash("system", "query", "model")
        hash2 = cache._compute_hash("system", "query", "model")

        assert hash1 == hash2, "Hash computation should be deterministic"

        # Different inputs should produce different hashes
        hash3 = cache._compute_hash("system", "different_query", "model")
        assert hash1 != hash3, "Different inputs should produce different hashes"
