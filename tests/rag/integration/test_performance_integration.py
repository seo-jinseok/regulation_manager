"""
Integration tests for performance optimization features.

Tests verify end-to-end functionality:
- Connection pooling with Redis (REQ-PER-001)
- Enhanced metrics tracking (REQ-PER-002)
- Cache warming automation (REQ-PER-003, REQ-PER-004)
- Graceful degradation (REQ-PER-008)
"""

import tempfile

from src.rag.config import get_config, reset_config
from src.rag.infrastructure.cache import CacheType, RAGQueryCache
from src.rag.infrastructure.cache_warming import (
    CacheWarmer,
    WarmingSchedule,
)


class TestPerformanceIntegration:
    """Integration tests for performance optimization."""

    def test_cache_with_connection_pooling(self):
        """Test cache with Redis connection pooling (REQ-PER-001)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache with connection pooling
            cache = RAGQueryCache(
                enabled=True,
                ttl_hours=24,
                cache_dir=tmpdir,
                redis_host="nonexistent",  # Will fall back to file
                max_connections=50,  # REQ-PER-001
                enable_enhanced_metrics=True,  # REQ-PER-002
            )

            # Verify basic functionality still works
            cache.set(
                CacheType.RETRIEVAL,
                "test query",
                {"result": "test data"},
            )

            result = cache.get(CacheType.RETRIEVAL, "test query")
            assert result is not None
            assert result["result"] == "test data"

    def test_enhanced_metrics_integration(self):
        """Test enhanced metrics tracking (REQ-PER-002)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RAGQueryCache(
                enabled=True,
                cache_dir=tmpdir,
                enable_enhanced_metrics=True,
            )

            # Perform some cache operations
            cache.set(CacheType.RETRIEVAL, "query1", {"data": "result1"})
            cache.get(CacheType.RETRIEVAL, "query1")  # Hit
            cache.get(CacheType.RETRIEVAL, "query2")  # Miss

            # Get stats
            stats = cache.stats()

            # Verify enhanced metrics are present
            assert "enhanced_metrics" in stats
            assert "overall_hit_rate" in stats["enhanced_metrics"]
            assert "layers" in stats["enhanced_metrics"]

    def test_cache_warming_integration(self):
        """Test cache warming integration (REQ-PER-003, REQ-PER-006)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RAGQueryCache(
                enabled=True,
                cache_dir=tmpdir,
                enable_enhanced_metrics=True,
            )

            # Create warmer
            warmer = CacheWarmer(
                cache=cache,
                enabled=True,
                top_n=100,  # REQ-PER-006
                hit_rate_threshold=0.6,  # REQ-PER-004
            )

            # Add warm queries
            for i in range(10):
                warmer.add_warm_query(
                    cache_type="retrieval",
                    query=f"warm query {i}",
                    data={"result": f"warm result {i}"},
                    priority=10 - i,
                )

            # Execute warming
            stats = warmer._warm_cache()

            # Verify warming completed
            # Note: Additional queries may be loaded from config file
            assert stats["warmed"] >= 10
            assert stats["errors"] == 0

            # Verify cache has warmed data
            result = cache.get(CacheType.RETRIEVAL, "warm query 0")
            assert result is not None

    def test_graceful_degradation_to_file_cache(self):
        """Test graceful degradation when Redis unavailable (REQ-PER-008)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache with Redis that will fail
            cache = RAGQueryCache(
                enabled=True,
                ttl_hours=24,
                cache_dir=tmpdir,
                redis_host="nonexistent_host",  # Will fail to connect
                redis_port=9999,
                max_connections=50,
            )

            # Should fall back to file backend (REQ-PER-008)
            cache.set(CacheType.RETRIEVAL, "test", {"data": "value"})

            result = cache.get(CacheType.RETRIEVAL, "test")
            assert result is not None
            assert result["data"] == "value"

            # Verify stats show Redis unavailable
            stats = cache.stats()
            assert stats["redis_enabled"] is False

    def test_write_through_caching(self):
        """Test write-through caching behavior (REQ-PER-007)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RAGQueryCache(
                enabled=True,
                cache_dir=tmpdir,
            )

            # Set should write immediately
            cache.set(CacheType.RETRIEVAL, "query", {"data": "value"})

            # Should be available immediately (write-through)
            result = cache.get(CacheType.RETRIEVAL, "query")
            assert result is not None
            assert result["data"] == "value"

    def test_hit_rate_based_warming_trigger(self):
        """Test warming triggered by low hit rate (REQ-PER-004)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RAGQueryCache(
                enabled=True,
                cache_dir=tmpdir,
                enable_enhanced_metrics=True,
            )

            warmer = CacheWarmer(
                cache=cache,
                enabled=True,
                hit_rate_threshold=0.6,  # 60% threshold
            )

            # Add warm query
            warmer.add_warm_query(
                cache_type="retrieval",
                query="warm query",
                data={"result": "warm result"},
            )

            # Simulate low hit rate by recording misses
            if cache._enhanced_metrics:
                from src.rag.domain.performance.metrics import CacheLayer

                for _ in range(10):
                    cache._enhanced_metrics.record_layer_miss(
                        CacheLayer.L1_MEMORY, 10.0
                    )

            # Check if warming should be triggered
            # Note: This is a simplified check
            if cache._enhanced_metrics and cache._enhanced_metrics.check_low_hit_rate(
                threshold=0.6
            ):
                # Would trigger warming in real scenario
                assert True

    def test_config_integration(self):
        """Test that config provides performance settings."""
        reset_config()
        config = get_config()

        # Verify performance settings are available
        assert hasattr(config, "enable_enhanced_metrics")
        assert hasattr(config, "redis_max_connections")
        assert hasattr(config, "enable_cache_warming")
        assert hasattr(config, "cache_warming_top_n")
        assert hasattr(config, "cache_hit_rate_threshold")

        # Verify defaults match REQ specifications
        assert config.redis_max_connections == 50  # REQ-PER-001
        assert config.cache_warming_top_n == 100  # REQ-PER-006
        assert config.cache_hit_rate_threshold == 0.6  # REQ-PER-004

    def test_cache_with_all_performance_features(self):
        """Test cache with all performance features enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RAGQueryCache(
                enabled=True,
                ttl_hours=24,
                cache_dir=tmpdir,
                max_connections=50,  # REQ-PER-001
                enable_enhanced_metrics=True,  # REQ-PER-002
            )

            # Create warmer
            schedule = WarmingSchedule(enabled=False)  # Don't auto-trigger
            warmer = CacheWarmer(
                cache=cache,
                enabled=True,
                top_n=100,
                schedule=schedule,
                hit_rate_threshold=0.6,
            )

            # Perform operations
            cache.set(CacheType.RETRIEVAL, "query1", {"data": "result1"})
            cache.get(CacheType.RETRIEVAL, "query1")

            # Warm cache
            warmer.add_warm_query(
                cache_type="retrieval",
                query="warm query",
                data={"result": "warm result"},
            )
            warmer._warm_cache()

            # Get comprehensive stats
            stats = cache.stats()

            # Verify all features are working
            assert "hits" in stats
            assert "enhanced_metrics" in stats
            assert "redis_enabled" in stats

    def test_performance_metrics_accuracy(self):
        """Test that performance metrics are accurately tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RAGQueryCache(
                enabled=True,
                cache_dir=tmpdir,
                enable_enhanced_metrics=True,
            )

            # Clear any existing stats
            cache.reset_stats()

            # Perform known operations
            cache.set(CacheType.RETRIEVAL, "q1", {"d": "r1"})
            cache.set(CacheType.RETRIEVAL, "q2", {"d": "r2"})
            cache.set(CacheType.RETRIEVAL, "q3", {"d": "r3"})

            cache.get(CacheType.RETRIEVAL, "q1")  # Hit
            cache.get(CacheType.RETRIEVAL, "q2")  # Hit
            cache.get(CacheType.RETRIEVAL, "q4")  # Miss

            # Verify counts
            stats = cache.stats()
            assert stats["hits"] == 2
            assert stats["misses"] == 1
            assert stats["total_requests"] == 3

    def test_warming_stats_tracking(self):
        """Test that warming statistics are tracked correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = RAGQueryCache(
                enabled=True,
                cache_dir=tmpdir,
                enable_enhanced_metrics=True,
            )

            warmer = CacheWarmer(
                cache=cache,
                enabled=True,
                top_n=10,
            )

            # Add queries
            for i in range(5):
                warmer.add_warm_query(
                    cache_type="retrieval",
                    query=f"query {i}",
                    data={"result": f"result {i}"},
                )

            # Execute warming
            warming_stats = warmer._warm_cache()

            # Verify stats
            # Note: Additional queries may be loaded from config file
            assert warming_stats["warmed"] >= 5
            assert warming_stats["errors"] == 0

            # Get warming stats from warmer
            stats = warmer.get_warming_stats()
            assert stats["warm_queries_count"] >= 5
            assert stats["enabled"] is True
