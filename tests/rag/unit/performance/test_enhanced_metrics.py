"""
Tests for enhanced cache metrics (REQ-PER-002).

Tests verify:
- Per-layer metrics tracking
- Connection pool statistics
- Cache warming metrics
- Overall health indicators
"""


from src.rag.domain.performance.metrics import (
    CacheLayer,
    CacheWarmingMetrics,
    ConnectionPoolMetrics,
    EnhancedCacheMetrics,
    LayerMetrics,
)


class TestLayerMetrics:
    """Tests for LayerMetrics."""

    def test_initialization(self):
        """Test LayerMetrics initialization."""
        metrics = LayerMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.errors == 0
        assert metrics.total_requests == 0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        metrics = LayerMetrics()
        metrics.record_hit(10.0)
        metrics.record_hit(15.0)
        metrics.record_hit(20.0)
        metrics.record_miss(5.0)

        assert metrics.total_requests == 4
        assert metrics.hit_rate == 0.75
        assert metrics.miss_rate == 0.25

    def test_hit_rate_no_requests(self):
        """Test hit rate with no requests."""
        metrics = LayerMetrics()
        assert metrics.hit_rate == 0.0
        assert metrics.miss_rate == 0.0

    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        metrics = LayerMetrics()

        # Record 100 latencies from 1ms to 100ms
        for i in range(1, 101):
            metrics.record_hit(float(i))

        assert metrics.p50_latency is not None
        assert 49 <= metrics.p50_latency <= 51  # Around 50ms
        assert metrics.p95_latency is not None
        assert 94 <= metrics.p95_latency <= 96  # Around 95ms
        assert metrics.p99_latency is not None
        assert 98 <= metrics.p99_latency <= 100  # Around 99ms

    def test_latency_with_no_samples(self):
        """Test latency with no samples."""
        metrics = LayerMetrics()
        assert metrics.p50_latency is None
        assert metrics.p95_latency is None
        assert metrics.p99_latency is None

    def test_max_latency_samples(self):
        """Test that latency samples don't exceed max_samples."""
        metrics = LayerMetrics(max_latency_samples=10)

        # Record 20 latencies
        for i in range(20):
            metrics.record_hit(float(i))

        # Should only keep last 10
        assert len(metrics.latencies) == 10
        assert list(range(10, 20)) == [int(l) for l in metrics.latencies]

    def test_record_eviction(self):
        """Test recording evictions."""
        metrics = LayerMetrics()
        metrics.record_eviction()
        metrics.record_eviction()
        assert metrics.evictions == 2

    def test_record_error(self):
        """Test recording errors."""
        metrics = LayerMetrics()
        metrics.record_error()
        metrics.record_error()
        metrics.record_error()
        assert metrics.errors == 3

    def test_reset(self):
        """Test resetting metrics."""
        metrics = LayerMetrics()
        metrics.record_hit(10.0)
        metrics.record_miss(5.0)
        metrics.record_eviction()
        metrics.record_error()

        metrics.reset()

        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.evictions == 0
        assert metrics.errors == 0
        assert len(metrics.latencies) == 0

    def test_to_dict(self):
        """Test serialization to dict."""
        metrics = LayerMetrics()
        metrics.record_hit(10.0)
        metrics.record_miss(5.0)

        d = metrics.to_dict()
        assert d["hits"] == 1
        assert d["misses"] == 1
        assert d["total_requests"] == 2
        assert d["hit_rate"] == "50.00%"
        assert d["miss_rate"] == "50.00%"
        assert "p50_latency_ms" in d


class TestConnectionPoolMetrics:
    """Tests for ConnectionPoolMetrics."""

    def test_initialization(self):
        """Test ConnectionPoolMetrics initialization."""
        metrics = ConnectionPoolMetrics()
        assert metrics.total_connections == 0
        assert metrics.active_connections == 0
        assert metrics.idle_connections == 0
        assert metrics.failed_connections == 0

    def test_pool_utilization(self):
        """Test pool utilization calculation."""
        metrics = ConnectionPoolMetrics(max_pool_size=100)
        metrics.total_connections = 50
        assert metrics.pool_utilization == 0.5

    def test_is_exhausted(self):
        """Test pool exhaustion detection."""
        metrics = ConnectionPoolMetrics(max_pool_size=10)
        metrics.active_connections = 9
        assert metrics.is_exhausted is False

        metrics.active_connections = 10
        assert metrics.is_exhausted is True

    def test_to_dict(self):
        """Test serialization to dict."""
        metrics = ConnectionPoolMetrics(max_pool_size=50)
        metrics.total_connections = 25
        metrics.active_connections = 10
        metrics.idle_connections = 15

        d = metrics.to_dict()
        assert d["total_connections"] == 25
        assert d["active_connections"] == 10
        assert d["idle_connections"] == 15
        assert d["max_pool_size"] == 50
        assert d["pool_utilization"] == "50.00%"


class TestCacheWarmingMetrics:
    """Tests for CacheWarmingMetrics."""

    def test_initialization(self):
        """Test CacheWarmingMetrics initialization."""
        metrics = CacheWarmingMetrics()
        assert metrics.warming_jobs_completed == 0
        assert metrics.warming_jobs_failed == 0
        assert metrics.total_entries_warmed == 0

    def test_record_warming_complete(self):
        """Test recording successful warming."""
        metrics = CacheWarmingMetrics()
        metrics.record_warming_start()
        import time

        time.sleep(0.01)  # Small delay
        metrics.record_warming_complete(entries_count=100)

        assert metrics.warming_jobs_completed == 1
        assert metrics.total_entries_warmed == 100
        assert metrics.last_warming_timestamp is not None
        assert metrics.last_warming_duration_ms is not None
        assert metrics.last_warming_duration_ms > 0

    def test_record_warming_failure(self):
        """Test recording failed warming."""
        metrics = CacheWarmingMetrics()
        metrics.record_warming_start()
        metrics.record_warming_failure()

        assert metrics.warming_jobs_failed == 1
        assert metrics.last_warming_duration_ms is None

    def test_avg_warming_time(self):
        """Test average warming time calculation."""
        metrics = CacheWarmingMetrics()

        # First job: 100ms
        metrics.record_warming_start()
        metrics._current_warming_start = time.time() - 0.1  # Simulate 100ms
        metrics.record_warming_complete(50)

        # Second job: 200ms
        metrics.record_warming_start()
        metrics._current_warming_start = time.time() - 0.2  # Simulate 200ms
        metrics.record_warming_complete(50)

        assert metrics.avg_warming_time_ms == 150.0

    def test_success_rate(self):
        """Test success rate calculation."""
        metrics = CacheWarmingMetrics()
        metrics.record_warming_start()
        metrics.record_warming_complete(100)

        metrics.record_warming_start()
        metrics.record_warming_failure()

        metrics.record_warming_start()
        metrics.record_warming_complete(100)

        assert metrics.warming_jobs_completed == 2
        assert metrics.warming_jobs_failed == 1
        assert metrics.success_rate == 2 / 3

    def test_to_dict(self):
        """Test serialization to dict."""
        metrics = CacheWarmingMetrics()
        metrics.record_warming_start()
        metrics.record_warming_complete(100)

        d = metrics.to_dict()
        assert d["warming_jobs_completed"] == 1
        assert d["total_entries_warmed"] == 100
        assert "last_warming_timestamp" in d


class TestEnhancedCacheMetrics:
    """Tests for EnhancedCacheMetrics."""

    def test_initialization(self):
        """Test EnhancedCacheMetrics initialization."""
        metrics = EnhancedCacheMetrics()

        # All layers should be initialized
        assert CacheLayer.L1_MEMORY in metrics._layers
        assert CacheLayer.L2_REDIS in metrics._layers
        assert CacheLayer.L3_CHROMA in metrics._layers

        # Connection pool and warming metrics should exist
        assert metrics._connection_pool is not None
        assert metrics._warming is not None

    def test_get_layer_metrics(self):
        """Test getting metrics for specific layer."""
        metrics = EnhancedCacheMetrics()
        l1_metrics = metrics.get_layer_metrics(CacheLayer.L1_MEMORY)

        assert isinstance(l1_metrics, LayerMetrics)
        assert l1_metrics.hits == 0
        assert l1_metrics.misses == 0

    def test_record_layer_hit(self):
        """Test recording hit for a layer."""
        metrics = EnhancedCacheMetrics()
        metrics.record_layer_hit(CacheLayer.L2_REDIS, latency_ms=15.0)

        l2_metrics = metrics.get_layer_metrics(CacheLayer.L2_REDIS)
        assert l2_metrics.hits == 1
        assert l2_metrics.total_requests == 1

    def test_record_layer_miss(self):
        """Test recording miss for a layer."""
        metrics = EnhancedCacheMetrics()
        metrics.record_layer_miss(CacheLayer.L1_MEMORY, latency_ms=5.0)

        l1_metrics = metrics.get_layer_metrics(CacheLayer.L1_MEMORY)
        assert l1_metrics.misses == 1

    def test_record_layer_eviction(self):
        """Test recording eviction for a layer."""
        metrics = EnhancedCacheMetrics()
        metrics.record_layer_eviction(CacheLayer.L3_CHROMA)

        l3_metrics = metrics.get_layer_metrics(CacheLayer.L3_CHROMA)
        assert l3_metrics.evictions == 1

    def test_record_layer_error(self):
        """Test recording error for a layer."""
        metrics = EnhancedCacheMetrics()
        metrics.record_layer_error(CacheLayer.L2_REDIS)

        l2_metrics = metrics.get_layer_metrics(CacheLayer.L2_REDIS)
        assert l2_metrics.errors == 1

    def test_get_overall_hit_rate(self):
        """Test overall hit rate calculation across all layers."""
        metrics = EnhancedCacheMetrics()

        # Record some hits and misses across layers
        metrics.record_layer_hit(CacheLayer.L1_MEMORY, 5.0)
        metrics.record_layer_hit(CacheLayer.L1_MEMORY, 5.0)
        metrics.record_layer_hit(CacheLayer.L2_REDIS, 10.0)
        metrics.record_layer_miss(CacheLayer.L1_MEMORY, 5.0)
        metrics.record_layer_miss(CacheLayer.L2_REDIS, 10.0)

        # 3 hits, 2 misses = 60% hit rate
        assert metrics.get_overall_hit_rate() == 0.6

    def test_check_low_hit_rate(self):
        """Test low hit rate detection (REQ-PER-004)."""
        metrics = EnhancedCacheMetrics()

        # Record low hit rate (3 hits, 7 misses = 30%)
        for _ in range(3):
            metrics.record_layer_hit(CacheLayer.L1_MEMORY, 5.0)
        for _ in range(7):
            metrics.record_layer_miss(CacheLayer.L1_MEMORY, 5.0)

        # Should be below 60% threshold
        assert metrics.check_low_hit_rate(threshold=0.6) is True

        # Add more hits to reach 70%
        for _ in range(7):
            metrics.record_layer_hit(CacheLayer.L1_MEMORY, 5.0)

        assert metrics.check_low_hit_rate(threshold=0.6) is False

    def test_update_connection_pool(self):
        """Test updating connection pool metrics (REQ-PER-002)."""
        metrics = EnhancedCacheMetrics()
        metrics.update_connection_pool(total=50, active=20, idle=30, failed=2, errors=1)

        pool_metrics = metrics.get_connection_pool_metrics()
        assert pool_metrics.total_connections == 50
        assert pool_metrics.active_connections == 20
        assert pool_metrics.idle_connections == 30
        assert pool_metrics.failed_connections == 2
        assert pool_metrics.connection_errors == 1

    def test_to_dict(self):
        """Test serialization of all metrics to dict."""
        metrics = EnhancedCacheMetrics()
        metrics.record_layer_hit(CacheLayer.L1_MEMORY, 10.0)
        metrics.record_layer_miss(CacheLayer.L2_REDIS, 5.0)
        metrics.update_connection_pool(total=50, active=20, idle=30)

        d = metrics.to_dict()

        assert "overall_hit_rate" in d
        assert "low_hit_rate_warning" in d
        assert "layers" in d
        assert "connection_pool" in d
        assert "warming" in d
        assert "l1_memory" in d["layers"]
        assert "l2_redis" in d["layers"]

    def test_reset_layer_metrics(self):
        """Test resetting metrics for specific layer."""
        metrics = EnhancedCacheMetrics()
        metrics.record_layer_hit(CacheLayer.L1_MEMORY, 10.0)
        metrics.record_layer_miss(CacheLayer.L1_MEMORY, 5.0)

        metrics.reset_layer_metrics(CacheLayer.L1_MEMORY)

        l1_metrics = metrics.get_layer_metrics(CacheLayer.L1_MEMORY)
        assert l1_metrics.hits == 0
        assert l1_metrics.misses == 0

    def test_reset_all_metrics(self):
        """Test resetting all metrics."""
        metrics = EnhancedCacheMetrics()
        metrics.record_layer_hit(CacheLayer.L1_MEMORY, 10.0)
        metrics.record_layer_hit(CacheLayer.L2_REDIS, 15.0)
        metrics.update_connection_pool(total=50, active=20, idle=30)

        metrics.reset_all()

        # All layer metrics should be reset
        for layer in CacheLayer:
            layer_metrics = metrics.get_layer_metrics(layer)
            assert layer_metrics.hits == 0
            assert layer_metrics.misses == 0

        # Connection pool should be reset
        pool_metrics = metrics.get_connection_pool_metrics()
        assert pool_metrics.total_connections == 0
