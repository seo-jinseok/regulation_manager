"""
Enhanced cache performance metrics for RAG system.

Provides comprehensive metrics collection for all cache layers:
- L1: In-memory cache
- L2: Redis cache (with connection pooling)
- L3: ChromaDB cache

Tracks:
- Hit/miss rates per cache layer
- Eviction rates
- Memory usage
- Latency percentiles (P50, P95, P99)
- Connection pool statistics
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Dict, List, Optional


class CacheLayer(Enum):
    """Cache layer types."""

    L1_MEMORY = "l1_memory"  # Fastest, limited size
    L2_REDIS = "l2_redis"  # Distributed, persistent
    L3_CHROMA = "l3_chroma"  # Vector similarity


@dataclass
class LayerMetrics:
    """Metrics for a single cache layer."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0

    # Latency tracking (in milliseconds)
    latencies: List[float] = field(default_factory=list)
    max_latency_samples: int = 1000

    @property
    def total_requests(self) -> int:
        """Total cache requests."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """Cache miss rate (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.misses / self.total_requests

    @property
    def p50_latency(self) -> Optional[float]:
        """P50 latency (median) in milliseconds."""
        if not self.latencies:
            return None
        sorted_latencies = sorted(self.latencies)
        return sorted_latencies[len(sorted_latencies) // 2]

    @property
    def p95_latency(self) -> Optional[float]:
        """P95 latency in milliseconds."""
        if not self.latencies:
            return None
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    @property
    def p99_latency(self) -> Optional[float]:
        """P99 latency in milliseconds."""
        if not self.latencies:
            return None
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]

    def record_hit(self, latency_ms: float) -> None:
        """Record a cache hit with latency."""
        self.hits += 1
        self._add_latency(latency_ms)

    def record_miss(self, latency_ms: float) -> None:
        """Record a cache miss with latency."""
        self.misses += 1
        self._add_latency(latency_ms)

    def record_eviction(self) -> None:
        """Record a cache eviction."""
        self.evictions += 1

    def record_error(self) -> None:
        """Record a cache error."""
        self.errors += 1

    def _add_latency(self, latency_ms: float) -> None:
        """Add latency sample, maintaining max sample size."""
        self.latencies.append(latency_ms)
        if len(self.latencies) > self.max_latency_samples:
            self.latencies.pop(0)

    def reset(self) -> None:
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.errors = 0
        self.latencies.clear()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "errors": self.errors,
            "total_requests": self.total_requests,
            "hit_rate": f"{self.hit_rate:.2%}",
            "miss_rate": f"{self.miss_rate:.2%}",
            "entry_count": self.entry_count,
            "total_size_bytes": self.total_size_bytes,
            "p50_latency_ms": self.p50_latency,
            "p95_latency_ms": self.p95_latency,
            "p99_latency_ms": self.p99_latency,
        }


@dataclass
class ConnectionPoolMetrics:
    """Metrics for connection pool performance."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    connection_errors: int = 0
    max_pool_size: int = 50

    @property
    def pool_utilization(self) -> float:
        """Pool utilization ratio (0.0 to 1.0)."""
        if self.max_pool_size == 0:
            return 0.0
        return self.total_connections / self.max_pool_size

    @property
    def is_exhausted(self) -> bool:
        """Check if pool is exhausted."""
        return self.active_connections >= self.max_pool_size

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_connections": self.total_connections,
            "active_connections": self.active_connections,
            "idle_connections": self.idle_connections,
            "failed_connections": self.failed_connections,
            "connection_errors": self.connection_errors,
            "max_pool_size": self.max_pool_size,
            "pool_utilization": f"{self.pool_utilization:.2%}",
            "is_exhausted": self.is_exhausted,
        }


@dataclass
class CacheWarmingMetrics:
    """Metrics for cache warming operations."""

    warming_jobs_completed: int = 0
    warming_jobs_failed: int = 0
    total_entries_warmed: int = 0
    total_warming_time_ms: float = 0.0
    last_warming_timestamp: Optional[float] = None
    last_warming_duration_ms: Optional[float] = None

    @property
    def avg_warming_time_ms(self) -> float:
        """Average warming time in milliseconds."""
        if self.warming_jobs_completed == 0:
            return 0.0
        return self.total_warming_time_ms / self.warming_jobs_completed

    @property
    def success_rate(self) -> float:
        """Warming job success rate."""
        total = self.warming_jobs_completed + self.warming_jobs_failed
        if total == 0:
            return 0.0
        return self.warming_jobs_completed / total

    def record_warming_start(self) -> None:
        """Record the start of a warming job."""
        self._current_warming_start = time.time()

    def record_warming_complete(self, entries_count: int) -> None:
        """Record successful completion of warming job."""
        if hasattr(self, "_current_warming_start"):
            duration_ms = (time.time() - self._current_warming_start) * 1000
            self.last_warming_duration_ms = duration_ms
            self.total_warming_time_ms += duration_ms
            delattr(self, "_current_warming_start")

        self.warming_jobs_completed += 1
        self.total_entries_warmed += entries_count
        self.last_warming_timestamp = time.time()

    def record_warming_failure(self) -> None:
        """Record failed warming job."""
        if hasattr(self, "_current_warming_start"):
            delattr(self, "_current_warming_start")

        self.warming_jobs_failed += 1

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "warming_jobs_completed": self.warming_jobs_completed,
            "warming_jobs_failed": self.warming_jobs_failed,
            "total_entries_warmed": self.total_entries_warmed,
            "avg_warming_time_ms": self.avg_warming_time_ms,
            "last_warming_timestamp": self.last_warming_timestamp,
            "last_warming_duration_ms": self.last_warming_duration_ms,
            "success_rate": f"{self.success_rate:.2%}",
        }


class EnhancedCacheMetrics:
    """
    Enhanced metrics collector for all cache layers.

    Provides:
    - Per-layer metrics (hit rate, latency, evictions)
    - Connection pool statistics
    - Cache warming metrics
    - Overall system health indicators
    """

    def __init__(self):
        """Initialize enhanced metrics."""
        self._layers: Dict[CacheLayer, LayerMetrics] = {}
        self._connection_pool: ConnectionPoolMetrics = ConnectionPoolMetrics()
        self._warming: CacheWarmingMetrics = CacheWarmingMetrics()
        self._lock = Lock()

        # Initialize layer metrics
        for layer in CacheLayer:
            self._layers[layer] = LayerMetrics()

    def get_layer_metrics(self, layer: CacheLayer) -> LayerMetrics:
        """Get metrics for a specific cache layer."""
        with self._lock:
            return self._layers[layer]

    def get_connection_pool_metrics(self) -> ConnectionPoolMetrics:
        """Get connection pool metrics."""
        with self._lock:
            return self._connection_pool

    def get_warming_metrics(self) -> CacheWarmingMetrics:
        """Get cache warming metrics."""
        with self._lock:
            return self._warming

    def get_overall_hit_rate(self) -> float:
        """Calculate overall cache hit rate across all layers."""
        with self._lock:
            total_hits = sum(layer.hits for layer in self._layers.values())
            total_requests = sum(
                layer.hits + layer.misses for layer in self._layers.values()
            )
            if total_requests == 0:
                return 0.0
            return total_hits / total_requests

    def check_low_hit_rate(self, threshold: float = 0.6) -> bool:
        """
        Check if overall hit rate is below threshold.

        Args:
            threshold: Hit rate threshold (default 0.6 = 60%).

        Returns:
            True if hit rate is below threshold.
        """
        return self.get_overall_hit_rate() < threshold

    def record_layer_hit(self, layer: CacheLayer, latency_ms: float = 0.0) -> None:
        """Record a cache hit for a specific layer."""
        with self._lock:
            self._layers[layer].record_hit(latency_ms)

    def record_layer_miss(self, layer: CacheLayer, latency_ms: float = 0.0) -> None:
        """Record a cache miss for a specific layer."""
        with self._lock:
            self._layers[layer].record_miss(latency_ms)

    def record_layer_eviction(self, layer: CacheLayer) -> None:
        """Record a cache eviction for a specific layer."""
        with self._lock:
            self._layers[layer].record_eviction()

    def record_layer_error(self, layer: CacheLayer) -> None:
        """Record a cache error for a specific layer."""
        with self._lock:
            self._layers[layer].record_error()

    def update_connection_pool(
        self,
        total: int,
        active: int,
        idle: int,
        failed: int = 0,
        errors: int = 0,
    ) -> None:
        """Update connection pool metrics."""
        with self._lock:
            self._connection_pool.total_connections = total
            self._connection_pool.active_connections = active
            self._connection_pool.idle_connections = idle
            self._connection_pool.failed_connections = failed
            self._connection_pool.connection_errors = errors

    def to_dict(self) -> Dict:
        """Convert all metrics to dictionary."""
        with self._lock:
            return {
                "overall_hit_rate": f"{self.get_overall_hit_rate():.2%}",
                "low_hit_rate_warning": self.check_low_hit_rate(),
                "layers": {
                    layer.value: metrics.to_dict()
                    for layer, metrics in self._layers.items()
                },
                "connection_pool": self._connection_pool.to_dict(),
                "warming": self._warming.to_dict(),
            }

    def reset_layer_metrics(self, layer: Optional[CacheLayer] = None) -> None:
        """Reset metrics for a specific layer or all layers."""
        with self._lock:
            if layer:
                self._layers[layer].reset()
            else:
                for layer_metrics in self._layers.values():
                    layer_metrics.reset()

    def reset_all(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for layer_metrics in self._layers.values():
                layer_metrics.reset()
            self._connection_pool = ConnectionPoolMetrics()
            self._warming = CacheWarmingMetrics()
