"""
Cache warming strategy for RAG system (REQ-PER-003, REQ-PER-004, REQ-PER-006, REQ-PER-010).

Provides automated cache warming for frequently accessed regulations:
- Pre-compute embeddings for top 100 regulations (REQ-PER-006)
- Schedule warming during low-traffic periods (REQ-PER-010)
- Trigger warming when hit rate drops below 60% (REQ-PER-004)
- Write-through caching for immediate refresh (REQ-PER-007)
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WarmQuery:
    """A query to warm the cache with."""

    cache_type: str
    query: str
    filter_options: Optional[Dict[str, Any]]
    kwargs: Dict[str, Any]
    data: Dict[str, Any]
    priority: int = 0  # Higher priority = warmed first


@dataclass
class WarmingSchedule:
    """Schedule for cache warming (REQ-PER-010)."""

    enabled: bool = False
    hour: int = 2  # 2:00 AM default (low-traffic period)
    minute: int = 0
    timezone: str = "UTC"

    def should_warm_now(self) -> bool:
        """Check if current time matches schedule."""
        if not self.enabled:
            return False

        now = datetime.now()
        return now.hour == self.hour and now.minute == self.minute


class CacheWarmer:
    """
    Automated cache warming service.

    Features:
    - Top 100 most frequent queries pre-caching (REQ-PER-006)
    - Scheduled warming during low-traffic periods (REQ-PER-010)
    - Hit rate-based warming trigger (REQ-PER-004)
    - Asynchronous warming to avoid blocking
    """

    def __init__(
        self,
        cache,
        enabled: bool = True,
        top_n: int = 100,  # REQ-PER-006
        schedule: Optional[WarmingSchedule] = None,
        hit_rate_threshold: float = 0.6,  # REQ-PER-004
    ):
        """
        Initialize cache warmer.

        Args:
            cache: RAGQueryCache instance.
            enabled: Whether warming is enabled.
            top_n: Number of top queries to warm (REQ-PER-006).
            schedule: Optional warming schedule (REQ-PER-010).
            hit_rate_threshold: Trigger warming when hit rate below this (REQ-PER-004).
        """
        self._cache = cache
        self._enabled = enabled
        self._top_n = top_n
        self._schedule = schedule or WarmingSchedule(enabled=False)
        self._hit_rate_threshold = hit_rate_threshold
        self._warm_queries: List[WarmQuery] = []
        self._lock = threading.Lock()
        self._warming_in_progress = False

        # Load warm queries from file if available
        self._load_warm_queries()

    def _load_warm_queries(self) -> None:
        """Load warm queries from configuration file."""
        try:
            from ..config import get_config

            config = get_config()
            warm_queries_path = config.cache_warm_queries_path_resolved

            if warm_queries_path and warm_queries_path.exists():
                import json

                with open(warm_queries_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    queries = data.get("queries", [])

                    for q in queries:
                        warm_query = WarmQuery(
                            cache_type=q.get("cache_type", "retrieval"),
                            query=q["query"],
                            filter_options=q.get("filter_options"),
                            kwargs=q.get("kwargs", {}),
                            data=q.get("data", {}),
                            priority=q.get("priority", 0),
                        )
                        self._warm_queries.append(warm_query)

                    logger.info(
                        f"Loaded {len(self._warm_queries)} warm queries from config"
                    )
        except Exception as e:
            logger.warning(f"Failed to load warm queries: {e}")

    def add_warm_query(
        self,
        cache_type: str,
        query: str,
        data: Dict[str, Any],
        filter_options: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> None:
        """
        Add a query to the warm list.

        Args:
            cache_type: Type of cache (retrieval, llm_response).
            query: Query text.
            data: Cached data.
            filter_options: Optional filter options.
            priority: Priority (higher = warmed first).
        """
        with self._lock:
            warm_query = WarmQuery(
                cache_type=cache_type,
                query=query,
                filter_options=filter_options,
                kwargs={},
                data=data,
                priority=priority,
            )
            self._warm_queries.append(warm_query)
            logger.debug(f"Added warm query: {query[:30]}... (priority={priority})")

    def warm_cache_async(self) -> None:
        """Trigger asynchronous cache warming."""
        if not self._enabled:
            logger.debug("Cache warming disabled, skipping")
            return

        if self._warming_in_progress:
            logger.debug("Warming already in progress, skipping")
            return

        # Run warming in background thread
        thread = threading.Thread(target=self._warm_cache, daemon=True)
        thread.start()

    def _warm_cache(self) -> Dict[str, int]:
        """
        Warm cache with frequent queries (REQ-PER-006).

        Returns:
            Dict with warming statistics.
        """
        if not self._enabled:
            return {"warmed": 0, "skipped": 0, "errors": 0}

        self._warming_in_progress = True
        stats = {"warmed": 0, "skipped": 0, "errors": 0}

        try:
            # Sort by priority (highest first)
            with self._lock:
                sorted_queries = sorted(
                    self._warm_queries, key=lambda x: x.priority, reverse=True
                )[: self._top_n]

            logger.info(f"Starting cache warming for {len(sorted_queries)} queries")

            from .cache import CacheType

            for warm_query in sorted_queries:
                try:
                    # Convert cache_type string to enum
                    cache_type_enum = CacheType[warm_query.cache_type.upper()]

                    # Skip if already cached
                    if self._cache.get(
                        cache_type_enum,
                        warm_query.query,
                        warm_query.filter_options,
                        **warm_query.kwargs,
                    ):
                        stats["skipped"] += 1
                        continue

                    # Store data (write-through caching, REQ-PER-007)
                    self._cache.set(
                        cache_type_enum,
                        warm_query.query,
                        warm_query.data,
                        warm_query.filter_options,
                        **warm_query.kwargs,
                    )
                    stats["warmed"] += 1

                except Exception as e:
                    logger.error(
                        f"Cache warming error for query '{warm_query.query[:30]}...': {e}"
                    )
                    stats["errors"] += 1

            logger.info(f"Cache warming complete: {stats}")

            # Record warming metrics
            if (
                hasattr(self._cache, "_enhanced_metrics")
                and self._cache._enhanced_metrics
            ):
                self._cache._enhanced_metrics.get_warming_metrics().record_warming_complete(
                    stats["warmed"]
                )

        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
            if (
                hasattr(self._cache, "_enhanced_metrics")
                and self._cache._enhanced_metrics
            ):
                self._cache._enhanced_metrics.get_warming_metrics().record_warming_failure()

        finally:
            self._warming_in_progress = False

        return stats

    def check_and_warm(self) -> None:
        """
        Check if warming should be triggered and warm if needed.

        Triggers warming if:
        - Scheduled time reached (REQ-PER-010)
        - Hit rate below threshold (REQ-PER-004)
        """
        if not self._enabled:
            return

        # Check schedule (REQ-PER-010)
        if self._schedule.should_warm_now():
            logger.info("Scheduled warming time reached (REQ-PER-010)")
            self.warm_cache_async()
            return

        # Check hit rate (REQ-PER-004)
        if hasattr(self._cache, "_enhanced_metrics") and self._cache._enhanced_metrics:
            if self._cache._enhanced_metrics.check_low_hit_rate(
                threshold=self._hit_rate_threshold
            ):
                logger.warning(
                    f"Hit rate below {self._hit_rate_threshold:.0%}, triggering warming (REQ-PER-004)"
                )
                self.warm_cache_async()

    def get_warming_stats(self) -> Dict[str, Any]:
        """Get warming statistics."""
        stats = {
            "enabled": self._enabled,
            "top_n": self._top_n,
            "warm_queries_count": len(self._warm_queries),
            "warming_in_progress": self._warming_in_progress,
            "schedule": {
                "enabled": self._schedule.enabled,
                "hour": self._schedule.hour,
                "minute": self._schedule.minute,
            },
            "hit_rate_threshold": self._hit_rate_threshold,
        }

        # Add metrics from enhanced metrics if available
        if hasattr(self._cache, "_enhanced_metrics") and self._cache._enhanced_metrics:
            warming_metrics = self._cache._enhanced_metrics.get_warming_metrics()
            stats["metrics"] = warming_metrics.to_dict()

        return stats
