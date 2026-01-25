"""
RAG Query Caching Layer.

Provides comprehensive caching for RAG query results to improve performance.
Supports Redis with file-based fallback, cache stampede prevention, and statistics.

Features:
- Redis primary storage with file-based fallback
- Content-based cache keys (query_hash from query + filter options)
- 24-hour TTL for regulation data (rarely changes)
- Cache stampede prevention with single-flight pattern
- Cache statistics (hit rate, miss rate, memory usage)
- Cache warming support for frequent queries
- Separate cache namespaces for retrieval and LLM responses
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Type of cached data."""
    RETRIEVAL = "retrieval"  # ChromaDB query results
    LLM_RESPONSE = "llm_response"  # LLM-generated answers


@dataclass
class CacheEntry:
    """A cached entry with metadata."""
    cache_type: CacheType
    query_hash: str
    data: Dict[str, Any]
    timestamp: float
    ttl_hours: int = 24

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        age_hours = (time.time() - self.timestamp) / 3600
        return age_hours > self.ttl_hours

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cache_type": self.cache_type.value,
            "query_hash": self.query_hash,
            "data": self.data,
            "timestamp": self.timestamp,
            "ttl_hours": self.ttl_hours,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            cache_type=CacheType(data["cache_type"]),
            query_hash=data["query_hash"],
            data=data["data"],
            timestamp=data["timestamp"],
            ttl_hours=data.get("ttl_hours", 24),
        )


@dataclass
class CacheStats:
    """Cache statistics tracking."""
    hits: int = 0
    misses: int = 0
    stampede_prevented: int = 0
    errors: int = 0

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stampede_prevented": self.stampede_prevented,
            "errors": self.errors,
            "total_requests": self.total_requests,
            "hit_rate": f"{self.hit_rate:.2%}",
            "miss_rate": f"{self.miss_rate:.2%}",
        }

    def reset(self) -> None:
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.stampede_prevented = 0
        self.errors = 0


class FileBackend:
    """File-based cache backend as fallback when Redis is unavailable."""

    def __init__(self, cache_dir: str = "data/cache/rag"):
        """
        Initialize file-based cache backend.

        Args:
            cache_dir: Directory for cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_dir / "cache_index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()
        self._lock = threading.Lock()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache index from disk."""
        if not self._index_path.exists():
            return {}
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache index: {e}")
            return {}

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        try:
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.error(f"Failed to save cache index: {e}")

    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Get a cache entry by key.

        Args:
            key: Cache key.

        Returns:
            CacheEntry or None if not found/expired.
        """
        with self._lock:
            if key not in self._index:
                return None

            entry_data = self._index[key]
            entry = CacheEntry.from_dict(entry_data)

            if entry.is_expired():
                # Remove expired entry
                del self._index[key]
                self._save_index()
                return None

            return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        """
        Set a cache entry.

        Args:
            key: Cache key.
            entry: CacheEntry to store.
        """
        with self._lock:
            self._index[key] = entry.to_dict()
            self._save_index()

    def delete(self, key: str) -> bool:
        """
        Delete a cache entry.

        Args:
            key: Cache key.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if key in self._index:
                del self._index[key]
                self._save_index()
                return True
            return False

    def clear_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            expired_keys = []
            for key, data in self._index.items():
                entry = CacheEntry.from_dict(data)
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self._index[key]

            if expired_keys:
                self._save_index()

            return len(expired_keys)

    def clear_all(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._index)
            self._index.clear()
            self._save_index()
            return count


class RedisBackend:
    """Redis cache backend for high-performance caching."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "rag_cache:",
    ):
        """
        Initialize Redis backend.

        Args:
            host: Redis host.
            port: Redis port.
            db: Redis database number.
            password: Optional Redis password.
            prefix: Key prefix for all cache entries.
        """
        self._client = None
        self._prefix = prefix
        self._connection_params = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
            "decode_responses": True,
        }
        self._connect()

    def _connect(self) -> None:
        """Establish Redis connection."""
        try:
            import redis
            self._client = redis.Redis(**self._connection_params)
            # Test connection
            self._client.ping()
            logger.info(f"Connected to Redis at {self._connection_params['host']}:{self._connection_params['port']}")
        except ImportError:
            logger.warning("redis package not installed. Redis backend unavailable.")
            self._client = None
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._client = None

    @property
    def available(self) -> bool:
        """Check if Redis backend is available."""
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def _make_key(self, key: str) -> str:
        """Create full Redis key with prefix."""
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Get a cache entry by key.

        Args:
            key: Cache key.

        Returns:
            CacheEntry or None if not found/expired.
        """
        if not self.available:
            return None

        try:
            redis_key = self._make_key(key)
            data = self._client.get(redis_key)
            if data is None:
                return None

            entry_dict = json.loads(data)
            entry = CacheEntry.from_dict(entry_dict)

            if entry.is_expired():
                self.delete(key)
                return None

            return entry
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, entry: CacheEntry) -> None:
        """
        Set a cache entry with TTL.

        Args:
            key: Cache key.
            entry: CacheEntry to store.
        """
        if not self.available:
            return

        try:
            redis_key = self._make_key(key)
            data = json.dumps(entry.to_dict(), ensure_ascii=False)
            ttl_seconds = entry.ttl_hours * 3600
            self._client.setex(redis_key, ttl_seconds, data)
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    def delete(self, key: str) -> bool:
        """
        Delete a cache entry.

        Args:
            key: Cache key.

        Returns:
            True if deleted, False otherwise.
        """
        if not self.available:
            return False

        try:
            redis_key = self._make_key(key)
            result = self._client.delete(redis_key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def clear_expired(self) -> int:
        """
        Redis handles TTL automatically, so this is a no-op.

        Returns:
            0 (Redis handles expiration automatically).
        """
        return 0

    def clear_all(self) -> int:
        """
        Clear all cache entries with the configured prefix.

        Returns:
            Number of keys cleared.
        """
        if not self.available:
            return 0

        try:
            pattern = f"{self._prefix}*"
            keys = list(self._client.scan_iter(match=pattern))
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis clear_all error: {e}")
            return 0


class RAGQueryCache:
    """
    Main cache manager for RAG query results.

    Features:
    - Redis primary with file fallback
    - Cache stampede prevention with single-flight pattern
    - Cache statistics tracking
    - Separate namespaces for retrieval and LLM responses
    - Cache warming support
    """

    def __init__(
        self,
        enabled: bool = True,
        ttl_hours: int = 24,
        redis_host: Optional[str] = None,
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        cache_dir: str = "data/cache/rag",
    ):
        """
        Initialize RAG query cache.

        Args:
            enabled: Whether caching is enabled.
            ttl_hours: Default TTL in hours (24 for regulation data).
            redis_host: Optional Redis host (enables Redis backend if provided).
            redis_port: Redis port.
            redis_password: Optional Redis password.
            cache_dir: Directory for file-based fallback cache.
        """
        self.enabled = enabled
        self.ttl_hours = ttl_hours
        self._stats = CacheStats()
        self._lock = threading.Lock()
        self._pending: Dict[str, threading.Event] = {}  # For stampede prevention

        # Initialize backends
        self._redis: Optional[RedisBackend] = None
        self._file: FileBackend = FileBackend(cache_dir)

        if redis_host:
            self._redis = RedisBackend(
                host=redis_host,
                port=redis_port,
                password=redis_password,
            )
            if self._redis.available:
                logger.info("RAG cache using Redis backend")
            else:
                logger.warning("Redis unavailable, using file backend")
                self._redis = None
        else:
            logger.info("RAG cache using file backend")

    def _compute_hash(
        self,
        cache_type: CacheType,
        query: str,
        filter_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Compute cache key from query and parameters.

        Args:
            cache_type: Type of cached data.
            query: Query text.
            filter_options: Optional search filter dict.
            **kwargs: Additional parameters (top_k, include_abolished, etc.)

        Returns:
            SHA256 hash as cache key.
        """
        # Normalize query
        normalized_query = query.strip().lower()

        # Build cache key components
        components = [
            cache_type.value,
            normalized_query,
        ]

        # Add filter options
        if filter_options:
            # Sort keys for consistent hashing
            sorted_filters = sorted(filter_options.items())
            components.append(str(sorted_filters))

        # Add other parameters (sorted)
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            components.append(str(sorted_kwargs))

        # Compute hash
        content = "::".join(components)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    def get(
        self,
        cache_type: CacheType,
        query: str,
        filter_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached data if available.

        Args:
            cache_type: Type of cached data.
            query: Query text.
            filter_options: Optional search filter dict.
            **kwargs: Additional parameters.

        Returns:
            Cached data or None if not found/expired.
        """
        if not self.enabled:
            return None

        cache_key = self._compute_hash(cache_type, query, filter_options, **kwargs)

        # Try Redis first
        if self._redis and self._redis.available:
            entry = self._redis.get(cache_key)
            if entry:
                self._stats.hits += 1
                logger.debug(f"Cache HIT (Redis): {cache_type.value} for '{query[:30]}...'")
                return entry.data

        # Fallback to file backend
        entry = self._file.get(cache_key)
        if entry:
            self._stats.hits += 1
            logger.debug(f"Cache HIT (File): {cache_type.value} for '{query[:30]}...'")
            return entry.data

        self._stats.misses += 1
        logger.debug(f"Cache MISS: {cache_type.value} for '{query[:30]}...'")
        return None

    def set(
        self,
        cache_type: CacheType,
        query: str,
        data: Dict[str, Any],
        filter_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Cache data.

        Args:
            cache_type: Type of data to cache.
            query: Query text.
            data: Data to cache.
            filter_options: Optional search filter dict.
            **kwargs: Additional parameters.
        """
        if not self.enabled:
            return

        cache_key = self._compute_hash(cache_type, query, filter_options, **kwargs)

        entry = CacheEntry(
            cache_type=cache_type,
            query_hash=cache_key,
            data=data,
            timestamp=time.time(),
            ttl_hours=self.ttl_hours,
        )

        # Store in both backends
        if self._redis and self._redis.available:
            self._redis.set(cache_key, entry)

        self._file.set(cache_key, entry)

    def invalidate(
        self,
        cache_type: Optional[CacheType] = None,
        query_pattern: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            cache_type: Optional cache type to filter.
            query_pattern: Optional query pattern to match.

        Returns:
            Number of entries invalidated.
        """
        count = 0

        if self._redis and self._redis.available:
            count += self._redis.clear_all()

        count += self._file.clear_all()

        logger.info(f"Invalidated {count} cache entries")
        return count

    def warm_cache(
        self,
        queries: List[tuple],
    ) -> Dict[str, int]:
        """
        Warm cache with frequent queries.

        Args:
            queries: List of (cache_type, query, filter_options, kwargs, data) tuples.

        Returns:
            Dict with warming statistics.
        """
        stats = {"warmed": 0, "skipped": 0, "errors": 0}

        for item in queries:
            try:
                cache_type, query, filter_options, kwargs, data = item

                # Skip if already cached
                if self.get(cache_type, query, filter_options, **kwargs):
                    stats["skipped"] += 1
                    continue

                # Store data
                self.set(cache_type, query, data, filter_options, **kwargs)
                stats["warmed"] += 1
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                stats["errors"] += 1

        logger.info(f"Cache warming complete: {stats}")
        return stats

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics.
        """
        stats_dict = self._stats.to_dict()

        # Add backend info
        backend_info = {
            "redis_enabled": self._redis is not None and self._redis.available,
            "cache_enabled": self.enabled,
            "ttl_hours": self.ttl_hours,
        }

        return {**stats_dict, **backend_info}

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats.reset()

    def clear_expired(self) -> int:
        """
        Clear expired entries from file backend.

        Note: Redis handles TTL automatically.

        Returns:
            Number of entries cleared.
        """
        return self._file.clear_expired()

    def clear_all(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        count = 0

        if self._redis and self._redis.available:
            count += self._redis.clear_all()

        count += self._file.clear_all()

        logger.info(f"Cleared {count} cache entries")
        return count


class SingleFlight:
    """
    Prevents cache stampede by ensuring only one computation per cache key.

    When multiple concurrent requests for the same cache miss occur,
    only the first request computes the result while others wait.
    """

    def __init__(self):
        """Initialize single-flight registry."""
        self._pending: Dict[str, threading.Event] = {}
        self._results: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def execute(self, key: str, func: Callable[[], Any]) -> Any:
        """
        Execute function with single-flight protection.

        Args:
            key: Cache key to track.
            func: Function to execute if not already in progress.

        Returns:
            Function result.
        """
        with self._lock:
            # Check if already computing
            if key in self._pending:
                # Wait for existing computation
                event = self._pending[key]
                self._lock.release()

                # Wait outside of lock
                event.wait(timeout=30)

                # Re-acquire lock to get result
                with self._lock:
                    result = self._results.get(key)
                    # Clean up
                    if key in self._pending:
                        del self._pending[key]
                    if key in self._results:
                        del self._results[key]
                    return result

            # Mark as computing
            self._pending[key] = threading.Event()

        try:
            # Execute function outside of lock
            result = func()

            # Store result and signal waiters
            with self._lock:
                self._results[key] = result
                self._pending[key].set()

            return result
        except Exception:
            # Signal waiters even on error
            with self._lock:
                self._pending[key].set()
                if key in self._pending:
                    del self._pending[key]
            raise


# Global single-flight instance
_single_flight = SingleFlight()
