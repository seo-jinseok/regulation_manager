"""
LLM Response Caching Layer.

Provides caching for LLM responses to reduce API costs and latency.
Uses content-based hashing for cache keys.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CacheEntry:
    """A cached LLM response entry."""

    query_hash: str
    response: str
    model: str
    timestamp: float
    ttl_days: int = 30

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        age_days = (time.time() - self.timestamp) / (24 * 3600)
        return age_days > self.ttl_days


class LLMResponseCache:
    """
    Caches LLM responses to disk for reuse.

    Features:
    - Content-based hashing for cache keys
    - TTL-based expiration
    - Model-specific caching
    - JSON persistence
    """

    def __init__(
        self,
        cache_dir: str = "data/llm_cache",
        ttl_days: int = 30,
        max_entries: int = 5000,
    ):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory for cache files.
            ttl_days: Default TTL in days.
            max_entries: Maximum number of cache entries.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days
        self.max_entries = max_entries
        self._index_path = self.cache_dir / "cache_index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache index from disk."""
        if not self._index_path.exists():
            return {}
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)

    def _compute_hash(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
    ) -> str:
        """Compute a hash for the query."""
        content = f"{model}::{system_prompt}::{user_message}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    def get(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
    ) -> Optional[str]:
        """
        Get a cached response if available.

        Args:
            system_prompt: The system prompt.
            user_message: The user message.
            model: The model name.

        Returns:
            Cached response or None if not found/expired.
        """
        query_hash = self._compute_hash(system_prompt, user_message, model)

        if query_hash not in self._index:
            return None

        entry_data = self._index[query_hash]
        entry = CacheEntry(**entry_data)

        if entry.is_expired():
            # Remove expired entry
            del self._index[query_hash]
            self._save_index()
            return None

        return entry.response

    def set(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        response: str,
    ) -> None:
        """
        Cache an LLM response.

        Args:
            system_prompt: The system prompt.
            user_message: The user message.
            model: The model name.
            response: The LLM response to cache.
        """
        query_hash = self._compute_hash(system_prompt, user_message, model)

        entry = CacheEntry(
            query_hash=query_hash,
            response=response,
            model=model,
            timestamp=time.time(),
            ttl_days=self.ttl_days,
        )

        self._index[query_hash] = asdict(entry)

        # Enforce max entries (LRU-like cleanup)
        if len(self._index) > self.max_entries:
            self._cleanup_oldest()

        self._save_index()

    def _cleanup_oldest(self) -> None:
        """Remove oldest entries when cache is full."""
        # Sort by timestamp and remove oldest 10%
        entries = sorted(
            self._index.items(),
            key=lambda x: x[1].get("timestamp", 0),
        )
        remove_count = len(entries) // 10
        for key, _ in entries[:remove_count]:
            del self._index[key]

    def clear_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        expired_keys = []
        for key, data in self._index.items():
            entry = CacheEntry(**data)
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            del self._index[key]

        if expired_keys:
            self._save_index()

        return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats.
        """
        total = len(self._index)
        expired = sum(
            1 for data in self._index.values() if CacheEntry(**data).is_expired()
        )
        return {
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired,
            "cache_dir": str(self.cache_dir),
        }
