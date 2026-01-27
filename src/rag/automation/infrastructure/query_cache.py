"""
Query Cache for LLM-generated test queries.

Infrastructure layer that caches generated queries to reduce LLM API costs
and ensure reproducibility across test runs.

Clean Architecture: Infrastructure implements domain interfaces and uses
external libraries (json, pathlib).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.rag.automation.domain.entities import (
    DifficultyLevel,
    EvaluationCase,
    PersonaType,
    QueryType,
)
from src.rag.automation.domain.value_objects import IntentAnalysis


class QueryCache:
    """
    Cache for LLM-generated test queries.

    Stores generated queries on disk to avoid redundant LLM API calls
    and ensure reproducibility. Cache keys are based on persona type,
    query counts, and random seed.

    Attributes:
        cache_dir: Directory for cache files.
        _enabled: Flag to enable/disable caching.
    """

    def __init__(self, cache_dir: str = ".query_cache", enabled: bool = True):
        """
        Initialize the query cache.

        Args:
            cache_dir: Directory path for cache storage.
            enabled: Whether caching is enabled.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = enabled

    def get(self, key: str) -> Optional[List[EvaluationCase]]:
        """
        Retrieve cached test cases.

        Args:
            key: Cache key (typically MD5 hash).

        Returns:
            List of EvaluationCase if cached, None otherwise.
        """
        if not self._enabled:
            return None

        cache_file = self.cache_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            return [self._dict_to_test_case(item) for item in data]
        except (json.JSONDecodeError, TypeError, KeyError):
            # Invalid cache file, treat as cache miss
            return None

    def _dict_to_test_case(self, data: Dict) -> EvaluationCase:
        """
        Convert dictionary to EvaluationCase with proper Enum conversion.

        Args:
            data: Dictionary from JSON cache.

        Returns:
            EvaluationCase instance.
        """
        # Convert string values to Enums
        persona_type = PersonaType(data["persona_type"])
        difficulty = DifficultyLevel(data["difficulty"])
        query_type = QueryType(data["query_type"])

        # Convert intent_analysis if present
        intent_analysis = None
        if data.get("intent_analysis"):
            ia_data = data["intent_analysis"]
            intent_analysis = IntentAnalysis(
                surface_intent=ia_data["surface_intent"],
                hidden_intent=ia_data["hidden_intent"],
                behavioral_intent=ia_data["behavioral_intent"],
            )

        return EvaluationCase(
            query=data["query"],
            persona_type=persona_type,
            difficulty=difficulty,
            query_type=query_type,
            intent_analysis=intent_analysis,
            expected_topics=data.get("expected_topics", []),
            expected_regulations=data.get("expected_regulations", []),
            metadata=data.get("metadata", {}),
        )

    def set(self, key: str, test_cases: List[EvaluationCase]) -> None:
        """
        Store test cases in cache.

        Args:
            key: Cache key.
            test_cases: List of EvaluationCase to cache.
        """
        if not self._enabled:
            return

        cache_file = self.cache_dir / f"{key}.json"

        # Convert EvaluationCase dataclasses to dicts for JSON serialization
        data = []
        for tc in test_cases:
            tc_dict = {
                "query": tc.query,
                "persona_type": tc.persona_type.value,
                "difficulty": tc.difficulty.value,
                "query_type": tc.query_type.value,
                "intent_analysis": (
                    {
                        "surface_intent": tc.intent_analysis.surface_intent,
                        "hidden_intent": tc.intent_analysis.hidden_intent,
                        "behavioral_intent": tc.intent_analysis.behavioral_intent,
                    }
                    if tc.intent_analysis
                    else None
                ),
                "expected_topics": tc.expected_topics,
                "expected_regulations": tc.expected_regulations,
                "metadata": tc.metadata,
            }
            data.append(tc_dict)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def clear(self) -> int:
        """
        Clear all cached queries.

        Returns:
            Number of cache files deleted.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count

    def exists(self, key: str) -> bool:
        """
        Check if a cache entry exists.

        Args:
            key: Cache key.

        Returns:
            True if cache exists, False otherwise.
        """
        if not self._enabled:
            return False
        return (self.cache_dir / f"{key}.json").exists()

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable caching."""
        self._enabled = value
