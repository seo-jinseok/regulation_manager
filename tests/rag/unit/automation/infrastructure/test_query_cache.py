"""
Unit tests for QueryCache infrastructure.

Tests query caching functionality including cache HIT/MISS scenarios,
serialization/deserialization, and cache management.
"""

import json
import tempfile
from pathlib import Path

from src.rag.automation.domain.entities import (
    DifficultyLevel,
    PersonaType,
    QueryType,
    TestCase,
)
from src.rag.automation.domain.value_objects import IntentAnalysis
from src.rag.automation.infrastructure.query_cache import QueryCache


class TestQueryCache:
    """Test QueryCache functionality."""

    def test_init_creates_cache_directory(self):
        """WHEN initializing cache, THEN should create cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            cache = QueryCache(str(cache_dir))

            assert cache_dir.exists()
            assert cache_dir.is_dir()

    def test_cache_miss_returns_none(self):
        """WHEN cache key does not exist, THEN should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = QueryCache(tmpdir)
            result = cache.get("nonexistent_key")

            assert result is None

    def test_cache_set_and_get(self):
        """WHEN storing and retrieving test cases, THEN should preserve data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = QueryCache(tmpdir)

            # Create test cases
            test_cases = [
                TestCase(
                    query="휴학 신청은 어떻게 하나요?",
                    persona_type=PersonaType.FRESHMAN,
                    difficulty=DifficultyLevel.EASY,
                    query_type=QueryType.PROCEDURAL,
                    intent_analysis=IntentAnalysis(
                        surface_intent="절차/신청 문의",
                        hidden_intent="휴학 신청 절차 필요",
                        behavioral_intent="신청서 제출",
                    ),
                ),
                TestCase(
                    query="장학금 자격이 뭐야?",
                    persona_type=PersonaType.FRESHMAN,
                    difficulty=DifficultyLevel.MEDIUM,
                    query_type=QueryType.ELIGIBILITY,
                    intent_analysis=IntentAnalysis(
                        surface_intent="자격/요건 확인",
                        hidden_intent="장학금 신청 자격 확인",
                        behavioral_intent="자격 확인 후 행동",
                    ),
                ),
            ]

            # Store and retrieve
            cache.set("test_key", test_cases)
            retrieved = cache.get("test_key")

            assert retrieved is not None
            assert len(retrieved) == 2
            assert retrieved[0].query == "휴학 신청은 어떻게 하나요?"
            assert retrieved[0].persona_type == PersonaType.FRESHMAN
            assert retrieved[0].difficulty == DifficultyLevel.EASY
            assert retrieved[0].query_type == QueryType.PROCEDURAL
            assert retrieved[0].intent_analysis is not None
            assert retrieved[0].intent_analysis.surface_intent == "절차/신청 문의"

    def test_cache_creates_json_file(self):
        """WHEN storing test cases, THEN should create JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = QueryCache(str(cache_dir))

            test_cases = [
                TestCase(
                    query="테스트 질문",
                    persona_type=PersonaType.JUNIOR,
                    difficulty=DifficultyLevel.MEDIUM,
                    query_type=QueryType.FACT_CHECK,
                    intent_analysis=IntentAnalysis(
                        surface_intent="정보 요청",
                        hidden_intent="정보 필요",
                        behavioral_intent="정보 습득",
                    ),
                )
            ]

            cache.set("file_test", test_cases)

            cache_file = cache_dir / "file_test.json"
            assert cache_file.exists()

            # Verify JSON structure
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["query"] == "테스트 질문"
            assert data[0]["persona_type"] == "junior"
            assert data[0]["difficulty"] == "medium"
            assert data[0]["query_type"] == "fact_check"

    def test_cache_exists(self):
        """WHEN checking cache existence, THEN should return correct status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = QueryCache(tmpdir)

            assert not cache.exists("new_key")

            test_cases = [
                TestCase(
                    query="테스트",
                    persona_type=PersonaType.GRADUATE,
                    difficulty=DifficultyLevel.HARD,
                    query_type=QueryType.COMPLEX,
                )
            ]
            cache.set("new_key", test_cases)

            assert cache.exists("new_key")

    def test_cache_clear_removes_all_files(self):
        """WHEN clearing cache, THEN should remove all cache files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = QueryCache(str(cache_dir))

            # Create multiple cache entries
            for i in range(3):
                test_cases = [
                    TestCase(
                        query=f"질문 {i}",
                        persona_type=PersonaType.FRESHMAN,
                        difficulty=DifficultyLevel.EASY,
                        query_type=QueryType.FACT_CHECK,
                    )
                ]
                cache.set(f"key_{i}", test_cases)

            # Verify files exist
            assert len(list(cache_dir.glob("*.json"))) == 3

            # Clear cache
            deleted_count = cache.clear()

            assert deleted_count == 3
            assert len(list(cache_dir.glob("*.json"))) == 0

    def test_cache_disabled_returns_none(self):
        """WHEN cache is disabled, THEN get should always return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = QueryCache(tmpdir, enabled=False)

            test_cases = [
                TestCase(
                    query="테스트",
                    persona_type=PersonaType.FRESHMAN,
                    difficulty=DifficultyLevel.EASY,
                    query_type=QueryType.FACT_CHECK,
                )
            ]

            cache.set("key", test_cases)
            result = cache.get("key")

            assert result is None
            assert not cache.exists("key")

    def test_cache_disabled_set_does_not_create_file(self):
        """WHEN cache is disabled, THEN set should not create files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = QueryCache(str(cache_dir), enabled=False)

            test_cases = [
                TestCase(
                    query="테스트",
                    persona_type=PersonaType.FRESHMAN,
                    difficulty=DifficultyLevel.EASY,
                    query_type=QueryType.FACT_CHECK,
                )
            ]

            cache.set("key", test_cases)

            assert not (cache_dir / "key.json").exists()

    def test_cache_with_empty_test_cases(self):
        """WHEN caching empty list, THEN should handle gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = QueryCache(tmpdir)

            cache.set("empty", [])
            retrieved = cache.get("empty")

            assert retrieved == []

    def test_cache_handles_corrupted_json(self):
        """WHEN cache file is corrupted, THEN should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache = QueryCache(str(cache_dir))

            # Create corrupted JSON file
            cache_file = cache_dir / "corrupted.json"
            with open(cache_file, "w") as f:
                f.write("{ invalid json")

            result = cache.get("corrupted")

            assert result is None

    def test_cache_enabled_property(self):
        """WHEN toggling cache enabled state, THEN should reflect correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = QueryCache(tmpdir)

            assert cache.enabled is True

            cache.enabled = False
            assert cache.enabled is False

            cache.enabled = True
            assert cache.enabled is True

    def test_cache_preserves_metadata(self):
        """WHEN caching test cases, THEN should preserve metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = QueryCache(tmpdir)

            test_cases = [
                TestCase(
                    query="테스트 질문",
                    persona_type=PersonaType.PROFESSOR,
                    difficulty=DifficultyLevel.HARD,
                    query_type=QueryType.PROCEDURAL,
                    metadata={"key": "value", "number": 123},
                    expected_topics=["휴학", "등록"],
                    expected_regulations=["규정001"],
                )
            ]

            cache.set("metadata_test", test_cases)
            retrieved = cache.get("metadata_test")

            assert retrieved is not None
            assert retrieved[0].metadata == {"key": "value", "number": 123}
            assert retrieved[0].expected_topics == ["휴학", "등록"]
            assert retrieved[0].expected_regulations == ["규정001"]

    def test_cache_multiple_keys(self):
        """WHEN storing multiple cache entries, THEN should keep them separate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = QueryCache(tmpdir)

            # Store different entries
            cache.set(
                "key1",
                [
                    TestCase(
                        query="질문 1",
                        persona_type=PersonaType.FRESHMAN,
                        difficulty=DifficultyLevel.EASY,
                        query_type=QueryType.FACT_CHECK,
                    )
                ],
            )
            cache.set(
                "key2",
                [
                    TestCase(
                        query="질문 2",
                        persona_type=PersonaType.PROFESSOR,
                        difficulty=DifficultyLevel.HARD,
                        query_type=QueryType.COMPLEX,
                    )
                ],
            )

            result1 = cache.get("key1")
            result2 = cache.get("key2")

            assert result1 is not None
            assert result2 is not None
            assert result1[0].query == "질문 1"
            assert result2[0].query == "질문 2"
            assert result1[0].persona_type == PersonaType.FRESHMAN
            assert result2[0].persona_type == PersonaType.PROFESSOR
