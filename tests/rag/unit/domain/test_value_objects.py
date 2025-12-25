"""
Unit tests for domain value objects.

TDD approach: These tests define the expected behavior of value objects.
"""

import pytest

from src.rag.domain.entities import ChunkLevel, RegulationStatus
from src.rag.domain.value_objects import (
    Query,
    SearchFilter,
    SyncResult,
    SyncState,
)


class TestQuery:
    """Tests for Query value object."""

    def test_valid_query(self):
        """Create a valid query."""
        query = Query(text="교원 연구년 자격")
        assert query.text == "교원 연구년 자격"
        assert query.include_abolished is False

    def test_query_with_abolished(self):
        """Query including abolished regulations."""
        query = Query(text="test", include_abolished=True)
        assert query.include_abolished is True

    def test_empty_query_raises(self):
        """Empty query text raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            Query(text="")

    def test_whitespace_query_raises(self):
        """Whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            Query(text="   ")

    def test_query_is_immutable(self):
        """Query is frozen (immutable)."""
        query = Query(text="test")
        with pytest.raises(AttributeError):
            query.text = "changed"  # type: ignore


class TestSearchFilter:
    """Tests for SearchFilter value object."""

    def test_empty_filter(self):
        """Empty filter has no constraints."""
        f = SearchFilter()
        assert f.status is None
        assert f.levels is None
        assert f.rule_codes is None

    def test_status_filter(self):
        """Filter by status."""
        f = SearchFilter(status=RegulationStatus.ACTIVE)
        assert f.status == RegulationStatus.ACTIVE

    def test_levels_filter(self):
        """Filter by chunk levels."""
        f = SearchFilter(levels=[ChunkLevel.ARTICLE, ChunkLevel.PARAGRAPH])
        assert len(f.levels) == 2

    def test_to_metadata_filter_empty(self):
        """Empty filter produces empty dict."""
        f = SearchFilter()
        meta = f.to_metadata_filter()
        assert meta == {}

    def test_to_metadata_filter_status(self):
        """Status filter in metadata."""
        f = SearchFilter(status=RegulationStatus.ACTIVE)
        meta = f.to_metadata_filter()
        assert meta["status"] == "active"

    def test_to_metadata_filter_levels(self):
        """Levels filter in metadata."""
        f = SearchFilter(levels=[ChunkLevel.ARTICLE])
        meta = f.to_metadata_filter()
        assert meta["level"] == {"$in": ["article"]}

    def test_to_metadata_filter_rule_codes(self):
        """Rule codes filter in metadata."""
        f = SearchFilter(rule_codes=["3-1-5", "3-1-6"])
        meta = f.to_metadata_filter()
        assert meta["rule_code"] == {"$in": ["3-1-5", "3-1-6"]}


class TestSyncResult:
    """Tests for SyncResult value object."""

    def test_total_processed(self):
        """Calculate total processed."""
        result = SyncResult(added=5, modified=10, removed=2, unchanged=100)
        assert result.total_processed == 117

    def test_has_changes_true(self):
        """has_changes is True when there are changes."""
        result = SyncResult(added=1, modified=0, removed=0, unchanged=100)
        assert result.has_changes is True

    def test_has_changes_false(self):
        """has_changes is False when no changes."""
        result = SyncResult(added=0, modified=0, removed=0, unchanged=100)
        assert result.has_changes is False

    def test_has_errors(self):
        """has_errors reflects error list."""
        result1 = SyncResult(errors=[])
        assert result1.has_errors is False

        result2 = SyncResult(errors=["Error 1"])
        assert result2.has_errors is True

    def test_str_representation(self):
        """String representation is readable."""
        result = SyncResult(added=5, modified=23, removed=2, unchanged=445)
        s = str(result)
        assert "5 added" in s
        assert "23 modified" in s
        assert "2 removed" in s


class TestSyncState:
    """Tests for SyncState value object."""

    def test_to_dict_and_back(self):
        """Serialize to dict and deserialize."""
        state = SyncState(
            last_sync="2025-01-15T10:30:00Z",
            json_file="규정집9-350.json",
            regulations={"3-1-5": "hash123", "3-1-6": "hash456"},
        )

        data = state.to_dict()
        restored = SyncState.from_dict(data)

        assert restored.last_sync == state.last_sync
        assert restored.json_file == state.json_file
        assert restored.regulations == state.regulations

    def test_empty_state(self):
        """Create empty state."""
        state = SyncState.empty()
        assert state.last_sync == ""
        assert state.json_file == ""
        assert state.regulations == {}
