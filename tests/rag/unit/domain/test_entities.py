"""
Unit tests for domain entities.

TDD approach: These tests define the expected behavior of entities.
"""

import pytest

from src.rag.domain.entities import (
    Chunk,
    ChunkLevel,
    Keyword,
    Regulation,
    RegulationStatus,
    SearchResult,
)


class TestChunkLevel:
    """Tests for ChunkLevel enum."""

    def test_from_string_valid(self):
        """Valid string converts to correct enum."""
        assert ChunkLevel.from_string("article") == ChunkLevel.ARTICLE
        assert ChunkLevel.from_string("paragraph") == ChunkLevel.PARAGRAPH
        assert ChunkLevel.from_string("chapter") == ChunkLevel.CHAPTER

    def test_from_string_invalid_defaults_to_text(self):
        """Invalid string defaults to TEXT."""
        assert ChunkLevel.from_string("unknown") == ChunkLevel.TEXT
        assert ChunkLevel.from_string("") == ChunkLevel.TEXT


class TestKeyword:
    """Tests for Keyword entity."""

    def test_from_dict(self):
        """Create Keyword from dict."""
        data = {"term": "교원", "weight": 0.9}
        kw = Keyword.from_dict(data)
        assert kw.term == "교원"
        assert kw.weight == 0.9


class TestChunk:
    """Tests for Chunk entity."""

    @pytest.fixture
    def sample_node(self) -> dict:
        """Sample JSON node for testing."""
        return {
            "id": "abc-123",
            "type": "article",
            "title": "제1조",
            "text": "본 규정의 목적은 다음과 같다.",
            "embedding_text": "본 규정의 목적은 다음과 같다.",
            "full_text": "[학칙 > 제1장 > 제1조] 본 규정의 목적은 다음과 같다.",
            "chunk_level": "article",
            "token_count": 15,
            "keywords": [{"term": "규정", "weight": 0.8}],
            "parent_path": ["학칙", "제1장"],
            "is_searchable": True,
            "effective_date": None,
        }

    def test_from_json_node_basic(self, sample_node):
        """Create Chunk from JSON node."""
        chunk = Chunk.from_json_node(sample_node, rule_code="3-1-5")

        assert chunk.id == "abc-123"
        assert chunk.rule_code == "3-1-5"
        assert chunk.level == ChunkLevel.ARTICLE
        assert chunk.title == "제1조"
        assert chunk.text == "본 규정의 목적은 다음과 같다."
        assert chunk.embedding_text == "본 규정의 목적은 다음과 같다."
        assert chunk.token_count == 15
        assert chunk.is_searchable is True
        assert len(chunk.keywords) == 1
        assert chunk.keywords[0].term == "규정"

    def test_from_json_node_with_effective_date(self, sample_node):
        """Chunk with effective_date."""
        sample_node["effective_date"] = "2024-01-01"
        chunk = Chunk.from_json_node(sample_node, rule_code="3-1-5")
        assert chunk.effective_date == "2024-01-01"

    def test_from_json_node_abolished_status(self, sample_node):
        """Chunk with abolished status from title."""
        sample_node["title"] = "시간강사위촉규정【폐지】"
        chunk = Chunk.from_json_node(sample_node, rule_code="3-1-5")
        assert chunk.status == RegulationStatus.ABOLISHED

    def test_from_json_node_defaults(self):
        """Chunk with minimal fields uses defaults."""
        node = {"id": "xyz", "text": "Some text"}
        chunk = Chunk.from_json_node(node, rule_code="1-0-1")

        assert chunk.id == "xyz"
        assert chunk.level == ChunkLevel.TEXT
        assert chunk.token_count == 0
        assert chunk.keywords == []
        assert chunk.status == RegulationStatus.ACTIVE

    def test_to_metadata(self, sample_node):
        """Convert Chunk to metadata dict."""
        chunk = Chunk.from_json_node(sample_node, rule_code="3-1-5")
        meta = chunk.to_metadata()

        assert meta["id"] == "abc-123"
        assert meta["rule_code"] == "3-1-5"
        assert meta["level"] == "article"
        assert meta["status"] == "active"
        assert "학칙 > 제1장" in meta["parent_path"]


class TestRegulation:
    """Tests for Regulation entity."""

    @pytest.fixture
    def sample_doc(self) -> dict:
        """Sample JSON document."""
        return {
            "title": "교원인사규정",
            "metadata": {"rule_code": "3-1-5"},
            "status": "active",
            "content": [],
        }

    def test_from_json_doc_active(self, sample_doc):
        """Create active Regulation from JSON doc."""
        reg = Regulation.from_json_doc(sample_doc)

        assert reg.rule_code == "3-1-5"
        assert reg.title == "교원인사규정"
        assert reg.status == RegulationStatus.ACTIVE
        assert reg.chunks == []

    def test_from_json_doc_abolished(self, sample_doc):
        """Create abolished Regulation from JSON doc."""
        sample_doc["status"] = "abolished"
        reg = Regulation.from_json_doc(sample_doc)
        assert reg.status == RegulationStatus.ABOLISHED


class TestSearchResult:
    """Tests for SearchResult entity."""

    def test_sorting_by_score(self):
        """SearchResults sort by score descending."""
        chunk1 = Chunk(
            id="1",
            rule_code="1",
            level=ChunkLevel.TEXT,
            title="",
            text="",
            embedding_text="",
            full_text="",
            parent_path=[],
            token_count=0,
            keywords=[],
            is_searchable=True,
        )
        chunk2 = Chunk(
            id="2",
            rule_code="2",
            level=ChunkLevel.TEXT,
            title="",
            text="",
            embedding_text="",
            full_text="",
            parent_path=[],
            token_count=0,
            keywords=[],
            is_searchable=True,
        )

        result1 = SearchResult(chunk=chunk1, score=0.8)
        result2 = SearchResult(chunk=chunk2, score=0.9)

        sorted_results = sorted([result1, result2])

        assert sorted_results[0].score == 0.9  # Higher score first
        assert sorted_results[1].score == 0.8
