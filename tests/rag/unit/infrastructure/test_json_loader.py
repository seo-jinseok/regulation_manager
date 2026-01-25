"""
Unit tests for JSON Document Loader.

Uses actual regulation JSON structure from the project.
"""

import json
import tempfile

import pytest

from src.rag.domain.entities import ChunkLevel
from src.rag.infrastructure.json_loader import JSONDocumentLoader


@pytest.fixture
def sample_regulation_json() -> dict:
    """Sample JSON matching the regulation schema v2.0."""
    return {
        "schema_version": "v4",
        "rag_enhanced": True,
        "rag_schema_version": "2.0",
        "docs": [
            {
                "title": "차례",
                "doc_type": "toc",
                "is_index_duplicate": True,
                "content": [],
            },
            {
                "title": "교원인사규정",
                "doc_type": "regulation",
                "status": "active",
                "metadata": {"rule_code": "3-1-5"},
                "content": [
                    {
                        "id": "node-1",
                        "type": "article",
                        "title": "제1조",
                        "text": "이 규정은 교원의 인사에 관한 사항을 정함을 목적으로 한다.",
                        "embedding_text": "이 규정은 교원의 인사에 관한 사항을 정함을 목적으로 한다.",
                        "full_text": "[교원인사규정 > 제1조] 이 규정은...",
                        "chunk_level": "article",
                        "token_count": 20,
                        "keywords": [{"term": "교원", "weight": 0.9}],
                        "parent_path": ["교원인사규정"],
                        "is_searchable": True,
                        "children": [
                            {
                                "id": "node-1-1",
                                "type": "paragraph",
                                "title": "",
                                "text": "전임교원에 적용한다.",
                                "embedding_text": "전임교원에 적용한다.",
                                "chunk_level": "paragraph",
                                "token_count": 8,
                                "keywords": [{"term": "교원", "weight": 0.9}],
                                "parent_path": ["교원인사규정", "제1조"],
                                "is_searchable": True,
                                "children": [],
                            }
                        ],
                    },
                    {
                        "id": "node-2",
                        "type": "article",
                        "title": "제2조",
                        "text": "",  # Empty text, should be skipped
                        "chunk_level": "article",
                        "is_searchable": False,
                        "children": [],
                    },
                ],
                "addenda": [
                    {
                        "id": "addendum-1",
                        "type": "addendum_item",
                        "title": "",
                        "text": "이 규정은 2024년 1월 1일부터 시행한다.",
                        "embedding_text": "이 규정은 2024년 1월 1일부터 시행한다.",
                        "chunk_level": "addendum_item",
                        "token_count": 12,
                        "keywords": [{"term": "시행", "weight": 0.6}],
                        "parent_path": ["교원인사규정", "부칙"],
                        "is_searchable": True,
                        "effective_date": "2024-01-01",
                        "children": [],
                    }
                ],
            },
            {
                "title": "시간강사위촉규정【폐지】",
                "doc_type": "regulation",
                "status": "abolished",
                "metadata": {"rule_code": "3-1-99"},
                "content": [
                    {
                        "id": "node-abolished",
                        "type": "article",
                        "title": "제1조",
                        "text": "폐지된 규정 내용",
                        "embedding_text": "폐지된 규정 내용",
                        "chunk_level": "article",
                        "is_searchable": True,
                        "children": [],
                    }
                ],
                "addenda": [],
            },
        ],
    }


@pytest.fixture
def temp_json_file(sample_regulation_json) -> str:
    """Create temporary JSON file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(sample_regulation_json, f, ensure_ascii=False)
        return f.name


class TestJSONDocumentLoader:
    """Tests for JSONDocumentLoader."""

    def test_load_all_chunks(self, temp_json_file):
        """Load all chunks from JSON file."""
        loader = JSONDocumentLoader()
        chunks = loader.load_all_chunks(temp_json_file)

        # Should skip TOC, include 2 regulations
        # 교원인사규정: 2 chunks (article + paragraph) + 1 addendum
        # 폐지규정: 1 chunk
        assert len(chunks) == 4

    def test_load_chunks_skips_toc(self, temp_json_file):
        """TOC/index documents are skipped."""
        loader = JSONDocumentLoader()
        chunks = loader.load_all_chunks(temp_json_file)

        # No chunks from TOC
        assert all(c.rule_code != "" for c in chunks)

    def test_load_chunks_extracts_rule_code(self, temp_json_file):
        """Rule code is extracted correctly."""
        loader = JSONDocumentLoader()
        chunks = loader.load_all_chunks(temp_json_file)

        rule_codes = {c.rule_code for c in chunks}
        assert "3-1-5" in rule_codes
        assert "3-1-99" in rule_codes

    def test_load_chunks_includes_addenda(self, temp_json_file):
        """Addenda are included in chunks."""
        loader = JSONDocumentLoader()
        chunks = loader.load_all_chunks(temp_json_file)

        addendum_chunks = [c for c in chunks if c.level == ChunkLevel.ADDENDUM_ITEM]
        assert len(addendum_chunks) == 1
        assert addendum_chunks[0].effective_date == "2024-01-01"

    def test_load_chunks_skips_non_searchable(self, temp_json_file):
        """Non-searchable nodes are skipped."""
        loader = JSONDocumentLoader()
        chunks = loader.load_all_chunks(temp_json_file)

        # node-2 has is_searchable=False and empty text
        ids = [c.id for c in chunks]
        assert "node-2" not in ids

    def test_load_chunks_recursive(self, temp_json_file):
        """Nested children are extracted."""
        loader = JSONDocumentLoader()
        chunks = loader.load_all_chunks(temp_json_file)

        # node-1-1 is a child of node-1
        ids = [c.id for c in chunks]
        assert "node-1-1" in ids

    def test_load_chunks_by_rule_codes(self, temp_json_file):
        """Load only specific rule codes."""
        loader = JSONDocumentLoader()
        chunks = loader.load_chunks_by_rule_codes(temp_json_file, {"3-1-5"})

        # Only 교원인사규정 chunks
        assert all(c.rule_code == "3-1-5" for c in chunks)
        assert len(chunks) == 3  # 2 content + 1 addendum

    def test_compute_state(self, temp_json_file):
        """Compute sync state for incremental sync."""
        loader = JSONDocumentLoader()
        state = loader.compute_state(temp_json_file)

        assert state.json_file.endswith(".json")
        assert "3-1-5" in state.regulations
        assert "3-1-99" in state.regulations
        assert len(state.regulations["3-1-5"]) == 16  # SHA256 truncated

    def test_compute_state_hash_changes(self, sample_regulation_json):
        """Hash changes when content changes."""
        loader = JSONDocumentLoader()

        # Create first file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path1 = f.name

        state1 = loader.compute_state(path1)
        hash1 = state1.regulations["3-1-5"]

        # Modify content
        sample_regulation_json["docs"][1]["content"][0]["text"] = "Modified text"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path2 = f.name

        state2 = loader.compute_state(path2)
        hash2 = state2.regulations["3-1-5"]

        # Hash should be different
        assert hash1 != hash2

    def test_compute_state_ignores_metadata_changes(self, sample_regulation_json):
        """Metadata-only changes should not affect content hash."""
        loader = JSONDocumentLoader()

        sample_regulation_json["docs"][1]["metadata"]["scan_date"] = "2025-01-01"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path1 = f.name

        state1 = loader.compute_state(path1)
        hash1 = state1.regulations["3-1-5"]

        sample_regulation_json["docs"][1]["metadata"]["scan_date"] = "2025-01-02"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path2 = f.name

        state2 = loader.compute_state(path2)
        hash2 = state2.regulations["3-1-5"]

        assert hash1 == hash2

    def test_get_regulation_titles(self, temp_json_file):
        """Get regulation titles mapping."""
        loader = JSONDocumentLoader()
        titles = loader.get_regulation_titles(temp_json_file)

        assert titles["3-1-5"] == "교원인사규정"
        assert titles["3-1-99"] == "시간강사위촉규정【폐지】"


class TestJSONDocumentLoaderCache:
    """Tests for caching behavior."""

    def test_clear_cache(self):
        """Test clear_cache clears the internal cache."""
        loader = JSONDocumentLoader()
        # Add something to cache by loading a file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"docs": []}, f)
            path = f.name
        loader.load_all_chunks(path)

        # Cache should have the file
        assert path in loader._cache

        # Clear cache
        loader.clear_cache()
        assert len(loader._cache) == 0


class TestJSONDocumentLoaderRegulationDoc:
    """Tests for get_regulation_doc method."""

    def test_get_regulation_doc_by_rule_code(self, sample_regulation_json):
        """Get regulation document by rule code."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        doc = loader.get_regulation_doc(path, "3-1-5")

        assert doc is not None
        assert doc["title"] == "교원인사규정"
        assert doc["metadata"]["rule_code"] == "3-1-5"

    def test_get_regulation_doc_by_title(self, sample_regulation_json):
        """Get regulation document by title."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        doc = loader.get_regulation_doc(path, "교원인사규정")

        assert doc is not None
        assert doc["metadata"]["rule_code"] == "3-1-5"

    def test_get_regulation_doc_not_found(self, sample_regulation_json):
        """Get regulation document returns None when not found."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        doc = loader.get_regulation_doc(path, "nonexistent")

        assert doc is None

    def test_get_regulation_doc_skips_non_regulation(self, sample_regulation_json):
        """Get regulation doc skips TOC documents."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        doc = loader.get_regulation_doc(path, "차례")

        # Should not find the TOC document
        assert doc is None


class TestJSONDocumentLoaderRegulationOverview:
    """Tests for get_regulation_overview method."""

    def test_get_regulation_overview_active(self, sample_regulation_json):
        """Get regulation overview for active regulation."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        overview = loader.get_regulation_overview(path, "3-1-5")

        assert overview is not None
        assert overview.rule_code == "3-1-5"
        assert overview.title == "교원인사규정"
        assert overview.status.value == "active"
        # Count includes both searchable and non-searchable articles
        assert overview.article_count >= 1
        assert overview.has_addenda is True

    def test_get_regulation_overview_abolished(self, sample_regulation_json):
        """Get regulation overview for abolished regulation."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        overview = loader.get_regulation_overview(path, "3-1-99")

        assert overview is not None
        assert overview.status.value == "abolished"

    def test_get_regulation_overview_not_found(self, sample_regulation_json):
        """Get regulation overview returns None when not found."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        overview = loader.get_regulation_overview(path, "nonexistent")

        assert overview is None

    def test_get_regulation_overview_with_chapters(self):
        """Test overview extracts chapter information."""
        data = {
            "docs": [
                {
                    "title": "테스트규정",
                    "doc_type": "regulation",
                    "metadata": {"rule_code": "TEST-001"},
                    "content": [
                        {
                            "type": "chapter",
                            "display_no": "제1장",
                            "title": "총칙",
                            "children": [
                                {
                                    "type": "article",
                                    "display_no": "제1조",
                                    "title": "목적",
                                },
                                {
                                    "type": "article",
                                    "display_no": "제2조",
                                    "title": "정의",
                                },
                            ],
                        },
                        {
                            "type": "chapter",
                            "display_no": "제2장",
                            "title": "인사",
                            "children": [
                                {
                                    "type": "article",
                                    "display_no": "제3조",
                                    "title": "임용",
                                },
                            ],
                        },
                    ],
                    "addenda": [],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        overview = loader.get_regulation_overview(path, "TEST-001")

        assert overview is not None
        assert len(overview.chapters) == 2
        assert overview.chapters[0].display_no == "제1장"
        assert overview.chapters[0].title == "총칙"
        # Article range for chapter 1 should be 제1조~제2조
        assert overview.chapters[0].article_range == "제1조~제2조"
        assert overview.chapters[1].article_range == "제3조"


class TestJSONDocumentLoaderFindCandidates:
    """Tests for find_regulation_candidates method."""

    def test_find_regulation_candidates(self, sample_regulation_json):
        """Find regulation candidates by query."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        candidates = loader.find_regulation_candidates(path, "교원인사")

        assert len(candidates) > 0
        codes = [c[0] for c in candidates]
        assert "3-1-5" in codes

    def test_find_candidates_sorts_by_length(self, sample_regulation_json):
        """Candidates are sorted by title length difference."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        candidates = loader.find_regulation_candidates(path, "교원인사")

        # First candidate should be closest match
        assert candidates[0][0] == "3-1-5"

    def test_find_candidates_empty_query(self, sample_regulation_json):
        """Empty query returns all candidates."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        candidates = loader.find_regulation_candidates(path, "")

        # Should return all regulations
        assert len(candidates) >= 2


class TestJSONDocumentLoaderGetAllRegulations:
    """Tests for get_all_regulations method."""

    def test_get_all_regulations(self, sample_regulation_json):
        """Get all regulation metadata."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        regulations = loader.get_all_regulations(path)

        assert len(regulations) == 2  # 2 regulations (excluding TOC)

        codes = [r[0] for r in regulations]
        titles = [r[1] for r in regulations]

        assert "3-1-5" in codes
        assert "3-1-99" in codes
        assert "교원인사규정" in titles

    def test_get_all_regulations_skips_non_regulation(self):
        """get_all_regulations skips non-regulation documents."""
        data = {
            "docs": [
                {
                    "title": "TOC",
                    "doc_type": "toc",
                    "metadata": {"rule_code": "N/A"},
                },
                {
                    "title": "Regulation",
                    "doc_type": "regulation",
                    "metadata": {"rule_code": "TEST-001"},
                },
                {
                    "title": "Index",
                    "doc_type": "index",
                    "metadata": {"rule_code": "N/A"},
                },
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        regulations = loader.get_all_regulations(path)

        assert len(regulations) == 1
        assert regulations[0][0] == "TEST-001"


class TestJSONDocumentLoaderEdgeCases:
    """Tests for edge cases and error handling."""

    def test_load_chunks_skips_docs_without_rule_code(self):
        """Documents without rule code are skipped."""
        data = {
            "docs": [
                {
                    "title": "No Rule Code",
                    "doc_type": "regulation",
                    "metadata": {},
                    "content": [
                        {
                            "id": "node-1",
                            "type": "article",
                            "title": "제1조",
                            "text": "내용",
                            "embedding_text": "내용",
                            "is_searchable": True,
                        }
                    ],
                    "addenda": [],
                }
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        chunks = loader.load_all_chunks(path)

        # Should skip document without rule code
        assert len(chunks) == 0

    def test_load_chunks_by_rule_codes_empty_set(self, sample_regulation_json):
        """Empty rule codes set returns no chunks."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(sample_regulation_json, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        chunks = loader.load_chunks_by_rule_codes(path, set())

        assert len(chunks) == 0

    def test_compute_state_skips_index_duplicates(self):
        """Index duplicate documents are skipped in state computation."""
        data = {
            "docs": [
                {
                    "title": "Index Duplicate",
                    "is_index_duplicate": True,
                    "doc_type": "toc",
                    "metadata": {"rule_code": "INDEX-001"},
                },
                {
                    "title": "Regulation",
                    "doc_type": "regulation",
                    "metadata": {"rule_code": "REG-001"},
                },
            ]
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)
            path = f.name

        loader = JSONDocumentLoader()
        state = loader.compute_state(path)

        # Should only have REG-001, not INDEX-001
        assert "REG-001" in state.regulations
        assert "INDEX-001" not in state.regulations
