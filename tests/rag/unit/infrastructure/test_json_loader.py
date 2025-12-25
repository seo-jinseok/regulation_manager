"""
Unit tests for JSON Document Loader.

Uses actual regulation JSON structure from the project.
"""

import json
import tempfile
from pathlib import Path

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
        chunks = loader.load_chunks_by_rule_codes(
            temp_json_file, {"3-1-5"}
        )

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

    def test_get_regulation_titles(self, temp_json_file):
        """Get regulation titles mapping."""
        loader = JSONDocumentLoader()
        titles = loader.get_regulation_titles(temp_json_file)

        assert titles["3-1-5"] == "교원인사규정"
        assert titles["3-1-99"] == "시간강사위촉규정【폐지】"
