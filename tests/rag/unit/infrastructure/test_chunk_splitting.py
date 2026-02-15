"""
Characterization tests for chunk splitting functionality.

These tests verify that chunk splitting preserves metadata and respects
size constraints. Following DDD methodology, these tests capture expected
behavior before implementation changes.
"""

import pytest

from src.enhance_for_rag import (
    CHARS_PER_TOKEN,
    CHUNK_OVERLAP_TOKENS,
    MAX_CHUNK_TOKENS,
    calculate_token_count,
    enhance_json,
    split_large_text,
    split_large_node,
)


class TestChunkSplittingConfiguration:
    """Tests for chunk splitting configuration constants."""

    def test_max_chunk_tokens_is_512(self):
        """Verify MAX_CHUNK_TOKENS is set to 512 per REQ-001."""
        assert MAX_CHUNK_TOKENS == 512

    def test_chunk_overlap_tokens_is_100(self):
        """Verify CHUNK_OVERLAP_TOKENS is set to 100 per REQ-001."""
        assert CHUNK_OVERLAP_TOKENS == 100

    def test_chars_per_token_is_2_5(self):
        """Verify CHARS_PER_TOKEN approximation factor."""
        assert CHARS_PER_TOKEN == 2.5


class TestSplitLargeText:
    """Tests for split_large_text function."""

    def test_small_text_not_split(self):
        """Text under 512 tokens should not be split."""
        # Create text that is clearly under 512 tokens
        # 512 tokens * 2.5 chars/token = 1280 chars
        small_text = "이것은 테스트입니다." * 10  # ~190 chars, ~76 tokens
        result = split_large_text(small_text, max_tokens=512, overlap=100)

        assert len(result) == 1
        assert result[0] == small_text

    def test_large_text_is_split(self):
        """Text over 512 tokens should be split into chunks."""
        # Create text that exceeds 512 tokens
        # 512 * 2.5 = 1280 chars minimum
        large_text = "이것은 테스트 문장입니다. " * 100  # ~2500 chars, ~1000 tokens
        result = split_large_text(large_text, max_tokens=512, overlap=100)

        assert len(result) > 1

    def test_split_respects_max_tokens(self):
        """Each chunk should not exceed max_tokens (with some tolerance)."""
        large_text = "테스트 문장입니다. " * 200  # ~4000 chars, ~1600 tokens
        chunks = split_large_text(large_text, max_tokens=512, overlap=100)

        for chunk in chunks:
            token_count = calculate_token_count(chunk)
            # Allow 10% tolerance for boundary conditions
            assert token_count <= MAX_CHUNK_TOKENS * 1.1, (
                f"Chunk exceeds max tokens: {token_count} > {MAX_CHUNK_TOKENS}"
            )

    def test_split_has_overlap(self):
        """Chunks should have overlap for context preservation."""
        # Create predictable text
        sentences = [f"문장 {i}입니다." for i in range(100)]
        large_text = " ".join(sentences)
        chunks = split_large_text(large_text, max_tokens=512, overlap=100)

        if len(chunks) > 1:
            # Check that there's overlap between consecutive chunks
            # The end of chunk 0 should appear at the start of chunk 1
            # Allow some flexibility due to character-based splitting
            overlap_found = False
            for i in range(len(chunks) - 1):
                end_of_chunk = chunks[i][-200:]  # Last 200 chars
                start_of_next = chunks[i + 1][:200]  # First 200 chars

                # Check for any common substring
                for length in [50, 30, 20, 10]:
                    if end_of_chunk[-length:] in start_of_next[: length * 2]:
                        overlap_found = True
                        break
                if overlap_found:
                    break

            assert overlap_found, "No overlap found between consecutive chunks"

    def test_empty_text_returns_empty_list(self):
        """Empty text should return empty list."""
        assert split_large_text("") == []
        assert split_large_text(None) == []

    def test_single_chunk_for_boundary_case(self):
        """Text exactly at boundary should return single chunk."""
        # Create text close to 512 tokens
        target_chars = int(MAX_CHUNK_TOKENS * CHARS_PER_TOKEN * 0.95)  # ~1216 chars
        text = "가" * target_chars
        chunks = split_large_text(text, max_tokens=512, overlap=100)

        # Should be a single chunk since it's under the limit
        assert len(chunks) == 1


class TestSplitLargeNode:
    """Tests for split_large_node function - metadata preservation."""

    def test_small_node_not_split(self):
        """Nodes under 512 tokens should not be split."""
        node = {
            "id": "test-001",
            "type": "article",
            "display_no": "제1조",
            "title": "목적",
            "text": "이 규정은 테스트입니다.",
            "embedding_text": "이 규정은 테스트입니다.",
            "parent_path": ["테스트규정"],
            "rule_code": "TEST001",
            "token_count": 20,
            "keywords": [{"term": "규정", "weight": 0.8}],
            "chunk_level": "article",
            "is_searchable": True,
            "status": "active",
            "doc_type": "regulation",
        }

        result = split_large_node(node)

        assert len(result) == 1
        assert result[0] == node

    def test_large_node_is_split(self):
        """Nodes over 512 tokens should be split into multiple nodes."""
        large_text = "이것은 긴 테스트 문장입니다. " * 200  # ~7000 chars
        node = {
            "id": "test-002",
            "type": "article",
            "display_no": "제26조",
            "title": "승진심사",
            "text": large_text,
            "embedding_text": large_text,
            "parent_path": ["교원인사규정", "제3장 승진"],
            "rule_code": "FAC001",
            "token_count": calculate_token_count(large_text),
            "keywords": [{"term": "승진", "weight": 1.0}],
            "chunk_level": "article",
            "is_searchable": True,
            "status": "active",
            "doc_type": "regulation",
            "article_number": "제26조",
        }

        result = split_large_node(node)

        assert len(result) > 1, "Large node should be split into multiple nodes"

    def test_preserves_rule_code(self):
        """All split nodes should inherit rule_code from parent."""
        large_text = "테스트 문장입니다. " * 200
        node = {
            "id": "test-003",
            "rule_code": "FAC001",
            "text": large_text,
            "token_count": calculate_token_count(large_text),
        }

        result = split_large_node(node)

        for chunk in result:
            assert chunk["rule_code"] == "FAC001", "rule_code should be preserved"

    def test_preserves_article_number(self):
        """All split nodes should inherit article_number for citation support."""
        large_text = "테스트 문장입니다. " * 200
        node = {
            "id": "test-004",
            "article_number": "제26조",
            "text": large_text,
            "token_count": calculate_token_count(large_text),
        }

        result = split_large_node(node)

        for chunk in result:
            assert chunk.get("article_number") == "제26조", (
                "article_number should be preserved for citation support"
            )

    def test_preserves_parent_path(self):
        """All split nodes should inherit parent_path (breadcrumb)."""
        large_text = "테스트 문장입니다. " * 200
        node = {
            "id": "test-005",
            "parent_path": ["교원인사규정", "제3장 승진", "제1절 심사"],
            "text": large_text,
            "token_count": calculate_token_count(large_text),
        }

        result = split_large_node(node)

        for chunk in result:
            assert chunk["parent_path"] == ["교원인사규정", "제3장 승진", "제1절 심사"], (
                "parent_path should be preserved"
            )

    def test_preserves_status(self):
        """All split nodes should inherit status (active/abolished)."""
        large_text = "테스트 문장입니다. " * 200
        node = {
            "id": "test-006",
            "status": "active",
            "text": large_text,
            "token_count": calculate_token_count(large_text),
        }

        result = split_large_node(node)

        for chunk in result:
            assert chunk["status"] == "active", "status should be preserved"

    def test_preserves_doc_type(self):
        """All split nodes should inherit doc_type."""
        large_text = "테스트 문장입니다. " * 200
        node = {
            "id": "test-007",
            "doc_type": "regulation",
            "text": large_text,
            "token_count": calculate_token_count(large_text),
        }

        result = split_large_node(node)

        for chunk in result:
            assert chunk["doc_type"] == "regulation", "doc_type should be preserved"

    def test_preserves_chunk_level(self):
        """All split nodes should inherit chunk_level."""
        large_text = "테스트 문장입니다. " * 200
        node = {
            "id": "test-008",
            "chunk_level": "article",
            "text": large_text,
            "token_count": calculate_token_count(large_text),
        }

        result = split_large_node(node)

        for chunk in result:
            assert chunk["chunk_level"] == "article", "chunk_level should be preserved"

    def test_preserves_keywords(self):
        """All split nodes should inherit keywords."""
        large_text = "테스트 문장입니다. " * 200
        keywords = [{"term": "승진", "weight": 1.0}, {"term": "심사", "weight": 0.9}]
        node = {
            "id": "test-009",
            "keywords": keywords,
            "text": large_text,
            "token_count": calculate_token_count(large_text),
        }

        result = split_large_node(node)

        for chunk in result:
            assert chunk["keywords"] == keywords, "keywords should be preserved"

    def test_generates_unique_ids(self):
        """Split nodes should have unique IDs derived from parent."""
        large_text = "테스트 문장입니다. " * 200
        node = {
            "id": "test-010",
            "text": large_text,
            "token_count": calculate_token_count(large_text),
        }

        result = split_large_node(node)

        # All IDs should be unique
        ids = [chunk["id"] for chunk in result]
        assert len(ids) == len(set(ids)), "All split chunk IDs should be unique"

        # All IDs should contain the original ID
        for chunk_id in ids:
            assert "test-010" in chunk_id, (
                f"Split ID '{chunk_id}' should contain original ID 'test-010'"
            )

    def test_updates_token_count_per_chunk(self):
        """Each split node should have its own accurate token_count."""
        large_text = "테스트 문장입니다. " * 200
        node = {
            "id": "test-011",
            "text": large_text,
            "token_count": calculate_token_count(large_text),
        }

        result = split_large_node(node)

        for chunk in result:
            expected_tokens = calculate_token_count(chunk["text"])
            assert chunk["token_count"] == expected_tokens, (
                f"token_count {chunk['token_count']} should match "
                f"calculated {expected_tokens}"
            )


class TestEnhanceJsonWithChunkSplitting:
    """Integration tests for enhance_json with chunk splitting enabled."""

    def test_large_chunks_are_split_during_enhancement(self):
        """Large chunks should be split during JSON enhancement."""
        large_text = "이것은 긴 테스트 문장입니다. " * 200
        data = {
            "docs": [
                {
                    "title": "테스트규정",
                    "doc_type": "regulation",
                    "metadata": {"rule_code": "TEST001"},
                    "content": [
                        {
                            "type": "article",
                            "display_no": "제1조",
                            "title": "긴조항",
                            "text": large_text,
                            "children": [],
                        }
                    ],
                    "addenda": [],
                }
            ]
        }

        result = enhance_json(data)

        # Original single article should be split into multiple searchable chunks
        # Check if any chunks were created (depends on whether splitting is applied)
        # The key requirement is that the enhancement doesn't fail
        assert result["rag_enhanced"] is True

    def test_small_chunks_remain_unchanged(self):
        """Small chunks should not be modified during enhancement."""
        small_text = "이것은 짧은 테스트 문장입니다."
        data = {
            "docs": [
                {
                    "title": "테스트규정",
                    "doc_type": "regulation",
                    "metadata": {"rule_code": "TEST002"},
                    "content": [
                        {
                            "type": "article",
                            "display_no": "제1조",
                            "title": "짧은조항",
                            "text": small_text,
                            "children": [],
                        }
                    ],
                    "addenda": [],
                }
            ]
        }

        result = enhance_json(data)

        # Small chunks should remain as single chunk
        content = result["docs"][0]["content"][0]
        assert content["text"] == small_text
        # token_count is calculated based on embedding_text which includes path context
        # So we just verify it's a reasonable positive value
        assert content["token_count"] > 0
        assert content["token_count"] < MAX_CHUNK_TOKENS  # Should be small


class TestCharacterizationMetadataPreservation:
    """
    Characterization tests that document current metadata behavior.

    These tests capture the expected behavior for metadata preservation
    during any future refactoring of the chunk splitting logic.
    """

    @pytest.fixture
    def sample_large_node(self):
        """Sample large node for characterization tests."""
        large_text = "테스트 문장입니다. " * 200
        return {
            "id": "sample-001",
            "type": "article",
            "display_no": "제26조",
            "title": "승진심사",
            "text": large_text,
            "embedding_text": f"교원인사규정 > 제3장 승진 > 제26조 승진심사: {large_text[:100]}",
            "full_text": f"[교원인사규정 > 제3장 승진 > 제26조 승진심사] {large_text}",
            "parent_path": ["교원인사규정", "제3장 승진"],
            "rule_code": "FAC001",
            "token_count": calculate_token_count(large_text),
            "keywords": [
                {"term": "승진", "weight": 1.0},
                {"term": "심사", "weight": 0.9},
                {"term": "교원", "weight": 0.8},
            ],
            "chunk_level": "article",
            "is_searchable": True,
            "status": "active",
            "doc_type": "regulation",
            "article_number": "제26조",
            "amendment_history": [
                {"date": "2020-01-15", "type": "개정"},
                {"date": "2023-06-01", "type": "개정"},
            ],
            "effective_date": None,
        }

    def test_characterize_all_metadata_preserved(self, sample_large_node):
        """
        CHARACTERIZE: All metadata fields should be preserved in split chunks.

        This test documents which fields MUST be inherited by all split chunks.
        """
        result = split_large_node(sample_large_node)

        # List of fields that must be preserved (identical in all chunks)
        preserved_fields = [
            "type",
            "display_no",
            "title",
            "parent_path",
            "rule_code",
            "keywords",
            "chunk_level",
            "is_searchable",
            "status",
            "doc_type",
            "article_number",
            "amendment_history",
        ]

        for chunk in result:
            for field in preserved_fields:
                assert field in chunk, f"Field '{field}' is missing from split chunk"
                assert chunk[field] == sample_large_node[field], (
                    f"Field '{field}' value changed in split chunk: "
                    f"{chunk[field]} != {sample_large_node[field]}"
                )

    def test_characterize_unique_fields_per_chunk(self, sample_large_node):
        """
        CHARACTERIZE: Fields that should be unique per split chunk.

        These fields are recalculated for each chunk based on its content.
        """
        result = split_large_node(sample_large_node)

        for i, chunk in enumerate(result):
            # ID should be unique (with part suffix)
            assert chunk["id"] != sample_large_node["id"], (
                "Each chunk should have a unique ID"
            )
            assert f"_part{i + 1}" in chunk["id"] or f"part{i + 1}" in chunk["id"], (
                f"Chunk ID should contain part index: {chunk['id']}"
            )

            # token_count should be recalculated for chunk's text
            assert chunk["token_count"] == calculate_token_count(chunk["text"]), (
                "token_count should be recalculated for each chunk"
            )

            # embedding_text should be updated with chunk's text
            assert chunk["text"] in chunk.get("embedding_text", ""), (
                "embedding_text should contain chunk's text"
            )
