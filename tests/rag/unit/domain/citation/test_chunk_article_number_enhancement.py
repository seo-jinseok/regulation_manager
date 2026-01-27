"""
Unit tests for Chunk article_number field enhancement.

These tests verify the new article_number field functionality
added to Chunk entities for Component 3 citation enhancement.
"""

from src.rag.domain.entities import Chunk, ChunkLevel


class TestChunkArticleNumberExtraction:
    """Tests for article number extraction in Chunk.from_json_node."""

    def test_extract_basic_article_number(self):
        """Extract 제N조 format into article_number field."""
        node = {
            "id": "test-1",
            "type": "article",
            "title": "제26조 (직원의 구분)",
            "text": "직원은 일반직, 기술직, 별정직으로 구분한다.",
            "chunk_level": "article",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Enhanced behavior: article_number field populated
        assert chunk.article_number == "제26조"
        assert chunk.title == "제26조 (직원의 구분)"

    def test_extract_sub_article_number(self):
        """Extract 제N조의M format into article_number field."""
        node = {
            "id": "test-2",
            "type": "article",
            "title": "제10조의2 (특별승급)",
            "text": "특별승급에 관한 사항은 다음과 같다.",
            "chunk_level": "article",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Enhanced behavior: sub-article extracted
        assert chunk.article_number == "제10조의2"
        assert chunk.title == "제10조의2 (특별승급)"

    def test_extract_chapter_number(self):
        """Extract 제N장 format into article_number field."""
        node = {
            "id": "test-3",
            "type": "chapter",
            "title": "제1장 총칙",
            "text": "本章의 目的은 다음과 같다.",
            "chunk_level": "chapter",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Enhanced behavior: chapter number extracted
        assert chunk.article_number == "제1장"
        assert chunk.title == "제1장 총칙"

    def test_extract_table_reference(self):
        """Extract 별표N format into article_number field."""
        node = {
            "id": "test-4",
            "type": "addendum",
            "title": "별표1 직원급별 봉급표",
            "text": "직원급별 봉급표는 다음과 같다.",
            "chunk_level": "addendum",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Enhanced behavior: table reference extracted
        assert chunk.article_number == "별표1"
        assert chunk.title == "별표1 직원급별 봉급표"

    def test_extract_form_reference(self):
        """Extract 서식N format into article_number field."""
        node = {
            "id": "test-5",
            "type": "addendum",
            "title": "서식1 휴직원부",
            "text": "휴직원부 서식은 다음과 같다.",
            "chunk_level": "addendum",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Enhanced behavior: form reference extracted
        assert chunk.article_number == "서식1"
        assert chunk.title == "서식1 휴직원부"

    def test_no_article_number_in_paragraph(self):
        """Paragraph-level chunks have None for article_number."""
        node = {
            "id": "test-6",
            "type": "paragraph",
            "title": "① 일반직",
            "text": "일반직은 정규직으로 한다.",
            "chunk_level": "paragraph",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # No article number for paragraph level
        assert chunk.article_number is None
        assert chunk.title == "① 일반직"

    def test_backward_compatibility_no_title(self):
        """Handle chunks without title gracefully."""
        node = {
            "id": "test-7",
            "type": "text",
            "text": "일반 텍스트입니다.",
            "chunk_level": "text",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Backward compatible: no crash, article_number is None
        assert chunk.article_number is None
        assert chunk.title == ""

    def test_large_article_numbers(self):
        """Extract large article numbers (e.g., 제100조)."""
        node = {
            "id": "test-8",
            "type": "article",
            "title": "제100조 (시행일)",
            "text": "본 규정은 2024년 1월 1일부터 시행한다.",
            "chunk_level": "article",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Large numbers supported
        assert chunk.article_number == "제100조"


class TestChunkMetadataRoundtrip:
    """Tests for article_number preservation in metadata roundtrip."""

    def test_article_number_in_to_metadata(self):
        """article_number field included in metadata."""
        node = {
            "id": "test-9",
            "type": "article",
            "title": "제26조 (직원의 구분)",
            "text": "직원은 일반직, 기술직, 별정직으로 구분한다.",
            "chunk_level": "article",
            "parent_path": ["직원복무규정"],
        }

        chunk = Chunk.from_json_node(node, rule_code="3-1-5")
        metadata = chunk.to_metadata()

        # Enhanced behavior: article_number in metadata
        assert "article_number" in metadata
        assert metadata["article_number"] == "제26조"

    def test_article_number_from_metadata_roundtrip(self):
        """article_number preserved in metadata roundtrip."""
        node = {
            "id": "test-10",
            "type": "article",
            "title": "제26조 (직원의 구분)",
            "text": "직원은 일반직, 기술직, 별정직으로 구분한다.",
            "chunk_level": "article",
            "parent_path": ["직원복무규정"],
        }

        original = Chunk.from_json_node(node, rule_code="3-1-5")
        metadata = original.to_metadata()
        restored = Chunk.from_metadata(
            doc_id=metadata["id"],
            text=original.text,
            metadata=metadata,
        )

        # Enhanced behavior: article_number preserved in roundtrip
        assert restored.article_number == original.article_number
        assert restored.article_number == "제26조"

    def test_empty_article_number_in_metadata(self):
        """Empty string used for None article_number in metadata."""
        node = {
            "id": "test-11",
            "type": "paragraph",
            "title": "① 일반직",
            "text": "일반직은 정규직으로 한다.",
            "chunk_level": "paragraph",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")
        metadata = chunk.to_metadata()

        # Empty string for None values
        assert metadata["article_number"] == ""

    def test_sub_article_roundtrip(self):
        """Sub-article numbers preserved in roundtrip."""
        node = {
            "id": "test-12",
            "type": "article",
            "title": "제10조의2 (특별승급)",
            "text": "특별승급에 관한 사항은 다음과 같다.",
            "chunk_level": "article",
        }

        original = Chunk.from_json_node(node, rule_code="test-code")
        metadata = original.to_metadata()
        restored = Chunk.from_metadata(
            doc_id=metadata["id"],
            text=original.text,
            metadata=metadata,
        )

        # Sub-article preserved
        assert restored.article_number == "제10조의2"

    def test_table_reference_roundtrip(self):
        """Table references preserved in roundtrip."""
        node = {
            "id": "test-13",
            "type": "addendum",
            "title": "별표1 직원급별 봉급표",
            "text": "직원급별 봉급표는 다음과 같다.",
            "chunk_level": "addendum",
        }

        original = Chunk.from_json_node(node, rule_code="test-code")
        metadata = original.to_metadata()
        restored = Chunk.from_metadata(
            doc_id=metadata["id"],
            text=original.text,
            metadata=metadata,
        )

        # Table reference preserved
        assert restored.article_number == "별표1"


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_existing_chunk_creation_still_works(self):
        """Existing chunk creation patterns still work."""
        # Old-style chunk creation without article_number
        chunk = Chunk(
            id="test-14",
            rule_code="test-code",
            level=ChunkLevel.ARTICLE,
            title="제26조",
            text="Some text",
            embedding_text="Some text",
            full_text="Full text",
            parent_path=[],
            token_count=10,
            keywords=[],
            is_searchable=True,
        )

        # article_number defaults to None
        assert chunk.article_number is None
        # Other fields work as before
        assert chunk.title == "제26조"
        assert chunk.level == ChunkLevel.ARTICLE

    def test_all_other_fields_unchanged(self):
        """article_number addition doesn't affect other fields."""
        node = {
            "id": "test-15",
            "type": "article",
            "title": "제26조 (직원의 구분)",
            "text": "직원은 일반직, 기술직, 별정직으로 구분한다.",
            "chunk_level": "article",
            "parent_path": ["직원복무규정", "제3장"],
            "token_count": 15,
            "keywords": [{"term": "직원", "weight": 0.9}],
            "is_searchable": True,
            "effective_date": "2024-01-01",
        }

        chunk = Chunk.from_json_node(node, rule_code="3-1-5")

        # All other fields work as before
        assert chunk.id == "test-15"
        assert chunk.rule_code == "3-1-5"
        assert chunk.level == ChunkLevel.ARTICLE
        assert chunk.text == "직원은 일반직, 기술직, 별정직으로 구분한다."
        assert chunk.parent_path == ["직원복무규정", "제3장"]
        assert chunk.token_count == 15
        assert chunk.is_searchable is True
        assert chunk.effective_date == "2024-01-01"
        assert len(chunk.keywords) == 1
        assert chunk.keywords[0].term == "직원"
        # New field also works
        assert chunk.article_number == "제26조"
