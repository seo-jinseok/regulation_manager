"""
Characterization tests for Chunk article number extraction.

These tests capture the CURRENT BEHAVIOR of how article numbers
are stored and extracted from Chunk entities. They document
what IS, not what SHOULD BE.

Purpose: Ensure behavior preservation during refactoring.
"""

from src.rag.domain.entities import Chunk


class TestChunkTitleArticleNumberFormats:
    """Characterize how article numbers appear in Chunk.title."""

    def test_characterize_basic_article_number_in_title(self):
        """
        CHARACTERIZE: Basic article number format in title.

        Current behavior: Article numbers like "제26조" appear in title field.
        No separate article_number field exists yet.
        """
        node = {
            "id": "test-1",
            "type": "article",
            "title": "제26조 (직원의 구분)",
            "text": "직원은 일반직, 기술직, 별정직으로 구분한다.",
            "chunk_level": "article",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Current state: article number is part of title string
        assert chunk.title == "제26조 (직원의 구분)"
        assert "제26조" in chunk.title
        # Note: No article_number field exists yet

    def test_characterize_sub_article_number_in_title(self):
        """
        CHARACTERIZE: Sub-article format (제N조의M) in title.

        Current behavior: Sub-articles like "제10조의2" appear in title.
        """
        node = {
            "id": "test-2",
            "type": "article",
            "title": "제10조의2 (특별승급)",
            "text": "특별승급에 관한 사항은 다음과 같다.",
            "chunk_level": "article",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Current state: sub-article is part of title
        assert chunk.title == "제10조의2 (특별승급)"
        assert "제10조의2" in chunk.title

    def test_characterize_table_reference_in_title(self):
        """
        CHARACTERIZE: Table reference (별표N) in title.

        Current behavior: Table references like "별표1" appear in title.
        """
        node = {
            "id": "test-3",
            "type": "addendum",
            "title": "별표1 직원급별 봉급표",
            "text": "직원급별 봉급표는 다음과 같다.",
            "chunk_level": "addendum",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Current state: 별표 reference is part of title
        assert chunk.title == "별표1 직원급별 봉급표"
        assert "별표1" in chunk.title

    def test_characterize_form_reference_in_title(self):
        """
        CHARACTERIZE: Form reference (서식N) in title.

        Current behavior: Form references like "서식1" appear in title.
        """
        node = {
            "id": "test-4",
            "type": "addendum",
            "title": "서식1 휴직원부",
            "text": "휴직원부 서식은 다음과 같다.",
            "chunk_level": "addendum",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Current state: 서식 reference is part of title
        assert chunk.title == "서식1 휴직원부"
        assert "서식1" in chunk.title

    def test_characterize_chapter_title_no_article_number(self):
        """
        CHARACTERIZE: Chapter titles don't have article numbers.

        Current behavior: Chapter titles like "제1장" don't follow 제N조 pattern.
        """
        node = {
            "id": "test-5",
            "type": "chapter",
            "title": "제1장 총칙",
            "text": "本章의 目的은 다음과 같다.",
            "chunk_level": "chapter",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Current state: chapter designation is part of title
        assert chunk.title == "제1장 총칙"
        assert "제1장" in chunk.title

    def test_characterize_paragraph_item_no_article_number(self):
        """
        CHARACTERIZE: Paragraph and item titles lack article numbers.

        Current behavior: Lower level chunks don't have article numbers in title.
        """
        node = {
            "id": "test-6",
            "type": "paragraph",
            "title": "① 일반직",
            "text": "일반직은 정규직으로 한다.",
            "chunk_level": "paragraph",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Current state: paragraph has no article number
        assert chunk.title == "① 일반직"
        assert "조" not in chunk.title


class TestChunkMetadataCitationFields:
    """Characterize what citation-relevant fields exist in Chunk metadata."""

    def test_characterize_metadata_fields_available_for_citation(self):
        """
        CHARACTERIZE: Fields available for citation matching in metadata.

        Current behavior: title and parent_path are primary citation fields.
        """
        node = {
            "id": "test-7",
            "type": "article",
            "title": "제26조 (직원의 구분)",
            "text": "직원은 일반직, 기술직, 별정직으로 구분한다.",
            "chunk_level": "article",
            "parent_path": ["직원복무규정", "제3장 직종"],
        }

        chunk = Chunk.from_json_node(node, rule_code="3-1-5")
        metadata = chunk.to_metadata()

        # Current state: citation-relevant fields in metadata
        assert "title" in metadata
        assert metadata["title"] == "제26조 (직원의 구분)"
        assert "parent_path" in metadata
        assert metadata["parent_path"] == "직원복무규정 > 제3장 직종"
        assert "level" in metadata
        assert metadata["level"] == "article"

    def test_characterize_metadata_from_metadata_roundtrip(self):
        """
        CHARACTERIZE: Roundtrip preservation of citation fields.

        Current behavior: Citation fields survive metadata roundtrip.
        """
        node = {
            "id": "test-8",
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

        # Current state: citation fields preserved in roundtrip
        assert restored.title == original.title
        assert restored.parent_path == original.parent_path
        assert restored.level == original.level


class TestArticleNumberEdgeCases:
    """Characterize edge cases in article number formats."""

    def test_characterize_article_with_large_number(self):
        """
        CHARACTERIZE: Article numbers can be large (e.g., 제100조).

        Current behavior: No upper limit on article number.
        """
        node = {
            "id": "test-9",
            "type": "article",
            "title": "제100조 (시행일)",
            "text": "본 규정은 2024년 1월 1일부터 시행한다.",
            "chunk_level": "article",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        assert chunk.title == "제100조 (시행일)"
        assert "제100조" in chunk.title

    def test_characterize_article_zero_padding(self):
        """
        CHARACTERIZE: Article numbers may have zero-padding.

        Current behavior: Both "제01조" and "제1조" may appear.
        """
        node_padded = {
            "id": "test-10",
            "type": "article",
            "title": "제01조 (목적)",
            "text": "본 규정의 목적은 다음과 같다.",
            "chunk_level": "article",
        }

        chunk = Chunk.from_json_node(node_padded, rule_code="test-code")

        # Current state: zero-padding preserved in title
        assert chunk.title == "제01조 (목적)"
        assert "제01조" in chunk.title

    def test_characterize_mixed_korean_arabic_numbers(self):
        """
        CHARACTERIZE: Some titles may use Korean numbers (일, 이, 삼).

        Current behavior: Korean numerals may appear instead of Arabic.
        """
        node = {
            "id": "test-11",
            "type": "article",
            "title": "제일조 (목적)",
            "text": "본 규정의 목적은 다음과 같다.",
            "chunk_level": "article",
        }

        chunk = Chunk.from_json_node(node, rule_code="test-code")

        # Current state: Korean numerals preserved
        assert chunk.title == "제일조 (목적)"
        assert "제일조" in chunk.title
