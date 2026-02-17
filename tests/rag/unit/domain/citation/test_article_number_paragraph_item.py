"""
Unit tests for Article Number Extractor - Paragraph/Item patterns.

Tests verify paragraph (항) and item (호) level citation extraction
functionality.
"""

from src.rag.domain.citation.article_number_extractor import (
    ArticleNumber,
    ArticleNumberExtractor,
    ArticleType,
)


class TestParagraphExtraction:
    """Tests for paragraph (항) level citation extraction."""

    def test_extract_paragraph_only(self):
        """Extract standalone paragraph number 제1항."""
        extractor = ArticleNumberExtractor()
        title = "제1항"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.PARAGRAPH
        assert result.number == 1
        assert str(result) == "제1항"

    def test_extract_article_with_paragraph(self):
        """Extract article with paragraph 제26조제1항."""
        extractor = ArticleNumberExtractor()
        title = "제26조제1항 (직원의 구분)"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.ARTICLE
        assert result.number == 26
        assert result.paragraph_number == 1
        assert result.to_citation_format() == "제26조제1항"

    def test_extract_sub_article_with_paragraph(self):
        """Extract sub-article with paragraph 제10조의2제3항."""
        extractor = ArticleNumberExtractor()
        title = "제10조의2제3항"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.SUB_ARTICLE
        assert result.number == 10
        assert result.sub_number == 2
        assert result.paragraph_number == 3
        assert result.to_citation_format() == "제10조의2제3항"

    def test_extract_article_with_large_paragraph(self):
        """Extract article with large paragraph number 제26조제10항."""
        extractor = ArticleNumberExtractor()
        title = "제26조제10항"

        result = extractor.extract(title)

        assert result is not None
        assert result.number == 26
        assert result.paragraph_number == 10
        assert result.to_citation_format() == "제26조제10항"


class TestItemExtraction:
    """Tests for item (호) level citation extraction."""

    def test_extract_item_only(self):
        """Extract standalone item number 제1호."""
        extractor = ArticleNumberExtractor()
        title = "제1호"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.ITEM
        assert result.number == 1
        assert str(result) == "제1호"

    def test_extract_article_with_item(self):
        """Extract article with item 제26조제1호."""
        extractor = ArticleNumberExtractor()
        title = "제26조제1호"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.ARTICLE
        assert result.number == 26
        assert result.item_number == 1
        assert result.to_citation_format() == "제26조제1호"

    def test_extract_article_with_paragraph_and_item(self):
        """Extract article with paragraph and item 제26조제1항제2호."""
        extractor = ArticleNumberExtractor()
        title = "제26조제1항제2호"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.ARTICLE
        assert result.number == 26
        assert result.paragraph_number == 1
        assert result.item_number == 2
        assert result.to_citation_format() == "제26조제1항제2호"

    def test_extract_sub_article_with_paragraph_and_item(self):
        """Extract sub-article with paragraph and item 제10조의2제3항제1호."""
        extractor = ArticleNumberExtractor()
        title = "제10조의2제3항제1호"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.SUB_ARTICLE
        assert result.number == 10
        assert result.sub_number == 2
        assert result.paragraph_number == 3
        assert result.item_number == 1
        assert result.to_citation_format() == "제10조의2제3항제1호"


class TestExtractAllWithParagraphItem:
    """Tests for extracting multiple citations with paragraph/item."""

    def test_extract_multiple_with_paragraphs(self):
        """Extract multiple article+paragraph combinations."""
        extractor = ArticleNumberExtractor()
        text = "제26조제1항 및 제27조제2항에 따라"

        results = extractor.extract_all(text)

        assert len(results) >= 2
        # Check first result
        assert results[0].number == 26
        assert results[0].paragraph_number == 1

    def test_extract_mixed_types_with_paragraph_item(self):
        """Extract mixed types including paragraph/item patterns."""
        extractor = ArticleNumberExtractor()
        text = "제26조제1항제1호와 제10조의2 및 별표1을 참고한다"

        results = extractor.extract_all(text)

        assert len(results) >= 3
        types = {r.type for r in results}
        assert ArticleType.ARTICLE in types
        assert ArticleType.SUB_ARTICLE in types or ArticleType.TABLE in types


class TestCitationFormatting:
    """Tests for citation formatting with paragraph/item."""

    def test_format_article_paragraph(self):
        """Format citation with article and paragraph."""
        result = ArticleNumber(
            type=ArticleType.ARTICLE,
            number=26,
            paragraph_number=1,
            prefix="제",
            suffix="조",
        )

        assert str(result) == "제26조제1항"
        assert result.to_citation_format() == "제26조제1항"

    def test_format_article_paragraph_item(self):
        """Format citation with article, paragraph, and item."""
        result = ArticleNumber(
            type=ArticleType.ARTICLE,
            number=26,
            paragraph_number=1,
            item_number=2,
            prefix="제",
            suffix="조",
        )

        assert str(result) == "제26조제1항제2호"
        assert result.to_citation_format() == "제26조제1항제2호"

    def test_format_sub_article_paragraph(self):
        """Format citation with sub-article and paragraph."""
        result = ArticleNumber(
            type=ArticleType.SUB_ARTICLE,
            number=10,
            sub_number=2,
            paragraph_number=3,
            prefix="제",
            suffix="조",
        )

        assert str(result) == "제10조의2제3항"
        assert result.to_citation_format() == "제10조의2제3항"


class TestBackwardCompatibility:
    """Tests ensuring backward compatibility."""

    def test_basic_article_unchanged(self):
        """Basic article extraction behavior unchanged."""
        extractor = ArticleNumberExtractor()
        result = extractor.extract("제26조 (직원의 구분)")

        assert result is not None
        assert result.type == ArticleType.ARTICLE
        assert result.number == 26
        assert result.paragraph_number is None
        assert result.item_number is None
        assert result.to_citation_format() == "제26조"

    def test_sub_article_unchanged(self):
        """Sub-article extraction behavior unchanged."""
        extractor = ArticleNumberExtractor()
        result = extractor.extract("제10조의2 (특별승급)")

        assert result is not None
        assert result.type == ArticleType.SUB_ARTICLE
        assert result.number == 10
        assert result.sub_number == 2
        assert result.paragraph_number is None
        assert result.item_number is None
        assert result.to_citation_format() == "제10조의2"

    def test_table_unchanged(self):
        """Table extraction behavior unchanged."""
        extractor = ArticleNumberExtractor()
        result = extractor.extract("별표1 봉급표")

        assert result is not None
        assert result.type == ArticleType.TABLE
        assert result.number == 1
        assert result.to_citation_format() == "별표1"

    def test_form_unchanged(self):
        """Form extraction behavior unchanged."""
        extractor = ArticleNumberExtractor()
        result = extractor.extract("서식1 휴직원부")

        assert result is not None
        assert result.type == ArticleType.FORM
        assert result.number == 1
        assert result.to_citation_format() == "서식1"
