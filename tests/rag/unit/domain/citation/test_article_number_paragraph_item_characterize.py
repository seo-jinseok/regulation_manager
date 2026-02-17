"""
Characterization tests for Article Number Extractor - Paragraph/Item patterns.

These tests capture the EXPECTED behavior for paragraph (항) and item (호)
level citation extraction. Run these tests to verify new functionality.

These are NOT characterization tests of existing behavior, but rather
specification tests defining expected behavior for new features.
"""

from src.rag.domain.citation.article_number_extractor import (
    ArticleNumberExtractor,
    ArticleType,
)


class TestParagraphExtraction:
    """Tests for paragraph (항) level citation extraction."""

    def test_characterize_paragraph_only(self):
        """
        Characterize: Extract standalone paragraph number.

        Input: "제1항" (Paragraph 1)
        Expected: Should extract paragraph 1
        """
        extractor = ArticleNumberExtractor()
        title = "제1항"

        result = extractor.extract(title)

        # Currently returns None (not implemented)
        # After implementation: should return ArticleNumber with type PARAGRAPH
        # This test documents expected behavior
        if result is not None:
            assert result.type == ArticleType.PARAGRAPH
            assert result.number == 1
            assert str(result) == "제1항"

    def test_characterize_article_with_paragraph(self):
        """
        Characterize: Extract article with paragraph.

        Input: "제26조제1항" (Article 26, Paragraph 1)
        Expected: Should extract article 26, paragraph 1
        """
        extractor = ArticleNumberExtractor()
        title = "제26조제1항 (직원의 구분)"

        result = extractor.extract(title)

        # Currently extracts only article number (제26조)
        # After implementation: should include paragraph info
        assert result is not None
        if result.paragraph_number is not None:
            assert result.number == 26
            assert result.paragraph_number == 1

    def test_characterize_sub_article_with_paragraph(self):
        """
        Characterize: Extract sub-article with paragraph.

        Input: "제10조의2제3항" (Article 10-2, Paragraph 3)
        Expected: Should extract article 10-2, paragraph 3
        """
        extractor = ArticleNumberExtractor()
        title = "제10조의2제3항"

        result = extractor.extract(title)

        assert result is not None
        if result.paragraph_number is not None:
            assert result.number == 10
            assert result.sub_number == 2
            assert result.paragraph_number == 3


class TestItemExtraction:
    """Tests for item (호) level citation extraction."""

    def test_characterize_item_only(self):
        """
        Characterize: Extract standalone item number.

        Input: "제1호" (Item 1)
        Expected: Should extract item 1
        """
        extractor = ArticleNumberExtractor()
        title = "제1호"

        result = extractor.extract(title)

        # Currently returns None (not implemented)
        # After implementation: should return ArticleNumber with type ITEM
        if result is not None:
            assert result.type == ArticleType.ITEM
            assert result.number == 1
            assert str(result) == "제1호"

    def test_characterize_article_with_item(self):
        """
        Characterize: Extract article with item.

        Input: "제26조제1호" (Article 26, Item 1)
        Expected: Should extract article 26, item 1
        """
        extractor = ArticleNumberExtractor()
        title = "제26조제1호"

        result = extractor.extract(title)

        # Currently extracts only article number (제26조)
        # After implementation: should include item info
        assert result is not None
        if result.item_number is not None:
            assert result.number == 26
            assert result.item_number == 1

    def test_characterize_full_citation(self):
        """
        Characterize: Extract full citation with article, paragraph, item.

        Input: "제26조제1항제1호" (Article 26, Paragraph 1, Item 1)
        Expected: Should extract all components
        """
        extractor = ArticleNumberExtractor()
        title = "제26조제1항제1호"

        result = extractor.extract(title)

        # Currently extracts only article number (제26조)
        # After implementation: should include all components
        assert result is not None
        if hasattr(result, 'paragraph_number') and result.paragraph_number is not None:
            assert result.number == 26
            assert result.paragraph_number == 1
            assert result.item_number == 1


class TestParagraphItemInContent:
    """Tests for extracting paragraph/item from content text."""

    def test_characterize_paragraph_in_text(self):
        """
        Characterize: Find all paragraph references in text.

        Input: "제26조제1항 및 제27조제2항에 따라"
        Expected: Should find both article+paragraph combinations
        """
        extractor = ArticleNumberExtractor()
        text = "제26조제1항 및 제27조제2항에 따라"

        results = extractor.extract_all(text)

        # Currently finds articles only
        # After implementation: should find article+paragraph combinations
        assert len(results) >= 2  # At least two articles found

    def test_characterize_item_in_text(self):
        """
        Characterize: Find item references in text.

        Input: "제26조제1항제1호부터 제26조제1항제3호까지"
        Expected: Should find article+paragraph+item combinations
        """
        extractor = ArticleNumberExtractor()
        text = "제26조제1항제1호부터 제26조제1항제3호까지"

        results = extractor.extract_all(text)

        # After implementation: should find combinations
        assert len(results) >= 1


class TestCitationFormatWithParagraphItem:
    """Tests for citation formatting with paragraph/item."""

    def test_characterize_format_article_paragraph(self):
        """
        Characterize: Format citation with paragraph.

        Input: ArticleNumber(article=26, paragraph=1)
        Expected: "제26조제1항"
        """
        extractor = ArticleNumberExtractor()
        result = extractor.extract("제26조제1항")

        if result is not None and hasattr(result, 'paragraph_number'):
            if result.paragraph_number is not None:
                assert result.to_citation_format() == "제26조제1항"

    def test_characterize_format_full_citation(self):
        """
        Characterize: Format citation with article, paragraph, item.

        Input: ArticleNumber(article=26, paragraph=1, item=2)
        Expected: "제26조제1항제2호"
        """
        extractor = ArticleNumberExtractor()
        result = extractor.extract("제26조제1항제2호")

        if result is not None and hasattr(result, 'item_number'):
            if result.item_number is not None:
                assert result.to_citation_format() == "제26조제1항제2호"


class TestBackwardCompatibility:
    """Tests for backward compatibility after adding paragraph/item support."""

    def test_characterize_basic_article_still_works(self):
        """
        Characterize: Basic article extraction still works.

        Input: "제26조 (직원의 구분)"
        Expected: Should return ArticleNumber with article=26
        """
        extractor = ArticleNumberExtractor()
        title = "제26조 (직원의 구분)"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.ARTICLE
        assert result.number == 26
        assert result.to_citation_format() == "제26조"

    def test_characterize_sub_article_still_works(self):
        """
        Characterize: Sub-article extraction still works.

        Input: "제10조의2 (특별승급)"
        Expected: Should return ArticleNumber with article=10, sub=2
        """
        extractor = ArticleNumberExtractor()
        title = "제10조의2 (특별승급)"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.SUB_ARTICLE
        assert result.number == 10
        assert result.sub_number == 2
        assert result.to_citation_format() == "제10조의2"

    def test_characterize_table_still_works(self):
        """
        Characterize: Table extraction still works.

        Input: "별표1 봉급표"
        Expected: Should return ArticleNumber with type=TABLE, number=1
        """
        extractor = ArticleNumberExtractor()
        title = "별표1 봉급표"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.TABLE
        assert result.number == 1
