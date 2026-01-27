"""
Unit tests for ArticleNumberExtractor.

These tests verify the article number extraction functionality
for various Korean regulation citation formats.
"""

from src.rag.domain.citation.article_number_extractor import (
    ArticleNumberExtractor,
    ArticleType,
)


class TestArticleNumberExtraction:
    """Tests for article number extraction from titles."""

    def test_extract_basic_article_number(self):
        """Extract 제N조 format."""
        extractor = ArticleNumberExtractor()
        title = "제26조 (직원의 구분)"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.ARTICLE
        assert result.number == 26
        assert str(result) == "제26조"
        assert result.to_citation_format() == "제26조"

    def test_extract_sub_article_number(self):
        """Extract 제N조의M format."""
        extractor = ArticleNumberExtractor()
        title = "제10조의2 (특별승급)"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.SUB_ARTICLE
        assert result.number == 10
        assert result.sub_number == 2
        assert str(result) == "제10조의2"

    def test_extract_chapter_number(self):
        """Extract 제N장 format."""
        extractor = ArticleNumberExtractor()
        title = "제1장 총칙"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.CHAPTER
        assert result.number == 1
        assert str(result) == "제1장"

    def test_extract_table_number(self):
        """Extract 별표N format."""
        extractor = ArticleNumberExtractor()
        title = "별표1 직원급별 봉급표"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.TABLE
        assert result.number == 1
        assert str(result) == "별표1"

    def test_extract_form_number(self):
        """Extract 서식N format."""
        extractor = ArticleNumberExtractor()
        title = "서식1 휴직원부"

        result = extractor.extract(title)

        assert result is not None
        assert result.type == ArticleType.FORM
        assert result.number == 1
        assert str(result) == "서식1"

    def test_extract_large_article_number(self):
        """Extract large article numbers (e.g., 제100조)."""
        extractor = ArticleNumberExtractor()
        title = "제100조 (시행일)"

        result = extractor.extract(title)

        assert result is not None
        assert result.number == 100

    def test_extract_no_article_number(self):
        """Return None for titles without article numbers."""
        extractor = ArticleNumberExtractor()
        title = "① 일반직"  # Paragraph/item level

        result = extractor.extract(title)

        assert result is None

    def test_extract_empty_title(self):
        """Handle empty title gracefully."""
        extractor = ArticleNumberExtractor()

        result = extractor.extract("")

        assert result is None

    def test_extract_preserves_full_text(self):
        """Preserve full matched text in result."""
        extractor = ArticleNumberExtractor()
        title = "제26조 (직원의 구분)"

        result = extractor.extract(title)

        assert result is not None
        assert result.full_text == "제26조"
        assert result.prefix == "제"
        assert result.suffix == "조"


class TestExtractAll:
    """Tests for extracting multiple article numbers from text."""

    def test_extract_multiple_articles(self):
        """Extract all article numbers from regulation text."""
        extractor = ArticleNumberExtractor()
        text = """
        제1장 총칙
        제1조 (목적) 본 규정의 목적은 다음과 같다.
        제2조 (정의) 본 규정에서 사용하는 용어의 정의는 다음과 같다.
        """

        results = extractor.extract_all(text)

        assert len(results) == 3
        assert results[0].type == ArticleType.CHAPTER
        assert results[1].type == ArticleType.ARTICLE
        assert results[2].type == ArticleType.ARTICLE

    def test_extract_all_mixed_types(self):
        """Extract mixed article types from text."""
        extractor = ArticleNumberExtractor()
        text = """
        제26조에 따라 직원을 구분한다.
        별표1과 서식2를 참고한다.
        """

        results = extractor.extract_all(text)

        assert len(results) == 3
        types = {r.type for r in results}
        assert ArticleType.ARTICLE in types
        assert ArticleType.TABLE in types
        assert ArticleType.FORM in types

    def test_extract_all_empty_text(self):
        """Handle empty text gracefully."""
        extractor = ArticleNumberExtractor()

        results = extractor.extract_all("")

        assert results == []

    def test_extract_all_maintains_order(self):
        """Maintain order of appearance in text."""
        extractor = ArticleNumberExtractor()
        text = "제1조와 제2조, 그리고 제3조에 따라"

        results = extractor.extract_all(text)

        assert len(results) == 3
        assert [r.number for r in results] == [1, 2, 3]


class TestIsArticleLevel:
    """Tests for article-level detection."""

    def test_is_article_level_true_for_article(self):
        """Return True for 제N조 titles."""
        extractor = ArticleNumberExtractor()

        assert extractor.is_article_level("제26조 (직원의 구분)") is True
        assert extractor.is_article_level("제1조 목적") is True

    def test_is_article_level_true_for_sub_article(self):
        """Return True for 제N조의M titles."""
        extractor = ArticleNumberExtractor()

        assert extractor.is_article_level("제10조의2 (특별승급)") is True

    def test_is_article_level_false_for_chapter(self):
        """Return False for 제N장 titles."""
        extractor = ArticleNumberExtractor()

        assert extractor.is_article_level("제1장 총칙") is False

    def test_is_article_level_false_for_table(self):
        """Return False for 별표N titles."""
        extractor = ArticleNumberExtractor()

        assert extractor.is_article_level("별표1 봉급표") is False

    def test_is_article_level_false_for_paragraph(self):
        """Return False for paragraph/item level titles."""
        extractor = ArticleNumberExtractor()

        assert extractor.is_article_level("① 일반직") is False
        assert extractor.is_article_level("1. 본회의") is False


class TestArticleNumberFormatting:
    """Tests for article number formatting."""

    def test_to_citation_format_article(self):
        """Format article number correctly."""
        extractor = ArticleNumberExtractor()
        result = extractor.extract("제26조 (직원의 구분)")

        assert result.to_citation_format() == "제26조"

    def test_to_citation_format_sub_article(self):
        """Format sub-article correctly."""
        extractor = ArticleNumberExtractor()
        result = extractor.extract("제10조의2 (특별승급)")

        assert result.to_citation_format() == "제10조의2"

    def test_to_citation_format_table(self):
        """Format table reference correctly."""
        extractor = ArticleNumberExtractor()
        result = extractor.extract("별표1 봉급표")

        assert result.to_citation_format() == "별표1"

    def test_to_citation_format_form(self):
        """Format form reference correctly."""
        extractor = ArticleNumberExtractor()
        result = extractor.extract("서식1 휴직원부")

        assert result.to_citation_format() == "서식1"

    def test_str_representation(self):
        """String representation matches expected format."""
        extractor = ArticleNumberExtractor()

        article = extractor.extract("제26조 (직원의 구분)")
        assert str(article) == "제26조"

        sub_article = extractor.extract("제10조의2 (특별승급)")
        assert str(sub_article) == "제10조의2"

        table = extractor.extract("별표1 봉급표")
        assert str(table) == "별표1"
