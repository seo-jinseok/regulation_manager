"""
Extended tests for formatter.py to improve coverage.

Focuses on additional testable methods:
- _extract_references
- _resolve_sort_no
- _infer_first_chapter
- _extract_header_metadata
- _parse_addenda_text
"""

import unittest

from src.formatter import RegulationFormatter


class TestExtractReferences(unittest.TestCase):
    """Tests for _extract_references method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_extract_article_references(self):
        """Test extraction of article references."""
        text = "제5조와 제10조제1항을 따른다."
        refs = self.formatter._extract_references(text)
        self.assertEqual(len(refs), 2)

    def test_extract_article_with_of_clause(self):
        """Test extraction of article with '의' clause."""
        text = "제29조의2에 따른다"
        refs = self.formatter._extract_references(text)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["text"], "제29조의2")

    def test_extract_paragraph_only(self):
        """Test extraction of paragraph references."""
        text = "제3항에 따라 처리한다"
        refs = self.formatter._extract_references(text)
        self.assertEqual(len(refs), 1)

    def test_extract_mixed_references(self):
        """Test extraction of mixed article and paragraph references."""
        text = "제5조제3항과 제10조제1항제2호를 따른다"
        refs = self.formatter._extract_references(text)
        self.assertGreater(len(refs), 0)

    def test_empty_text(self):
        """Test empty text returns empty list."""
        refs = self.formatter._extract_references("")
        self.assertEqual(len(refs), 0)

    def test_none_text(self):
        """Test None text returns empty list."""
        refs = self.formatter._extract_references(None)
        self.assertEqual(len(refs), 0)

    def test_text_without_references(self):
        """Test text without references returns empty list."""
        text = "이것은 일반 텍스트입니다."
        refs = self.formatter._extract_references(text)
        self.assertEqual(len(refs), 0)


class TestResolveSortNo(unittest.TestCase):
    """Tests for _resolve_sort_no method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_article_basic(self):
        """Test basic article number."""
        result = self.formatter._resolve_sort_no("제29조", "article")
        self.assertEqual(result["main"], 29)
        self.assertEqual(result["sub"], 0)

    def test_article_with_sub(self):
        """Test article number with sub-article."""
        result = self.formatter._resolve_sort_no("제29조의2", "article")
        self.assertEqual(result["main"], 29)
        self.assertEqual(result["sub"], 2)

    def test_article_spaces(self):
        """Test article number with spaces."""
        result = self.formatter._resolve_sort_no("제 29 조 의 2", "article")
        self.assertEqual(result["main"], 29)

    def test_chapter(self):
        """Test chapter number."""
        result = self.formatter._resolve_sort_no("제1장", "chapter")
        self.assertEqual(result["main"], 1)

    def test_section(self):
        """Test section number."""
        result = self.formatter._resolve_sort_no("제2절", "section")
        self.assertEqual(result["main"], 2)

    def test_paragraph_circled(self):
        """Test paragraph with circled number."""
        result = self.formatter._resolve_sort_no("①", "paragraph")
        self.assertEqual(result["main"], 1)

    def test_paragraph_circled_range(self):
        """Test paragraph with circled numbers in range."""
        # ① is U+2460, should map to 1
        result = self.formatter._resolve_sort_no("①", "paragraph")
        self.assertEqual(result["main"], 1)

        # ⑩ is U+2469, should map to 10
        result = self.formatter._resolve_sort_no("⑩", "paragraph")
        self.assertEqual(result["main"], 10)

    def test_item_number(self):
        """Test item number."""
        result = self.formatter._resolve_sort_no("1.", "item")
        self.assertEqual(result["main"], 1)

    def test_subitem_hangul(self):
        """Test subitem with hangul character."""
        result = self.formatter._resolve_sort_no("가.", "subitem")
        self.assertEqual(result["main"], 1)  # 가 is first

    def test_subitem_hangul_last(self):
        """Test last hangul subitem."""
        result = self.formatter._resolve_sort_no("하.", "subitem")
        self.assertEqual(result["main"], 14)  # 하 is last in the list

    def test_unknown_node_type(self):
        """Test unknown node type returns zeros."""
        result = self.formatter._resolve_sort_no("something", "unknown")
        self.assertEqual(result["main"], 0)
        self.assertEqual(result["sub"], 0)


class TestInferFirstChapter(unittest.TestCase):
    """Tests for _infer_first_chapter method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_empty_articles(self):
        """Test empty articles list."""
        result = self.formatter._infer_first_chapter([])
        self.assertEqual(result, [])

    def test_all_articles_have_chapter(self):
        """Test when all articles already have chapter."""
        articles = [
            {"chapter": "제1장 총칙", "content": ["content1"]},
            {"chapter": "제2장 목적", "content": ["content2"]},
        ]
        result = self.formatter._infer_first_chapter(articles)
        # No inference needed, all have chapters
        self.assertEqual(result[0]["chapter"], "제1장 총칙")

    def test_infer_chapter1(self):
        """Test inference of chapter 1."""
        articles = [
            {"content": ["intro"]},  # No chapter
            {"chapter": "제2장", "content": ["content"]},
        ]
        result = self.formatter._infer_first_chapter(articles)
        self.assertEqual(result[0]["chapter"], "제1장 총칙")

    def test_first_article_has_chapter1(self):
        """Test when first article already has chapter 1."""
        articles = [
            {"chapter": "제1장 총칙", "content": ["content1"]},
            {"chapter": "제2장", "content": ["content2"]},
        ]
        result = self.formatter._infer_first_chapter(articles)
        # No inference needed
        self.assertEqual(result[0]["chapter"], "제1장 총칙")

    def test_no_chapter_at_all(self):
        """Test when no chapter info exists anywhere."""
        articles = [
            {"content": ["content1"]},
            {"content": ["content2"]},
        ]
        result = self.formatter._infer_first_chapter(articles)
        # No inference possible without any chapter reference
        self.assertIsNone(result[0].get("chapter"))

    def test_first_chapter_is_chapter2(self):
        """Test when first explicit chapter is chapter 2."""
        articles = [
            {"content": ["intro"]},
            {"chapter": "제2장 목적", "content": ["content"]},
        ]
        result = self.formatter._infer_first_chapter(articles)
        # Should infer 제1장 총칙 for articles before 제2장
        self.assertEqual(result[0]["chapter"], "제1장 총칙")

    def test_first_chapter_is_chapter3(self):
        """Test when first explicit chapter is chapter 3."""
        articles = [
            {"content": ["intro1"]},
            {"content": ["intro2"]},
            {"chapter": "제3장", "content": ["content"]},
        ]
        result = self.formatter._infer_first_chapter(articles)
        # Should infer 제1장 총칙 for articles before 제3장
        self.assertEqual(result[0]["chapter"], "제1장 총칙")
        self.assertEqual(result[1]["chapter"], "제1장 총칙")


class TestExtractHeaderMetadata(unittest.TestCase):
    """Tests for _extract_header_metadata method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_empty_html(self):
        """Test empty HTML returns empty list."""
        result = self.formatter._extract_header_metadata("")
        self.assertEqual(result, [])

    def test_none_html(self):
        """Test None HTML returns empty list."""
        result = self.formatter._extract_header_metadata(None)
        self.assertEqual(result, [])

    def test_valid_header_extraction(self):
        """Test valid header with rule code and page."""
        html = """
        <html><body>
        <div class="HeaderArea">교원인사규정 3-1-5~1</div>
        </body></html>
        """
        result = self.formatter._extract_header_metadata(html)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["rule_code"], "3-1-5")
        self.assertEqual(result[0]["page"], "1")
        self.assertEqual(result[0]["prefix"], "교원인사규정")

    def test_multiple_headers(self):
        """Test multiple headers in HTML."""
        html = """
        <html><body>
        <div class="HeaderArea">교원인사규정 3-1-5~1</div>
        <div class="HeaderArea">학칙 1-1-1~2</div>
        </body></html>
        """
        result = self.formatter._extract_header_metadata(html)
        self.assertEqual(len(result), 2)

    def test_header_without_page(self):
        """Test header without page number."""
        html = """
        <html><body>
        <div class="HeaderArea">교원인사규정 3-1-5</div>
        </body></html>
        """
        result = self.formatter._extract_header_metadata(html)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]["page"])

    def test_various_dash_types_normalized(self):
        """Test various dash types are normalized."""
        html = """
        <html><body>
        <div class="HeaderArea">규정 3—1—5</div>
        </body></html>
        """
        result = self.formatter._extract_header_metadata(html)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["rule_code"], "3-1-5")


class TestParseAddendaText(unittest.TestCase):
    """Tests for _parse_addenda_text method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_empty_text(self):
        """Test empty text returns empty list."""
        result = self.formatter._parse_addenda_text("")
        self.assertEqual(len(result), 0)

    def test_article_style(self):
        """Test article-style parsing."""
        text = """제1조(시행일) 이 규정은 2024년 1월 1일부터 시행한다.
제2조(경과조치) ..."""
        result = self.formatter._parse_addenda_text(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["type"], "addendum_item")
        self.assertEqual(result[0]["display_no"], "제1조")

    def test_numbered_item_style(self):
        """Test numbered item style."""
        text = """1. (시행일) 이 규정은 2024년 1월 1일부터 시행한다.
2. (적용범위) ..."""
        result = self.formatter._parse_addenda_text(text)
        self.assertEqual(len(result), 2)

    def test_paragraph_style(self):
        """Test paragraph style with circled numbers."""
        text = """① 첫 번째 항목
② 두 번째 항목"""
        result = self.formatter._parse_addenda_text(text)
        self.assertEqual(len(result), 2)

    def test_mixed_styles(self):
        """Test mixed parsing styles."""
        text = """제1조(시행일) 이 규정은 2024년 1월 1일부터 시행한다.
① 첫 번째 항목
② 두 번째 항목"""
        result = self.formatter._parse_addenda_text(text)
        self.assertGreater(len(result), 0)

    def test_continuation_lines(self):
        """Test continuation lines are appended."""
        text = """제1조(시행일) 이 규정은
2024년 1월 1일부터 시행한다."""
        result = self.formatter._parse_addenda_text(text)
        self.assertGreater(len(result), 0)
        # Text should be combined
        self.assertIn("2024년", result[0]["text"])

    def test_pipe_continuation(self):
        """Test pipe-prefixed continuation lines."""
        text = """제1조(시행일) 이 규정은
| 2024년 1월 1일부터 시행한다."""
        result = self.formatter._parse_addenda_text(text)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
