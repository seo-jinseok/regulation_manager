"""
Extended tests for formatter.py to improve coverage from 69% to 85%.

Focuses on additional methods:
- _build_index_nodes
- _populate_index_docs
- _extract_references
- _infer_first_chapter
- _resolve_sort_no
- _extract_header_metadata
- _extract_clean_title
- _parse_addenda_text
- _extract_html_segments
"""

import unittest

from src.formatter import RegulationFormatter


class TestExtractReferences(unittest.TestCase):
    """Tests for _extract_references method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_empty_text(self):
        """Test empty text returns empty list."""
        result = self.formatter._extract_references("")
        self.assertEqual(result, [])

    def test_none_text(self):
        """Test None text returns empty list."""
        result = self.formatter._extract_references(None)
        self.assertEqual(result, [])

    def test_single_article_reference(self):
        """Test single article reference."""
        text = "제5조에 따르면"
        result = self.formatter._extract_references(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "제5조")

    def test_article_with_item(self):
        """Test article with item reference."""
        text = "제10조제2항"
        result = self.formatter._extract_references(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "제10조제2항")

    def test_multiple_references(self):
        """Test multiple references in text."""
        text = "제5조와 제10조제2항에 따라"
        result = self.formatter._extract_references(text)
        self.assertEqual(len(result), 2)

    def test_paragraph_only_reference(self):
        """Test paragraph-only reference (제N항)."""
        text = "제2항에 따라"
        result = self.formatter._extract_references(text)
        self.assertEqual(len(result), 1)
        self.assertIn("제2항", result[0]["text"])

    def test_item_only_reference(self):
        """Test item-only reference (제N호)."""
        text = "제1호에 해당"
        result = self.formatter._extract_references(text)
        self.assertEqual(len(result), 1)
        self.assertIn("제1호", result[0]["text"])


class TestInferFirstChapter(unittest.TestCase):
    """Tests for _infer_first_chapter method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_empty_articles(self):
        """Test empty articles returns as-is."""
        result = self.formatter._infer_first_chapter([])
        self.assertEqual(result, [])

    def test_first_article_has_chapter(self):
        """Test when first article already has chapter."""
        articles = [
            {"chapter": "제1장 총칙", "content": ["text1"]},
            {"chapter": "제1장 총칙", "content": ["text2"]},
        ]
        result = self.formatter._infer_first_chapter(articles)
        # No change since first article has chapter
        self.assertEqual(result[0]["chapter"], "제1장 총칙")

    def test_infers_chapter_1_for_articles_before_chapter_2(self):
        """Test inferring '제1장 총칙' for articles before 제2장."""
        articles = [
            {"article_no": "제1조", "content": ["text1"], "chapter": None},
            {"article_no": "제2조", "content": ["text2"], "chapter": None},
            {"article_no": "제3조", "content": ["text3"], "chapter": "제2장"},
        ]
        result = self.formatter._infer_first_chapter(articles)
        # First two articles should get "제1장 총칙"
        self.assertEqual(result[0]["chapter"], "제1장 총칙")
        self.assertEqual(result[1]["chapter"], "제1장 총칙")
        self.assertEqual(result[2]["chapter"], "제2장")

    def test_no_inference_for_chapter_1(self):
        """Test no inference if first explicit chapter is 제1장."""
        articles = [
            {"article_no": "제1조", "content": ["text1"], "chapter": None},
            {"article_no": "제2조", "content": ["text2"], "chapter": "제1장"},
        ]
        result = self.formatter._infer_first_chapter(articles)
        # No inference needed since 제1장 is already first chapter
        self.assertIsNone(result[0]["chapter"])

    def test_no_explicit_chapter(self):
        """Test articles with no explicit chapter."""
        articles = [
            {"article_no": "제1조", "content": ["text1"], "chapter": None},
            {"article_no": "제2조", "content": ["text2"], "chapter": None},
        ]
        result = self.formatter._infer_first_chapter(articles)
        # No change since no explicit chapter found
        self.assertIsNone(result[0]["chapter"])


class TestResolveSortNo(unittest.TestCase):
    """Tests for _resolve_sort_no method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_article_type_simple(self):
        """Test simple article number."""
        result = self.formatter._resolve_sort_no("제29조", "article")
        self.assertEqual(result["main"], 29)
        self.assertEqual(result["sub"], 0)

    def test_article_with_sub(self):
        """Test article with sub number (제29조의2)."""
        result = self.formatter._resolve_sort_no("제29조의2", "article")
        self.assertEqual(result["main"], 29)
        self.assertEqual(result["sub"], 2)

    def test_chapter_type(self):
        """Test chapter number."""
        result = self.formatter._resolve_sort_no("제1장", "chapter")
        self.assertEqual(result["main"], 1)
        self.assertEqual(result["sub"], 0)

    def test_section_type(self):
        """Test section number."""
        result = self.formatter._resolve_sort_no("제1절", "section")
        self.assertEqual(result["main"], 1)
        self.assertEqual(result["sub"], 0)

    def test_paragraph_circled_number(self):
        """Test paragraph with circled number (①)."""
        result = self.formatter._resolve_sort_no("①", "paragraph")
        self.assertEqual(result["main"], 1)
        self.assertEqual(result["sub"], 0)

    def test_paragraph_high_circled_number(self):
        """Test paragraph with high circled number (⑮)."""
        result = self.formatter._resolve_sort_no("⑮", "paragraph")
        self.assertEqual(result["main"], 15)

    def test_item_number(self):
        """Test item number (1.)."""
        result = self.formatter._resolve_sort_no("1.", "item")
        self.assertEqual(result["main"], 1)
        self.assertEqual(result["sub"], 0)

    def test_subitem_hangul(self):
        """Test subitem with Hangul (가.)."""
        result = self.formatter._resolve_sort_no("가.", "subitem")
        self.assertEqual(result["main"], 1)
        self.assertEqual(result["sub"], 0)

    def test_subitem_last_hangul(self):
        """Test last subitem with Hangul (하.)."""
        result = self.formatter._resolve_sort_no("하.", "subitem")
        self.assertEqual(result["main"], 14)

    def test_unknown_node_type(self):
        """Test unknown node type returns defaults."""
        result = self.formatter._resolve_sort_no("unknown", "unknown")
        self.assertEqual(result["main"], 0)
        self.assertEqual(result["sub"], 0)


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

    def test_extracts_rule_code_and_page(self):
        """Test extraction of rule code and page number."""
        html = """
        <div class="HeaderArea">
            3-1-5~10
        </div>
        """
        result = self.formatter._extract_header_metadata(html)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["rule_code"], "3-1-5")
        self.assertEqual(result[0]["page"], "10")

    def test_extracts_prefix(self):
        """Test extraction of title prefix."""
        html = """
        <div class="HeaderArea">
            교원인사규정 3-1-5~1
        </div>
        """
        result = self.formatter._extract_header_metadata(html)
        self.assertEqual(len(result), 1)
        self.assertIn("교원인사규정", result[0]["prefix"])

    def test_normalizes_dashes(self):
        """Test dash normalization in rule codes."""
        html = """
        <div class="HeaderArea">
            규정 3—1—5
        </div>
        """
        result = self.formatter._extract_header_metadata(html)
        self.assertEqual(result[0]["rule_code"], "3-1-5")

    def test_multiple_headers(self):
        """Test multiple header areas."""
        html = """
        <div class="HeaderArea">3-1-1~1</div>
        <div class="HeaderArea">3-1-2~5</div>
        """
        result = self.formatter._extract_header_metadata(html)
        self.assertEqual(len(result), 2)


class TestExtractCleanTitle(unittest.TestCase):
    """Tests for _extract_clean_title method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_explicit_title(self):
        """Test explicit title from doc_data."""
        doc_data = {"title": "교원인사규정", "preamble": []}
        title, preamble = self.formatter._extract_clean_title(doc_data)
        self.assertEqual(title, "교원인사규정")
        self.assertEqual(preamble, "")

    def test_title_from_preamble(self):
        """Test title extraction from preamble."""
        doc_data = {
            "title": None,
            "preamble": ["제1장 총칙", "교원인사규정", "제2조 목적"],
        }
        title, preamble = self.formatter._extract_clean_title(doc_data)
        # Should find "교원인사규정" as it ends with "규정"
        self.assertEqual(title, "교원인사규정")

    def test_preamble_with_metadata_brackets(self):
        """Test preamble with metadata brackets."""
        doc_data = {
            "title": None,
            "preamble": ["<시행 2020.01.01>", "교원인사규정"],
        }
        title, preamble = self.formatter._extract_clean_title(doc_data)
        # Should skip the metadata line
        self.assertEqual(title, "교원인사규정")

    def test_fallback_to_last_line(self):
        """Test fallback to last line when no title found."""
        doc_data = {"title": None, "preamble": ["Some text", "Last line"]}
        title, preamble = self.formatter._extract_clean_title(doc_data)
        # Falls back to last line
        self.assertEqual(title, "Last line")

    def test_preamble_as_string(self):
        """Test preamble as string instead of list."""
        doc_data = {"title": None, "preamble": "교원인사규정\n제1조"}
        title, preamble = self.formatter._extract_clean_title(doc_data)
        self.assertEqual(title, "교원인사규정")


class TestParseAddendaText(unittest.TestCase):
    """Tests for _parse_addenda_text method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_empty_text(self):
        """Test empty text returns empty nodes."""
        result = self.formatter._parse_addenda_text("")
        self.assertEqual(result, [])

    def test_article_style(self):
        """Test article-style parsing (제1조)."""
        text = "제1조(시행일) 이 규정은 2020년부터 시행한다."
        result = self.formatter._parse_addenda_text(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "addendum_item")
        self.assertEqual(result[0]["display_no"], "제1조")

    def test_numbered_item_style(self):
        """Test numbered item style (1.)."""
        text = "1.(시행일) 2020년 1월 1일부터 시행한다."
        result = self.formatter._parse_addenda_text(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "addendum_item")
        self.assertEqual(result[0]["display_no"], "1.")

    def test_paragraph_style(self):
        """Test paragraph style (①)."""
        text = "① 첫 번째 항목\n② 두 번째 항목"
        result = self.formatter._parse_addenda_text(text)
        # Both become separate top-level items
        self.assertEqual(len(result), 2)

    def test_nested_paragraph_under_article(self):
        """Test nested paragraphs under article."""
        text = "제1조 시행일\n① 2020년부터\n② 2021년부터"
        result = self.formatter._parse_addenda_text(text)
        self.assertEqual(len(result), 1)
        # Should have children
        self.assertEqual(len(result[0]["children"]), 2)

    def test_text_content_appends(self):
        """Test text content appending to current node."""
        text = "제1조\n  추가 텍스트"
        result = self.formatter._parse_addenda_text(text)
        self.assertEqual(len(result), 1)
        self.assertIn("추가 텍스트", result[0]["text"])


class TestBuildIndexNodes(unittest.TestCase):
    """Tests for _build_index_nodes method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_empty_entries(self):
        """Test empty entries returns empty list."""
        result = self.formatter._build_index_nodes([])
        self.assertEqual(result, [])

    def test_builds_text_nodes(self):
        """Test building text nodes from entries."""
        entries = [
            {"title": "교원인사규정", "rule_code": "3-1-5"},
            {"title": "학칙", "rule_code": "1-1-1"},
        ]
        result = self.formatter._build_index_nodes(entries)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["type"], "text")
        self.assertEqual(result[0]["title"], "교원인사규정")
        self.assertEqual(result[0]["metadata"]["rule_code"], "3-1-5")

    def test_sort_no_increments(self):
        """Test sort_no increments correctly."""
        entries = [
            {"title": "A", "rule_code": "1-1-1"},
            {"title": "B", "rule_code": "1-1-2"},
            {"title": "C", "rule_code": "1-1-3"},
        ]
        result = self.formatter._build_index_nodes(entries)
        self.assertEqual(result[0]["sort_no"]["main"], 1)
        self.assertEqual(result[1]["sort_no"]["main"], 2)
        self.assertEqual(result[2]["sort_no"]["main"], 3)

    def test_none_rule_code(self):
        """Test entries without rule code."""
        entries = [{"title": "No Code"}]
        result = self.formatter._build_index_nodes(entries)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["metadata"], {})


class TestPopulateIndexDocs(unittest.TestCase):
    """Tests for _populate_index_docs method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_toc_population(self):
        """Test TOC document population."""
        docs = [
            {"title": "차례", "content": [], "addenda": [], "attached_files": []},
        ]
        extracted_metadata = {
            "toc": [
                {"title": "교원인사규정", "rule_code": "3-1-5"},
            ]
        }

        self.formatter._populate_index_docs(docs, extracted_metadata)

        self.assertEqual(docs[0]["doc_type"], "toc")
        self.assertEqual(len(docs[0]["content"]), 1)

    def test_index_alpha_population(self):
        """Test index_alpha document population."""
        docs = [
            {"title": "찾아보기", "preamble": "가나다순", "content": []},
        ]
        extracted_metadata = {
            "index_by_alpha": [
                {"title": "가나다순 목록"},
            ]
        }

        self.formatter._populate_index_docs(docs, extracted_metadata)

        self.assertEqual(docs[0]["doc_type"], "index_alpha")
        self.assertGreater(len(docs[0]["content"]), 0)

    def test_skips_content_docs(self):
        """Test that documents with content are not modified."""
        docs = [
            {"title": "규정", "content": [{"id": 1}]},
        ]
        extracted_metadata = {"toc": [{"title": "Item"}]}

        original_content = docs[0]["content"]
        self.formatter._populate_index_docs(docs, extracted_metadata)

        # Content should remain unchanged
        self.assertEqual(docs[0]["content"], original_content)


class TestExtractHtmlSegments(unittest.TestCase):
    """Tests for _extract_html_segments method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_no_attached_files(self):
        """Test with no attached files."""
        attached_files = []
        html_content = "<html><body>Content</body></html>"
        # Should not raise error
        self.formatter._extract_html_segments(attached_files, html_content)

    def test_simple_html(self):
        """Test simple HTML extraction."""
        attached_files = [{"title": "[별표 1]", "text": "Table content"}]
        html_content = "[별표 1] <span>Table content</span>"

        # Function should find and extract
        # We can't test full extraction without actual HTML parsing
        # But we can verify it doesn't crash
        self.formatter._extract_html_segments(attached_files, html_content)


class TestHierarchyMethods(unittest.TestCase):
    """Tests for hierarchy building methods."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_create_node_defaults(self):
        """Test _create_node with defaults."""
        result = self.formatter._create_node(
            "article",
            "제1조",
            None,
            "Test text",
        )
        self.assertEqual(result["type"], "article")
        self.assertEqual(result["display_no"], "제1조")
        self.assertEqual(result["title"], "")
        self.assertEqual(result["text"], "Test text")
        self.assertEqual(result["sort_no"], {"main": 0, "sub": 0})
        self.assertEqual(result["confidence_score"], 1.0)
        self.assertEqual(result["references"], [])
        self.assertEqual(result["metadata"], {})
        self.assertEqual(result["children"], [])

    def test_create_node_full(self):
        """Test _create_node with all parameters."""
        result = self.formatter._create_node(
            "chapter",
            "제1장",
            "총칙",
            "Chapter text",
            sort_no={"main": 1, "sub": 0},
            children=[{"id": "child"}],
            confidence_score=0.8,
            references=[{"text": "ref"}],
            metadata={"key": "value"},
        )
        self.assertEqual(result["type"], "chapter")
        self.assertEqual(result["title"], "총칙")
        self.assertEqual(result["confidence_score"], 0.8)
        self.assertEqual(len(result["references"]), 1)
        self.assertEqual(result["metadata"]["key"], "value")
        self.assertEqual(len(result["children"]), 1)

    def test_get_parent_list_chapter_level(self):
        """Test _get_parent_list for chapter level."""
        roots = []
        current_nodes = {
            "chapter": {"name": None, "node": None},
            "section": {"name": None, "node": None},
            "subsection": {"name": None, "node": None},
        }

        result = self.formatter._get_parent_list("chapter", current_nodes, roots)
        self.assertEqual(result, roots)

    def test_get_parent_list_section_level(self):
        """Test _get_parent_list for section level."""
        roots = []
        chapter_node = {"children": []}
        current_nodes = {
            "chapter": {"name": "제1장", "node": chapter_node},
            "section": {"name": None, "node": None},
            "subsection": {"name": None, "node": None},
        }

        result = self.formatter._get_parent_list("section", current_nodes, roots)
        self.assertEqual(result, chapter_node["children"])

    def test_get_parent_list_article_level(self):
        """Test _get_parent_list for article level."""
        roots = []
        section_node = {"children": []}
        current_nodes = {
            "chapter": {"name": None, "node": None},
            "section": {"name": "제1절", "node": section_node},
            "subsection": {"name": None, "node": None},
        }

        result = self.formatter._get_parent_list("article", current_nodes, roots)
        self.assertEqual(result, section_node["children"])


if __name__ == "__main__":
    unittest.main()
