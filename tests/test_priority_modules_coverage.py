# -*- coding: utf-8 -*-
"""
Focused tests for priority modules to improve coverage from 64% to 85%.

Targets:
1. src/parsing/reference_resolver.py (56% -> 80%+)
2. src/parsing/regulation_parser.py (64% -> 80%+)
3. src/analyze_json.py (58% -> 80%+)
4. src/converter.py (64% -> 80%+)
5. src/inspect_json.py (70% -> 80%+)
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from src.analyze_json import (
    _validate_nodes,
    analyze_json,
    check_doc,
    check_structure,
)
from src.converter import HwpToMarkdownReader
from src.inspect_json import _resolve_input_path as inspect_resolve_input_path
from src.inspect_json import analyze_json as inspect_analyze_json
from src.inspect_json import main as inspect_main
from src.parsing.reference_resolver import ReferenceResolver
from src.parsing.regulation_parser import RegulationParser


class TestReferenceResolverUncoveredPaths(unittest.TestCase):
    """Tests for uncovered paths in ReferenceResolver (56% -> 80%+)."""

    def setUp(self):
        self.resolver = ReferenceResolver()

    def test_extract_references_basic(self):
        """Test extract_references with various reference patterns."""
        # Lines 48-71: extract_references method
        text = "제5조 및 제10조제1항에 따라 처리한다. 제3조의2를 참조할 것."
        refs = self.resolver.extract_references(text)
        self.assertEqual(len(refs), 3)
        self.assertEqual(refs[0]["text"], "제5조")
        self.assertEqual(refs[1]["text"], "제10조제1항")
        self.assertEqual(refs[2]["text"], "제3조의2")

    def test_extract_references_empty_string(self):
        """Test extract_references with empty string."""
        refs = self.resolver.extract_references("")
        self.assertEqual(refs, [])

    def test_extract_references_none(self):
        """Test extract_references with None."""
        refs = self.resolver.extract_references(None)
        self.assertEqual(refs, [])

    def test_extract_references_whitespace_only(self):
        """Test extract_references with whitespace only."""
        refs = self.resolver.extract_references("   \n\t  ")
        self.assertEqual(refs, [])

    def test_extract_references_with_spaces(self):
        """Test extract_references handles spaces in references."""
        # The regex pattern allows optional spaces: r"제\s*\d+\s*조"
        text = "제 5 조 및 제 10 조"  # Spaces between number and 조
        refs = self.resolver.extract_references(text)
        self.assertEqual(len(refs), 2)
        # Normalized reference (spaces are preserved in the match)
        self.assertEqual(refs[0]["text"], "제 5 조")
        self.assertEqual(refs[1]["text"], "제 10 조")

    def test_normalize_token(self):
        """Test _normalize_token removes whitespace."""
        # Line 270: _normalize_token
        result = self.resolver._normalize_token("제 5 조")
        self.assertEqual(result, "제5조")

    def test_normalize_token_with_tabs(self):
        """Test _normalize_token removes tabs."""
        result = self.resolver._normalize_token("제\t\t5\t조")
        self.assertEqual(result, "제5조")

    def test_format_article_key_with_sub(self):
        """Test _format_article_key with sub article."""
        result = self.resolver._format_article_key(5, 2)
        self.assertEqual(result, "제5조의2")

    def test_format_article_key_without_sub(self):
        """Test _format_article_key without sub article."""
        result = self.resolver._format_article_key(10, 0)
        self.assertEqual(result, "제10조")

    def test_parse_reference_token_full(self):
        """Test _parse_reference_token with full reference."""
        # Lines 231-263: _parse_reference_token
        result = self.resolver._parse_reference_token("제6조제2항제3호")
        self.assertEqual(result["article_main"], 6)
        self.assertEqual(result["article_sub"], 0)
        self.assertEqual(result["paragraph_no"], 2)
        self.assertEqual(result["item_no"], 3)

    def test_parse_reference_token_article_only(self):
        """Test _parse_reference_token with article only."""
        result = self.resolver._parse_reference_token("제5조")
        self.assertEqual(result["article_main"], 5)
        self.assertIsNone(result["paragraph_no"])
        self.assertIsNone(result["item_no"])

    def test_parse_reference_token_with_sub_article(self):
        """Test _parse_reference_token with sub article."""
        result = self.resolver._parse_reference_token("제6조의2")
        self.assertEqual(result["article_main"], 6)
        self.assertEqual(result["article_sub"], 2)

    def test_parse_reference_token_item_only(self):
        """Test _parse_reference_token with item only."""
        result = self.resolver._parse_reference_token("제3호")
        self.assertIsNone(result["article_main"])
        self.assertEqual(result["item_no"], 3)

    def test_build_reference_index(self):
        """Test _build_reference_index creates proper index."""
        doc = {
            "content": [
                {
                    "type": "article",
                    "id": "art1",
                    "display_no": "제5조",
                    "children": [
                        {
                            "type": "paragraph",
                            "id": "para1",
                            "sort_no": {"main": 1},
                            "children": [
                                {"type": "item", "id": "item1", "sort_no": {"main": 1}}
                            ],
                        }
                    ],
                }
            ]
        }
        index = self.resolver._build_reference_index(doc)
        self.assertIn("articles", index)
        self.assertIn("제5조", index["articles"])

    def test_build_article_reference_index(self):
        """Test _build_article_reference_index."""
        article_node = {
            "id": "art1",
            "children": [
                {
                    "type": "paragraph",
                    "id": "para1",
                    "sort_no": {"main": 1},
                    "children": [
                        {"type": "item", "id": "item1", "sort_no": {"main": 1}}
                    ],
                },
                {"type": "item", "id": "item2", "sort_no": {"main": 2}},
            ],
        }
        index = self.resolver._build_article_reference_index(article_node)
        self.assertEqual(index["id"], "art1")
        self.assertEqual(index["paragraphs"][1], "para1")
        self.assertEqual(index["items_by_paragraph"][1][1], "item1")
        self.assertEqual(index["items_direct"][2], "item2")

    def test_resolve_reference_target_article_only(self):
        """Test _resolve_reference_target with article only."""
        doc_index = {
            "articles": {
                "제5조": {
                    "id": "art5",
                    "paragraphs": {},
                    "items_by_paragraph": {},
                    "items_direct": {},
                }
            }
        }
        result = self.resolver._resolve_reference_target(
            "제5조",
            doc_index=doc_index,
            current_article=None,
            current_paragraph=None,
            current_item=None,
        )
        self.assertEqual(result, "art5")

    def test_resolve_reference_target_with_paragraph(self):
        """Test _resolve_reference_target with paragraph."""
        doc_index = {
            "articles": {
                "제5조": {
                    "id": "art5",
                    "paragraphs": {2: "para2"},
                    "items_by_paragraph": {},
                    "items_direct": {},
                }
            }
        }
        result = self.resolver._resolve_reference_target(
            "제5조제2항",
            doc_index=doc_index,
            current_article=None,
            current_paragraph=None,
            current_item=None,
        )
        self.assertEqual(result, "para2")

    def test_resolve_reference_target_not_found(self):
        """Test _resolve_reference_target when not found."""
        doc_index = {"articles": {}}
        result = self.resolver._resolve_reference_target(
            "제999조",
            doc_index=doc_index,
            current_article=None,
            current_paragraph=None,
            current_item=None,
        )
        self.assertIsNone(result)

    def test_resolve_references_in_nodes_updates_refs(self):
        """Test _resolve_references_in_nodes updates reference metadata."""
        nodes = [
            {
                "type": "article",
                "id": "art1",
                "display_no": "제1조",
                "references": [{"text": "제2조", "target": "제2조"}],
                "children": [],
            }
        ]
        doc_index = {
            "articles": {
                "제2조": {
                    "id": "art2",
                    "paragraphs": {},
                    "items_by_paragraph": {},
                    "items_direct": {},
                }
            }
        }
        self.resolver._resolve_references_in_nodes(
            nodes,
            doc_rule_code="1-1-1",
            doc_index=doc_index,
            current_article=None,
            current_paragraph=None,
            current_item=None,
        )
        self.assertEqual(nodes[0]["references"][0]["target_node_id"], "art2")
        self.assertEqual(nodes[0]["references"][0]["target_doc_rule_code"], "1-1-1")

    def test_resolve_references_in_nodes_with_empty_target(self):
        """Test _resolve_references_in_nodes with empty target."""
        nodes = [
            {
                "type": "article",
                "id": "art1",
                "display_no": "제1조",
                "references": [{"text": "", "target": ""}],
                "children": [],
            }
        ]
        doc_index = {"articles": {}}
        # Should not crash with empty target
        self.resolver._resolve_references_in_nodes(
            nodes,
            doc_rule_code="1-1-1",
            doc_index=doc_index,
            current_article=None,
            current_paragraph=None,
            current_item=None,
        )

    def test_resolve_all_with_addenda(self):
        """Test resolve_all processes both content and addenda."""
        docs = [
            {
                "metadata": {"rule_code": "1-1-1"},
                "content": [
                    {
                        "type": "article",
                        "id": "c1",
                        "display_no": "제1조",
                        "references": [],
                        "children": [],
                    }
                ],
                "addenda": [
                    {
                        "type": "article",
                        "id": "a1",
                        "display_no": "부칙",
                        "references": [],
                        "children": [],
                    }
                ],
            }
        ]
        result = self.resolver.resolve_all(docs)
        self.assertEqual(len(result), 1)

    def test_resolve_all_without_metadata(self):
        """Test resolve_all skips docs without metadata."""
        docs = [
            {
                "content": [
                    {
                        "type": "article",
                        "id": "c1",
                        "display_no": "제1조",
                        "references": [],
                        "children": [],
                    }
                ]
            }
        ]
        result = self.resolver.resolve_all(docs)
        self.assertEqual(len(result), 1)


class TestRegulationParserUncoveredPaths(unittest.TestCase):
    """Tests for uncovered paths in RegulationParser (64% -> 80%+)."""

    def setUp(self):
        self.parser = RegulationParser()

    def test_parse_flat_with_toc(self):
        """Test parsing with table of contents."""
        text = """차례

학칙
제1장 총칙
제1조 목적
"""
        result = self.parser.parse_flat(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "차례")

    def test_parse_flat_with_index(self):
        """Test parsing with index."""
        text = """찾아보기

가: 학칙
나: 교원인사규정
"""
        result = self.parser.parse_flat(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "찾아보기")

    def test_parse_flat_with_part(self):
        """Test parsing with part (편) divisions."""
        text = """제1편 총칙

제1장 목적
제1조 목적이다
"""
        result = self.parser.parse_flat(text)
        self.assertGreater(len(result), 0)

    def test_parse_flat_with_chapter(self):
        """Test parsing with chapter divisions."""
        text = """제1장 총칙

제1조 목적
본 대학의 목적은 다음과 같다.
"""
        result = self.parser.parse_flat(text)
        self.assertEqual(len(result), 1)

    def test_parse_flat_with_section(self):
        """Test parsing with section (절) divisions."""
        text = """제1장 총칙
제1절 목적

제1조 목적
"""
        result = self.parser.parse_flat(text)
        self.assertEqual(len(result), 1)

    def test_parse_flat_with_subsection(self):
        """Test parsing with subsection (관) divisions."""
        text = """제1장 총칙
제1관 총칙

제1조 목적
"""
        result = self.parser.parse_flat(text)
        self.assertEqual(len(result), 1)

    def test_parse_flat_article_1_split(self):
        """Test Article 1 split scenario (lines 153-197)."""
        text = """학칙

부칙
규정

제1조 목적
본 규정의 목적은 다음과 같다.
"""
        result = self.parser.parse_flat(text)
        # Should detect and split at Article 1
        self.assertGreater(len(result), 0)

    def test_parse_flat_article_with_title(self):
        """Test parsing article with title."""
        text = """제1조(목적) 본 규정의 목적이다"""
        result = self.parser.parse_flat(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["articles"][0]["article_no"], "제1조")
        self.assertEqual(result[0]["articles"][0]["title"], "목적")

    def test_parse_flat_article_with_sub_article(self):
        """Test parsing article with sub article (제N조의M)."""
        text = """제5조의2 정의
용어의 정의는 다음과 같다."""
        result = self.parser.parse_flat(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["articles"][0]["article_no"], "제5조의2")

    def test_parse_flat_with_paragraph(self):
        """Test parsing with paragraph (①-⑮)."""
        text = """제1조 목적
① 첫 번째 항
② 두 번째 항"""
        result = self.parser.parse_flat(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["articles"][0]["paragraphs"]), 2)

    def test_parse_flat_with_items(self):
        """Test parsing with items (1., 2., 3.)."""
        text = """제1조 목적
1. 첫 번째 항목
2. 두 번째 항목"""
        result = self.parser.parse_flat(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["articles"][0]["paragraphs"][0]["items"]), 2)

    def test_parse_flat_with_subitems(self):
        """Test parsing with subitems (가., 나., 다.)."""
        text = """제1조 목적
1. 항목
가. 첫 번째 하위항목
나. 두 번째 하위항목"""
        result = self.parser.parse_flat(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            len(result[0]["articles"][0]["paragraphs"][0]["items"][0]["subitems"]), 2
        )

    def test_parse_flat_with_appendices(self):
        """Test parsing with appendices (부칙)."""
        text = """제1조 목적
본 규정의 목적이다.

부칙
제1조 시행일
본 규정은 2024년 3월 1일부터 시행한다."""
        result = self.parser.parse_flat(text)
        # 부칙 triggers mode switch, creates new regulation
        self.assertGreaterEqual(len(result), 1)
        # Check that appendices were parsed
        has_appendices = any(len(r.get("appendices", [])) > 0 for r in result)
        self.assertTrue(has_appendices)

    def test_parse_flat_with_appendix_table(self):
        """Test parsing with appendix table reference."""
        text = """[별표 1]

부칙
제1조 시행일"""
        result = self.parser.parse_flat(text)
        self.assertEqual(len(result), 1)

    def test_resolve_sort_no_article(self):
        """Test resolve_sort_no for articles."""
        result = self.parser.resolve_sort_no("제29조의2", "article")
        self.assertEqual(result["main"], 29)
        self.assertEqual(result["sub"], 2)

    def test_resolve_sort_no_article_simple(self):
        """Test resolve_sort_no for simple articles."""
        result = self.parser.resolve_sort_no("제5조", "article")
        self.assertEqual(result["main"], 5)
        self.assertEqual(result["sub"], 0)

    def test_resolve_sort_no_chapter(self):
        """Test resolve_sort_no for chapters."""
        result = self.parser.resolve_sort_no("제1장", "chapter")
        self.assertEqual(result["main"], 1)

    def test_resolve_sort_no_paragraph(self):
        """Test resolve_sort_no for paragraphs."""
        result = self.parser.resolve_sort_no("①", "paragraph")
        self.assertEqual(result["main"], 1)

    def test_resolve_sort_no_paragraph_extended(self):
        """Test resolve_sort_no for extended paragraph range."""
        result = self.parser.resolve_sort_no("⑳", "paragraph")
        self.assertGreater(result["main"], 0)

    def test_resolve_sort_no_item(self):
        """Test resolve_sort_no for items."""
        result = self.parser.resolve_sort_no("1.", "item")
        self.assertEqual(result["main"], 1)

    def test_resolve_sort_no_subitem(self):
        """Test resolve_sort_no for subitems."""
        result = self.parser.resolve_sort_no("가.", "subitem")
        self.assertEqual(result["main"], 1)

    def test_resolve_sort_no_subitem_last(self):
        """Test resolve_sort_no for last subitem."""
        result = self.parser.resolve_sort_no("하.", "subitem")
        self.assertGreater(result["main"], 0)

    def test_resolve_sort_no_part(self):
        """Test resolve_sort_no for parts."""
        result = self.parser.resolve_sort_no("제1편", "part")
        self.assertEqual(result["main"], 1)

    def test_create_node(self):
        """Test create_node with all parameters."""
        node = self.parser.create_node(
            node_type="article",
            display_no="제1조",
            title="목적",
            text="목적 내용",
            sort_no={"main": 1, "sub": 0},
            children=[],
            confidence_score=0.9,
            references=[{"text": "제2조"}],
            metadata={"key": "value"},
        )
        self.assertEqual(node["type"], "article")
        self.assertEqual(node["display_no"], "제1조")
        self.assertEqual(node["title"], "목적")
        self.assertEqual(node["confidence_score"], 0.9)
        self.assertEqual(len(node["references"]), 1)

    def test_create_node_minimal(self):
        """Test create_node with minimal parameters."""
        node = self.parser.create_node(
            node_type="article", display_no="제1조", title=None, text=None
        )
        self.assertEqual(node["type"], "article")
        self.assertEqual(node["title"], "")
        self.assertEqual(node["text"], "")


class TestAnalyzeJsonUncoveredPaths(unittest.TestCase):
    """Tests for uncovered paths in analyze_json.py (58% -> 80%+)."""

    def test_check_structure_missing_docs(self):
        """Test check_structure with missing docs key."""
        data = {"file_name": "test.json"}
        errors = check_structure(data)
        self.assertIn("Missing root key: docs", errors)

    def test_check_structure_missing_file_name(self):
        """Test check_structure with missing file_name key."""
        data = {"docs": []}
        errors = check_structure(data)
        self.assertIn("Missing root key: file_name", errors)

    def test_check_structure_valid(self):
        """Test check_structure with valid structure."""
        data = {"docs": [], "file_name": "test.json"}
        errors = check_structure(data)
        self.assertEqual(len(errors), 0)

    def test_validate_nodes_invalid_node(self):
        """Test _validate_nodes with invalid node type."""
        nodes = ["not a dict", {"type": "article", "children": []}]
        errors = _validate_nodes(nodes, "test")
        self.assertIn("test #0: invalid node (not a dict)", errors)

    def test_validate_nodes_missing_type(self):
        """Test _validate_nodes with missing type."""
        nodes = [{"id": "test", "children": []}]
        errors = _validate_nodes(nodes, "test")
        self.assertIn("test #0: missing 'type'", errors)

    def test_validate_nodes_invalid_children(self):
        """Test _validate_nodes with invalid children type."""
        nodes = [{"type": "article", "children": "not a list"}]
        errors = _validate_nodes(nodes, "test")
        self.assertIn("test #0: 'children' is not a list", errors)

    def test_validate_nodes_invalid_sort_no(self):
        """Test _validate_nodes with invalid sort_no type."""
        nodes = [{"type": "article", "sort_no": "not a dict", "children": []}]
        errors = _validate_nodes(nodes, "test")
        self.assertIn("test #0: 'sort_no' is not a dict", errors)

    def test_check_doc_missing_title(self):
        """Test check_doc with missing title."""
        doc = {"content": [], "addenda": [], "attached_files": []}
        errors = check_doc(doc, 0)
        self.assertIn("Doc #0: Missing 'title' field", errors)

    def test_check_doc_empty_title(self):
        """Test check_doc with empty title."""
        doc = {"title": "   ", "content": [], "addenda": [], "attached_files": []}
        errors = check_doc(doc, 0)
        self.assertIn("Doc #0: Empty 'title' value", errors)

    def test_check_doc_invalid_content(self):
        """Test check_doc with invalid content type."""
        doc = {
            "title": "Test",
            "content": "not a list",
            "addenda": [],
            "attached_files": [],
        }
        errors = check_doc(doc, 0)
        self.assertIn("Doc #0: 'content' is not a list", errors)

    def test_check_doc_invalid_addenda(self):
        """Test check_doc with invalid addenda type."""
        doc = {
            "title": "Test",
            "content": [],
            "addenda": "not a list",
            "attached_files": [],
        }
        errors = check_doc(doc, 0)
        self.assertIn("Doc #0: 'addenda' is not a list", errors)

    def test_check_doc_invalid_attached_files(self):
        """Test check_doc with invalid attached_files type."""
        doc = {
            "title": "Test",
            "content": [],
            "addenda": [],
            "attached_files": "not a list",
        }
        errors = check_doc(doc, 0)
        self.assertIn("Doc #0: 'attached_files' is not a list", errors)

    def test_check_doc_invalid_attached_file_item(self):
        """Test check_doc with invalid attached file item."""
        doc = {
            "title": "Test",
            "content": [],
            "addenda": [],
            "attached_files": ["not a dict"],
        }
        errors = check_doc(doc, 0)
        # Check that error message contains key parts
        self.assertTrue(any("invalid type" in str(err).lower() for err in errors))

    def test_check_doc_missing_attached_file_title(self):
        """Test check_doc with missing attached file title."""
        doc = {
            "title": "Test",
            "content": [],
            "addenda": [],
            "attached_files": [{"url": "test"}],
        }
        errors = check_doc(doc, 0)
        self.assertIn("Doc #0: attached_files item #0 missing title", errors)

    def test_check_doc_valid_structure(self):
        """Test check_doc with valid structure."""
        doc = {
            "title": "Test Doc",
            "content": [{"type": "article", "children": []}],
            "addenda": [],
            "attached_files": [{"title": "Attachment 1"}],
        }
        errors = check_doc(doc, 0)
        self.assertEqual(len(errors), 0)

    def test_analyze_json_file_not_found(self, capsys=None):
        """Test analyze_json with non-existent file."""
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        analyze_json("/nonexistent/path/file.json")

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        self.assertIn("File found:", output)


class TestInspectJsonUncoveredPaths(unittest.TestCase):
    """Tests for uncovered paths in inspect_json.py (70% -> 80%+)."""

    def test_analyze_json_with_list(self):
        """Test analyze_json with list input."""
        data = [{"title": "Doc1", "content": [], "preamble": "test preamble"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            inspect_analyze_json(temp_path)

            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            # Just check it runs without error and produces output
            self.assertTrue(len(output) > 0)
            # The title is printed as part of the regulation section
            self.assertIn("Regulation 1", output)
        finally:
            os.unlink(temp_path)

    def test_analyze_json_with_dict(self):
        """Test analyze_json with dict input."""
        data = {
            "file_name": "test.hwp",
            "docs": [{"title": "Doc1", "content": [], "preamble": "preamble text"}],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            inspect_analyze_json(temp_path)

            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            # Check for file name being printed
            self.assertTrue("test.hwp" in output or "File" in output)
        finally:
            os.unlink(temp_path)

    def test_analyze_json_with_chapter_in_content(self):
        """Test analyze_json detects chapter in content."""
        data = [
            {
                "title": "Doc1",
                "content": [
                    {
                        "type": "article",
                        "display_no": "제1조",
                        "title": "총칙",
                        "text": "제2장 학생에 관한 규정",
                        "children": [],
                    }
                ],
                "preamble": "",
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            inspect_analyze_json(temp_path)

            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            self.assertIn("Articles with 'Chapter' in content: 1", output)
        finally:
            os.unlink(temp_path)

    def test_analyze_json_with_addenda_in_last_article(self):
        """Test analyze_json detects addenda info in last article."""
        data = [
            {
                "title": "Doc1",
                "content": [
                    {
                        "type": "article",
                        "display_no": "부칙",
                        "title": "시행일",
                        "text": "부칙 시행일은 2024년 3월 1일이다",
                        "children": [],
                    }
                ],
                "preamble": "",
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            sys.stdout = StringIO()

            inspect_analyze_json(temp_path)

            output = sys.stdout.getvalue()
            sys.stdout = old_stdout

            self.assertIn("Addenda info", output)
        finally:
            os.unlink(temp_path)

    def test_resolve_input_path_with_value(self):
        """Test _resolve_input_path with provided value."""
        result = inspect_resolve_input_path("/custom/path.json")
        self.assertEqual(result, "/custom/path.json")

    def test_resolve_input_path_default(self):
        """Test _resolve_input_path default behavior."""
        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data" / "output"
            data_dir.mkdir(parents=True)
            (data_dir / "test.json").write_text("{}")

            original_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                result = inspect_resolve_input_path(None)
                self.assertTrue(result.endswith("test.json"))
            finally:
                os.chdir(original_cwd)

    def test_main_invalid_path(self):
        """Test main with invalid path."""
        import sys

        old_argv = sys.argv
        try:
            sys.argv = ["inspect_json", "/nonexistent/path/that/does/not/exist.json"]
            # The main function doesn't catch FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                inspect_main()
        finally:
            sys.argv = old_argv


class TestConverterUncoveredPaths(unittest.TestCase):
    """Tests for uncovered paths in converter.py (64% -> 80%+)."""

    @patch("subprocess.Popen")
    @patch("tempfile.TemporaryDirectory")
    def test_load_data_fallback_to_index_html(self, mock_tmp, mock_popen):
        """Test fallback to index.html when index.xhtml doesn't exist."""
        mock_tmp.return_value.__enter__.return_value = "/tmp/dir"

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = iter(["<html></html>"])
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process

        # Create a test HWP file
        with tempfile.NamedTemporaryFile(suffix=".hwp", delete=False) as f:
            f.write(b"test hwp content")
            temp_hwp = f.name

        try:
            # Ensure file exists first
            self.assertTrue(Path(temp_hwp).exists())

            reader = HwpToMarkdownReader()

            # Mock exists to return False for index.xhtml but True for index.html
            def mock_exists(self):
                path_str = str(self)
                # The HWP file should exist
                if temp_hwp in path_str:
                    return True
                # index.html exists, index.xhtml doesn't
                if "index.html" in path_str and "xhtml" not in path_str:
                    return True
                return False

            with patch.object(Path, "exists", mock_exists):
                mock_file = mock_open(read_data="<html><body>Test</body></html>")
                with patch("builtins.open", mock_file):
                    docs = reader.load_data(Path(temp_hwp))
                    self.assertEqual(len(docs), 1)
        finally:
            os.unlink(temp_hwp)

    @patch("subprocess.Popen")
    @patch("tempfile.TemporaryDirectory")
    def test_load_data_fallback_to_glob(self, mock_tmp, mock_popen):
        """Test fallback to glob when specific files don't exist."""
        mock_tmp.return_value.__enter__.return_value = "/tmp/dir"

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = iter(["<html></html>"])
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process

        # Create a test HWP file
        with tempfile.NamedTemporaryFile(suffix=".hwp", delete=False) as f:
            f.write(b"test hwp content")
            temp_hwp = f.name

        try:
            # Ensure file exists first
            self.assertTrue(Path(temp_hwp).exists())

            reader = HwpToMarkdownReader()

            # Mock exists to handle all cases
            def mock_exists(self):
                path_str = str(self)
                # The HWP file should exist
                if temp_hwp in path_str:
                    return True
                # No index.xhtml or index.html files exist
                return False

            def mock_glob(self, pattern):
                if "html" in pattern:
                    mock_path = MagicMock()
                    mock_path.exists.return_value = True
                    mock_path.__str__ = lambda: "/tmp/dir/output.html"
                    return [mock_path]
                return []

            with patch.object(Path, "exists", mock_exists):
                with patch.object(Path, "glob", mock_glob):
                    mock_file = mock_open(read_data="<html><body>Test</body></html>")
                    with patch("builtins.open", mock_file):
                        docs = reader.load_data(Path(temp_hwp))
                        self.assertEqual(len(docs), 1)
        finally:
            os.unlink(temp_hwp)

    @patch("subprocess.Popen")
    @patch("tempfile.TemporaryDirectory")
    def test_load_data_no_html_found(self, mock_tmp, mock_popen):
        """Test when no HTML output is found."""
        mock_tmp.return_value.__enter__.return_value = "/tmp/dir"

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = iter(["<html></html>"])
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process

        # Create a test HWP file
        with tempfile.NamedTemporaryFile(suffix=".hwp", delete=False) as f:
            f.write(b"test hwp content")
            temp_hwp = f.name

        try:
            reader = HwpToMarkdownReader()

            with patch("pathlib.Path.exists", return_value=False):
                with patch("pathlib.Path.glob", return_value=[]):
                    with self.assertRaises(FileNotFoundError):
                        reader.load_data(Path(temp_hwp))
        finally:
            os.unlink(temp_hwp)

    @patch("subprocess.Popen")
    @patch("tempfile.TemporaryDirectory")
    def test_load_data_hwp5html_failure(self, mock_tmp, mock_popen):
        """Test when hwp5html conversion fails."""
        mock_tmp.return_value.__enter__.return_value = "/tmp/dir"

        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process

        # Create a test HWP file
        with tempfile.NamedTemporaryFile(suffix=".hwp", delete=False) as f:
            f.write(b"test hwp content")
            temp_hwp = f.name

        try:
            reader = HwpToMarkdownReader()

            with self.assertRaises(RuntimeError):
                reader.load_data(Path(temp_hwp))
        finally:
            os.unlink(temp_hwp)

    @patch("subprocess.Popen")
    @patch("tempfile.TemporaryDirectory")
    def test_load_data_without_markdownify(self, mock_tmp, mock_popen):
        """Test when markdownify is not available - uses HTML content directly."""
        mock_tmp.return_value.__enter__.return_value = "/tmp/dir"

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = iter(["<html></html>"])
        mock_process.wait = MagicMock()
        mock_popen.return_value = mock_process

        # Create a test HWP file
        with tempfile.NamedTemporaryFile(suffix=".hwp", delete=False) as f:
            f.write(b"test hwp content")
            temp_hwp = f.name

        try:
            # Patch md to None before importing reader
            with patch("src.converter.md", None):
                # Need to reload to apply the patch
                from importlib import reload

                import src.converter

                reload(src.converter)
                from src.converter import (
                    HwpToMarkdownReader as HwpToMarkdownReaderReloaded,
                )

                reader = HwpToMarkdownReaderReloaded()

                with patch.object(Path, "exists", return_value=True):
                    mock_file = mock_open(read_data="<html><body>Test</body></html>")
                    with patch("builtins.open", mock_file):
                        docs = reader.load_data(Path(temp_hwp))
                        self.assertEqual(len(docs), 1)
                        # When md is None, falls back to HTML content
                        self.assertIn("html_content", docs[0].metadata)
        finally:
            os.unlink(temp_hwp)


if __name__ == "__main__":
    unittest.main()
