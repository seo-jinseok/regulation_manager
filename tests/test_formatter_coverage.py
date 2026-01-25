"""
Additional tests for formatter.py to improve coverage.

Focuses on testable helper methods and edge cases:
- _doc_has_content
- _index_kind
- _reorder_and_trim_docs
- _merge_adjacent_docs
- _assign_doc_types
- _parse_toc_rule_codes
"""

import unittest

from src.formatter import RegulationFormatter


class TestDocHasContent(unittest.TestCase):
    """Tests for _doc_has_content method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_doc_with_content(self):
        """Test document with content returns True."""
        doc = {"content": [{"type": "article"}]}
        self.assertTrue(self.formatter._doc_has_content(doc))

    def test_doc_with_addenda(self):
        """Test document with addenda returns True."""
        doc = {"addenda": [{"type": "addendum"}]}
        self.assertTrue(self.formatter._doc_has_content(doc))

    def test_doc_with_attached_files(self):
        """Test document with attached_files returns True."""
        doc = {"attached_files": [{"name": "file.pdf"}]}
        self.assertTrue(self.formatter._doc_has_content(doc))

    def test_doc_with_all_three(self):
        """Test document with all content types returns True."""
        doc = {
            "content": [{}],
            "addenda": [{}],
            "attached_files": [{}],
        }
        self.assertTrue(self.formatter._doc_has_content(doc))

    def test_empty_doc(self):
        """Test empty document returns False."""
        doc = {}
        self.assertFalse(self.formatter._doc_has_content(doc))

    def test_doc_with_empty_content_lists(self):
        """Test document with empty content lists returns False."""
        doc = {"content": [], "addenda": [], "attached_files": []}
        self.assertFalse(self.formatter._doc_has_content(doc))

    def test_doc_with_none_values(self):
        """Test document with None values returns False."""
        doc = {"content": None, "addenda": None, "attached_files": None}
        self.assertFalse(self.formatter._doc_has_content(doc))


class TestIndexKind(unittest.TestCase):
    """Tests for _index_kind method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_toc_title(self):
        """Test '차례' title returns 'toc'."""
        doc = {"title": "차례"}
        self.assertEqual(self.formatter._index_kind(doc), "toc")

    def test_mokcha_title(self):
        """Test '목차' title returns 'toc'."""
        doc = {"title": "목차"}
        self.assertEqual(self.formatter._index_kind(doc), "toc")

    def test_index_alpha_title(self):
        """Test '찾아보기' with 가나다순 returns 'index_alpha'."""
        doc = {"title": "찾아보기", "preamble": "<가나다순>"}
        self.assertEqual(self.formatter._index_kind(doc), "index_alpha")

    def test_index_alpha_no_brackets(self):
        """Test '찾아보기' with 가나다순 without brackets."""
        doc = {"title": "찾아보기", "preamble": "가나다순 목록"}
        self.assertEqual(self.formatter._index_kind(doc), "index_alpha")

    def test_index_dept_title(self):
        """Test '찾아보기' with 소관부서별 returns 'index_dept'."""
        doc = {"title": "찾아보기", "preamble": "<소관부서별>"}
        self.assertEqual(self.formatter._index_kind(doc), "index_dept")

    def test_index_dept_no_brackets(self):
        """Test '찾아보기' with 소관부서별 without brackets."""
        doc = {"title": "찾아보기", "preamble": "소관부서별 목록"}
        self.assertEqual(self.formatter._index_kind(doc), "index_dept")

    def test_generic_index_title(self):
        """Test '찾아보기' without specific markers returns 'index'."""
        doc = {"title": "찾아보기", "preamble": "일반 내용"}
        self.assertEqual(self.formatter._index_kind(doc), "index")

    def test_empty_title(self):
        """Test empty title returns None."""
        doc = {"title": ""}
        self.assertIsNone(self.formatter._index_kind(doc))

    def test_none_title(self):
        """Test None title returns None."""
        doc = {"title": None}
        self.assertIsNone(self.formatter._index_kind(doc))

    def test_non_index_title(self):
        """Test non-index title returns None."""
        doc = {"title": "교원인사규정"}
        self.assertIsNone(self.formatter._index_kind(doc))


class TestReorderAndTrimDocs(unittest.TestCase):
    """Tests for _reorder_and_trim_docs method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_empty_list(self):
        """Test with empty list returns empty list."""
        result = self.formatter._reorder_and_trim_docs([])
        self.assertEqual(result, [])

    def test_single_content_doc(self):
        """Test single content document."""
        docs = [{"title": "규정", "content": [{}]}]
        result = self.formatter._reorder_and_trim_docs(docs)
        self.assertEqual(len(result), 1)

    def test_index_docs_ordering(self):
        """Test index documents are ordered correctly."""
        docs = [
            {"title": "찾아보기", "preamble": "소관부서별"},  # index_dept
            {"title": "차례"},  # toc
            {"title": "찾아보기", "preamble": "가나다순"},  # index_alpha
        ]
        result = self.formatter._reorder_and_trim_docs(docs)
        titles = [d.get("title") for d in result]
        # Order should be: toc (0), index_alpha (1), index_dept (2)
        self.assertEqual(titles, ["차례", "찾아보기", "찾아보기"])

    def test_content_docs_after_index(self):
        """Test content documents come after index documents."""
        docs = [
            {"title": "교원인사규정", "content": [{}]},
            {"title": "차례"},
            {"title": "학칙", "content": [{}]},
        ]
        result = self.formatter._reorder_and_trim_docs(docs)
        titles = [d.get("title") for d in result]
        # toc should be first
        self.assertEqual(titles[0], "차례")
        # content docs after
        self.assertIn("교원인사규정", titles[1:])
        self.assertIn("학칙", titles[1:])

    def test_empty_docs_dropped(self):
        """Test empty docs without index kind are dropped."""
        docs = [
            {"title": "빈문서"},
            {"title": "규정", "content": [{}]},
        ]
        result = self.formatter._reorder_and_trim_docs(docs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "규정")

    def test_part_cleared_for_index_docs(self):
        """Test part field is cleared for index docs."""
        docs = [
            {"title": "차례", "part": "제1편"},
        ]
        result = self.formatter._reorder_and_trim_docs(docs)
        self.assertIsNone(result[0].get("part"))


class TestMergeAdjacentDocs(unittest.TestCase):
    """Tests for _merge_adjacent_docs method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_empty_list(self):
        """Test with empty list returns empty list."""
        result = self.formatter._merge_adjacent_docs([])
        self.assertEqual(result, [])

    def test_single_doc(self):
        """Test single document returns as-is."""
        docs = [{"title": "규정"}]
        result = self.formatter._merge_adjacent_docs(docs)
        self.assertEqual(len(result), 1)

    def test_noise_doc_dropped(self):
        """Test noise doc (규정집 관리 현황표) is dropped."""
        docs = [
            {"title": "규정1"},
            {"title": "규정집 관리 현황표"},
            {"title": "규정2"},
        ]
        result = self.formatter._merge_adjacent_docs(docs)
        titles = [d.get("title") for d in result]
        self.assertNotIn("규정집 관리 현황표", titles)

    def test_merge_no_title_same_part(self):
        """Test doc without title is merged into previous if same part."""
        docs = [
            {
                "title": "규정",
                "part": "제1편",
                "content": [{"id": 1}],
                "addenda": [],
                "attached_files": [],
                "metadata": {},  # Initialize metadata to avoid KeyError
            },
            {
                "title": "",
                "part": "제1편",
                "content": [{"id": 2}],
                "addenda": [{"id": 3}],
                "attached_files": [{"name": "file.pdf"}],
                "metadata": {"rule_code": "1-1-1"},
            },
        ]
        result = self.formatter._merge_adjacent_docs(docs)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]["content"]), 2)
        self.assertEqual(len(result[0]["addenda"]), 1)
        self.assertEqual(len(result[0]["attached_files"]), 1)
        self.assertEqual(result[0]["metadata"]["rule_code"], "1-1-1")

    def test_no_merge_different_part(self):
        """Test doc without title is NOT merged if different part."""
        docs = [
            {
                "title": "규정",
                "part": "제1편",
                "content": [{"id": 1}],
                "addenda": [],
                "attached_files": [],
            },
            {
                "title": "",
                "part": "제2편",
                "content": [{"id": 2}],
                "addenda": [],
                "attached_files": [],
            },
        ]
        result = self.formatter._merge_adjacent_docs(docs)
        self.assertEqual(len(result), 2)

    def test_no_merge_when_previous_has_no_part(self):
        """Test no merge when previous doc has no part."""
        docs = [
            {
                "title": "규정",
                "part": None,
                "content": [{"id": 1}],
                "addenda": [],
                "attached_files": [],
            },
            {
                "title": "",
                "part": "제1편",
                "content": [{"id": 2}],
                "addenda": [],
                "attached_files": [],
            },
        ]
        result = self.formatter._merge_adjacent_docs(docs)
        self.assertEqual(len(result), 2)

    def test_metadata_none_values_filtered(self):
        """Test None values in metadata are filtered during merge."""
        docs = [
            {
                "title": "규정",
                "part": "제1편",
                "content": [],
                "addenda": [],
                "attached_files": [],
                "metadata": {"rule_code": "1-1-1"},
            },
            {
                "title": "",
                "part": "제1편",
                "content": [{}],  # Need content for merge to happen
                "addenda": [],
                "attached_files": [],
                "metadata": {"rule_code": None, "extra": "value"},
            },
        ]
        result = self.formatter._merge_adjacent_docs(docs)
        self.assertEqual(len(result), 1)
        # rule_code should not be overwritten with None
        self.assertEqual(result[0]["metadata"]["rule_code"], "1-1-1")
        # extra should be added
        self.assertEqual(result[0]["metadata"]["extra"], "value")


class TestAssignDocTypes(unittest.TestCase):
    """Tests for _assign_doc_types method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_index_doc_gets_type(self):
        """Test index documents get appropriate doc_type."""
        docs = [
            {"title": "차례"},
            {"title": "찾아보기", "preamble": "가나다순"},
            {"title": "찾아보기", "preamble": "소관부서별"},
        ]
        self.formatter._assign_doc_types(docs)
        self.assertEqual(docs[0]["doc_type"], "toc")
        self.assertEqual(docs[1]["doc_type"], "index_alpha")
        self.assertEqual(docs[2]["doc_type"], "index_dept")

    def test_regulation_with_rule_code(self):
        """Test doc with rule_code gets 'regulation' type."""
        docs = [{"title": "규정", "metadata": {"rule_code": "1-1-1"}}]
        self.formatter._assign_doc_types(docs)
        self.assertEqual(docs[0]["doc_type"], "regulation")

    def test_doc_with_content_gets_note(self):
        """Test doc with content but no rule_code gets 'note' type."""
        docs = [{"title": "비고", "content": [{}]}]
        self.formatter._assign_doc_types(docs)
        self.assertEqual(docs[0]["doc_type"], "note")

    def test_empty_doc_gets_unknown(self):
        """Test empty doc gets 'unknown' type."""
        docs = [{"title": "제목"}]
        self.formatter._assign_doc_types(docs)
        self.assertEqual(docs[0]["doc_type"], "unknown")

    def test_existing_doc_type_preserved(self):
        """Test existing doc_type is not overwritten."""
        docs = [{"title": "규정", "doc_type": "custom"}]
        self.formatter._assign_doc_types(docs)
        self.assertEqual(docs[0]["doc_type"], "custom")


class TestParseTocRuleCodes(unittest.TestCase):
    """Tests for _parse_toc_rule_codes method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_empty_preamble(self):
        """Test empty preamble returns empty dict."""
        result = self.formatter._parse_toc_rule_codes("")
        self.assertEqual(result, {})

    def test_none_preamble(self):
        """Test None preamble returns empty dict."""
        result = self.formatter._parse_toc_rule_codes(None)
        self.assertEqual(result, {})

    def test_basic_rule_code(self):
        """Test basic rule code extraction."""
        preamble = "교원인사규정 3-1-5"
        result = self.formatter._parse_toc_rule_codes(preamble)
        self.assertEqual(result["교원인사규정"], "3-1-5")

    def test_rule_code_with_range(self):
        """Test rule code with page range."""
        preamble = "교원인사규정 3-1-5 ~ 10"
        result = self.formatter._parse_toc_rule_codes(preamble)
        self.assertEqual(result["교원인사규정"], "3-1-5")

    def test_multiple_entries(self):
        """Test multiple rule code entries."""
        preamble = "교원인사규정 3-1-5\n학칙 1-1-1"
        result = self.formatter._parse_toc_rule_codes(preamble)
        self.assertEqual(result["교원인사규정"], "3-1-5")
        self.assertEqual(result["학칙"], "1-1-1")

    def test_title_with_spaces(self):
        """Test title with spaces is preserved."""
        preamble = "대학원 학사 규정 3-2-1"
        result = self.formatter._parse_toc_rule_codes(preamble)
        self.assertEqual(result["대학원 학사 규정"], "3-2-1")

    def test_pipe_delimited_lines(self):
        """Test pipe-delimited TOC format."""
        preamble = "| 교원인사규정 | 3-1-5 |"
        result = self.formatter._parse_toc_rule_codes(preamble)
        # After stripping, it becomes "교원인사규정 | 3-1-5 |"
        # The regex matches, stripping trailing content
        self.assertEqual(len(result), 1)
        # The title may include trailing pipe
        self.assertTrue("교원인사규정" in list(result.keys())[0])

    def test_various_dash_types(self):
        """Test various dash types are normalized."""
        preamble = "규정 3—1—5"  # em dashes
        result = self.formatter._parse_toc_rule_codes(preamble)
        self.assertEqual(result["규정"], "3-1-5")

    def test_en_dash_normalization(self):
        """Test en-dash normalization."""
        preamble = "규정 3–1–5"  # en dashes
        result = self.formatter._parse_toc_rule_codes(preamble)
        self.assertEqual(result["규정"], "3-1-5")

    def test_empty_lines_skipped(self):
        """Test empty lines are skipped."""
        preamble = "교원인사규정 3-1-5\n\n학칙 1-1-1"
        result = self.formatter._parse_toc_rule_codes(preamble)
        self.assertEqual(len(result), 2)

    def test_invalid_line_format_skipped(self):
        """Test lines not matching pattern are skipped."""
        preamble = "교원인사규정 3-1-5\nInvalid line\n학칙 1-1-1"
        result = self.formatter._parse_toc_rule_codes(preamble)
        self.assertEqual(len(result), 2)


class TestTitlesMatch(unittest.TestCase):
    """Tests for _titles_match method."""

    def setUp(self):
        self.formatter = RegulationFormatter()

    def test_identical_titles(self):
        """Test identical titles match."""
        result = self.formatter._titles_match("교원인사규정", "교원인사규정")
        self.assertTrue(result)

    def test_titles_with_different_spacing(self):
        """Test titles with different spacing match."""
        result = self.formatter._titles_match("교원 인사 규정", "교원인사규정")
        self.assertTrue(result)

    def test_titles_with_special_chars(self):
        """Test titles with special characters match."""
        result = self.formatter._titles_match("제1장 총칙", "제1장 총칙")
        self.assertTrue(result)

    def test_different_titles_no_match(self):
        """Test different titles don't match."""
        result = self.formatter._titles_match("교원인사규정", "학칙")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
