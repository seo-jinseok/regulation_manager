"""
Unit tests for enhance_for_rag.py
"""

import pytest
from src.enhance_for_rag import (
    extract_amendment_history,
    extract_keywords,
    determine_status,
    build_path_label,
    build_full_text,
    enhance_node,
    enhance_document,
    enhance_json,
)


class TestExtractAmendmentHistory:
    """Tests for extract_amendment_history function."""

    def test_basic_revision(self):
        """Test extraction of basic revision pattern."""
        text = "이사와 감사는 이사회에서 선임한다. (개정 2006.11.06.)"
        result = extract_amendment_history(text)
        assert len(result) == 1
        assert result[0]["date"] == "2006-11-06"
        assert result[0]["type"] == "개정"

    def test_multiple_revisions(self):
        """Test extraction of multiple revisions in separate parentheses."""
        text = "임원의 임기는 4년으로 한다. (개정 2006.11.06.) 단, 중임할 수 있다. (개정 2022.04.21.)"
        result = extract_amendment_history(text)
        assert len(result) == 2
        assert result[0]["date"] == "2006-11-06"
        assert result[1]["date"] == "2022-04-21"

    def test_new_provision(self):
        """Test extraction of 신설 (new provision) pattern."""
        text = "추천위원회는 건학이념을 구현할 수 있는 자를 추천하여야 한다. (신설 2018.02.08.)"
        result = extract_amendment_history(text)
        assert len(result) == 1
        assert result[0]["date"] == "2018-02-08"
        assert result[0]["type"] == "신설"

    def test_deletion(self):
        """Test extraction of 삭제 (deletion) pattern."""
        text = "(삭제 2008.01.29.)"
        result = extract_amendment_history(text)
        assert len(result) == 1
        assert result[0]["date"] == "2008-01-29"
        assert result[0]["type"] == "삭제"

    def test_bracket_new_article(self):
        """Test extraction of [본조신설] pattern."""
        text = "위원회 운영에 필요한 사항은 별도로 정한다. [본조신설 2006.11.06.]"
        result = extract_amendment_history(text)
        assert len(result) == 1
        assert result[0]["date"] == "2006-11-06"
        assert result[0]["type"] == "신설"

    def test_no_history(self):
        """Test text with no amendment history."""
        text = "이 법인은 대한민국의 교육이념에 입각하여 교육을 실시함을 목적으로 한다."
        result = extract_amendment_history(text)
        assert len(result) == 0

    def test_date_with_spaces(self):
        """Test dates with extra spaces."""
        text = "(개정 2006. 11. 06.)"
        result = extract_amendment_history(text)
        assert len(result) == 1
        assert result[0]["date"] == "2006-11-06"

    def test_sorted_by_date(self):
        """Test that results are sorted by date."""
        text = "(개정 2022.04.21.) (개정 2006.11.06.)"
        result = extract_amendment_history(text)
        assert result[0]["date"] == "2006-11-06"
        assert result[1]["date"] == "2022-04-21"


class TestExtractKeywords:
    """Tests for extract_keywords function."""

    def test_basic_keywords(self):
        """Test extraction of basic keywords."""
        text = "이사회는 이사와 감사로 구성한다."
        result = extract_keywords(text)
        assert "이사" in result
        assert "감사" in result

    def test_education_keywords(self):
        """Test extraction of education-related keywords."""
        text = "교원은 학생의 교육을 담당한다."
        result = extract_keywords(text)
        assert "교원" in result
        assert "학생" in result
        assert "교육" in result

    def test_empty_text(self):
        """Test empty text returns empty list."""
        result = extract_keywords("")
        assert result == []

    def test_none_text(self):
        """Test None text returns empty list."""
        result = extract_keywords(None)
        assert result == []

    def test_no_keywords(self):
        """Test text with no matching keywords."""
        text = "이것은 테스트입니다."
        result = extract_keywords(text)
        assert result == []

    def test_sorted_and_unique(self):
        """Test that results are sorted and unique."""
        text = "교원은 교원이다. 학생은 학생이다."
        result = extract_keywords(text)
        # Should not have duplicates
        assert len(result) == len(set(result))
        # Should be sorted
        assert result == sorted(result)


class TestDetermineStatus:
    """Tests for determine_status function."""

    def test_active_status(self):
        """Test that normal titles return active status."""
        status, abolished_date = determine_status("교원인사규정")
        assert status == "active"
        assert abolished_date is None

    def test_abolished_status_with_bracket(self):
        """Test that titles with 【폐지】 return abolished status."""
        status, abolished_date = determine_status("시간강사위촉규정【폐지】")
        assert status == "abolished"

    def test_abolished_status_without_bracket(self):
        """Test that titles with 폐지 return abolished status."""
        status, abolished_date = determine_status("창업보육센터규정【폐지】")
        assert status == "abolished"


class TestBuildPathLabel:
    """Tests for build_path_label function."""

    def test_with_display_no_and_title(self):
        """Test node with both display_no and title."""
        node = {"display_no": "제1조", "title": "목적"}
        assert build_path_label(node) == "제1조 목적"

    def test_with_only_display_no(self):
        """Test node with only display_no."""
        node = {"display_no": "①", "title": ""}
        assert build_path_label(node) == "①"

    def test_with_only_title(self):
        """Test node with only title."""
        node = {"display_no": "", "title": "제1장 총칙"}
        assert build_path_label(node) == "제1장 총칙"

    def test_with_neither(self):
        """Test node with neither display_no nor title."""
        node = {"display_no": "", "title": ""}
        assert build_path_label(node) == ""


class TestBuildFullText:
    """Tests for build_full_text function."""

    def test_with_path(self):
        """Test building full text with parent path."""
        parent_path = ["동의대학교학칙", "제1장 총칙"]
        node = {"display_no": "제1조", "title": "목적", "text": "본 대학교는..."}
        result = build_full_text(parent_path, node)
        assert result == "[동의대학교학칙 > 제1장 총칙 > 제1조 목적] 본 대학교는..."

    def test_without_path(self):
        """Test building full text without parent path."""
        node = {"display_no": "", "title": "", "text": "본 규정은..."}
        result = build_full_text([], node)
        assert result == "본 규정은..."

    def test_empty_text(self):
        """Test node with empty text."""
        node = {"display_no": "제1조", "title": "목적", "text": ""}
        result = build_full_text(["Path"], node)
        assert result == ""


class TestEnhanceNode:
    """Tests for enhance_node function."""

    def test_basic_enhancement(self):
        """Test basic node enhancement."""
        node = {
            "type": "article",
            "display_no": "제1조",
            "title": "목적",
            "text": "이 규정은 교원의 인사에 관한 사항을 규정한다. (개정 2006.11.06.)",
            "children": [],
        }
        enhance_node(node, [], "테스트규정")
        
        assert "parent_path" in node
        assert node["parent_path"] == ["테스트규정"]
        assert "full_text" in node
        assert "keywords" in node
        assert "교원" in node["keywords"]
        assert "규정" in node["keywords"]
        assert "amendment_history" in node
        assert node["amendment_history"][0]["date"] == "2006-11-06"

    def test_nested_enhancement(self):
        """Test enhancement of nested nodes."""
        node = {
            "type": "article",
            "display_no": "제1조",
            "title": "목적",
            "text": "",
            "children": [
                {
                    "type": "paragraph",
                    "display_no": "①",
                    "title": "",
                    "text": "이 규정은 학생의 학사에 관한 사항을 규정한다.",
                    "children": [],
                }
            ],
        }
        enhance_node(node, [], "학칙")
        
        child = node["children"][0]
        assert child["parent_path"] == ["학칙", "제1조 목적"]


class TestEnhanceDocument:
    """Tests for enhance_document function."""

    def test_active_document(self):
        """Test enhancing an active document."""
        doc = {
            "title": "교원인사규정",
            "doc_type": "regulation",
            "content": [],
            "addenda": [],
        }
        enhance_document(doc)
        assert doc["status"] == "active"
        assert "is_index_duplicate" not in doc

    def test_abolished_document(self):
        """Test enhancing an abolished document."""
        doc = {
            "title": "시간강사위촉규정【폐지】",
            "doc_type": "regulation",
            "content": [],
            "addenda": [],
        }
        enhance_document(doc)
        assert doc["status"] == "abolished"

    def test_index_document(self):
        """Test enhancing an index document."""
        doc = {
            "title": "차례",
            "doc_type": "toc",
            "content": [],
            "addenda": [],
        }
        enhance_document(doc)
        assert doc["is_index_duplicate"] is True


class TestEnhanceJson:
    """Tests for enhance_json function."""

    def test_metadata_added(self):
        """Test that RAG metadata is added."""
        data = {"docs": []}
        result = enhance_json(data)
        assert result["rag_enhanced"] is True
        assert result["rag_schema_version"] == "1.0"

    def test_all_docs_enhanced(self):
        """Test that all documents are enhanced."""
        data = {
            "docs": [
                {"title": "규정1", "doc_type": "regulation", "content": [], "addenda": []},
                {"title": "규정2【폐지】", "doc_type": "regulation", "content": [], "addenda": []},
            ]
        }
        result = enhance_json(data)
        assert result["docs"][0]["status"] == "active"
        assert result["docs"][1]["status"] == "abolished"
