"""
Unit tests for enhance_for_rag.py
"""

from src.enhance_for_rag import (
    build_full_text,
    build_path_label,
    calculate_token_count,
    determine_chunk_level,
    determine_status,
    enhance_document,
    enhance_json,
    enhance_node,
    extract_amendment_history,
    extract_effective_date,
    extract_keywords,
    extract_keywords_simple,
    is_node_searchable,
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
        """Test extraction of basic keywords with weights."""
        text = "이사회는 이사와 감사로 구성한다."
        result = extract_keywords(text)
        terms = [k["term"] for k in result]
        assert "이사" in terms
        assert "감사" in terms
        # Check weights are present
        assert all("weight" in k for k in result)

    def test_education_keywords(self):
        """Test extraction of education-related keywords with weights."""
        text = "교원은 학생의 교육을 담당한다."
        result = extract_keywords(text)
        terms = [k["term"] for k in result]
        assert "교원" in terms
        assert "학생" in terms
        assert "교육" in terms

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

    def test_sorted_by_weight(self):
        """Test that results are sorted by weight descending."""
        text = "교원은 교원이다. 학생은 학생이다. 위원장이 의결한다."
        result = extract_keywords(text)
        # Check sorted by weight descending
        weights = [k["weight"] for k in result]
        assert weights == sorted(weights, reverse=True)
        # No duplicates
        terms = [k["term"] for k in result]
        assert len(terms) == len(set(terms))

    def test_simple_extraction(self):
        """Test extract_keywords_simple for backward compatibility."""
        text = "이사회는 이사와 감사로 구성한다."
        result = extract_keywords_simple(text)
        assert "이사" in result
        assert "감사" in result
        # Result is a simple list
        assert all(isinstance(k, str) for k in result)


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

    def test_dedupes_label_against_last_path_segment(self):
        """Whitespace-variant label duplicates should not repeat in full_text."""
        parent_path = ["교원인사규정", "부칙"]
        node = {"display_no": "", "title": "부 칙", "text": "내용"}
        result = build_full_text(parent_path, node)
        assert "부칙 > 부 칙" not in result
        assert "[교원인사규정 > 부칙]" in result


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
        # keywords is now a list of dicts
        assert "keywords" in node
        terms = [k["term"] for k in node["keywords"]]
        assert "교원" in terms
        assert "규정" in terms
        assert "amendment_history" in node
        assert node["amendment_history"][0]["date"] == "2006-11-06"
        # New fields
        assert "chunk_level" in node
        assert node["chunk_level"] == "article"
        assert "is_searchable" in node
        assert node["is_searchable"] is True
        assert "embedding_text" in node
        assert "token_count" in node

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
        assert result["rag_schema_version"] == "2.0"

    def test_all_docs_enhanced(self):
        """Test that all documents are enhanced."""
        data = {
            "docs": [
                {
                    "title": "규정1",
                    "doc_type": "regulation",
                    "content": [],
                    "addenda": [],
                },
                {
                    "title": "규정2【폐지】",
                    "doc_type": "regulation",
                    "content": [],
                    "addenda": [],
                },
            ]
        }
        result = enhance_json(data)
        assert result["docs"][0]["status"] == "active"
        assert result["docs"][1]["status"] == "abolished"


class TestNewHelperFunctions:
    """Tests for new RAG helper functions."""

    def test_determine_chunk_level_article(self):
        """Test chunk level for article type."""
        node = {"type": "article"}
        assert determine_chunk_level(node) == "article"

    def test_determine_chunk_level_paragraph(self):
        """Test chunk level for paragraph type."""
        node = {"type": "paragraph"}
        assert determine_chunk_level(node) == "paragraph"

    def test_determine_chunk_level_unknown(self):
        """Test chunk level for unknown type defaults to text."""
        node = {"type": "unknown_type"}
        assert determine_chunk_level(node) == "text"

    def test_calculate_token_count(self):
        """Test token count calculation."""
        # Korean text: approximately 2.5 chars per token
        text = "이 규정은 교원의 인사에 관한 사항을 규정한다."  # 23 chars
        count = calculate_token_count(text)
        assert count > 0
        assert count == int(len(text) / 2.5)

    def test_calculate_token_count_empty(self):
        """Test token count for empty text."""
        assert calculate_token_count("") == 0
        assert calculate_token_count(None) == 0

    def test_is_node_searchable_with_text(self):
        """Test searchable for node with text."""
        node = {"text": "Some content", "children": [{"type": "item"}]}
        assert is_node_searchable(node) is True

    def test_is_node_searchable_leaf(self):
        """Test searchable for leaf node without text."""
        node = {"text": "", "children": []}
        assert is_node_searchable(node) is True

    def test_is_node_searchable_parent_no_text(self):
        """Test searchable for parent node without text."""
        node = {"text": "", "children": [{"type": "item"}]}
        assert is_node_searchable(node) is False

    def test_extract_effective_date_basic(self):
        """Test effective date extraction."""
        text = "이 규정은 2024년 1월 1일부터 시행한다."
        assert extract_effective_date(text) == "2024-01-01"

    def test_extract_effective_date_dot_format(self):
        """Test effective date with dot format."""
        text = "이 변경 정관은 2023. 12. 25.부터 시행한다."
        result = extract_effective_date(text)
        assert result == "2023-12-25"

    def test_extract_effective_date_not_found(self):
        """Test when no effective date is found."""
        text = "이 규정은 공포한 날부터 시행한다."
        assert extract_effective_date(text) is None

    def test_extract_effective_date_empty(self):
        """Test with empty text."""
        assert extract_effective_date("") is None
        assert extract_effective_date(None) is None


class TestEnhanceNodeNewFields:
    """Tests for new fields in enhance_node."""

    def test_addendum_effective_date(self):
        """Test effective_date extraction for addendum nodes."""
        node = {
            "type": "addendum_item",
            "display_no": "1.",
            "title": "",
            "text": "이 규정은 2024년 3월 1일부터 시행한다.",
            "children": [],
        }
        enhance_node(node, ["부칙"], "테스트규정")
        assert "effective_date" in node
        assert node["effective_date"] == "2024-03-01"

    def test_embedding_text_with_context(self):
        """Test that embedding_text contains path context for better search."""
        node = {
            "type": "article",
            "display_no": "제1조",
            "title": "목적",
            "text": "본 규정의 목적",
            "children": [],
        }
        enhance_node(node, [], "학칙")
        # embedding_text should include path context for better search
        assert "학칙" in node["embedding_text"]
        assert "제1조 목적" in node["embedding_text"]
        assert "본 규정의 목적" in node["embedding_text"]
        # full_text should include path as well
        assert "[" in node["full_text"] and ">" in node["full_text"]


class TestBuildEmbeddingText:
    """Tests for build_embedding_text function."""

    def test_includes_path_context(self):
        """Test that embedding text includes path context."""
        from src.enhance_for_rag import build_embedding_text

        parent_path = ["동의대학교학칙", "제3장 학사", "제1절 수업"]
        node = {
            "text": "수업일수는 연간 16주 이상으로 한다.",
            "display_no": "제15조",
            "title": "수업일수",
        }
        result = build_embedding_text(parent_path, node)
        assert "제3장 학사" in result
        assert "제1절 수업" in result
        assert "제15조 수업일수" in result
        assert "수업일수는 연간 16주 이상" in result

    def test_truncates_long_path(self):
        """Test that only last 3 path segments are used."""
        from src.enhance_for_rag import build_embedding_text

        parent_path = ["규정", "제1편", "제1장", "제1절", "제1관"]
        node = {"text": "내용", "display_no": "제1조", "title": ""}
        result = build_embedding_text(parent_path, node)
        # Only last 3 segments
        assert "규정" not in result
        assert "제1편" not in result
        assert "제1장" in result
        assert "제1절" in result
        assert "제1관" in result

    def test_empty_text_returns_empty(self):
        """Test that empty text returns empty string."""
        from src.enhance_for_rag import build_embedding_text

        result = build_embedding_text(["path"], {"text": ""})
        assert result == ""

    def test_no_path_includes_label(self):
        """Test that label is included even without path."""
        from src.enhance_for_rag import build_embedding_text

        node = {"text": "내용", "display_no": "제1조", "title": "목적"}
        result = build_embedding_text([], node)
        assert "제1조 목적" in result
        assert "내용" in result

    def test_dedupes_label_against_last_path_segment(self):
        """Whitespace-variant label duplicates should not repeat in embedding_text."""
        from src.enhance_for_rag import build_embedding_text

        parent_path = ["교원인사규정", "부칙"]
        node = {"text": "내용", "display_no": "", "title": "부 칙"}
        result = build_embedding_text(parent_path, node)
        assert "부칙 > 부 칙" not in result
        assert "부칙: 내용" in result


class TestExtractLastRevisionDate:
    """Tests for extract_last_revision_date function."""

    def test_single_addendum(self):
        """Test with single addendum containing effective_date."""
        from src.enhance_for_rag import extract_last_revision_date

        doc = {
            "addenda": [
                {"effective_date": "2020-01-01", "children": []}
            ]
        }
        assert extract_last_revision_date(doc) == "2020-01-01"

    def test_multiple_addenda(self):
        """Test with multiple addenda, returns the latest date."""
        from src.enhance_for_rag import extract_last_revision_date

        doc = {
            "addenda": [
                {"effective_date": "2018-03-01", "children": []},
                {"effective_date": "2024-06-15", "children": []},
                {"effective_date": "2020-01-01", "children": []},
            ]
        }
        assert extract_last_revision_date(doc) == "2024-06-15"

    def test_nested_effective_date(self):
        """Test with effective_date in children nodes."""
        from src.enhance_for_rag import extract_last_revision_date

        doc = {
            "addenda": [
                {
                    "children": [
                        {"effective_date": "2023-05-01", "children": []}
                    ]
                }
            ]
        }
        assert extract_last_revision_date(doc) == "2023-05-01"

    def test_no_addenda(self):
        """Test document with no addenda."""
        from src.enhance_for_rag import extract_last_revision_date

        doc = {"addenda": []}
        assert extract_last_revision_date(doc) is None

    def test_no_effective_date(self):
        """Test addenda without effective_date."""
        from src.enhance_for_rag import extract_last_revision_date

        doc = {
            "addenda": [
                {"title": "부칙", "children": []}
            ]
        }
        assert extract_last_revision_date(doc) is None


class TestEnhanceDocumentLastRevisionDate:
    """Tests for last_revision_date field in enhance_document."""

    def test_populates_last_revision_date(self):
        """Test that enhance_document populates last_revision_date in metadata."""
        from src.enhance_for_rag import enhance_document

        doc = {
            "title": "테스트규정",
            "doc_type": "regulation",
            "metadata": {},
            "content": [],
            "addenda": [
                {
                    "type": "addendum_item",
                    "text": "이 규정은 2024년 3월 1일부터 시행한다.",
                    "children": [],
                }
            ],
        }
        enhance_document(doc)
        assert doc["metadata"].get("last_revision_date") == "2024-03-01"

    def test_no_metadata_creates_one(self):
        """Test that metadata dict is created if missing."""
        from src.enhance_for_rag import enhance_document

        doc = {
            "title": "테스트규정",
            "doc_type": "regulation",
            "content": [],
            "addenda": [
                {
                    "type": "addendum_item",
                    "text": "이 규정은 2023년 1월 1일부터 시행한다.",
                    "children": [],
                }
            ],
        }
        enhance_document(doc)
        assert "metadata" in doc
        assert doc["metadata"].get("last_revision_date") == "2023-01-01"

