"""
Unit tests for formatters module.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.rag.interface.formatters import (
    DEFAULT_RELEVANCE_THRESHOLD,
    build_display_path,
    clean_path_segments,
    extract_display_text,
    filter_by_relevance,
    get_confidence_info,
    get_relevance_label,
    get_relevance_label_combined,
    infer_regulation_title_from_tables,
    normalize_markdown_emphasis,
    normalize_markdown_table,
    normalize_relevance_scores,
    render_full_view_nodes,
    strip_path_prefix,
)


# Mock SearchResult and Chunk for testing
@dataclass
class MockChunk:
    id: str
    text: str = ""
    title: str = ""
    parent_path: Optional[List[str]] = None
    rule_code: str = ""


@dataclass
class MockSearchResult:
    chunk: MockChunk
    score: float


@dataclass
class MockTable:
    path: Optional[List[str]] = None


# ============================================================================
# normalize_relevance_scores tests
# ============================================================================


class TestNormalizeRelevanceScores:
    def test_empty_list(self):
        """Empty list should return empty dict."""
        result = normalize_relevance_scores([])
        assert result == {}

    def test_single_item(self):
        """Single item should return 1.0 (100% relevance)."""
        sources = [MockSearchResult(chunk=MockChunk(id="1"), score=0.5)]
        result = normalize_relevance_scores(sources)
        assert result == {"1": 1.0}

    def test_equal_scores(self):
        """All equal scores should return 1.0 for each."""
        sources = [
            MockSearchResult(chunk=MockChunk(id="1"), score=0.5),
            MockSearchResult(chunk=MockChunk(id="2"), score=0.5),
            MockSearchResult(chunk=MockChunk(id="3"), score=0.5),
        ]
        result = normalize_relevance_scores(sources)
        assert result == {"1": 1.0, "2": 1.0, "3": 1.0}

    def test_multiple_different_scores(self):
        """Different scores should be normalized to 0-1 range."""
        sources = [
            MockSearchResult(chunk=MockChunk(id="1"), score=0.9),  # highest -> 1.0
            MockSearchResult(chunk=MockChunk(id="2"), score=0.5),  # middle -> 0.5
            MockSearchResult(chunk=MockChunk(id="3"), score=0.1),  # lowest -> 0.0
        ]
        result = normalize_relevance_scores(sources)

        assert result["1"] == 1.0
        assert result["3"] == 0.0
        assert 0.4 < result["2"] < 0.6  # approximately 0.5

    def test_preserves_order(self):
        """Higher original scores should have higher normalized scores."""
        sources = [
            MockSearchResult(chunk=MockChunk(id="a"), score=0.2),
            MockSearchResult(chunk=MockChunk(id="b"), score=0.8),
            MockSearchResult(chunk=MockChunk(id="c"), score=0.5),
        ]
        result = normalize_relevance_scores(sources)

        assert result["b"] > result["c"] > result["a"]


# ============================================================================
# filter_by_relevance tests
# ============================================================================


class TestFilterByRelevance:
    def test_empty_list(self):
        """Empty list should return empty list."""
        result = filter_by_relevance([], {})
        assert result == []

    def test_filters_below_threshold(self):
        """Results below threshold should be filtered out."""
        sources = [
            MockSearchResult(chunk=MockChunk(id="1"), score=0.9),
            MockSearchResult(chunk=MockChunk(id="2"), score=0.5),
            MockSearchResult(chunk=MockChunk(id="3"), score=0.1),
        ]
        norm_scores = {"1": 1.0, "2": 0.5, "3": 0.05}  # "3" is below 10%

        result = filter_by_relevance(sources, norm_scores, threshold=0.10)

        assert len(result) == 2
        assert result[0].chunk.id == "1"
        assert result[1].chunk.id == "2"

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        sources = [
            MockSearchResult(chunk=MockChunk(id="1"), score=0.9),
            MockSearchResult(chunk=MockChunk(id="2"), score=0.5),
        ]
        norm_scores = {"1": 1.0, "2": 0.4}

        # 50% threshold should filter out "2"
        result = filter_by_relevance(sources, norm_scores, threshold=0.50)

        assert len(result) == 1
        assert result[0].chunk.id == "1"

    def test_default_threshold(self):
        """Default threshold should be 0.10."""
        assert DEFAULT_RELEVANCE_THRESHOLD == 0.10


# ============================================================================
# get_relevance_label tests
# ============================================================================


class TestGetRelevanceLabel:
    def test_very_high(self):
        """80%+ should be '매우 높음'."""
        icon, label = get_relevance_label(80)
        assert icon == "🟢"
        assert label == "매우 높음"

        icon, label = get_relevance_label(100)
        assert icon == "🟢"
        assert label == "매우 높음"

    def test_high(self):
        """50-79% should be '높음'."""
        icon, label = get_relevance_label(50)
        assert icon == "🟡"
        assert label == "높음"

        icon, label = get_relevance_label(79)
        assert icon == "🟡"
        assert label == "높음"

    def test_medium(self):
        """30-49% should be '보통'."""
        icon, label = get_relevance_label(30)
        assert icon == "🟠"
        assert label == "보통"

        icon, label = get_relevance_label(49)
        assert icon == "🟠"
        assert label == "보통"

    def test_low(self):
        """Below 30% should be '낮음'."""
        icon, label = get_relevance_label(29)
        assert icon == "🔴"
        assert label == "낮음"

        icon, label = get_relevance_label(0)
        assert icon == "🔴"
        assert label == "낮음"

    def test_combined(self):
        """Combined should return 'icon label' format."""
        result = get_relevance_label_combined(85)
        assert result == "🟢 매우 높음"


# ============================================================================
# clean_path_segments tests
# ============================================================================


class TestCleanPathSegments:
    def test_empty_list(self):
        """Empty list should return empty list."""
        assert clean_path_segments([]) == []

    def test_no_duplicates(self):
        """List without duplicates should be unchanged."""
        segments = ["규정명", "제1장", "제1조"]
        assert clean_path_segments(segments) == segments

    def test_removes_whitespace_duplicates(self):
        """Duplicates differing only by whitespace should be removed."""
        segments = ["부칙", "부 칙"]
        result = clean_path_segments(segments)
        assert result == ["부칙"]

    def test_removes_fullwidth_space_duplicates(self):
        """Duplicates with fullwidth spaces should be removed."""
        segments = ["제1조", "제 1 조"]  # with fullwidth spaces
        result = clean_path_segments(segments)
        assert result == ["제1조"]

    def test_preserves_different_segments(self):
        """Different segments should be preserved."""
        segments = ["규정명", "부칙", "부 칙", "제1조"]
        result = clean_path_segments(segments)
        assert result == ["규정명", "부칙", "제1조"]


# ============================================================================
# extract_display_text tests
# ============================================================================


class TestExtractDisplayText:
    def test_with_path_prefix(self):
        """Text with path prefix should have it removed."""
        text = "규정명 > 제1조 > 제1항: 본문 내용입니다."
        result = extract_display_text(text)
        assert result == "본문 내용입니다."

    def test_without_path_prefix(self):
        """Text without path prefix should be unchanged."""
        text = "본문 내용입니다."
        result = extract_display_text(text)
        assert result == "본문 내용입니다."

    def test_cleans_number_colon_format(self):
        """Number followed by colon should be cleaned."""
        text = "1.: 첫 번째 항목"
        result = extract_display_text(text)
        assert "1.:" not in result


# ============================================================================
# render_full_view_nodes tests
# ============================================================================


class TestRenderFullViewNodes:
    def test_inline_paragraph_numbering(self):
        nodes = [
            {
                "type": "paragraph",
                "display_no": "①",
                "title": "",
                "text": "재직 중인 교원중에서 지정할 수 있다.",
                "children": [],
            }
        ]
        rendered = render_full_view_nodes(nodes)
        assert "① 재직 중인 교원중에서 지정할 수 있다." in rendered

    def test_inline_item_numbering(self):
        nodes = [
            {
                "type": "item",
                "display_no": "1.",
                "title": "",
                "text": "교수 : 정",
                "children": [],
            }
        ]
        rendered = render_full_view_nodes(nodes)
        assert "1. 교수 : 정" in rendered

    def test_article_with_title_keeps_heading(self):
        nodes = [
            {
                "type": "article",
                "display_no": "제1조",
                "title": "목적",
                "text": "이 규정은 목적을 규정한다.",
                "children": [],
            }
        ]
        rendered = render_full_view_nodes(nodes)
        assert "### 제1조 목적" in rendered
        assert "이 규정은 목적을 규정한다." in rendered

    def test_article_without_title_inlines_text(self):
        nodes = [
            {
                "type": "article",
                "display_no": "제16조",
                "title": "",
                "text": "내용이 이어진다.",
                "children": [],
            }
        ]
        rendered = render_full_view_nodes(nodes)
        assert "제16조 내용이 이어진다." in rendered

    def test_injects_table_markdown(self):
        nodes = [
            {
                "type": "paragraph",
                "display_no": "①",
                "title": "",
                "text": "기준은 다음과 같다.\n[TABLE:1]",
                "metadata": {
                    "tables": [
                        {
                            "format": "markdown",
                            "markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |",
                        },
                    ]
                },
                "children": [],
            }
        ]
        rendered = render_full_view_nodes(nodes)
        assert "[TABLE:1]" not in rendered
        assert "| A | B |" in rendered

    def test_keeps_unknown_table_placeholder(self):
        nodes = [
            {
                "type": "paragraph",
                "display_no": "①",
                "title": "",
                "text": "기준은 다음과 같다.\n[TABLE:2]",
                "metadata": {
                    "tables": [
                        {"format": "markdown", "markdown": "| A | B |"},
                    ]
                },
                "children": [],
            }
        ]
        rendered = render_full_view_nodes(nodes)
        assert "[TABLE:2]" in rendered


class TestNormalizeMarkdownTable:
    def test_promotes_first_data_row_when_header_blank(self):
        markdown = (
            "|  |  |  |  |\n"
            "| --- | --- | --- | --- |\n"
            "| 직 위 | 근무기간 | 교육업적 점수 | 연구업적 점수 |\n"
            "| 조교수 | 3년 | 480점 이상 | 300점 이상 |\n"
        )
        normalized = normalize_markdown_table(markdown)
        lines = normalized.splitlines()
        assert lines[0] == "| 직 위 | 근무기간 | 교육업적 점수 | 연구업적 점수 |"
        assert "조교수" in normalized

    def test_keeps_existing_header(self):
        markdown = "| 구분 | 값 |\n| --- | --- |\n| A | 1 |\n"
        normalized = normalize_markdown_table(markdown)
        assert normalized.strip() == markdown.strip()


# ============================================================================
# normalize_markdown_emphasis tests
# ============================================================================


class TestNormalizeMarkdownEmphasis:
    def test_moves_double_quotes_outside_bold(self):
        text = '**"교수님이 학교에 가기 싫어하는 상황"**'
        assert (
            normalize_markdown_emphasis(text)
            == '"**교수님이 학교에 가기 싫어하는 상황**"'
        )

    def test_moves_single_quotes_outside_bold(self):
        text = "**'교원인사규정'**"
        assert normalize_markdown_emphasis(text) == "'**교원인사규정**'"

    def test_moves_curly_quotes_outside_bold(self):
        text = "**“교원인사규정”**"
        assert normalize_markdown_emphasis(text) == "“**교원인사규정**”"

    def test_leaves_plain_bold_untouched(self):
        text = "**교원인사규정**"
        assert normalize_markdown_emphasis(text) == text


# ============================================================================
# strip_path_prefix tests
# ============================================================================


class TestStripPathPrefix:
    def test_strips_parent_path_prefix(self):
        text = "교원인사규정 > 부칙 > 부 칙 > 2. 조교수로 재직중인 교원"
        parent_path = ["교원인사규정", "부칙", "부 칙"]
        assert strip_path_prefix(text, parent_path) == "2. 조교수로 재직중인 교원"

    def test_keeps_text_when_no_match(self):
        text = "제1조 목적 이 규정은 목적을 규정한다."
        parent_path = ["교원인사규정"]
        assert strip_path_prefix(text, parent_path) == text


# ============================================================================
# infer_regulation_title_from_tables tests
# ============================================================================


class TestInferRegulationTitleFromTables:
    def test_uses_table_path_title_over_fallback(self):
        tables = [{"path": ["교원인사규정", "부칙"]}]
        assert (
            infer_regulation_title_from_tables(tables, "JA교원인사규정")
            == "교원인사규정"
        )

    def test_uses_object_path_when_present(self):
        tables = [MockTable(path=None), MockTable(path=["교원인사규정", "별첨"])]
        assert (
            infer_regulation_title_from_tables(tables, "JA교원인사규정")
            == "교원인사규정"
        )

    def test_returns_fallback_when_paths_missing(self):
        tables = [MockTable(path=None), {"path": []}]
        assert (
            infer_regulation_title_from_tables(tables, "교원인사규정")
            == "교원인사규정"
        )


# ============================================================================
# get_confidence_info tests
# ============================================================================


class TestGetConfidenceInfo:
    def test_high_confidence(self):
        """70%+ should be '높음'."""
        icon, label, desc = get_confidence_info(0.7)
        assert icon == "🟢"
        assert label == "높음"
        assert "신뢰" in desc

        icon, label, desc = get_confidence_info(1.0)
        assert icon == "🟢"

    def test_medium_confidence(self):
        """40-69% should be '보통'."""
        icon, label, desc = get_confidence_info(0.4)
        assert icon == "🟡"
        assert label == "보통"
        assert "확인" in desc

        icon, label, desc = get_confidence_info(0.69)
        assert icon == "🟡"

    def test_low_confidence(self):
        """Below 40% should be '낮음'."""
        icon, label, desc = get_confidence_info(0.39)
        assert icon == "🔴"
        assert label == "낮음"
        assert "행정실" in desc

        icon, label, desc = get_confidence_info(0.0)
        assert icon == "🔴"


# ============================================================================
# build_display_path tests
# ============================================================================


class TestBuildDisplayPath:
    def test_simple_path(self):
        """Simple parent path should be joined."""
        result = build_display_path(
            chunk_parent_path=["규정명", "제1장", "제1조"],
            chunk_text="내용",
            chunk_title="제목",
        )
        assert result == "규정명 > 제1장 > 제1조"

    def test_empty_parent_path(self):
        """Empty parent path should use title."""
        result = build_display_path(
            chunk_parent_path=[],
            chunk_text="내용",
            chunk_title="제목",
        )
        assert result == "제목"

    def test_removes_duplicates(self):
        """Duplicate segments should be removed."""
        result = build_display_path(
            chunk_parent_path=["규정명", "부칙", "부 칙"],
            chunk_text="내용",
            chunk_title="제목",
        )
        assert result == "규정명 > 부칙"
