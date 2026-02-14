"""
Characterization tests for FullViewUseCase.

These tests document the CURRENT behavior of full view retrieval,
not what it SHOULD do. Tests capture actual outputs for regression detection.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Optional, Tuple


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_loader():
    """Create a mock document loader for testing."""
    loader = MagicMock()
    loader.get_all_regulations.return_value = []
    loader.get_regulation_doc.return_value = None
    return loader


@pytest.fixture
def sample_regulation_doc():
    """Create a sample regulation document for testing."""
    return {
        "title": "테스트 규정",
        "metadata": {"rule_code": "TEST-001"},
        "content": [
            {
                "type": "chapter",
                "display_no": "제1장",
                "title": "총칙",
                "children": [
                    {
                        "type": "article",
                        "display_no": "제1조",
                        "title": "목적",
                        "text": "이 규정은 테스트를 목적으로 한다.",
                    }
                ],
            }
        ],
        "addenda": [],
    }


@pytest.fixture
def sample_regulations_list():
    """Create a sample list of regulations."""
    return [
        ("TEST-001", "테스트 규정"),
        ("TEST-002", "다른 규정"),
    ]


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestRegulationMatch:
    """Tests for RegulationMatch dataclass."""

    def test_regulation_match_creation(self):
        """RegulationMatch can be created with required fields."""
        from src.rag.application.full_view_usecase import RegulationMatch

        match = RegulationMatch(title="테스트 규정", rule_code="TEST-001", score=4)
        assert match.title == "테스트 규정"
        assert match.rule_code == "TEST-001"
        assert match.score == 4


class TestRegulationView:
    """Tests for RegulationView dataclass."""

    def test_regulation_view_creation(self):
        """RegulationView can be created with required fields."""
        from src.rag.application.full_view_usecase import RegulationView

        view = RegulationView(
            title="테스트 규정",
            rule_code="TEST-001",
            toc=["제1장 총칙"],
            content=[{"type": "chapter"}],
            addenda=[],
        )
        assert view.title == "테스트 규정"
        assert view.rule_code == "TEST-001"
        assert len(view.toc) == 1


class TestTableMatch:
    """Tests for TableMatch dataclass."""

    def test_table_match_creation(self):
        """TableMatch can be created with required fields."""
        from src.rag.application.full_view_usecase import TableMatch

        match = TableMatch(
            path=["제1장", "제1조"],
            display_no="별표 1",
            title="양식",
            text="설명 텍스트",
            markdown="|컬럼1|컬럼2|",
            table_index=1,
        )
        assert match.path == ["제1장", "제1조"]
        assert match.table_index == 1


# ============================================================================
# FullViewUseCase Initialization Tests
# ============================================================================


class TestFullViewUseCaseInit:
    """Tests for FullViewUseCase initialization."""

    def test_init_with_loader(self, mock_loader):
        """FullViewUseCase initializes with loader."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        assert usecase.loader == mock_loader

    def test_init_with_custom_json_path(self, mock_loader):
        """FullViewUseCase accepts custom JSON path."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader, json_path="/custom/path.json")
        assert usecase.json_path == "/custom/path.json"


# ============================================================================
# find_matches Tests
# ============================================================================


class TestFindMatches:
    """Tests for find_matches method."""

    def test_find_matches_empty_regulations(self, mock_loader):
        """find_matches returns empty list when no regulations."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_all_regulations.return_value = []

        result = usecase.find_matches("테스트")
        assert result == []

    def test_find_matches_exact_title_match(self, mock_loader, sample_regulations_list):
        """find_matches returns high score for exact title match."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_all_regulations.return_value = sample_regulations_list

        result = usecase.find_matches("테스트 규정")
        assert len(result) >= 1
        assert result[0].score == 4  # exact match

    def test_find_matches_partial_match(self, mock_loader, sample_regulations_list):
        """find_matches returns results for partial matches."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_all_regulations.return_value = sample_regulations_list

        result = usecase.find_matches("테스트")
        assert len(result) >= 1

    def test_find_matches_no_match(self, mock_loader, sample_regulations_list):
        """find_matches returns empty list when no match."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_all_regulations.return_value = sample_regulations_list

        result = usecase.find_matches("존재하지않는규정")
        assert result == []

    def test_find_matches_strips_full_view_markers(self, mock_loader, sample_regulations_list):
        """find_matches strips full view markers from query."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_all_regulations.return_value = sample_regulations_list

        # Query with "전문" marker should still find matches
        result = usecase.find_matches("테스트 규정 전문")
        assert len(result) >= 1


# ============================================================================
# get_full_view Tests
# ============================================================================


class TestGetFullView:
    """Tests for get_full_view method."""

    def test_get_full_view_returns_view(self, mock_loader, sample_regulation_doc):
        """get_full_view returns RegulationView for valid identifier."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_regulation_doc.return_value = sample_regulation_doc

        result = usecase.get_full_view("TEST-001")

        assert result is not None
        assert result.title == "테스트 규정"
        assert result.rule_code == "TEST-001"

    def test_get_full_view_not_found(self, mock_loader):
        """get_full_view returns None for non-existent regulation."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_regulation_doc.return_value = None

        result = usecase.get_full_view("NONEXISTENT")
        assert result is None

    def test_get_full_view_builds_toc(self, mock_loader, sample_regulation_doc):
        """get_full_view builds table of contents."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_regulation_doc.return_value = sample_regulation_doc

        result = usecase.get_full_view("TEST-001")

        assert result is not None
        assert isinstance(result.toc, list)


# ============================================================================
# get_article_view Tests
# ============================================================================


class TestGetArticleView:
    """Tests for get_article_view method."""

    def test_get_article_view_finds_article(self, mock_loader, sample_regulation_doc):
        """get_article_view finds article by number."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_regulation_doc.return_value = sample_regulation_doc

        result = usecase.get_article_view("TEST-001", 1)

        assert result is not None
        assert result.get("type") == "article"

    def test_get_article_view_not_found(self, mock_loader, sample_regulation_doc):
        """get_article_view returns None for non-existent article."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_regulation_doc.return_value = sample_regulation_doc

        result = usecase.get_article_view("TEST-001", 999)
        assert result is None

    def test_get_article_view_no_doc(self, mock_loader):
        """get_article_view returns None when doc not found."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_regulation_doc.return_value = None

        result = usecase.get_article_view("TEST-001", 1)
        assert result is None


# ============================================================================
# get_chapter_node Tests
# ============================================================================


class TestGetChapterNode:
    """Tests for get_chapter_node method."""

    def test_get_chapter_node_finds_chapter(self, mock_loader, sample_regulation_doc):
        """get_chapter_node finds chapter by number."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_regulation_doc.return_value = sample_regulation_doc

        result = usecase.get_chapter_node(sample_regulation_doc, 1)

        assert result is not None
        assert result.get("type") == "chapter"

    def test_get_chapter_node_none_doc(self, mock_loader):
        """get_chapter_node returns None for None doc."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)

        result = usecase.get_chapter_node(None, 1)
        assert result is None

    def test_get_chapter_node_not_found(self, mock_loader, sample_regulation_doc):
        """get_chapter_node returns None for non-existent chapter."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)

        result = usecase.get_chapter_node(sample_regulation_doc, 999)
        assert result is None


# ============================================================================
# find_tables Tests
# ============================================================================


class TestFindTables:
    """Tests for find_tables method."""

    def test_find_tables_empty_result(self, mock_loader):
        """find_tables returns empty list when no tables."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_regulation_doc.return_value = {
            "title": "Test",
            "content": [],
            "addenda": [],
        }

        result = usecase.find_tables("TEST-001")
        assert result == []

    def test_find_tables_with_table_no(self, mock_loader):
        """find_tables filters by table number."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_regulation_doc.return_value = {
            "title": "Test",
            "content": [
                {
                    "type": "article",
                    "text": "별표 1 내용",
                    "metadata": {
                        "tables": [{"markdown": "|A|B|"}]
                    },
                }
            ],
            "addenda": [],
        }

        result = usecase.find_tables("TEST-001", table_no=1)
        # Result depends on actual implementation
        assert isinstance(result, list)

    def test_find_tables_no_doc(self, mock_loader):
        """find_tables returns empty list when doc not found."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_regulation_doc.return_value = None

        result = usecase.find_tables("NONEXISTENT")
        assert result == []


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_clean_attachment_title(self):
        """_clean_attachment_title removes brackets."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        result = FullViewUseCase._clean_attachment_title("[별표 1]")
        assert "[" not in result
        assert "]" not in result

    def test_clean_attachment_title_empty(self):
        """_clean_attachment_title handles empty string."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        result = FullViewUseCase._clean_attachment_title("")
        assert result == ""

    def test_extract_attachment_label_with_number(self):
        """_extract_attachment_label extracts label and number."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        label, number = FullViewUseCase._extract_attachment_label("별표 1")
        assert label == "별표"
        assert number == 1

    def test_extract_attachment_label_no_match(self):
        """_extract_attachment_label returns empty for no match."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        label, number = FullViewUseCase._extract_attachment_label("일반 텍스트")
        assert label == ""
        assert number is None


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_full_view_markers_defined(self):
        """FULL_VIEW_MARKERS constant is defined."""
        from src.rag.application.full_view_usecase import FULL_VIEW_MARKERS

        assert "전문" in FULL_VIEW_MARKERS
        assert "전체" in FULL_VIEW_MARKERS

    def test_attachment_markers_defined(self):
        """ATTACHMENT_MARKERS constant is defined."""
        from src.rag.application.full_view_usecase import ATTACHMENT_MARKERS

        assert "별표" in ATTACHMENT_MARKERS

    def test_attachment_labels_defined(self):
        """ATTACHMENT_LABELS constant is defined."""
        from src.rag.application.full_view_usecase import ATTACHMENT_LABELS

        assert "별표" in ATTACHMENT_LABELS
        assert "별지" in ATTACHMENT_LABELS


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_find_matches_with_full_view_marker_in_query(
        self, mock_loader, sample_regulations_list
    ):
        """find_matches handles queries with full view markers."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_all_regulations.return_value = sample_regulations_list

        # Various full view markers
        for marker in ["전문", "전체", "원문", "full text"]:
            result = usecase.find_matches(f"테스트 규정 {marker}")
            assert isinstance(result, list)

    def test_get_full_view_with_empty_content(self, mock_loader):
        """get_full_view handles documents with empty content."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_regulation_doc.return_value = {
            "title": "Empty",
            "metadata": {"rule_code": "EMPTY-001"},
            "content": [],
            "addenda": [],
        }

        result = usecase.get_full_view("EMPTY-001")
        assert result is not None
        assert result.content == []

    def test_find_matches_loader_exception(self, mock_loader):
        """find_matches handles loader exceptions gracefully."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_all_regulations.side_effect = Exception("Loader error")

        result = usecase.find_matches("test")
        assert result == []

    def test_get_full_view_loader_exception(self, mock_loader):
        """get_full_view handles loader exceptions gracefully."""
        from src.rag.application.full_view_usecase import FullViewUseCase

        usecase = FullViewUseCase(loader=mock_loader)
        mock_loader.get_regulation_doc.side_effect = Exception("Loader error")

        result = usecase.get_full_view("TEST-001")
        assert result is None
