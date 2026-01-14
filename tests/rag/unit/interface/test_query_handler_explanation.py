"""
Integration-style tests for QueryHandler search/ask methods with match explanation.

Tests that 매칭 정보 lines appear correctly in QueryHandler output.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.rag.domain.entities import ChunkLevel
from src.rag.interface.query_handler import QueryHandler, QueryOptions, QueryResult, QueryType


# ============================================================================
# Mock Classes
# ============================================================================


@dataclass
class MockKeyword:
    """Mock Keyword entity."""
    term: str
    weight: float = 1.0


@dataclass
class MockChunk:
    """Mock Chunk entity for testing."""
    id: str
    text: str
    title: str = ""
    rule_code: str = "1-1-1"
    parent_path: Optional[List[str]] = None
    keywords: Optional[List[MockKeyword]] = None
    level: Optional[ChunkLevel] = None
    display_no: Optional[str] = None


@dataclass
class MockSearchResult:
    """Mock SearchResult for testing."""
    chunk: MockChunk
    score: float
    rank: int = 1


class FakeVectorStore:
    """Fake vector store that returns predetermined results."""

    def __init__(self, count_value: int = 100):
        self._count = count_value

    def count(self) -> int:
        return self._count


class FakeSearchUseCase:
    """Fake SearchUseCase that returns predetermined results."""

    def __init__(self, results: List[MockSearchResult]):
        self._results = results
        self._last_rewrite = None

    def search_unique(self, query, top_k=5, **kwargs) -> List[MockSearchResult]:
        return self._results[:top_k]

    def get_last_query_rewrite(self):
        return self._last_rewrite


@dataclass
class MockAnswer:
    """Mock LLM answer."""
    text: str
    sources: List[MockSearchResult] = field(default_factory=list)
    confidence: float = 0.8


# ============================================================================
# Test QueryHandler.search() with explanation
# ============================================================================


class TestQueryHandlerSearchExplanation:
    """Test that search() includes 매칭 정보 in output."""

    def test_search_includes_matching_info_line(self):
        """Search result content should include 매칭 정보 line."""
        # Setup mock chunk with keywords
        chunk = MockChunk(
            id="chunk1",
            text="연구년 신청 절차에 대한 내용입니다.",
            title="제15조 연구년",
            rule_code="3-1-5",
            parent_path=["교원인사규정", "제3장 연구년제"],
            keywords=[MockKeyword(term="연구년"), MockKeyword(term="신청")],
            level=ChunkLevel.ARTICLE,
            display_no="제15조",
        )
        mock_result = MockSearchResult(chunk=chunk, score=0.85, rank=1)

        # Create handler with fake store
        store = FakeVectorStore(count_value=100)
        handler = QueryHandler(store=store)

        # Patch SearchUseCase to return our mock results
        with patch.object(handler, 'store', store):
            with patch('src.rag.interface.query_handler.SearchUseCase') as MockSU:
                mock_su_instance = FakeSearchUseCase([mock_result])
                MockSU.return_value = mock_su_instance

                result = handler.search("연구년 신청", options=QueryOptions(top_k=5))

        assert result.success is True
        assert result.type == QueryType.SEARCH
        assert "매칭 정보" in result.content
        # Should include matched keywords
        assert "연구년" in result.content

    def test_search_shows_article_number_from_display_no(self):
        """Search should show article number extracted from display_no."""
        chunk = MockChunk(
            id="chunk1",
            text="휴직에 대한 규정입니다.",
            title="휴직",  # No article number in title
            rule_code="3-1-10",
            parent_path=["교원인사규정"],
            level=ChunkLevel.ARTICLE,
            display_no="제20조",  # Article number in display_no
        )
        mock_result = MockSearchResult(chunk=chunk, score=0.9, rank=1)

        store = FakeVectorStore(count_value=100)
        handler = QueryHandler(store=store)

        with patch.object(handler, 'store', store):
            with patch('src.rag.interface.query_handler.SearchUseCase') as MockSU:
                mock_su_instance = FakeSearchUseCase([mock_result])
                MockSU.return_value = mock_su_instance

                result = handler.search("휴직", options=QueryOptions())

        assert result.success is True
        # 매칭 정보 should include article number from display_no
        assert "제20조" in result.content

    def test_search_with_no_keywords_shows_fallback(self):
        """Search result without keywords should show fallback text."""
        chunk = MockChunk(
            id="chunk1",
            text="일반 내용입니다.",
            title="제1조",
            rule_code="1-1-1",
            parent_path=["학칙"],
            keywords=[],  # No keywords
        )
        mock_result = MockSearchResult(chunk=chunk, score=0.7, rank=1)

        store = FakeVectorStore(count_value=100)
        handler = QueryHandler(store=store)

        with patch.object(handler, 'store', store):
            with patch('src.rag.interface.query_handler.SearchUseCase') as MockSU:
                mock_su_instance = FakeSearchUseCase([mock_result])
                MockSU.return_value = mock_su_instance

                result = handler.search("검색어", options=QueryOptions())

        assert result.success is True
        # Should still have 매칭 정보 line even without keywords
        assert "매칭 정보" in result.content


# ============================================================================
# Test QueryHandler.ask() with explanation
# ============================================================================


class TestQueryHandlerAskExplanation:
    """Test that ask() includes 매칭 정보 in sources."""

    def test_ask_includes_matching_info_in_sources(self):
        """Ask result sources should include 매칭 정보."""
        # Setup mock chunks with keywords
        chunk = MockChunk(
            id="chunk1",
            text="연구년 신청은 매년 3월에 가능합니다.",
            title="제15조 연구년 신청",
            rule_code="3-1-5",
            parent_path=["교원인사규정", "연구년제"],
            keywords=[MockKeyword(term="연구년"), MockKeyword(term="신청")],
            level=ChunkLevel.ARTICLE,
            display_no="제15조",
        )
        mock_result = MockSearchResult(chunk=chunk, score=0.9, rank=1)
        mock_answer = MockAnswer(
            text="연구년 신청은 매년 3월에 가능합니다.",
            sources=[mock_result],
            confidence=0.85,
        )

        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        with patch.object(handler, 'store', store):
            with patch('src.rag.interface.query_handler.SearchUseCase') as MockSU:
                mock_su_instance = MagicMock()
                mock_su_instance.ask.return_value = mock_answer
                mock_su_instance.get_last_query_rewrite.return_value = None
                MockSU.return_value = mock_su_instance

                result = handler.ask("연구년 신청 시기는?", options=QueryOptions())

        assert result.success is True
        assert result.type == QueryType.ASK
        # Sources section should include 매칭 정보
        assert "매칭 정보" in result.content

    def test_ask_shows_matched_keywords_in_source(self):
        """Ask should show matched keywords in source section."""
        chunk = MockChunk(
            id="chunk1",
            text="휴직 신청 절차입니다.",
            title="제10조",
            rule_code="3-1-10",
            parent_path=["교원인사규정"],
            keywords=[MockKeyword(term="휴직"), MockKeyword(term="신청")],
        )
        mock_result = MockSearchResult(chunk=chunk, score=0.85, rank=1)
        mock_answer = MockAnswer(
            text="휴직 신청은 인사팀에 문의하세요.",
            sources=[mock_result],
            confidence=0.8,
        )

        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        with patch.object(handler, 'store', store):
            with patch('src.rag.interface.query_handler.SearchUseCase') as MockSU:
                mock_su_instance = MagicMock()
                mock_su_instance.ask.return_value = mock_answer
                mock_su_instance.get_last_query_rewrite.return_value = None
                MockSU.return_value = mock_su_instance

                result = handler.ask("휴직 신청 방법", options=QueryOptions())

        assert result.success is True
        # Should include matched keyword in content
        assert "휴직" in result.content


# ============================================================================
# Test article number extraction priority
# ============================================================================


class TestArticleNumberExtraction:
    """Test that article numbers are extracted with correct priority."""

    def test_display_no_takes_priority_over_title(self):
        """display_no should be preferred over title for article number."""
        chunk = MockChunk(
            id="chunk1",
            text="내용",
            title="제5조 목적",  # Has article number
            rule_code="1-1-1",
            parent_path=["규정"],
            level=ChunkLevel.ARTICLE,
            display_no="제10조",  # Different article number - should take priority
        )
        mock_result = MockSearchResult(chunk=chunk, score=0.9, rank=1)

        store = FakeVectorStore(count_value=100)
        handler = QueryHandler(store=store)

        with patch.object(handler, 'store', store):
            with patch('src.rag.interface.query_handler.SearchUseCase') as MockSU:
                mock_su_instance = FakeSearchUseCase([mock_result])
                MockSU.return_value = mock_su_instance

                result = handler.search("검색어", options=QueryOptions())

        assert result.success is True
        # Should show 제10조 from display_no, not 제5조 from title
        assert "제10조" in result.content

    def test_title_used_when_display_no_missing(self):
        """title should be used when display_no is not available."""
        chunk = MockChunk(
            id="chunk1",
            text="내용",
            title="제7조 적용범위",
            rule_code="1-1-1",
            parent_path=["규정"],
            level=ChunkLevel.ARTICLE,
            display_no=None,  # No display_no
        )
        mock_result = MockSearchResult(chunk=chunk, score=0.9, rank=1)

        store = FakeVectorStore(count_value=100)
        handler = QueryHandler(store=store)

        with patch.object(handler, 'store', store):
            with patch('src.rag.interface.query_handler.SearchUseCase') as MockSU:
                mock_su_instance = FakeSearchUseCase([mock_result])
                MockSU.return_value = mock_su_instance

                result = handler.search("검색어", options=QueryOptions())

        assert result.success is True
        # Should show 제7조 from title
        assert "제7조" in result.content

    def test_text_used_as_last_resort(self):
        """text should be used when both display_no and title lack article number."""
        chunk = MockChunk(
            id="chunk1",
            text="제12조에 따라 처리됩니다.",  # Article number in text
            title="적용범위",  # No article number
            rule_code="1-1-1",
            parent_path=["규정"],
            level=ChunkLevel.ARTICLE,
            display_no="",  # Empty display_no
        )
        mock_result = MockSearchResult(chunk=chunk, score=0.9, rank=1)

        store = FakeVectorStore(count_value=100)
        handler = QueryHandler(store=store)

        with patch.object(handler, 'store', store):
            with patch('src.rag.interface.query_handler.SearchUseCase') as MockSU:
                mock_su_instance = FakeSearchUseCase([mock_result])
                MockSU.return_value = mock_su_instance

                result = handler.search("검색어", options=QueryOptions())

        assert result.success is True
        # Should show 제12조 from text
        assert "제12조" in result.content
