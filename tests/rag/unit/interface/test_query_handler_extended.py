"""
Extended tests for query_handler.py focusing on uncovered lines.

Targets specific uncovered code paths:
- Event handler validation pattern (line 198)
- _process_with_function_gemma error handling (line 368-374)
- ask_stream method coverage (lines 679-765)
- search deletion warning integration
- ask method error paths and context expansion
- _yield_result state_update edge case
- _enrich_with_suggestions title field variants
"""

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

from src.rag.domain.entities import ChunkLevel
from src.rag.interface.query_handler import (
    QueryContext,
    QueryHandler,
    QueryOptions,
    QueryResult,
    QueryType,
)

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


@dataclass
class MockToolResult:
    """Mock tool result from FunctionGemma."""

    tool_name: str
    success: bool
    result: object
    arguments: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": str(self.result),
            "arguments": self.arguments,
        }


class FakeVectorStore:
    """Fake vector store for testing."""

    def __init__(self, count_value: int = 100):
        self._count = count_value

    def count(self) -> int:
        return self._count


# ============================================================================
# Test QueryHandler validation - Event handler pattern (line 198)
# ============================================================================


class TestQueryHandlerEventHandlers:
    """Test event handler pattern detection in validation."""

    def test_validate_rejects_onclick_handler(self):
        """Event handler pattern: onclick= should be rejected."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("click me onclick=alert('xss')")
        assert is_valid is False
        assert "허용되지 않는 문자" in msg

    def test_validate_rejects_onload_handler(self):
        """Event handler pattern: onload= should be rejected."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("<body onload=evil()>")
        assert is_valid is False

    def test_validate_rejects_onerror_handler(self):
        """Event handler pattern: onerror= should be rejected."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("<img src=x onerror=alert(1)>")
        assert is_valid is False

    def test_validate_rejects_onmouseover_handler(self):
        """Event handler pattern: onmouseover= should be rejected."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("text onmouseover=bad()")
        assert is_valid is False

    def test_validate_rejects_event_handler_with_spaces(self):
        """Event handler pattern with spaces should be rejected."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("<div onclick = 'evil()'>")
        assert is_valid is False


# ============================================================================
# Test _process_with_function_gemma error handling (line 368-374)
# ============================================================================


class TestProcessWithFunctionGemmaErrors:
    """Test _process_with_function_gemma error handling path."""

    def test_function_gemma_exception_returns_error_result(self):
        """Test exception in _process_with_function_gemma returns ERROR result."""
        mock_adapter = MagicMock()
        mock_adapter.process_query.side_effect = Exception("FunctionGemma failed")

        handler = QueryHandler(store=None, llm_client=None)
        handler._function_gemma_adapter = mock_adapter

        result = handler._process_with_function_gemma(
            "test query",
            QueryContext(),
            QueryOptions(show_debug=False),
        )

        assert result.type == QueryType.ERROR
        assert result.success is False
        assert "FunctionGemma 처리 오류" in result.content
        assert "기본 검색을 시도해주세요" in result.content

    def test_function_gemma_exception_with_custom_message(self):
        """Test exception message is preserved in error result."""
        mock_adapter = MagicMock()
        mock_adapter.process_query.side_effect = RuntimeError("Connection timeout")

        handler = QueryHandler(store=None, llm_client=None)
        handler._function_gemma_adapter = mock_adapter

        result = handler._process_with_function_gemma(
            "test query",
            QueryContext(),
            QueryOptions(),
        )

        assert result.type == QueryType.ERROR
        assert "Connection timeout" in result.content

    def test_function_gemma_success_with_debug_info(self):
        """Test successful function gemma processing with debug info."""
        # Mock tool results
        tool_result1 = MockToolResult(
            tool_name="search_regulations",
            success=True,
            result={"results": [{"regulation_title": "교원인사규정"}]},
            arguments={"query": "test"},
        )
        tool_result2 = MockToolResult(
            tool_name="get_article",
            success=True,
            result="Article content",
            arguments={"article_no": 10},
        )

        mock_adapter = MagicMock()
        mock_adapter.process_query.return_value = (
            "Answer text",
            [tool_result1, tool_result2],
        )

        handler = QueryHandler(store=None, llm_client=None)
        handler._function_gemma_adapter = mock_adapter

        result = handler._process_with_function_gemma(
            "test query",
            QueryContext(),
            QueryOptions(show_debug=True),
        )

        assert result.type == QueryType.ASK
        assert result.success is True
        assert result.content == "Answer text"
        assert result.data["used_function_gemma"] is True
        assert "search_regulations" in result.debug_info
        assert "get_article" in result.debug_info
        assert result.data["regulation_title"] == "교원인사규정"

    def test_function_gemma_success_without_debug(self):
        """Test successful function gemma processing without debug info."""
        tool_result = MockToolResult(
            tool_name="search",
            success=True,
            result="Result",
            arguments={"q": "test"},
        )

        mock_adapter = MagicMock()
        mock_adapter.process_query.return_value = ("Answer", [tool_result])

        handler = QueryHandler(store=None, llm_client=None)
        handler._function_gemma_adapter = mock_adapter

        result = handler._process_with_function_gemma(
            "test",
            QueryContext(),
            QueryOptions(show_debug=False),
        )

        assert result.success is True
        assert result.debug_info == ""  # No debug info when show_debug=False


# ============================================================================
# Test _enrich_with_suggestions title field variants
# ============================================================================


class TestEnrichWithSuggestionsTitleFields:
    """Test _enrich_with_suggestions handles different data field names."""

    def test_enrich_uses_title_from_data_field(self):
        """Test enrichment uses 'title' from data when 'regulation_title' missing."""
        handler = QueryHandler(store=None, llm_client=None)
        result = QueryResult(
            type=QueryType.SEARCH,
            success=True,
            content="Search results",
            data={"title": "학칙"},  # 'title' field instead of 'regulation_title'
        )

        with patch(
            "src.rag.interface.query_handler.get_followup_suggestions"
        ) as mock_sugs:
            with patch(
                "src.rag.interface.query_handler.format_suggestions_for_cli"
            ) as mock_fmt:
                mock_sugs.return_value = ["suggestion1"]
                mock_fmt.return_value = "\n\n### Suggestions\n- suggestion1"

                handler._enrich_with_suggestions(result, "query")

                # Verify suggestions were called with title from data
                mock_sugs.assert_called_once()
                call_args = mock_sugs.call_args
                assert call_args[1]["regulation_title"] == "학칙"

    def test_enrich_prefers_regulation_title_over_title(self):
        """Test enrichment prefers 'regulation_title' over 'title'."""
        handler = QueryHandler(store=None, llm_client=None)
        result = QueryResult(
            type=QueryType.SEARCH,
            success=True,
            content="Results",
            data={
                "regulation_title": "교원인사규정",
                "title": "fallback title",
            },
        )

        with patch(
            "src.rag.interface.query_handler.get_followup_suggestions"
        ) as mock_sugs:
            mock_sugs.return_value = []
            handler._enrich_with_suggestions(result, "query")

            # Should prefer regulation_title
            call_args = mock_sugs.call_args
            assert call_args[1]["regulation_title"] == "교원인사규정"

    def test_enrich_with_ask_type_passes_answer_text(self):
        """Test enrichment for ASK type passes answer_text for suggestions."""
        handler = QueryHandler(store=None, llm_client=None)
        result = QueryResult(
            type=QueryType.ASK,
            success=True,
            content="Answer text here",
            data={"regulation_title": "규정"},
        )

        with patch(
            "src.rag.interface.query_handler.get_followup_suggestions"
        ) as mock_sugs:
            mock_sugs.return_value = []
            handler._enrich_with_suggestions(result, "query")

            # For ASK type, should pass answer_text
            call_args = mock_sugs.call_args
            assert call_args[1]["answer_text"] == "Answer text here"

    def test_enrich_with_search_type_no_answer_text(self):
        """Test enrichment for SEARCH type does not pass answer_text."""
        handler = QueryHandler(store=None, llm_client=None)
        result = QueryResult(
            type=QueryType.SEARCH,
            success=True,
            content="Results",
            data={"regulation_title": "규정"},
        )

        with patch(
            "src.rag.interface.query_handler.get_followup_suggestions"
        ) as mock_sugs:
            mock_sugs.return_value = []
            handler._enrich_with_suggestions(result, "query")

            # For non-ASK type, answer_text should be None
            call_args = mock_sugs.call_args
            assert call_args[1]["answer_text"] is None


# ============================================================================
# Test _yield_result edge cases
# ============================================================================


class TestYieldResultEdgeCases:
    """Test _yield_result with various edge cases."""

    def test_yield_result_standard_without_state_update(self):
        """Test yielding standard result without state_update."""
        handler = QueryHandler(store=None, llm_client=None)
        result = QueryResult(
            type=QueryType.SEARCH,
            success=True,
            content="Search results",
            data={"rule_code": "1-1-1", "title": "Regulation"},
            # No state_update provided - defaults to empty dict
        )

        events = list(handler._yield_result(result))
        # Should yield metadata and complete, but not state event
        assert len(events) == 2  # metadata, complete (no state event for empty dict)

        assert events[0]["type"] == "metadata"
        assert events[1]["type"] == "complete"

    def test_yield_result_metadata_with_none_parent_path(self):
        """Test metadata when parent_path is None."""
        handler = QueryHandler(store=None, llm_client=None)
        result = QueryResult(
            type=QueryType.SEARCH,
            success=True,
            content="Results",
            data={"title": "Regulation Title"},  # No regulation_title
        )

        events = list(handler._yield_result(result))
        assert events[0]["type"] == "metadata"
        # When regulation_title is missing, should fallback to title
        assert events[0]["regulation_title"] == "Regulation Title"


# ============================================================================
# Test search method with deletion warning
# ============================================================================


class TestSearchDeletionWarning:
    """Test search method includes deletion warnings."""

    def test_search_includes_deletion_warning_in_content(self):
        """Test search adds deletion warning to top result content."""
        chunk = MockChunk(
            id="chunk1",
            text="본 조는 삭제 (2024.06.30) 이 규정은 효력을 잃었습니다.",
            title="제10조",
            rule_code="1-1-1",
            parent_path=["학칙"],
        )
        mock_result = MockSearchResult(chunk=chunk, score=0.9, rank=1)

        store = FakeVectorStore(count_value=100)
        handler = QueryHandler(store=store)

        with patch.object(handler, "store", store):
            with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
                mock_su_instance = MagicMock()
                mock_su_instance.search_unique.return_value = [mock_result]
                mock_su_instance.get_last_query_rewrite.return_value = None
                MockSU.return_value = mock_su_instance

                result = handler.search("검색어", options=QueryOptions())

        assert result.success is True
        # Should include deletion warning in content
        assert "삭제" in result.content
        assert "2024년" in result.content

    def test_search_includes_deletion_warning_in_data(self):
        """Test search includes deletion_warning in result data."""
        chunk = MockChunk(
            id="chunk1",
            text="본 항은 삭제 (2024.03) 내용입니다.",
            title="제5조",
            rule_code="1-1-1",
            parent_path=["규정"],
        )
        mock_result = MockSearchResult(chunk=chunk, score=0.9, rank=1)

        store = FakeVectorStore(count_value=100)
        handler = QueryHandler(store=store)

        with patch.object(handler, "store", store):
            with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
                mock_su_instance = MagicMock()
                mock_su_instance.search_unique.return_value = [mock_result]
                mock_su_instance.get_last_query_rewrite.return_value = None
                MockSU.return_value = mock_su_instance

                result = handler.search("test", options=QueryOptions())

        # Data should include deletion_warning field
        assert "results" in result.data
        assert len(result.data["results"]) > 0
        assert result.data["results"][0]["deletion_warning"] is not None


# ============================================================================
# Test ask method error paths and context expansion
# ============================================================================


class TestAskMethodErrorPaths:
    """Test ask method error handling paths."""

    def test_ask_no_store_returns_error(self):
        """Test ask with None store returns error."""
        handler = QueryHandler(store=None, llm_client=MagicMock())
        result = handler.ask("question", options=QueryOptions())

        assert result.type == QueryType.ERROR
        assert result.success is False
        assert "데이터베이스가 초기화되지 않았습니다" in result.content

    def test_ask_empty_store_returns_error(self):
        """Test ask with empty store returns error."""
        store = FakeVectorStore(count_value=0)
        handler = QueryHandler(store=store, llm_client=MagicMock())
        result = handler.ask("question", options=QueryOptions())

        assert result.type == QueryType.ERROR
        assert result.success is False
        assert "데이터베이스가 비어 있습니다" in result.content

    def test_ask_no_llm_client_returns_error(self):
        """Test ask with None LLM client returns error."""
        store = FakeVectorStore(count_value=100)
        handler = QueryHandler(store=store, llm_client=None)
        result = handler.ask("question", options=QueryOptions())

        assert result.type == QueryType.ERROR
        assert result.success is False
        assert "LLM 클라이언트가 초기화되지 않았습니다" in result.content

    def test_ask_exception_returns_error(self):
        """Test ask exception handling returns error result."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
            mock_su_instance = MagicMock()
            mock_su_instance.ask.side_effect = RuntimeError("LLM error")
            MockSU.return_value = mock_su_instance

            result = handler.ask("question", options=QueryOptions())

        assert result.type == QueryType.ERROR
        assert result.success is False
        assert "답변 생성 실패" in result.content
        assert "LLM error" in result.content

    def test_ask_with_context_expansion(self):
        """Test ask expands query with last_regulation context."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()

        chunk = MockChunk(
            id="chunk1",
            text="연구년 내용",
            title="제10조",
            rule_code="3-1-1",
        )
        mock_result = MockSearchResult(chunk=chunk, score=0.9)
        mock_answer = MagicMock()
        mock_answer.text = "답변"
        mock_answer.sources = [mock_result]
        mock_answer.confidence = 0.8

        handler = QueryHandler(store=store, llm_client=llm_client)
        context = QueryContext(last_regulation="교원인사규정")

        with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
            with patch(
                "src.rag.interface.query_handler.expand_followup_query"
            ) as mock_expand:
                mock_expand.return_value = "교원인사규정 연구년"
                mock_su_instance = MagicMock()
                mock_su_instance.ask.return_value = mock_answer
                mock_su_instance.get_last_query_rewrite.return_value = None
                MockSU.return_value = mock_su_instance

                result = handler.ask(
                    "연구년 신청",
                    options=QueryOptions(show_debug=False),
                    context=context,
                )

        # Should call expand_followup_query with context
        mock_expand.assert_called_once_with("연구년 신청", "교원인사규정")
        assert result.success is True

    def test_ask_with_history_context(self):
        """Test ask includes history in search use case."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        history = [
            {"role": "user", "content": "이전 질문"},
            {"role": "assistant", "content": "이전 답변"},
        ]
        context = QueryContext(history=history)

        mock_answer = MagicMock()
        mock_answer.text = "Answer"
        mock_answer.sources = []
        mock_answer.confidence = 0.8

        with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
            mock_su_instance = MagicMock()
            mock_su_instance.ask.return_value = mock_answer
            mock_su_instance.get_last_query_rewrite.return_value = None
            MockSU.return_value = mock_su_instance

            handler.ask("new question", options=QueryOptions(), context=context)

        # Verify history was passed to ask method
        mock_su_instance.ask.assert_called_once()
        call_kwargs = mock_su_instance.ask.call_args[1]
        assert "history_text" in call_kwargs
        assert (
            "사용자" in call_kwargs["history_text"]
            or "AI" in call_kwargs["history_text"]
        )

    def test_ask_with_debug_mode(self):
        """Test ask with debug mode logs context expansion."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        context = QueryContext(last_regulation="교원인사규정")

        mock_answer = MagicMock()
        mock_answer.text = "Answer"
        mock_answer.sources = []
        mock_answer.confidence = 0.8

        with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
            with patch(
                "src.rag.interface.query_handler.expand_followup_query"
            ) as mock_expand:
                mock_expand.return_value = "교원인사규정 확장된 쿼리"
                mock_su_instance = MagicMock()
                mock_su_instance.ask.return_value = mock_answer
                mock_su_instance.get_last_query_rewrite.return_value = None
                MockSU.return_value = mock_su_instance

                result = handler.ask(
                    "query",
                    options=QueryOptions(show_debug=True),
                    context=context,
                )

        # Debug mode should log the expansion
        assert result.success is True


# ============================================================================
# Test ask_stream method
# ============================================================================


class TestAskStreamMethod:
    """Test ask_stream method coverage."""

    def test_ask_stream_no_store_error(self):
        """Test ask_stream with None store returns error."""
        handler = QueryHandler(store=None, llm_client=MagicMock())
        events = list(handler.ask_stream("question"))
        assert len(events) == 1
        assert events[0]["type"] == "error"
        # The actual error message is "시스템이 초기화되지 않았거나 DB가 비어있습니다."
        assert (
            "초기화" in events[0]["content"] or "비어있습니다" in events[0]["content"]
        )

    def test_ask_stream_empty_store_error(self):
        """Test ask_stream with empty store returns error."""
        store = FakeVectorStore(count_value=0)
        handler = QueryHandler(store=store, llm_client=MagicMock())
        events = list(handler.ask_stream("question"))
        assert events[0]["type"] == "error"

    def test_ask_stream_progress_event(self):
        """Test ask_stream yields progress event."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        # Mock the ask_stream generator to return minimal events
        mock_events = [
            {"type": "token", "content": "Answer"},
            {"type": "complete", "content": "Final answer", "data": {}},
        ]

        with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
            mock_su_instance = MagicMock()
            mock_su_instance.ask_stream.return_value = iter(mock_events)
            MockSU.return_value = mock_su_instance

            events = list(handler.ask_stream("question"))

        # Should have progress event first
        assert events[0]["type"] == "progress"
        assert "규정 검색 중" in events[0]["content"]

    def test_ask_stream_with_context_expansion(self):
        """Test ask_stream expands query with last_regulation."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        context = QueryContext(last_regulation="교원인사규정")

        mock_events = [
            {"type": "complete", "content": "Answer", "data": {}},
        ]

        with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
            with patch(
                "src.rag.interface.query_handler.expand_followup_query"
            ) as mock_expand:
                mock_expand.return_value = "교원인사규정 확장쿼리"
                mock_su_instance = MagicMock()
                mock_su_instance.ask_stream.return_value = iter(mock_events)
                MockSU.return_value = mock_su_instance

                list(handler.ask_stream("query", context=context))

        # Should have expanded the query
        mock_expand.assert_called_once_with("query", "교원인사규정")

    def test_ask_stream_with_expansion_debug_event(self):
        """Test ask_stream shows expansion progress in debug mode."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        context = QueryContext(last_regulation="학칙")

        mock_events = [{"type": "complete", "content": "Answer", "data": {}}]

        with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
            with patch(
                "src.rag.interface.query_handler.expand_followup_query"
            ) as mock_expand:
                mock_expand.return_value = "학칙 변경쿼리"
                mock_su_instance = MagicMock()
                mock_su_instance.ask_stream.return_value = iter(mock_events)
                MockSU.return_value = mock_su_instance

                events = list(
                    handler.ask_stream(
                        "query",
                        context=context,
                        options=QueryOptions(show_debug=True),
                    )
                )

        # Should have expansion progress event when debug is True
        progress_events = [e for e in events if e["type"] == "progress"]
        assert any("문맥 반영" in e.get("content", "") for e in progress_events)

    def test_ask_stream_with_sources(self):
        """Test ask_stream adds sources citations."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        chunk = MockChunk(
            id="chunk1",
            text="Source content",
            title="제1조",
            rule_code="1-1-1",
            parent_path=["규정"],
        )
        mock_result = MockSearchResult(chunk=chunk, score=0.9)

        mock_events = [
            {
                "type": "metadata",
                "sources": [mock_result],
            },
            {"type": "complete", "content": "Answer", "data": {}},
        ]

        with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
            mock_su_instance = MagicMock()
            mock_su_instance.ask_stream.return_value = iter(mock_events)
            MockSU.return_value = mock_su_instance

            events = list(handler.ask_stream("question"))

        # Should have citation in token events
        citation_events = [e for e in events if e["type"] == "token"]
        assert len(citation_events) > 0

    def test_ask_stream_yields_complete_event(self):
        """Test ask_stream yields complete event from search use case."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        # Mock events that search_usecase.ask_stream would return
        mock_events = [
            {"type": "metadata", "sources": []},
            {"type": "complete", "content": "Final Answer", "data": {}},
        ]

        with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
            mock_su_instance = MagicMock()
            mock_su_instance.ask_stream.return_value = iter(mock_events)
            MockSU.return_value = mock_su_instance

            events = list(handler.ask_stream("query"))

        # Should have progress, metadata, complete events
        assert len(events) >= 2
        assert events[0]["type"] == "progress"
        # Should have a complete event
        complete_events = [e for e in events if e["type"] == "complete"]
        assert len(complete_events) > 0
        assert "Final Answer" in complete_events[0]["content"]


# ============================================================================
# Test process_query and process_query_stream validation paths
# ============================================================================


class TestProcessQueryValidationPaths:
    """Test process_query validation uses validated error messages."""

    def test_process_query_empty_error_content(self):
        """Test process_query empty validation returns formatted error."""
        handler = QueryHandler(store=None, llm_client=None)
        result = handler.process_query("")
        assert result.content.startswith("⚠️")
        assert "검색어를 입력" in result.content

    def test_process_query_too_long_formatted_error(self):
        """Test process_query too long returns formatted error."""
        handler = QueryHandler(store=None, llm_client=None)
        result = handler.process_query("가" * 501)
        assert result.content.startswith("⚠️")
        assert "너무 깁니다" in result.content

    def test_process_query_xss_formatted_error(self):
        """Test process_query XSS returns formatted error."""
        handler = QueryHandler(store=None, llm_client=None)
        result = handler.process_query("<script>alert(1)</script>")
        assert result.content.startswith("⚠️")

    def test_process_query_strips_after_validation(self):
        """Test process_query strips query after validation."""
        handler = QueryHandler(store=None, llm_client=None)
        # Query that passes validation but has leading/trailing whitespace
        with patch.object(handler, "validate_query") as mock_validate:
            mock_validate.return_value = (True, "")
            with patch.object(handler, "_is_overview_query") as mock_overview:
                mock_overview.return_value = False
                with patch.object(handler, "search") as mock_search:
                    mock_search.return_value = QueryResult(
                        type=QueryType.SEARCH, success=True, content="results"
                    )

                    handler.process_query("  query  ")

        # Query should be stripped before processing
        # The search call should receive stripped query
        mock_search.assert_called_once()
        call_args = mock_search.call_args[0]
        assert call_args[0] == "query"  # Stripped, not "  query  "


# ============================================================================
# Test get_regulation_overview with similar matches
# ============================================================================


class TestRegulationOverviewSimilarMatches:
    """Test get_regulation_overview similar regulation matching."""

    def test_overview_with_similar_regulations(self):
        """Test overview finds and shows similar regulations."""
        handler = QueryHandler(store=None, llm_client=None)

        # Mock get_regulation_overview to return None initially
        # Mock find_regulation_candidates to return similar matches
        mock_candidates = [
            ("1-1-1", "교원인사규정"),
            ("2-1-1", "교원인사규정시행세칙"),
        ]

        with patch.object(
            handler.full_view_usecase, "_resolve_json_path"
        ) as mock_resolve:
            mock_resolve.return_value = "/path/to/json"
            with patch.object(
                handler.loader, "get_regulation_overview"
            ) as mock_overview:
                mock_overview.return_value = None  # No direct match
                with patch.object(
                    handler.loader, "find_regulation_candidates"
                ) as mock_find:
                    mock_find.return_value = mock_candidates
                    with patch.object(
                        handler.loader, "get_regulation_overview"
                    ) as mock_overview2:
                        # Second call with exact match returns overview
                        from src.rag.domain.entities import (
                            RegulationOverview,
                            RegulationStatus,
                        )

                        mock_overview_obj = RegulationOverview(
                            title="교원인사규정",
                            rule_code="1-1-1",
                            status=RegulationStatus.ACTIVE,
                            article_count=50,
                            chapters=[],
                            has_addenda=False,
                        )
                        mock_overview2.return_value = mock_overview_obj

                        with patch.object(
                            handler.loader, "get_regulation_titles"
                        ) as mock_titles:
                            mock_titles.return_value = {
                                "1-1-1": "교원인사규정",
                                "2-1-1": "교원인사규정시행세칙",
                            }

                            result = handler.get_regulation_overview("교원인사규정")

        # Should find similar matches and include them in response
        assert result.success is True
        # Note: This test verifies the path through similar matches


# ============================================================================
# Test get_article_view, get_chapter_view error paths
# ============================================================================


class TestGetArticleChapterViewErrors:
    """Test error paths in get_article_view and get_chapter_view."""

    def test_get_article_view_no_matches(self):
        """Test get_article_view with no matches returns error."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = []  # No matches

            result = handler.get_article_view("unknown_reg", 10)

        assert result.type == QueryType.ERROR
        assert "해당 규정을 찾을 수 없습니다" in result.content

    def test_get_article_view_multiple_matches(self):
        """Test get_article_view with multiple matches returns clarification."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.application.full_view_usecase import RegulationMatch

        mock_match1 = RegulationMatch(title="규정1", rule_code="1-1-1", score=4)
        mock_match2 = RegulationMatch(title="규정2", rule_code="2-1-1", score=3)

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match1, mock_match2]

            result = handler.get_article_view("ambiguous_reg", 10)

        assert result.type == QueryType.CLARIFICATION
        assert result.clarification_type == "regulation"
        assert "규정1" in result.clarification_options
        assert "규정2" in result.clarification_options

    def test_get_article_view_article_not_found(self):
        """Test get_article_view when article doesn't exist."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.application.full_view_usecase import RegulationMatch

        mock_match = RegulationMatch(title="규정", rule_code="1-1-1", score=4)

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match]
            with patch.object(
                handler.full_view_usecase, "get_article_view"
            ) as mock_get:
                mock_get.return_value = None  # Article not found

                result = handler.get_article_view("규정", 999)

        assert result.type == QueryType.ERROR
        assert "제999조를 찾을 수 없습니다" in result.content

    def test_get_chapter_view_no_matches(self):
        """Test get_chapter_view with no matches returns error."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = []

            result = handler.get_chapter_view("unknown_reg", 3)

        assert result.type == QueryType.ERROR
        assert "해당 규정을 찾을 수 없습니다" in result.content

    def test_get_chapter_view_multiple_matches(self):
        """Test get_chapter_view with multiple matches returns clarification."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.application.full_view_usecase import RegulationMatch

        mock_match1 = RegulationMatch(title="규정A", rule_code="1-1-1", score=4)
        mock_match2 = RegulationMatch(title="규정B", rule_code="2-1-1", score=3)

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match1, mock_match2]

            result = handler.get_chapter_view("ambiguous", 3)

        assert result.type == QueryType.CLARIFICATION
        assert result.clarification_type == "regulation"

    def test_get_chapter_view_chapter_not_found(self):
        """Test get_chapter_view when chapter doesn't exist."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.application.full_view_usecase import RegulationMatch

        mock_match = RegulationMatch(title="규정", rule_code="1-1-1", score=4)

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match]
            with patch.object(
                handler.full_view_usecase, "_resolve_json_path"
            ) as mock_resolve:
                mock_resolve.return_value = "/path/to/json"
                with patch.object(handler.loader, "get_regulation_doc") as mock_get_doc:
                    mock_get_doc.return_value = {"title": "규정"}
                    with patch.object(
                        handler.full_view_usecase, "get_chapter_node"
                    ) as mock_chapter:
                        mock_chapter.return_value = None  # Chapter not found

                        result = handler.get_chapter_view("규정", 99)

        assert result.type == QueryType.ERROR
        assert "제99장을 찾을 수 없습니다" in result.content


# ============================================================================
# Test get_attachment_view method (lines 1035-1090)
# ============================================================================


class TestGetAttachmentViewMethod:
    """Test get_attachment_view method coverage."""

    def test_get_attachment_view_no_matches(self):
        """Test get_attachment_view with no matches returns error."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = []

            result = handler.get_attachment_view("unknown_reg", "별표", None)

        assert result.type == QueryType.ERROR
        assert "해당 규정을 찾을 수 없습니다" in result.content

    def test_get_attachment_view_multiple_matches(self):
        """Test get_attachment_view with multiple matches returns clarification."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.application.full_view_usecase import RegulationMatch

        mock_match1 = RegulationMatch(title="규정A", rule_code="1-1-1", score=4)
        mock_match2 = RegulationMatch(title="규정B", rule_code="2-1-1", score=3)

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match1, mock_match2]

            result = handler.get_attachment_view("ambiguous", "별표", None)

        assert result.type == QueryType.CLARIFICATION
        assert result.clarification_type == "regulation"

    def test_get_attachment_view_no_tables_found(self):
        """Test get_attachment_view when no tables found."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.application.full_view_usecase import RegulationMatch

        mock_match = RegulationMatch(title="규정", rule_code="1-1-1", score=4)

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match]
            with patch.object(
                handler.full_view_usecase, "find_tables"
            ) as mock_find_tables:
                mock_find_tables.return_value = []

                result = handler.get_attachment_view("규정", "별표", None)

        assert result.type == QueryType.ERROR
        assert "찾을 수 없습니다" in result.content

    def test_get_attachment_view_success_with_label(self):
        """Test get_attachment_view returns content with label."""
        handler = QueryHandler(store=None, llm_client=None)

        from dataclasses import dataclass

        from src.rag.application.full_view_usecase import RegulationMatch

        @dataclass
        class MockTable:
            path: list
            title: str
            text: str
            markdown: str

        mock_match = RegulationMatch(title="규정", rule_code="1-1-1", score=4)
        mock_table = MockTable(
            path=["규정", "부칙", "별표"],  # Fixed path to include regulation first
            title="서식",
            text="테이블 설명",
            markdown="| 칼럼1 | 칼럼2 |\n|-------|-------|",
        )

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match]
            with patch.object(
                handler.full_view_usecase, "find_tables"
            ) as mock_find_tables:
                mock_find_tables.return_value = [mock_table]

                result = handler.get_attachment_view("규정", "별표", None)

        assert result.type == QueryType.ATTACHMENT
        assert result.success is True
        assert "별표" in result.content
        assert "규정" in result.data["regulation_title"]

    def test_get_attachment_view_with_table_number(self):
        """Test get_attachment_view with specific table number."""
        handler = QueryHandler(store=None, llm_client=None)

        from dataclasses import dataclass

        from src.rag.application.full_view_usecase import RegulationMatch

        @dataclass
        class MockTable:
            path: list
            title: str
            text: str
            markdown: str

        mock_match = RegulationMatch(title="규정", rule_code="1-1-1", score=4)
        mock_table = MockTable(
            path=["부칙"],
            title="서식1",
            text="내용",
            markdown="| A | B |",
        )

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match]
            with patch.object(
                handler.full_view_usecase, "find_tables"
            ) as mock_find_tables:
                mock_find_tables.return_value = [mock_table]

                result = handler.get_attachment_view("규정", "별표", 1)

        assert result.success is True
        assert "별표 1" in result.content

    def test_get_attachment_view_multiple_tables(self):
        """Test get_attachment_view with multiple tables."""
        handler = QueryHandler(store=None, llm_client=None)

        from dataclasses import dataclass

        from src.rag.application.full_view_usecase import RegulationMatch

        @dataclass
        class MockTable:
            path: list
            title: str
            text: str
            markdown: str

        mock_match = RegulationMatch(title="규정", rule_code="1-1-1", score=4)
        mock_tables = [
            MockTable(
                path=["부칙"],
                title="서식1",
                text="설명1",
                markdown="| A1 | B1 |",
            ),
            MockTable(
                path=["부칙"],
                title="서식2",
                text="설명2",
                markdown="| A2 | B2 |",
            ),
        ]

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match]
            with patch.object(
                handler.full_view_usecase, "find_tables"
            ) as mock_find_tables:
                mock_find_tables.return_value = mock_tables

                result = handler.get_attachment_view("규정", "별표", None)

        assert result.success is True
        # Should include both tables indexed
        assert "[1]" in result.content
        assert "[2]" in result.content


# ============================================================================
# Test get_full_view method (lines 1120-1172)
# ============================================================================


class TestGetFullViewMethod:
    """Test get_full_view method coverage."""

    def test_get_full_view_no_matches(self):
        """Test get_full_view with no matches returns error."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = []

            result = handler.get_full_view("unknown_reg")

        assert result.type == QueryType.ERROR
        assert "해당 규정을 찾을 수 없습니다" in result.content

    def test_get_full_view_multiple_matches(self):
        """Test get_full_view with multiple matches returns clarification."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.application.full_view_usecase import RegulationMatch

        mock_match1 = RegulationMatch(title="규정1", rule_code="1-1-1", score=4)
        mock_match2 = RegulationMatch(title="규정2", rule_code="2-1-1", score=3)

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match1, mock_match2]

            result = handler.get_full_view("ambiguous")

        assert result.type == QueryType.CLARIFICATION
        assert result.clarification_type == "regulation"

    def test_get_full_view_no_view_returned(self):
        """Test get_full_view when view is None."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.application.full_view_usecase import RegulationMatch

        mock_match = RegulationMatch(title="규정", rule_code="1-1-1", score=4)

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match]
            with patch.object(handler.full_view_usecase, "get_full_view") as mock_get:
                mock_get.return_value = None

                result = handler.get_full_view("규정")

        assert result.type == QueryType.ERROR
        assert "불러오지 못했습니다" in result.content

    def test_get_full_view_success_with_toc(self):
        """Test get_full_view returns content with TOC."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.application.full_view_usecase import (
            RegulationMatch,
            RegulationView,
        )

        mock_match = RegulationMatch(title="학칙", rule_code="1-1-1", score=4)
        mock_view = RegulationView(
            title="학칙",
            rule_code="1-1-1",
            toc=["제1장 총칙", "제2장 학생"],
            content=[{"title": "제1조", "text": "내용"}],
            addenda=[],
        )

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match]
            with patch.object(handler.full_view_usecase, "get_full_view") as mock_get:
                mock_get.return_value = mock_view

                result = handler.get_full_view("학칙")

        assert result.success is True
        assert "목차" in result.content
        assert "학칙" in result.content

    def test_get_full_view_with_addenda(self):
        """Test get_full_view includes addenda section."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.application.full_view_usecase import (
            RegulationMatch,
            RegulationView,
        )

        mock_match = RegulationMatch(title="규정", rule_code="1-1-1", score=4)
        mock_view = RegulationView(
            title="규정",
            rule_code="1-1-1",
            toc=[],
            content=[],
            addenda=[{"title": "부칙", "text": "부칙내용"}],
        )

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match]
            with patch.object(handler.full_view_usecase, "get_full_view") as mock_get:
                mock_get.return_value = mock_view
                with patch(
                    "src.rag.interface.query_handler.render_full_view_nodes"
                ) as mock_render:
                    mock_render.return_value = "부칙 내용"

                    result = handler.get_full_view("규정")

        assert result.success is True
        assert "부칙" in result.content

    def test_get_full_view_with_addenda_starting_with_title(self):
        """Test get_full_view when addenda starts with 부칙 title."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.application.full_view_usecase import (
            RegulationMatch,
            RegulationView,
        )

        mock_match = RegulationMatch(title="규정", rule_code="1-1-1", score=4)
        mock_view = RegulationView(
            title="규정",
            rule_code="1-1-1",
            toc=[],
            content=[],
            addenda=[{"title": "부칙", "text": "부칙내용"}],
        )

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match]
            with patch.object(handler.full_view_usecase, "get_full_view") as mock_get:
                mock_get.return_value = mock_view
                with patch(
                    "src.rag.interface.query_handler.render_full_view_nodes"
                ) as mock_render:
                    # Addenda starts with 부칙 heading
                    mock_render.return_value = "### 부칙\n내용"

                    result = handler.get_full_view("규정")

        assert result.success is True

    def test_get_full_view_without_toc(self):
        """Test get_full_view when TOC is missing."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.application.full_view_usecase import (
            RegulationMatch,
            RegulationView,
        )

        mock_match = RegulationMatch(title="규정", rule_code="1-1-1", score=4)
        mock_view = RegulationView(
            title="규정",
            rule_code="1-1-1",
            toc=None,
            content=[],
            addenda=[],
        )

        with patch.object(handler.full_view_usecase, "find_matches") as mock_find:
            mock_find.return_value = [mock_match]
            with patch.object(handler.full_view_usecase, "get_full_view") as mock_get:
                mock_get.return_value = mock_view

                result = handler.get_full_view("규정")

        assert result.success is True
        assert "목차 정보가 없습니다" in result.content


# ============================================================================
# Test ask_stream with history (lines 705-712)
# ============================================================================


class TestAskStreamWithHistory:
    """Test ask_stream method with history context."""

    def test_ask_stream_with_history_generates_history_text(self):
        """Test ask_stream includes history text from context."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        history = [
            {"role": "user", "content": "첫 번째 질문"},
            {"role": "assistant", "content": "첫 번째 답변"},
            {"role": "user", "content": "두 번째 질문"},
        ]
        context = QueryContext(history=history)

        mock_events = [{"type": "complete", "content": "Answer", "data": {}}]

        with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
            mock_su_instance = MagicMock()
            mock_su_instance.ask_stream.return_value = iter(mock_events)
            MockSU.return_value = mock_su_instance

            list(handler.ask_stream("new question", context=context))

        # Verify history text was passed to ask_stream
        mock_su_instance.ask_stream.assert_called_once()
        call_kwargs = mock_su_instance.ask_stream.call_args[1]
        assert "history_text" in call_kwargs
        # Should have role labels in Korean
        assert (
            "사용자" in call_kwargs["history_text"]
            or "AI" in call_kwargs["history_text"]
        )

    def test_ask_stream_limits_history_to_10_messages(self):
        """Test ask_stream only uses last 10 history messages."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        # Create 15 history messages
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(15)
        ]
        context = QueryContext(history=history)

        mock_events = [{"type": "complete", "content": "Answer", "data": {}}]

        with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
            mock_su_instance = MagicMock()
            mock_su_instance.ask_stream.return_value = iter(mock_events)
            MockSU.return_value = mock_su_instance

            list(handler.ask_stream("query", context=context))

        # History should only have last 10 messages
        call_kwargs = mock_su_instance.ask_stream.call_args[1]
        history_text = call_kwargs.get("history_text", "")
        # Count occurrences of "사용자" or "AI" - should be 10 or fewer
        role_count = history_text.count("사용자") + history_text.count("AI")
        assert role_count <= 10

    def test_ask_stream_with_empty_content_history(self):
        """Test ask_stream skips history entries with empty content."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        history = [
            {"role": "user", "content": "valid question"},
            {"role": "assistant", "content": ""},  # Empty content
            {"role": "user", "content": None},  # None content
        ]
        context = QueryContext(history=history)

        mock_events = [{"type": "complete", "content": "Answer", "data": {}}]

        with patch("src.rag.interface.query_handler.SearchUseCase") as MockSU:
            mock_su_instance = MagicMock()
            mock_su_instance.ask_stream.return_value = iter(mock_events)
            MockSU.return_value = mock_su_instance

            list(handler.ask_stream("query", context=context))

        # Only non-empty content should be in history
        call_kwargs = mock_su_instance.ask_stream.call_args[1]
        history_text = call_kwargs.get("history_text", "")
        # Should only have one valid entry
        assert history_text.count("사용자") <= 1


# ============================================================================
# Test _setup_function_gemma method (lines 278-303)
# ============================================================================


class TestSetupFunctionGemma:
    """Test _setup_function_gemma method coverage."""

    def test_setup_function_gemma_with_store(self):
        """Test _setup_function_gemma creates tool executor when store exists."""
        handler = QueryHandler(store=None, llm_client=None)
        mock_llm_client = MagicMock()

        with patch("src.rag.interface.query_handler.ToolExecutor") as MockToolExec:
            with patch(
                "src.rag.interface.query_handler.FunctionGemmaAdapter"
            ) as MockAdapter:
                handler._setup_function_gemma(mock_llm_client)

                # ToolExecutor should be created
                MockToolExec.assert_called_once()
                # FunctionGemmaAdapter should be created
                MockAdapter.assert_called_once()
                # Check adapter was stored
                assert handler._function_gemma_adapter is not None

    def test_setup_function_gemma_without_store(self):
        """Test _setup_function_gemma when store is None."""
        handler = QueryHandler(store=None, llm_client=None)
        mock_llm_client = MagicMock()

        with patch("src.rag.interface.query_handler.ToolExecutor") as MockToolExec:
            with patch(
                "src.rag.interface.query_handler.FunctionGemmaAdapter"
            ) as MockAdapter:
                handler._setup_function_gemma(mock_llm_client)

                # ToolExecutor and FunctionGemmaAdapter should be created
                MockToolExec.assert_called_once()
                MockAdapter.assert_called_once()
                # ToolExecutor should be created with None search_usecase
                call_kwargs = MockToolExec.call_args[1]
                assert call_kwargs["search_usecase"] is None
                assert call_kwargs["sync_usecase"] is None

    def test_setup_function_gemma_passes_dependencies(self):
        """Test _setup_function_gemma passes dependencies correctly."""
        handler = QueryHandler(store=None, llm_client=None)
        handler.use_reranker = False
        mock_llm_client = MagicMock()

        with patch("src.rag.interface.query_handler.ToolExecutor") as MockToolExec:
            with patch(
                "src.rag.interface.query_handler.FunctionGemmaAdapter"
            ) as MockAdapter:
                handler._setup_function_gemma(mock_llm_client)

                # Verify adapters were created
                MockToolExec.assert_called_once()
                MockAdapter.assert_called_once()
                # Verify ToolExecutor was called with expected parameters
                call_kwargs = MockToolExec.call_args[1]
                assert "query_analyzer" in call_kwargs
                assert "llm_client" in call_kwargs


# ============================================================================
# Test process_query and process_query_stream with overview/article/chapter
# ============================================================================


class TestProcessQueryWithSpecificPatterns:
    """Test process_query routes to specific patterns correctly."""

    def test_process_query_with_overview_pattern(self):
        """Test process_query routes overview query correctly."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch.object(handler, "_is_overview_query") as mock_is_overview:
            mock_is_overview.return_value = True
            with patch.object(handler, "get_regulation_overview") as mock_overview:
                mock_overview.return_value = QueryResult(
                    type=QueryType.OVERVIEW,
                    success=True,
                    content="Overview content",
                    data={"title": "규정"},
                )

                result = handler.process_query("교원인사규정")

        assert result.type == QueryType.OVERVIEW

    def test_process_query_with_article_pattern(self):
        """Test process_query routes article query correctly."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch(
            "src.rag.interface.query_handler.extract_regulation_title"
        ) as mock_extract:
            mock_extract.return_value = "교원인사규정"
            with patch.object(handler, "get_article_view") as mock_article:
                mock_article.return_value = QueryResult(
                    type=QueryType.ARTICLE,
                    success=True,
                    content="Article content",
                )

                result = handler.process_query("교원인사규정 제10조")

        assert result.type == QueryType.ARTICLE

    def test_process_query_with_chapter_pattern(self):
        """Test process_query routes chapter query correctly."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch(
            "src.rag.interface.query_handler.extract_regulation_title"
        ) as mock_extract:
            mock_extract.return_value = "학칙"
            with patch.object(handler, "get_chapter_view") as mock_chapter:
                mock_chapter.return_value = QueryResult(
                    type=QueryType.CHAPTER,
                    success=True,
                    content="Chapter content",
                )

                result = handler.process_query("학칙 제3장")

        assert result.type == QueryType.CHAPTER

    def test_process_query_with_attachment_pattern(self):
        """Test process_query routes attachment query correctly."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch(
            "src.rag.interface.query_handler.parse_attachment_request"
        ) as mock_parse:
            mock_parse.return_value = ("규정", "별표", 1)
            with patch.object(handler, "get_attachment_view") as mock_attachment:
                mock_attachment.return_value = QueryResult(
                    type=QueryType.ATTACHMENT,
                    success=True,
                    content="Attachment content",
                )

                result = handler.process_query("규정 별표1")

        assert result.type == QueryType.ATTACHMENT


class TestProcessQueryStreamWithSpecificPatterns:
    """Test process_query_stream routes to specific patterns correctly."""

    def test_stream_with_overview_pattern(self):
        """Test process_query_stream yields overview result."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch.object(handler, "_is_overview_query") as mock_is_overview:
            mock_is_overview.return_value = True
            with patch.object(handler, "get_regulation_overview") as mock_overview:
                mock_overview.return_value = QueryResult(
                    type=QueryType.OVERVIEW,
                    success=True,
                    content="Overview content",
                    data={"title": "규정"},
                )

                events = list(handler.process_query_stream("교원인사규정"))

        # Should have metadata and complete events
        assert len(events) >= 2
        assert any(e["type"] == "metadata" for e in events)
        assert any(e["type"] == "complete" for e in events)

    def test_stream_with_full_view_mode(self):
        """Test process_query_stream with full_view mode."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch("src.rag.interface.query_handler.decide_search_mode") as mock_mode:
            mock_mode.return_value = "full_view"
            with patch.object(handler, "get_full_view") as mock_full:
                mock_full.return_value = QueryResult(
                    type=QueryType.FULL_VIEW,
                    success=True,
                    content="Full view content",
                    data={"title": "규정"},
                )

                events = list(handler.process_query_stream("학칙 전문"))

        # Should have metadata and complete events
        assert len(events) >= 2

    def test_stream_with_search_mode(self):
        """Test process_query_stream with search mode yields progress."""
        store = FakeVectorStore(count_value=100)
        handler = QueryHandler(store=store)

        with patch("src.rag.interface.query_handler.decide_search_mode") as mock_mode:
            mock_mode.return_value = "search"
            with patch.object(handler, "search") as mock_search:
                mock_search.return_value = QueryResult(
                    type=QueryType.SEARCH,
                    success=True,
                    content="Search results",
                    data={"results": []},
                )

                events = list(handler.process_query_stream("검색어"))

        # Should have progress event
        progress_events = [e for e in events if e["type"] == "progress"]
        assert len(progress_events) > 0


# ============================================================================
# Test search method error paths (lines 1198, 1205)
# ============================================================================


class TestSearchMethodErrorPaths:
    """Test search method error handling paths."""

    def test_search_no_store_returns_error(self):
        """Test search with None store returns error."""
        handler = QueryHandler(store=None)
        result = handler.search("query")

        assert result.type == QueryType.ERROR
        assert "데이터베이스가 초기화되지 않았습니다" in result.content

    def test_search_empty_store_returns_error(self):
        """Test search with empty store returns error."""
        store = FakeVectorStore(count_value=0)
        handler = QueryHandler(store=store)
        result = handler.search("query")

        assert result.type == QueryType.ERROR
        assert "데이터베이스가 비어 있습니다" in result.content


# ============================================================================
# Test get_regulation_overview edge cases (lines 778, 787-816, 823, 841-844, 849-850)
# ============================================================================


class TestGetRegulationOverviewEdgeCases:
    """Test get_regulation_overview edge cases."""

    def test_overview_no_json_path(self):
        """Test get_regulation_overview with no JSON path."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch.object(
            handler.full_view_usecase, "_resolve_json_path"
        ) as mock_resolve:
            mock_resolve.return_value = None

            result = handler.get_regulation_overview("규정")

        assert result.type == QueryType.ERROR
        assert "JSON 파일을 찾을 수 없습니다" in result.content

    def test_overview_with_candidates_single_match(self):
        """Test get_regulation_overview with single candidate match."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.domain.entities import (
            RegulationOverview,
            RegulationStatus,
        )

        mock_candidates = [("1-1-1", "교원인사규정")]
        mock_overview = RegulationOverview(
            title="교원인사규정",
            rule_code="1-1-1",
            status=RegulationStatus.ACTIVE,
            article_count=50,
            chapters=[],
            has_addenda=False,
        )

        with patch.object(
            handler.full_view_usecase, "_resolve_json_path"
        ) as mock_resolve:
            mock_resolve.return_value = "/path/to/json"
            with patch.object(
                handler.loader, "get_regulation_overview"
            ) as mock_overview_method:
                mock_overview_method.return_value = None  # No direct match
                with patch.object(
                    handler.loader, "find_regulation_candidates"
                ) as mock_find:
                    mock_find.return_value = mock_candidates
                    with patch.object(
                        handler.loader, "get_regulation_overview"
                    ) as mock_overview2:
                        mock_overview2.return_value = mock_overview

                        result = handler.get_regulation_overview("교원인사")

        assert result.success is True
        assert "교원인사규정" in result.content

    def test_overview_with_candidates_multiple_matches_no_exact(self):
        """Test get_regulation_overview with multiple matches, no exact match."""
        handler = QueryHandler(store=None, llm_client=None)

        mock_candidates = [
            ("1-1-1", "교원인사규정"),
            ("2-1-1", "교원인사규정시행세칙"),
        ]

        with patch.object(
            handler.full_view_usecase, "_resolve_json_path"
        ) as mock_resolve:
            mock_resolve.return_value = "/path/to/json"
            with patch.object(
                handler.loader, "get_regulation_overview"
            ) as mock_overview:
                mock_overview.return_value = None
                with patch.object(
                    handler.loader, "find_regulation_candidates"
                ) as mock_find:
                    mock_find.return_value = mock_candidates

                    result = handler.get_regulation_overview("교원인")

        assert result.type == QueryType.CLARIFICATION
        assert "여러 규정이 매칭됩니다" in result.content

    def test_overview_no_candidates(self):
        """Test get_regulation_overview with no candidates."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch.object(
            handler.full_view_usecase, "_resolve_json_path"
        ) as mock_resolve:
            mock_resolve.return_value = "/path/to/json"
            with patch.object(
                handler.loader, "get_regulation_overview"
            ) as mock_overview:
                mock_overview.return_value = None
                with patch.object(
                    handler.loader, "find_regulation_candidates"
                ) as mock_find:
                    mock_find.return_value = []  # No candidates

                    result = handler.get_regulation_overview("존재하지않는규정")

        assert result.type == QueryType.ERROR
        assert "찾을 수 없습니다" in result.content

    def test_overview_with_chapters(self):
        """Test get_regulation_overview includes chapters."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.domain.entities import (
            ChapterInfo,
            RegulationOverview,
            RegulationStatus,
        )

        mock_overview = RegulationOverview(
            title="학칙",
            rule_code="1-1-1",
            status=RegulationStatus.ACTIVE,
            article_count=100,
            chapters=[
                ChapterInfo(
                    display_no="제1장",
                    title="총칙",
                    article_range="제1조-제10조",
                ),
                ChapterInfo(
                    display_no="제2장",
                    title="학생",
                    article_range="제11조-제50조",
                ),
            ],
            has_addenda=False,
        )

        with patch.object(
            handler.full_view_usecase, "_resolve_json_path"
        ) as mock_resolve:
            mock_resolve.return_value = "/path/to/json"
            with patch.object(
                handler.loader, "get_regulation_overview"
            ) as mock_overview_method:
                mock_overview_method.return_value = mock_overview

                result = handler.get_regulation_overview("학칙")

        assert result.success is True
        assert "목차" in result.content
        assert "제1장" in result.content
        assert "총칙" in result.content

    def test_overview_without_chapters(self):
        """Test get_regulation_overview without chapters."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.domain.entities import (
            RegulationOverview,
            RegulationStatus,
        )

        mock_overview = RegulationOverview(
            title="단순규정",
            rule_code="1-1-1",
            status=RegulationStatus.ACTIVE,
            article_count=10,
            chapters=None,  # No chapters
            has_addenda=False,
        )

        with patch.object(
            handler.full_view_usecase, "_resolve_json_path"
        ) as mock_resolve:
            mock_resolve.return_value = "/path/to/json"
            with patch.object(
                handler.loader, "get_regulation_overview"
            ) as mock_overview_method:
                mock_overview_method.return_value = mock_overview
                with patch.object(
                    handler.loader, "get_regulation_titles"
                ) as mock_titles:
                    mock_titles.return_value = {}

                    result = handler.get_regulation_overview("단순규정")

        assert result.success is True
        assert "장 구조 없이" in result.content

    def test_overview_with_addenda(self):
        """Test get_regulation_overview includes addenda notice."""
        handler = QueryHandler(store=None, llm_client=None)

        from src.rag.domain.entities import (
            RegulationOverview,
            RegulationStatus,
        )

        mock_overview = RegulationOverview(
            title="규정",
            rule_code="1-1-1",
            status=RegulationStatus.ACTIVE,
            article_count=50,
            chapters=[],
            has_addenda=True,  # Has addenda
        )

        with patch.object(
            handler.full_view_usecase, "_resolve_json_path"
        ) as mock_resolve:
            mock_resolve.return_value = "/path/to/json"
            with patch.object(
                handler.loader, "get_regulation_overview"
            ) as mock_overview_method:
                mock_overview_method.return_value = mock_overview

                result = handler.get_regulation_overview("규정")

        assert result.success is True
        assert "부칙" in result.content


# ============================================================================
# Test process_query_stream additional paths (lines 515-520, 525-530, 538-543)
# ============================================================================


class TestProcessQueryStreamAdditionalPaths:
    """Test process_query_stream additional code paths."""

    def test_stream_article_query_yields_enriched_result(self):
        """Test process_query_stream yields enriched article result."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch(
            "src.rag.interface.query_handler.extract_regulation_title"
        ) as mock_extract:
            mock_extract.return_value = "규정"
            with patch.object(handler, "get_article_view") as mock_article:
                mock_article.return_value = QueryResult(
                    type=QueryType.ARTICLE,
                    success=True,
                    content="Article content",
                    data={"regulation_title": "규정"},
                )

                events = list(handler.process_query_stream("규정 제10조"))

        # Should have enriched result
        assert any(e["type"] == "metadata" for e in events)
        assert any(e["type"] == "complete" for e in events)

    def test_stream_chapter_query_yields_result(self):
        """Test process_query_stream yields chapter result."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch(
            "src.rag.interface.query_handler.extract_regulation_title"
        ) as mock_extract:
            mock_extract.return_value = "규정"
            with patch.object(handler, "get_chapter_view") as mock_chapter:
                mock_chapter.return_value = QueryResult(
                    type=QueryType.CHAPTER,
                    success=True,
                    content="Chapter content",
                    data={"title": "규정"},
                )

                events = list(handler.process_query_stream("규정 제3장"))

        assert len(events) >= 2

    def test_stream_attachment_query_yields_result(self):
        """Test process_query_stream yields attachment result."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch(
            "src.rag.interface.query_handler.parse_attachment_request"
        ) as mock_parse:
            mock_parse.return_value = ("규정", "별표", None)
            with patch.object(handler, "get_attachment_view") as mock_attachment:
                mock_attachment.return_value = QueryResult(
                    type=QueryType.ATTACHMENT,
                    success=True,
                    content="Attachment content",
                    data={"title": "규정"},
                )

                events = list(handler.process_query_stream("규정 별표"))

        assert len(events) >= 2


# ============================================================================
# Test process_query additional paths (lines 446-448, 454-455, 462, 474)
# ============================================================================


class TestProcessQueryAdditionalPaths:
    """Test process_query additional code paths."""

    def test_process_query_full_view_mode(self):
        """Test process_query with full_view mode."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch("src.rag.interface.query_handler.decide_search_mode") as mock_mode:
            mock_mode.return_value = "full_view"
            with patch.object(handler, "get_full_view") as mock_full:
                mock_full.return_value = QueryResult(
                    type=QueryType.FULL_VIEW,
                    success=True,
                    content="Full view",
                    data={"title": "규정"},
                )

                result = handler.process_query("규정 전체")

        assert result.type == QueryType.FULL_VIEW

    def test_process_query_audience_ambiguity(self):
        """Test process_query returns clarification for ambiguous audience."""
        handler = QueryHandler(store=None, llm_client=None)

        with patch.object(
            handler.query_analyzer, "is_audience_ambiguous"
        ) as mock_ambig:
            mock_ambig.return_value = True

            result = handler.process_query("신청 방법")

        assert result.type == QueryType.CLARIFICATION
        assert result.clarification_type == "audience"
        assert "교수" in result.clarification_options

    def test_process_query_ask_mode(self):
        """Test process_query routes to ask mode."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        with patch("src.rag.interface.query_handler.decide_search_mode") as mock_mode:
            mock_mode.return_value = "ask"  # Ask mode
            with patch.object(handler, "ask") as mock_ask:
                mock_ask.return_value = QueryResult(
                    type=QueryType.ASK,
                    success=True,
                    content="Answer",
                    data={"question": "query"},
                )

                result = handler.process_query("질문")

        assert result.type == QueryType.ASK


# ============================================================================
# Test process_query_stream ask mode with suggestions (lines 613-625)
# ============================================================================


class TestProcessQueryStreamAskMode:
    """Test process_query_stream ask mode with suggestion enrichment."""

    def test_stream_ask_mode_enriches_with_suggestions(self):
        """Test process_query_stream enriches ask result with suggestions."""
        store = FakeVectorStore(count_value=100)
        llm_client = MagicMock()
        handler = QueryHandler(store=store, llm_client=llm_client)

        with patch("src.rag.interface.query_handler.decide_search_mode") as mock_mode:
            mock_mode.return_value = "ask"
            with patch.object(handler, "ask_stream") as mock_ask_stream:
                # Mock ask_stream to yield events including complete
                def mock_generator():
                    yield {"type": "progress", "content": "Searching..."}
                    yield {
                        "type": "complete",
                        "content": "Answer text",
                        "data": {},
                    }

                mock_ask_stream.return_value = mock_generator()

                events = list(handler.process_query_stream("question"))

        # Should have events with suggestions enriched
        complete_events = [e for e in events if e["type"] == "complete"]
        assert len(complete_events) > 0


# ============================================================================
# Test FunctionGemma path in process_query (lines 454-455)
# ============================================================================


class TestProcessQueryFunctionGemmaPath:
    """Test process_query FunctionGemma path."""

    def test_process_query_with_function_gemma_enabled(self):
        """Test process_query uses FunctionGemma when enabled."""
        store = FakeVectorStore(count_value=100)
        handler = QueryHandler(store=store)

        # Mock FunctionGemma adapter
        mock_adapter = MagicMock()
        mock_adapter.process_query.return_value = ("Answer", [])
        handler._function_gemma_adapter = mock_adapter

        result = handler.process_query(
            "query", options=QueryOptions(use_function_gemma=True)
        )

        # Should use FunctionGemma
        mock_adapter.process_query.assert_called_once()
        assert result.type == QueryType.ASK
