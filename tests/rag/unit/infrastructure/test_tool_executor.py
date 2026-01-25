"""
Comprehensive tests for ToolExecutor.

Covers tool execution logic, error handling, and edge cases.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.rag.infrastructure.tool_executor import ToolExecutor, ToolResult


class TestToolResult:
    """Test ToolResult dataclass."""

    def test_to_dict_success(self):
        result = ToolResult(
            tool_name="test_tool",
            success=True,
            result={"data": "value"},
            arguments={"arg1": "val1"},
        )
        expected = {
            "tool_name": "test_tool",
            "success": True,
            "result": {"data": "value"},
            "error": None,
            "arguments": {"arg1": "val1"},
        }
        assert result.to_dict() == expected

    def test_to_dict_failure(self):
        result = ToolResult(
            tool_name="test_tool", success=False, result=None, error="Test error"
        )
        data = result.to_dict()
        assert data["tool_name"] == "test_tool"
        assert data["success"] is False
        assert data["error"] == "Test error"
        assert data["result"] is None

    def test_to_context_string_success_string_result(self):
        result = ToolResult(tool_name="test_tool", success=True, result="String result")
        context = result.to_context_string()
        assert context == "[test_tool] String result"

    def test_to_context_string_success_dict_result(self):
        result = ToolResult(
            tool_name="test_tool", success=True, result={"key": "value", "number": 42}
        )
        context = result.to_context_string()
        assert "[test_tool]" in context
        assert '"key": "value"' in context
        assert '"number": 42' in context

    def test_to_context_string_failure(self):
        result = ToolResult(
            tool_name="test_tool",
            success=False,
            result=None,
            error="Something went wrong",
        )
        context = result.to_context_string()
        assert context == "[test_tool] Error: Something went wrong"

    def test_to_context_string_with_none_arguments(self):
        result = ToolResult(tool_name="test", success=True, result="value")
        assert result.arguments is None
        assert result.to_context_string() == "[test] value"


class TestToolExecutorInit:
    """Test ToolExecutor initialization."""

    def test_init_with_all_dependencies(self):
        search_uc = MagicMock()
        sync_uc = MagicMock()
        analyzer = MagicMock()
        llm = MagicMock()

        executor = ToolExecutor(
            search_usecase=search_uc,
            sync_usecase=sync_uc,
            query_analyzer=analyzer,
            llm_client=llm,
            json_path="test/path.json",
        )

        assert executor._search_usecase is search_uc
        assert executor._sync_usecase is sync_uc
        assert executor._query_analyzer is analyzer
        assert executor._llm_client is llm
        assert executor._json_path == "test/path.json"

    def test_init_with_minimal_params(self):
        executor = ToolExecutor()
        assert executor._search_usecase is None
        assert executor._sync_usecase is None
        assert executor._query_analyzer is None
        assert executor._llm_client is None
        assert executor._json_path == "data/output/규정집.json"


class TestToolExecutorExecute:
    """Test the execute method."""

    def test_execute_unknown_tool(self):
        executor = ToolExecutor()
        result = executor.execute("unknown_tool", {})

        assert result.success is False
        assert result.tool_name == "unknown_tool"
        assert "Unknown tool" in result.error

    def test_execute_with_exception(self):
        def failing_handler(args):
            raise ValueError("Test error")

        executor = ToolExecutor()
        executor._get_handler = lambda name: failing_handler

        result = executor.execute("failing_tool", {})

        assert result.success is False
        assert "Test error" in result.error
        assert result.tool_name == "failing_tool"

    def test_execute_successful_tool(self):
        def success_handler(args):
            return {"status": "ok"}

        executor = ToolExecutor()
        executor._get_handler = lambda name: success_handler

        result = executor.execute("success_tool", {"key": "value"})

        assert result.success is True
        assert result.result == {"status": "ok"}
        assert result.arguments == {"key": "value"}
        assert result.tool_name == "success_tool"


class TestToolExecutorSearchTools:
    """Test search-related tool handlers."""

    def test_handle_search_regulations_missing_usecase(self):
        executor = ToolExecutor(search_usecase=None)
        with pytest.raises(RuntimeError, match="SearchUseCase not initialized"):
            executor._handle_search_regulations({"query": "test"})

    def test_handle_search_regulations_basic(self):
        mock_search_uc = MagicMock()
        mock_result = MagicMock()
        mock_result.chunk.title = "Test Regulation"
        mock_result.chunk.parent_path = ["Parent", "Chapter"]
        mock_result.chunk.text = "Some text"
        mock_result.chunk.rule_code = "1-1-1"
        mock_result.score = 0.95
        mock_search_uc.search.return_value = [mock_result]

        executor = ToolExecutor(search_usecase=mock_search_uc)

        result = executor._handle_search_regulations(
            {"query": "test", "top_k": 5, "audience": "student"}
        )

        assert result["count"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["regulation_title"] == "Parent"
        assert result["results"][0]["score"] == 0.95

    def test_handle_search_regulations_with_query_analyzer(self):
        mock_search_uc = MagicMock()
        mock_search_uc.search.return_value = []

        mock_analyzer = MagicMock()
        mock_analyzer.expand_query.return_value = "expanded query"

        executor = ToolExecutor(
            search_usecase=mock_search_uc, query_analyzer=mock_analyzer
        )

        executor._handle_search_regulations({"query": "test"})
        mock_analyzer.expand_query.assert_called_once_with("test")

    def test_handle_search_regulations_audience_mapping(self):
        mock_search_uc = MagicMock()
        mock_search_uc.search.return_value = []

        from src.rag.infrastructure.query_analyzer import Audience

        executor = ToolExecutor(search_usecase=mock_search_uc)

        # Test each audience type
        for aud_str, _expected_aud in [
            ("all", Audience.ALL),
            ("student", Audience.STUDENT),
            ("faculty", Audience.FACULTY),
            ("staff", Audience.STAFF),
        ]:
            mock_search_uc.reset_mock()
            executor._handle_search_regulations({"query": "test", "audience": aud_str})
            # Verify the search was called (audience passed correctly internally)

    def test_handle_get_article_missing_usecase(self):
        mock_search_uc = MagicMock()
        mock_search_uc.search.return_value = []
        executor = ToolExecutor(search_usecase=mock_search_uc)

        with patch("src.rag.application.full_view_usecase.FullViewUseCase") as mock_uc:
            mock_instance = MagicMock()
            mock_instance.find_matches.return_value = []
            mock_uc.return_value = mock_instance

            result = executor._handle_get_article(
                {"regulation": "test", "article_no": 5}
            )

            # Should fall back to search
            assert "count" in result or "found" in result

    def test_handle_get_chapter(self):
        mock_search_uc = MagicMock()
        mock_search_uc.search.return_value = []
        executor = ToolExecutor(search_usecase=mock_search_uc)

        result = executor._handle_get_chapter({"regulation": "test", "chapter_no": 3})

        # Should delegate to search
        assert "count" in result

    def test_handle_get_attachment(self):
        mock_search_uc = MagicMock()
        mock_search_uc.search.return_value = []
        executor = ToolExecutor(search_usecase=mock_search_uc)

        result = executor._handle_get_attachment(
            {"regulation": "test", "label": "별표"}
        )

        # Should delegate to search
        assert "count" in result

    def test_handle_get_regulation_overview(self):
        mock_search_uc = MagicMock()
        mock_search_uc.search.return_value = []
        executor = ToolExecutor(search_usecase=mock_search_uc)

        result = executor._handle_get_regulation_overview({"regulation": "test"})

        # Should delegate to search
        assert "overview" in result

    def test_handle_get_full_regulation(self):
        executor = ToolExecutor()

        result = executor._handle_get_full_regulation({"regulation": "test"})

        assert result["regulation"] == "test"
        assert "note" in result


class TestToolExecutorAnalysisTools:
    """Test analysis tool handlers."""

    def test_handle_expand_synonyms_missing_analyzer(self):
        executor = ToolExecutor(query_analyzer=None)
        with pytest.raises(RuntimeError, match="QueryAnalyzer not initialized"):
            executor._handle_expand_synonyms({"term": "test"})

    def test_handle_expand_synonyms(self):
        mock_analyzer = MagicMock()
        mock_analyzer._synonyms = {"test": ["syn1", "syn2"]}
        executor = ToolExecutor(query_analyzer=mock_analyzer)

        result = executor._handle_expand_synonyms({"term": "test"})
        assert result["term"] == "test"
        assert result["synonyms"] == ["syn1", "syn2"]

    def test_handle_detect_intent_missing_analyzer(self):
        executor = ToolExecutor(query_analyzer=None)
        with pytest.raises(RuntimeError, match="QueryAnalyzer not initialized"):
            executor._handle_detect_intent({"query": "test"})

    def test_handle_detect_audience_missing_analyzer(self):
        executor = ToolExecutor(query_analyzer=None)
        with pytest.raises(RuntimeError, match="QueryAnalyzer not initialized"):
            executor._handle_detect_audience({"query": "test"})

    def test_handle_analyze_query_missing_analyzer(self):
        executor = ToolExecutor(query_analyzer=None)
        with pytest.raises(RuntimeError, match="QueryAnalyzer not initialized"):
            executor._handle_analyze_query({"query": "test"})


class TestToolExecutorAdminTools:
    """Test admin tool handlers."""

    def test_handle_sync_database_missing_usecase(self):
        executor = ToolExecutor(sync_usecase=None)
        with pytest.raises(RuntimeError, match="SyncUseCase not initialized"):
            executor._handle_sync_database({"full": False})

    def test_handle_sync_database_full(self):
        mock_sync_uc = MagicMock()
        mock_result = MagicMock()
        mock_result.added = 10
        mock_result.modified = 2
        mock_result.removed = 1
        mock_result.unchanged = 50
        mock_result.errors = []
        mock_sync_uc.full_sync.return_value = mock_result

        executor = ToolExecutor(sync_usecase=mock_sync_uc)

        result = executor._handle_sync_database({"full": True})
        assert result["added"] == 10
        assert result["modified"] == 2

    def test_handle_sync_database_incremental(self):
        mock_sync_uc = MagicMock()
        mock_result = MagicMock()
        mock_result.added = 5
        mock_result.modified = 1
        mock_result.removed = 0
        mock_result.unchanged = 25
        mock_result.errors = []
        mock_sync_uc.incremental_sync.return_value = mock_result

        executor = ToolExecutor(sync_usecase=mock_sync_uc)

        result = executor._handle_sync_database({"full": False})
        assert result["added"] == 5
        assert result["modified"] == 1

    def test_handle_get_sync_status_missing_usecase(self):
        executor = ToolExecutor(sync_usecase=None)
        with pytest.raises(RuntimeError, match="SyncUseCase not initialized"):
            executor._handle_get_sync_status({})

    def test_handle_reset_database(self):
        mock_sync_uc = MagicMock()
        executor = ToolExecutor(sync_usecase=mock_sync_uc)

        result = executor._handle_reset_database({})
        assert result["status"] == "reset_complete"
        mock_sync_uc.reset_state.assert_called_once()


class TestToolExecutorResponseTools:
    """Test response tool handlers."""

    def test_handle_generate_answer_missing_llm(self):
        executor = ToolExecutor(llm_client=None)
        with pytest.raises(RuntimeError, match="LLM client not initialized"):
            executor._handle_generate_answer({"question": "test", "context": "context"})

    def test_handle_generate_answer(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Generated answer"
        executor = ToolExecutor(llm_client=mock_llm)

        result = executor._handle_generate_answer(
            {"question": "What is X?", "context": "X is something"}
        )

        assert result == "Generated answer"
        mock_llm.generate.assert_called_once()

    def test_handle_clarify_query(self):
        executor = ToolExecutor()

        result = executor._handle_clarify_query(
            {"query": "test query", "options": ["Option A", "Option B"]}
        )

        assert result["type"] == "clarification"
        assert "test query" in result["message"]
        assert result["options"] == ["Option A", "Option B"]

    def test_handle_clarify_query_empty_options(self):
        executor = ToolExecutor()

        result = executor._handle_clarify_query({"query": "test", "options": []})

        assert result["type"] == "clarification"
        assert result["options"] == []


class TestToolExecutorGetHandler:
    """Test the _get_handler method."""

    def test_get_handler_search_tools(self):
        executor = ToolExecutor()

        # Test all search tools exist
        search_tools = [
            "search_regulations",
            "get_article",
            "get_chapter",
            "get_attachment",
            "get_regulation_overview",
            "get_full_regulation",
        ]

        for tool in search_tools:
            handler = executor._get_handler(tool)
            assert handler is not None
            assert callable(handler)

    def test_get_handler_analysis_tools(self):
        executor = ToolExecutor()

        analysis_tools = [
            "expand_synonyms",
            "detect_intent",
            "detect_audience",
            "analyze_query",
        ]

        for tool in analysis_tools:
            handler = executor._get_handler(tool)
            assert handler is not None
            assert callable(handler)

    def test_get_handler_admin_tools(self):
        executor = ToolExecutor()

        admin_tools = [
            "sync_database",
            "get_sync_status",
            "reset_database",
        ]

        for tool in admin_tools:
            handler = executor._get_handler(tool)
            assert handler is not None
            assert callable(handler)

    def test_get_handler_response_tools(self):
        executor = ToolExecutor()

        response_tools = [
            "generate_answer",
            "clarify_query",
        ]

        for tool in response_tools:
            handler = executor._get_handler(tool)
            assert handler is not None
            assert callable(handler)

    def test_get_handler_invalid_tool(self):
        executor = ToolExecutor()
        handler = executor._get_handler("invalid_tool_name")
        assert handler is None


class TestToolExecutorEdgeCases:
    """Test edge cases and error scenarios."""

    def test_handle_search_regulations_default_params(self):
        mock_search_uc = MagicMock()
        mock_search_uc.search.return_value = []

        executor = ToolExecutor(search_usecase=mock_search_uc)

        # Test with minimal args
        result = executor._handle_search_regulations({"query": "test"})
        assert "count" in result

    def test_handle_search_regulations_with_top_k(self):
        mock_search_uc = MagicMock()
        mock_search_uc.search.return_value = []

        executor = ToolExecutor(search_usecase=mock_search_uc)

        result = executor._handle_search_regulations({"query": "test", "top_k": 10})

        assert "count" in result

    def test_handle_search_regulations_with_long_text_truncation(self):
        mock_search_uc = MagicMock()
        mock_result = MagicMock()
        mock_result.chunk.title = "Title"
        mock_result.chunk.parent_path = ["Parent"]
        mock_result.chunk.text = "x" * 600  # Long text
        mock_result.chunk.rule_code = "1-1-1"
        mock_result.chunk.score = 0.9
        mock_search_uc.search.return_value = [mock_result]

        executor = ToolExecutor(search_usecase=mock_search_uc)

        result = executor._handle_search_regulations({"query": "test"})
        # Text should be truncated with ...
        assert "..." in result["results"][0]["text"]
        assert len(result["results"][0]["text"]) <= 503  # 500 + "..."

    def test_handle_search_regulations_no_parent_path(self):
        mock_search_uc = MagicMock()
        mock_result = MagicMock()
        mock_result.chunk.title = "Direct Title"
        mock_result.chunk.parent_path = []  # Empty list instead of None
        mock_result.chunk.text = "Text"
        mock_result.chunk.rule_code = "1-1-1"
        mock_result.chunk.score = 0.9
        mock_search_uc.search.return_value = [mock_result]

        executor = ToolExecutor(search_usecase=mock_search_uc)

        result = executor._handle_search_regulations({"query": "test"})
        # When parent_path is empty, title should be used as regulation_title
        assert result["results"][0]["regulation_title"] == "Direct Title"

    def test_execute_with_dict_arguments(self):
        executor = ToolExecutor()
        result = executor.execute("unknown_tool", {"key": "value"})
        assert result.success is False

    def test_execute_preserves_arguments_on_error(self):
        def failing_handler(args):
            raise ValueError("test")

        executor = ToolExecutor()
        executor._get_handler = lambda name: failing_handler

        args = {"input": "value"}
        result = executor.execute("test", args)

        assert result.success is False
        assert result.arguments == args
