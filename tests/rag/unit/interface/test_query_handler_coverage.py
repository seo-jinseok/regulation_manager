"""
Extended tests for query_handler.py to improve coverage from 44% to 85%.

Covers missing lines in process_query, process_query_stream, and business logic methods.
Focuses on edge cases, error paths, and less-tested code paths.
"""

from unittest.mock import MagicMock, patch

from src.rag.interface.query_handler import (
    QueryContext,
    QueryHandler,
    QueryOptions,
    QueryResult,
    QueryType,
)


class TestQueryHandlerNormalization:
    """Test query normalization and validation."""

    def test_normalize_query_empty_string(self):
        """Test normalization with empty string."""
        handler = QueryHandler(store=None, llm_client=None)
        result = handler._normalize_query("")
        assert result == ""

    def test_normalize_query_whitespace(self):
        """Test normalization with whitespace."""
        handler = QueryHandler(store=None, llm_client=None)
        result = handler._normalize_query("   ")
        assert result == "   "

    def test_normalize_query_normal_text(self):
        """Test normalization with normal text."""
        handler = QueryHandler(store=None, llm_client=None)
        result = handler._normalize_query("교원 연구년")
        assert result == "교원 연구년"


class TestQueryHandlerValidationExtended:
    """Extended validation tests."""

    def test_validate_query_normal_korean(self):
        """Test validation with normal Korean text."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("교원인사규정")
        assert is_valid is True
        assert msg == ""

    def test_validate_query_with_newlines(self):
        """Test validation with newlines allowed."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("교원\n연구년")
        assert is_valid is True

    def test_validate_query_with_tabs(self):
        """Test validation with tabs allowed."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("교원\t연구년")
        assert is_valid is True

    def test_validate_query_carriage_return(self):
        """Test validation with carriage return allowed."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("교원\r연구년")
        assert is_valid is True

    def test_validate_query_xss_pattern(self):
        """Test XSS pattern detection."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("<script>alert('xss')</script>")
        assert is_valid is False

    def test_validate_query_sql_drop(self):
        """Test SQL DROP pattern detection."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("DROP TABLE users")
        assert is_valid is False

    def test_validate_query_sql_delete(self):
        """Test SQL DELETE pattern detection."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("DELETE FROM users")
        assert is_valid is False

    def test_validate_query_template_injection(self):
        """Test template injection pattern detection."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("${7*7}")
        assert is_valid is False

    def test_validate_query_iframe(self):
        """Test iframe tag detection."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("<iframe src='evil.com'>")
        assert is_valid is False

    def test_validate_query_embed(self):
        """Test embed tag detection."""
        handler = QueryHandler(store=None, llm_client=None)
        is_valid, msg = handler.validate_query("<embed src='evil.swf'>")
        assert is_valid is False


class TestQueryHandlerProcessQuery:
    """Test process_query method edge cases."""

    def test_process_query_empty_validation_error(self):
        """Test process_query with empty query (line 262-263)."""
        handler = QueryHandler(store=None, llm_client=None)
        result = handler.process_query("")
        assert result.type == QueryType.ERROR
        assert result.success is False
        assert "검색어를 입력" in result.content

    def test_process_query_too_long_error(self):
        """Test process_query with query too long (line 258-259)."""
        handler = QueryHandler(store=None, llm_client=None)
        long_query = "가" * 501
        result = handler.process_query(long_query)
        assert result.type == QueryType.ERROR
        assert result.success is False
        assert "너무 깁니다" in result.content

    def test_process_query_xss_error(self):
        """Test process_query with XSS pattern (line 266-268)."""
        handler = QueryHandler(store=None, llm_client=None)
        result = handler.process_query("<script>alert('xss')</script>")
        assert result.type == QueryType.ERROR
        assert result.success is False
        assert "허용되지 않는 문자" in result.content

    def test_process_query_control_char_error(self):
        """Test process_query with control character (line 271-272)."""
        handler = QueryHandler(store=None, llm_client=None)
        query_with_null = "교원\x00연구년"
        result = handler.process_query(query_with_null)
        assert result.type == QueryType.ERROR
        assert result.success is False
        assert "제어 문자" in result.content


class TestQueryHandlerResultEnrichment:
    """Test result enrichment with suggestions."""

    def test_enrich_with_suggestions_error_type(self):
        """Test enrichment skips ERROR type (line 629-630)."""
        handler = QueryHandler(store=None, llm_client=None)
        result = QueryResult(type=QueryType.ERROR, success=False, content="Error")
        enriched = handler._enrich_with_suggestions(result, "query")
        assert enriched == result  # Should return unchanged

    def test_enrich_with_suggestions_clarification_type(self):
        """Test enrichment skips CLARIFICATION type (line 629-630)."""
        handler = QueryHandler(store=None, llm_client=None)
        result = QueryResult(
            type=QueryType.CLARIFICATION,
            success=True,
            content="Select option",
        )
        enriched = handler._enrich_with_suggestions(result, "query")
        assert enriched == result  # Should return unchanged

    def test_enrich_result_with_suggestions(self):
        """Test enrichment adds suggestions to content."""
        handler = QueryHandler(store=None, llm_client=None)
        result = QueryResult(
            type=QueryType.SEARCH,
            success=True,
            content="Search results",
            data={"regulation_title": "교원인사규정"},
        )

        with patch(
            "src.rag.interface.query_handler.get_followup_suggestions"
        ) as mock_sugs:
            with patch(
                "src.rag.interface.query_handler.format_suggestions_for_cli"
            ) as mock_fmt:
                mock_sugs.return_value = ["suggestion1", "suggestion2"]
                mock_fmt.return_value = (
                    "\n\n### Suggestions\n- suggestion1\n- suggestion2"
                )

                enriched = handler._enrich_with_suggestions(result, "query")

                # Should have suggestions appended
                assert enriched.suggestions == ["suggestion1", "suggestion2"]
                assert "suggestion1" in enriched.content


class TestQueryHandlerYieldResult:
    """Test _yield_result helper method."""

    def test_yield_result_clarification(self):
        """Test yielding clarification result (line 650-656)."""
        handler = QueryHandler(store=None, llm_client=None)
        result = QueryResult(
            type=QueryType.CLARIFICATION,
            success=True,
            clarification_type="audience",
            clarification_options=["교수", "학생"],
            content="Select audience",
        )

        events = list(handler._yield_result(result))
        assert len(events) == 1
        assert events[0]["type"] == "clarification"
        assert events[0]["clarification_type"] == "audience"
        assert events[0]["options"] == ["교수", "학생"]

    def test_yield_result_error(self):
        """Test yielding error result (line 657-658)."""
        handler = QueryHandler(store=None, llm_client=None)
        result = QueryResult(
            type=QueryType.ERROR, success=False, content="Error message"
        )

        events = list(handler._yield_result(result))
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert events[0]["content"] == "Error message"

    def test_yield_result_standard(self):
        """Test yielding standard result (line 660-677)."""
        handler = QueryHandler(store=None, llm_client=None)
        result = QueryResult(
            type=QueryType.SEARCH,
            success=True,
            content="Search results",
            data={"rule_code": "1-1-1", "title": "Regulation Title"},
            state_update={"last_regulation": "Regulation"},
        )

        events = list(handler._yield_result(result))
        assert len(events) == 3  # metadata, complete, state

        # Check metadata event
        assert events[0]["type"] == "metadata"
        assert events[0]["rule_code"] == "1-1-1"
        assert events[0]["regulation_title"] == "Regulation Title"

        # Check complete event
        assert events[1]["type"] == "complete"
        assert events[1]["content"] == "Search results"

        # Check state event
        assert events[2]["type"] == "state"
        assert events[2]["update"]["last_regulation"] == "Regulation"


class TestQueryHandlerProcessQueryStream:
    """Test process_query_stream method."""

    def test_stream_empty_query_error(self):
        """Test stream with empty query."""
        handler = QueryHandler(store=None, llm_client=None)
        events = list(handler.process_query_stream(""))
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "검색어를 입력" in events[0]["content"]

    def test_stream_validation_error(self):
        """Test stream with validation error."""
        handler = QueryHandler(store=None, llm_client=None)
        events = list(handler.process_query_stream("<script>alert('xss')</script>"))
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "허용되지 않는 문자" in events[0]["content"]

    def test_stream_too_long_error(self):
        """Test stream with query too long."""
        handler = QueryHandler(store=None, llm_client=None)
        long_query = "가" * 501
        events = list(handler.process_query_stream(long_query))
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "너무 깁니다" in events[0]["content"]


class TestQueryHandlerGetLastQueryRewrite:
    """Test get_last_query_rewrite method."""

    def test_get_last_query_rewrite_initial(self):
        """Test initial state of last query rewrite."""
        handler = QueryHandler(store=None, llm_client=None)
        result = handler.get_last_query_rewrite()
        assert result is None


class TestQueryHandlerIsOverviewQuery:
    """Test _is_overview_query method."""

    def test_is_overview_query_regulation_only(self):
        """Test overview query with regulation name only."""
        handler = QueryHandler(store=None, llm_client=None)
        # This requires mocking REGULATION_ONLY_PATTERN
        with patch(
            "src.rag.interface.query_handler.REGULATION_ONLY_PATTERN"
        ) as mock_pattern:
            mock_pattern.match.return_value = MagicMock()
            result = handler._is_overview_query("교원인사규정")
            assert result is True

    def test_is_overview_query_rule_code_only(self):
        """Test overview query with rule code only."""
        handler = QueryHandler(store=None, llm_client=None)
        with patch("src.rag.interface.query_handler.RULE_CODE_PATTERN") as mock_pattern:
            mock_pattern.match.return_value = MagicMock()
            result = handler._is_overview_query("1-1-1")
            assert result is True

    def test_is_overview_query_negative(self):
        """Test overview query returns False for normal query."""
        handler = QueryHandler(store=None, llm_client=None)
        with patch(
            "src.rag.interface.query_handler.REGULATION_ONLY_PATTERN"
        ) as mock_reg_pattern:
            with patch(
                "src.rag.interface.query_handler.RULE_CODE_PATTERN"
            ) as mock_rule_pattern:
                mock_reg_pattern.match.return_value = None
                mock_rule_pattern.match.return_value = None
                result = handler._is_overview_query("교원 연구년 신청 방법")
                assert result is False


class TestQueryResultDataClass:
    """Test QueryResult dataclass edge cases."""

    def test_query_result_default_debug_info(self):
        """Test QueryResult default debug_info is empty string."""
        result = QueryResult(type=QueryType.SEARCH, success=True)
        assert result.debug_info == ""

    def test_query_result_default_suggestions(self):
        """Test QueryResult default suggestions is empty list."""
        result = QueryResult(type=QueryType.SEARCH, success=True)
        assert result.suggestions == []

    def test_query_result_with_all_fields(self):
        """Test QueryResult with all fields populated."""
        result = QueryResult(
            type=QueryType.ASK,
            success=True,
            content="Answer",
            data={"key": "value"},
            state_update={"key": "value"},
            clarification_type="audience",
            clarification_options=["opt1", "opt2"],
            debug_info="debug",
            suggestions=["sug1"],
        )
        assert result.type == QueryType.ASK
        assert result.success is True
        assert result.content == "Answer"
        assert result.data["key"] == "value"
        assert result.state_update["key"] == "value"
        assert result.clarification_type == "audience"
        assert result.clarification_options == ["opt1", "opt2"]
        assert result.debug_info == "debug"
        assert result.suggestions == ["sug1"]


class TestQueryOptionsDataClass:
    """Test QueryOptions dataclass."""

    def test_query_options_all_defaults(self):
        """Test QueryOptions with all defaults."""
        options = QueryOptions()
        assert options.top_k == 5
        assert options.force_mode is None
        assert options.include_abolished is False
        assert options.use_rerank is True
        assert options.audience_override is None
        assert options.show_debug is False
        assert options.llm_provider is None
        assert options.llm_model is None
        assert options.llm_base_url is None
        assert options.use_function_gemma is False


class TestQueryContextDataClass:
    """Test QueryContext dataclass."""

    def test_query_context_with_state(self):
        """Test QueryContext with custom state."""
        context = QueryContext(state={"key": "value"})
        assert context.state["key"] == "value"
        assert context.history == []
        assert context.interactive is False


class TestQueryTypeEnum:
    """Test QueryType enum values."""

    def test_query_type_values(self):
        """Test all QueryType enum values."""
        assert QueryType.OVERVIEW.value == "overview"
        assert QueryType.ARTICLE.value == "article"
        assert QueryType.CHAPTER.value == "chapter"
        assert QueryType.ATTACHMENT.value == "attachment"
        assert QueryType.FULL_VIEW.value == "full_view"
        assert QueryType.SEARCH.value == "search"
        assert QueryType.ASK.value == "ask"
        assert QueryType.CLARIFICATION.value == "clarification"
        assert QueryType.ERROR.value == "error"
