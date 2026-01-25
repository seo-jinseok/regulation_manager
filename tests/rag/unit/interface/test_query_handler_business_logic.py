"""
Comprehensive tests for QueryHandler business logic methods.

Focuses on testable business logic methods without extensive mocking.
"""

from src.rag.interface.query_handler import (
    QueryContext,
    QueryOptions,
    QueryResult,
    QueryType,
    detect_deletion_warning,
)


class TestDetectDeletionWarning:
    """Test deletion warning detection logic."""

    def test_no_warning(self):
        assert detect_deletion_warning("Normal text content") is None
        assert detect_deletion_warning("제1조 규정 내용") is None

    def test_deletion_with_date_yyyy_mm_dd(self):
        # Pattern: 삭제 (YYYY.MM.DD)
        result = detect_deletion_warning("본 조는 삭제 (2024.06.30)")
        assert "2024년 06월 30일" in result  # Note: leading zeros
        assert "삭제" in result

    def test_deletion_with_date_yyyy_mm(self):
        # Pattern: 삭제 (YYYY.MM)
        result = detect_deletion_warning("본 항은 삭제 (2024.03)")
        assert "2024년 03월" in result  # Note: leading zeros
        assert "삭제" in result

    def test_deletion_with_date_yyyy_only(self):
        # Pattern: 삭제 (YYYY)
        result = detect_deletion_warning("삭제(2024)")
        assert "2024년" in result
        assert "삭제" in result

    def test_deletion_without_date(self):
        # Pattern: 본 조는 삭제
        result = detect_deletion_warning("본 조는 삭제")
        assert "삭제" in result
        assert "최신 규정을 확인하세요" in result

    def test_abolishment_with_date(self):
        # Pattern: 폐지 (YYYY.MM.DD)
        result = detect_deletion_warning("폐지 (2023.12.31)")
        assert "2023년 12월 31일" in result
        assert "폐지" in result

    def test_deletion_in_brackets(self):
        # Pattern: (삭제)
        result = detect_deletion_warning("본 조(삭제)")
        assert "삭제" in result

    def test_deletion_in_brackets_square(self):
        # Pattern: [삭제]
        result = detect_deletion_warning("본 항[삭제]")
        assert "삭제" in result

    def test_item_deletion(self):
        # Pattern: 본 항은 삭제
        result = detect_deletion_warning("본 항은 삭제")
        assert "삭제" in result


class TestQueryResult:
    """Test QueryResult dataclass."""

    def test_create_default_result(self):
        result = QueryResult(type=QueryType.SEARCH, success=True)
        assert result.type == QueryType.SEARCH
        assert result.success is True
        assert result.content == ""
        assert result.data == {}
        assert result.state_update == {}
        assert result.suggestions == []

    def test_create_result_with_content(self):
        result = QueryResult(
            type=QueryType.ASK,
            success=True,
            content="Answer text",
            data={"answer": "value"},
        )
        assert result.content == "Answer text"
        assert result.data["answer"] == "value"

    def test_create_clarification_result(self):
        result = QueryResult(
            type=QueryType.CLARIFICATION,
            success=True,
            clarification_type="audience",
            clarification_options=["교수", "학생"],
            content="Select audience",
        )
        assert result.clarification_type == "audience"
        assert result.suggestions == []

    def test_create_error_result(self):
        result = QueryResult(
            type=QueryType.ERROR, success=False, content="Error message"
        )
        assert result.type == QueryType.ERROR
        assert result.success is False


class TestQueryContext:
    """Test QueryContext dataclass."""

    def test_default_context(self):
        context = QueryContext()
        assert context.state == {}
        assert context.history == []
        assert context.interactive is False
        assert context.last_regulation is None
        assert context.last_rule_code is None

    def test_context_with_history(self):
        history = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        context = QueryContext(history=history)
        assert len(context.history) == 2
        assert context.history[0]["role"] == "user"

    def test_context_with_last_regulation(self):
        context = QueryContext(last_regulation="교원인사규정", last_rule_code="1-1-1")
        assert context.last_regulation == "교원인사규정"
        assert context.last_rule_code == "1-1-1"


class TestQueryOptions:
    """Test QueryOptions dataclass."""

    def test_default_options(self):
        options = QueryOptions()
        assert options.top_k == 5
        assert options.force_mode is None
        assert options.include_abolished is False
        assert options.use_rerank is True
        assert options.audience_override is None
        assert options.show_debug is False
        assert options.use_function_gemma is False

    def test_custom_options(self):
        options = QueryOptions(
            top_k=10,
            force_mode="search",
            include_abolished=True,
            use_rerank=False,
            show_debug=True,
        )
        assert options.top_k == 10
        assert options.force_mode == "search"
        assert options.include_abolished is True
        assert options.use_rerank is False
        assert options.show_debug is True


class TestQueryType:
    """Test QueryType enum."""

    def test_all_types_exist(self):
        assert QueryType.OVERVIEW.value == "overview"
        assert QueryType.ARTICLE.value == "article"
        assert QueryType.CHAPTER.value == "chapter"
        assert QueryType.ATTACHMENT.value == "attachment"
        assert QueryType.FULL_VIEW.value == "full_view"
        assert QueryType.SEARCH.value == "search"
        assert QueryType.ASK.value == "ask"
        assert QueryType.CLARIFICATION.value == "clarification"
        assert QueryType.ERROR.value == "error"


class TestDeletionPatternMatching:
    """Test various deletion warning patterns."""

    def test_pattern_exact_format_yyyy_mm_dd(self):
        # Pattern: 삭제 (YYYY.MM.DD)
        result = detect_deletion_warning("본 조는 삭제 (2024.06.30)")
        assert result is not None
        assert "2024년 06월 30일" in result

    def test_pattern_format_with_angle_bracket(self):
        # Pattern: 삭제<YYYY.MM.DD> - Using < not 〈
        result = detect_deletion_warning("본 조는 삭제<2024.06.30>")
        assert result is not None
        assert "2024년 06월 30일" in result

    def test_pattern_with_square_bracket_date(self):
        # Pattern: 삭제[YYYY.MM.DD]
        result = detect_deletion_warning("본 조는 삭제[2024.06.30]")
        assert result is not None
        assert "2024년 06월 30일" in result

    def test_pattern_yyyy_mm_only(self):
        # Pattern: 삭제 (YYYY.MM)
        result = detect_deletion_warning("본 항은 삭제 (2024.03)")
        assert result is not None
        assert "2024년 03월" in result

    def test_pattern_yyyy_only(self):
        # Pattern: 삭제 (YYYY)
        result = detect_deletion_warning("삭제 (2024)")
        assert result is not None
        assert "2024년" in result

    def test_pattern_slashes(self):
        # Pattern: 삭제 (YYYY/MM/DD)
        result = detect_deletion_warning("본 조는 삭제 (2024/06/30)")
        assert result is not None
        assert "2024년 06월 30일" in result

    def test_pattern_hyphens(self):
        # Pattern: 삭제 (YYYY-MM-DD)
        result = detect_deletion_warning("본 조는 삭제 (2024-06-30)")
        assert result is not None
        assert "2024년 06월 30일" in result

    def test_abolishment_pattern(self):
        # Pattern: 폐지 (YYYY.MM.DD)
        result = detect_deletion_warning("폐지 (2023.12.31)")
        assert result is not None
        assert "2023년 12월 31일" in result
        assert "폐지" in result

    def test_pattern_square_brackets(self):
        """[삭제] pattern"""
        result = detect_deletion_warning("본 조 [삭제]")
        assert result is not None

    def test_combined_text_pattern(self):
        """'본 조는 삭제' pattern"""
        result = detect_deletion_warning("본 조는 삭제되었습니다.")
        # This pattern specifically looks for "본 조는? 삭제"
        assert "삭제" in result or result is None

    def test_pattern_item_deletion(self):
        """'본 항은? 삭제' pattern"""
        result = detect_deletion_warning("본 항은 삭제")
        assert result is not None


class TestDeletionWarningEdgeCases:
    """Test edge cases for deletion warning detection."""

    def test_empty_string(self):
        assert detect_deletion_warning("") is None

    def test_only_whitespace(self):
        assert detect_deletion_warning("   ") is None

    def test_deletion_in_middle_of_text(self):
        result = detect_deletion_warning("규정 내용입니다 삭제(2024) 계속됩니다")
        assert result is not None
        assert "삭제" in result

    def test_multiple_dates_different_patterns(self):
        result = detect_deletion_warning("2023년 12월 31일 삭제(2024.03)")
        # Should match one of the patterns
        assert result is not None

    def test_no_match_similar_text(self):
        # Text that mentions deletion but not in the warning format
        assert detect_deletion_warning("삭제된 조항입니다") is None
        assert detect_deletion_warning("삭제해주세요") is None


class TestQueryTypeIntegration:
    """Integration tests for QueryResult with QueryType."""

    def test_error_result_format(self):
        result = QueryResult(
            type=QueryType.ERROR, success=False, content="⚠️ Error message"
        )
        assert "Error message" in result.content

    def test_search_result_with_data(self):
        result = QueryResult(
            type=QueryType.SEARCH,
            success=True,
            content="Results",
            data={"query": "test", "results": []},
        )
        assert result.data["query"] == "test"
        assert result.data["results"] == []

    def test_ask_result_with_sources(self):
        result = QueryResult(
            type=QueryType.ASK,
            success=True,
            content="Answer",
            data={"sources": [{"title": "Regulation"}]},
        )
        assert "sources" in result.data
        assert len(result.data["sources"]) == 1
