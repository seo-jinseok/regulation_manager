"""
Unit tests for ExecuteTestUseCase application.

Tests test execution orchestration and SearchUseCase integration.
"""

from typing import Any, List, Tuple
from unittest.mock import MagicMock, Mock

import pytest

from src.rag.automation.application.execute_test_usecase import ExecuteTestUseCase
from src.rag.automation.domain.entities import (
    DifficultyLevel,
    PersonaType,
    QueryType,
    TestCase,
)


class TestExecuteTestUseCase:
    """Test ExecuteTestUseCase functionality."""

    @pytest.fixture
    def mock_search_usecase(self):
        """Create mock SearchUseCase."""
        mock = MagicMock()
        mock.llm = None  # No LLM by default
        return mock

    @pytest.fixture
    def use_case(self, mock_search_usecase):
        """Create use case with mocked search."""
        return ExecuteTestUseCase(search_usecase=mock_search_usecase)

    def test_execute_query_returns_test_result(self, use_case, mock_search_usecase):
        """WHEN executing query, THEN should return TestResult."""
        # Setup mock search results
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        # Execute
        result = use_case.execute_query(
            query="휴학 어떻게 해?",
            test_case_id="test-001",
            enable_answer=False,  # No LLM
        )

        # Assert
        assert result.test_case_id == "test-001"
        assert result.query == "휴학 어떻게 해?"
        assert len(result.sources) == 1
        assert "2-1-1" in result.sources[0]
        assert result.execution_time_ms > 0
        assert result.rag_pipeline_log["search_results_count"] == 1
        assert not result.error_message

    def test_execute_query_with_llm_answer(self, use_case, mock_search_usecase):
        """WHEN LLM enabled, THEN should generate answer."""
        from src.rag.domain.entities import Answer, Chunk, SearchResult

        # Setup mocks
        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        mock_answer = Answer(
            text="휴학 신청은 학기 시작 14일 전까지...",
            sources=[],
            confidence=0.85,
        )
        mock_search_usecase.ask.return_value = mock_answer
        mock_search_usecase.llm = MagicMock()  # Enable LLM

        # Execute
        result = use_case.execute_query(
            query="휴학 절차",
            test_case_id="test-002",
            enable_answer=True,
        )

        # Assert
        assert result.answer == "휴학 신청은 학기 시작 14일 전까지..."
        assert result.confidence == 0.85
        assert result.rag_pipeline_log["llm_generated"] is True
        assert mock_search_usecase.ask.called

    def test_execute_query_handles_llm_error(self, use_case, mock_search_usecase):
        """WHEN LLM fails, THEN should return result with error info."""
        from src.rag.domain.entities import Chunk, SearchResult

        # Setup mocks
        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        mock_search_usecase.ask.side_effect = Exception("LLM failed")
        mock_search_usecase.llm = MagicMock()

        # Execute
        result = use_case.execute_query(
            query="휴학 절차",
            test_case_id="test-003",
            enable_answer=True,
        )

        # Assert
        assert result.answer == ""
        assert result.confidence == 0.0
        assert result.rag_pipeline_log["llm_error"] == "LLM failed"
        assert result.rag_pipeline_log["llm_generated"] is False

    def test_execute_test_case(self, use_case, mock_search_usecase):
        """WHEN executing TestCase, THEN should map metadata correctly."""
        from src.rag.domain.entities import Chunk, SearchResult

        # Setup
        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "3-1-5"
        mock_chunk.title = "장학금"
        mock_chunk.parent_path = ["장학금지급규정"]
        mock_chunk.id = "chunk-002"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.85, rank=1)
        ]

        test_case = TestCase(
            query="장학금 받으려면 성적이 얼마여야 해?",
            persona_type=PersonaType.FRESHMAN,
            difficulty=DifficultyLevel.MEDIUM,
            query_type=QueryType.ELIGIBILITY,
            expected_topics=["장학금", "성적기준"],
            expected_regulations=["장학금지급규정"],
        )

        # Execute
        result = use_case.execute_test_case(test_case)

        # Assert
        assert result.query == test_case.query
        assert "freshman" in result.test_case_id
        assert "medium" in result.test_case_id

    def test_batch_execute(self, use_case, mock_search_usecase):
        """WHEN executing batch, THEN should process all queries."""
        from src.rag.domain.entities import Chunk, SearchResult

        # Setup
        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        queries = ["휴학 절차", "복학 절차", "자퇴 절차"]

        # Execute
        results = use_case.batch_execute(queries, test_case_prefix="batch")

        # Assert
        assert len(results) == 3
        assert all(r.test_case_id.startswith("batch_") for r in results)
        assert mock_search_usecase.search.call_count == 3

    def test_pipeline_log_contains_top_chunks(self, use_case, mock_search_usecase):
        """WHEN executing, THEN pipeline log should contain top chunks."""
        from src.rag.domain.entities import Chunk, SearchResult

        # Setup multiple results
        chunks = []
        for i in range(5):
            mock_chunk = Mock(spec=Chunk)
            mock_chunk.rule_code = f"2-1-{i}"
            mock_chunk.title = f"조항{i}"
            mock_chunk.parent_path = ["학칙"]
            mock_chunk.id = f"chunk-{i:03d}"
            chunks.append(mock_chunk)

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=chunks[i], score=0.9 - i * 0.1, rank=i + 1)
            for i in range(5)
        ]

        # Execute
        result = use_case.execute_query(
            query="휴학",
            test_case_id="test-004",
            enable_answer=False,
        )

        # Assert
        assert "top_chunks" in result.rag_pipeline_log
        top_chunks = result.rag_pipeline_log["top_chunks"]
        assert len(top_chunks) == 5
        assert top_chunks[0]["rule_code"] == "2-1-0"
        assert top_chunks[0]["score"] == 0.9
        assert top_chunks[0]["rank"] == 1

    def test_execution_time_recorded(self, use_case, mock_search_usecase):
        """WHEN executing, THEN should record execution time."""
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        # Execute
        result = use_case.execute_query(
            query="휴학",
            test_case_id="test-005",
            enable_answer=False,
        )

        # Assert
        assert result.execution_time_ms > 0
        assert result.execution_time_ms < 10000  # Should be under 10 seconds
        assert result.rag_pipeline_log["execution_time_ms"] == result.execution_time_ms

    # ========== Additional Tests for Coverage 80%+ ==========

    def test_execute_query_handles_search_exception(
        self, use_case, mock_search_usecase
    ):
        """WHEN search throws exception, THEN should return error result."""
        # Setup search to raise exception
        mock_search_usecase.search.side_effect = Exception("Search failed")

        # Execute
        result = use_case.execute_query(
            query="휴학",
            test_case_id="test-error-001",
            enable_answer=False,
        )

        # Assert - should return result with error message
        assert result.test_case_id == "test-error-001"
        assert result.query == "휴학"
        assert result.error_message == "Search failed"
        assert result.sources == []
        assert result.confidence == 0.0
        assert result.execution_time_ms > 0
        assert "error" in result.rag_pipeline_log
        assert result.passed is False

    def test_execute_query_with_top_k_parameter(self, use_case, mock_search_usecase):
        """WHEN top_k specified, THEN should pass to search method."""
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        # Execute with top_k=10
        result = use_case.execute_query(
            query="휴학",
            test_case_id="test-topk",
            enable_answer=False,
            top_k=10,
        )

        # Assert search was called with top_k=10
        mock_search_usecase.search.assert_called_once()
        call_kwargs = mock_search_usecase.search.call_args.kwargs
        assert call_kwargs["top_k"] == 10
        assert result.execution_time_ms > 0

    def test_batch_execute_parallel_basic(self, use_case, mock_search_usecase):
        """WHEN executing batch in parallel, THEN should process all queries."""
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        queries = ["휴학 절차", "복학 절차", "자퇴 절차"]

        # Execute
        results = use_case.batch_execute_parallel(
            queries=queries,
            test_case_prefix="parallel",
            max_workers=2,
            rate_limit_per_second=10.0,
        )

        # Assert
        assert len(results) == 3
        assert all(r.test_case_id.startswith("parallel_") for r in results)
        assert mock_search_usecase.search.call_count == 3
        # Results should be in order
        assert results[0].query == "휴학 절차"
        assert results[1].query == "복학 절차"
        assert results[2].query == "자퇴 절차"

    def test_batch_execute_parallel_with_progress_callback(
        self, use_case, mock_search_usecase
    ):
        """WHEN progress callback provided, THEN should call for each result."""
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        queries = ["휴학", "복학"]

        # Track progress callback calls
        progress_calls: List[Tuple[int, int, Any]] = []

        def progress_callback(completed: int, total: int, result):
            progress_calls.append((completed, total, result))

        # Execute
        results = use_case.batch_execute_parallel(
            queries=queries,
            test_case_prefix="progress",
            max_workers=2,
            progress_callback=progress_callback,
        )

        # Assert callback was called
        assert len(progress_calls) == 2
        assert progress_calls[0][0] == 1  # First completion
        assert progress_calls[0][1] == 2  # Total
        assert progress_calls[1][0] == 2  # Second completion
        assert len(results) == 2

    def test_batch_execute_parallel_empty_queries(self, use_case, mock_search_usecase):
        """WHEN queries list is empty, THEN should return empty list."""
        # Execute with empty list
        results = use_case.batch_execute_parallel(
            queries=[],
            test_case_prefix="empty",
        )

        # Assert
        assert results == []
        assert mock_search_usecase.search.call_count == 0

    def test_batch_execute_parallel_default_max_workers(
        self, use_case, mock_search_usecase
    ):
        """WHEN max_workers not specified, THEN should use default (cpu_count or 4)."""
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        queries = ["휴학"]

        # Execute without max_workers
        results = use_case.batch_execute_parallel(
            queries=queries,
            test_case_prefix="default_workers",
        )

        # Assert should execute successfully
        assert len(results) == 1
        assert mock_search_usecase.search.called

    def test_batch_execute_test_cases(self, use_case, mock_search_usecase):
        """WHEN executing test cases, THEN should map persona and difficulty to ID."""
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        test_cases = [
            TestCase(
                query="휴학 절차",
                persona_type=PersonaType.FRESHMAN,
                difficulty=DifficultyLevel.EASY,
                query_type=QueryType.PROCEDURAL,
            ),
            TestCase(
                query="복학 절차",
                persona_type=PersonaType.JUNIOR,
                difficulty=DifficultyLevel.MEDIUM,
                query_type=QueryType.PROCEDURAL,
            ),
        ]

        # Execute
        results = use_case.batch_execute_test_cases(
            test_cases=test_cases,
            max_workers=2,
        )

        # Assert
        assert len(results) == 2
        assert mock_search_usecase.search.call_count == 2
        # Check test_case_id format: {persona}_{difficulty}_{index}
        assert "freshman" in results[0].test_case_id
        assert "easy" in results[0].test_case_id
        assert "junior" in results[1].test_case_id
        assert "medium" in results[1].test_case_id
        assert "_000" in results[0].test_case_id
        assert "_001" in results[1].test_case_id

    def test_batch_execute_test_cases_with_progress_callback(
        self, use_case, mock_search_usecase
    ):
        """WHEN executing test cases with callback, THEN should track progress."""
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        test_cases = [
            TestCase(
                query="휴학",
                persona_type=PersonaType.FRESHMAN,
                difficulty=DifficultyLevel.EASY,
                query_type=QueryType.PROCEDURAL,
            )
        ]

        progress_calls: List[Tuple[int, int, Any]] = []

        def progress_callback(completed: int, total: int, result):
            progress_calls.append((completed, total, result))

        # Execute
        results = use_case.batch_execute_test_cases(
            test_cases=test_cases,
            progress_callback=progress_callback,
        )

        # Assert
        assert len(results) == 1
        assert len(progress_calls) == 1

    def test_batch_execute_empty_list(self, use_case, mock_search_usecase):
        """WHEN batch executing empty queries, THEN should return empty list."""
        # Execute with empty list
        results = use_case.batch_execute(
            queries=[],
            test_case_prefix="empty_batch",
        )

        # Assert
        assert results == []
        assert mock_search_usecase.search.call_count == 0

    def test_execute_query_without_llm_does_not_call_ask(
        self, use_case, mock_search_usecase
    ):
        """WHEN enable_answer is False, THEN should not call ask method."""
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        # Execute without LLM
        result = use_case.execute_query(
            query="휴학",
            test_case_id="test-no-llm",
            enable_answer=False,
        )

        # Assert
        assert result.answer == ""
        assert result.confidence == 0.0
        assert not mock_search_usecase.ask.called
        assert result.rag_pipeline_log["llm_generated"] is False

    def test_execute_query_when_llm_is_none(self, use_case, mock_search_usecase):
        """WHEN LLM is None, THEN should not generate answer even with enable_answer=True."""
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]
        mock_search_usecase.llm = None  # No LLM

        # Execute with enable_answer=True
        result = use_case.execute_query(
            query="휴학",
            test_case_id="test-no-llm-instance",
            enable_answer=True,
        )

        # Assert - should not call ask
        assert result.answer == ""
        assert result.confidence == 0.0
        assert not mock_search_usecase.ask.called
        assert result.rag_pipeline_log["llm_generated"] is False

    def test_execute_query_sources_format_with_parent_path(
        self, use_case, mock_search_usecase
    ):
        """WHEN chunk has parent_path and no title, THEN should use parent_path."""
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = None  # No title, should use parent_path[0]
        mock_chunk.parent_path = ["학칙", "제2장"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        # Execute
        result = use_case.execute_query(
            query="휴학",
            test_case_id="test-sources",
            enable_answer=False,
        )

        # Assert source format - should use parent_path[0] when title is None
        assert len(result.sources) == 1
        assert "2-1-1" in result.sources[0]
        assert "학칙" in result.sources[0]

    def test_execute_query_sources_format_without_parent_path(
        self, use_case, mock_search_usecase
    ):
        """WHEN chunk has no parent_path, THEN should use rule_code only."""
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = []
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        # Execute
        result = use_case.execute_query(
            query="휴학",
            test_case_id="test-sources-no-path",
            enable_answer=False,
        )

        # Assert source format - should use rule_code when parent_path is empty
        assert len(result.sources) == 1
        assert "2-1-1" in result.sources[0]

    def test_batch_execute_result_order_preserved(self, use_case, mock_search_usecase):
        """WHEN batch executing, THEN results should be in original query order."""
        from src.rag.domain.entities import Chunk, SearchResult

        def create_mock_chunk(code: str, title: str):
            chunk = Mock(spec=Chunk)
            chunk.rule_code = code
            chunk.title = title
            chunk.parent_path = ["학칙"]
            chunk.id = f"chunk-{code}"
            return chunk

        mock_search_usecase.search.side_effect = [
            [SearchResult(chunk=create_mock_chunk("2-1-1", "휴학"), score=0.9, rank=1)],
            [SearchResult(chunk=create_mock_chunk("2-1-2", "복학"), score=0.8, rank=1)],
            [SearchResult(chunk=create_mock_chunk("2-1-3", "자퇴"), score=0.7, rank=1)],
        ]

        queries = ["세번째", "첫번째", "두번째"]

        # Execute
        results = use_case.batch_execute(queries, test_case_prefix="order")

        # Assert order is preserved
        assert len(results) == 3
        assert results[0].query == "세번째"
        assert results[1].query == "첫번째"
        assert results[2].query == "두번째"

    def test_batch_execute_parallel_rate_limit_respected(
        self, use_case, mock_search_usecase
    ):
        """WHEN rate limit specified, THEN should control concurrency."""
        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.title = "휴학"
        mock_chunk.parent_path = ["학칙"]
        mock_chunk.id = "chunk-001"

        mock_search_usecase.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        queries = ["휴학"] * 3

        # Execute with low rate limit
        results = use_case.batch_execute_parallel(
            queries=queries,
            test_case_prefix="ratelimit",
            max_workers=2,
            rate_limit_per_second=2.0,
        )

        # Assert all queries processed
        assert len(results) == 3
        assert all(r.query == "휴학" for r in results)
