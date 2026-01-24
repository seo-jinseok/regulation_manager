"""
Unit tests for ExecuteTestUseCase application.

Tests test execution orchestration and SearchUseCase integration.
"""

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
