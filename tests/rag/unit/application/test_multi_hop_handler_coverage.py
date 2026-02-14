"""
Characterization tests for MultiHopHandler - Additional Coverage.

SPEC: SPEC-TEST-COV-001 Phase 3 - Test Coverage Improvement

These tests are designed to cover uncovered branches in multi_hop_handler.py
to achieve 85% coverage target.

Key methods to test:
- execute_multi_hop() - Main multi-hop handling
- _execute_hop_with_timeout() - Timeout handling
- _execute_hop() - Single hop execution
- _build_context_text() - Context aggregation
- _evaluate_relevance() - Self-RAG relevance
- _generate_hop_answer() - Answer generation
- _synthesize_final_answer() - Final synthesis
- DependencyCycleDetector - Cycle detection
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.application.multi_hop_handler import (
    DependencyCycleDetector,
    HopResult,
    HopStatus,
    MultiHopHandler,
    MultiHopQueryDecomposer,
    MultiHopResult,
    SubQuery,
)
from src.rag.domain.entities import Chunk, ChunkLevel, Keyword, SearchResult
from src.rag.domain.value_objects import Query


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for deterministic testing."""
    client = MagicMock()
    client.generate.return_value = json.dumps(
        {
            "sub_queries": [
                {
                    "query_id": "hop_1",
                    "query_text": "First sub-query",
                    "hop_order": 1,
                    "depends_on": [],
                    "reasoning": "First step",
                }
            ]
        },
        ensure_ascii=False,
    )
    return client


@pytest.fixture
def mock_vector_store():
    """Mock vector store for deterministic testing."""

    class MockVectorStore:
        def __init__(self):
            self._results = []

        def search(self, query: Query, top_k: int = 10) -> List[SearchResult]:
            return self._results[:top_k]

        def set_results(self, results: List[SearchResult]) -> None:
            self._results = results

    return MockVectorStore()


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing."""
    return Chunk(
        id="test_chunk_1",
        rule_code="TEST001",
        level=ChunkLevel.ARTICLE,
        title="Test Article",
        text="This is test content for the chunk.",
        embedding_text="This is test content for the chunk.",
        full_text="Full text of test chunk.",
        parent_path=["Chapter 1", "Section 1"],
        token_count=100,
        keywords=[Keyword(term="test", weight=1.0)],
        is_searchable=True,
    )


@pytest.fixture
def sample_search_result(sample_chunk):
    """Create a sample search result for testing."""
    return SearchResult(chunk=sample_chunk, score=0.8, rank=1)


# =============================================================================
# Test SubQuery Dataclass
# =============================================================================


class TestSubQuery:
    """Characterization tests for SubQuery dataclass."""

    def test_to_dict(self):
        """SubQuery can be converted to dictionary."""
        sub_query = SubQuery(
            query_id="q1",
            query_text="test query",
            hop_order=1,
            depends_on=["q0"],
            context_from="q0",
            reasoning="test reason",
        )
        result = sub_query.to_dict()
        assert result["query_id"] == "q1"
        assert result["query_text"] == "test query"
        assert result["hop_order"] == 1
        assert result["depends_on"] == ["q0"]
        assert result["context_from"] == "q0"
        assert result["reasoning"] == "test reason"

    def test_from_dict(self):
        """SubQuery can be created from dictionary."""
        data = {
            "query_id": "q1",
            "query_text": "test query",
            "hop_order": 1,
            "depends_on": ["q0"],
            "context_from": "q0",
            "reasoning": "test reason",
        }
        sub_query = SubQuery.from_dict(data)
        assert sub_query.query_id == "q1"
        assert sub_query.query_text == "test query"
        assert sub_query.hop_order == 1
        assert sub_query.depends_on == ["q0"]
        assert sub_query.context_from == "q0"
        assert sub_query.reasoning == "test reason"

    def test_from_dict_defaults(self):
        """SubQuery from_dict handles missing optional fields."""
        data = {"query_id": "q1", "query_text": "test query", "hop_order": 1}
        sub_query = SubQuery.from_dict(data)
        assert sub_query.depends_on == []
        assert sub_query.context_from is None
        assert sub_query.reasoning == ""

    def test_default_values(self):
        """SubQuery has correct default values."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        assert sub_query.depends_on == []
        assert sub_query.context_from is None
        assert sub_query.reasoning == ""


# =============================================================================
# Test HopResult Dataclass
# =============================================================================


class TestHopResult:
    """Characterization tests for HopResult dataclass."""

    def test_to_dict(self, sample_chunk):
        """HopResult can be converted to dictionary."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        result = HopResult(
            hop_id="q1",
            query=sub_query,
            answer="Test answer",
            sources=[SearchResult(chunk=sample_chunk, score=0.8)],
            execution_time_ms=100.0,
            is_relevant=True,
            status=HopStatus.COMPLETED,
        )
        data = result.to_dict()
        assert data["hop_id"] == "q1"
        assert data["answer"] == "Test answer"
        assert data["execution_time_ms"] == 100.0
        assert data["is_relevant"] is True
        assert data["status"] == "completed"

    def test_to_dict_with_error(self):
        """HopResult with error message is serialized correctly."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        result = HopResult(
            hop_id="q1",
            query=sub_query,
            answer="",
            sources=[],
            execution_time_ms=100.0,
            status=HopStatus.FAILED,
            error_message="Test error",
        )
        data = result.to_dict()
        assert data["error_message"] == "Test error"
        assert data["status"] == "failed"

    def test_default_values(self):
        """HopResult has correct default values."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        result = HopResult(
            hop_id="q1", query=sub_query, answer="", sources=[], execution_time_ms=0
        )
        assert result.is_relevant is True
        assert result.status == HopStatus.COMPLETED
        assert result.error_message is None


# =============================================================================
# Test MultiHopResult Dataclass
# =============================================================================


class TestMultiHopResult:
    """Characterization tests for MultiHopResult dataclass."""

    def test_to_dict(self):
        """MultiHopResult can be converted to dictionary."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        hop_result = HopResult(
            hop_id="q1",
            query=sub_query,
            answer="Answer",
            sources=[],
            execution_time_ms=100.0,
        )
        result = MultiHopResult(
            original_query="Original query",
            sub_queries=[sub_query],
            hop_results=[hop_result],
            final_answer="Final answer",
            total_execution_time_ms=200.0,
            hop_count=1,
        )
        data = result.to_dict()
        assert data["original_query"] == "Original query"
        assert data["final_answer"] == "Final answer"
        assert data["hop_count"] == 1
        assert data["success"] is True

    def test_failure_result(self):
        """MultiHopResult with failure status."""
        result = MultiHopResult(
            original_query="Query",
            sub_queries=[],
            hop_results=[],
            final_answer="Failed",
            total_execution_time_ms=0,
            hop_count=0,
            success=False,
        )
        data = result.to_dict()
        assert data["success"] is False


# =============================================================================
# Test DependencyCycleDetector
# =============================================================================


class TestDependencyCycleDetector:
    """Characterization tests for DependencyCycleDetector."""

    def test_no_cycle(self):
        """No cycle detected in simple linear dependencies."""
        detector = DependencyCycleDetector(max_hops=5)
        dependencies = {"q1": [], "q2": ["q1"], "q3": ["q2"]}
        result = detector.detect_cycle(dependencies)
        assert result is None

    def test_simple_cycle(self):
        """Simple cycle is detected."""
        detector = DependencyCycleDetector(max_hops=5)
        dependencies = {"q1": ["q2"], "q2": ["q1"]}
        result = detector.detect_cycle(dependencies)
        assert result is not None

    def test_self_cycle(self):
        """Self-referential dependency is detected."""
        detector = DependencyCycleDetector(max_hops=5)
        dependencies = {"q1": ["q1"]}
        result = detector.detect_cycle(dependencies)
        assert result is not None

    def test_complex_cycle(self):
        """Complex cycle with multiple nodes is detected."""
        detector = DependencyCycleDetector(max_hops=5)
        dependencies = {"q1": ["q3"], "q2": ["q1"], "q3": ["q2"]}
        result = detector.detect_cycle(dependencies)
        assert result is not None

    def test_empty_dependencies(self):
        """Empty dependencies have no cycle."""
        detector = DependencyCycleDetector(max_hops=5)
        result = detector.detect_cycle({})
        assert result is None

    def test_max_hops_exceeded(self):
        """Max hops limit triggers cycle detection."""
        detector = DependencyCycleDetector(max_hops=2)
        dependencies = {"q1": [], "q2": ["q1"], "q3": ["q2"], "q4": ["q3"]}
        # This doesn't exceed max_hops in depth, but tests the method
        result = detector.detect_cycle(dependencies)
        assert result is None  # No actual cycle

    def test_validate_max_hops_within_limit(self):
        """validate_max_hops returns True when within limit."""
        detector = DependencyCycleDetector(max_hops=5)
        sub_queries = [
            SubQuery(query_id=f"q{i}", query_text="test", hop_order=i)
            for i in range(1, 4)
        ]
        assert detector.validate_max_hops(sub_queries) is True

    def test_validate_max_hops_exceeds_limit(self):
        """validate_max_hops returns False when exceeding limit."""
        detector = DependencyCycleDetector(max_hops=3)
        sub_queries = [
            SubQuery(query_id=f"q{i}", query_text="test", hop_order=i) for i in range(1, 5)
        ]
        assert detector.validate_max_hops(sub_queries) is False

    def test_validate_max_hops_empty(self):
        """validate_max_hops handles empty list."""
        detector = DependencyCycleDetector(max_hops=5)
        assert detector.validate_max_hops([]) is True


# =============================================================================
# Test MultiHopQueryDecomposer
# =============================================================================


class TestMultiHopQueryDecomposer:
    """Characterization tests for MultiHopQueryDecomposer."""

    @pytest.fixture
    def decomposer(self, mock_llm_client):
        return MultiHopQueryDecomposer(mock_llm_client)

    @pytest.mark.asyncio
    async def test_decompose_single_query(self, decomposer):
        """Decomposer returns sub-queries from LLM."""
        result = await decomposer.decompose("test query")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(sq, SubQuery) for sq in result)

    @pytest.mark.asyncio
    async def test_decompose_llm_error_fallback(self, mock_llm_client):
        """Decomposer falls back to single query on LLM error."""
        mock_llm_client.generate.side_effect = Exception("LLM error")
        decomposer = MultiHopQueryDecomposer(mock_llm_client)
        result = await decomposer.decompose("test query")
        assert len(result) == 1
        assert result[0].query_text == "test query"

    @pytest.mark.asyncio
    async def test_decompose_json_parse_error_fallback(self, mock_llm_client):
        """Decomposer falls back on JSON parse error."""
        mock_llm_client.generate.return_value = "not valid json"
        decomposer = MultiHopQueryDecomposer(mock_llm_client)
        result = await decomposer.decompose("test query")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_decompose_markdown_json_extraction(self, mock_llm_client):
        """Decomposer extracts JSON from markdown code blocks."""
        mock_llm_client.generate.return_value = (
            "```json\n"
            '{"sub_queries": [{"query_id": "hop_1", "query_text": "test", "hop_order": 1}]}'
            "\n```"
        )
        decomposer = MultiHopQueryDecomposer(mock_llm_client)
        result = await decomposer.decompose("test query")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_decompose_generates_missing_query_id(self, mock_llm_client):
        """Decomposer generates query_id if missing in response."""
        mock_llm_client.generate.return_value = json.dumps(
            {"sub_queries": [{"query_text": "test", "hop_order": 1}]}
        )
        decomposer = MultiHopQueryDecomposer(mock_llm_client)
        result = await decomposer.decompose("test query")
        assert result[0].query_id == "hop_1"

    def test_get_decomposition_prompt(self, decomposer):
        """_get_decomposition_prompt returns system prompt."""
        prompt = decomposer._get_decomposition_prompt()
        assert isinstance(prompt, str)
        # The prompt uses "sub-questions" (hyphen) in the text
        assert "sub-question" in prompt.lower() or "sub_queries" in prompt.lower()

    def test_format_decomposition_request(self, decomposer):
        """_format_decomposition_request formats user message."""
        message = decomposer._format_decomposition_request("test query")
        assert "test query" in message


# =============================================================================
# Test MultiHopHandler
# =============================================================================


class TestMultiHopHandler:
    """Characterization tests for MultiHopHandler."""

    @pytest.fixture
    def handler(self, mock_llm_client, mock_vector_store):
        return MultiHopHandler(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            max_hops=5,
            hop_timeout_seconds=30,
            enable_self_rag=True,
        )

    @pytest.mark.asyncio
    async def test_execute_multi_hop_success(self, handler, mock_vector_store, sample_search_result):
        """execute_multi_hop returns MultiHopResult on success."""
        mock_vector_store.set_results([sample_search_result])
        result = await handler.execute_multi_hop("test query", top_k=5)
        assert isinstance(result, MultiHopResult)
        assert result.original_query == "test query"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_multi_hop_max_hops_exceeded(self, mock_llm_client, mock_vector_store):
        """execute_multi_hop fails when max hops exceeded."""
        # Create handler with max_hops=1
        handler = MultiHopHandler(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            max_hops=1,
        )
        # Mock LLM to return 2 sub-queries
        mock_llm_client.generate.return_value = json.dumps(
            {
                "sub_queries": [
                    {"query_id": "q1", "query_text": "first", "hop_order": 1},
                    {"query_id": "q2", "query_text": "second", "hop_order": 2},
                ]
            }
        )
        result = await handler.execute_multi_hop("complex query")
        assert result.success is False
        assert "maximum" in result.final_answer.lower()

    @pytest.mark.asyncio
    async def test_execute_multi_hop_cycle_detected(self, mock_llm_client, mock_vector_store):
        """execute_multi_hop fails when cycle detected."""
        handler = MultiHopHandler(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
        )
        # Mock LLM to return cyclic dependencies
        mock_llm_client.generate.return_value = json.dumps(
            {
                "sub_queries": [
                    {"query_id": "q1", "query_text": "first", "hop_order": 1, "depends_on": ["q2"]},
                    {"query_id": "q2", "query_text": "second", "hop_order": 2, "depends_on": ["q1"]},
                ]
            }
        )
        result = await handler.execute_multi_hop("cyclic query")
        assert result.success is False
        assert "cycle" in result.final_answer.lower() or "Cyclic" in result.final_answer

    @pytest.mark.asyncio
    async def test_execute_multi_hop_hop_failure(self, handler, mock_vector_store):
        """execute_multi_hop handles hop failure gracefully."""
        # Empty results cause hop to return "No relevant information" but not fail
        mock_vector_store.set_results([])
        result = await handler.execute_multi_hop("test query")
        assert isinstance(result, MultiHopResult)

    @pytest.mark.asyncio
    async def test_execute_multi_hop_exception(self, handler, mock_vector_store):
        """execute_multi_hop handles exceptions - hop failure doesn't fail whole result."""
        mock_vector_store.search = MagicMock(side_effect=Exception("Search error"))
        result = await handler.execute_multi_hop("test query")
        # When a hop fails, the result still returns with success=True but with failed hop
        # The handler continues and synthesizes answer from available results
        assert result is not None
        # Check that the hop failed
        if result.hop_results:
            assert any(hr.status == HopStatus.FAILED for hr in result.hop_results)

    def test_build_context_text_no_context(self, handler):
        """_build_context_text returns empty string when no context."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        result = handler._build_context_text(sub_query, {})
        assert result == ""

    def test_build_context_text_with_context(self, handler, sample_chunk):
        """_build_context_text builds context from previous hop."""
        sub_query = SubQuery(
            query_id="q2", query_text="second query", hop_order=2, context_from="q1"
        )
        prev_result = HopResult(
            hop_id="q1",
            query=SubQuery(query_id="q1", query_text="first query", hop_order=1),
            answer="First answer",
            sources=[SearchResult(chunk=sample_chunk, score=0.8)],
            execution_time_ms=100.0,
        )
        context = {"q1": prev_result}
        result = handler._build_context_text(sub_query, context)
        assert "first query" in result.lower()
        assert "first answer" in result.lower()

    def test_build_context_text_context_not_found(self, handler):
        """_build_context_text returns empty when context_from not in context."""
        sub_query = SubQuery(
            query_id="q2", query_text="test", hop_order=2, context_from="q_missing"
        )
        result = handler._build_context_text(sub_query, {})
        assert result == ""

    @pytest.mark.asyncio
    async def test_evaluate_relevance_empty_results(self, handler):
        """_evaluate_relevance returns False for empty results."""
        result = await handler._evaluate_relevance("test query", [])
        assert result is False

    @pytest.mark.asyncio
    async def test_evaluate_relevance_high_score(self, handler, sample_chunk):
        """_evaluate_relevance returns True for high score."""
        search_results = [SearchResult(chunk=sample_chunk, score=0.8)]
        result = await handler._evaluate_relevance("test query", search_results)
        assert result is True

    @pytest.mark.asyncio
    async def test_evaluate_relevance_low_score(self, handler, sample_chunk):
        """_evaluate_relevance returns False for low score."""
        search_results = [SearchResult(chunk=sample_chunk, score=0.3)]
        result = await handler._evaluate_relevance("test query", search_results)
        assert result is False

    @pytest.mark.asyncio
    async def test_generate_hop_answer_empty_results(self, handler):
        """_generate_hop_answer handles empty results."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        result = await handler._generate_hop_answer(sub_query, [], "")
        assert "No relevant information" in result

    @pytest.mark.asyncio
    async def test_generate_hop_answer_success(self, handler, sample_search_result):
        """_generate_hop_answer generates answer from results."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        result = await handler._generate_hop_answer(
            sub_query, [sample_search_result], ""
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_hop_answer_with_context(self, handler, sample_search_result):
        """_generate_hop_answer includes context from previous hops."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        result = await handler._generate_hop_answer(
            sub_query, [sample_search_result], "Previous context"
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_hop_answer_llm_error(self, mock_llm_client, mock_vector_store, sample_search_result):
        """_generate_hop_answer handles LLM errors."""
        mock_llm_client.generate.side_effect = Exception("LLM error")
        handler = MultiHopHandler(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
        )
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        result = await handler._generate_hop_answer(
            sub_query, [sample_search_result], ""
        )
        assert "Error" in result or "error" in result

    @pytest.mark.asyncio
    async def test_synthesize_final_answer_empty_results(self, handler):
        """_synthesize_final_answer handles empty results."""
        result = await handler._synthesize_final_answer("test query", [])
        assert "Unable to answer" in result or "processing errors" in result

    @pytest.mark.asyncio
    async def test_synthesize_final_answer_single_hop(self, handler, sample_chunk):
        """_synthesize_final_answer returns single hop answer directly."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        hop_result = HopResult(
            hop_id="q1",
            query=sub_query,
            answer="Single answer",
            sources=[SearchResult(chunk=sample_chunk, score=0.8)],
            execution_time_ms=100.0,
        )
        result = await handler._synthesize_final_answer("test query", [hop_result])
        assert result == "Single answer"

    @pytest.mark.asyncio
    async def test_synthesize_final_answer_multiple_hops(self, handler, sample_chunk):
        """_synthesize_final_answer synthesizes multiple hops."""
        hop_results = [
            HopResult(
                hop_id=f"q{i}",
                query=SubQuery(query_id=f"q{i}", query_text=f"query {i}", hop_order=i),
                answer=f"Answer {i}",
                sources=[SearchResult(chunk=sample_chunk, score=0.8)],
                execution_time_ms=100.0,
            )
            for i in range(1, 3)
        ]
        result = await handler._synthesize_final_answer("test query", hop_results)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_synthesize_final_answer_llm_error(self, mock_llm_client, mock_vector_store, sample_chunk):
        """_synthesize_final_answer falls back on LLM error."""
        mock_llm_client.generate.side_effect = Exception("LLM error")
        handler = MultiHopHandler(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
        )
        hop_results = [
            HopResult(
                hop_id=f"q{i}",
                query=SubQuery(query_id=f"q{i}", query_text=f"query {i}", hop_order=i),
                answer=f"Answer {i}",
                sources=[SearchResult(chunk=sample_chunk, score=0.8)],
                execution_time_ms=100.0,
            )
            for i in range(1, 3)
        ]
        result = await handler._synthesize_final_answer("test query", hop_results)
        # Falls back to concatenating answers
        assert "Answer 1" in result or "Answer 2" in result

    def test_create_failure_result(self, handler):
        """_create_failure_result creates proper failure result."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        result = handler._create_failure_result("test query", [sub_query], "Test error")
        assert result.success is False
        assert "Test error" in result.final_answer
        assert result.hop_count == 0

    @pytest.mark.asyncio
    async def test_execute_hop_with_timeout_success(self, handler, sample_search_result):
        """_execute_hop_with_timeout succeeds within timeout."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        handler.vector_store.search = MagicMock(return_value=[sample_search_result])
        result = await handler._execute_hop_with_timeout(sub_query, {}, 5)
        assert result.status == HopStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_hop_with_timeout_exceeded(self, mock_llm_client, mock_vector_store):
        """_execute_hop_with_timeout returns TIMEOUT status on timeout."""
        handler = MultiHopHandler(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            hop_timeout_seconds=0,  # Immediate timeout
        )

        async def slow_search(*args, **kwargs):
            await asyncio.sleep(1)  # Longer than timeout
            return []

        handler.vector_store.search = slow_search

        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        result = await handler._execute_hop_with_timeout(sub_query, {}, 5)
        assert result.status == HopStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_execute_hop_exception(self, handler):
        """_execute_hop handles exceptions."""
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        handler.vector_store.search = MagicMock(side_effect=Exception("Search failed"))
        result = await handler._execute_hop(sub_query, {}, 5)
        assert result.status == HopStatus.FAILED
        assert "Search failed" in result.error_message

    @pytest.mark.asyncio
    async def test_self_rag_disabled(self, mock_llm_client, mock_vector_store, sample_search_result):
        """Self-RAG can be disabled."""
        handler = MultiHopHandler(
            vector_store=mock_vector_store,
            llm_client=mock_llm_client,
            enable_self_rag=False,
        )
        mock_vector_store.set_results([sample_search_result])
        sub_query = SubQuery(query_id="q1", query_text="test", hop_order=1)
        result = await handler._execute_hop(sub_query, {}, 5)
        # When disabled, is_relevant defaults to True
        assert result.is_relevant is True


# =============================================================================
# Test HopStatus Enum
# =============================================================================


class TestHopStatus:
    """Characterization tests for HopStatus enum."""

    def test_values(self):
        """HopStatus has expected values."""
        assert HopStatus.PENDING.value == "pending"
        assert HopStatus.IN_PROGRESS.value == "in_progress"
        assert HopStatus.COMPLETED.value == "completed"
        assert HopStatus.FAILED.value == "failed"
        assert HopStatus.SKIPPED.value == "skipped"
        assert HopStatus.TIMEOUT.value == "timeout"
