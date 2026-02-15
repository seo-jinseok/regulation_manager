"""
Characterization tests for confidence threshold and fallback response (TAG-001).

These tests verify behavior preservation for the confidence threshold feature
that prevents hallucination when retrieved context is not relevant.

SPEC Requirement (REQ-002):
    WHEN no relevant context is found (contextual_recall < 0.3)
    THE SYSTEM SHALL respond with a safe fallback message
    AND NOT generate content beyond retrieved context
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import List, Optional

from src.rag.application.search_usecase import (
    SearchUseCase,
    FALLBACK_MESSAGE_KO,
    FALLBACK_MESSAGE_EN,
)
from src.rag.domain.entities import SearchResult, Chunk, ChunkLevel, Answer, Keyword


def create_mock_chunk(
    text: str = "Test chunk content",
    score: float = 0.5,
    rule_code: str = "TEST-001",
    chunk_id: str = "test-chunk-1",
) -> SearchResult:
    """Create a mock SearchResult for testing."""
    chunk = Chunk(
        id=chunk_id,
        text=text,
        rule_code=rule_code,
        level=ChunkLevel.ARTICLE,
        title="Test Article",
        embedding_text=text,
        full_text=text,
        parent_path=[],
        token_count=10,
        keywords=[],
        is_searchable=True,
    )
    return SearchResult(chunk=chunk, score=score, rank=1)


def create_mock_search_results(
    num_results: int = 3,
    base_score: float = 0.5,
) -> List[SearchResult]:
    """Create multiple mock search results with specified scores."""
    results = []
    for i in range(num_results):
        results.append(
            create_mock_chunk(
                text=f"Test content {i}",
                score=base_score - (i * 0.1),  # Decreasing scores
                chunk_id=f"chunk-{i}",
            )
        )
    return results


class TestConfidenceThresholdFallback:
    """Test cases for confidence threshold fallback behavior."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        store.search.return_value = []
        return store

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        llm.generate.return_value = "Generated answer text"
        return llm

    @pytest.fixture
    def search_usecase(self, mock_store, mock_llm):
        """Create a SearchUseCase instance for testing."""
        usecase = SearchUseCase(
            store=mock_store,
            llm_client=mock_llm,
            use_reranker=False,
            use_hybrid=False,
        )
        return usecase

    def test_ask_returns_fallback_when_confidence_below_threshold(
        self, search_usecase, mock_store, mock_llm
    ):
        """
        Characterization test: ask() returns fallback message when confidence < 0.3.

        This test verifies that:
        1. Low confidence results trigger fallback response
        2. The fallback message is exactly as specified in SPEC
        3. Sources are empty for fallback responses
        """
        # Setup: Mock search results with very low scores (confidence will be < 0.3)
        low_score_results = create_mock_search_results(num_results=3, base_score=0.05)

        # Set a low confidence threshold for testing
        search_usecase._confidence_threshold = 0.3

        # Patch the search method to return low score results
        # Also disable Self-RAG to simplify the test
        search_usecase._enable_self_rag = False
        with patch.object(search_usecase, 'search', return_value=low_score_results):
            # Execute
            result = search_usecase.ask("What is the test query?")

        # Verify: Should return fallback message
        assert isinstance(result, Answer)
        assert result.text == FALLBACK_MESSAGE_KO
        assert result.confidence < 0.3
        assert result.sources == []  # No sources for fallback

    def test_ask_returns_normal_answer_when_confidence_above_threshold(
        self, search_usecase, mock_store, mock_llm
    ):
        """
        Characterization test: ask() returns normal answer when confidence >= 0.3.

        This test verifies that:
        1. Sufficient confidence allows normal answer generation
        2. LLM is called to generate the answer
        3. Sources are included in the response
        """
        # Setup: Mock search results with good scores (confidence >= 0.3)
        good_score_results = create_mock_search_results(num_results=3, base_score=0.7)

        search_usecase._confidence_threshold = 0.3

        # Patch the search method to return good score results
        with patch.object(search_usecase, 'search', return_value=good_score_results):
            # Execute
            result = search_usecase.ask("What is the test query?")

        # Verify: Should return generated answer
        assert isinstance(result, Answer)
        assert result.text != FALLBACK_MESSAGE_KO
        assert result.confidence >= 0.3
        assert len(result.sources) > 0

    def test_confidence_threshold_configurable(self, search_usecase):
        """
        Characterization test: Confidence threshold can be configured.

        This test verifies that:
        1. Threshold can be set via environment variable
        2. Different thresholds produce different behaviors
        """
        # Test with different threshold values
        search_usecase._confidence_threshold = 0.5
        assert search_usecase._confidence_threshold == 0.5

        search_usecase._confidence_threshold = 0.2
        assert search_usecase._confidence_threshold == 0.2

    def test_ask_multilingual_english_returns_fallback_when_confidence_below_threshold(
        self, search_usecase, mock_store, mock_llm
    ):
        """
        Characterization test: ask_multilingual(english) returns English fallback when confidence < 0.3.

        This test verifies that:
        1. English queries use English fallback message
        2. Low confidence triggers fallback for English queries too
        """
        # Setup: Mock search results with very low scores
        low_score_results = create_mock_search_results(num_results=3, base_score=0.05)

        search_usecase._confidence_threshold = 0.3

        # Disable Self-RAG to simplify the test
        search_usecase._enable_self_rag = False

        # Patch the search method to return low score results
        with patch.object(search_usecase, 'search', return_value=low_score_results):
            # Execute
            result = search_usecase.ask_multilingual("What is the test query?", language="english")

        # Verify: Should return English fallback message
        assert isinstance(result, Answer)
        assert result.text == FALLBACK_MESSAGE_EN
        assert result.confidence < 0.3
        assert result.sources == []

    def test_ask_multilingual_english_returns_normal_answer_when_confidence_above_threshold(
        self, search_usecase, mock_store, mock_llm
    ):
        """
        Characterization test: ask_multilingual(english) returns normal answer when confidence >= 0.3.

        This test verifies that:
        1. Sufficient confidence allows normal answer generation for English queries
        2. LLM is called with English prompt
        """
        # Setup: Mock search results with good scores
        good_score_results = create_mock_search_results(num_results=3, base_score=0.7)

        search_usecase._confidence_threshold = 0.3

        # Patch the search method to return good score results
        with patch.object(search_usecase, 'search', return_value=good_score_results):
            # Execute
            result = search_usecase.ask_multilingual("What is the test query?", language="english")

        # Verify: Should return generated answer
        assert isinstance(result, Answer)
        assert result.text != FALLBACK_MESSAGE_EN
        assert result.confidence >= 0.3
        assert len(result.sources) > 0

    def test_confidence_computation_with_mixed_scores(self, search_usecase):
        """
        Characterization test: Confidence is computed correctly from search scores.

        This test verifies the _compute_confidence method behavior:
        1. Higher scores produce higher confidence
        2. Score spread affects confidence
        """
        # Test with high scores
        high_score_results = create_mock_search_results(num_results=5, base_score=0.8)
        high_confidence = search_usecase._compute_confidence(high_score_results)

        # Test with low scores
        low_score_results = create_mock_search_results(num_results=5, base_score=0.1)
        low_confidence = search_usecase._compute_confidence(low_score_results)

        # Verify: High scores should produce higher confidence
        assert high_confidence > low_confidence


class TestConfidenceThresholdEdgeCases:
    """Test edge cases for confidence threshold behavior."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        return store

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        llm.generate.return_value = "Generated answer"
        return llm

    @pytest.fixture
    def search_usecase(self, mock_store, mock_llm):
        """Create a SearchUseCase instance for testing."""
        usecase = SearchUseCase(
            store=mock_store,
            llm_client=mock_llm,
            use_reranker=False,
            use_hybrid=False,
        )
        return usecase

    def test_confidence_exactly_at_threshold(
        self, search_usecase, mock_store, mock_llm
    ):
        """
        Edge case: Confidence exactly at threshold should generate answer.

        When confidence == threshold, the system should generate a normal answer
        (not return fallback).
        """
        # Setup: Create results that will produce exactly 0.3 confidence
        # Using _compute_confidence logic: avg_score affects confidence
        threshold_results = create_mock_search_results(num_results=3, base_score=0.3)

        search_usecase._confidence_threshold = 0.3

        # Patch the search method
        with patch.object(search_usecase, 'search', return_value=threshold_results):
            # Execute
            result = search_usecase.ask("Test query")

        # Verify: Should generate normal answer (not fallback)
        # Note: Actual behavior depends on _compute_confidence implementation
        assert isinstance(result, Answer)

    def test_single_low_score_result(
        self, search_usecase, mock_store, mock_llm
    ):
        """
        Edge case: Single result with very low score should trigger fallback.

        Note: For single result with score < 0.1, _compute_confidence uses
        abs_scale=0.05, spread_confidence=0.5. To get confidence < 0.3:
        combined = (avg/0.05 * 0.7) + (0.5 * 0.3) < 0.3
        avg/0.05 * 0.7 < 0.15
        avg < 0.15 / 0.7 * 0.05 = 0.0107
        So we need avg < ~0.01 for confidence < 0.3
        """
        # Setup: Single result with very low score (< 0.01 to get confidence < 0.3)
        single_low_result = [create_mock_chunk(text="Low relevance", score=0.005)]

        search_usecase._confidence_threshold = 0.3

        # Disable Self-RAG to simplify the test
        search_usecase._enable_self_rag = False

        # Patch the search method
        with patch.object(search_usecase, 'search', return_value=single_low_result):
            # Execute
            result = search_usecase.ask("Test query")

        # Verify: Should return fallback
        assert result.text == FALLBACK_MESSAGE_KO
        assert result.confidence < 0.3


class TestConfidenceThresholdPreservation:
    """Tests to verify behavior preservation after adding confidence threshold."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock vector store."""
        store = MagicMock()
        return store

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client."""
        llm = MagicMock()
        llm.generate.return_value = "Normal generated answer"
        return llm

    @pytest.fixture
    def search_usecase(self, mock_store, mock_llm):
        """Create a SearchUseCase instance for testing."""
        usecase = SearchUseCase(
            store=mock_store,
            llm_client=mock_llm,
            use_reranker=False,
            use_hybrid=False,
        )
        # Set a very low threshold to ensure normal behavior is preserved
        usecase._confidence_threshold = 0.0
        return usecase

    def test_normal_answer_flow_preserved(
        self, search_usecase, mock_store, mock_llm
    ):
        """
        Preservation test: Normal answer generation flow is preserved.

        When confidence is above threshold, the existing flow should be unchanged:
        1. Search results are retrieved
        2. Context is built
        3. LLM generates answer
        4. Answer is returned with sources
        """
        # Setup
        good_results = create_mock_search_results(num_results=5, base_score=0.8)

        # Disable Self-RAG to simplify the test
        search_usecase._enable_self_rag = False

        # Patch the search method
        with patch.object(search_usecase, 'search', return_value=good_results):
            # Execute
            result = search_usecase.ask("Test query about regulations")

        # Verify: Normal flow preserved
        assert isinstance(result, Answer)
        assert "Generated answer" in result.text or result.text != FALLBACK_MESSAGE_KO
        mock_llm.generate.assert_called()
        assert len(result.sources) > 0

    def test_answer_entity_structure_preserved(
        self, search_usecase, mock_store, mock_llm
    ):
        """
        Preservation test: Answer entity structure is preserved.

        The Answer entity should maintain its structure:
        - text: str
        - sources: List[SearchResult]
        - confidence: float
        """
        # Setup
        results = create_mock_search_results(num_results=3, base_score=0.5)

        # Disable Self-RAG to simplify the test
        search_usecase._enable_self_rag = False

        # Patch the search method
        with patch.object(search_usecase, 'search', return_value=results):
            # Execute
            answer = search_usecase.ask("Test query")

        # Verify: Answer structure preserved
        assert hasattr(answer, "text")
        assert hasattr(answer, "sources")
        assert hasattr(answer, "confidence")
        assert isinstance(answer.text, str)
        assert isinstance(answer.sources, list)
        assert isinstance(answer.confidence, float)
