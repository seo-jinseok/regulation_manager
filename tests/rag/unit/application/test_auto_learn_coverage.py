"""
Characterization tests for AutoLearnUseCase.

These tests document the CURRENT behavior of auto learning,
not what it SHOULD do. Tests capture actual outputs for regression detection.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockFeedbackEntry:
    """Mock feedback entry for testing."""
    query: str
    matched_intents: List[str] = field(default_factory=list)
    rule_code: str = ""
    rating: int = -1


@pytest.fixture
def mock_feedback_collector():
    """Create a mock feedback collector for testing."""
    collector = MagicMock()
    collector.get_negative_feedback.return_value = []
    return collector


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    client = MagicMock()
    client.generate.return_value = '{"suggestions": ["개선 제안"]}'
    return client


@pytest.fixture
def sample_negative_feedback():
    """Create sample negative feedback entries."""
    return [
        MockFeedbackEntry(query="휴학 신청", matched_intents=["휴학"], rating=-1),
        MockFeedbackEntry(query="휴학 신청", matched_intents=["휴학"], rating=-1),
        MockFeedbackEntry(query="복학 방법", matched_intents=[], rating=-1),
    ]


# ============================================================================
# ImprovementSuggestion Tests
# ============================================================================


class TestImprovementSuggestion:
    """Tests for ImprovementSuggestion dataclass."""

    def test_suggestion_creation_basic(self):
        """ImprovementSuggestion can be created with required fields."""
        from src.rag.application.auto_learn import ImprovementSuggestion

        suggestion = ImprovementSuggestion(
            type="intent",
            priority="high",
            description="Test suggestion",
        )
        assert suggestion.type == "intent"
        assert suggestion.priority == "high"
        assert suggestion.description == "Test suggestion"
        assert suggestion.suggested_value == {}
        assert suggestion.affected_queries == []

    def test_suggestion_with_all_fields(self):
        """ImprovementSuggestion can include all fields."""
        from src.rag.application.auto_learn import ImprovementSuggestion

        suggestion = ImprovementSuggestion(
            type="synonym",
            priority="medium",
            description="Add synonym",
            suggested_value={"term": "휴학"},
            affected_queries=["휴학 신청"],
        )
        assert suggestion.suggested_value["term"] == "휴학"


# ============================================================================
# AnalysisResult Tests
# ============================================================================


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_result_creation(self):
        """AnalysisResult can be created with required fields."""
        from src.rag.application.auto_learn import AnalysisResult

        result = AnalysisResult(
            total_negative_feedback=10,
            unique_problematic_queries=5,
        )
        assert result.total_negative_feedback == 10
        assert result.unique_problematic_queries == 5
        assert result.suggestions == []


# ============================================================================
# AutoLearnUseCase Initialization Tests
# ============================================================================


class TestAutoLearnUseCaseInit:
    """Tests for AutoLearnUseCase initialization."""

    def test_init_with_feedback_collector(self, mock_feedback_collector):
        """AutoLearnUseCase initializes with feedback collector."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        usecase = AutoLearnUseCase(feedback_collector=mock_feedback_collector)
        assert usecase._feedback == mock_feedback_collector

    def test_init_with_llm_client(self, mock_feedback_collector, mock_llm_client):
        """AutoLearnUseCase accepts LLM client."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        usecase = AutoLearnUseCase(
            feedback_collector=mock_feedback_collector,
            llm_client=mock_llm_client,
        )
        assert usecase._llm_client == mock_llm_client

    def test_init_without_feedback_collector(self):
        """AutoLearnUseCase can be initialized without feedback collector."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        usecase = AutoLearnUseCase()
        assert usecase._feedback is None


# ============================================================================
# analyze_feedback Tests
# ============================================================================


class TestAnalyzeFeedback:
    """Tests for analyze_feedback method."""

    def test_analyze_feedback_no_collector(self):
        """analyze_feedback returns empty result when no collector."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        usecase = AutoLearnUseCase()
        result = usecase.analyze_feedback()

        assert result.total_negative_feedback == 0
        assert result.unique_problematic_queries == 0

    def test_analyze_feedback_no_negative(self, mock_feedback_collector):
        """analyze_feedback returns empty result when no negative feedback."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        usecase = AutoLearnUseCase(feedback_collector=mock_feedback_collector)
        mock_feedback_collector.get_negative_feedback.return_value = []

        result = usecase.analyze_feedback()

        assert result.total_negative_feedback == 0

    def test_analyze_feedback_with_negative(
        self, mock_feedback_collector, sample_negative_feedback, tmp_path
    ):
        """analyze_feedback processes negative feedback."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        # Create mock intents and synonyms files
        intents_path = tmp_path / "intents.json"
        intents_path.write_text('{"intents": [{"triggers": ["휴학"]}]}')
        synonyms_path = tmp_path / "synonyms.json"
        synonyms_path.write_text('{"terms": {"휴학": ["휴학원"]}}')

        usecase = AutoLearnUseCase(
            feedback_collector=mock_feedback_collector,
            intents_path=str(intents_path),
            synonyms_path=str(synonyms_path),
        )
        mock_feedback_collector.get_negative_feedback.return_value = sample_negative_feedback

        result = usecase.analyze_feedback()

        assert result.total_negative_feedback == 3
        assert result.unique_problematic_queries == 2


# ============================================================================
# suggest_with_llm Tests
# ============================================================================


class TestSuggestWithLLM:
    """Tests for suggest_with_llm method."""

    def test_suggest_with_llm_no_client(self):
        """suggest_with_llm returns None when no LLM client."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        usecase = AutoLearnUseCase()
        result = usecase.suggest_with_llm("휴학 신청")

        assert result is None

    def test_suggest_with_llm_with_client(self, mock_llm_client):
        """suggest_with_llm uses LLM client for suggestions."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        usecase = AutoLearnUseCase(llm_client=mock_llm_client)
        result = usecase.suggest_with_llm("휴학 신청")

        # Result depends on LLM response parsing
        assert result is not None or mock_llm_client.generate.called


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_analyze_query_pattern_no_intents(self, mock_feedback_collector):
        """_analyze_query_pattern detects missing intents."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        usecase = AutoLearnUseCase(feedback_collector=mock_feedback_collector)

        entries = [MockFeedbackEntry(query="test", matched_intents=[])]
        result = usecase._analyze_query_pattern("test", entries)

        assert result is not None
        assert result.type == "intent"

    def test_analyze_query_pattern_with_intents(self, mock_feedback_collector):
        """_analyze_query_pattern suggests rerank review when intents matched."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        usecase = AutoLearnUseCase(feedback_collector=mock_feedback_collector)

        entries = [
            MockFeedbackEntry(query="test", matched_intents=["intent1"], rule_code="RULE-1")
        ]
        result = usecase._analyze_query_pattern("test", entries)

        assert result is not None
        assert result.type == "rerank"


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_analyze_feedback_exception_handling(self, mock_feedback_collector):
        """analyze_feedback may propagate exceptions (documents current behavior)."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        usecase = AutoLearnUseCase(feedback_collector=mock_feedback_collector)
        mock_feedback_collector.get_negative_feedback.side_effect = Exception("Error")

        # Current behavior: exception is not caught, so it propagates
        with pytest.raises(Exception):
            usecase.analyze_feedback()

    def test_check_missing_intents_file_not_found(self, mock_feedback_collector, tmp_path):
        """_check_missing_intents handles missing file."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        usecase = AutoLearnUseCase(
            feedback_collector=mock_feedback_collector,
            intents_path=str(tmp_path / "nonexistent.json"),
        )

        result = usecase._check_missing_intents({"test": []})
        assert result == []

    def test_check_missing_synonyms_file_not_found(self, mock_feedback_collector, tmp_path):
        """_check_missing_synonyms handles missing file."""
        from src.rag.application.auto_learn import AutoLearnUseCase

        usecase = AutoLearnUseCase(
            feedback_collector=mock_feedback_collector,
            synonyms_path=str(tmp_path / "nonexistent.json"),
        )

        result = usecase._check_missing_synonyms({"test": []})
        assert result == []
