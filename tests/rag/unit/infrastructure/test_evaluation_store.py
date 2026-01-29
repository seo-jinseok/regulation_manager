"""
Unit tests for EvaluationStore.

Tests JSON file-based storage operations including save, load,
historical retrieval, and statistics calculation.
"""

import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.rag.domain.evaluation.models import EvaluationResult
from src.rag.infrastructure.storage.evaluation_store import (
    EvaluationStatistics,
    EvaluationStore,
)


@pytest.fixture
def temp_storage_dir():
    """Create temporary directory for test storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_evaluation_result():
    """Create sample EvaluationResult for testing."""
    return EvaluationResult(
        query="What are the admission requirements?",
        answer="Undergraduate admission requires high school diploma.",
        contexts=["Context 1", "Context 2"],
        faithfulness=0.92,
        answer_relevancy=0.88,
        contextual_precision=0.85,
        contextual_recall=0.90,
        overall_score=0.89,
        passed=True,
        failure_reasons=[],
        metadata={"test": "sample"},
    )


@pytest.fixture
def evaluation_store(temp_storage_dir):
    """Create EvaluationStore with temporary directory."""
    return EvaluationStore(storage_dir=temp_storage_dir)


class TestEvaluationStore:
    """Test suite for EvaluationStore class."""

    def test_initialization_creates_directory(self, temp_storage_dir):
        """Test that store initialization creates storage directory."""
        store = EvaluationStore(storage_dir=temp_storage_dir)
        assert Path(temp_storage_dir).exists()
        assert store.storage_dir == Path(temp_storage_dir)

    def test_save_evaluation_creates_json_file(
        self, evaluation_store, sample_evaluation_result
    ):
        """Test that save_evaluation creates a JSON file."""
        evaluation_store.save_evaluation(sample_evaluation_result)

        files = list(evaluation_store.storage_dir.glob("evaluation_*.json"))
        assert len(files) == 1

        # Verify file content
        with open(files[0], "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["query"] == sample_evaluation_result.query
        assert data["overall_score"] == sample_evaluation_result.overall_score

    def test_save_multiple_evaluations_creates_multiple_files(
        self, evaluation_store, sample_evaluation_result
    ):
        """Test that saving multiple evaluations creates multiple files."""
        base_time = datetime.now()
        for i in range(3):
            result = EvaluationResult(
                query=f"Query {i}",
                answer=f"Answer {i}",
                contexts=[],
                faithfulness=0.8 + i * 0.05,
                answer_relevancy=0.8 + i * 0.05,
                contextual_precision=0.8 + i * 0.05,
                contextual_recall=0.8 + i * 0.05,
                overall_score=0.8 + i * 0.05,
                passed=True,
                timestamp=base_time + timedelta(seconds=i + 1),  # Different timestamps
            )
            evaluation_store.save_evaluation(result)

        files = list(evaluation_store.storage_dir.glob("evaluation_*.json"))
        assert len(files) == 3

    def test_get_latest_evaluation_returns_most_recent(
        self, evaluation_store, sample_evaluation_result
    ):
        """Test that get_latest_evaluation returns the most recent result."""
        # Save multiple results with different timestamps
        base_time = datetime.now()
        for i in range(3):
            result = EvaluationResult(
                query=f"Query {i}",
                answer=f"Answer {i}",
                contexts=[],
                faithfulness=0.8 + i * 0.05,
                answer_relevancy=0.8 + i * 0.05,
                contextual_precision=0.8 + i * 0.05,
                contextual_recall=0.8 + i * 0.05,
                overall_score=0.8 + i * 0.05,
                passed=True,
                timestamp=base_time + timedelta(seconds=i + 1),
            )
            evaluation_store.save_evaluation(result)

        latest = evaluation_store.get_latest_evaluation()
        assert latest is not None
        assert latest.query == "Query 2"

    def test_get_latest_evaluation_returns_none_when_empty(self, evaluation_store):
        """Test that get_latest_evaluation returns None when no evaluations."""
        latest = evaluation_store.get_latest_evaluation()
        assert latest is None

    def test_get_history_since_filters_by_date(
        self, evaluation_store, sample_evaluation_result
    ):
        """Test that get_history_since filters results by date."""
        now = datetime.now()

        # Save old result
        old_timestamp = now - timedelta(days=5)
        old_result = EvaluationResult(
            query="Old query",
            answer="Old answer",
            contexts=[],
            faithfulness=0.75,
            answer_relevancy=0.75,
            contextual_precision=0.75,
            contextual_recall=0.75,
            overall_score=0.75,
            passed=True,
            timestamp=old_timestamp,
        )
        evaluation_store.save_evaluation(old_result)

        # Save new result
        new_result = EvaluationResult(
            query="New query",
            answer="New answer",
            contexts=[],
            faithfulness=0.85,
            answer_relevancy=0.85,
            contextual_precision=0.85,
            contextual_recall=0.85,
            overall_score=0.85,
            passed=True,
            timestamp=now,
        )
        evaluation_store.save_evaluation(new_result)

        # Get history since 3 days ago (should only include new result)
        cutoff = now - timedelta(days=3)
        history = evaluation_store.get_history_since(cutoff)

        assert len(history) == 1
        assert history[0].query == "New query"

    def test_get_baseline_returns_first_evaluation(
        self, evaluation_store, sample_evaluation_result
    ):
        """Test that get_baseline returns the first stored evaluation."""
        base_time = datetime.now()
        for i in range(3):
            result = EvaluationResult(
                query=f"Query {i}",
                answer=f"Answer {i}",
                contexts=[],
                faithfulness=0.8 + i * 0.05,
                answer_relevancy=0.8 + i * 0.05,
                contextual_precision=0.8 + i * 0.05,
                contextual_recall=0.8 + i * 0.05,
                overall_score=0.8 + i * 0.05,
                passed=True,
                timestamp=base_time + timedelta(seconds=i + 1),
            )
            evaluation_store.save_evaluation(result)

        baseline = evaluation_store.get_baseline()
        assert baseline is not None
        assert baseline.query == "Query 0"

    def test_get_baseline_returns_none_when_empty(self, evaluation_store):
        """Test that get_baseline returns None when no evaluations."""
        baseline = evaluation_store.get_baseline()
        assert baseline is None

    def test_get_statistics_calculates_aggregates(
        self, evaluation_store, sample_evaluation_result
    ):
        """Test that get_statistics correctly calculates aggregate metrics."""
        # Save multiple results with known scores and different timestamps
        scores = [0.75, 0.85, 0.95]
        base_time = datetime.now()
        for i, score in enumerate(scores):
            result = EvaluationResult(
                query="Query",
                answer="Answer",
                contexts=[],
                faithfulness=score,
                answer_relevancy=score,
                contextual_precision=score,
                contextual_recall=score,
                overall_score=score,
                passed=score >= 0.8,
                timestamp=base_time + timedelta(seconds=i + 1),
            )
            evaluation_store.save_evaluation(result)

        stats = evaluation_store.get_statistics()

        assert stats.total_evaluations == 3
        assert stats.avg_overall_score == 0.85
        assert stats.min_score == 0.75
        assert stats.max_score == 0.95
        assert stats.pass_rate == pytest.approx(2 / 3, abs=0.0001)

    def test_get_statistics_with_days_parameter(self, evaluation_store):
        """Test get_statistics with days parameter filters correctly."""
        now = datetime.now()

        # Save old result (10 days ago)
        old_result = EvaluationResult(
            query="Old",
            answer="Old",
            contexts=[],
            faithfulness=0.70,
            answer_relevancy=0.70,
            contextual_precision=0.70,
            contextual_recall=0.70,
            overall_score=0.70,
            passed=True,
            timestamp=now - timedelta(days=10),
        )
        evaluation_store.save_evaluation(old_result)

        # Save new result (1 day ago)
        new_result = EvaluationResult(
            query="New",
            answer="New",
            contexts=[],
            faithfulness=0.90,
            answer_relevancy=0.90,
            contextual_precision=0.90,
            contextual_recall=0.90,
            overall_score=0.90,
            passed=True,
            timestamp=now - timedelta(days=1),
        )
        evaluation_store.save_evaluation(new_result)

        # Get statistics for last 5 days (should only include new)
        stats = evaluation_store.get_statistics(days=5)

        assert stats.total_evaluations == 1
        assert stats.avg_overall_score == 0.90

    def test_get_statistics_returns_empty_when_no_evaluations(self, evaluation_store):
        """Test that get_statistics returns zeros when no evaluations."""
        stats = evaluation_store.get_statistics()

        assert stats.total_evaluations == 0
        assert stats.avg_overall_score == 0.0
        assert stats.pass_rate == 0.0

    def test_compare_to_baseline(self, evaluation_store):
        """Test compare_to_baseline returns comparison metrics."""
        # Save baseline
        baseline = EvaluationResult(
            query="Baseline query",
            answer="Baseline answer",
            contexts=[],
            faithfulness=0.80,
            answer_relevancy=0.80,
            contextual_precision=0.80,
            contextual_recall=0.80,
            overall_score=0.80,
            passed=True,
            timestamp=datetime.now() - timedelta(seconds=2),
        )
        evaluation_store.save_evaluation(baseline)

        # Save current (improved)
        current = EvaluationResult(
            query="Current query",
            answer="Current answer",
            contexts=[],
            faithfulness=0.90,
            answer_relevancy=0.90,
            contextual_precision=0.90,
            contextual_recall=0.90,
            overall_score=0.90,
            passed=True,
        )

        comparison = evaluation_store.compare_to_baseline(current)

        assert comparison["has_baseline"] is True
        assert comparison["baseline_overall"] == 0.80
        assert comparison["current_overall"] == 0.90
        assert comparison["overall_delta"] == pytest.approx(0.10, abs=0.001)
        assert comparison["is_improved"] is True

    def test_compare_to_baseline_without_baseline(self, evaluation_store):
        """Test compare_to_baseline when no baseline exists."""
        current = EvaluationResult(
            query="Query",
            answer="Answer",
            contexts=[],
            faithfulness=0.85,
            answer_relevancy=0.85,
            contextual_precision=0.85,
            contextual_recall=0.85,
            overall_score=0.85,
            passed=True,
        )

        comparison = evaluation_store.compare_to_baseline(current)

        assert comparison["has_baseline"] is False
        assert "message" in comparison

    def test_calculate_trend_improving(self, evaluation_store):
        """Test trend calculation for improving scores."""
        # Save results with increasing scores and different timestamps
        base_time = datetime.now()
        for i in range(4):
            result = EvaluationResult(
                query="Query",
                answer="Answer",
                contexts=[],
                faithfulness=0.7 + i * 0.05,
                answer_relevancy=0.7 + i * 0.05,
                contextual_precision=0.7 + i * 0.05,
                contextual_recall=0.7 + i * 0.05,
                overall_score=0.7 + i * 0.05,
                passed=True,
                timestamp=base_time + timedelta(seconds=i + 1),
            )
            evaluation_store.save_evaluation(result)

        stats = evaluation_store.get_statistics()
        assert stats.trend == "improving"

    def test_calculate_trend_declining(self, evaluation_store):
        """Test trend calculation for declining scores."""
        # Save results with decreasing scores and different timestamps
        base_time = datetime.now()
        for i in range(4):
            result = EvaluationResult(
                query="Query",
                answer="Answer",
                contexts=[],
                faithfulness=0.9 - i * 0.05,
                answer_relevancy=0.9 - i * 0.05,
                contextual_precision=0.9 - i * 0.05,
                contextual_recall=0.9 - i * 0.05,
                overall_score=0.9 - i * 0.05,
                passed=True,
                timestamp=base_time + timedelta(seconds=i + 1),
            )
            evaluation_store.save_evaluation(result)

        stats = evaluation_store.get_statistics()
        assert stats.trend == "declining"

    def test_calculate_trend_stable(self, evaluation_store):
        """Test trend calculation for stable scores."""
        # Save results with similar scores
        for _ in range(4):
            result = EvaluationResult(
                query="Query",
                answer="Answer",
                contexts=[],
                faithfulness=0.85,
                answer_relevancy=0.85,
                contextual_precision=0.85,
                contextual_recall=0.85,
                overall_score=0.85,
                passed=True,
            )
            evaluation_store.save_evaluation(result)

        stats = evaluation_store.get_statistics()
        assert stats.trend == "stable"


class TestEvaluationStatistics:
    """Test suite for EvaluationStatistics dataclass."""

    def test_to_dict_conversion(self):
        """Test that to_dict converts statistics to dictionary."""
        stats = EvaluationStatistics(
            total_evaluations=10,
            avg_faithfulness=0.88,
            avg_answer_relevancy=0.86,
            avg_contextual_precision=0.84,
            avg_contextual_recall=0.87,
            avg_overall_score=0.86,
            pass_rate=0.80,
            min_score=0.75,
            max_score=0.95,
            std_deviation=0.05,
            timestamp_range=(datetime.now(), datetime.now()),
            trend="improving",
        )

        data = stats.to_dict()

        assert data["total_evaluations"] == 10
        assert data["avg_overall_score"] == 0.86
        assert data["trend"] == "improving"
        assert "timestamp_range" in data
