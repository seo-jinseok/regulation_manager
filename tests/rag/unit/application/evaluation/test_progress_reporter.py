"""
Unit tests for ProgressReporter.

Tests for SPEC-RAG-EVAL-001 Milestone 3: Automation Pipeline.
"""

from datetime import timedelta
from typing import List
from unittest.mock import MagicMock

import pytest

from src.rag.application.evaluation.progress_reporter import (
    PersonaProgressInfo,
    ProgressInfo,
    ProgressReporter,
    create_progress_bar,
)


@pytest.fixture
def callback_calls():
    """Track callback calls."""
    return []


@pytest.fixture
def progress_callback(callback_calls):
    """Create a callback that records calls."""
    def callback(progress: ProgressInfo):
        callback_calls.append(progress)
    return callback


@pytest.fixture
def reporter(progress_callback):
    """Create a ProgressReporter with callback."""
    return ProgressReporter(
        total_queries=100,
        callback=progress_callback,
        report_interval=0.0,  # Report every update
    )


class TestProgressInfo:
    """Tests for ProgressInfo."""

    def test_creation(self):
        """Test creating a ProgressInfo."""
        info = ProgressInfo(
            total_queries=100,
            completed_queries=50,
            failed_queries=5,
            current_persona="professor",
            current_query="test query",
            start_time=0.0,
            elapsed_seconds=60.0,
            queries_per_second=0.83,
            estimated_remaining_seconds=54.0,
            completion_percentage=55.0,
        )

        assert info.total_queries == 100
        assert info.completed_queries == 50
        assert info.failed_queries == 5
        assert info.completion_percentage == 55.0

    def test_format_elapsed(self):
        """Test formatting elapsed time."""
        info = ProgressInfo(
            total_queries=100,
            completed_queries=50,
            failed_queries=0,
            current_persona=None,
            current_query=None,
            start_time=0.0,
            elapsed_seconds=3661.5,  # 1 hour, 1 min, 1.5 sec
            queries_per_second=1.0,
            estimated_remaining_seconds=0,
            completion_percentage=50.0,
        )

        assert info.format_elapsed() == "01:01:01"

    def test_format_eta(self):
        """Test formatting ETA."""
        info = ProgressInfo(
            total_queries=100,
            completed_queries=50,
            failed_queries=0,
            current_persona=None,
            current_query=None,
            start_time=0.0,
            elapsed_seconds=60.0,
            queries_per_second=1.0,
            estimated_remaining_seconds=125.0,  # 2 min, 5 sec
            completion_percentage=50.0,
        )

        assert info.format_eta() == "00:02:05"

    def test_format_eta_negative(self):
        """Test formatting negative ETA."""
        info = ProgressInfo(
            total_queries=100,
            completed_queries=100,
            failed_queries=0,
            current_persona=None,
            current_query=None,
            start_time=0.0,
            elapsed_seconds=100.0,
            queries_per_second=1.0,
            estimated_remaining_seconds=-1.0,
            completion_percentage=100.0,
        )

        assert info.format_eta() == "--:--:--"

    def test_to_dict(self):
        """Test serialization."""
        info = ProgressInfo(
            total_queries=100,
            completed_queries=50,
            failed_queries=5,
            current_persona="test",
            current_query="query",
            start_time=0.0,
            elapsed_seconds=60.0,
            queries_per_second=0.83,
            estimated_remaining_seconds=54.0,
            completion_percentage=55.0,
        )

        data = info.to_dict()

        assert data["total_queries"] == 100
        assert data["completed_queries"] == 50


class TestPersonaProgressInfo:
    """Tests for PersonaProgressInfo."""

    def test_creation(self):
        """Test creating a PersonaProgressInfo."""
        info = PersonaProgressInfo(
            persona_name="professor",
            total=50,
            completed=25,
            failed=2,
            avg_score=0.85,
        )

        assert info.persona_name == "professor"
        assert info.total == 50
        assert info.completed == 25
        assert info.avg_score == 0.85

    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        info = PersonaProgressInfo(
            persona_name="test",
            total=100,
            completed=75,
            failed=5,
        )

        assert info.completion_percentage == 75.0

    def test_to_dict(self):
        """Test serialization."""
        info = PersonaProgressInfo(
            persona_name="test",
            total=100,
            completed=50,
            failed=5,
            avg_score=0.8,
        )

        data = info.to_dict()

        assert data["persona_name"] == "test"
        assert data["completion_percentage"] == 50.0


class TestProgressReporter:
    """Tests for ProgressReporter."""

    def test_init(self):
        """Test initialization."""
        reporter = ProgressReporter(total_queries=100)

        assert reporter.total_queries == 100
        assert reporter._completed_queries == 0

    def test_init_with_personas(self):
        """Test initialization with persona counts."""
        reporter = ProgressReporter(
            total_queries=100,
            persona_counts={"freshman": 50, "professor": 50},
        )

        assert "freshman" in reporter._persona_progress
        assert "professor" in reporter._persona_progress

    def test_update(self, reporter, callback_calls):
        """Test updating progress."""
        progress = reporter.update(completed=1, persona="test", query="q1")

        assert progress.completed_queries == 1
        assert progress.current_persona == "test"
        assert len(callback_calls) == 1

    def test_update_with_failure(self, reporter, callback_calls):
        """Test updating with a failure."""
        progress = reporter.update(completed=1, failed=True)

        assert progress.failed_queries == 1
        assert progress.completed_queries == 0

    def test_update_with_score(self):
        """Test updating with a score for persona."""
        reporter = ProgressReporter(
            total_queries=100,
            persona_counts={"test": 50},
        )
        reporter.update(completed=1, persona="test", score=0.9)

        persona_progress = reporter.get_persona_progress()
        assert "test" in persona_progress
        assert persona_progress["test"].avg_score == 0.9

    def test_get_progress(self, reporter):
        """Test getting current progress."""
        reporter.update(completed=1)
        reporter.update(completed=1)

        progress = reporter.get_progress()

        assert progress.completed_queries == 2
        assert progress.total_queries == 100

    def test_get_eta(self, reporter):
        """Test getting ETA."""
        reporter.update(completed=50)

        eta = reporter.get_eta()

        assert isinstance(eta, timedelta)

    def test_complete(self, reporter, callback_calls):
        """Test marking as complete."""
        reporter.update(completed=100)
        progress = reporter.complete()

        assert progress.completed_queries == 100

    def test_format_cli_progress(self, reporter):
        """Test CLI progress formatting."""
        reporter.update(completed=50)
        formatted = reporter.format_cli_progress()

        assert "50%" in formatted or "50.0%" in formatted
        assert "50/100" in formatted

    def test_format_persona_summary(self):
        """Test persona summary formatting."""
        reporter = ProgressReporter(
            total_queries=100,
            persona_counts={"freshman": 50, "professor": 50},
        )

        reporter.update(completed=25, persona="freshman", score=0.8)
        reporter.update(completed=25, persona="professor", score=0.9)

        summary = reporter.format_persona_summary()

        assert "freshman" in summary
        assert "professor" in summary

    def test_callback_exception_handling(self, callback_calls):
        """Test that callback exceptions don't break updates."""
        def bad_callback(progress):
            raise ValueError("Bad callback")

        reporter = ProgressReporter(
            total_queries=100,
            callback=bad_callback,
            report_interval=0.0,
        )

        # Should not raise
        progress = reporter.update(completed=1)

        assert progress.completed_queries == 1


class TestCreateProgressBar:
    """Tests for create_progress_bar helper."""

    def test_empty(self):
        """Test progress bar at zero."""
        bar = create_progress_bar(0, 100, width=10)

        assert "0.0%" in bar
        assert "░" * 10 in bar

    def test_half(self):
        """Test progress bar at 50%."""
        bar = create_progress_bar(50, 100, width=10)

        assert "50.0%" in bar
        assert "█" * 5 in bar
        assert "░" * 5 in bar

    def test_full(self):
        """Test progress bar at 100%."""
        bar = create_progress_bar(100, 100, width=10)

        assert "100.0%" in bar
        assert "█" * 10 in bar

    def test_custom_chars(self):
        """Test progress bar with custom characters."""
        bar = create_progress_bar(50, 100, width=4, fill_char="#", empty_char="-")

        assert "##--" in bar

    def test_zero_total(self):
        """Test progress bar with zero total."""
        bar = create_progress_bar(0, 0, width=10)

        assert "0.0%" in bar


class TestProgressReporterEdgeCases:
    """Edge case tests for ProgressReporter."""

    def test_zero_queries(self):
        """Test with zero total queries."""
        reporter = ProgressReporter(total_queries=0)

        progress = reporter.get_progress()

        assert progress.total_queries == 0
        assert progress.completion_percentage == 0.0

    def test_speed_calculation(self):
        """Test queries per second calculation."""
        reporter = ProgressReporter(total_queries=100, report_interval=0.0)

        # First update
        reporter.update(completed=1)
        progress1 = reporter.get_progress()

        # Second update
        reporter.update(completed=1)
        progress2 = reporter.get_progress()

        # QPS should be positive after updates
        assert progress2.queries_per_second > 0

    def test_eta_when_no_progress(self):
        """Test ETA when no progress has been made."""
        reporter = ProgressReporter(total_queries=100)

        progress = reporter.get_progress()

        # ETA should be 0 when no progress and no speed
        assert progress.estimated_remaining_seconds == 0.0

    def test_multiple_persona_updates(self):
        """Test updating multiple personas."""
        reporter = ProgressReporter(
            total_queries=100,
            persona_counts={"a": 50, "b": 50},
        )

        reporter.update(completed=1, persona="a", score=0.8)
        reporter.update(completed=1, persona="b", score=0.9)

        persona_progress = reporter.get_persona_progress()

        assert persona_progress["a"].completed == 1
        assert persona_progress["b"].completed == 1
