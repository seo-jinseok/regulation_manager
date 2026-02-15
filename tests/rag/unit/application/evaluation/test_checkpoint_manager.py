"""
Unit tests for CheckpointManager.

Tests for SPEC-RAG-EVAL-001 Milestone 3: Automation Pipeline.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from src.rag.application.evaluation.checkpoint_manager import (
    CheckpointManager,
    EvaluationProgress,
    PersonaProgress,
)


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Create a CheckpointManager with temp directory."""
    return CheckpointManager(checkpoint_dir=temp_checkpoint_dir)


@pytest.fixture
def sample_progress():
    """Create a sample EvaluationProgress."""
    return EvaluationProgress(
        session_id="test-session-001",
        started_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:01:00",
        status="running",
        total_queries=100,
        completed_queries=50,
        failed_queries=5,
        personas={
            "freshman": PersonaProgress(
                persona_name="freshman",
                total_queries=25,
                completed_queries=15,
                failed_queries=2,
            ),
        },
    )


class TestPersonaProgress:
    """Tests for PersonaProgress."""

    def test_creation(self):
        """Test creating a PersonaProgress."""
        progress = PersonaProgress(
            persona_name="professor",
            total_queries=30,
            completed_queries=10,
            failed_queries=2,
        )

        assert progress.persona_name == "professor"
        assert progress.total_queries == 30
        assert progress.completed_queries == 10
        assert progress.failed_queries == 2

    def test_completion_rate(self):
        """Test completion rate calculation."""
        progress = PersonaProgress(
            persona_name="test",
            total_queries=100,
            completed_queries=25,
            failed_queries=0,
        )

        assert progress.completion_rate == 25.0

    def test_completion_rate_zero_total(self):
        """Test completion rate with zero total."""
        progress = PersonaProgress(
            persona_name="test",
            total_queries=0,
            completed_queries=0,
            failed_queries=0,
        )

        assert progress.completion_rate == 0.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        progress = PersonaProgress(
            persona_name="test",
            total_queries=10,
            completed_queries=5,
            failed_queries=1,
        )

        data = progress.to_dict()

        assert data["persona_name"] == "test"
        assert data["total_queries"] == 10
        assert data["completed_queries"] == 5

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "persona_name": "test",
            "total_queries": 10,
            "completed_queries": 5,
            "failed_queries": 1,
            "completed_query_ids": ["q1", "q2"],
            "failed_query_ids": ["q3"],
            "last_query": "q5",
            "last_timestamp": "2024-01-01T00:00:00",
        }

        progress = PersonaProgress.from_dict(data)

        assert progress.persona_name == "test"
        assert progress.total_queries == 10
        assert progress.completed_queries == 5


class TestEvaluationProgress:
    """Tests for EvaluationProgress."""

    def test_creation(self, sample_progress):
        """Test creating an EvaluationProgress."""
        assert sample_progress.session_id == "test-session-001"
        assert sample_progress.total_queries == 100
        assert sample_progress.completed_queries == 50
        assert sample_progress.failed_queries == 5

    def test_completion_rate(self, sample_progress):
        """Test completion rate calculation."""
        assert sample_progress.completion_rate == 50.0

    def test_is_complete(self, sample_progress):
        """Test is_complete property."""
        assert not sample_progress.is_complete

        sample_progress.status = "completed"
        assert sample_progress.is_complete

    def test_can_resume(self, sample_progress):
        """Test can_resume property."""
        sample_progress.status = "running"
        assert not sample_progress.can_resume

        sample_progress.status = "paused"
        assert sample_progress.can_resume

        sample_progress.status = "failed"
        assert sample_progress.can_resume

    def test_to_dict(self, sample_progress):
        """Test serialization."""
        data = sample_progress.to_dict()

        assert data["session_id"] == "test-session-001"
        assert data["total_queries"] == 100
        assert "personas" in data

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "session_id": "test",
            "started_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:01:00",
            "status": "running",
            "total_queries": 100,
            "completed_queries": 50,
            "failed_queries": 5,
            "personas": {},
            "results": [],
            "errors": [],
            "metadata": {},
        }

        progress = EvaluationProgress.from_dict(data)

        assert progress.session_id == "test"
        assert progress.total_queries == 100


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_init(self, temp_checkpoint_dir):
        """Test initialization."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        assert manager.checkpoint_dir == Path(temp_checkpoint_dir)

    def test_create_session(self, checkpoint_manager):
        """Test creating a new session."""
        progress = checkpoint_manager.create_session(
            session_id="test-001",
            total_queries=100,
            personas=["freshman", "professor"],
        )

        assert progress.session_id == "test-001"
        assert progress.total_queries == 100
        assert progress.status == "running"
        assert "freshman" in progress.personas
        assert "professor" in progress.personas

    def test_save_and_load_checkpoint(self, checkpoint_manager, sample_progress):
        """Test saving and loading a checkpoint."""
        checkpoint_manager.save_checkpoint("test-001", sample_progress)

        loaded = checkpoint_manager.load_checkpoint("test-001")

        assert loaded is not None
        assert loaded.session_id == sample_progress.session_id
        assert loaded.total_queries == sample_progress.total_queries

    def test_load_nonexistent_checkpoint(self, checkpoint_manager):
        """Test loading a checkpoint that doesn't exist."""
        loaded = checkpoint_manager.load_checkpoint("nonexistent")

        assert loaded is None

    def test_update_progress(self, checkpoint_manager):
        """Test updating progress."""
        # Create initial session
        checkpoint_manager.create_session(
            session_id="test-001",
            total_queries=100,
            personas=["freshman"],
        )

        # Update with a result
        updated = checkpoint_manager.update_progress(
            session_id="test-001",
            persona="freshman",
            query_id="q1",
            result={"score": 0.85},
        )

        assert updated is not None
        assert updated.completed_queries == 1

    def test_update_progress_with_error(self, checkpoint_manager):
        """Test updating progress with an error."""
        checkpoint_manager.create_session(
            session_id="test-001",
            total_queries=100,
            personas=["freshman"],
        )

        updated = checkpoint_manager.update_progress(
            session_id="test-001",
            persona="freshman",
            query_id="q1",
            error="API timeout",
        )

        assert updated is not None
        assert updated.failed_queries == 1
        assert "API timeout" in updated.errors

    def test_pause_and_resume_session(self, checkpoint_manager):
        """Test pausing and resuming a session."""
        checkpoint_manager.create_session(
            session_id="test-001",
            total_queries=100,
        )

        paused = checkpoint_manager.pause_session("test-001")
        assert paused is not None
        assert paused.status == "paused"

        resumed = checkpoint_manager.resume_session("test-001")
        assert resumed is not None
        assert resumed.status == "running"

    def test_clear_checkpoint(self, checkpoint_manager, sample_progress):
        """Test clearing a checkpoint."""
        checkpoint_manager.save_checkpoint("test-001", sample_progress)

        cleared = checkpoint_manager.clear_checkpoint("test-001")
        assert cleared is True

        loaded = checkpoint_manager.load_checkpoint("test-001")
        assert loaded is None

    def test_clear_nonexistent_checkpoint(self, checkpoint_manager):
        """Test clearing a checkpoint that doesn't exist."""
        cleared = checkpoint_manager.clear_checkpoint("nonexistent")
        assert cleared is False

    def test_list_sessions(self, checkpoint_manager):
        """Test listing sessions."""
        # Create multiple sessions
        checkpoint_manager.create_session("session-1", total_queries=50)
        checkpoint_manager.create_session("session-2", total_queries=100)

        # Complete one
        progress = checkpoint_manager.load_checkpoint("session-1")
        progress.status = "completed"
        checkpoint_manager.save_checkpoint("session-1", progress)

        sessions = checkpoint_manager.list_sessions()

        assert len(sessions) == 2

    def test_get_pending_queries(self, checkpoint_manager):
        """Test getting pending queries."""
        checkpoint_manager.create_session(
            session_id="test-001",
            total_queries=100,
            personas=["freshman"],
        )

        pending = checkpoint_manager.get_pending_queries(
            session_id="test-001",
            persona="freshman",
        )

        # Should have pending queries (exact count depends on implementation)
        assert isinstance(pending, list)


class TestCheckpointFileIntegrity:
    """Tests for checkpoint file integrity."""

    def test_handles_corrupted_file(self, temp_checkpoint_dir):
        """Test handling of corrupted checkpoint file."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Create a corrupted file
        checkpoint_path = Path(temp_checkpoint_dir) / "checkpoint_corrupted.json"
        with open(checkpoint_path, "w") as f:
            f.write("{ invalid json }")

        loaded = manager.load_checkpoint("corrupted")

        assert loaded is None

    def test_atomic_write(self, checkpoint_manager, sample_progress):
        """Test that writes are atomic (no partial files)."""
        checkpoint_manager.save_checkpoint("test-001", sample_progress)

        # Should not have temp files
        temp_files = list(checkpoint_manager.checkpoint_dir.glob("*.tmp"))
        assert len(temp_files) == 0

        # Should have checkpoint file
        checkpoint_files = list(checkpoint_manager.checkpoint_dir.glob("checkpoint_*.json"))
        assert len(checkpoint_files) == 1
