"""
Unit tests for JSONSessionRepository infrastructure.

Tests session persistence and retrieval.
"""

import json
import tempfile
from datetime import datetime

import pytest

from src.rag.automation.domain.entities import (
    DifficultyLevel,
    PersonaType,
    QueryType,
    TestCase,
    TestSession,
)
from src.rag.automation.domain.value_objects import IntentAnalysis
from src.rag.automation.infrastructure.json_session_repository import (
    JSONSessionRepository,
)


class TestJSONSessionRepository:
    """Test JSONSessionRepository functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def repository(self, temp_dir):
        """Create repository with temporary directory."""
        return JSONSessionRepository(base_path=temp_dir)

    @pytest.fixture
    def sample_session(self):
        """Create sample test session."""
        intent = IntentAnalysis(
            surface_intent="휴학 절차 문의",
            hidden_intent="휴학 신청 방법과 필요 서류 확인",
            behavioral_intent="휴학 신청서 제출",
        )

        test_case = TestCase(
            query="휴학 신청은 어떻게 하나요?",
            persona_type=PersonaType.FRESHMAN,
            difficulty=DifficultyLevel.EASY,
            query_type=QueryType.PROCEDURAL,
            intent_analysis=intent,
        )

        return TestSession(
            session_id="test-session-001",
            started_at=datetime.fromisoformat("2025-01-24T10:00:00"),
            total_test_cases=1,
            test_cases=[test_case],
            metadata={"test_framework": "automation-v1"},
        )

    def test_save_session(self, repository, sample_session):
        """WHEN saving session, THEN should create JSON file."""
        repository.save(sample_session)

        session_path = repository._get_session_path("test-session-001")
        assert session_path.exists()

    def test_load_session(self, repository, sample_session):
        """WHEN loading session, THEN should return correct data."""
        repository.save(sample_session)

        loaded = repository.load("test-session-001")

        assert loaded is not None
        assert loaded.session_id == "test-session-001"
        assert loaded.total_test_cases == 1
        assert len(loaded.test_cases) == 1

    def test_load_nonexistent_session(self, repository):
        """WHEN loading nonexistent session, THEN should return None."""
        loaded = repository.load("nonexistent")

        assert loaded is None

    def test_loaded_session_preserves_intent(self, repository, sample_session):
        """WHEN loading session, THEN intent analysis should be preserved."""
        repository.save(sample_session)

        loaded = repository.load("test-session-001")
        test_case = loaded.test_cases[0]

        assert test_case.intent_analysis is not None
        assert test_case.intent_analysis.surface_intent == "휴학 절차 문의"
        assert test_case.intent_analysis.behavioral_intent == "휴학 신청서 제출"

    def test_list_all_sessions(self, repository):
        """WHEN listing sessions, THEN should return all sessions."""
        # Create multiple sessions
        for i in range(3):
            test_case = TestCase(
                query=f"Query {i}",
                persona_type=PersonaType.FRESHMAN,
                difficulty=DifficultyLevel.EASY,
                query_type=QueryType.PROCEDURAL,
            )

            session = TestSession(
                session_id=f"session-{i}",
                started_at=datetime.fromisoformat("2025-01-24T10:00:00"),
                total_test_cases=1,
                test_cases=[test_case],
            )

            repository.save(session)

        sessions = repository.list_all()

        assert len(sessions) == 3

    def test_delete_session(self, repository, sample_session):
        """WHEN deleting session, THEN should remove file."""
        repository.save(sample_session)

        result = repository.delete("test-session-001")

        assert result is True

        session_path = repository._get_session_path("test-session-001")
        assert not session_path.exists()

    def test_delete_nonexistent_session(self, repository):
        """WHEN deleting nonexistent session, THEN should return False."""
        result = repository.delete("nonexistent")

        assert result is False

    def test_json_structure(self, repository, sample_session, temp_dir):
        """WHEN saving session, THEN JSON should have correct structure."""
        repository.save(sample_session)

        session_path = repository._get_session_path("test-session-001")

        with open(session_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["session_id"] == "test-session-001"
        assert "started_at" in data
        assert "test_cases" in data
        assert len(data["test_cases"]) == 1
        assert data["test_cases"][0]["query"] == "휴학 신청은 어떻게 하나요?"
