"""
Unit tests for GenerateTestUseCase application.

Tests test generation orchestration.
"""

import tempfile

import pytest

from src.rag.automation.application.generate_test_usecase import GenerateTestUseCase
from src.rag.automation.domain.entities import PersonaType
from src.rag.automation.infrastructure.json_session_repository import (
    JSONSessionRepository,
)


class TestGenerateTestUseCase:
    """Test GenerateTestUseCase functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def use_case(self, temp_dir):
        """Create use case with temporary repository."""
        repository = JSONSessionRepository(base_path=temp_dir)
        return GenerateTestUseCase(session_repository=repository)

    def test_execute_creates_session(self, use_case):
        """WHEN executing use case, THEN should create test session."""
        session = use_case.execute(
            session_id="test-001",
            tests_per_persona=3,
        )

        assert session.session_id == "test-001"
        assert session.total_test_cases == 30  # 10 personas * 3 tests
        assert len(session.test_cases) == 30

    def test_execute_with_custom_distribution(self, use_case):
        """WHEN using custom distribution, THEN should apply correctly."""
        from src.rag.automation.domain.value_objects import DifficultyDistribution

        custom_dist = DifficultyDistribution(
            easy_ratio=0.5,
            medium_ratio=0.3,
            hard_ratio=0.2,
        )

        session = use_case.execute(
            session_id="test-002",
            tests_per_persona=10,
            difficulty_distribution=custom_dist,
        )

        # 10 personas * 10 tests = 100 total
        assert session.total_test_cases == 100

        # Count difficulties
        from src.rag.automation.domain.entities import DifficultyLevel

        easy_count = sum(
            1 for tc in session.test_cases if tc.difficulty == DifficultyLevel.EASY
        )
        medium_count = sum(
            1 for tc in session.test_cases if tc.difficulty == DifficultyLevel.MEDIUM
        )
        hard_count = sum(
            1 for tc in session.test_cases if tc.difficulty == DifficultyLevel.HARD
        )

        assert easy_count == 50  # 50%
        assert medium_count == 30  # 30%
        assert hard_count == 20  # 20%

    def test_execute_for_single_persona(self, use_case):
        """WHEN executing for single persona, THEN should create only that persona."""
        session = use_case.execute_for_persona(
            session_id="test-003",
            persona_type="freshman",
            tests_per_difficulty={"easy": 2, "medium": 2, "hard": 2},
        )

        assert session.total_test_cases == 6

        # All test cases should be for freshman
        assert all(tc.persona_type == PersonaType.FRESHMAN for tc in session.test_cases)

    def test_execute_saves_session(self, use_case):
        """WHEN executing, THEN should save session to repository."""
        session = use_case.execute(
            session_id="test-004",
            tests_per_persona=1,
        )

        # Load from repository
        loaded = use_case.get_session("test-004")

        assert loaded is not None
        assert loaded.session_id == "test-004"
        assert loaded.total_test_cases == 10  # 10 personas * 1 test

    def test_get_session_nonexistent(self, use_case):
        """WHEN getting nonexistent session, THEN should return None."""
        loaded = use_case.get_session("nonexistent")

        assert loaded is None

    def test_list_all_sessions(self, use_case):
        """WHEN listing sessions, THEN should return all created sessions."""
        # Create 3 sessions
        for i in range(3):
            use_case.execute(
                session_id=f"test-{i:03d}",
                tests_per_persona=1,
            )

        sessions = use_case.list_all_sessions()

        assert len(sessions) == 3

    def test_all_test_cases_have_intent(self, use_case):
        """WHEN executing, THEN all test cases should have intent analysis."""
        session = use_case.execute(
            session_id="test-005",
            tests_per_persona=3,
        )

        assert all(tc.intent_analysis is not None for tc in session.test_cases)

    def test_metadata_preserved(self, use_case):
        """WHEN executing with metadata, THEN should be preserved."""
        metadata = {
            "rag_version": "1.0.0",
            "test_framework": "automation-v1",
        }

        session = use_case.execute(
            session_id="test-006",
            tests_per_persona=1,
            metadata=metadata,
        )

        assert session.metadata == metadata

    def test_session_has_all_persona_types(self, use_case):
        """WHEN executing, THEN session should include all persona types."""
        session = use_case.execute(
            session_id="test-007",
            tests_per_persona=1,
        )

        persona_types = {tc.persona_type for tc in session.test_cases}

        # Should have all 10 persona types
        assert len(persona_types) == 10
