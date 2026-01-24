"""
Unit tests for automation domain entities.

These tests specify the intended behavior of Persona, TestCase, and TestSession entities.
Following Test-First DDD approach for greenfield development.
"""

from datetime import datetime

from src.rag.automation.domain.entities import (
    DifficultyLevel,
    Persona,
    PersonaType,
    QueryType,
    TestCase,
    TestSession,
)


class TestPersona:
    """Test Persona entity behavior."""

    def test_persona_creation_freshman(self):
        """WHEN creating a freshman persona, THEN it should have correct attributes."""
        persona = Persona(
            persona_type=PersonaType.FRESHMAN,
            name="신입생",
            description="학교 시스템에 익숙하지 않음, 비공식적 표현",
            characteristics=["비공식적 표현 사용", "학교 시스템 미숙"],
            query_styles=["간단한 질문", "구어체"],
        )

        assert persona.persona_type == PersonaType.FRESHMAN
        assert persona.name == "신입생"
        assert len(persona.characteristics) == 2
        assert len(persona.query_styles) == 2

    def test_persona_all_types_defined(self):
        """THEN all 10 persona types should be defined."""
        expected_types = {
            PersonaType.FRESHMAN,
            PersonaType.JUNIOR,
            PersonaType.GRADUATE,
            PersonaType.NEW_PROFESSOR,
            PersonaType.PROFESSOR,
            PersonaType.NEW_STAFF,
            PersonaType.STAFF_MANAGER,
            PersonaType.PARENT,
            PersonaType.DISTRESSED_STUDENT,
            PersonaType.DISSATISFIED_MEMBER,
        }

        assert len(PersonaType) == 10
        assert set(PersonaType) == expected_types

    def test_persona_equality(self):
        """WHEN comparing two personas with same type, THEN they should be equal."""
        persona1 = Persona(
            persona_type=PersonaType.FRESHMAN,
            name="신입생",
            description="학교 시스템에 익숴하지 않음",
        )
        persona2 = Persona(
            persona_type=PersonaType.FRESHMAN,
            name="Freshman",
            description="Not familiar with school system",
        )

        # Personas are equal if type is the same
        assert persona1.persona_type == persona2.persona_type


class TestTestCase:
    """Test TestCase entity behavior."""

    def test_test_case_creation_easy(self):
        """WHEN creating an easy test case, THEN it should have correct difficulty."""
        test_case = TestCase(
            query="휴학 신청은 어떻게 하나요?",
            persona_type=PersonaType.FRESHMAN,
            difficulty=DifficultyLevel.EASY,
            query_type=QueryType.PROCEDURAL,
            intent_analysis=None,  # Will be set by QueryGenerator
        )

        assert test_case.difficulty == DifficultyLevel.EASY
        assert test_case.query_type == QueryType.PROCEDURAL
        assert test_case.persona_type == PersonaType.FRESHMAN
        assert test_case.query == "휴학 신청은 어떻게 하나요?"

    def test_test_case_with_intent(self):
        """WHEN creating test case with intent analysis, THEN it should preserve intent."""
        from src.rag.automation.domain.value_objects import IntentAnalysis

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

        assert test_case.intent_analysis is not None
        assert test_case.intent_analysis.surface_intent == "휴학 절차 문의"
        assert test_case.intent_analysis.behavioral_intent == "휴학 신청서 제출"

    def test_test_case_all_query_types(self):
        """THEN all query types should be defined."""
        expected_types = {
            QueryType.FACT_CHECK,
            QueryType.PROCEDURAL,
            QueryType.ELIGIBILITY,
            QueryType.COMPARISON,
            QueryType.AMBIGUOUS,
            QueryType.EMOTIONAL,
            QueryType.COMPLEX,
            QueryType.SLANG,
        }

        assert len(QueryType) == 8
        assert set(QueryType) == expected_types


class TestTestSession:
    """Test TestSession entity behavior."""

    def test_test_session_creation(self):
        """WHEN creating a test session, THEN it should initialize correctly."""
        session = TestSession(
            session_id="test-session-001",
            started_at=datetime.now(),
            total_test_cases=0,
            test_cases=[],
        )

        assert session.session_id == "test-session-001"
        assert session.total_test_cases == 0
        assert len(session.test_cases) == 0
        assert session.completed_at is None

    def test_test_session_add_test_case(self):
        """WHEN adding a test case, THEN it should update the count."""
        session = TestSession(
            session_id="test-session-002",
            started_at=datetime.now(),
            total_test_cases=0,
            test_cases=[],
        )

        test_case = TestCase(
            query="휴학 신청은 어떻게 하나요?",
            persona_type=PersonaType.FRESHMAN,
            difficulty=DifficultyLevel.EASY,
            query_type=QueryType.PROCEDURAL,
        )

        session.test_cases.append(test_case)
        session.total_test_cases = len(session.test_cases)

        assert session.total_test_cases == 1
        assert len(session.test_cases) == 1

    def test_test_session_completion(self):
        """WHEN completing a session, THEN it should set completion time."""
        started = datetime.now()
        session = TestSession(
            session_id="test-session-003",
            started_at=started,
            total_test_cases=0,
            test_cases=[],
        )

        completed = datetime.now()
        session.completed_at = completed

        assert session.completed_at is not None
        assert session.completed_at >= started

    def test_test_session_metadata(self):
        """WHEN creating session with metadata, THEN it should preserve metadata."""
        metadata = {
            "rag_version": "1.0.0",
            "test_framework": "automation-v1",
            "total_personas_tested": 10,
        }

        session = TestSession(
            session_id="test-session-004",
            started_at=datetime.now(),
            total_test_cases=0,
            test_cases=[],
            metadata=metadata,
        )

        assert session.metadata == metadata
        assert session.metadata["total_personas_tested"] == 10
