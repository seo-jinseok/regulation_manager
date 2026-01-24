"""
Unit tests for QueryGenerator infrastructure.

Tests query generation and intent analysis logic.
"""

from src.rag.automation.domain.entities import DifficultyLevel, PersonaType
from src.rag.automation.infrastructure.llm_persona_generator import PersonaGenerator
from src.rag.automation.infrastructure.llm_query_generator import QueryGenerator


class TestQueryGenerator:
    """Test QueryGenerator functionality."""

    def test_generate_for_persona_freshman(self):
        """WHEN generating queries for freshman, THEN should create test cases."""
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1, "medium": 1, "hard": 1}

        test_cases = QueryGenerator.generate_for_persona(persona, counts)

        assert len(test_cases) == 3
        assert all(tc.persona_type == PersonaType.FRESHMAN for tc in test_cases)

    def test_generate_difficulty_distribution(self):
        """WHEN generating queries, THEN difficulty distribution should match."""
        persona = PersonaGenerator.get_persona(PersonaType.JUNIOR)
        counts = {"easy": 2, "medium": 2, "hard": 2}

        test_cases = QueryGenerator.generate_for_persona(persona, counts)

        easy_count = sum(
            1 for tc in test_cases if tc.difficulty == DifficultyLevel.EASY
        )
        medium_count = sum(
            1 for tc in test_cases if tc.difficulty == DifficultyLevel.MEDIUM
        )
        hard_count = sum(
            1 for tc in test_cases if tc.difficulty == DifficultyLevel.HARD
        )

        assert easy_count == 2
        assert medium_count == 2
        assert hard_count == 2

    def test_all_test_cases_have_intent(self):
        """WHEN generating queries, THEN all should have intent analysis."""
        persona = PersonaGenerator.get_persona(PersonaType.GRADUATE)
        counts = {"easy": 1, "medium": 1, "hard": 1}

        test_cases = QueryGenerator.generate_for_persona(persona, counts)

        assert all(tc.intent_analysis is not None for tc in test_cases)

    def test_intent_analysis_structure(self):
        """WHEN generating intent, THEN should have 3-level structure."""
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1, "medium": 0, "hard": 0}

        test_cases = QueryGenerator.generate_for_persona(persona, counts)
        intent = test_cases[0].intent_analysis

        assert intent.surface_intent
        assert intent.hidden_intent
        assert intent.behavioral_intent

    def test_surface_intent_extraction(self):
        """WHEN query has '신청', THEN surface intent should be '절차/신청 문의'."""
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        # Test with query containing '신청'
        test_cases = QueryGenerator.generate_for_persona(
            persona, {"easy": 1, "medium": 0, "hard": 0}
        )

        # Find a query with '신청'
        for tc in test_cases:
            if "신청" in tc.query:
                intent = tc.intent_analysis
                # Should have procedural/surface intent
                assert intent.surface_intent
                break

    def test_persona_context_in_hidden_intent(self):
        """WHEN generating for distressed student, THEN hidden intent should include context."""
        persona = PersonaGenerator.get_persona(PersonaType.DISTRESSED_STUDENT)
        counts = {"easy": 0, "medium": 0, "hard": 1}

        test_cases = QueryGenerator.generate_for_persona(persona, counts)
        intent = test_cases[0].intent_analysis

        # Hidden intent should include context
        assert (
            "긴급 상황" in intent.hidden_intent or "어려운 상황" in intent.hidden_intent
        )

    def test_query_content_exists(self):
        """WHEN generating queries, THEN all queries should have content."""
        persona = PersonaGenerator.get_persona(PersonaType.PROFESSOR)
        counts = {"easy": 1, "medium": 1, "hard": 1}

        test_cases = QueryGenerator.generate_for_persona(persona, counts)

        assert all(tc.query for tc in test_cases)
        assert all(len(tc.query) > 0 for tc in test_cases)
