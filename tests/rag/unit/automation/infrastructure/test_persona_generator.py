"""
Unit tests for PersonaGenerator infrastructure.

Tests persona generation and difficulty distribution logic.
"""


from src.rag.automation.domain.entities import PersonaType
from src.rag.automation.domain.value_objects import DifficultyDistribution
from src.rag.automation.infrastructure.llm_persona_generator import PersonaGenerator


class TestPersonaGenerator:
    """Test PersonaGenerator functionality."""

    def test_get_all_personas_returns_10(self):
        """WHEN getting all personas, THEN should return exactly 10."""
        personas = PersonaGenerator.get_all_personas()

        assert len(personas) == 10

    def test_all_persona_types_present(self):
        """THEN all persona types should be present."""
        personas = PersonaGenerator.get_all_personas()
        persona_types = {p.persona_type for p in personas}

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

        assert persona_types == expected_types

    def test_get_persona_by_type(self):
        """WHEN getting persona by type, THEN should return correct persona."""
        freshman = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        assert freshman.persona_type == PersonaType.FRESHMAN
        assert freshman.name == "신입생"
        assert "학교 시스템에 익숙하지 않음" in freshman.description

    def test_difficulty_distribution_default(self):
        """WHEN getting default distribution, THEN should be 30-40-30."""
        distribution = PersonaGenerator.get_difficulty_distribution()

        assert distribution.easy_ratio == 0.3
        assert distribution.medium_ratio == 0.4
        assert distribution.hard_ratio == 0.3

    def test_calculate_test_case_counts(self):
        """WHEN calculating counts for 10 tests, THEN should be 3-4-3."""
        distribution = DifficultyDistribution(
            easy_ratio=0.3,
            medium_ratio=0.4,
            hard_ratio=0.3,
        )

        counts = PersonaGenerator.calculate_test_case_counts(10, distribution)

        assert counts["easy"] == 3
        assert counts["medium"] == 4
        assert counts["hard"] == 3
        assert sum(counts.values()) == 10

    def test_calculate_test_case_counts_rounding(self):
        """WHEN calculating counts for 11 tests, THEN rounding should work."""
        distribution = DifficultyDistribution(
            easy_ratio=0.3,
            medium_ratio=0.4,
            hard_ratio=0.3,
        )

        counts = PersonaGenerator.calculate_test_case_counts(11, distribution)

        assert counts["easy"] == 3  # 11 * 0.3 = 3.3 -> 3
        assert counts["medium"] == 4  # 11 * 0.4 = 4.4 -> 4
        assert counts["hard"] == 4  # remainder
        assert sum(counts.values()) == 11

    def test_persona_has_required_fields(self):
        """THEN each persona should have required fields."""
        personas = PersonaGenerator.get_all_personas()

        for persona in personas:
            assert persona.name
            assert persona.description
            assert len(persona.characteristics) > 0
            assert len(persona.query_styles) > 0
            assert len(persona.context_hints) > 0
