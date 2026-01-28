"""
Test Extended Domain Entities for RAG Testing Automation.

Tests extended personas, ambiguous queries, multi-turn scenarios,
and edge cases defined in the extended domain layer.
"""

import pytest

from src.rag.automation.domain.entities import (
    DifficultyLevel,
    FollowUpType,
    PersonaType,
)
from src.rag.automation.domain.extended_entities import (
    EXTENDED_PERSONA_DEFINITIONS,
    AmbiguityType,
    AmbiguousQuery,
    EdgeCaseCategory,
    EdgeCaseScenario,
    ExtendedPersona,
    ExtendedPersonaType,
    MultiTurnConversationScenario,
    get_all_extended_personas,
    get_extended_persona,
)


class TestExtendedPersonaType:
    """Tests for ExtendedPersonaType enum."""

    def test_all_extended_persona_types_defined(self):
        """Test that all 10 extended persona types are defined."""
        expected_types = {
            ExtendedPersonaType.INTERNATIONAL_STUDENT,
            ExtendedPersonaType.ADJUNCT_PROFESSOR,
            ExtendedPersonaType.RESEARCHER,
            ExtendedPersonaType.APPLICANT,
            ExtendedPersonaType.COMMUNITY_MEMBER,
            ExtendedPersonaType.TRANSFER_STUDENT,
            ExtendedPersonaType.RETURNING_STUDENT,
            ExtendedPersonaType.RETIREE_STUDENT,
            ExtendedPersonaType.ONLINE_STUDENT,
            ExtendedPersonaType.DISABLED_STUDENT,
        }
        assert set(ExtendedPersonaType) == expected_types

    def test_persona_type_values(self):
        """Test that persona type values are correctly defined."""
        assert (
            ExtendedPersonaType.INTERNATIONAL_STUDENT.value == "international_student"
        )
        assert ExtendedPersonaType.ADJUNCT_PROFESSOR.value == "adjunct_professor"
        assert ExtendedPersonaType.RESEARCHER.value == "researcher"


class TestExtendedPersona:
    """Tests for ExtendedPersona dataclass."""

    def test_extended_persona_attributes(self):
        """Test that ExtendedPersona has all required attributes."""
        persona = ExtendedPersona(
            persona_type=PersonaType.FRESHMAN,
            name="Test Persona",
            description="Test description",
            characteristics=["test"],
            query_styles=["test style"],
            context_hints=["test hint"],
            language_proficiency="native",
            cultural_context=["test"],
            technical_expertise="basic",
            urgency_level="normal",
            accessibility_needs=[],
        )
        assert persona.persona_type == PersonaType.FRESHMAN
        assert persona.name == "Test Persona"
        assert persona.language_proficiency == "native"
        assert persona.technical_expertise == "basic"
        assert persona.urgency_level == "normal"

    def test_international_student_persona(self):
        """Test international student persona definition."""
        persona = EXTENDED_PERSONA_DEFINITIONS[
            ExtendedPersonaType.INTERNATIONAL_STUDENT
        ]
        assert persona.name == "유학생"
        assert "언어" in persona.description and "장벽" in persona.description
        assert persona.language_proficiency == "intermediate"
        assert "student_visa" in persona.cultural_context

    def test_adjunct_professor_persona(self):
        """Test adjunct professor persona definition."""
        persona = EXTENDED_PERSONA_DEFINITIONS[ExtendedPersonaType.ADJUNCT_PROFESSOR]
        assert persona.name == "시간강사"
        assert "비정규직" in persona.description
        assert persona.urgency_level == "high"

    def test_researcher_persona(self):
        """Test researcher persona definition."""
        persona = EXTENDED_PERSONA_DEFINITIONS[ExtendedPersonaType.RESEARCHER]
        assert persona.name == "연구원"
        assert "연구 중심" in persona.description
        assert persona.technical_expertise == "advanced"

    def test_applicant_persona(self):
        """Test applicant persona definition."""
        persona = EXTENDED_PERSONA_DEFINITIONS[ExtendedPersonaType.APPLICANT]
        assert persona.name == "입학 지원자"
        assert "입학 절차" in persona.description

    def test_disabled_student_persona(self):
        """Test disabled student persona definition."""
        persona = EXTENDED_PERSONA_DEFINITIONS[ExtendedPersonaType.DISABLED_STUDENT]
        assert persona.name == "장애학생"
        assert "접근성" in persona.description
        assert len(persona.accessibility_needs) > 0
        assert persona.urgency_level == "high"


class TestAmbiguityType:
    """Tests for AmbiguityType enum."""

    def test_all_ambiguity_types_defined(self):
        """Test that all 5 ambiguity types are defined."""
        expected_types = {
            AmbiguityType.MISSING_CONTEXT,
            AmbiguityType.MULTIPLE_INTERPRETATIONS,
            AmbiguityType.UNCLEAR_INTENT,
            AmbiguityType.VAGUE_TERMINOLOGY,
            AmbiguityType.INCOMPLETE_THOUGHT,
        }
        assert set(AmbiguityType) == expected_types

    def test_ambiguity_type_values(self):
        """Test that ambiguity type values are correctly defined."""
        assert AmbiguityType.MISSING_CONTEXT.value == "missing_context"
        assert (
            AmbiguityType.MULTIPLE_INTERPRETATIONS.value == "multiple_interpretations"
        )


class TestAmbiguousQuery:
    """Tests for AmbiguousQuery dataclass."""

    def test_ambiguous_query_creation(self):
        """Test creating an ambiguous query."""
        query = AmbiguousQuery(
            query_id="amb-001",
            query="그거 언제까지야?",
            ambiguity_type=AmbiguityType.MISSING_CONTEXT,
            difficulty=DifficultyLevel.MEDIUM,
            persona_type=PersonaType.FRESHMAN,
            context_hints=["deadline"],
            expected_interpretations=["휴학 마감", "장학금 마감"],
            expected_clarifications=["절차 명시 필요"],
        )
        assert query.query_id == "amb-001"
        assert query.ambiguity_type == AmbiguityType.MISSING_CONTEXT
        assert query.difficulty == DifficultyLevel.MEDIUM
        assert len(query.expected_interpretations) == 2

    def test_ambiguous_query_default_values(self):
        """Test ambiguous query default values."""
        query = AmbiguousQuery(
            query_id="amb-002",
            query="test query",
            ambiguity_type=AmbiguityType.UNCLEAR_INTENT,
            difficulty=DifficultyLevel.HARD,
        )
        assert query.should_detect_ambiguity is True
        assert query.should_request_clarification is True
        assert query.context_hints == []
        assert query.expected_interpretations == []


class TestEdgeCaseCategory:
    """Tests for EdgeCaseCategory enum."""

    def test_all_edge_case_categories_defined(self):
        """Test that all 8 edge case categories are defined."""
        expected_categories = {
            EdgeCaseCategory.EMOTIONAL,
            EdgeCaseCategory.COMPLEX_SYNTHESIS,
            EdgeCaseCategory.CROSS_REFERENCED,
            EdgeCaseCategory.DEADLINE_CRITICAL,
            EdgeCaseCategory.CONTRADICTORY,
            EdgeCaseCategory.LANGUAGE_BARRIER,
            EdgeCaseCategory.EXCEPTIONAL,
            EdgeCaseCategory.TECHNICAL,
        }
        assert set(EdgeCaseCategory) == expected_categories


class TestEdgeCaseScenario:
    """Tests for EdgeCaseScenario dataclass."""

    def test_edge_case_scenario_creation(self):
        """Test creating an edge case scenario."""
        scenario = EdgeCaseScenario(
            scenario_id="edge-001",
            name="Frustrated Student",
            category=EdgeCaseCategory.EMOTIONAL,
            difficulty=DifficultyLevel.HARD,
            persona_type=PersonaType.DISTRESSED_STUDENT,
            query="학교 너무 힘들어...",
            is_emotional=True,
            is_urgent=False,
            is_confused=True,
            is_frustrated=True,
            expected_empathy_level="empathetic",
            expected_response_speed="normal",
            should_escalate=True,
            expected_regulations=["상담 센터"],
            expected_actions=["상담 안내"],
            should_show_empathy=True,
            should_provide_contact=True,
            should_offer_alternatives=True,
        )
        assert scenario.scenario_id == "edge-001"
        assert scenario.category == EdgeCaseCategory.EMOTIONAL
        assert scenario.is_emotional is True
        assert scenario.should_escalate is True

    def test_edge_case_scenario_defaults(self):
        """Test edge case scenario default values."""
        scenario = EdgeCaseScenario(
            scenario_id="edge-002",
            name="Test Scenario",
            category=EdgeCaseCategory.DEADLINE_CRITICAL,
            difficulty=DifficultyLevel.MEDIUM,
            persona_type=PersonaType.JUNIOR,
            query="urgent query",
        )
        assert scenario.is_emotional is False
        assert scenario.is_urgent is False
        assert scenario.expected_regulations == []
        assert scenario.expected_actions == []


class TestMultiTurnConversationScenario:
    """Tests for MultiTurnConversationScenario dataclass."""

    def test_multi_turn_scenario_creation(self):
        """Test creating a multi-turn conversation scenario."""
        scenario = MultiTurnConversationScenario(
            scenario_id="mt-001",
            name="Freshman Dormitory Inquiry",
            description="Freshman asks about dormitory application",
            persona_type=PersonaType.FRESHMAN,
            difficulty=DifficultyLevel.EASY,
            context_window_size=3,
            initial_query="기숙사 어떻게 신청해요?",
            initial_expected_intent="절차 문의",
            turns=[
                {
                    "turn_number": 2,
                    "follow_up_type": FollowUpType.CLARIFICATION,
                    "query": "신청 기간은 언제인가요?",
                    "expected_intent": "기간 확인",
                }
            ],
            expected_context_preservation_rate=0.9,
            expected_topic_transitions=["신청 절차", "기간"],
        )
        assert scenario.scenario_id == "mt-001"
        assert scenario.persona_type == PersonaType.FRESHMAN
        assert scenario.context_window_size == 3
        assert len(scenario.turns) == 1

    def test_multi_turn_scenario_defaults(self):
        """Test multi-turn scenario default values."""
        scenario = MultiTurnConversationScenario(
            scenario_id="mt-002",
            name="Test Scenario",
            description="Test description",
            persona_type=PersonaType.JUNIOR,
            difficulty=DifficultyLevel.MEDIUM,
            initial_query="test query",
            initial_expected_intent="test intent",
        )
        assert scenario.context_window_size == 3
        assert scenario.expected_context_preservation_rate == 0.8
        assert scenario.turns == []


class TestExtendedPersonaFunctions:
    """Tests for extended persona utility functions."""

    def test_get_all_extended_personas(self):
        """Test getting all extended personas."""
        personas = get_all_extended_personas()
        assert len(personas) == 10
        assert all(isinstance(p, ExtendedPersona) for p in personas)

    def test_get_extended_persona_by_type(self):
        """Test getting a specific extended persona."""
        persona = get_extended_persona(ExtendedPersonaType.INTERNATIONAL_STUDENT)
        assert persona.name == "유학생"
        assert isinstance(persona, ExtendedPersona)

    def test_get_extended_persona_invalid_type(self):
        """Test getting an invalid extended persona type."""
        with pytest.raises(KeyError):
            get_extended_persona("invalid_type")


class TestExtendedPersonaCompleteness:
    """Tests for completeness of extended persona definitions."""

    @pytest.mark.parametrize(
        "persona_type",
        [
            ExtendedPersonaType.INTERNATIONAL_STUDENT,
            ExtendedPersonaType.ADJUNCT_PROFESSOR,
            ExtendedPersonaType.RESEARCHER,
            ExtendedPersonaType.APPLICANT,
            ExtendedPersonaType.COMMUNITY_MEMBER,
            ExtendedPersonaType.TRANSFER_STUDENT,
            ExtendedPersonaType.RETURNING_STUDENT,
            ExtendedPersonaType.RETIREE_STUDENT,
            ExtendedPersonaType.ONLINE_STUDENT,
            ExtendedPersonaType.DISABLED_STUDENT,
        ],
    )
    def test_all_extended_personas_have_required_attributes(self, persona_type):
        """Test that all extended personas have required attributes."""
        persona = get_extended_persona(persona_type)
        assert persona.name is not None
        assert persona.description is not None
        assert len(persona.characteristics) > 0
        assert len(persona.query_styles) > 0
        assert len(persona.context_hints) > 0
        assert persona.language_proficiency in [
            "native",
            "fluent",
            "intermediate",
            "basic",
        ]
        assert persona.technical_expertise in ["basic", "intermediate", "advanced"]
        assert persona.urgency_level in ["low", "normal", "high", "urgent"]

    def test_all_extended_personas_have_unique_names(self):
        """Test that all extended personas have unique names."""
        personas = get_all_extended_personas()
        names = [p.name for p in personas]
        assert len(names) == len(set(names))


class TestAmbiguityTypeCompleteness:
    """Tests for completeness of ambiguity type definitions."""

    @pytest.mark.parametrize(
        "ambiguity_type",
        [
            AmbiguityType.MISSING_CONTEXT,
            AmbiguityType.MULTIPLE_INTERPRETATIONS,
            AmbiguityType.UNCLEAR_INTENT,
            AmbiguityType.VAGUE_TERMINOLOGY,
            AmbiguityType.INCOMPLETE_THOUGHT,
        ],
    )
    def test_all_ambiguity_types_have_valid_values(self, ambiguity_type):
        """Test that all ambiguity types have valid values."""
        assert isinstance(ambiguity_type.value, str)
        assert len(ambiguity_type.value) > 0


class TestEdgeCaseCategoryCompleteness:
    """Tests for completeness of edge case category definitions."""

    @pytest.mark.parametrize(
        "category",
        [
            EdgeCaseCategory.EMOTIONAL,
            EdgeCaseCategory.COMPLEX_SYNTHESIS,
            EdgeCaseCategory.CROSS_REFERENCED,
            EdgeCaseCategory.DEADLINE_CRITICAL,
            EdgeCaseCategory.CONTRADICTORY,
        ],
    )
    def test_all_edge_case_categories_have_valid_values(self, category):
        """Test that all edge case categories have valid values."""
        assert isinstance(category.value, str)
        assert len(category.value) > 0
