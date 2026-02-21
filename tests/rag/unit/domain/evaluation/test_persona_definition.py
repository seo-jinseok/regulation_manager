"""
Unit tests for PersonaDefinition and related types.

Tests for SPEC-RAG-QUALITY-010 Milestone 6: Persona Evaluation System.
TDD RED Phase - Write failing tests first.
"""

import pytest

from src.rag.domain.evaluation.persona_definition import (
    PersonaDefinition,
    PersonaType,
    DEFAULT_PERSONAS,
)


class TestPersonaType:
    """Tests for PersonaType enum."""

    def test_all_persona_types_exist(self):
        """Test that all 6 persona types are defined."""
        expected_types = [
            "freshman",
            "student",
            "professor",
            "staff",
            "parent",
            "international",
        ]

        for type_name in expected_types:
            assert any(pt.value == type_name for pt in PersonaType), (
                f"PersonaType.{type_name.upper()} should exist"
            )

    def test_persona_type_values(self):
        """Test that persona type values match expected Korean names."""
        assert PersonaType.FRESHMAN.value == "freshman"
        assert PersonaType.STUDENT.value == "student"
        assert PersonaType.PROFESSOR.value == "professor"
        assert PersonaType.STAFF.value == "staff"
        assert PersonaType.PARENT.value == "parent"
        assert PersonaType.INTERNATIONAL.value == "international"

    def test_persona_type_count(self):
        """Test that exactly 6 persona types exist."""
        assert len(PersonaType) == 6


class TestPersonaDefinition:
    """Tests for PersonaDefinition dataclass."""

    def test_create_persona_definition(self):
        """Test creating a basic PersonaDefinition."""
        persona = PersonaDefinition(
            persona_id="freshman",
            name="신입생",
            description="대학 규정을 처음 접하는 1학년 학생",
            language_level="simple",
            citation_preference="minimal",
            key_requirements=["간단명료한 답변", "최소 인용", "친절한 설명"],
        )

        assert persona.persona_id == "freshman"
        assert persona.name == "신입생"
        assert persona.description == "대학 규정을 처음 접하는 1학년 학생"
        assert persona.language_level == "simple"
        assert persona.citation_preference == "minimal"
        assert len(persona.key_requirements) == 3

    def test_persona_definition_to_dict(self):
        """Test serialization to dictionary."""
        persona = PersonaDefinition(
            persona_id="professor",
            name="교수",
            description="교원 대상 규정",
            language_level="formal",
            citation_preference="detailed",
            key_requirements=["정책/규정 중심", "전문 용어", "조항 인용"],
        )

        data = persona.to_dict()

        assert data["persona_id"] == "professor"
        assert data["name"] == "교수"
        assert data["language_level"] == "formal"
        assert data["citation_preference"] == "detailed"

    @pytest.mark.parametrize(
        "level",
        ["simple", "normal", "formal", "technical"],
    )
    def test_valid_language_levels(self, level):
        """Test that valid language levels are accepted."""
        persona = PersonaDefinition(
            persona_id="test",
            name="테스트",
            description="테스트용",
            language_level=level,
            citation_preference="normal",
            key_requirements=[],
        )

        assert persona.language_level == level

    @pytest.mark.parametrize(
        "pref",
        ["minimal", "normal", "detailed"],
    )
    def test_valid_citation_preferences(self, pref):
        """Test that valid citation preferences are accepted."""
        persona = PersonaDefinition(
            persona_id="test",
            name="테스트",
            description="테스트용",
            language_level="normal",
            citation_preference=pref,
            key_requirements=[],
        )

        assert persona.citation_preference == pref


class TestDefaultPersonas:
    """Tests for DEFAULT_PERSONAS dictionary."""

    def test_default_personas_exist(self):
        """Test that DEFAULT_PERSONAS is defined and contains all personas."""
        assert DEFAULT_PERSONAS is not None
        assert isinstance(DEFAULT_PERSONAS, dict)
        assert len(DEFAULT_PERSONAS) == 6

    def test_freshman_persona_definition(self):
        """Test that freshman persona is correctly defined."""
        freshman = DEFAULT_PERSONAS["freshman"]

        assert freshman.persona_id == "freshman"
        assert freshman.name == "신입생"
        assert freshman.language_level == "simple"
        assert freshman.citation_preference == "minimal"
        assert "간단명료한 답변" in freshman.key_requirements
        assert "최소 인용" in freshman.key_requirements
        assert "친절한 설명" in freshman.key_requirements

    def test_student_persona_definition(self):
        """Test that student (재학생) persona is correctly defined."""
        student = DEFAULT_PERSONAS["student"]

        assert student.persona_id == "student"
        assert student.name == "재학생"
        assert student.language_level == "normal"
        assert "절차 중심" in student.key_requirements
        assert "구체적 안내" in student.key_requirements
        assert "실용적 정보" in student.key_requirements

    def test_professor_persona_definition(self):
        """Test that professor persona is correctly defined."""
        professor = DEFAULT_PERSONAS["professor"]

        assert professor.persona_id == "professor"
        assert professor.name == "교수"
        assert professor.language_level == "technical"
        assert professor.citation_preference == "detailed"
        assert "정책/규정 중심" in professor.key_requirements
        assert "전문 용어" in professor.key_requirements
        assert "조항 인용" in professor.key_requirements

    def test_staff_persona_definition(self):
        """Test that staff persona is correctly defined."""
        staff = DEFAULT_PERSONAS["staff"]

        assert staff.persona_id == "staff"
        assert staff.name == "직원"
        assert staff.language_level == "formal"
        assert "행정 절차" in staff.key_requirements
        assert "담당 부서 정보" in staff.key_requirements
        assert "처리 기한" in staff.key_requirements

    def test_parent_persona_definition(self):
        """Test that parent persona is correctly defined."""
        parent = DEFAULT_PERSONAS["parent"]

        assert parent.persona_id == "parent"
        assert parent.name == "학부모"
        assert parent.language_level == "simple"
        assert "친절한 설명" in parent.key_requirements
        assert "연락처 포함" in parent.key_requirements
        assert "이해하기 쉬운 용어" in parent.key_requirements

    def test_international_persona_definition(self):
        """Test that international student persona is correctly defined."""
        intl = DEFAULT_PERSONAS["international"]

        assert intl.persona_id == "international"
        assert intl.name == "외국인 유학생"
        assert "간단한 한국어" in intl.key_requirements
        assert "복잡한 용어 설명" in intl.key_requirements

    def test_all_personas_have_required_fields(self):
        """Test that all personas have all required fields populated."""
        required_fields = ["persona_id", "name", "description", "language_level",
                          "citation_preference", "key_requirements"]

        for persona_id, persona in DEFAULT_PERSONAS.items():
            for field in required_fields:
                assert hasattr(persona, field), (
                    f"Persona {persona_id} missing field {field}"
                )
                value = getattr(persona, field)
                if field == "key_requirements":
                    assert isinstance(value, list) and len(value) > 0, (
                        f"Persona {persona_id} has empty key_requirements"
                    )
                else:
                    assert value is not None and value != "", (
                        f"Persona {persona_id} has empty {field}"
                    )


class TestPersonaDefinitionFromPersonaType:
    """Tests for getting PersonaDefinition from PersonaType."""

    def test_get_persona_by_type(self):
        """Test getting persona definition by PersonaType enum."""
        persona = DEFAULT_PERSONAS[PersonaType.FRESHMAN.value]
        assert persona.name == "신입생"

        persona = DEFAULT_PERSONAS[PersonaType.PROFESSOR.value]
        assert persona.name == "교수"

    def test_get_all_personas_from_types(self):
        """Test that all PersonaTypes map to valid personas."""
        for persona_type in PersonaType:
            persona = DEFAULT_PERSONAS[persona_type.value]
            assert persona is not None
            assert persona.persona_id == persona_type.value
