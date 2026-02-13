"""
Unit tests for PersonaAwareGenerator.

Tests the persona prompt enhancement functionality.
"""

import pytest
from src.rag.domain.personas import PersonaAwareGenerator, PersonaPromptBuilder, create_persona_prompt


class TestPersonaAwareGenerator:
    """Unit tests for PersonaAwareGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a PersonaAwareGenerator instance."""
        return PersonaAwareGenerator()

    @pytest.fixture
    def base_prompt(self):
        """Create a base prompt for testing."""
        return "당신은 동의대학교 규정 전문가입니다.\n질문에 답변하세요."

    def test_persona_id_mapping(self, generator):
        """Test persona ID mapping from evaluator to internal names."""
        assert generator.get_persona_name("student-undergraduate") == "freshman"
        assert generator.get_persona_name("student-graduate") == "graduate"
        assert generator.get_persona_name("professor") == "professor"
        assert generator.get_persona_name("staff-admin") == "staff"
        assert generator.get_persona_name("parent") == "parent"
        assert generator.get_persona_name("student-international") == "international"

    def test_enhance_prompt_for_professor(self, generator, base_prompt):
        """Test prompt enhancement for professor persona."""
        enhanced = generator.enhance_prompt(base_prompt, "professor")

        # Should be longer than base
        assert len(enhanced) > len(base_prompt)

        # Should contain professor-specific instructions
        assert "교수님" in enhanced
        assert "공식적이고 학술적인 표현" in enhanced

    def test_enhance_prompt_for_parent(self, generator, base_prompt):
        """Test prompt enhancement for parent persona."""
        enhanced = generator.enhance_prompt(base_prompt, "parent")

        # Should be longer than base
        assert len(enhanced) > len(base_prompt)

        # Should contain parent-specific instructions
        assert "학부모" in enhanced
        assert "쉬운 용어" in enhanced

    def test_enhance_prompt_for_international(self, generator, base_prompt):
        """Test prompt enhancement for international student."""
        enhanced = generator.enhance_prompt(base_prompt, "student-international")

        # Should be longer than base
        assert len(enhanced) > len(base_prompt)

        # Should contain international-specific instructions
        assert "외국인유학생" in enhanced
        assert "English" in enhanced

    def test_supports_persona(self, generator):
        """Test persona support checking."""
        assert generator.supports_persona("professor") is True
        assert generator.supports_persona("parent") is True
        assert generator.supports_persona("student-international") is True
        assert generator.supports_persona("unknown_persona") is False

    def test_get_supported_personas(self, generator):
        """Test getting list of supported personas."""
        personas = generator.get_supported_personas()

        assert len(personas) == 6
        assert "student-undergraduate" in personas
        assert "professor" in personas
        assert "parent" in personas

    def test_get_persona_preferences(self, generator):
        """Test getting persona preferences."""
        # Professor preferences
        prof_prefs = generator.get_persona_preferences("professor")
        assert prof_prefs["detail_level"] == "comprehensive"
        assert prof_prefs["citation_style"] == "detailed"

        # Parent preferences
        parent_prefs = generator.get_persona_preferences("parent")
        assert parent_prefs["detail_level"] == "simple"
        assert parent_prefs["parent_friendly"] is True


class TestPersonaPromptBuilder:
    """Unit tests for PersonaPromptBuilder."""

    @pytest.fixture
    def base_prompt(self):
        """Create a base prompt."""
        return "Base prompt for testing."

    def test_builder_for_persona(self, base_prompt):
        """Test builder with persona enhancement."""
        builder = PersonaPromptBuilder(base_prompt)
        enhanced = builder.for_persona("professor").build()

        assert len(enhanced) > len(base_prompt)

    def test_builder_with_completeness(self, base_prompt):
        """Test builder with completeness instructions."""
        builder = PersonaPromptBuilder(base_prompt)
        enhanced = builder.with_completeness_instructions().build()

        assert "Completeness Requirements" in enhanced
        assert "누락 금지" in enhanced

    def test_builder_with_citation_quality(self, base_prompt):
        """Test builder with citation quality instructions."""
        builder = PersonaPromptBuilder(base_prompt)
        enhanced = builder.with_citation_quality_instructions().build()

        assert "Citation Quality Requirements" in enhanced
        assert "정확한 인용" in enhanced

    def test_builder_full_chain(self, base_prompt):
        """Test builder with all enhancements chained."""
        builder = PersonaPromptBuilder(base_prompt)
        enhanced = (
            builder
            .for_persona("parent")
            .with_completeness_instructions()
            .with_citation_quality_instructions()
            .build()
        )

        # Should contain all enhancements
        assert "학부모" in enhanced
        assert "Completeness Requirements" in enhanced
        assert "Citation Quality Requirements" in enhanced


class TestCreatePersonaPrompt:
    """Unit tests for create_persona_prompt convenience function."""

    def test_create_with_all_options(self):
        """Test creating prompt with all options enabled."""
        base = "Base prompt"
        enhanced = create_persona_prompt(
            base,
            persona="professor",
            include_completeness=True,
            include_citation_quality=True,
        )

        assert "교수님" in enhanced
        assert "Completeness Requirements" in enhanced
        assert "Citation Quality Requirements" in enhanced

    def test_create_minimal(self):
        """Test creating minimal prompt."""
        base = "Base prompt"
        enhanced = create_persona_prompt(
            base,
            persona="parent",
            include_completeness=False,
            include_citation_quality=False,
        )

        assert "학부모" in enhanced
        assert "Completeness Requirements" not in enhanced
        assert "Citation Quality Requirements" not in enhanced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
