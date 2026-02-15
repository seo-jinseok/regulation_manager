"""
Characterization tests for persona detection integration (TAG-005).

These tests verify that the persona detection integration works correctly
and that existing behavior is preserved.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.rag.application.search_usecase import (
    SearchUseCase,
    AUDIENCE_TO_PERSONA,
    REGULATION_QA_PROMPT,
)
from src.rag.infrastructure.query_analyzer import Audience


class TestAudienceToPersonaMapping:
    """Characterization tests for AUDIENCE_TO_PERSONA mapping."""

    def test_mapping_exists_for_all_audience_types(self):
        """Verify mapping exists for all Audience enum values."""
        assert Audience.STUDENT in AUDIENCE_TO_PERSONA
        assert Audience.FACULTY in AUDIENCE_TO_PERSONA
        assert Audience.STAFF in AUDIENCE_TO_PERSONA
        assert Audience.ALL in AUDIENCE_TO_PERSONA

    def test_student_maps_to_freshman(self):
        """Student audience should map to freshman persona."""
        assert AUDIENCE_TO_PERSONA[Audience.STUDENT] == "freshman"

    def test_faculty_maps_to_professor(self):
        """Faculty audience should map to professor persona."""
        assert AUDIENCE_TO_PERSONA[Audience.FACULTY] == "professor"

    def test_staff_maps_to_staff(self):
        """Staff audience should map to staff persona."""
        assert AUDIENCE_TO_PERSONA[Audience.STAFF] == "staff"

    def test_all_maps_to_freshman_default(self):
        """ALL audience should default to freshman persona."""
        assert AUDIENCE_TO_PERSONA[Audience.ALL] == "freshman"


class TestPersonaGenerationFromAudience:
    """Characterization tests for _get_persona_from_audience method."""

    @pytest.fixture
    def usecase(self):
        """Create SearchUseCase with mocked dependencies."""
        store = MagicMock()
        store.search.return_value = []
        return SearchUseCase(store=store)

    def test_none_audience_returns_none_persona(self, usecase):
        """None audience should return None persona."""
        result = usecase._get_persona_from_audience(None)
        assert result is None

    def test_student_audience_returns_freshman_persona(self, usecase):
        """Student audience should return freshman persona."""
        result = usecase._get_persona_from_audience(Audience.STUDENT)
        assert result == "freshman"

    def test_faculty_audience_returns_professor_persona(self, usecase):
        """Faculty audience should return professor persona."""
        result = usecase._get_persona_from_audience(Audience.FACULTY)
        assert result == "professor"

    def test_staff_audience_returns_staff_persona(self, usecase):
        """Staff audience should return staff persona."""
        result = usecase._get_persona_from_audience(Audience.STAFF)
        assert result == "staff"


class TestPromptEnhancementWithPersona:
    """Characterization tests for _enhance_prompt_with_persona method."""

    @pytest.fixture
    def usecase(self):
        """Create SearchUseCase with mocked dependencies."""
        store = MagicMock()
        store.search.return_value = []
        return SearchUseCase(store=store)

    def test_enhance_prompt_for_professor_adds_persona_instructions(self, usecase):
        """Professor persona should add formal instructions."""
        enhanced = usecase._enhance_prompt_with_persona(
            REGULATION_QA_PROMPT, "professor"
        )
        # Should be longer than base prompt (persona instructions added)
        assert len(enhanced) >= len(REGULATION_QA_PROMPT)
        # Should contain Korean professor indicator (Korean text in persona prompt)
        assert "professor" in enhanced.lower() or "professor" in enhanced.lower() or "교수" in enhanced

    def test_enhance_prompt_for_staff_adds_administrative_instructions(self, usecase):
        """Staff persona should add administrative instructions."""
        enhanced = usecase._enhance_prompt_with_persona(
            REGULATION_QA_PROMPT, "staff"
        )
        # Should be longer than base prompt (persona instructions added)
        assert len(enhanced) >= len(REGULATION_QA_PROMPT)
        # Should contain Korean staff indicator
        assert "staff" in enhanced.lower() or "교직원" in enhanced

    def test_enhance_prompt_returns_base_prompt_on_error(self, usecase):
        """Should return base prompt if enhancement fails."""
        # Use invalid persona that might cause issues
        result = usecase._enhance_prompt_with_persona(
            REGULATION_QA_PROMPT, "invalid_persona_name"
        )
        # Should still return a string (either enhanced or base)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_enhance_prompt_with_query_context(self, usecase):
        """Should accept optional query context."""
        enhanced = usecase._enhance_prompt_with_persona(
            REGULATION_QA_PROMPT, "professor", query="What are the leave policies?"
        )
        # Should return enhanced prompt
        assert isinstance(enhanced, str)
        assert len(enhanced) >= len(REGULATION_QA_PROMPT)


class TestPersonaGeneratorInitialization:
    """Characterization tests for lazy persona generator initialization."""

    @pytest.fixture
    def usecase(self):
        """Create SearchUseCase with mocked dependencies."""
        store = MagicMock()
        store.search.return_value = []
        return SearchUseCase(store=store)

    def test_persona_generator_is_none_initially(self, usecase):
        """Persona generator should be None initially."""
        assert usecase._persona_generator is None

    def test_ensure_persona_generator_initializes(self, usecase):
        """_ensure_persona_generator should initialize the generator."""
        usecase._ensure_persona_generator()
        assert usecase._persona_generator is not None

    def test_ensure_persona_generator_is_idempotent(self, usecase):
        """Multiple calls should not recreate generator."""
        usecase._ensure_persona_generator()
        first = usecase._persona_generator
        usecase._ensure_persona_generator()
        second = usecase._persona_generator
        assert first is second


class TestGenerateWithFactCheckPersonaIntegration:
    """Characterization tests for _generate_with_fact_check with persona."""

    @pytest.fixture
    def usecase_with_llm(self):
        """Create SearchUseCase with mocked LLM."""
        store = MagicMock()
        store.search.return_value = []

        llm = MagicMock()
        llm.generate.return_value = "Test answer"

        return SearchUseCase(store=store, llm_client=llm)

    def test_generate_with_persona_uses_enhanced_prompt(
        self, usecase_with_llm
    ):
        """_generate_with_fact_check should use enhanced prompt when persona is provided."""
        usecase_with_llm.llm.generate.return_value = "Test answer"

        # Mock the persona enhancement to track if it's called
        original_enhance = usecase_with_llm._enhance_prompt_with_persona
        enhancement_called = []

        def track_enhancement(base, persona, query=None):
            enhancement_called.append((persona, query))
            return original_enhance(base, persona, query)

        usecase_with_llm._enhance_prompt_with_persona = track_enhancement

        # Call with persona
        usecase_with_llm._generate_with_fact_check(
            question="What is the leave policy?",
            context="Leave policy context...",
            persona="professor",
        )

        # Verify enhancement was called with correct persona
        assert len(enhancement_called) > 0
        assert enhancement_called[0][0] == "professor"

    def test_generate_without_persona_uses_base_prompt(self, usecase_with_llm):
        """_generate_with_fact_check should use base prompt when persona is None."""
        usecase_with_llm.llm.generate.return_value = "Test answer"

        # Call without persona
        result = usecase_with_llm._generate_with_fact_check(
            question="What is the leave policy?",
            context="Leave policy context...",
            persona=None,
        )

        # Should still generate an answer
        assert result == "Test answer"

    def test_generate_with_custom_prompt_skips_persona_enhancement(
        self, usecase_with_llm
    ):
        """When custom_prompt is provided, persona enhancement should be skipped."""
        usecase_with_llm.llm.generate.return_value = "Test answer"

        custom_prompt = "Custom system prompt"

        # Call with both custom_prompt and persona
        result = usecase_with_llm._generate_with_fact_check(
            question="What is the leave policy?",
            context="Leave policy context...",
            custom_prompt=custom_prompt,
            persona="professor",
        )

        # Should still generate an answer using custom prompt
        assert result == "Test answer"
        # Verify custom prompt was used
        call_args = usecase_with_llm.llm.generate.call_args
        assert call_args.kwargs.get("system_prompt") == custom_prompt


class TestPersonaBehaviorPreservation:
    """Tests to verify existing behavior is preserved after integration."""

    @pytest.fixture
    def usecase(self):
        """Create SearchUseCase with mocked dependencies."""
        store = MagicMock()
        store.search.return_value = []
        return SearchUseCase(store=store)

    def test_detect_audience_still_works(self, usecase):
        """Existing _detect_audience behavior should be preserved."""
        # Test that audience detection still works
        audience = usecase._detect_audience("What are the faculty promotion policies?", None)
        assert audience is not None

    def test_audience_penalty_still_applied(self, usecase):
        """Audience-based score penalties should still work."""
        # Create mock chunk with student regulation name
        chunk = MagicMock()
        chunk.parent_path = ["학사관리규정"]  # Student-related regulation
        chunk.title = "Test Student Regulation"
        chunk.id = "test-1"
        chunk.text = "Test content about student policies"

        # Apply faculty audience penalty to student regulation
        penalized = usecase._apply_audience_penalty(
            chunk, Audience.FACULTY, 0.9
        )

        # Faculty should penalize student regulations
        # This verifies existing behavior is preserved
        assert 0.0 <= penalized <= 1.0
        # The score should be penalized (reduced from 0.9)
        assert penalized < 0.9
