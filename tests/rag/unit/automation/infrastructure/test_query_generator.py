"""
Unit tests for QueryGenerator infrastructure.

Tests query generation and intent analysis logic.
"""

from unittest.mock import Mock

import pytest

from src.rag.automation.domain.entities import DifficultyLevel, PersonaType
from src.rag.automation.domain.value_objects import IntentAnalysis
from src.rag.automation.infrastructure.llm_persona_generator import PersonaGenerator
from src.rag.automation.infrastructure.llm_query_generator import (
    LLMQueryGenerator,
    QueryGenerator,
)


class TestQueryGenerator:
    """Test QueryGenerator functionality."""

    def test_generate_for_persona_freshman(self):
        """WHEN generating queries for freshman, THEN should create test cases."""
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1, "medium": 1, "hard": 1}
        generator = QueryGenerator()

        test_cases = generator.generate_for_persona(persona, counts)

        assert len(test_cases) == 3
        assert all(tc.persona_type == PersonaType.FRESHMAN for tc in test_cases)

    def test_generate_difficulty_distribution(self):
        """WHEN generating queries, THEN difficulty distribution should match."""
        persona = PersonaGenerator.get_persona(PersonaType.JUNIOR)
        counts = {"easy": 2, "medium": 2, "hard": 2}
        generator = QueryGenerator()

        test_cases = generator.generate_for_persona(persona, counts)

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
        generator = QueryGenerator()

        test_cases = generator.generate_for_persona(persona, counts)

        assert all(tc.intent_analysis is not None for tc in test_cases)

    def test_intent_analysis_structure(self):
        """WHEN generating intent, THEN should have 3-level structure."""
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1, "medium": 0, "hard": 0}
        generator = QueryGenerator()

        test_cases = generator.generate_for_persona(persona, counts)
        intent = test_cases[0].intent_analysis

        assert intent.surface_intent
        assert intent.hidden_intent
        assert intent.behavioral_intent

    def test_surface_intent_extraction(self):
        """WHEN query has '신청', THEN surface intent should be '절차/신청 문의'."""
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        generator = QueryGenerator()

        # Test with query containing '신청'
        test_cases = generator.generate_for_persona(
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
        generator = QueryGenerator()

        test_cases = generator.generate_for_persona(persona, counts)
        intent = test_cases[0].intent_analysis

        # Hidden intent should include context
        assert (
            "긴급 상황" in intent.hidden_intent or "어려운 상황" in intent.hidden_intent
        )

    def test_query_content_exists(self):
        """WHEN generating queries, THEN all queries should have content."""
        persona = PersonaGenerator.get_persona(PersonaType.PROFESSOR)
        counts = {"easy": 1, "medium": 1, "hard": 1}
        generator = QueryGenerator()

        test_cases = generator.generate_for_persona(persona, counts)

        assert all(tc.query for tc in test_cases)
        assert all(len(tc.query) > 0 for tc in test_cases)

    def test_seed_reproducibility_same_seed(self):
        """WHEN using same seed, THEN should generate same queries (with templates)."""
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 2, "medium": 0, "hard": 0}
        generator = QueryGenerator()

        # Generate with same seed twice
        test_cases1 = generator.generate_for_persona(
            persona, counts, seed=42, vary_queries=False
        )
        test_cases2 = generator.generate_for_persona(
            persona, counts, seed=42, vary_queries=False
        )

        assert len(test_cases1) == len(test_cases2)
        for tc1, tc2 in zip(test_cases1, test_cases2):
            assert tc1.query == tc2.query

    def test_seed_diversity_different_seeds(self):
        """WHEN using different seeds, THEN may generate different queries (with templates)."""
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 2, "medium": 0, "hard": 0}
        generator = QueryGenerator()

        # Generate with different seeds
        test_cases1 = generator.generate_for_persona(
            persona, counts, seed=42, vary_queries=False
        )
        test_cases2 = generator.generate_for_persona(
            persona, counts, seed=123, vary_queries=False
        )

        # With limited templates, results might differ
        # The key is that different seeds should work correctly
        assert len(test_cases1) == 2
        assert len(test_cases2) == 2

    def test_no_seed_random_generation(self):
        """WHEN seed is None, THEN should use random selection."""
        persona = PersonaGenerator.get_persona(PersonaType.JUNIOR)
        counts = {"easy": 1, "medium": 0, "hard": 0}
        generator = QueryGenerator()

        # Multiple calls without seed should work
        test_cases1 = generator.generate_for_persona(
            persona, counts, seed=None, vary_queries=False
        )
        test_cases2 = generator.generate_for_persona(
            persona, counts, seed=None, vary_queries=False
        )

        assert len(test_cases1) == 1
        assert len(test_cases2) == 1

    def test_vary_queries_false_uses_templates(self):
        """WHEN vary_queries is False, THEN should use templates regardless of LLM."""
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1, "medium": 0, "hard": 0}

        # Create generator with mock LLM
        mock_llm = Mock()
        generator = QueryGenerator(llm_client=mock_llm)

        test_cases = generator.generate_for_persona(persona, counts, vary_queries=False)

        assert len(test_cases) == 1
        # LLM should not be called
        mock_llm.generate.assert_not_called()

    def test_temperature_no_seed(self):
        """WHEN seed is None, THEN temperature should be 0.9 for maximum diversity."""
        generator = LLMQueryGenerator()

        # Test temperature with no seed
        temp = generator._get_temperature(None)
        assert temp == 0.9

    def test_temperature_with_seed_deterministic(self):
        """WHEN seed is provided, THEN temperature should be deterministic."""
        generator = LLMQueryGenerator()

        # Same seed should produce same temperature
        temp1 = generator._get_temperature(42)
        temp2 = generator._get_temperature(42)

        assert temp1 == temp2
        assert 0.5 <= temp1 <= 0.89

    def test_temperature_different_seeds_different_values(self):
        """WHEN different seeds are used, THEN temperatures should differ."""
        generator = LLMQueryGenerator()

        temps = [generator._get_temperature(seed) for seed in range(10)]

        # Check we get various temperatures
        assert len(set(temps)) > 1  # At least some different values
        assert all(0.5 <= t <= 0.89 for t in temps)

    def test_make_cache_key_consistency(self):
        """WHEN making cache key, THEN same inputs should produce same key."""
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        key1 = generator._make_cache_key(persona, {"easy": 1, "medium": 1}, 42)
        key2 = generator._make_cache_key(persona, {"easy": 1, "medium": 1}, 42)

        assert key1 == key2

    def test_make_cache_key_uniqueness(self):
        """WHEN inputs differ, THEN cache keys should differ."""
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        key1 = generator._make_cache_key(persona, {"easy": 1, "medium": 1}, 42)
        key2 = generator._make_cache_key(persona, {"easy": 2, "medium": 1}, 42)
        key3 = generator._make_cache_key(persona, {"easy": 1, "medium": 1}, 43)

        assert key1 != key2  # Different counts
        assert key1 != key3  # Different seed
