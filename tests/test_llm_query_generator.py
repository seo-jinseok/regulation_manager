"""
Tests for LLMQueryGenerator.

Test both template-based and LLM-based query generation.
"""

import pytest
from src.rag.automation.domain.entities import Persona, PersonaType
from src.rag.automation.infrastructure.llm_query_generator import LLMQueryGenerator, QueryGenerator
from src.rag.automation.infrastructure.mock_llm_client import MockLLMClientForQueryGen


@pytest.fixture
def freshman_persona():
    """Create a freshman persona for testing."""
    return Persona(
        persona_type=PersonaType.FRESHMAN,
        name="신입생",
        description="학교 시스템에 익숙하지 않음",
        characteristics=["학교 생활 미숙", "궁금한 것 많음"],
        query_styles=["간단한 질문", "구체적인 도움 요청"],
        context_hints=["입학 첫 해", "규정을 잘 모름"],
    )


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client for testing."""
    return MockLLMClientForQueryGen(use_template_responses=True)


class TestLLMQueryGenerator:
    """Test suite for LLMQueryGenerator."""

    def test_init_without_llm_client(self):
        """Test initialization without LLM client (template-only mode)."""
        generator = LLMQueryGenerator()
        assert generator.llm is None
        assert generator.use_llm is False

    def test_init_with_llm_client(self, mock_llm_client):
        """Test initialization with LLM client."""
        generator = LLMQueryGenerator(mock_llm_client)
        assert generator.llm is mock_llm_client
        assert generator.use_llm is True

    def test_template_based_generation(self, freshman_persona):
        """Test template-based query generation (backward compatibility)."""
        generator = LLMQueryGenerator()  # No LLM client
        counts = {"easy": 2, "medium": 1, "hard": 0}

        test_cases = generator.generate_for_persona(
            persona=freshman_persona,
            count_per_difficulty=counts,
            vary_queries=False,  # Use templates
        )

        assert len(test_cases) == 3
        assert all(tc.persona_type == PersonaType.FRESHMAN for tc in test_cases)
        assert sum(1 for tc in test_cases if tc.difficulty.value == "easy") == 2
        assert sum(1 for tc in test_cases if tc.difficulty.value == "medium") == 1

    def test_llm_based_generation(self, freshman_persona, mock_llm_client):
        """Test LLM-based query generation."""
        generator = LLMQueryGenerator(mock_llm_client)
        counts = {"easy": 1, "medium": 1, "hard": 1}

        test_cases = generator.generate_for_persona(
            persona=freshman_persona,
            count_per_difficulty=counts,
            vary_queries=True,  # Use LLM
        )

        # Verify LLM was called
        assert mock_llm_client.call_count > 0
        # Verify test cases were generated
        assert len(test_cases) >= 1

    def test_seed_reproducibility(self, freshman_persona, mock_llm_client):
        """Test that same seed produces same results."""
        generator = LLMQueryGenerator(mock_llm_client)
        counts = {"easy": 1, "medium": 1, "hard": 0}

        # Generate with seed
        test_cases_1 = generator.generate_for_persona(
            persona=freshman_persona,
            count_per_difficulty=counts,
            vary_queries=False,  # Use templates for deterministic test
            seed=42,
        )

        # Reset and generate again with same seed
        mock_llm_client.reset_call_count()
        test_cases_2 = generator.generate_for_persona(
            persona=freshman_persona,
            count_per_difficulty=counts,
            vary_queries=False,
            seed=42,
        )

        # Should produce same queries (template-based)
        queries_1 = [tc.query for tc in test_cases_1]
        queries_2 = [tc.query for tc in test_cases_2]
        assert queries_1 == queries_2

    def test_fallback_to_templates_on_json_error(self, freshman_persona):
        """Test fallback to templates when LLM returns invalid JSON."""
        # Create a mock client that returns invalid JSON
        class BrokenLLMClient:
            def generate(self, system_prompt, user_message, temperature=0.0):
                return "This is not valid JSON"
            def get_embedding(self, text):
                return [0.0] * 384

        generator = LLMQueryGenerator(BrokenLLMClient())
        counts = {"easy": 1, "medium": 0, "hard": 0}

        test_cases = generator.generate_for_persona(
            persona=freshman_persona,
            count_per_difficulty=counts,
            vary_queries=True,
        )

        # Should fall back to templates
        assert len(test_cases) >= 1

    def test_intent_analysis_generation(self, freshman_persona):
        """Test that intent analysis is generated for each query."""
        generator = LLMQueryGenerator()
        counts = {"easy": 1, "medium": 0, "hard": 0}

        test_cases = generator.generate_for_persona(
            persona=freshman_persona,
            count_per_difficulty=counts,
        )

        assert len(test_cases) == 1
        assert test_cases[0].intent_analysis is not None
        assert test_cases[0].intent_analysis.surface_intent != ""
        assert test_cases[0].intent_analysis.hidden_intent != ""
        assert test_cases[0].intent_analysis.behavioral_intent != ""

    def test_use_llm_property(self, mock_llm_client):
        """Test use_llm property setter."""
        generator = LLMQueryGenerator(mock_llm_client)

        # Initially True (LLM client provided)
        assert generator.use_llm is True

        # Set to False
        generator.use_llm = False
        assert generator.use_llm is False

        # Set back to True
        generator.use_llm = True
        assert generator.use_llm is True


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_query_generator_alias(self):
        """Test that QueryGenerator alias works."""
        # Should work without arguments (backward compatibility)
        generator = QueryGenerator()
        assert generator is not None
        assert isinstance(generator, LLMQueryGenerator)

    def test_classmethod_fallback_removed(self):
        """Test that old classmethod pattern is not available."""
        # The old pattern QueryGenerator.generate_for_persona(cls, ...) should not work
        # because the method now requires an instance (self)
        # Verify it's not a classmethod by checking it needs an instance
        import inspect
        
        # Should be a regular method that requires self, not a classmethod
        sig = inspect.signature(QueryGenerator.generate_for_persona)
        params = list(sig.parameters.keys())
        
        # First parameter should be 'self', not 'cls'
        assert params[0] == 'self', "Method should use 'self' not 'cls'"        
        # Should require at least self and persona
        assert len(params) >= 2, "Method should have at least self and persona parameters"


class TestCacheKeyGeneration:
    """Test cache key generation for query caching."""

    def test_cache_key_different_inputs(self, freshman_persona):
        """Test that different inputs produce different cache keys."""
        generator = LLMQueryGenerator()

        key1 = generator._make_cache_key(freshman_persona, {"easy": 1}, 42)
        key2 = generator._make_cache_key(freshman_persona, {"easy": 2}, 42)
        key3 = generator._make_cache_key(freshman_persona, {"easy": 1}, 43)

        assert key1 != key2  # Different counts
        assert key1 != key3  # Different seeds

    def test_cache_key_same_inputs(self, freshman_persona):
        """Test that same inputs produce same cache keys."""
        generator = LLMQueryGenerator()

        key1 = generator._make_cache_key(freshman_persona, {"easy": 1}, 42)
        key2 = generator._make_cache_key(freshman_persona, {"easy": 1}, 42)

        assert key1 == key2  # Same inputs


class TestLLMQueryGeneration:
    """Test LLM-based query generation with realistic scenarios."""

    def test_professor_persona_queries(self, mock_llm_client):
        """Test query generation for professor persona."""
        professor_persona = Persona(
            persona_type=PersonaType.PROFESSOR,
            name="정교수",
            description="세부 규정 확인 필요",
            characteristics=["연구 중심", "행정 절차 관심"],
            query_styles=["공식적인 어조", "구체적인 질문"],
            context_hints=["교원 평가", "연구비"],
        )

        generator = LLMQueryGenerator(mock_llm_client)
        counts = {"easy": 1, "medium": 2, "hard": 1}

        test_cases = generator.generate_for_persona(
            persona=professor_persona,
            count_per_difficulty=counts,
            vary_queries=True,
        )

        # Should generate queries
        assert len(test_cases) >= 1

    def test_vary_queries_false_uses_templates(self, freshman_persona, mock_llm_client):
        """Test that vary_queries=False forces template usage even with LLM."""
        generator = LLMQueryGenerator(mock_llm_client)
        counts = {"easy": 1, "medium": 0, "hard": 0}

        # Reset call count
        mock_llm_client.reset_call_count()

        test_cases = generator.generate_for_persona(
            persona=freshman_persona,
            count_per_difficulty=counts,
            vary_queries=False,  # Force templates
        )

        # LLM should not have been called
        assert mock_llm_client.call_count == 0
        # Should still have test cases from templates
        assert len(test_cases) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
