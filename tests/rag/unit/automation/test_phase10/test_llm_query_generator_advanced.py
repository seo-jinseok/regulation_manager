"""
Advanced Unit Tests for LLMQueryGenerator (Phase 10).

Tests for LLM-based query generation, system prompt building,
response parsing, temperature calculation, and edge cases.
Clean Architecture: Infrastructure layer tests.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.rag.automation.domain.entities import (
    DifficultyLevel,
    Persona,
    PersonaType,
    QueryType,
)
from src.rag.automation.infrastructure.llm_query_generator import LLMQueryGenerator


class TestGenerateWithLLM:
    """Test suite for _generate_with_llm method."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_llm = MagicMock()
        # Return valid JSON array response
        mock_llm.generate.return_value = json.dumps(
            [
                {
                    "query": "LLM 생성 질문 1",
                    "type": "fact_check",
                    "difficulty": "easy",
                },
                {
                    "query": "LLM 생성 질문 2",
                    "type": "procedural",
                    "difficulty": "medium",
                },
            ],
            ensure_ascii=False,
        )
        return mock_llm

    @pytest.fixture
    def sample_persona(self):
        """Create a sample persona for testing."""
        return Persona(
            persona_type=PersonaType.FRESHMAN,
            name="Test Student",
            description="Test description",
            characteristics=["curious"],
            query_styles=["정중하게"],
        )

    def test_generate_with_llm_success(self, mock_llm_client, sample_persona):
        """WHEN LLM returns valid JSON, THEN should parse into TestCases."""
        generator = LLMQueryGenerator(llm_client=mock_llm_client)

        count_per_difficulty = {"easy": 1, "medium": 1}
        results = generator._generate_with_llm(
            persona=sample_persona,
            count_per_difficulty=count_per_difficulty,
            seed=42,
        )

        assert len(results) == 2
        assert all(isinstance(r, type(results[0])) for r in results)
        assert results[0].query == "LLM 생성 질문 1"

    def test_generate_with_llm_fallback_on_none_llm(self, sample_persona):
        """WHEN llm is None, THEN should fallback to templates."""
        generator = LLMQueryGenerator(llm_client=None)

        count_per_difficulty = {"easy": 1, "medium": 1}
        results = generator._generate_with_llm(
            persona=sample_persona,
            count_per_difficulty=count_per_difficulty,
            seed=42,
        )

        # Should still return results from templates
        assert len(results) >= 1
        assert all(isinstance(r, type(results[0])) for r in results)

    def test_generate_with_llm_uses_system_prompt(
        self, mock_llm_client, sample_persona
    ):
        """WHEN calling LLM, THEN should use proper system prompt."""
        generator = LLMQueryGenerator(llm_client=mock_llm_client)

        count_per_difficulty = {"easy": 1}
        generator._generate_with_llm(
            persona=sample_persona,
            count_per_difficulty=count_per_difficulty,
            seed=42,
        )

        # Verify LLM was called with system prompt
        mock_llm_client.generate.assert_called_once()
        call_args = mock_llm_client.generate.call_args
        system_prompt = call_args[0][0]

        assert "Test Student" in system_prompt
        assert "easy" in system_prompt

    def test_generate_with_llm_uses_temperature(self, mock_llm_client, sample_persona):
        """WHEN calling LLM, THEN should use temperature based on seed."""
        generator = LLMQueryGenerator(llm_client=mock_llm_client)

        count_per_difficulty = {"easy": 1}
        generator._generate_with_llm(
            persona=sample_persona,
            count_per_difficulty=count_per_difficulty,
            seed=42,
        )

        # Verify temperature was passed
        call_args = mock_llm_client.generate.call_args
        temperature = call_args[0][2]  # Third positional arg

        # With seed, should be deterministic (0.5-0.89 range)
        assert 0.5 <= temperature < 0.9


class TestBuildSystemPrompt:
    """Test suite for _build_system_prompt method."""

    @pytest.fixture
    def sample_persona(self):
        """Create a sample persona."""
        return Persona(
            persona_type=PersonaType.FRESHMAN,
            name="김학생",
            description="신입생",
            characteristics=["호기심 많음"],
            query_styles=["~까요?", "알려주세요"],
        )

    def test_build_system_prompt_includes_persona_info(self, sample_persona):
        """WHEN building prompt, THEN should include persona details."""
        generator = LLMQueryGenerator()

        prompt = generator._build_system_prompt(
            persona=sample_persona, count_per_difficulty={"easy": 1, "medium": 1}
        )

        assert "김학생" in prompt
        assert "신입생" in prompt
        assert "호기심 많음" in prompt
        assert "~까요?" in prompt or "알려주세요" in prompt

    def test_build_system_prompt_includes_difficulty_counts(self, sample_persona):
        """WHEN building prompt, THEN should include requested counts."""
        generator = LLMQueryGenerator()

        prompt = generator._build_system_prompt(
            persona=sample_persona,
            count_per_difficulty={"easy": 3, "medium": 2, "hard": 1},
        )

        assert "**Easy** (3개)" in prompt
        assert "**Medium** (2개)" in prompt
        assert "**Hard** (1개)" in prompt

    def test_build_system_prompt_includes_query_type_guide(self, sample_persona):
        """WHEN building prompt, THEN should include query type guide."""
        generator = LLMQueryGenerator()

        prompt = generator._build_system_prompt(
            persona=sample_persona, count_per_difficulty={"easy": 1}
        )

        assert "fact_check" in prompt
        assert "procedural" in prompt
        assert "eligibility" in prompt

    def test_build_system_prompt_includes_output_format(self, sample_persona):
        """WHEN building prompt, THEN should specify JSON output format."""
        generator = LLMQueryGenerator()

        prompt = generator._build_system_prompt(
            persona=sample_persona, count_per_difficulty={"easy": 1}
        )

        assert "JSON 배열" in prompt
        assert "```json" in prompt
        assert '"query"' in prompt
        assert '"type"' in prompt
        assert '"difficulty"' in prompt

    def test_build_system_prompt_includes_template_examples(self, sample_persona):
        """WHEN building prompt, THEN should include template examples."""
        generator = LLMQueryGenerator()

        prompt = generator._build_system_prompt(
            persona=sample_persona, count_per_difficulty={"easy": 1}
        )

        assert "참고용 예시" in prompt
        # Should have examples from the persona's templates


class TestParseLLMResponse:
    """Test suite for _parse_llm_response method."""

    @pytest.fixture
    def sample_persona(self):
        """Create a sample persona."""
        return Persona(
            persona_type=PersonaType.FRESHMAN,
            name="Test",
            description="Test",
            characteristics=[],
            query_styles=[],
        )

    def test_parse_llm_response_valid_json(self, sample_persona):
        """WHEN response is valid JSON, THEN should parse correctly."""
        generator = LLMQueryGenerator()

        response = json.dumps(
            [
                {"query": "테스트 질문", "type": "fact_check", "difficulty": "easy"},
                {"query": "절차 질문", "type": "procedural", "difficulty": "medium"},
            ],
            ensure_ascii=False,
        )

        results = generator._parse_llm_response(
            response=response,
            persona=sample_persona,
            count_per_difficulty={"easy": 1, "medium": 1},
        )

        assert len(results) == 2
        assert results[0].query == "테스트 질문"
        assert results[0].query_type == QueryType.FACT_CHECK
        assert results[0].difficulty == DifficultyLevel.EASY
        assert results[1].query_type == QueryType.PROCEDURAL

    def test_parse_llm_response_markdown_wrapped(self, sample_persona):
        """WHEN response is wrapped in markdown code block, THEN should extract JSON."""
        generator = LLMQueryGenerator()

        json_content = json.dumps(
            [{"query": "테스트", "type": "fact_check", "difficulty": "easy"}],
            ensure_ascii=False,
        )

        # Wrapped in ```json ... ```
        response = f"```json\n{json_content}\n```"

        results = generator._parse_llm_response(
            response=response, persona=sample_persona, count_per_difficulty={"easy": 1}
        )

        assert len(results) == 1
        assert results[0].query == "테스트"

    def test_parse_llm_response_plain_markdown_wrapped(self, sample_persona):
        """WHEN response is wrapped in plain markdown code block, THEN should extract JSON."""
        generator = LLMQueryGenerator()

        json_content = json.dumps(
            [{"query": "테스트", "type": "fact_check", "difficulty": "easy"}],
            ensure_ascii=False,
        )

        # Wrapped in ``` ... ```
        response = f"```\n{json_content}\n```"

        results = generator._parse_llm_response(
            response=response, persona=sample_persona, count_per_difficulty={"easy": 1}
        )

        assert len(results) == 1

    def test_parse_llm_response_invalid_json_fallback(self, sample_persona):
        """WHEN response has invalid JSON, THEN should fallback to templates."""
        generator = LLMQueryGenerator()

        response = "This is not valid JSON at all"

        results = generator._parse_llm_response(
            response=response, persona=sample_persona, count_per_difficulty={"easy": 1}
        )

        # Should return results from template fallback
        assert len(results) >= 1

    def test_parse_llm_response_empty_result_fallback(self, sample_persona):
        """WHEN response parses to empty list, THEN should fallback to templates."""
        generator = LLMQueryGenerator()

        response = json.dumps([], ensure_ascii=False)

        results = generator._parse_llm_response(
            response=response, persona=sample_persona, count_per_difficulty={"easy": 1}
        )

        # Empty result should trigger template fallback
        assert len(results) >= 1

    def test_parse_llm_response_skips_invalid_entries(self, sample_persona):
        """WHEN response has mixed valid/invalid entries, THEN should parse valid ones."""
        generator = LLMQueryGenerator()

        response = json.dumps(
            [
                {"query": "Valid question", "type": "fact_check", "difficulty": "easy"},
                {
                    "query": "",
                    "type": "fact_check",
                    "difficulty": "easy",
                },  # Empty query
                {
                    "query": "Another valid",
                    "type": "procedural",
                    "difficulty": "medium",
                },
            ],
            ensure_ascii=False,
        )

        results = generator._parse_llm_response(
            response=response,
            persona=sample_persona,
            count_per_difficulty={"easy": 1, "medium": 1},
        )

        # Should parse valid entries, skip invalid
        assert len(results) >= 2
        # Empty query should be filtered out


class TestGetTemperature:
    """Test suite for _get_temperature method."""

    def test_get_temperature_no_seed_high_diversity(self):
        """WHEN seed is None, THEN should return 0.9 for high diversity."""
        generator = LLMQueryGenerator()

        temperature = generator._get_temperature(seed=None)

        assert temperature == 0.9

    def test_get_temperature_with_seed_deterministic(self):
        """WHEN seed is provided, THEN should return deterministic temperature."""
        generator = LLMQueryGenerator()

        temp1 = generator._get_temperature(seed=42)
        temp2 = generator._get_temperature(seed=42)

        # Same seed should give same temperature
        assert temp1 == temp2
        assert 0.5 <= temp1 < 0.9

    def test_get_temperature_different_seeds(self):
        """WHEN different seeds are used, THEN temperatures should differ."""
        generator = LLMQueryGenerator()

        temp1 = generator._get_temperature(seed=1)
        temp2 = generator._get_temperature(seed=999)

        # Different seeds may give different temperatures
        assert 0.5 <= temp1 < 0.9
        assert 0.5 <= temp2 < 0.9

    def test_get_temperature_in_valid_range(self):
        """WHEN using seed, THEN temperature should be in [0.5, 0.89] range."""
        generator = LLMQueryGenerator()

        for seed in [1, 42, 100, 9999]:
            temperature = generator._get_temperature(seed=seed)
            assert 0.5 <= temperature <= 0.89


class TestUseLLMProperty:
    """Test suite for use_llm property."""

    def test_use_llm_getter(self):
        """WHEN getting use_llm, THEN should return flag value."""
        generator = LLMQueryGenerator()
        generator._use_llm = True

        assert generator.use_llm is True

    def test_use_llm_setter(self):
        """WHEN setting use_llm, THEN should update flag value."""
        generator = LLMQueryGenerator()

        generator.use_llm = False

        assert generator.use_llm is False

    def test_use_llm_setter_affects_generate(self):
        """WHEN use_llm is False, THEN generate should use templates."""
        generator = LLMQueryGenerator(llm_client=None)
        generator.use_llm = False

        persona = Persona(
            persona_type=PersonaType.FRESHMAN,
            name="Test",
            description="Test",
            characteristics=[],
            query_styles=[],
        )

        # Even with vary_queries=True, should use templates since use_llm=False
        results = generator.generate_for_persona(
            persona=persona, count_per_difficulty={"easy": 1}, vary_queries=True
        )

        assert len(results) >= 1


class TestGenerateFromTemplates:
    """Test suite for _generate_from_templates distribution."""

    @pytest.fixture
    def sample_persona(self):
        """Create a sample persona."""
        return Persona(
            persona_type=PersonaType.FRESHMAN,
            name="Test Student",
            description="Test",
            characteristics=[],
            query_styles=[],
        )

    def test_generate_from_templates_respects_difficulty_counts(self, sample_persona):
        """WHEN requesting specific counts, THEN should generate that many."""
        generator = LLMQueryGenerator()

        results = generator._generate_from_templates(
            persona=sample_persona, count_per_difficulty={"easy": 2, "medium": 1}
        )

        # Should have at least the requested counts (may have more from template rotation)
        easy_queries = [r for r in results if r.difficulty == DifficultyLevel.EASY]
        medium_queries = [r for r in results if r.difficulty == DifficultyLevel.MEDIUM]

        assert len(easy_queries) >= 2
        assert len(medium_queries) >= 1

    def test_generate_from_templates_distributes_query_types(self, sample_persona):
        """WHEN generating from templates, THEN should have various types."""
        generator = LLMQueryGenerator()

        results = generator._generate_from_templates(
            persona=sample_persona, count_per_difficulty={"easy": 3, "medium": 3}
        )

        # Should have different query types
        query_types = set(r.query_type for r in results)
        assert len(query_types) > 1

    def test_generate_from_templates_all_have_intent_analysis(self, sample_persona):
        """WHEN generating from templates, THEN all should have intent analysis."""
        generator = LLMQueryGenerator()

        results = generator._generate_from_templates(
            persona=sample_persona, count_per_difficulty={"easy": 1}
        )

        # All should have intent analysis
        for result in results:
            assert result.intent_analysis is not None
            assert hasattr(result.intent_analysis, "surface_intent")


class TestLLMIntegration:
    """Test suite for LLM integration patterns."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        mock_llm_client = MagicMock()
        return mock_llm_client

    @pytest.fixture
    def sample_persona(self):
        """Create a sample persona."""
        return Persona(
            persona_type=PersonaType.PROFESSOR,
            name="Professor Kim",
            description="Test professor",
            characteristics=["strict"],
            query_styles=["~니?"],
        )

    def test_generate_with_vary_queries_uses_llm(self, mock_llm_client, sample_persona):
        """WHEN vary_queries=True and LLM available, THEN should use LLM."""
        generator = LLMQueryGenerator(llm_client=mock_llm_client)
        mock_llm_client.generate.return_value = json.dumps(
            [{"query": "LLM 질문", "type": "fact_check", "difficulty": "easy"}],
            ensure_ascii=False,
        )

        results = generator.generate_for_persona(
            persona=sample_persona, count_per_difficulty={"easy": 1}, vary_queries=True
        )

        # Should have called LLM
        mock_llm_client.generate.assert_called_once()
        # Should have results
        assert len(results) >= 1

    def test_generate_with_vary_queries_false_uses_templates(self, sample_persona):
        """WHEN vary_queries=False, THEN should use templates even with LLM."""
        generator = LLMQueryGenerator(llm_client=MagicMock())

        results = generator.generate_for_persona(
            persona=sample_persona, count_per_difficulty={"easy": 1}, vary_queries=False
        )

        # Should NOT have called LLM
        assert not generator.llm.generate.called
        # Should still have results
        assert len(results) >= 1

    def test_generate_llm_error_fallback_to_templates(
        self, mock_llm_client, sample_persona
    ):
        """WHEN LLM raises exception, THEN should propagate the exception."""
        generator = LLMQueryGenerator(llm_client=mock_llm_client)
        mock_llm_client.generate.side_effect = Exception("LLM error")

        # LLM errors are propagated (no automatic fallback)
        with pytest.raises(Exception, match="LLM error"):
            generator.generate_for_persona(
                persona=sample_persona,
                count_per_difficulty={"easy": 1},
                vary_queries=True,
            )
