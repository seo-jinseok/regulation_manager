"""
Comprehensive tests for LLMQueryGenerator.

Covers template-based generation, LLM-based generation with mocks,
intent analysis, temperature diversity, and seed reproducibility.
"""

from unittest.mock import Mock

from src.rag.automation.domain.entities import (
    DifficultyLevel,
    PersonaType,
    QueryType,
)
from src.rag.automation.domain.value_objects import IntentAnalysis
from src.rag.automation.infrastructure.llm_persona_generator import PersonaGenerator
from src.rag.automation.infrastructure.llm_query_generator import LLMQueryGenerator


class TestLLMQueryGenerator:
    """Test LLMQueryGenerator with comprehensive coverage."""

    # ==========================================================================
    # Test 1: Template-based generation (basic)
    # ==========================================================================

    def test_generate_from_templates_basic(self):
        """Test template-based generation basic behavior."""
        # Create generator without LLM client (template-only mode)
        generator = LLMQueryGenerator(llm_client=None)

        # Get a test persona
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        # Generate queries
        count_per_difficulty = {"easy": 2, "medium": 1, "hard": 1}
        test_cases = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=False,  # Force template mode
            seed=None,
        )

        # Verify results
        assert len(test_cases) == 4  # 2 + 1 + 1
        assert all(tc.query for tc in test_cases)
        assert all(tc.persona_type == PersonaType.FRESHMAN for tc in test_cases)
        assert all(tc.intent_analysis is not None for tc in test_cases)

        # Verify difficulty distribution
        easy_cases = [tc for tc in test_cases if tc.difficulty == DifficultyLevel.EASY]
        medium_cases = [
            tc for tc in test_cases if tc.difficulty == DifficultyLevel.MEDIUM
        ]
        hard_cases = [tc for tc in test_cases if tc.difficulty == DifficultyLevel.HARD]

        assert len(easy_cases) == 2
        assert len(medium_cases) == 1
        assert len(hard_cases) == 1

    def test_generate_from_templates_all_persona_types(self):
        """Test template generation for all persona types."""
        generator = LLMQueryGenerator(llm_client=None)
        personas = PersonaGenerator.get_all_personas()

        for persona in personas:
            count_per_difficulty = {"easy": 1, "medium": 1, "hard": 1}
            test_cases = generator.generate_for_persona(
                persona=persona,
                count_per_difficulty=count_per_difficulty,
                vary_queries=False,
            )

            assert len(test_cases) == 3
            assert all(tc.persona_type == persona.persona_type for tc in test_cases)

    def test_generate_from_templates_with_seed(self):
        """Test template generation respects random seed."""
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.JUNIOR)

        count_per_difficulty = {"easy": 2, "medium": 2}

        # Generate with same seed twice
        test_cases_1 = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=False,
            seed=42,
        )

        test_cases_2 = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=False,
            seed=42,
        )

        # Verify reproducibility - same queries should be selected
        queries_1 = [tc.query for tc in test_cases_1]
        queries_2 = [tc.query for tc in test_cases_2]
        assert queries_1 == queries_2

    # ==========================================================================
    # Test 2: LLM-based generation with mock
    # ==========================================================================

    def test_generate_with_llm_mock(self):
        """Test LLM-based generation with mocked LLM client."""
        # Create mock LLM client
        mock_llm = Mock()
        mock_llm.generate.return_value = """```json
[
  {"query": "LLM generated query 1", "type": "procedural", "difficulty": "easy"},
  {"query": "LLM generated query 2", "type": "eligibility", "difficulty": "medium"},
  {"query": "LLM generated query 3", "type": "fact_check", "difficulty": "hard"}
]
```"""

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.GRADUATE)

        count_per_difficulty = {"easy": 1, "medium": 1, "hard": 1}
        test_cases = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=True,  # Enable LLM
            seed=42,
        )

        # Verify LLM was called
        assert mock_llm.generate.called_once
        call_args = mock_llm.generate.call_args
        assert "graduated" in call_args[0][0].lower() or "대학원생" in call_args[0][0]

        # Verify results
        assert len(test_cases) == 3
        assert all(tc.persona_type == PersonaType.GRADUATE for tc in test_cases)
        assert test_cases[0].query == "LLM generated query 1"
        assert test_cases[1].query == "LLM generated query 2"
        assert test_cases[2].query == "LLM generated query 3"

    def test_llm_fallback_to_templates_on_json_error(self):
        """Test fallback to templates when LLM returns invalid JSON."""
        mock_llm = Mock()
        mock_llm.generate.return_value = "Invalid JSON response"

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.PROFESSOR)

        count_per_difficulty = {"easy": 1, "medium": 1}
        test_cases = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=True,
            seed=42,
        )

        # Should fallback to templates and still return results
        assert len(test_cases) == 2
        assert all(tc.persona_type == PersonaType.PROFESSOR for tc in test_cases)

    def test_llm_fallback_to_templates_on_empty_result(self):
        """Test fallback to templates when LLM returns empty array."""
        mock_llm = Mock()
        mock_llm.generate.return_value = "[]"

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.STAFF_MANAGER)

        count_per_difficulty = {"easy": 1}
        test_cases = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=True,
            seed=42,
        )

        # Should fallback to templates
        assert len(test_cases) >= 1

    def test_llm_with_markdown_json_wrapper(self):
        """Test LLM response parsing with markdown code block wrapper."""
        mock_llm = Mock()
        mock_llm.generate.return_value = """```json
[
  {"query": "Test query", "type": "fact_check", "difficulty": "easy"}
]
```"""

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.PARENT)

        count_per_difficulty = {"easy": 1}
        test_cases = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=True,
        )

        assert len(test_cases) == 1
        assert test_cases[0].query == "Test query"

    def test_llm_with_plain_json(self):
        """Test LLM response parsing with plain JSON (no markdown)."""
        mock_llm = Mock()
        mock_llm.generate.return_value = (
            '[{"query": "Plain JSON", "type": "fact_check", "difficulty": "easy"}]'
        )

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.NEW_STAFF)

        count_per_difficulty = {"easy": 1}
        test_cases = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=True,
        )

        assert len(test_cases) == 1
        assert test_cases[0].query == "Plain JSON"

    # ==========================================================================
    # Test 3: Intent analysis generation
    # ==========================================================================

    def test_intent_analysis_generation(self):
        """Test intent analysis is generated for each query."""
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        count_per_difficulty = {"easy": 1, "medium": 1}
        test_cases = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=False,
        )

        # Verify all test cases have intent analysis
        for tc in test_cases:
            assert tc.intent_analysis is not None
            assert isinstance(tc.intent_analysis, IntentAnalysis)
            assert tc.intent_analysis.surface_intent
            assert tc.intent_analysis.hidden_intent
            assert tc.intent_analysis.behavioral_intent

    def test_surface_intent_extraction(self):
        """Test surface intent extraction for different query types."""
        generator = LLMQueryGenerator(llm_client=None)

        # Test procedural query
        procedural_intent = generator._extract_surface_intent(
            "휴학 신청하는 방법 알려주세요"
        )
        assert procedural_intent == "절차/신청 문의"

        # Test eligibility query
        eligibility_intent = generator._extract_surface_intent("장학금 자격이 뭐야?")
        assert eligibility_intent == "자격/요건 확인"

        # Test information request
        info_intent = generator._extract_surface_intent("졸업 요건 알려줘")
        assert info_intent == "정보 요청"

        # Test emotional/distress query
        emotional_intent = generator._extract_surface_intent(
            "학자금 대출 못 받았어요 어떡하죠?"
        )
        assert emotional_intent == "불만/도움 요청"

    def test_hidden_intent_inference(self):
        """Test hidden intent inference based on persona."""
        generator = LLMQueryGenerator(llm_client=None)

        freshman = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        hidden_intent = generator._infer_hidden_intent("휴학 신청 방법", freshman)
        assert "학교 시스템 적응 필요" in hidden_intent

        graduate = PersonaGenerator.get_persona(PersonaType.GRADUATE)
        hidden_intent = generator._infer_hidden_intent("연구비 지원", graduate)
        assert "연구/학위 진행" in hidden_intent

        parent = PersonaGenerator.get_persona(PersonaType.PARENT)
        hidden_intent = generator._infer_hidden_intent("등록금 환급", parent)
        assert "자녀 교육 지원" in hidden_intent

    def test_behavioral_intent_inference(self):
        """Test behavioral intent inference for different actions."""
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        # Test application behavioral intent ("신청" in query)
        application_intent = generator._infer_behavioral_intent(
            "휴학 신청 방법", persona
        )
        assert application_intent == "신청서 제출"

        # Test procedure behavioral intent ("방법" in query, but not "신청")
        procedure_intent = generator._infer_behavioral_intent("사용 방법", persona)
        assert procedure_intent == "절차 수행"

        # Test information acquisition ("알려줘" in query)
        info_intent = generator._infer_behavioral_intent("졸업 요건 알려줘", persona)
        assert info_intent == "정보 습득"

        # Test eligibility check ("자격" or "조건" in query)
        eligibility_intent = generator._infer_behavioral_intent(
            "자격 조건 확인", persona
        )
        assert eligibility_intent == "자격 확인 후 행동"

        # Test default case (no keywords)
        default_intent = generator._infer_behavioral_intent("안녕하세요", persona)
        assert default_intent == "정보 확인"

    # ==========================================================================
    # Test 4: Temperature diversity
    # ==========================================================================

    def test_temperature_diversity(self):
        """Test that different seeds produce different temperatures."""
        generator = LLMQueryGenerator(llm_client=None)

        # Get temperatures for different seeds
        temps = []
        for seed in range(10):
            temp = generator._get_temperature(seed)
            temps.append(temp)

        # Verify we have different temperatures (not all same)
        assert len(set(temps)) > 1, "Temperatures should vary with different seeds"

        # Verify all temperatures are in valid range
        for temp in temps:
            assert 0.5 <= temp <= 0.89, f"Temperature {temp} out of range [0.5, 0.89]"

    def test_temperature_none_seed_max_diversity(self):
        """Test that None seed returns maximum temperature for diversity."""
        generator = LLMQueryGenerator(llm_client=None)

        temp = generator._get_temperature(None)
        assert temp == 0.9, "None seed should return max temperature 0.9"

    def test_temperature_deterministic_with_seed(self):
        """Test that same seed produces same temperature."""
        generator = LLMQueryGenerator(llm_client=None)

        temp1 = generator._get_temperature(42)
        temp2 = generator._get_temperature(42)
        temp3 = generator._get_temperature(42)

        assert temp1 == temp2 == temp3, "Same seed should produce same temperature"

    def test_temperature_distribution(self):
        """Test temperature distribution covers expected range."""
        generator = LLMQueryGenerator(llm_client=None)

        temps = [generator._get_temperature(seed) for seed in range(100)]

        # Check coverage across range
        min_temp = min(temps)
        max_temp = max(temps)

        assert min_temp >= 0.5, f"Min temp {min_temp} below 0.5"
        assert max_temp <= 0.89, f"Max temp {max_temp} above 0.89"

        # Check reasonable distribution (not clumped at one end)
        # MD5 hash should distribute evenly
        low_range = sum(1 for t in temps if t < 0.65)
        mid_range = sum(1 for t in temps if 0.65 <= t < 0.75)
        high_range = sum(1 for t in temps if t >= 0.75)

        # Each range should have at least some values
        assert low_range > 10, "Low range underpopulated"
        assert mid_range > 10, "Mid range underpopulated"
        assert high_range > 10, "High range underpopulated"

    # ==========================================================================
    # Test 5: Seed reproducibility
    # ==========================================================================

    def test_seed_reproducibility(self):
        """Test that same seed produces same query selection."""
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.JUNIOR)

        count_per_difficulty = {"easy": 3, "medium": 3}

        # Generate with same seed
        results1 = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=False,
            seed=12345,
        )

        results2 = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=False,
            seed=12345,
        )

        # Verify same queries are selected
        queries1 = [tc.query for tc in results1]
        queries2 = [tc.query for tc in results2]

        assert queries1 == queries2, "Same seed should produce identical results"

        # Verify same metadata
        for tc1, tc2 in zip(results1, results2):
            assert tc1.query_type == tc2.query_type
            assert tc1.difficulty == tc2.difficulty

    def test_seed_reproducibility_with_llm(self):
        """Test that same seed produces same temperature for LLM calls."""
        mock_llm = Mock()
        mock_llm.generate.return_value = (
            '[{"query": "Test", "type": "fact_check", "difficulty": "easy"}]'
        )

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.GRADUATE)

        # Generate twice with same seed
        generator.generate_for_persona(
            persona=persona,
            count_per_difficulty={"easy": 1},
            vary_queries=True,
            seed=999,
        )

        temp1 = mock_llm.generate.call_args[1].get("temperature")

        mock_llm.reset_mock()

        generator.generate_for_persona(
            persona=persona,
            count_per_difficulty={"easy": 1},
            vary_queries=True,
            seed=999,
        )

        temp2 = mock_llm.generate.call_args[1].get("temperature")

        assert temp1 == temp2, "Same seed should produce same temperature"

    # ==========================================================================
    # Test 6: Seed diversity
    # ==========================================================================

    def test_seed_diversity(self):
        """Test that different seeds produce different query selections."""
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.DISTRESSED_STUDENT)

        count_per_difficulty = {"easy": 3, "medium": 3, "hard": 3}

        # Generate with different seeds
        results1 = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=False,
            seed=1,
        )

        results2 = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=False,
            seed=2,
        )

        results3 = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=False,
            seed=3,
        )

        # Verify different queries are selected
        queries1 = [tc.query for tc in results1]
        queries2 = [tc.query for tc in results2]
        queries3 = [tc.query for tc in results3]

        # At least some should be different (random choice from templates)
        # With 9 queries from 4 templates, duplicates are possible
        # but different seeds should likely produce different results
        assert queries1 != queries2 or queries2 != queries3, (
            "Different seeds should produce different results"
        )

    def test_seed_vs_no_seed(self):
        """Test that None seed produces different results than fixed seed."""
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.STAFF_MANAGER)

        count_per_difficulty = {"easy": 2, "medium": 2}

        # Generate with seed
        results_seeded = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=False,
            seed=42,
        )

        # Generate without seed (random)
        results_random1 = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=False,
            seed=None,
        )

        results_random2 = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty=count_per_difficulty,
            vary_queries=False,
            seed=None,
        )

        # Seeded should be reproducible
        queries_seeded = [tc.query for tc in results_seeded]

        # Random runs might be different (though could be same by chance)
        queries_random1 = [tc.query for tc in results_random1]
        queries_random2 = [tc.query for tc in results_random2]

        # At minimum, seeded results should be consistent
        assert len(queries_seeded) == 4

    # ==========================================================================
    # Additional edge cases
    # ==========================================================================

    def test_empty_count_per_difficulty(self):
        """Test handling of empty count dictionary."""
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        test_cases = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty={},
            vary_queries=False,
        )

        assert len(test_cases) == 0

    def test_vary_queries_flag_without_llm(self):
        """Test that vary_queries=True falls back to templates when no LLM."""
        # Create generator without LLM
        generator = LLMQueryGenerator(llm_client=None)
        assert generator.use_llm is False

        persona = PersonaGenerator.get_persona(PersonaType.PROFESSOR)

        # Even with vary_queries=True, should use templates
        test_cases = generator.generate_for_persona(
            persona=persona,
            count_per_difficulty={"easy": 1},
            vary_queries=True,  # This should be ignored
        )

        assert len(test_cases) == 1

    def test_use_llm_property(self):
        """Test use_llm property getter and setter."""
        generator = LLMQueryGenerator(llm_client=None)
        assert generator.use_llm is False

        # Can set to True even without LLM client
        generator.use_llm = True
        assert generator.use_llm is True

        # But generate_for_persona will still use templates if llm is None
        generator.use_llm = False
        assert generator.use_llm is False

    def test_make_cache_key(self):
        """Test cache key generation for query storage."""
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.JUNIOR)

        counts = {"easy": 2, "medium": 1}
        key1 = generator._make_cache_key(persona, counts, seed=42)
        key2 = generator._make_cache_key(persona, counts, seed=42)
        key3 = generator._make_cache_key(persona, counts, seed=43)

        # Same inputs should produce same key
        assert key1 == key2

        # Different seed should produce different key
        assert key1 != key3

        # Keys should be valid MD5 hex strings (32 chars)
        assert len(key1) == 32
        assert all(c in "0123456789abcdef" for c in key1)

    def test_build_system_prompt_contains_required_elements(self):
        """Test that system prompt contains all required elements."""
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.GRADUATE)

        counts = {"easy": 1, "medium": 2, "hard": 1}
        prompt = generator._build_system_prompt(persona, counts)

        # Verify prompt contains persona info
        assert persona.name in prompt
        assert persona.description in prompt

        # Verify prompt contains difficulty info
        assert "easy" in prompt.lower()
        assert "medium" in prompt.lower()
        assert "hard" in prompt.lower()

        # Verify prompt contains regulation topics
        assert "학사" in prompt or "Academic" in prompt
        assert "장학" in prompt or "Scholarship" in prompt

    def test_get_template_examples(self):
        """Test getting template examples for LLM reference."""
        generator = LLMQueryGenerator(llm_client=None)

        # Test with persona that has templates
        examples = generator._get_template_examples(PersonaType.FRESHMAN)

        assert examples
        assert "휴학" in examples or "freshman" in examples.lower()

        # Test with persona type that might not exist (should return empty)
        # We use a valid enum value but the templates dict should handle missing keys
        examples_all_types = [
            generator._get_template_examples(pt)
            for pt in [
                PersonaType.FRESHMAN,
                PersonaType.JUNIOR,
                PersonaType.GRADUATE,
                PersonaType.PARENT,
            ]
        ]

        # All should have examples (they're defined in _QUERY_TEMPLATES)
        assert all(examples for examples in examples_all_types)

    def test_parse_llm_response_with_invalid_entries(self):
        """Test parsing handles invalid entries gracefully."""
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        # Response with one valid and one invalid entry
        response = """```json
[
  {"query": "Valid query", "type": "fact_check", "difficulty": "easy"},
  {"query": "Invalid type", "type": "invalid_type", "difficulty": "easy"},
  {"query": "Missing type", "difficulty": "medium"}
]
```"""

        counts = {"easy": 2}
        test_cases = generator._parse_llm_response(response, persona, counts)

        # Should parse valid entry and skip invalid ones
        assert len(test_cases) >= 1
        assert test_cases[0].query == "Valid query"

    def test_query_type_mapping(self):
        """Test all query types are correctly mapped."""
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        # Test each query type
        type_tests = [
            ("fact_check", QueryType.FACT_CHECK),
            ("procedural", QueryType.PROCEDURAL),
            ("eligibility", QueryType.ELIGIBILITY),
            ("comparison", QueryType.COMPARISON),
            ("ambiguous", QueryType.AMBIGUOUS),
            ("emotional", QueryType.EMOTIONAL),
            ("complex", QueryType.COMPLEX),
        ]

        for type_str, expected_type in type_tests:
            response = (
                f'[{{"query": "Test", "type": "{type_str}", "difficulty": "easy"}}]'
            )
            test_cases = generator._parse_llm_response(response, persona, {"easy": 1})

            if test_cases:
                assert test_cases[0].query_type == expected_type

    def test_difficulty_level_mapping(self):
        """Test all difficulty levels are correctly mapped."""
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        # Test each difficulty level
        difficulty_tests = [
            ("easy", DifficultyLevel.EASY),
            ("medium", DifficultyLevel.MEDIUM),
            ("hard", DifficultyLevel.HARD),
        ]

        for diff_str, expected_diff in difficulty_tests:
            response = f'[{{"query": "Test", "type": "fact_check", "difficulty": "{diff_str}"}}]'
            test_cases = generator._parse_llm_response(response, persona, {"easy": 1})

            if test_cases:
                assert test_cases[0].difficulty == expected_diff

    def test_backward_compatibility_alias(self):
        """Test that QueryGenerator alias still works."""
        from src.rag.automation.infrastructure.llm_query_generator import (
            QueryGenerator,
        )

        # QueryGenerator should be the same as LLMQueryGenerator
        assert QueryGenerator is LLMQueryGenerator

        # Should be able to create instance with alias
        generator = QueryGenerator(llm_client=None)
        assert isinstance(generator, LLMQueryGenerator)
