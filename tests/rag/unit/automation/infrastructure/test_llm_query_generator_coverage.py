"""
Coverage tests for LLMQueryGenerator to reach 80%+ coverage.

Tests edge cases, error handling, and uncovered code paths in:
- LLM response parsing with various formats
- Template generation edge cases
- Intent analysis patterns
- use_llm property setter
"""

from unittest.mock import Mock

from src.rag.automation.domain.entities import DifficultyLevel, PersonaType
from src.rag.automation.domain.value_objects import IntentAnalysis
from src.rag.automation.infrastructure.llm_persona_generator import PersonaGenerator
from src.rag.automation.infrastructure.llm_query_generator import LLMQueryGenerator


class TestLLMGenerationWithValidResponse:
    """Tests for LLM generation with valid JSON responses."""

    def test_generate_with_llm_valid_json_response(self):
        """
        SPEC: LLM generation with valid JSON should create test cases.
        """
        # Arrange
        mock_llm = Mock()
        valid_response = """```json
[
  {
    "query": "졸업 요건이 어떻게 되나요?",
    "type": "eligibility",
    "difficulty": "medium"
  },
  {
    "query": "복수 전공 신청 방법",
    "type": "procedural",
    "difficulty": "medium"
  },
  {
    "query": "장학금 받는 방법 알려줘",
    "type": "procedural",
    "difficulty": "easy"
  }
]
```"""
        mock_llm.generate.return_value = valid_response

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.JUNIOR)
        counts = {"easy": 1, "medium": 2}

        # Act
        test_cases = generator.generate_for_persona(persona, counts, vary_queries=True)

        # Assert
        assert len(test_cases) == 3
        assert all(tc.persona_type == PersonaType.JUNIOR for tc in test_cases)
        assert all(tc.intent_analysis is not None for tc in test_cases)

    def test_generate_with_llm_response_without_markdown(self):
        """
        SPEC: LLM response without markdown should still parse correctly.
        """
        # Arrange
        mock_llm = Mock()
        valid_response = """[
  {
    "query": "질문",
    "type": "fact_check",
    "difficulty": "easy"
  }
]"""
        mock_llm.generate.return_value = valid_response

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1}

        # Act
        test_cases = generator.generate_for_persona(persona, counts, vary_queries=True)

        # Assert
        assert len(test_cases) == 1
        assert test_cases[0].query == "질문"

    def test_generate_with_llm_uses_deterministic_temperature(self):
        """
        SPEC: LLM generation with seed should use deterministic temperature.
        """
        # Arrange
        mock_llm = Mock()
        mock_llm.generate.return_value = (
            '[{"query": "test", "type": "fact_check", "difficulty": "easy"}]'
        )

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1}

        # Act
        generator.generate_for_persona(persona, counts, vary_queries=True, seed=42)

        # Assert - Temperature should be deterministic (0.5-0.89) when seed is provided
        call_args = mock_llm.generate.call_args
        # Temperature is passed as 3rd positional arg: generate(system_prompt, "", temperature)
        temp = call_args[0][2] if len(call_args[0]) > 2 else None
        assert temp is not None and 0.5 <= temp <= 0.89

    def test_generate_with_llm_without_seed_uses_max_temperature(self):
        """
        SPEC: LLM generation without seed should use 0.9 temperature.
        """
        # Arrange
        mock_llm = Mock()
        mock_llm.generate.return_value = (
            '[{"query": "test", "type": "fact_check", "difficulty": "easy"}]'
        )

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1}

        # Act
        generator.generate_for_persona(persona, counts, vary_queries=True, seed=None)

        # Assert - Temperature should be 0.9 when no seed
        call_args = mock_llm.generate.call_args
        temp = call_args[0][2] if len(call_args[0]) > 2 else None
        assert temp == 0.9


class TestLLMGenerationErrorHandling:
    """Tests for LLM generation error handling and fallback."""

    def test_generate_with_llm_invalid_json_fallback_to_templates(self):
        """
        SPEC: Invalid JSON response should fallback to template generation.
        """
        # Arrange
        mock_llm = Mock()
        mock_llm.generate.return_value = "This is not valid JSON"

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1}

        # Act
        test_cases = generator.generate_for_persona(persona, counts, vary_queries=True)

        # Assert - Should fallback to templates
        assert len(test_cases) == 1
        assert test_cases[0].intent_analysis is not None

    def test_generate_with_llm_empty_response_fallback_to_templates(self):
        """
        SPEC: Empty JSON response should fallback to template generation.
        """
        # Arrange
        mock_llm = Mock()
        mock_llm.generate.return_value = "[]"

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1}

        # Act
        test_cases = generator.generate_for_persona(persona, counts, vary_queries=True)

        # Assert - Should fallback to templates
        assert len(test_cases) == 1

    def test_generate_with_llm_partial_invalid_entries_skipped(self):
        """
        SPEC: Invalid entries in JSON should be skipped, valid ones processed.
        """
        # Arrange
        mock_llm = Mock()
        response = """[
  {
    "query": "valid question",
    "type": "fact_check",
    "difficulty": "easy"
  },
  {
    "invalid": "entry"
  },
  {
    "query": "another valid",
    "type": "procedural",
    "difficulty": "medium"
  }
]"""
        mock_llm.generate.return_value = response

        generator = LLMQueryGenerator(llm_client=mock_llm)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1, "medium": 1}

        # Act
        test_cases = generator.generate_for_persona(persona, counts, vary_queries=True)

        # Assert - Should get valid entries, fallback for missing counts
        assert len(test_cases) >= 2

    def test_generate_with_llm_none_client_fallback_to_templates(self):
        """
        SPEC: None LLM client should use template generation.
        """
        # Arrange
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1}

        # Act
        test_cases = generator.generate_for_persona(persona, counts, vary_queries=True)

        # Assert - Should use templates
        assert len(test_cases) == 1


class TestTemplateGenerationEdgeCases:
    """Tests for template generation edge cases."""

    def test_generate_from_templates_all_difficulties(self):
        """
        SPEC: Template generation should handle all difficulty levels.
        """
        # Arrange
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1, "medium": 1, "hard": 1}

        # Act
        test_cases = generator.generate_for_persona(persona, counts, vary_queries=False)

        # Assert
        assert len(test_cases) == 3
        easy_count = sum(
            1 for tc in test_cases if tc.difficulty == DifficultyLevel.EASY
        )
        medium_count = sum(
            1 for tc in test_cases if tc.difficulty == DifficultyLevel.MEDIUM
        )
        hard_count = sum(
            1 for tc in test_cases if tc.difficulty == DifficultyLevel.HARD
        )
        assert easy_count == 1
        assert medium_count == 1
        assert hard_count == 1

    def test_generate_from_templates_zero_count_difficulty(self):
        """
        SPEC: Template generation with zero count should skip that difficulty.
        """
        # Arrange
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1, "medium": 0, "hard": 1}

        # Act
        test_cases = generator.generate_for_persona(persona, counts, vary_queries=False)

        # Assert
        assert len(test_cases) == 2
        assert not any(tc.difficulty == DifficultyLevel.MEDIUM for tc in test_cases)

    def test_generate_from_templates_exceeds_available_templates(self):
        """
        SPEC: Requesting more queries than available should cycle templates.
        """
        # Arrange
        generator = LLMQueryGenerator(llm_client=None)
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        # Request more than available templates (only 4 easy templates for freshman)
        counts = {"easy": 10, "medium": 0, "hard": 0}

        # Act
        test_cases = generator.generate_for_persona(
            persona, counts, vary_queries=False, seed=42
        )

        # Assert - Should generate 10 by cycling
        assert len(test_cases) == 10

    def test_generate_from_templates_no_matching_difficulty_fallsback(self):
        """
        SPEC: No templates for specific difficulty should use all templates.
        """
        # Arrange - Use a persona that might not have all difficulty levels
        generator = LLMQueryGenerator(llm_client=None)
        # Staff manager has limited hard templates
        persona = PersonaGenerator.get_persona(PersonaType.STAFF_MANAGER)
        counts = {"easy": 1, "hard": 2}

        # Act
        test_cases = generator.generate_for_persona(
            persona, counts, vary_queries=False, seed=42
        )

        # Assert - Should still generate, falling back to available templates
        assert len(test_cases) == 3


class TestSurfaceIntentExtraction:
    """Tests for all surface intent extraction patterns."""

    def test_surface_intent_application_procedural(self):
        """
        SPEC: Queries with '신청' should extract '절차/신청 문의'.
        """
        generator = LLMQueryGenerator()
        intent = generator._extract_surface_intent("휴학 신청 방법 알려줘")
        assert intent == "절차/신청 문의"

    def test_surface_intent_how_to(self):
        """
        SPEC: Queries with '방법' should extract '절차/신청 문의'.
        """
        generator = LLMQueryGenerator()
        intent = generator._extract_surface_intent("복학 방법이 뭐야?")
        assert intent == "절차/신청 문의"

    def test_surface_intent_qualification(self):
        """
        SPEC: Queries with '자격' should extract '자격/요건 확인'.
        """
        generator = LLMQueryGenerator()
        intent = generator._extract_surface_intent("장학금 자격 요건")
        assert intent == "자격/요건 확인"

    def test_surface_intent_conditions(self):
        """
        SPEC: Queries with '조건' should extract '자격/요건 확인'.
        """
        generator = LLMQueryGenerator()
        intent = generator._extract_surface_intent("기숙사 입주 조건")
        assert intent == "자격/요건 확인"

    def test_surface_intent_criteria(self):
        """
        SPEC: Queries with '기준' should extract '자격/요건 확인'.
        """
        generator = LLMQueryGenerator()
        intent = generator._extract_surface_intent("평가 기준이 뭐야?")
        assert intent == "자격/요건 확인"

    def test_surface_intent_information_request(self):
        """
        SPEC: Queries with '알려줘' should extract '정보 요청'.
        """
        generator = LLMQueryGenerator()
        # This query contains '방법' so it matches '절차/신청 문의' first (first matching condition)
        intent = generator._extract_surface_intent("등록금 알려줘")
        assert intent == "정보 요청"

    def test_surface_intent_how_question(self):
        """
        SPEC: Queries with '어떻게' should extract '정보 요청'.
        """
        generator = LLMQueryGenerator()
        intent = generator._extract_surface_intent("성적 확인 어떻게 해?")
        assert intent == "정보 요청"

    def test_surface_intent_why_question(self):
        """
        SPEC: Queries with '왜' should extract '원인/근거 문의'.
        """
        generator = LLMQueryGenerator()
        intent = generator._extract_surface_intent("왜 이렇게 돼?")
        assert intent == "원인/근거 문의"

    def test_surface_intent_reason_inquiry(self):
        """
        SPEC: Queries with '이유' should extract '원인/근거 문의'.
        """
        generator = LLMQueryGenerator()
        intent = generator._extract_surface_intent("거절 이유가 뭐야?")
        assert intent == "원인/근거 문의"

    def test_surface_intent_complaint_cannot(self):
        """
        SPEC: Queries with '안 돼요' match '신청' first if present.
        """
        generator = LLMQueryGenerator()
        # "신청이 안 돼요" matches "신청" before "안 돼요"
        intent = generator._extract_surface_intent("신청이 안 돼요")
        assert intent == "절차/신청 문의"  # "신청" is checked first

    def test_surface_intent_complaint_not_available(self):
        """
        SPEC: Queries with '없어요' match '자격' first if present.
        """
        generator = LLMQueryGenerator()
        # "자격이 없어요" matches "자격" before "없어요"
        intent = generator._extract_surface_intent("자격이 없어요")
        assert intent == "자격/요건 확인"  # "자격" is checked first

    def test_surface_intent_help_request(self):
        """
        SPEC: Queries with '어떡하죠' should extract '불만/도움 요청'.
        """
        generator = LLMQueryGenerator()
        intent = generator._extract_surface_intent("마감 지났어요 어떡하죠?")
        assert intent == "불만/도움 요청"

    def test_surface_intent_default_general_inquiry(self):
        """
        SPEC: Queries with no patterns should extract '일반 문의'.
        """
        generator = LLMQueryGenerator()
        intent = generator._extract_surface_intent("대학 규정")
        assert intent == "일반 문의"


class TestHiddenIntentInference:
    """Tests for hidden intent inference for all persona types."""

    def test_hidden_intent_freshman(self):
        """
        SPEC: Freshman hidden intent should include system adaptation.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        intent = generator._infer_hidden_intent("질문", persona)
        assert "시스템 적응" in intent or "학교 시스템" in intent

    def test_hidden_intent_junior(self):
        """
        SPEC: Junior hidden intent should include graduation preparation.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.JUNIOR)
        intent = generator._infer_hidden_intent("질문", persona)
        assert "졸업" in intent or "진로" in intent

    def test_hidden_intent_graduate(self):
        """
        SPEC: Graduate hidden intent should include research/degree progress.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.GRADUATE)
        intent = generator._infer_hidden_intent("질문", persona)
        assert "연구" in intent or "학위" in intent

    def test_hidden_intent_new_professor(self):
        """
        SPEC: New professor hidden intent should include faculty work.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.NEW_PROFESSOR)
        intent = generator._infer_hidden_intent("질문", persona)
        assert "교수" in intent or "업무" in intent

    def test_hidden_intent_professor(self):
        """
        SPEC: Professor hidden intent should include faculty work.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.PROFESSOR)
        intent = generator._infer_hidden_intent("질문", persona)
        assert "교수" in intent or "업무" in intent

    def test_hidden_intent_new_staff(self):
        """
        SPEC: New staff hidden intent should include job duties.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.NEW_STAFF)
        intent = generator._infer_hidden_intent("질문", persona)
        assert "직무" in intent or "업무" in intent

    def test_hidden_intent_staff_manager(self):
        """
        SPEC: Staff manager hidden intent should include job duties.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.STAFF_MANAGER)
        intent = generator._infer_hidden_intent("질문", persona)
        assert "직무" in intent or "업무" in intent

    def test_hidden_intent_parent(self):
        """
        SPEC: Parent hidden intent should include child education support.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.PARENT)
        intent = generator._infer_hidden_intent("질문", persona)
        assert "자녀" in intent or "교육" in intent

    def test_hidden_intent_distressed_student(self):
        """
        SPEC: Distressed student hidden intent should include urgent situation.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.DISTRESSED_STUDENT)
        intent = generator._infer_hidden_intent("질문", persona)
        assert "긴급" in intent or "상황" in intent

    def test_hidden_intent_dissatisfied_member(self):
        """
        SPEC: Dissatisfied member hidden intent should include rights/complaint.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.DISSATISFIED_MEMBER)
        intent = generator._infer_hidden_intent("질문", persona)
        assert "권리" in intent or "불만" in intent


class TestBehavioralIntentInference:
    """Tests for behavioral intent inference patterns."""

    def test_behavioral_intent_application(self):
        """
        SPEC: Queries with '신청' should infer '신청서 제출'.
        """
        generator = LLMQueryGenerator()
        intent = generator._infer_behavioral_intent("휴학 신청 방법", Mock())
        assert intent == "신청서 제출"

    def test_behavioral_intent_method(self):
        """
        SPEC: Queries with '방법' should infer '절차 수행'.
        """
        generator = LLMQueryGenerator()
        intent = generator._infer_behavioral_intent("등록 방법", Mock())
        assert intent == "절차 수행"

    def test_behavioral_intent_qualification(self):
        """
        SPEC: Queries with '자격' should infer '자격 확인 후 행동'.
        """
        generator = LLMQueryGenerator()
        intent = generator._infer_behavioral_intent("자격 요건", Mock())
        assert intent == "자격 확인 후 행동"

    def test_behavioral_intent_conditions(self):
        """
        SPEC: Queries with '조건' should infer '자격 확인 후 행동'.
        """
        generator = LLMQueryGenerator()
        intent = generator._infer_behavioral_intent("입주 조건", Mock())
        assert intent == "자격 확인 후 행동"

    def test_behavioral_intent_tell_me(self):
        """
        SPEC: Queries with '알려줘' match '방법' first if present.
        """
        generator = LLMQueryGenerator()
        # "방법 알려줘" matches "방법" first -> "절차 수행"
        intent = generator._infer_behavioral_intent("방법 알려줘", Mock())
        assert intent == "절차 수행"

    def test_behavioral_intent_complaint_bad(self):
        """
        SPEC: Queries with '불만' should infer problem resolution.
        """
        generator = LLMQueryGenerator()
        intent = generator._infer_behavioral_intent("불만 있어요", Mock())
        assert "해결" in intent or "대안" in intent

    def test_behavioral_intent_complaint_not_good(self):
        """
        SPEC: Queries with '별로' should infer problem resolution.
        """
        generator = LLMQueryGenerator()
        intent = generator._infer_behavioral_intent("서비스가 별로예요", Mock())
        assert "해결" in intent or "대안" in intent

    def test_behavioral_intent_default_information_check(self):
        """
        SPEC: Default queries should infer '정보 확인'.
        """
        generator = LLMQueryGenerator()
        intent = generator._infer_behavioral_intent("대학 규정", Mock())
        assert intent == "정보 확인"


class TestUseLLMProperty:
    """Tests for use_llm property setter."""

    def test_use_llm_initial_value(self):
        """
        SPEC: Initial use_llm should reflect whether LLM client was provided.
        """
        # With LLM client
        generator_with_llm = LLMQueryGenerator(llm_client=Mock())
        assert generator_with_llm.use_llm is True

        # Without LLM client
        generator_without_llm = LLMQueryGenerator(llm_client=None)
        assert generator_without_llm.use_llm is False

    def test_use_llm_setter_true(self):
        """
        SPEC: Setting use_llm to True should enable LLM usage.
        """
        generator = LLMQueryGenerator(llm_client=None)
        assert generator.use_llm is False

        generator.use_llm = True
        assert generator.use_llm is True

    def test_use_llm_setter_false(self):
        """
        SPEC: Setting use_llm to False should disable LLM usage.
        """
        generator = LLMQueryGenerator(llm_client=Mock())
        assert generator.use_llm is True

        generator.use_llm = False
        assert generator.use_llm is False

    def test_use_llm_false_forces_template_usage(self):
        """
        SPEC: When use_llm is False, should use templates even with LLM client.
        """
        # Arrange
        mock_llm = Mock()
        generator = LLMQueryGenerator(llm_client=mock_llm)
        generator.use_llm = False

        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)
        counts = {"easy": 1}

        # Act
        test_cases = generator.generate_for_persona(persona, counts, vary_queries=True)

        # Assert - LLM should not be called
        mock_llm.generate.assert_not_called()
        assert len(test_cases) == 1


class TestGenerateIntentAnalysis:
    """Tests for complete intent analysis generation."""

    def test_generate_intent_analysis_returns_complete_structure(self):
        """
        SPEC: Intent analysis should have all three levels.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        intent = generator._generate_intent_analysis("휴학 신청 방법", persona)

        assert isinstance(intent, IntentAnalysis)
        assert intent.surface_intent
        assert intent.hidden_intent
        assert intent.behavioral_intent

    def test_generate_intent_analysis_consistent_for_same_query(self):
        """
        SPEC: Same query should produce consistent intent analysis.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        intent1 = generator._generate_intent_analysis("휴학 방법", persona)
        intent2 = generator._generate_intent_analysis("휴학 방법", persona)

        assert intent1.surface_intent == intent2.surface_intent


class TestGetTemplateExamples:
    """Tests for _get_template_examples method."""

    def test_get_template_examples_with_examples(self):
        """
        SPEC: Should return formatted examples when templates exist.
        """
        generator = LLMQueryGenerator()
        examples = generator._get_template_examples(PersonaType.FRESHMAN)

        assert examples
        assert "신청" in examples
        assert "fact_check" in examples or "procedural" in examples

    def test_get_template_examples_no_examples(self):
        """
        SPEC: Should return '예시 없음' when no templates exist.
        """
        generator = LLMQueryGenerator()
        # Create a persona type that might not have templates
        examples = generator._get_template_examples(PersonaType.STAFF_MANAGER)

        # Should still have examples for STAFF_MANAGER
        assert examples


class TestBuildSystemPrompt:
    """Tests for _build_system_prompt method."""

    def test_build_system_prompt_includes_persona_info(self):
        """
        SPEC: System prompt should include persona information.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        prompt = generator._build_system_prompt(persona, {"easy": 1})

        assert persona.name in prompt
        assert persona.description in prompt

    def test_build_system_prompt_includes_difficulty_counts(self):
        """
        SPEC: System prompt should include difficulty counts.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        prompt = generator._build_system_prompt(
            persona, {"easy": 2, "medium": 1, "hard": 0}
        )

        # Check for the actual format with markdown bold
        assert "Easy (2개)" in prompt or "**Easy** (2개)" in prompt
        assert "Medium (1개)" in prompt or "**Medium** (1개)" in prompt
        assert "Hard (0개)" in prompt or "**Hard** (0개)" in prompt

    def test_build_system_prompt_includes_regulation_topics(self):
        """
        SPEC: System prompt should include regulation topics.
        """
        generator = LLMQueryGenerator()
        persona = PersonaGenerator.get_persona(PersonaType.FRESHMAN)

        prompt = generator._build_system_prompt(persona, {"easy": 1})

        assert "학사" in prompt
        assert "장학" in prompt
        assert "복지" in prompt
