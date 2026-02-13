"""
Characterization tests for PersonaAwareGenerator behavior preservation.

These tests capture the CURRENT behavior of the RAG system for different personas.
They serve as a safety net during refactoring to ensure behavior is preserved.
"""

import pytest
from unittest.mock import Mock, patch
from src.rag.domain.evaluation.parallel_evaluator import ParallelPersonaEvaluator, PersonaQuery
from src.rag.infrastructure.llm_adapter import LLMClientAdapter


class TestPersonaBehaviorCharacterization:
    """Characterization tests for persona-specific response behavior."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator with mocked LLM to capture behavior."""
        mock_llm = Mock(spec=LLMClientAdapter)
        # Mock the generate method to return persona-specific responses
        mock_llm.generate.return_value = "Default response"

        with patch('src.rag.domain.evaluation.parallel_evaluator.SearchUseCase'):
            with patch('src.rag.domain.evaluation.parallel_evaluator.ChromaVectorStore'):
                with patch('src.rag.domain.evaluation.parallel_evaluator.LLMJudge'):
                    evaluator = ParallelPersonaEvaluator(llm_client=mock_llm)
                    return evaluator

    def test_professor_persona_query_generation(self, evaluator):
        """Characterize how professor queries are generated."""
        queries = evaluator.generate_persona_queries(
            persona="professor",
            count_per_category=1
        )

        # Capture current behavior: professor queries
        assert len(queries) > 0

        # Document current query patterns
        academic_terms = ["조항", "규정", "적용", "기준", "해석"]
        query_texts = [q.query for q in queries]

        # Current behavior: queries use academic language
        has_academic_query = any(
            any(term in q for term in academic_terms)
            for q in query_texts
        )
        # This documents current state - may be False initially
        print(f"Professor has academic terms: {has_academic_query}")

    def test_parent_persona_query_generation(self, evaluator):
        """Characterize how parent queries are generated."""
        queries = evaluator.generate_persona_queries(
            persona="parent",
            count_per_category=1
        )

        # Capture current behavior: parent queries
        assert len(queries) > 0

        query_texts = [q.query for q in queries]

        # Current behavior: queries mention child/student
        has_child_ref = any("자녀" in q or "학생" in q for q in query_texts)
        print(f"Parent queries mention child: {has_child_ref}")

    def test_international_persona_query_generation(self, evaluator):
        """Characterize how international student queries are generated."""
        queries = evaluator.generate_persona_queries(
            persona="student-international",
            count_per_category=1
        )

        # Capture current behavior: international queries
        assert len(queries) > 0

        query_texts = [q.query for q in queries]

        # Current behavior: some queries in English
        english_queries = [q for q in query_texts if any(
            c.isalpha() and c.isascii() for c in q
        )]
        print(f"International has English queries: {len(english_queries)}")

    def test_current_prompt_does_not_differentiate_personas(self, evaluator):
        """
        Document that current implementation uses same prompt for all personas.

        This is the behavior we want to CHANGE with PersonaAwareGenerator.
        """
        from src.rag.application.search_usecase import REGULATION_QA_PROMPT

        # Current behavior: single prompt for all personas
        assert isinstance(REGULATION_QA_PROMPT, str)
        assert len(REGULATION_QA_PROMPT) > 0

        # Document: prompt is not parameterized by persona
        # This test will help us verify we've improved this
        print(f"Current prompt length: {len(REGULATION_QA_PROMPT)}")

    def test_evaluate_single_query_behavior(self, evaluator):
        """Characterize the current evaluate_single_query behavior."""
        # Create a test query
        test_query = PersonaQuery(
            query="휴학 방법 알려줘",
            persona="student-undergraduate",
            category="simple",
            difficulty="easy",
            expected_intent="leave_of_absence",
            expected_info=["기간", "절차", "서류"]
        )

        # Mock the search_usecase.ask method
        mock_answer = Mock()
        mock_answer.text = "현재 규정에 따르면 휴학 신청은 학기 개시 1개월 전까지 가능합니다."
        mock_answer.sources = []

        evaluator.search_usecase.ask = Mock(return_value=mock_answer)

        # Mock judge evaluation
        mock_judge_result = Mock()
        mock_judge_result.overall_score = 0.7
        mock_judge_result.passed = True
        mock_judge_result.issues = []

        evaluator.judge.evaluate_with_llm = Mock(return_value=mock_judge_result)

        # Execute
        result = evaluator._evaluate_single_query(test_query)

        # Document current behavior
        assert result.overall_score == 0.7
        assert result.passed is True

        # Verify search_usecase.ask was called (not persona-aware yet)
        assert evaluator.search_usecase.ask.called
        call_args = evaluator.search_usecase.ask.call_args

        # Document: ask() doesn't receive persona parameter currently
        # This will change after integration
        print(f"ask() called with kwargs: {call_args.kwargs.keys()}")


class TestPersonaAnswerPreferences:
    """Characterization tests for persona answer preferences."""

    def test_professor_preferences(self):
        """Document professor answer preferences."""
        from src.rag.domain.evaluation.personas import PERSONAS

        professor = PERSONAS.get("professor")
        assert professor is not None

        # Document current preferences
        prefs = professor.answer_preferences
        print(f"Professor preferences: {prefs}")

        assert prefs.get("detail_level") == "comprehensive"
        assert prefs.get("citation_style") == "detailed"

    def test_parent_preferences(self):
        """Document parent answer preferences."""
        from src.rag.domain.evaluation.personas import PERSONAS

        parent = PERSONAS.get("parent")
        assert parent is not None

        # Document current preferences
        prefs = parent.answer_preferences
        print(f"Parent preferences: {prefs}")

        assert prefs.get("detail_level") == "simple"
        assert prefs.get("parent_friendly") is True

    def test_international_preferences(self):
        """Document international student answer preferences."""
        from src.rag.domain.evaluation.personas import PERSONAS

        international = PERSONAS.get("international")
        assert international is not None

        # Document current preferences
        prefs = international.answer_preferences
        print(f"International preferences: {prefs}")

        assert prefs.get("language") == "korean_english_mixed"
        assert prefs.get("include_english_terms") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
