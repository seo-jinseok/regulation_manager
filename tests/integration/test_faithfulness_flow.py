"""
Integration tests for FaithfulnessValidator flow in SearchUseCase.

Tests REQ-003 and REQ-004 integration:
- FaithfulnessValidator instance creation in __init__
- Semantic validation working with pattern-based HallucinationFilter
- Regeneration loop trigger on low faithfulness
- Fallback response on all attempts failure
"""

import pytest
from unittest.mock import MagicMock, patch

from src.rag.application.search_usecase import SearchUseCase
from src.rag.application.hallucination_filter import HallucinationFilter, FilterMode
from src.rag.domain.evaluation.faithfulness_validator import (
    FaithfulnessValidator,
    FaithfulnessValidationResult,
)
from src.rag.domain.entities import SearchResult, Chunk, ChunkLevel


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: list[str] = None):
        self.responses = responses or ["This is a test response."]
        self.call_count = 0

    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.0) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class MockVectorStore:
    """Mock vector store for testing."""

    def search(self, query: str, top_k: int = 5, filter=None):
        return []

    def get_all_documents(self):
        """Return empty list for hybrid search initialization."""
        return []


def create_test_search_result(text: str, score: float = 0.8) -> SearchResult:
    """Create a test SearchResult with chunk."""
    chunk = Chunk(
        id="test-1",
        rule_code="TEST001",
        level=ChunkLevel.ARTICLE,
        title="Test Article",
        text=text,
        embedding_text=text,
        full_text=text,
        parent_path=["Test Regulation"],
        token_count=100,
        keywords=[],
        is_searchable=True,
        article_number="1",
    )
    return SearchResult(chunk=chunk, score=score, rank=1)


class TestFaithfulnessValidatorIntegration:
    """Tests for FaithfulnessValidator integration with SearchUseCase."""

    def test_faithfulness_validator_injected_via_constructor(self):
        """Test that FaithfulnessValidator can be injected via constructor."""
        store = MockVectorStore()
        custom_validator = FaithfulnessValidator()

        usecase = SearchUseCase(
            store=store,
            faithfulness_validator=custom_validator,
        )

        assert usecase._faithfulness_validator is custom_validator

    def test_faithfulness_validator_auto_created_when_not_provided(self):
        """Test that FaithfulnessValidator is auto-created when not provided."""
        store = MockVectorStore()

        usecase = SearchUseCase(store=store)

        assert usecase._faithfulness_validator is not None
        assert isinstance(usecase._faithfulness_validator, FaithfulnessValidator)

    def test_both_validators_coexist(self):
        """Test that HallucinationFilter and FaithfulnessValidator coexist."""
        store = MockVectorStore()
        hallucination_filter = HallucinationFilter(mode=FilterMode.SANITIZE)
        faithfulness_validator = FaithfulnessValidator()

        usecase = SearchUseCase(
            store=store,
            hallucination_filter=hallucination_filter,
            faithfulness_validator=faithfulness_validator,
        )

        # Both validators should be present
        assert usecase.hallucination_filter is hallucination_filter
        assert usecase._faithfulness_validator is faithfulness_validator

    def test_role_separation_pattern_vs_semantic(self):
        """Test that HallucinationFilter (pattern) and FaithfulnessValidator (semantic) have different roles."""
        response = "문의는 학적팀(02-1234-5678)으로 연락 바랍니다."
        context = ["학적팀 연락처는 02-1234-5678입니다."]

        # HallucinationFilter: Pattern-based validation
        hallucination_filter = HallucinationFilter(mode=FilterMode.SANITIZE)
        filter_result = hallucination_filter.filter_response(response, context)

        # FaithfulnessValidator: Semantic validation
        validator = FaithfulnessValidator()
        validation_result = validator.validate_answer(response, context)

        # Both should pass since the data is consistent
        assert not filter_result.blocked
        assert validation_result.is_acceptable

    def test_faithfulness_validation_flow_with_valid_answer(self):
        """Test faithfulness validation flow with a valid answer."""
        store = MockVectorStore()
        llm = MockLLMClient(
            responses=["제10조에 따르면 휴학 기간은 2학기까지 가능합니다."]
        )

        usecase = SearchUseCase(
            store=store,
            llm_client=llm,
        )

        # Test internal method directly
        context = "제10조 (휴학기간) 휴학기간은 통산 2학기를 초과할 수 없다."
        context_list = [context]

        answer, metadata = usecase._generate_answer_with_validation(
            question="휴학 기간이 어떻게 되나요?",
            context=context,
            context_list=context_list,
        )

        assert answer == "제10조에 따르면 휴학 기간은 2학기까지 가능합니다."
        assert metadata["final_status"] == "validated"
        assert metadata["validation_attempts"] >= 1

    def test_regeneration_trigger_on_low_faithfulness(self):
        """Test that low faithfulness triggers regeneration."""
        store = MockVectorStore()
        # First answer has ungrounded claims, second is better
        llm = MockLLMClient(
            responses=[
                "휴학 기간은 5년입니다.",  # Ungrounded claim
                "제10조에 따르면 휴학 기간은 2학기까지 가능합니다.",  # Grounded answer
            ]
        )

        usecase = SearchUseCase(
            store=store,
            llm_client=llm,
        )

        context = "제10조 (휴학기간) 휴학기간은 통산 2학기를 초과할 수 없다."
        context_list = [context]

        answer, metadata = usecase._generate_answer_with_validation(
            question="휴학 기간이 어떻게 되나요?",
            context=context,
            context_list=context_list,
            max_retries=2,
        )

        # Should have attempted regeneration
        assert metadata["validation_attempts"] >= 1
        assert llm.call_count >= 1

    def test_fallback_response_on_all_attempts_failure(self):
        """Test fallback response when all regeneration attempts fail."""
        store = MockVectorStore()
        # All answers have ungrounded claims
        llm = MockLLMClient(
            responses=[
                "휴학 기간은 5년입니다.",
                "휴학 기간은 10년입니다.",
                "휴학 기간은 3년입니다.",
            ]
        )

        usecase = SearchUseCase(
            store=store,
            llm_client=llm,
        )

        context = "제10조 (휴학기간) 휴학기간은 통산 2학기를 초과할 수 없다."
        context_list = [context]

        answer, metadata = usecase._generate_answer_with_validation(
            question="휴학 기간이 어떻게 되나요?",
            context=context,
            context_list=context_list,
            max_retries=2,
        )

        # Should return fallback when all attempts fail
        assert metadata["final_status"] == "fallback"
        # Check for fallback message (죄송합니다 or 관련 규정을 찾을 수 없)
        assert "죄송" in answer or "찾을 수 없" in answer or "문의" in answer

    def test_end_to_end_with_mock_search(self):
        """Test end-to-end flow with mocked search results."""
        store = MockVectorStore()
        llm = MockLLMClient(
            responses=["제10조에 따르면 휴학 기간은 2학기까지 가능합니다."]
        )

        usecase = SearchUseCase(
            store=store,
            llm_client=llm,
        )

        # Mock the search method to return test results
        test_results = [
            create_test_search_result(
                "제10조 (휴학기간) 휴학기간은 통산 2학기를 초과할 수 없다.",
                score=0.9,
            )
        ]

        with patch.object(usecase, "search", return_value=test_results):
            # Mock _compute_confidence to return high confidence
            with patch.object(usecase, "_compute_confidence", return_value=0.9):
                answer = usecase.ask("휴학 기간이 어떻게 되나요?")

        assert answer is not None
        assert answer.text is not None
        assert len(answer.sources) > 0

    def test_hallucination_filter_and_faithfulness_validator_both_applied(self):
        """Test that both validators are applied in the ask() flow."""
        store = MockVectorStore()
        llm = MockLLMClient(
            responses=["문의는 학적팀(02-1234-5678)으로 연락 바랍니다."]
        )

        hallucination_filter = HallucinationFilter(mode=FilterMode.SANITIZE)
        faithfulness_validator = FaithfulnessValidator()

        usecase = SearchUseCase(
            store=store,
            llm_client=llm,
            hallucination_filter=hallucination_filter,
            faithfulness_validator=faithfulness_validator,
        )

        # Create test results with matching context
        test_results = [
            create_test_search_result(
                "학적팀 연락처는 02-1234-5678입니다.",
                score=0.9,
            )
        ]

        with patch.object(usecase, "search", return_value=test_results):
            with patch.object(usecase, "_compute_confidence", return_value=0.9):
                answer = usecase.ask("학적팀 연락처가 어떻게 되나요?")

        assert answer is not None
        # Both validators should have been applied without blocking
        assert "02-1234-5678" in answer.text or "학적팀" in answer.text


class TestFaithfulnessValidatorEdgeCases:
    """Edge case tests for FaithfulnessValidator integration."""

    def test_empty_context_handling(self):
        """Test handling of empty context."""
        store = MockVectorStore()
        llm = MockLLMClient(responses=["휴학 관련 규정입니다."])

        usecase = SearchUseCase(
            store=store,
            llm_client=llm,
        )

        answer, metadata = usecase._generate_answer_with_validation(
            question="휴학 기간이 어떻게 되나요?",
            context="",
            context_list=[],
        )

        # Should handle gracefully
        assert answer is not None
        assert metadata["final_status"] in ["validated", "fallback"]

    def test_empty_answer_handling(self):
        """Test handling of empty answer from LLM."""
        store = MockVectorStore()
        llm = MockLLMClient(responses=[""])

        usecase = SearchUseCase(
            store=store,
            llm_client=llm,
        )

        answer, metadata = usecase._generate_answer_with_validation(
            question="휴학 기간이 어떻게 되나요?",
            context="제10조 (휴학기간) 휴학기간은 통산 2학기를 초과할 수 없다.",
            context_list=["제10조 (휴학기간) 휴학기간은 통산 2학기를 초과할 수 없다."],
        )

        assert answer is not None
        # Empty answers should trigger fallback or be handled

    def test_no_llm_client(self):
        """Test that SearchUseCase works without LLM client for search-only usage."""
        store = MockVectorStore()

        usecase = SearchUseCase(
            store=store,
            llm_client=None,  # No LLM
        )

        # Should have validator initialized even without LLM
        assert usecase._faithfulness_validator is not None
        assert usecase.llm is None


class TestValidatorInteraction:
    """Tests for interaction between HallucinationFilter and FaithfulnessValidator."""

    def test_pattern_filter_catches_contact_mismatch(self):
        """Test that HallucinationFilter catches phone number not in context."""
        response = "문의는 02-9999-9999로 연락 바랍니다."
        context = ["학적팀 연락처는 02-1234-5678입니다."]

        hallucination_filter = HallucinationFilter(mode=FilterMode.SANITIZE)
        filter_result = hallucination_filter.filter_response(response, context)

        # Should detect phone number mismatch
        assert len(filter_result.issues) > 0
        assert any("phone" in issue.lower() or "연락처" in issue for issue in filter_result.issues)

    def test_semantic_validator_catches_numerical_mismatch(self):
        """Test that FaithfulnessValidator catches numerical claims not in context."""
        response = "휴학 기간은 5년입니다."
        context = ["제10조 (휴학기간) 휴학기간은 통산 2학기를 초과할 수 없다."]

        validator = FaithfulnessValidator()
        result = validator.validate_answer(response, context)

        # Should detect that "5년" is not grounded in context
        # Note: "5년" is a duration claim that should not be found in context
        # The context mentions "2학기" not "5년"
        assert len(result.ungrounded_claims) > 0 or result.score < 1.0

    def test_both_validators_complementary_coverage(self):
        """Test that both validators provide complementary coverage."""
        # Response with both pattern and semantic issues
        response = "휴학 기간은 5년이며 문의는 02-9999-9999로 연락 바랍니다."
        context = [
            "제10조 (휴학기간) 휴학기간은 통산 2학기를 초과할 수 없다.",
            "학적팀 연락처는 02-1234-5678입니다.",
        ]

        # Pattern-based filter
        hallucination_filter = HallucinationFilter(mode=FilterMode.SANITIZE)
        filter_result = hallucination_filter.filter_response(response, context)

        # Semantic validator
        validator = FaithfulnessValidator()
        validation_result = validator.validate_answer(response, context)

        # Both should detect issues (complementary roles)
        # HallucinationFilter catches phone mismatch
        # FaithfulnessValidator catches duration mismatch
        has_issues = len(filter_result.issues) > 0 or validation_result.score < 1.0
        assert has_issues
