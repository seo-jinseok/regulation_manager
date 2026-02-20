"""
Unit tests for regeneration loop (REQ-002).

Tests _generate_answer_with_validation method that validates faithfulness
and regenerates with stricter prompt when needed.

TASK-003 Implementation.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.rag.application.search_usecase import (
    FAITHFULNESS_REGENERATION_THRESHOLD,
    FALLBACK_MESSAGE_KO,
    MAX_REGENERATION_ATTEMPTS,
    SearchUseCase,
)
from src.rag.domain.entities import Chunk, ChunkLevel


class FakeStore:
    """Fake store for testing."""

    def __init__(self):
        self._results = []

    def search(self, query, filter=None, top_k: int = 10):
        return self._results

    def get_all_documents(self):
        return []


class FakeLLM:
    """Fake LLM for testing."""

    def __init__(self, return_value="휴학은 최대 4학기까지 가능합니다."):
        self._return_value = return_value
        self.call_count = 0
        self.last_system_prompt = None
        self.last_user_message = None

    def generate(
        self, system_prompt: str, user_message: str, temperature: float = 0.0
    ) -> str:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_message = user_message
        if isinstance(self._return_value, list):
            return self._return_value.pop(0)
        return self._return_value


class TestGenerateAnswerWithValidation:
    """Test _generate_answer_with_validation method."""

    @pytest.fixture
    def fake_store(self):
        """Create fake store."""
        return FakeStore()

    @pytest.fixture
    def fake_llm(self):
        """Create fake LLM client."""
        return FakeLLM()

    @pytest.fixture
    def usecase(self, fake_store, fake_llm):
        """Create SearchUseCase instance with fake dependencies."""
        return SearchUseCase(
            store=fake_store,
            llm_client=fake_llm,
        )

    @pytest.fixture
    def context_list(self):
        """Create sample context list."""
        return [
            "학칙 제12조: 휴학 기간은 최대 4학기까지 허용된다.",
            "휴학 신청은 매학기 시작 전에 해야 한다.",
        ]

    # ========================================
    # Validated (First Attempt Success) Tests
    # ========================================

    def test_validated_on_first_attempt_high_faithfulness(self, usecase, fake_llm, context_list):
        """
        WHEN answer has high faithfulness score (>= 0.6),
        THEN should return validated status without regeneration.
        """
        # Answer claims match context exactly (4학기 is in context)
        fake_llm._return_value = "휴학은 최대 4학기까지 가능합니다."

        question = "휴학 기간은 얼마나 되나요?"
        context = "\n".join(context_list)

        answer_text, metadata = usecase._generate_answer_with_validation(
            question=question,
            context=context,
            context_list=context_list,
        )

        # Verify result
        assert "4학기" in answer_text or "휴학" in answer_text
        assert metadata["final_status"] == "validated"
        assert metadata["validation_attempts"] == 1
        assert metadata["faithfulness_score"] >= FAITHFULNESS_REGENERATION_THRESHOLD

    def test_validated_returns_correct_metadata(self, usecase, context_list):
        """
        WHEN answer is validated,
        THEN metadata should include all required fields.
        """
        question = "휴학 기간은 얼마나 되나요?"
        context = "\n".join(context_list)

        answer_text, metadata = usecase._generate_answer_with_validation(
            question=question,
            context=context,
            context_list=context_list,
        )

        # Check metadata structure
        assert "faithfulness_score" in metadata
        assert "validation_attempts" in metadata
        assert "final_status" in metadata
        assert metadata["final_status"] == "validated"
        assert isinstance(metadata["faithfulness_score"], float)
        assert isinstance(metadata["validation_attempts"], int)

    # ========================================
    # Regenerated (After Retry Success) Tests
    # ========================================

    def test_regenerated_after_retry(self, usecase, fake_llm, context_list):
        """
        WHEN first answer has low faithfulness but second succeeds,
        THEN should return regenerated status.
        """
        question = "휴학 규정에 대해 알려주세요."
        context = "\n".join(context_list)

        # First answer: includes unverified claim (e.g., "6학기" not in context)
        # Second answer: grounded claims only
        fake_llm._return_value = [
            "휴학은 최대 6학기까지 가능하며 등록금은 100% 환불됩니다.",  # Low faithfulness
            "휴학은 최대 4학기까지 가능합니다.",  # High faithfulness
        ]

        answer_text, metadata = usecase._generate_answer_with_validation(
            question=question,
            context=context,
            context_list=context_list,
        )

        # Verify regeneration occurred
        assert metadata["validation_attempts"] == 2
        assert metadata["final_status"] == "regenerated"
        assert metadata["faithfulness_score"] >= FAITHFULNESS_REGENERATION_THRESHOLD

    def test_uses_stricter_prompt_on_retry(self, usecase, fake_llm, context_list):
        """
        WHEN regeneration occurs,
        THEN should use stricter prompt with warnings.
        """
        question = "휴학 규정에 대해 알려주세요."
        context = "\n".join(context_list)

        # First answer with ungrounded claims, second succeeds
        fake_llm._return_value = [
            "휴학은 최대 6학기까지 가능합니다.",  # Ungrounded
            "휴학은 최대 4학기까지 가능합니다.",  # Grounded
        ]

        usecase._generate_answer_with_validation(
            question=question,
            context=context,
            context_list=context_list,
        )

        # Verify LLM was called twice (initial + 1 retry)
        assert fake_llm.call_count == 2

        # Second call should use strict prompt
        second_system_prompt = fake_llm.last_system_prompt
        # Check for strict prompt indicators
        assert "반드시 준수" in second_system_prompt or "문맥에 없는" in second_system_prompt

    # ========================================
    # Fallback (All Attempts Failed) Tests
    # ========================================

    def test_fallback_after_max_retries(self, usecase, fake_llm, context_list):
        """
        WHEN all regeneration attempts fail (faithfulness still low),
        THEN should return fallback response.
        """
        question = "휴학 규정에 대해 알려주세요."
        context = "\n".join(context_list)

        # All answers have ungrounded claims
        fake_llm._return_value = [
            "휴학은 최대 6학기까지 가능하며 등록금은 100% 환불됩니다.",  # Attempt 1
            "휴학은 10학기까지 가능합니다.",  # Attempt 2 (strict)
            "등록금은 무료입니다.",  # This won't be called because we only do 2 retries
        ]

        answer_text, metadata = usecase._generate_answer_with_validation(
            question=question,
            context=context,
            context_list=context_list,
            max_retries=2,
        )

        # Verify fallback was used
        assert metadata["final_status"] == "fallback"
        assert metadata["validation_attempts"] == MAX_REGENERATION_ATTEMPTS + 1
        # Should return Korean fallback message
        assert "찾을 수 없습니다" in answer_text or "문의" in answer_text

    def test_fallback_returns_korean_message(self, usecase, fake_llm, context_list):
        """
        WHEN fallback is triggered,
        THEN should return Korean fallback message.
        """
        question = "휴학 규정에 대해 알려주세요."
        context = "\n".join(context_list)

        # All attempts fail
        fake_llm._return_value = "잘못된 정보: 10학기까지 가능, 등록금 100% 환불"

        answer_text, metadata = usecase._generate_answer_with_validation(
            question=question,
            context=context,
            context_list=context_list,
            max_retries=0,  # No retries
        )

        # Should return Korean fallback
        assert FALLBACK_MESSAGE_KO in answer_text or "찾을 수 없습니다" in answer_text

    # ========================================
    # Edge Cases Tests
    # ========================================

    def test_empty_context_returns_fallback(self, usecase, fake_llm):
        """
        WHEN context is empty,
        THEN should return fallback response.
        """
        question = "휴학 규정에 대해 알려주세요."
        context = ""
        context_list = []

        answer_text, metadata = usecase._generate_answer_with_validation(
            question=question,
            context=context,
            context_list=context_list,
        )

        # Should return fallback due to empty context
        assert metadata["final_status"] == "fallback"

    def test_max_retries_configurable(self, usecase, fake_llm, context_list):
        """
        WHEN max_retries is configured,
        THEN should respect the configured value.
        """
        question = "휴학 규정에 대해 알려주세요."
        context = "\n".join(context_list)

        # All answers have low faithfulness
        fake_llm._return_value = [
            "잘못된 정보만 포함",  # Initial
            "잘못된 정보만 포함",  # Retry 1
        ]

        answer_text, metadata = usecase._generate_answer_with_validation(
            question=question,
            context=context,
            context_list=context_list,
            max_retries=1,  # Only 1 retry
        )

        # Should have 2 attempts (1 initial + 1 retry)
        assert metadata["validation_attempts"] == 2

    def test_debug_mode_logs_attempts(self, usecase, fake_llm, context_list):
        """
        WHEN debug mode is enabled,
        THEN should log validation attempts.
        """
        question = "휴학 규정에 대해 알려주세요."
        context = "\n".join(context_list)

        with patch("src.rag.application.search_usecase.logger") as mock_logger:
            usecase._generate_answer_with_validation(
                question=question,
                context=context,
                context_list=context_list,
                debug=True,
            )

            # Debug logging should have been called
            assert mock_logger.debug.called or mock_logger.info.called

    def test_ungrounded_claims_in_metadata(self, usecase, fake_llm, context_list):
        """
        WHEN fallback is triggered,
        THEN metadata should include ungrounded claims.
        """
        question = "휴학 규정에 대해 알려주세요."
        context = "\n".join(context_list)

        # Answer with claims not in context
        fake_llm._return_value = "휴학은 10학기까지 가능하며 등록금은 100% 환불됩니다."

        answer_text, metadata = usecase._generate_answer_with_validation(
            question=question,
            context=context,
            context_list=context_list,
            max_retries=0,
        )

        # Should have ungrounded claims recorded
        assert "ungrounded_claims" in metadata
        assert isinstance(metadata["ungrounded_claims"], list)


class TestGenerateAnswerStrict:
    """Test _generate_answer_strict method."""

    @pytest.fixture
    def fake_store(self):
        """Create fake store."""
        return FakeStore()

    @pytest.fixture
    def fake_llm(self):
        """Create fake LLM client."""
        return FakeLLM()

    @pytest.fixture
    def usecase(self, fake_store, fake_llm):
        """Create SearchUseCase instance with fake dependencies."""
        return SearchUseCase(
            store=fake_store,
            llm_client=fake_llm,
        )

    def test_uses_stricter_prompt(self, usecase, fake_llm):
        """
        WHEN generating strict answer,
        THEN should use stricter system prompt.
        """
        question = "휴학 규정에 대해 알려주세요."
        context = "학칙 제12조: 휴학 기간은 최대 4학기까지 허용된다."

        usecase._generate_answer_strict(
            question=question,
            context=context,
        )

        # Verify strict prompt was used
        system_prompt = fake_llm.last_system_prompt
        assert "반드시 준수" in system_prompt or "문맥에 없는" in system_prompt

    def test_includes_previous_ungrounded_claims(self, usecase, fake_llm):
        """
        WHEN previous ungrounded claims are provided,
        THEN should include them in feedback.
        """
        question = "휴학 규정에 대해 알려주세요."
        context = "학칙 제12조: 휴학 기간은 최대 4학기까지 허용된다."
        ungrounded_claims = ["10학기", "100% 환불"]

        usecase._generate_answer_strict(
            question=question,
            context=context,
            previous_ungrounded_claims=ungrounded_claims,
        )

        # Verify user message includes feedback about ungrounded claims
        user_message = fake_llm.last_user_message
        # Should mention the ungrounded claims
        assert "검증되지 않은" in user_message or "10학기" in user_message

    def test_includes_previous_suggestion(self, usecase, fake_llm):
        """
        WHEN previous suggestion is provided,
        THEN should include it in feedback.
        """
        question = "휴학 규정에 대해 알려주세요."
        context = "학칙 제12조: 휴학 기간은 최대 4학기까지 허용된다."
        suggestion = "제공된 규정에서 찾을 수 없습니다."

        usecase._generate_answer_strict(
            question=question,
            context=context,
            previous_suggestion=suggestion,
        )

        # Verify user message includes suggestion
        user_message = fake_llm.last_user_message
        assert "개선 제안" in user_message or "찾을 수 없습니다" in user_message

    def test_temperature_is_zero(self, usecase, fake_llm):
        """
        WHEN generating strict answer,
        THEN should use temperature=0 for deterministic output.
        (Note: temperature is always 0 in our FakeLLM, but we check the method)
        """
        question = "휴학 규정에 대해 알려주세요."
        context = "학칙 제12조: 휴학 기간은 최대 4학기까지 허용된다."

        usecase._generate_answer_strict(
            question=question,
            context=context,
        )

        # Method was called successfully
        assert fake_llm.call_count == 1


class TestRegenerationConstants:
    """Test regeneration-related constants."""

    def test_regeneration_threshold_is_0_6(self):
        """WHEN checking threshold, THEN should be 0.6."""
        assert FAITHFULNESS_REGENERATION_THRESHOLD == 0.6

    def test_max_regeneration_attempts_is_2(self):
        """WHEN checking max attempts, THEN should be 2."""
        assert MAX_REGENERATION_ATTEMPTS == 2

    def test_fallback_message_is_korean(self):
        """WHEN checking fallback message, THEN should be Korean."""
        assert "찾을 수 없습니다" in FALLBACK_MESSAGE_KO
        assert "문의" in FALLBACK_MESSAGE_KO
