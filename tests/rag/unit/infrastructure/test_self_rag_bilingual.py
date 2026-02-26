"""
Tests for SPEC-RAG-003 Phase 1: Self-RAG Bilingual Support.

Tests:
- English keyword detection (Task 1.1)
- Bilingual prompt (Task 1.2)
- University topic override (Task 1.3)
- Bilingual rejection messages (Task 1.4)
"""

from unittest.mock import MagicMock

import pytest

from src.rag.config import reset_config
from src.rag.infrastructure.self_rag import (
    REGULATION_KEYWORDS,
    SelfRAGEvaluator,
)


class TestEnglishKeywordDetection:
    """Task 1.1: English keywords in REGULATION_KEYWORDS."""

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_english_keywords_present(self):
        """English regulation keywords MUST be in the list."""
        expected_en = [
            "tuition", "scholarship", "dormitory", "graduation",
            "registration", "course", "grade", "visa", "international",
            "leave", "absence", "professor", "faculty",
        ]
        for keyword in expected_en:
            assert keyword in REGULATION_KEYWORDS, (
                f"English keyword '{keyword}' missing from REGULATION_KEYWORDS"
            )

    def test_korean_keywords_still_present(self):
        """Korean keywords MUST still be present."""
        expected_ko = ["규정", "학칙", "장학금", "휴학", "졸업", "등록"]
        for keyword in expected_ko:
            assert keyword in REGULATION_KEYWORDS

    def test_english_query_detected_by_keywords(self):
        """English queries with known keywords MUST be detected."""
        evaluator = SelfRAGEvaluator(llm_client=None)
        en_queries = [
            "What are the tuition fees for international students?",
            "How do I apply for a leave of absence?",
            "What are the dormitory rules?",
            "How to apply for scholarship?",
            "What is the graduation requirement?",
        ]
        for query in en_queries:
            assert evaluator._has_regulation_keywords(query), (
                f"English query should match keywords: {query}"
            )

    def test_english_query_needs_retrieval(self):
        """English regulation queries MUST trigger retrieval (no LLM)."""
        evaluator = SelfRAGEvaluator(llm_client=None)
        assert evaluator.needs_retrieval(
            "What are the tuition fees for international students?"
        ) is True
        assert evaluator.needs_retrieval(
            "How to apply for scholarship?"
        ) is True


class TestBilingualPrompt:
    """Task 1.2: Bilingual RETRIEVAL_NEEDED_PROMPT."""

    def test_prompt_contains_english_examples(self):
        """Prompt MUST reference English examples."""
        prompt = SelfRAGEvaluator.RETRIEVAL_NEEDED_PROMPT
        assert "English" in prompt or "english" in prompt
        assert "tuition" in prompt.lower()
        assert "scholarship" in prompt.lower()

    def test_prompt_contains_korean_examples(self):
        """Prompt MUST still reference Korean examples."""
        prompt = SelfRAGEvaluator.RETRIEVAL_NEEDED_PROMPT
        assert "규정" in prompt or "학칙" in prompt
        assert "장학금" in prompt or "졸업" in prompt


class TestUniversityTopicOverride:
    """Task 1.3: University topic override mechanism."""

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_has_university_topic_words_2plus(self):
        """MUST detect 2+ university topic words."""
        evaluator = SelfRAGEvaluator(llm_client=None)
        assert evaluator._has_university_topic_words(
            "How do I apply for a student visa at university?"
        ) is True  # student, visa, university = 3

    def test_does_not_trigger_with_1_word(self):
        """MUST NOT trigger with only 1 topic word."""
        evaluator = SelfRAGEvaluator(llm_client=None)
        assert evaluator._has_university_topic_words(
            "What is the weather today university?"
        ) is False  # only 'university' = 1

    def test_override_when_llm_says_no(self):
        """LLM says NO but 2+ topic words → override to YES."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "[RETRIEVE_NO]"
        evaluator = SelfRAGEvaluator(llm_client=mock_llm)
        # Use words only in _UNIVERSITY_TOPIC_WORDS, not in REGULATION_KEYWORDS
        # "campus" + "semester" + "exam" → 3 topic words, 0 regulation keywords
        result = evaluator.needs_retrieval("Where can I find the campus exam for this semester?")
        assert result is True
        assert evaluator.get_metrics()["override_count"] >= 1


class TestBilingualRejection:
    """Task 1.4: Bilingual rejection messages."""

    def test_detect_query_language_english(self):
        """ASCII-heavy queries should be detected as English."""
        from src.rag.application.search_usecase import _detect_query_language
        assert _detect_query_language("What is the tuition fee?") == "en"
        assert _detect_query_language("How to apply for scholarship?") == "en"

    def test_detect_query_language_korean(self):
        """Korean queries should be detected as Korean."""
        from src.rag.application.search_usecase import _detect_query_language
        assert _detect_query_language("장학금 신청 방법이 뭔가요?") == "ko"
        assert _detect_query_language("휴학 절차 알려주세요") == "ko"

    def test_rejection_messages_exist(self):
        """Both language rejection messages MUST exist."""
        from src.rag.application.search_usecase import _REJECTION_MESSAGES
        assert "ko" in _REJECTION_MESSAGES
        assert "en" in _REJECTION_MESSAGES
        assert len(_REJECTION_MESSAGES["ko"]) > 10
        assert len(_REJECTION_MESSAGES["en"]) > 10

    def test_rejection_messages_contain_examples(self):
        """Rejection messages MUST contain query examples."""
        from src.rag.application.search_usecase import _REJECTION_MESSAGES
        assert "예시" in _REJECTION_MESSAGES["ko"] or "▶" in _REJECTION_MESSAGES["ko"]
        assert "Examples" in _REJECTION_MESSAGES["en"] or "▶" in _REJECTION_MESSAGES["en"]
