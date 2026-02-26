"""
Characterization tests for SPEC-RAG-003 behavior preservation.

These tests capture the CURRENT BEHAVIOR of components being modified:
- SelfRAGEvaluator: keyword detection, needs_retrieval
- SearchUseCase: answer generation, rejection messages
- HybridSearcher: weight fusion
- RetrievalEvaluator: threshold logic

Any regression in these tests means SPEC-RAG-003 broke existing behavior.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.rag.config import reset_config
from src.rag.infrastructure.self_rag import (
    REGULATION_KEYWORDS,
    SelfRAGEvaluator,
)


@pytest.mark.characterization
class TestSelfRAGKeywordPreserve:
    """Preserve existing Korean keyword detection behavior."""

    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_korean_regulation_keywords_detected(self):
        """Korean regulation queries MUST continue to be detected."""
        evaluator = SelfRAGEvaluator(llm_client=None)
        korean_queries = [
            "장학금 종류와 신청방법 알려주세요",
            "휴학 절차가 어떻게 되나요?",
            "졸업 요건이 뭔가요?",
            "등록금 납부 기간은 언제인가요?",
            "교수 임용 규정이 어떻게 되나요?",
        ]
        for query in korean_queries:
            assert evaluator._has_regulation_keywords(query), (
                f"REGRESSION: Korean query should have keywords detected: {query}"
            )

    def test_needs_retrieval_without_llm_defaults_true(self):
        """Without LLM, needs_retrieval MUST return True."""
        evaluator = SelfRAGEvaluator(llm_client=None)
        assert evaluator.needs_retrieval("아무 질문") is True
        assert evaluator.needs_retrieval("hello") is True

    def test_keyword_bypass_increments_metrics(self):
        """Keyword bypass MUST increment bypass_count metric."""
        evaluator = SelfRAGEvaluator(llm_client=None)
        evaluator.needs_retrieval("장학금 신청 방법")
        metrics = evaluator.get_metrics()
        assert metrics["bypass_count"] >= 1

    def test_llm_retrieve_no_with_keywords_overrides(self):
        """If LLM says NO but keywords exist, MUST override to YES."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "[RETRIEVE_NO]"
        evaluator = SelfRAGEvaluator(llm_client=mock_llm)
        # This has Korean keywords so it'll bypass LLM entirely
        assert evaluator.needs_retrieval("장학금 신청") is True


@pytest.mark.characterization
class TestSelfRAGRelevancePreserve:
    """Preserve relevance evaluation behavior."""

    def test_evaluate_relevance_no_llm_returns_results(self):
        """Without LLM, evaluate_relevance returns all results as relevant."""
        evaluator = SelfRAGEvaluator(llm_client=None)
        mock_result = Mock()
        mock_result.chunk = Mock()
        mock_result.chunk.text = "테스트"
        mock_result.chunk.title = "테스트 규정"
        is_relevant, results = evaluator.evaluate_relevance("test", [mock_result])
        assert is_relevant is True
        assert len(results) == 1

    def test_evaluate_relevance_empty_results(self):
        """Empty results MUST return (False, [])."""
        evaluator = SelfRAGEvaluator(llm_client=None)
        is_relevant, results = evaluator.evaluate_relevance("test", [])
        assert is_relevant is False
        assert len(results) == 0


@pytest.mark.characterization
class TestRetrievalEvaluatorPreserve:
    """Preserve RetrievalEvaluator threshold behavior."""

    def test_default_thresholds_from_config(self):
        """Default thresholds MUST use config values."""
        from src.rag.infrastructure.retrieval_evaluator import RetrievalEvaluator

        evaluator = RetrievalEvaluator()
        # Document current thresholds
        assert "simple" in evaluator._thresholds
        assert "medium" in evaluator._thresholds
        assert "complex" in evaluator._thresholds

    def test_empty_results_need_correction(self):
        """Empty results MUST trigger correction."""
        from src.rag.infrastructure.retrieval_evaluator import RetrievalEvaluator

        evaluator = RetrievalEvaluator()
        assert evaluator.needs_correction("test", []) is True

    def test_evaluate_empty_returns_zero(self):
        """Empty results MUST score 0.0."""
        from src.rag.infrastructure.retrieval_evaluator import RetrievalEvaluator

        evaluator = RetrievalEvaluator()
        assert evaluator.evaluate("test", []) == 0.0


@pytest.mark.characterization
class TestHybridSearcherPreserve:
    """Preserve HybridSearcher weight behavior."""

    def test_default_weights(self):
        """Default weights MUST be bm25=0.3, dense=0.7."""
        from src.rag.infrastructure.hybrid_search import HybridSearcher

        searcher = HybridSearcher(tokenize_mode="simple")
        assert searcher.bm25_weight == 0.3
        assert searcher.dense_weight == 0.7

    def test_custom_weights_accepted(self):
        """Custom weights MUST be accepted."""
        from src.rag.infrastructure.hybrid_search import HybridSearcher

        searcher = HybridSearcher(
            bm25_weight=0.5,
            dense_weight=0.5,
            tokenize_mode="simple",
        )
        assert searcher.bm25_weight == 0.5
        assert searcher.dense_weight == 0.5


@pytest.mark.characterization
class TestFallbackMessagesPreserve:
    """Preserve fallback message content."""

    def test_korean_fallback_message_exists(self):
        """Korean fallback message MUST exist."""
        from src.rag.application.search_usecase import FALLBACK_MESSAGE_KO

        assert "규정" in FALLBACK_MESSAGE_KO
        assert len(FALLBACK_MESSAGE_KO) > 10

    def test_english_fallback_message_exists(self):
        """English fallback message MUST exist."""
        from src.rag.application.search_usecase import FALLBACK_MESSAGE_EN

        assert "regulation" in FALLBACK_MESSAGE_EN.lower()
        assert len(FALLBACK_MESSAGE_EN) > 10
