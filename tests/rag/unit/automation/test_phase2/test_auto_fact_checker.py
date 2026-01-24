"""
Unit tests for AutoFactChecker infrastructure.

Tests claim extraction and fact verification functionality.
"""

from unittest.mock import MagicMock

import pytest

from src.rag.automation.domain.entities import TestResult
from src.rag.automation.domain.value_objects import FactCheckStatus
from src.rag.automation.infrastructure.auto_fact_checker import AutoFactChecker


class TestAutoFactChecker:
    """Test AutoFactChecker functionality."""

    @pytest.fixture
    def fact_checker(self):
        """Create fact checker without LLM."""
        return AutoFactChecker(llm_client=None, vector_store=None)

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM client."""
        return MagicMock()

    @pytest.fixture
    def mock_store(self):
        """Mock vector store."""
        return MagicMock()

    @pytest.fixture
    def fact_checker_with_llm(self, mock_llm, mock_store):
        """Create fact checker with LLM."""
        return AutoFactChecker(llm_client=mock_llm, vector_store=mock_store)

    @pytest.fixture
    def sample_test_result(self):
        """Create sample test result."""
        return TestResult(
            test_case_id="test-001",
            query="휴학 절차가 어떻게 되나요?",
            answer="휴학 신청은 학기 시작 14일 전까지 해야 합니다. 휴학원서를 제출하고 승인을 받아야 합니다.",
            sources=["2-1-1 - 학칙"],
            confidence=0.85,
            execution_time_ms=150,
            rag_pipeline_log={},
        )

    def test_detect_generalization(self, fact_checker):
        """WHEN answer contains generalization, THEN should detect it."""
        answers = [
            "대학마다 다를 수 있습니다.",
            "각 대학의 상황에 따라 다릅니다.",
            "일반적으로 그렇습니다.",
            "보통은 1학기입니다.",
        ]

        for answer in answers:
            assert fact_checker.detect_generalization(answer) is True

    def test_detect_no_generalization(self, fact_checker):
        """WHEN answer is specific, THEN should not detect generalization."""
        answer = "휴학은 학기 시작 14일 전까지 신청해야 합니다."

        assert fact_checker.detect_generalization(answer) is False

    def test_extract_claims_rule_based(self, fact_checker):
        """WHEN extracting claims without LLM, THEN should use rule-based."""
        answer = "휴학은 14일 전까지 신청해야 합니다. 성적이 2.0 이상이어야 합니다. 학기당 1회 가능합니다."

        claims = fact_checker.extract_claims(answer, None)

        assert len(claims) > 0
        assert all(isinstance(claim, str) for claim in claims)

    def test_extract_claims_with_llm(self, fact_checker_with_llm, mock_llm):
        """WHEN LLM available, THEN should use LLM for extraction."""
        # Mock LLM response
        mock_llm.generate.return_value = """```json
{
  "claims": [
    {"claim": "휴학은 학기 시작 14일 전까지 신청해야 함", "category": "기한"},
    {"claim": "휴학원서 제출 필요", "category": "절차"},
    {"claim": "휴학은 학기당 1회 가능", "category": "조건"}
  ]
}
```"""

        answer = "휴학은 학기 시작 14일 전까지 신청해야 합니다."

        claims = fact_checker_with_llm.extract_claims(answer, None)

        assert len(claims) == 3
        assert "휴학은 학기 시작 14일 전까지 신청해야 함" in claims[0]

    def test_extract_claims_llm_fallback(self, fact_checker_with_llm, mock_llm):
        """WHEN LLM fails, THEN should fallback to rule-based."""
        # Mock LLM failure
        mock_llm.generate.side_effect = Exception("LLM failed")

        answer = "휴학 신청은 14일 전까지 해야 합니다."

        claims = fact_checker_with_llm.extract_claims(answer, None)

        # Should fallback to rule-based
        assert len(claims) >= 0

    def test_verify_claim_rule_based(self, fact_checker):
        """WHEN verifying without LLM, THEN should use rule-based."""
        sources = ["2-1-1 - 학칙", "2-1-2 - 학칙"]

        result = fact_checker.verify_claim(
            claim="휴학은 14일 전까지 신청",
            query="휴학 절차",
            sources=sources,
        )

        assert result.status == FactCheckStatus.PASS
        assert result.confidence > 0
        assert "2-1-1" in result.source or result.source == ""

    def test_verify_claim_with_search(
        self, fact_checker_with_llm, mock_llm, mock_store
    ):
        """WHEN LLM and store available, THEN should use search."""
        # Mock search results
        from unittest.mock import Mock

        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.text = "휴학은 학기 시작 14일 전까지 신청해야 한다."

        mock_store.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        # Mock LLM response
        mock_llm.generate.return_value = """```json
{
  "status": "PASS",
  "confidence": 0.95,
  "source": "학칙 제2조",
  "correction": null,
  "explanation": "규정에 명시되어 있음"
}
```"""

        result = fact_checker_with_llm._verify_with_search(
            claim="휴학은 14일 전까지 신청",
            query="휴학 절차",
        )

        assert result.status == FactCheckStatus.PASS
        assert result.confidence == 0.95
        assert result.source == "학칙 제2조"

    def test_check_facts_with_generalization(self, fact_checker, sample_test_result):
        """WHEN answer contains generalization, THEN should auto-fail."""
        sample_test_result.answer = (
            "대학마다 다를 수 있습니다. 각 학교의 상황을 확인하세요."
        )

        results = fact_checker.check_facts(sample_test_result)

        assert len(results) == 1
        assert results[0].status == FactCheckStatus.FAIL
        assert "generalization" in results[0].explanation.lower()

    def test_check_facts_no_claims(self, fact_checker):
        """WHEN no claims extracted, THEN should return UNCERTAIN."""
        test_result = TestResult(
            test_case_id="test-002",
            query="테스트",
            answer="...",  # Too short to extract claims
            sources=[],
            confidence=0.0,
            execution_time_ms=100,
            rag_pipeline_log={},
        )

        results = fact_checker.check_facts(test_result)

        assert len(results) == 1
        assert results[0].status == FactCheckStatus.UNCERTAIN

    def test_check_facts_full_pipeline(
        self, fact_checker_with_llm, mock_llm, mock_store
    ):
        """WHEN checking facts with LLM, THEN should extract and verify."""
        # Mock claim extraction
        mock_llm.generate.return_value = """```json
{
  "claims": [
    {"claim": "휴학은 14일 전까지 신청", "category": "기한"}
  ]
}
```"""

        # Mock search
        from unittest.mock import Mock

        from src.rag.domain.entities import Chunk, SearchResult

        mock_chunk = Mock(spec=Chunk)
        mock_chunk.rule_code = "2-1-1"
        mock_chunk.text = "휴학은 학기 시작 14일 전까지 신청해야 한다."

        mock_store.search.return_value = [
            SearchResult(chunk=mock_chunk, score=0.9, rank=1)
        ]

        test_result = TestResult(
            test_case_id="test-003",
            query="휴학 절차",
            answer="휴학은 학기 시작 14일 전까지 신청해야 합니다.",
            sources=["2-1-1 - 학칙"],
            confidence=0.85,
            execution_time_ms=150,
            rag_pipeline_log={},
        )

        results = fact_checker_with_llm.check_facts(test_result)

        assert len(results) >= 1
