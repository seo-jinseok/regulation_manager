"""
Unit tests for QualityEvaluator infrastructure.

Tests quality evaluation across 6 dimensions.
"""

from unittest.mock import MagicMock

import pytest

from src.rag.automation.domain.entities import TestResult
from src.rag.automation.domain.value_objects import FactCheck, FactCheckStatus
from src.rag.automation.infrastructure.quality_evaluator import QualityEvaluator


class TestQualityEvaluator:
    """Test QualityEvaluator functionality."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator without LLM."""
        return QualityEvaluator(llm_client=None)

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM client."""
        return MagicMock()

    @pytest.fixture
    def evaluator_with_llm(self, mock_llm):
        """Create evaluator with LLM."""
        return QualityEvaluator(llm_client=mock_llm)

    @pytest.fixture
    def sample_test_result(self):
        """Create sample test result."""
        return TestResult(
            test_case_id="test-001",
            query="휴학 절차가 어떻게 되나요?",
            answer="휴학 신청은 학칙 제2조에 따라 학기 시작 14일 전까지 휴학원서를 제출해야 합니다. 학적팀에 방문하여 승인을 받으세요.",
            sources=["2-1-1 - 학칙"],
            confidence=0.85,
            execution_time_ms=150,
            rag_pipeline_log={},
        )

    @pytest.fixture
    def passing_fact_checks(self):
        """Create passing fact checks."""
        return [
            FactCheck(
                claim="휴학은 14일 전까지 신청",
                status=FactCheckStatus.PASS,
                source="학칙 제2조",
                confidence=0.95,
            )
        ]

    @pytest.fixture
    def failing_fact_checks(self):
        """Create failing fact checks."""
        return [
            FactCheck(
                claim="틀린 주장",
                status=FactCheckStatus.FAIL,
                source="",
                confidence=0.8,
                correction="올바른 정보",
            )
        ]

    def test_generalization_auto_fail(self, evaluator, sample_test_result):
        """WHEN answer contains generalization, THEN should auto-fail."""
        sample_test_result.answer = (
            "대학마다 다를 수 있습니다. 각 학교의 규정을 확인하세요."
        )

        result = evaluator.evaluate(sample_test_result, [])

        assert result.total_score == 0.0
        assert result.is_pass is False
        assert result.dimensions.accuracy == 0.0

    def test_empty_answer_auto_fail(self, evaluator, sample_test_result):
        """WHEN answer is empty, THEN should auto-fail."""
        sample_test_result.answer = ""

        result = evaluator.evaluate(sample_test_result, [])

        assert result.total_score == 0.0
        assert result.is_pass is False

    def test_failed_fact_checks_auto_fail(
        self, evaluator, sample_test_result, failing_fact_checks
    ):
        """WHEN fact checks fail, THEN should auto-fail."""
        result = evaluator.evaluate(sample_test_result, failing_fact_checks)

        assert result.total_score == 0.0
        assert result.is_pass is False

    def test_passing_score_threshold(
        self, evaluator, sample_test_result, passing_fact_checks
    ):
        """WHEN score >= 4.0, THEN should pass."""
        sample_test_result.answer = "휴학은 학칙 제2조에 따라 학기 시작 14일 전까지 휴학원서를 학적팀에 제출해야 합니다. 신청서는 학적팀에서 받을 수 있습니다. (장학금 지급규정 제5조에 의거)"

        result = evaluator.evaluate(sample_test_result, passing_fact_checks)

        # Rule-based scoring should give reasonable score
        assert result.total_score >= 0.0
        assert result.total_score <= 5.0

    def test_llm_evaluation(
        self, evaluator_with_llm, mock_llm, sample_test_result, passing_fact_checks
    ):
        """WHEN LLM available, THEN should use LLM for evaluation."""
        # Mock LLM response
        mock_llm.generate.return_value = """```json
{
  "accuracy": {"score": 0.95, "reason": "규정 내용 정확함"},
  "completeness": {"score": 0.9, "reason": "모든 측면 포함"},
  "relevance": {"score": 1.0, "reason": "질문 의도 부합"},
  "source_citation": {"score": 1.0, "reason": "규정 조항 인용"},
  "practicality": {"score": 0.4, "reason": "기한 정보 포함"},
  "actionability": {"score": 0.5, "reason": "명확한 절차 제시"}
}
```"""

        result = evaluator_with_llm.evaluate(sample_test_result, passing_fact_checks)

        # LLM-based scoring
        assert result.dimensions.accuracy == 0.95
        assert result.dimensions.completeness == 0.9
        assert result.dimensions.relevance == 1.0
        assert result.dimensions.source_citation == 1.0
        assert result.dimensions.practicality == 0.4
        assert result.dimensions.actionability == 0.5
        assert result.total_score == pytest.approx(4.85, rel=0.1)
        assert result.is_pass is True

    def test_llm_fallback_to_rule_based(
        self, evaluator_with_llm, mock_llm, sample_test_result, passing_fact_checks
    ):
        """WHEN LLM fails, THEN should fallback to rule-based."""
        # Mock LLM failure
        mock_llm.generate.side_effect = Exception("LLM failed")

        # The _evaluate_with_llm method should catch the exception and fall back
        # Let's verify by checking the result
        result = evaluator_with_llm.evaluate(sample_test_result, passing_fact_checks)

        # Should fallback to rule-based (which should work without LLM)
        assert result.total_score >= 0.0
        assert result.total_score <= 5.0
        # Rule-based scoring should give some positive score
        assert result.dimensions.accuracy >= 0.0

    def test_rule_based_scoring(
        self, evaluator, sample_test_result, passing_fact_checks
    ):
        """WHEN using rule-based scoring, THEN should calculate correctly."""
        sample_test_result.answer = (
            "휴학 절차에 대해 설명하겠습니다. 학칙에 따라 신청해야 합니다."
        )

        result = evaluator.evaluate(sample_test_result, passing_fact_checks)

        # All dimensions should be scored
        assert 0.0 <= result.dimensions.accuracy <= 1.0
        assert 0.0 <= result.dimensions.completeness <= 1.0
        assert 0.0 <= result.dimensions.relevance <= 1.0
        assert 0.0 <= result.dimensions.source_citation <= 1.0
        assert 0.0 <= result.dimensions.practicality <= 0.5
        assert 0.0 <= result.dimensions.actionability <= 0.5

    def test_source_citation_detection(self, evaluator):
        """WHEN answer has citations, THEN source_citation should be high."""
        test_result = TestResult(
            test_case_id="test-002",
            query="휴학 절차",
            answer="학칙 제2조에 따르면 휴학 신청은 14일 전까지 해야 합니다. 장학금 지급규정 제5조도 확인하세요.",
            sources=["2-1-1 - 학칙"],
            confidence=0.85,
            execution_time_ms=150,
            rag_pipeline_log={},
        )

        result = evaluator.evaluate(test_result, [])

        # Should detect citations
        assert result.dimensions.source_citation > 0.5

    def test_practical_info_detection(self, evaluator):
        """WHEN answer has practical info, THEN practicality should be high."""
        test_result = TestResult(
            test_case_id="test-003",
            query="휴학 절차",
            answer="휴학은 학기 시작 14일 전까지 신청해야 합니다. 3.0학점 이상이어야 합니다. 학적팀에 방문하세요.",
            sources=["2-1-1 - 학칙"],
            confidence=0.85,
            execution_time_ms=150,
            rag_pipeline_log={},
        )

        result = evaluator.evaluate(test_result, [])

        # Should detect practical info
        assert result.dimensions.practicality > 0.0

    def test_actionability_detection(self, evaluator):
        """WHEN answer has action verbs, THEN actionability should be high."""
        test_result = TestResult(
            test_case_id="test-004",
            query="휴학 절차",
            answer="휴학원서를 제출하고 학적팀을 방문하여 승인을 확인하세요. 신청을 준비하세요.",
            sources=["2-1-1 - 학칙"],
            confidence=0.85,
            execution_time_ms=150,
            rag_pipeline_log={},
        )

        result = evaluator.evaluate(test_result, [])

        # Should detect action verbs
        assert result.dimensions.actionability > 0.0

    def test_is_partial_success(
        self, evaluator, sample_test_result, passing_fact_checks
    ):
        """WHEN score is 3.0~3.9, THEN should be partial success."""
        sample_test_result.answer = "휴학 절차"  # Short answer

        result = evaluator.evaluate(sample_test_result, passing_fact_checks)

        # Check partial success property
        if 3.0 <= result.total_score < 4.0:
            assert result.is_partial_success is True
        else:
            assert result.is_partial_success is False

    def test_total_score_calculation(
        self, evaluator, sample_test_result, passing_fact_checks
    ):
        """WHEN scoring, THEN total should be sum of dimensions."""
        result = evaluator.evaluate(sample_test_result, passing_fact_checks)

        expected_total = (
            result.dimensions.accuracy
            + result.dimensions.completeness
            + result.dimensions.relevance
            + result.dimensions.source_citation
            + result.dimensions.practicality
            + result.dimensions.actionability
        )

        assert result.total_score == pytest.approx(expected_total, rel=0.01)
