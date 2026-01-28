"""
Unit tests for RAG Quality Evaluator domain component.

Tests the LLM-as-Judge evaluation framework.
"""

import pytest

from src.rag.domain.evaluation.models import (
    EvaluationFramework,
    EvaluationResult,
    EvaluationThresholds,
    MetricScore,
)
from src.rag.domain.evaluation.quality_evaluator import RAGQualityEvaluator


class TestRAGQualityEvaluator:
    """Test RAG Quality Evaluator functionality."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator with default settings."""
        return RAGQualityEvaluator(
            framework=EvaluationFramework.RAGAS,
            judge_model="gpt-4o",
        )

    @pytest.fixture
    def sample_query_answer(self):
        """Create sample query and answer."""
        return {
            "query": "휴학 절차가 어떻게 되나요?",
            "answer": "휴학 신청은 학칙 제2조에 따라 학기 시작 14일 전까지 휴학원서를 제출해야 합니다.",
            "contexts": ["학칙 제2조: 휴학은 학기 시작 14일 전까지 신청해야 한다."],
        }

    def test_evaluator_initialization(self, evaluator):
        """WHEN evaluator is created, THEN should have correct attributes."""
        assert evaluator.framework == EvaluationFramework.RAGAS
        assert evaluator.judge_model == "gpt-4o"
        assert evaluator.thresholds is not None
        assert evaluator.thresholds.faithfulness == 0.90
        assert evaluator.thresholds.answer_relevancy == 0.85

    def test_evaluation_thresholds_defaults(self):
        """WHEN thresholds created, THEN should have correct default values."""
        thresholds = EvaluationThresholds()

        assert thresholds.faithfulness == 0.90
        assert thresholds.answer_relevancy == 0.85
        assert thresholds.contextual_precision == 0.80
        assert thresholds.contextual_recall == 0.80

        # Critical thresholds
        assert thresholds.faithfulness_critical == 0.70
        assert thresholds.relevancy_critical == 0.70

    def test_thresholds_check_minimum(self):
        """WHEN checking minimum threshold, THEN should return correct status."""
        thresholds = EvaluationThresholds()

        # Above threshold
        assert not thresholds.is_below_minimum("faithfulness", 0.92)

        # Below threshold
        assert thresholds.is_below_minimum("faithfulness", 0.85)

        # At threshold
        assert not thresholds.is_below_minimum("faithfulness", 0.90)

    def test_thresholds_check_critical(self):
        """WHEN checking critical threshold, THEN should return correct status."""
        thresholds = EvaluationThresholds()

        # Above critical
        assert not thresholds.is_below_critical("faithfulness", 0.75)

        # Below critical
        assert thresholds.is_below_critical("faithfulness", 0.65)

        # At critical
        assert not thresholds.is_below_critical("faithfulness", 0.70)

    @pytest.mark.asyncio
    async def test_evaluate_single_query(self, evaluator, sample_query_answer):
        """WHEN evaluating single query, THEN should return EvaluationResult."""
        result = await evaluator.evaluate(
            query=sample_query_answer["query"],
            answer=sample_query_answer["answer"],
            contexts=sample_query_answer["contexts"],
        )

        assert isinstance(result, EvaluationResult)
        assert result.query == sample_query_answer["query"]
        assert result.answer == sample_query_answer["answer"]
        assert result.contexts == sample_query_answer["contexts"]

        # Check scores are in valid range
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.answer_relevancy <= 1.0
        assert 0.0 <= result.contextual_precision <= 1.0
        assert 0.0 <= result.contextual_recall <= 1.0

        # Check overall score is calculated
        assert 0.0 <= result.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluation_result_to_dict(self, evaluator, sample_query_answer):
        """WHEN converting result to dict, THEN should include all fields."""
        result = await evaluator.evaluate(
            query=sample_query_answer["query"],
            answer=sample_query_answer["answer"],
            contexts=sample_query_answer["contexts"],
        )

        result_dict = result.to_dict()

        assert "query" in result_dict
        assert "answer" in result_dict
        assert "faithfulness" in result_dict
        assert "answer_relevancy" in result_dict
        assert "contextual_precision" in result_dict
        assert "contextual_recall" in result_dict
        assert "overall_score" in result_dict
        assert "passed" in result_dict
        assert "timestamp" in result_dict
        assert "metadata" in result_dict

    @pytest.mark.asyncio
    async def test_evaluation_result_from_dict(self):
        """WHEN creating result from dict, THEN should recreate object."""
        data = {
            "query": "Test query",
            "answer": "Test answer",
            "contexts": ["Context 1"],
            "faithfulness": 0.92,
            "answer_relevancy": 0.88,
            "contextual_precision": 0.85,
            "contextual_recall": 0.87,
            "overall_score": 0.88,
            "passed": True,
            "failure_reasons": [],
            "timestamp": "2025-01-28T12:00:00",
            "metadata": {},
        }

        result = EvaluationResult.from_dict(data)

        assert result.query == "Test query"
        assert result.faithfulness == 0.92
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, evaluator):
        """WHEN evaluating batch, THEN should return list of results."""
        test_cases = [
            {
                "query": f"Query {i}",
                "answer": f"Answer {i}",
                "contexts": [f"Context {i}"],
            }
            for i in range(5)
        ]

        results = await evaluator.evaluate_batch(test_cases)

        assert len(results) == 5
        assert all(isinstance(r, EvaluationResult) for r in results)


class TestMetricScore:
    """Test MetricScore model."""

    def test_metric_score_creation(self):
        """WHEN creating metric score, THEN should store values correctly."""
        score = MetricScore(
            name="faithfulness",
            score=0.92,
            passed=True,
            reason="Answer is factually consistent",
        )

        assert score.name == "faithfulness"
        assert score.score == 0.92
        assert score.passed is True
        assert "factually consistent" in score.reason


class TestEvaluationResult:
    """Test EvaluationResult model."""

    def test_evaluation_result_passed_all_metrics(self):
        """WHEN all metrics pass thresholds, THEN result should be passed."""
        result = EvaluationResult(
            query="Test query",
            answer="Test answer",
            contexts=["Context"],
            faithfulness=0.92,
            answer_relevancy=0.88,
            contextual_precision=0.85,
            contextual_recall=0.87,
            overall_score=0.88,
            passed=True,
        )

        assert result.passed is True
        assert len(result.failure_reasons) == 0

    def test_evaluation_result_failed_metric(self):
        """WHEN metric fails threshold, THEN result should include failure reason."""
        result = EvaluationResult(
            query="Test query",
            answer="Test answer",
            contexts=["Context"],
            faithfulness=0.85,  # Below 0.90 threshold
            answer_relevancy=0.88,
            contextual_precision=0.85,
            contextual_recall=0.87,
            overall_score=0.86,
            passed=False,
            failure_reasons=["Faithfulness below threshold: 0.850 < 0.900"],
        )

        assert result.passed is False
        assert len(result.failure_reasons) == 1
        assert "Faithfulness" in result.failure_reasons[0]
