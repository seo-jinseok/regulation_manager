"""
Unit tests for RAG Quality Evaluator domain component.

Tests the LLM-as-Judge evaluation framework with RAGAS metrics.
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
        """Create evaluator with mock evaluation (no API key required)."""
        return RAGQualityEvaluator(
            framework=EvaluationFramework.RAGAS,
            judge_model="gpt-4o",
            use_ragas=False,  # Force mock evaluation
        )

    @pytest.fixture
    def evaluator_with_ragas(self):
        """Create evaluator that will try to use RAGAS."""
        return RAGQualityEvaluator(
            framework=EvaluationFramework.RAGAS,
            judge_model="gpt-4o",
            judge_api_key="test-key",  # Mock API key
            use_ragas=True,
        )

    @pytest.fixture
    def sample_query_answer(self):
        """Create sample query and answer."""
        return {
            "query": "휴학 절차가 어떻게 되나요?",
            "answer": "휴학 신청은 학칙 제2조에 따라 학기 시작 14일 전까지 휴학원서를 제출해야 합니다.",
            "contexts": ["학칙 제2조: 휴학은 학기 시작 14일 전까지 신청해야 한다."],
            "ground_truth": "휴학 신청은 학기 시작 14일 전까지 해야 한다.",
        }

    @pytest.fixture
    def good_rag_sample(self):
        """Create high-quality RAG sample (should pass all thresholds)."""
        return {
            "query": "성적 정정은 어떻게 하나요?",
            "answer": "성적 정정은 학기 시작 후 2주 이내에 성적정정원을 제출해야 합니다.",
            "contexts": [
                "학칙 제15조: 성적 정정은 학기 시작 후 2주 이내에 신청해야 한다.",
                "성적 정정 서류: 성적정정원, 관련 증빙 서류",
            ],
            "ground_truth": "성적 정정은 학기 시작 후 2주 이내에 신청 가능",
        }

    @pytest.fixture
    def poor_rag_sample(self):
        """Create low-quality RAG sample (should fail some thresholds)."""
        return {
            "query": "등록금 납부 방법은?",
            "answer": "장학금을 신청하려면 성적이 우수해야 합니다.",  # Irrelevant answer
            "contexts": [
                "등록금은 학기 시작 전까지 납부해야 한다.",
                "가상계좌, 신용카드 등으로 납부 가능",
            ],
            "ground_truth": "등록금은 학기 시작 전에 가상계좌나 카드로 납부",
        }

    def test_evaluator_initialization(self, evaluator):
        """WHEN evaluator is created, THEN should have correct attributes."""
        assert evaluator.framework == EvaluationFramework.RAGAS
        assert evaluator.judge_model == "gpt-4o"
        assert evaluator.thresholds is not None
        assert evaluator.thresholds.faithfulness == 0.90
        assert evaluator.thresholds.answer_relevancy == 0.85
        assert evaluator.use_ragas is False  # Mock evaluation

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

        # Check metadata
        assert "evaluation_method" in result.metadata
        assert result.metadata["evaluation_method"] == "mock"

    @pytest.mark.asyncio
    async def test_evaluate_with_ground_truth(self, evaluator, sample_query_answer):
        """WHEN evaluating with ground truth, THEN should include recall score."""
        result = await evaluator.evaluate(
            query=sample_query_answer["query"],
            answer=sample_query_answer["answer"],
            contexts=sample_query_answer["contexts"],
            ground_truth=sample_query_answer["ground_truth"],
        )

        # Recall should be calculated
        assert 0.0 <= result.contextual_recall <= 1.0

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

    @pytest.mark.asyncio
    async def test_good_rag_passes_thresholds(self, evaluator, good_rag_sample):
        """WHEN RAG quality is good, THEN should pass all thresholds."""
        result = await evaluator.evaluate(
            query=good_rag_sample["query"],
            answer=good_rag_sample["answer"],
            contexts=good_rag_sample["contexts"],
            ground_truth=good_rag_sample["ground_truth"],
        )

        # Good samples should have decent scores (using mock evaluation)
        # Mock evaluation uses keyword overlap, so good samples should still pass
        assert result.overall_score >= 0.5

    @pytest.mark.asyncio
    async def test_poor_rag_fails_thresholds(self, evaluator, poor_rag_sample):
        """WHEN RAG quality is poor, THEN should fail some thresholds."""
        result = await evaluator.evaluate(
            query=poor_rag_sample["query"],
            answer=poor_rag_sample["answer"],
            contexts=poor_rag_sample["contexts"],
        )

        # Poor answer is irrelevant to query, should have low relevancy
        # Mock evaluation uses keyword overlap
        # Answer: "장학금을 신청하려면 성적이 우수해야 합니다."
        # Query: "등록금 납부 방법은?"
        # No keyword overlap = low score
        assert result.answer_relevancy < 0.85 or result.overall_score < 0.8

    @pytest.mark.asyncio
    async def test_faithfulness_calculation(self, evaluator):
        """WHEN calculating faithfulness, THEN should check context grounding."""
        # Answer with good context support
        result = await evaluator.evaluate(
            query="휴학 기간은?",
            answer="휴학 기간은 1년 이내로 한다.",  # Contains keywords from context
            contexts=["휴학 기간은 1년 이내로 허가한다."],
        )

        # Should have decent faithfulness due to keyword overlap
        assert 0.5 <= result.faithfulness <= 1.0

        # Check faithfulness-specific reason
        assert isinstance(result.faithfulness, float)

    @pytest.mark.asyncio
    async def test_answer_relevancy_calculation(self, evaluator):
        """WHEN calculating answer relevancy, THEN should check query addressing."""
        # Relevant answer
        result = await evaluator.evaluate(
            query="휴학 절차",
            answer="휴학 절차는 다음과 같습니다: 서류 제출, 승인 대기, 완료",
            contexts=[],
        )

        # Should have good relevancy due to keyword overlap
        assert 0.5 <= result.answer_relevancy <= 1.0

    @pytest.mark.asyncio
    async def test_contextual_precision_calculation(self, evaluator):
        """WHEN calculating contextual precision, THEN should check ranking quality."""
        result = await evaluator.evaluate(
            query="휴학 규정",
            answer="휴학에 관한 규정입니다.",
            contexts=[
                "휴학은 학기 시작 14일 전까지 신청한다.",  # Relevant
                "성적은 A+B/C/D/F로 표기한다.",  # Less relevant
            ],
        )

        # Should calculate precision score
        assert 0.5 <= result.contextual_precision <= 1.0

    @pytest.mark.asyncio
    async def test_contextual_recall_calculation(self, evaluator):
        """WHEN calculating contextual recall, THEN should check information completeness."""
        result = await evaluator.evaluate(
            query="휴학 규정",
            answer="휴학 신청은 학기 시작 14일 전까지 해야 합니다.",
            contexts=["휴학은 학기 시작 14일 전까지 신청한다."],
            ground_truth="휴학 신청 기간은 학기 시작 14일 전까지다.",
        )

        # Should calculate recall score
        assert 0.5 <= result.contextual_recall <= 1.0

    def test_mock_faithfulness_fallback(self, evaluator):
        """WHEN RAGAS unavailable, THEN should use mock faithfulness."""
        score = evaluator._mock_faithfulness(
            answer="휴학은 14일 전까지 신청해야 한다.",
            contexts=["휴학은 학기 시작 14일 전까지 해야 한다."],
        )

        assert isinstance(score, MetricScore)
        assert score.name == "faithfulness"
        assert 0.0 <= score.score <= 1.0
        assert score.reason is not None
        assert isinstance(score.passed, bool)

    def test_mock_answer_relevancy_fallback(self, evaluator):
        """WHEN RAGAS unavailable, THEN should use mock relevancy."""
        score = evaluator._mock_answer_relevancy(
            query="휴학 절차",
            answer="휴학 절차는 서류 제출 후 승인을 받아야 합니다.",
        )

        assert isinstance(score, MetricScore)
        assert score.name == "answer_relevancy"
        assert 0.0 <= score.score <= 1.0
        assert score.reason is not None

    def test_mock_contextual_precision_fallback(self, evaluator):
        """WHEN RAGAS unavailable, THEN should use mock precision."""
        score = evaluator._mock_contextual_precision(
            query="휴학 규정",
            contexts=[
                "휴학은 14일 전까지 신청한다.",
                "성적 정정은 2주 이내에 한다.",
            ],
        )

        assert isinstance(score, MetricScore)
        assert score.name == "contextual_precision"
        assert 0.0 <= score.score <= 1.0
        assert score.reason is not None

    def test_mock_contextual_recall_fallback(self, evaluator):
        """WHEN RAGAS unavailable, THEN should use mock recall."""
        # With ground truth
        score_with_gt = evaluator._mock_contextual_recall(
            contexts=["휴학은 14일 전까지 신청한다."],
            ground_truth="휴학 신청은 14일 전까지 해야 한다.",
        )

        assert isinstance(score_with_gt, MetricScore)
        assert score_with_gt.name == "contextual_recall"
        assert 0.0 <= score_with_gt.score <= 1.0

        # Without ground truth
        score_without_gt = evaluator._mock_contextual_recall(
            contexts=["휴학은 14일 전까지 신청한다."],
            ground_truth=None,
        )

        assert isinstance(score_without_gt, MetricScore)
        # Should assume good coverage when no ground truth
        assert score_without_gt.score >= 0.5

    @pytest.mark.asyncio
    async def test_critical_threshold_alert(self, evaluator):
        """WHEN faithfulness below critical, THEN should include critical alert."""
        # Create a sample that will fail faithfulness significantly
        result = await evaluator.evaluate(
            query="휴학 규정",
            answer="완전히 다른 내용에 대한 답변입니다.",  # No context support
            contexts=["휴학은 14일 전까지 신청한다."],
        )

        # If faithfulness is very low, should trigger critical alert
        if result.faithfulness < 0.70:
            assert any("CRITICAL" in reason for reason in result.failure_reasons)

    @pytest.mark.asyncio
    async def test_empty_contexts_handling(self, evaluator):
        """WHEN contexts are empty, THEN should handle gracefully."""
        result = await evaluator.evaluate(
            query="휴학 규정",
            answer="휴학에 관한 답변",
            contexts=[],
        )

        # Should still produce valid scores
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.contextual_precision <= 1.0
        assert 0.0 <= result.contextual_recall <= 1.0

    @pytest.mark.asyncio
    async def test_evaluation_result_aggregation(self, evaluator):
        """WHEN evaluating multiple samples, THEN should aggregate correctly."""
        test_cases = [
            {
                "query": "휴학 기간",
                "answer": "1년 이내",
                "contexts": ["휴학 기간은 1년 이내로 한다."],
            },
            {
                "query": "성적 정정",
                "answer": "2주 이내 신청",
                "contexts": ["성적 정정은 학기 시작 후 2주 이내에 한다."],
            },
            {
                "query": "등록금 납부",
                "answer": "학기 시작 전",
                "contexts": ["등록금은 학기 시작 전에 납부해야 한다."],
            },
        ]

        results = await evaluator.evaluate_batch(test_cases)

        # Check all results are valid
        assert len(results) == 3
        assert all(r.overall_score >= 0.0 for r in results)

        # Calculate average scores
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        assert 0.0 <= avg_faithfulness <= 1.0


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

    def test_evaluation_result_multiple_failures(self):
        """WHEN multiple metrics fail, THEN should list all failure reasons."""
        result = EvaluationResult(
            query="Test query",
            answer="Test answer",
            contexts=["Context"],
            faithfulness=0.85,  # Below 0.90
            answer_relevancy=0.75,  # Below 0.85
            contextual_precision=0.70,  # Below 0.80
            contextual_recall=0.87,
            overall_score=0.79,
            passed=False,
            failure_reasons=[
                "Faithfulness below threshold: 0.850 < 0.900",
                "Answer Relevancy below threshold: 0.750 < 0.850",
                "Contextual Precision below threshold: 0.700 < 0.800",
            ],
        )

        assert result.passed is False
        assert len(result.failure_reasons) == 3

    def test_overall_score_calculation(self):
        """WHEN calculating overall score, THEN should average all metrics."""
        result = EvaluationResult(
            query="Test query",
            answer="Test answer",
            contexts=["Context"],
            faithfulness=0.92,
            answer_relevancy=0.88,
            contextual_precision=0.85,
            contextual_recall=0.87,
            overall_score=(0.92 + 0.88 + 0.85 + 0.87) / 4.0,
            passed=True,
        )

        expected = (0.92 + 0.88 + 0.85 + 0.87) / 4.0
        assert abs(result.overall_score - expected) < 0.001


class TestCustomThresholds:
    """Test custom evaluation thresholds."""

    def test_custom_thresholds(self):
        """WHEN setting custom thresholds, THEN should use custom values."""
        custom_thresholds = EvaluationThresholds(
            faithfulness=0.95,
            answer_relevancy=0.90,
            contextual_precision=0.85,
            contextual_recall=0.85,
        )

        evaluator = RAGQualityEvaluator(
            thresholds=custom_thresholds,
            use_ragas=False,
        )

        assert evaluator.thresholds.faithfulness == 0.95
        assert evaluator.thresholds.answer_relevancy == 0.90
        assert evaluator.thresholds.contextual_precision == 0.85
        assert evaluator.thresholds.contextual_recall == 0.85

    @pytest.mark.asyncio
    async def test_custom_thresholds_evaluation(self):
        """WHEN evaluating with custom thresholds, THEN should use custom values."""
        custom_thresholds = EvaluationThresholds(
            faithfulness=0.95,  # Higher threshold
            answer_relevancy=0.90,
            contextual_precision=0.85,
            contextual_recall=0.85,
        )

        evaluator = RAGQualityEvaluator(
            thresholds=custom_thresholds,
            use_ragas=False,
        )

        result = await evaluator.evaluate(
            query="휴학 규정",
            answer="휴학은 14일 전까지 신청",
            contexts=["휴학은 14일 전까지 신청한다."],
        )

        # With higher threshold, more samples should fail
        # Check that the evaluator uses custom threshold
        assert evaluator.thresholds.faithfulness == 0.95

        # The score should be compared against 0.95, not 0.90
        passed_faithfulness = result.faithfulness >= 0.95
        assert result.passed == passed_faithfulness or result.faithfulness < 0.95
