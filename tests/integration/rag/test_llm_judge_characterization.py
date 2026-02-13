"""
Characterization tests for LLMJudge behavior preservation.

These tests capture the CURRENT behavior of LLMJudge to ensure
that integration of improved prompts does not break existing functionality.
"""

import pytest
from src.rag.domain.evaluation import LLMJudge, JudgeResult
from src.rag.infrastructure.llm_adapter import LLMClientAdapter
from src.rag.config import get_config


@pytest.fixture
def llm_judge():
    """Create LLMJudge instance for testing."""
    config = get_config()
    llm_client = LLMClientAdapter(
        provider=config.llm_provider,
        model=config.llm_model,
        base_url=config.llm_base_url,
    )
    return LLMJudge(llm_client=llm_client)


class TestLLMJudgeEvaluateBehavior:
    """Characterization tests for evaluate() method."""

    def test_evaluate_returns_judge_result(self, llm_judge):
        """Characterize: evaluate() returns JudgeResult with all metrics."""
        result = llm_judge.evaluate(
            query="휴학 방법 알려줘",
            answer="휴학 신청은 학기 시작 전에 해야 합니다.",
            sources=[],
        )
        # Capture current behavior
        assert isinstance(result, JudgeResult)
        assert hasattr(result, 'accuracy')
        assert hasattr(result, 'completeness')
        assert hasattr(result, 'citations')
        assert hasattr(result, 'context_relevance')
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'passed')

    def test_evaluate_accuracy_scoring(self, llm_judge):
        """Characterize: accuracy scoring for hallucination-free answers."""
        result = llm_judge.evaluate(
            query="휴학 기간은?",
            answer="일반휴학은 1년 이내 가능합니다.",
            sources=[{"score": 0.8}],
        )
        # Capture current scoring behavior
        assert 0.0 <= result.accuracy <= 1.0

    def test_evaluate_hallucination_detection(self, llm_judge):
        """Characterize: hallucination detection with fake phone numbers."""
        result = llm_judge.evaluate(
            query="휴학 신처 방법",
            answer="02-1234-5678로 전화하세요.",
            sources=[],
        )
        # Capture current behavior - should detect hallucination
        assert result.accuracy == 0.0

    def test_evaluate_citation_scoring(self, llm_judge):
        """Characterize: citation scoring for regulation references."""
        # Test with perfect citation
        result = llm_judge.evaluate(
            query="휴학 규정",
            answer="「직원복무규정」제26조에 따라...",
            sources=[],
        )
        # Capture current citation scoring
        assert 0.0 <= result.citations <= 1.0

    def test_evaluate_context_relevance(self, llm_judge):
        """Characterize: context relevance scoring."""
        result = llm_judge.evaluate(
            query="휴학 방법",
            answer="휴학은 학기 시작 전에 신청해야 합니다.",
            sources=[
                {"score": 0.9},
                {"score": 0.8},
                {"score": 0.7},
            ],
        )
        # Capture current relevance scoring
        assert 0.0 <= result.context_relevance <= 1.0


class TestLLMJudgeThresholds:
    """Characterization tests for quality thresholds."""

    def test_quality_thresholds_exist(self, llm_judge):
        """Characterize: LLMJudge has defined thresholds."""
        # Capture current threshold values
        assert hasattr(llm_judge, 'THRESHOLDS')
        assert 'overall' in llm_judge.THRESHOLDS
        assert 'accuracy' in llm_judge.THRESHOLDS
        assert 'completeness' in llm_judge.THRESHOLDS
        assert 'citations' in llm_judge.THRESHOLDS
        assert 'context_relevance' in llm_judge.THRESHOLDS

    def test_pass_fail_determination(self, llm_judge):
        """Characterize: pass/fail determination logic."""
        # Test with passing scores
        result = llm_judge.evaluate(
            query="테스트",
            answer="완벽한 답변입니다. 「규정」제1조에 따라 처리합니다.",
            sources=[{"score": 0.9}],
        )
        # Capture current pass/fail behavior
        assert isinstance(result.passed, bool)
