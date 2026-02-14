"""
Characterization tests for LLMJudge module.

These tests document the current behavior of the LLMJudge evaluation system
without prescribing how it should behave. They serve as a safety net for
future refactoring.

Module under test: src/rag/domain/evaluation/llm_judge.py
"""

import pytest
from dataclasses import asdict
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.rag.domain.evaluation.llm_judge import (
    LLMJudge,
    JudgeResult,
    QualityLevel,
    EvaluationSummary,
    EvaluationBatch,
)


class TestQualityLevel:
    """Characterization tests for QualityLevel enum."""

    def test_quality_level_values(self):
        """Document QualityLevel enum values."""
        # Arrange & Act
        levels = {level.name: level.value for level in QualityLevel}

        # Assert - Document current enum values
        assert levels["EXCELLENT"] == "excellent"
        assert levels["GOOD"] == "good"
        assert levels["ACCEPTABLE"] == "acceptable"
        assert levels["POOR"] == "poor"
        assert levels["FAILING"] == "failing"


class TestJudgeResult:
    """Characterization tests for JudgeResult dataclass."""

    def test_judge_result_creation_with_minimal_fields(self):
        """Document JudgeResult creation with minimal required fields."""
        # Arrange & Act
        result = JudgeResult(
            query="test query",
            answer="test answer",
            sources=[{"content": "source"}],
            accuracy=0.9,
            completeness=0.8,
            citations=0.7,
            context_relevance=0.85,
            overall_score=0.82,
            passed=True,
        )

        # Assert - Document field initialization
        assert result.query == "test query"
        assert result.answer == "test answer"
        assert len(result.sources) == 1
        assert result.accuracy == 0.9
        assert result.completeness == 0.8
        assert result.citations == 0.7
        assert result.context_relevance == 0.85
        assert result.overall_score == 0.82
        assert result.passed is True

    def test_judge_result_auto_generates_evaluation_id(self):
        """Document that evaluation_id is auto-generated when not provided."""
        # Arrange & Act
        result = JudgeResult(
            query="test",
            answer="answer",
            sources=[],
            accuracy=0.5,
            completeness=0.5,
            citations=0.5,
            context_relevance=0.5,
            overall_score=0.5,
            passed=False,
        )

        # Assert
        assert result.evaluation_id != ""
        assert result.evaluation_id.startswith("eval_")

    def test_judge_result_auto_generates_timestamp(self):
        """Document that timestamp is auto-generated when not provided."""
        # Arrange & Act
        result = JudgeResult(
            query="test",
            answer="answer",
            sources=[],
            accuracy=0.5,
            completeness=0.5,
            citations=0.5,
            context_relevance=0.5,
            overall_score=0.5,
            passed=False,
        )

        # Assert - timestamp should be ISO format
        assert result.timestamp != ""
        # Should be parseable as datetime
        parsed = datetime.fromisoformat(result.timestamp)
        assert parsed is not None

    def test_judge_result_default_values(self):
        """Document default values for optional fields."""
        # Arrange & Act
        result = JudgeResult(
            query="test",
            answer="answer",
            sources=[],
            accuracy=0.5,
            completeness=0.5,
            citations=0.5,
            context_relevance=0.5,
            overall_score=0.5,
            passed=False,
        )

        # Assert - Document defaults
        assert result.reasoning == {}
        assert result.issues == []
        assert result.strengths == []


class TestLLMJudgeInit:
    """Characterization tests for LLMJudge initialization."""

    def test_init_with_default_llm_client(self):
        """Document LLMJudge initialization with default client."""
        # Arrange & Act
        with patch("src.rag.domain.evaluation.llm_judge.LLMClientAdapter") as mock_adapter:
            with patch("src.rag.domain.evaluation.llm_judge.get_config") as mock_config:
                mock_config.return_value = MagicMock(
                    llm_provider="test",
                    llm_model="test-model",
                    llm_base_url=None,
                )
                judge = LLMJudge()

        # Assert
        assert judge.llm_client is not None

    def test_init_with_custom_llm_client(self):
        """Document LLMJudge initialization with custom client."""
        # Arrange
        mock_client = MagicMock()

        # Act
        judge = LLMJudge(llm_client=mock_client)

        # Assert
        assert judge.llm_client is mock_client

    def test_thresholds_constants(self):
        """Document LLMJudge threshold constants."""
        # Assert - Document threshold values
        assert LLMJudge.THRESHOLDS["overall"] == 0.80
        assert LLMJudge.THRESHOLDS["accuracy"] == 0.85
        assert LLMJudge.THRESHOLDS["completeness"] == 0.75
        assert LLMJudge.THRESHOLDS["citations"] == 0.70
        assert LLMJudge.THRESHOLDS["context_relevance"] == 0.75

    def test_hallucination_patterns(self):
        """Document hallucination detection patterns."""
        # Assert - Document patterns used for hallucination detection
        patterns = LLMJudge.HALLUCINATION_PATTERNS

        assert len(patterns) == 5
        # Phone number pattern
        assert r"02-\d{3,4}-\d{4}" in patterns
        # University names to avoid
        assert "서울대" in " ".join(patterns)

    def test_avoidance_phrases(self):
        """Document avoidance phrase patterns."""
        # Assert - Document phrases that indicate generic avoidance
        phrases = LLMJudge.AVOIDANCE_PHRASES

        assert "대학마다 다릅니다" in phrases
        assert "확인해주세요" in phrases
        assert "일반적으로" in phrases


class TestLLMJudgeEvaluate:
    """Characterization tests for LLMJudge.evaluate method."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        return MagicMock()

    @pytest.fixture
    def judge(self, mock_llm_client):
        """Create LLMJudge with mock client."""
        return LLMJudge(llm_client=mock_llm_client)

    def test_evaluate_returns_judge_result(self, judge):
        """Document that evaluate returns JudgeResult."""
        # Arrange
        query = "test query"
        answer = "test answer"
        sources = [{"content": "source", "score": 0.8}]

        # Act
        result = judge.evaluate(query, answer, sources)

        # Assert
        assert isinstance(result, JudgeResult)

    def test_evaluate_with_high_quality_sources(self, judge):
        """Document evaluation with high quality sources."""
        # Arrange
        query = "test query"
        answer = "This is a good answer with regulation references."
        sources = [{"content": "source content", "score": 0.9}]

        # Act
        result = judge.evaluate(query, answer, sources)

        # Assert - Document behavior with high quality sources
        assert result.accuracy > 0.8  # Should be high with good source score

    def test_evaluate_with_low_quality_sources(self, judge):
        """Document evaluation with low quality sources."""
        # Arrange
        query = "test query"
        answer = "This is an answer."
        sources = [{"content": "source content", "score": 0.3}]

        # Act
        result = judge.evaluate(query, answer, sources)

        # Assert - Document behavior with low quality sources
        assert result.accuracy < 0.8

    def test_evaluate_with_no_sources(self, judge):
        """Document evaluation with no sources."""
        # Arrange
        query = "test query"
        answer = "This is an answer."
        sources = []

        # Act
        result = judge.evaluate(query, answer, sources)

        # Assert - Document behavior with no sources
        assert result.accuracy == 0.5  # Default score without sources

    def test_evaluate_with_expected_info(self, judge):
        """Document evaluation with expected information."""
        # Arrange
        query = "test query"
        answer = "This answer contains expected information points."
        sources = [{"content": "source", "score": 0.8}]
        expected_info = ["expected", "information"]

        # Act
        result = judge.evaluate(query, answer, sources, expected_info)

        # Assert - Completeness should reflect expected info coverage
        assert 0.0 <= result.completeness <= 1.0

    def test_evaluate_with_empty_answer(self, judge):
        """Document evaluation with empty answer."""
        # Arrange
        query = "test query"
        answer = ""
        sources = [{"content": "source", "score": 0.8}]

        # Act
        result = judge.evaluate(query, answer, sources)

        # Assert - Empty answer should result in low accuracy
        assert result.accuracy == 0.0

    def test_evaluate_with_whitespace_only_answer(self, judge):
        """Document evaluation with whitespace-only answer."""
        # Arrange
        query = "test query"
        answer = "   "
        sources = [{"content": "source", "score": 0.8}]

        # Act
        result = judge.evaluate(query, answer, sources)

        # Assert
        assert result.accuracy == 0.0

    def test_evaluate_with_hallucination_pattern(self, judge):
        """Document detection of hallucination patterns."""
        # Arrange
        query = "test query"
        answer = "Call us at 02-1234-5678 for more information."
        sources = [{"content": "source", "score": 0.8}]

        # Act
        result = judge.evaluate(query, answer, sources)

        # Assert - Hallucination pattern should result in failure
        assert result.accuracy == 0.0
        assert result.passed is False

    def test_evaluate_with_wrong_university_name(self, judge):
        """Document detection of wrong university names."""
        # Arrange
        query = "test query"
        answer = "You should contact 서울대 for more information."
        sources = [{"content": "source", "score": 0.8}]

        # Act
        result = judge.evaluate(query, answer, sources)

        # Assert - Wrong university name should be detected
        assert result.accuracy == 0.0

    def test_evaluate_with_avoidance_phrase(self, judge):
        """Document detection of avoidance phrases."""
        # Arrange
        query = "test query"
        answer = "대학마다 다릅니다"  # Short answer with avoidance phrase
        sources = [{"content": "source", "score": 0.8}]

        # Act
        result = judge.evaluate(query, answer, sources)

        # Assert - Avoidance phrase should reduce score
        assert result.accuracy == 0.3


class TestLLMJudgeAccuracyEvaluation:
    """Characterization tests for accuracy evaluation."""

    @pytest.fixture
    def judge(self):
        """Create LLMJudge with mock client."""
        mock_client = MagicMock()
        return LLMJudge(llm_client=mock_client)

    def test_evaluate_accuracy_high_source_score(self, judge):
        """Document accuracy with high source score."""
        # Arrange
        sources = [{"score": 0.9}]

        # Act
        accuracy = judge._evaluate_accuracy("query", "answer", sources)

        # Assert
        assert accuracy == 0.95

    def test_evaluate_accuracy_medium_source_score(self, judge):
        """Document accuracy with medium source score."""
        # Arrange
        sources = [{"score": 0.7}]

        # Act
        accuracy = judge._evaluate_accuracy("query", "answer", sources)

        # Assert
        assert accuracy == 0.85

    def test_evaluate_accuracy_low_source_score(self, judge):
        """Document accuracy with low source score."""
        # Arrange
        sources = [{"score": 0.5}]

        # Act
        accuracy = judge._evaluate_accuracy("query", "answer", sources)

        # Assert
        assert accuracy == 0.75

    def test_evaluate_accuracy_very_low_source_score(self, judge):
        """Document accuracy with very low source score."""
        # Arrange
        sources = [{"score": 0.3}]

        # Act
        accuracy = judge._evaluate_accuracy("query", "answer", sources)

        # Assert
        assert accuracy == 0.65


class TestLLMJudgeCompletenessEvaluation:
    """Characterization tests for completeness evaluation."""

    @pytest.fixture
    def judge(self):
        """Create LLMJudge with mock client."""
        mock_client = MagicMock()
        return LLMJudge(llm_client=mock_client)

    def test_completeness_with_no_expected_info_long_answer(self, judge):
        """Document completeness estimation for long answers."""
        # Arrange
        answer = "a" * 400  # Long answer

        # Act
        completeness = judge._evaluate_completeness("query", answer, None)

        # Assert
        assert completeness == 0.85

    def test_completeness_with_no_expected_info_medium_answer(self, judge):
        """Document completeness estimation for medium answers."""
        # Arrange
        answer = "a" * 200  # Medium answer

        # Act
        completeness = judge._evaluate_completeness("query", answer, None)

        # Assert
        assert completeness == 0.75

    def test_completeness_with_no_expected_info_short_answer(self, judge):
        """Document completeness estimation for short answers."""
        # Arrange
        answer = "a" * 100  # Short answer

        # Act
        completeness = judge._evaluate_completeness("query", answer, None)

        # Assert
        assert completeness == 0.60

    def test_completeness_with_expected_info(self, judge):
        """Document completeness with expected information."""
        # Arrange
        answer = "This contains expected and information"
        expected_info = ["expected", "information", "missing"]

        # Act
        completeness = judge._evaluate_completeness("query", answer, expected_info)

        # Assert - 2 out of 3 expected items covered
        assert completeness == pytest.approx(2/3, rel=0.1)


class TestLLMJudgeCitationsEvaluation:
    """Characterization tests for citations evaluation."""

    @pytest.fixture
    def judge(self):
        """Create LLMJudge with mock client."""
        mock_client = MagicMock()
        return LLMJudge(llm_client=mock_client)

    def test_citations_perfect_format(self, judge):
        """Document perfect citation format detection."""
        # Arrange
        answer = "학칙규정 제10조에 따르면..."

        # Act
        citations = judge._evaluate_citations(answer)

        # Assert
        assert citations == 1.0

    def test_citations_good_format(self, judge):
        """Document good citation format detection."""
        # Arrange
        answer = "규정에 따르면 제5조에..."

        # Act
        citations = judge._evaluate_citations(answer)

        # Assert
        assert citations == 0.85

    def test_citations_fair_format(self, judge):
        """Document fair citation format detection."""
        # Arrange
        answer = "규정에 따르면..."

        # Act
        citations = judge._evaluate_citations(answer)

        # Assert
        assert citations == 0.60

    def test_citations_poor_format(self, judge):
        """Document poor citation format detection."""
        # Arrange - "관련"과 "조"가 있는데 "규정"은 없는 경우
        # 실제 코드에서는 "규정"이 먼저 체크되므로 "관련 규정"은 0.60이 됨
        # "관련" + "조"만 있을 때 0.30
        answer = "관련 조항을 참조하세요..."

        # Act
        citations = judge._evaluate_citations(answer)

        # Assert - Has 관련 and 조 without 규정, so score is 0.30
        assert citations == 0.30

    def test_citations_fair_format_only_regulation(self, judge):
        """Document fair citation format with only regulation mention."""
        # Arrange - Only has 규정, no 제X조 pattern
        answer = "규정에 명시된 대로..."

        # Act
        citations = judge._evaluate_citations(answer)

        # Assert - Only 규정 without 제X조 gives 0.60
        assert citations == 0.60

    def test_citations_no_citation(self, judge):
        """Document no citation detection."""
        # Arrange
        answer = "This answer has no citations."

        # Act
        citations = judge._evaluate_citations(answer)

        # Assert
        assert citations == 0.0


class TestLLMJudgeContextRelevanceEvaluation:
    """Characterization tests for context relevance evaluation."""

    @pytest.fixture
    def judge(self):
        """Create LLMJudge with mock client."""
        mock_client = MagicMock()
        return LLMJudge(llm_client=mock_client)

    def test_context_relevance_empty_sources(self, judge):
        """Document context relevance with empty sources."""
        # Arrange
        sources = []

        # Act
        relevance = judge._evaluate_context_relevance(sources)

        # Assert - Returns minimal base score
        assert relevance == 0.2

    def test_context_relevance_single_source(self, judge):
        """Document context relevance with single source."""
        # Arrange
        sources = [{"score": 0.8}]

        # Act
        relevance = judge._evaluate_context_relevance(sources)

        # Assert
        assert relevance == 0.8

    def test_context_relevance_multiple_sources(self, judge):
        """Document context relevance with multiple sources."""
        # Arrange
        sources = [
            {"score": 0.9},
            {"score": 0.8},
            {"score": 0.7},
        ]

        # Act
        relevance = judge._evaluate_context_relevance(sources)

        # Assert - First sources weighted higher
        assert relevance < 0.9  # Weighted average

    def test_context_relevance_source_with_no_score(self, judge):
        """Document context relevance when source has no score."""
        # Arrange
        sources = [{}]

        # Act
        relevance = judge._evaluate_context_relevance(sources)

        # Assert - Default score is 0.5
        assert relevance == 0.5

    def test_context_relevance_source_with_zero_score(self, judge):
        """Document context relevance when source has zero score."""
        # Arrange
        sources = [{"score": 0}]

        # Act
        relevance = judge._evaluate_context_relevance(sources)

        # Assert - Zero score converted to minimum
        assert relevance == 0.3


class TestLLMJudgePassFail:
    """Characterization tests for pass/fail determination."""

    @pytest.fixture
    def judge(self):
        """Create LLMJudge with mock client."""
        mock_client = MagicMock()
        return LLMJudge(llm_client=mock_client)

    def test_determine_pass_fail_all_thresholds_met(self, judge):
        """Document pass when all thresholds met."""
        # Arrange
        accuracy = 0.90
        completeness = 0.80
        citations = 0.75
        context_relevance = 0.80
        overall = 0.82

        # Act
        passed = judge._determine_pass_fail(
            accuracy, completeness, citations, context_relevance, overall
        )

        # Assert
        assert passed is True

    def test_determine_pass_fail_below_accuracy_threshold(self, judge):
        """Document fail when below accuracy threshold."""
        # Arrange
        accuracy = 0.80  # Below 0.85 threshold
        completeness = 0.80
        citations = 0.75
        context_relevance = 0.80
        overall = 0.82

        # Act
        passed = judge._determine_pass_fail(
            accuracy, completeness, citations, context_relevance, overall
        )

        # Assert
        assert passed is False

    def test_determine_pass_fail_below_completeness_threshold(self, judge):
        """Document fail when below completeness threshold."""
        # Arrange
        accuracy = 0.90
        completeness = 0.70  # Below 0.75 threshold
        citations = 0.75
        context_relevance = 0.80
        overall = 0.82

        # Act
        passed = judge._determine_pass_fail(
            accuracy, completeness, citations, context_relevance, overall
        )

        # Assert
        assert passed is False


class TestLLMJudgeExplanationMethods:
    """Characterization tests for explanation generation."""

    @pytest.fixture
    def judge(self):
        """Create LLMJudge with mock client."""
        mock_client = MagicMock()
        return LLMJudge(llm_client=mock_client)

    def test_explain_accuracy_excellent(self, judge):
        """Document accuracy explanation for excellent score."""
        result = judge._explain_accuracy(0.95)
        assert "완벽한 정확도" in result

    def test_explain_accuracy_good(self, judge):
        """Document accuracy explanation for good score."""
        result = judge._explain_accuracy(0.88)
        assert "우수한 정확도" in result

    def test_explain_accuracy_fair(self, judge):
        """Document accuracy explanation for fair score."""
        result = judge._explain_accuracy(0.78)
        assert "양호한 정확도" in result

    def test_explain_accuracy_poor(self, judge):
        """Document accuracy explanation for poor score."""
        result = judge._explain_accuracy(0.55)
        assert "낮은 정확도" in result

    def test_explain_accuracy_very_poor(self, judge):
        """Document accuracy explanation for very poor score."""
        result = judge._explain_accuracy(0.30)
        assert "매우 낮은 정확도" in result


class TestEvaluationSummary:
    """Characterization tests for EvaluationSummary."""

    def test_evaluation_summary_creation(self):
        """Document EvaluationSummary creation."""
        # Arrange & Act
        summary = EvaluationSummary(
            evaluation_id="test_id",
            timestamp="2024-01-01T00:00:00",
            total_queries=10,
            passed=8,
            failed=2,
            pass_rate=0.8,
        )

        # Assert
        assert summary.evaluation_id == "test_id"
        assert summary.total_queries == 10
        assert summary.passed == 8
        assert summary.failed == 2
        assert summary.pass_rate == 0.8

    def test_evaluation_summary_to_dict(self):
        """Document EvaluationSummary serialization."""
        # Arrange
        summary = EvaluationSummary(
            evaluation_id="test_id",
            timestamp="2024-01-01T00:00:00",
            total_queries=10,
            passed=8,
            failed=2,
            pass_rate=0.8,
        )

        # Act
        data = summary.to_dict()

        # Assert
        assert data["evaluation_id"] == "test_id"
        assert data["total_queries"] == 10
        assert data["passed"] == 8
        assert "by_persona" in data
        assert "by_category" in data


class TestEvaluationBatch:
    """Characterization tests for EvaluationBatch."""

    @pytest.fixture
    def mock_judge(self):
        """Create mock LLMJudge."""
        return MagicMock(spec=LLMJudge)

    def test_evaluation_batch_add_result(self, mock_judge):
        """Document adding results to batch."""
        # Arrange
        batch = EvaluationBatch(judge=mock_judge)
        result = JudgeResult(
            query="test",
            answer="answer",
            sources=[],
            accuracy=0.9,
            completeness=0.8,
            citations=0.7,
            context_relevance=0.85,
            overall_score=0.82,
            passed=True,
        )

        # Act
        batch.add_result(result)

        # Assert
        assert len(batch.results) == 1

    def test_evaluation_batch_get_summary_empty(self, mock_judge):
        """Document summary for empty batch."""
        # Arrange
        batch = EvaluationBatch(judge=mock_judge)

        # Act
        summary = batch.get_summary()

        # Assert
        assert summary.total_queries == 0
        assert summary.passed == 0
        assert summary.failed == 0

    def test_evaluation_batch_get_summary_with_results(self, mock_judge):
        """Document summary calculation with results."""
        # Arrange
        batch = EvaluationBatch(judge=mock_judge)

        # Add passing result
        batch.add_result(JudgeResult(
            query="q1", answer="a1", sources=[],
            accuracy=0.9, completeness=0.9, citations=0.9,
            context_relevance=0.9, overall_score=0.9, passed=True,
        ))

        # Add failing result
        batch.add_result(JudgeResult(
            query="q2", answer="a2", sources=[],
            accuracy=0.5, completeness=0.5, citations=0.5,
            context_relevance=0.5, overall_score=0.5, passed=False,
        ))

        # Act
        summary = batch.get_summary()

        # Assert
        assert summary.total_queries == 2
        assert summary.passed == 1
        assert summary.failed == 1
        assert summary.pass_rate == 0.5


class TestLLMJudgeIdentifyIssues:
    """Characterization tests for issue identification."""

    @pytest.fixture
    def judge(self):
        """Create LLMJudge with mock client."""
        mock_client = MagicMock()
        return LLMJudge(llm_client=mock_client)

    def test_identify_issues_low_accuracy(self, judge):
        """Document issue identification for low accuracy."""
        issues = judge._identify_issues(
            accuracy=0.4,
            completeness=0.9,
            citations=0.9,
            context_relevance=0.9,
            answer="test answer",
        )

        assert any("환각" in issue or "사실 오류" in issue for issue in issues)

    def test_identify_issues_low_completeness(self, judge):
        """Document issue identification for low completeness."""
        issues = judge._identify_issues(
            accuracy=0.9,
            completeness=0.5,
            citations=0.9,
            context_relevance=0.9,
            answer="test answer",
        )

        assert any("누락" in issue for issue in issues)

    def test_identify_issues_low_citations(self, judge):
        """Document issue identification for low citations."""
        issues = judge._identify_issues(
            accuracy=0.9,
            completeness=0.9,
            citations=0.5,
            context_relevance=0.9,
            answer="test answer",
        )

        assert any("인용" in issue for issue in issues)

    def test_identify_issues_low_context_relevance(self, judge):
        """Document issue identification for low context relevance."""
        issues = judge._identify_issues(
            accuracy=0.9,
            completeness=0.9,
            citations=0.9,
            context_relevance=0.5,
            answer="test answer",
        )

        assert any("관련성" in issue or "검색" in issue for issue in issues)


class TestLLMJudgeIdentifyStrengths:
    """Characterization tests for strength identification."""

    @pytest.fixture
    def judge(self):
        """Create LLMJudge with mock client."""
        mock_client = MagicMock()
        return LLMJudge(llm_client=mock_client)

    def test_identify_strengths_high_accuracy(self, judge):
        """Document strength identification for high accuracy."""
        strengths = judge._identify_strengths(
            accuracy=0.98,
            completeness=0.7,
            citations=0.7,
            context_relevance=0.7,
            answer="test answer",
        )

        assert any("정확도" in s for s in strengths)

    def test_identify_strengths_high_completeness(self, judge):
        """Document strength identification for high completeness."""
        strengths = judge._identify_strengths(
            accuracy=0.7,
            completeness=0.95,
            citations=0.7,
            context_relevance=0.7,
            answer="test answer",
        )

        assert any("포괄적" in s for s in strengths)

    def test_identify_strengths_detailed_answer(self, judge):
        """Document strength identification for detailed answer."""
        # Arrange - Answer must be > 200 characters and contain "제"
        long_answer = (
            "This is a detailed answer with 제10조 reference and more content. "
            "We need to make this answer long enough to exceed 200 characters. "
            "Adding more content to reach the threshold for detailed answer detection."
        )

        strengths = judge._identify_strengths(
            accuracy=0.7,
            completeness=0.7,
            citations=0.7,
            context_relevance=0.7,
            answer=long_answer,
        )

        assert any("상세" in s for s in strengths)
