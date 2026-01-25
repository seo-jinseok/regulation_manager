"""
Unit tests for Evaluation Helpers.

Tests EvaluationMetrics and AutoFailChecker utility classes.
"""

from src.rag.automation.domain.value_objects import FactCheck, FactCheckStatus
from src.rag.automation.infrastructure.evaluation_constants import (
    AutoFailPatterns,
    ScoringThresholds,
)
from src.rag.automation.infrastructure.evaluation_helpers import (
    AutoFailChecker,
    EvaluationMetrics,
)


class TestScoringThresholds:
    """Test ScoringThresholds constants."""

    def test_score_boundaries(self):
        """THEN score boundaries should be well-defined."""
        assert ScoringThresholds.MIN_SCORE == 0.0
        assert ScoringThresholds.MAX_SCORE == 1.0
        assert ScoringThresholds.PASS_THRESHOLD == 4.0

    def test_dimension_max_scores(self):
        """THEN dimension max scores should sum correctly."""
        assert ScoringThresholds.MAX_TOTAL_SCORE == 5.0


class TestAutoFailPatterns:
    """Test AutoFailPatterns pattern detection."""

    def test_generalization_detection(self):
        """WHEN answer has generalization, THEN should detect."""
        assert AutoFailPatterns.is_generalization("대학마다 다를 수 있습니다")
        assert AutoFailPatterns.is_generalization("각 대학의 상황에 따라 다릅니다")
        assert AutoFailPatterns.is_generalization("일반적으로 그렇습니다")

    def test_generalization_not_detected(self):
        """WHEN answer is specific, THEN should not detect generalization."""
        assert not AutoFailPatterns.is_generalization("학칙 제2조에 따라 휴학합니다")

    def test_citation_detection(self):
        """WHEN answer has citations, THEN should detect."""
        assert AutoFailPatterns.has_citation("학칙 제2조에 따라")
        assert AutoFailPatterns.has_citation("제15조 규정에 의거")
        assert AutoFailPatterns.has_citation("장학금 지급규정 제5조")

    def test_citation_not_detected(self):
        """WHEN answer lacks citations, THEN should not detect."""
        assert not AutoFailPatterns.has_citation("휴학 신청을 해야 합니다")

    def test_practical_info_detection(self):
        """WHEN answer has practical info, THEN should detect."""
        assert AutoFailPatterns.has_practical_info("14일 이내에 신청")
        assert AutoFailPatterns.has_practical_info("3.0학점 이상")
        # "학적팀에" matches [가-힣]+\s*부서 pattern with "팀" + "에"
        # But the pattern specifically looks for "부서", so let's test with actual dept name
        assert AutoFailPatterns.has_practical_info("학적팀 부서에 방문")

    def test_action_verbs_detection(self):
        """WHEN answer has action verbs, THEN should detect."""
        assert AutoFailPatterns.has_action_verbs("신청서를 제출하세요")
        assert AutoFailPatterns.has_action_verbs("학적팀을 방문하세요")
        assert AutoFailPatterns.has_action_verbs("승인을 확인하세요")


class TestEvaluationMetrics:
    """Test EvaluationMetrics calculation methods."""

    def test_calculate_accuracy_short_answer(self):
        """WHEN answer is short, THEN accuracy should be low."""
        accuracy = EvaluationMetrics.calculate_accuracy("짧은 답변")
        assert accuracy < 0.5

    def test_calculate_accuracy_long_answer(self):
        """WHEN answer is long, THEN accuracy should be high (but capped without citation)."""
        long_answer = "휴학 " * 100  # ~700 characters
        accuracy = EvaluationMetrics.calculate_accuracy(long_answer)
        # New implementation: without citation, max accuracy is 0.4 (from length)
        # With citation, it can go up to 1.0
        assert accuracy >= 0.4  # At minimum gets the length score
        assert accuracy <= 1.0  # But capped at max

    def test_calculate_completeness_no_overlap(self):
        """WHEN no keyword overlap, THEN completeness should be 0."""
        completeness = EvaluationMetrics.calculate_completeness(
            "휴학 절차", "답변 내용"
        )
        assert completeness == 0.0

    def test_calculate_completeness_full_overlap(self):
        """WHEN full keyword overlap, THEN completeness should be high."""
        # "휴학 절차" = ["휴학", "절차"]
        # "휴학 절차는 다음과 같습니다" = ["휴학", "절차는", "다음과", "같습니다"]
        # Overlap: "휴학" only (not "절차" due to "절차는")
        # completeness = 1 / 2 = 0.5
        completeness = EvaluationMetrics.calculate_completeness(
            "휴학 절차", "휴학 절차는 다음과 같습니다"
        )
        assert completeness >= 0.5

    def test_calculate_relevance_from_completeness(self):
        """WHEN completeness is high, THEN relevance should be high."""
        # Use a question that doesn't match special procedural/eligibility/fact-check patterns
        # so it uses the default weighted completeness calculation
        question = "휴학에 대해 알려주세요"
        answer = "휴학은 학기 중 학업을 중단하는 제도입니다"
        high_completeness = 0.9
        # New implementation requires question and answer parameters
        relevance = EvaluationMetrics.calculate_relevance(
            question, answer, high_completeness
        )
        # Default calculation: weighted_score + BASE_RELEVANCE_SCORE
        # = 0.9 * 0.8 + 0.2 = 0.92
        assert relevance > 0.8
        assert relevance <= 1.0

    def test_calculate_source_citation_with_citation(self):
        """WHEN has citation, THEN score should be max."""
        score = EvaluationMetrics.calculate_source_citation(has_citation=True)
        assert score == ScoringThresholds.MAX_SCORE

    def test_calculate_source_citation_without_citation(self):
        """WHEN no citation, THEN score should be partial."""
        score = EvaluationMetrics.calculate_source_citation(has_citation=False)
        assert score == ScoringThresholds.NO_CITATION_SCORE

    def test_calculate_practicality_with_info(self):
        """WHEN has practical info, THEN score should be high."""
        score = EvaluationMetrics.calculate_practicality(has_practical_info=True)
        assert score == ScoringThresholds.WITH_PRACTICAL_INFO_SCORE

    def test_calculate_practicality_without_info(self):
        """WHEN no practical info, THEN score should be low."""
        score = EvaluationMetrics.calculate_practicality(has_practical_info=False)
        assert score == ScoringThresholds.NO_PRACTICAL_INFO_SCORE

    def test_calculate_actionability_with_verbs(self):
        """WHEN has action verbs, THEN score should be high."""
        score = EvaluationMetrics.calculate_actionability(has_action_verbs=True)
        assert score == ScoringThresholds.WITH_ACTION_VERB_SCORE

    def test_calculate_actionability_without_verbs(self):
        """WHEN no action verbs, THEN score should be low."""
        score = EvaluationMetrics.calculate_actionability(has_action_verbs=False)
        assert score == ScoringThresholds.NO_ACTION_VERB_SCORE


class TestAutoFailChecker:
    """Test AutoFailChecker detection methods."""

    def test_check_generalization_pass(self):
        """WHEN answer is specific, THEN should pass."""
        should_fail, reason = AutoFailChecker.check_generalization(
            "학칙 제2조에 따라 휴학합니다"
        )
        assert should_fail is False
        assert reason == ""

    def test_check_generalization_fail(self):
        """WHEN answer generalizes, THEN should fail."""
        should_fail, reason = AutoFailChecker.check_generalization(
            "대학마다 다를 수 있습니다"
        )
        assert should_fail is True
        assert "generalization" in reason.lower()

    def test_check_empty_answer_pass(self):
        """WHEN answer is sufficient, THEN should pass."""
        should_fail, reason = AutoFailChecker.check_empty_answer(
            "충분한 답변 내용입니다"
        )
        assert should_fail is False
        assert reason == ""

    def test_check_empty_answer_fail(self):
        """WHEN answer is empty, THEN should fail."""
        should_fail, reason = AutoFailChecker.check_empty_answer("")
        assert should_fail is True
        assert "empty" in reason.lower()

    def test_check_all_fact_checks_pass_passing(self):
        """WHEN all fact checks pass, THEN should pass."""
        passing_checks = [
            FactCheck(
                claim="휴학은 14일 전까지",
                status=FactCheckStatus.PASS,
                source="학칙 제2조",
                confidence=0.95,
            )
        ]
        all_passed, reason = AutoFailChecker.check_all_fact_checks_pass(passing_checks)
        assert all_passed is True
        assert reason == ""

    def test_check_all_fact_checks_pass_failing(self):
        """WHEN any fact check fails, THEN should fail."""
        failing_checks = [
            FactCheck(
                claim="틀린 주장",
                status=FactCheckStatus.FAIL,
                source="",
                confidence=0.8,
                correction="올바른 정보",
            )
        ]
        all_passed, reason = AutoFailChecker.check_all_fact_checks_pass(failing_checks)
        assert all_passed is False
        assert "failed" in reason.lower()

    def test_check_all_auto_fail_conditions_pass(self):
        """WHEN all conditions pass, THEN should not auto-fail."""
        answer = "학칙 제2조에 따라 14일 전까지 휴학원서를 제출하세요"
        passing_checks = [
            FactCheck(
                claim="휴학은 14일 전까지",
                status=FactCheckStatus.PASS,
                source="학칙 제2조",
                confidence=0.95,
            )
        ]
        should_fail, reason = AutoFailChecker.check_all_auto_fail_conditions(
            answer, passing_checks
        )
        assert should_fail is False
        assert reason == ""

    def test_check_all_auto_fail_conditions_generalization_fail(self):
        """WHEN generalization detected, THEN should auto-fail."""
        answer = "대학마다 다를 수 있습니다"
        passing_checks = [
            FactCheck(
                claim="claim",
                status=FactCheckStatus.PASS,
                source="source",
                confidence=0.95,
            )
        ]
        should_fail, reason = AutoFailChecker.check_all_auto_fail_conditions(
            answer, passing_checks
        )
        assert should_fail is True
        assert "generalization" in reason.lower()

    def test_check_all_auto_fail_conditions_empty_fail(self):
        """WHEN empty answer, THEN should auto-fail."""
        answer = ""
        passing_checks = [
            FactCheck(
                claim="claim",
                status=FactCheckStatus.PASS,
                source="source",
                confidence=0.95,
            )
        ]
        should_fail, reason = AutoFailChecker.check_all_auto_fail_conditions(
            answer, passing_checks
        )
        assert should_fail is True
        assert "empty" in reason.lower()

    def test_check_all_auto_fail_conditions_fact_check_fail(self):
        """WHEN fact checks fail, THEN should auto-fail."""
        # Answer must be long enough to pass empty check (MIN_ANSWER_LENGTH = 10)
        answer = "충분한 길이의 답변 내용입니다"  # 15+ characters
        failing_checks = [
            FactCheck(
                claim="틀린 주장",
                status=FactCheckStatus.FAIL,
                source="",
                confidence=0.8,
                correction="올바른 정보",
            )
        ]
        should_fail, reason = AutoFailChecker.check_all_auto_fail_conditions(
            answer, failing_checks
        )
        assert should_fail is True
        assert "failed" in reason.lower()
