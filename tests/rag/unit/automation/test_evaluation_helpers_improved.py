"""
Test improved evaluation metrics for RAG quality assessment.

Tests the enhanced rule-based evaluation that considers question types,
citation density, and intent alignment.
"""

from src.rag.automation.infrastructure.evaluation_constants import AutoFailPatterns
from src.rag.automation.infrastructure.evaluation_helpers import (
    AutoFailChecker,
    EvaluationMetrics,
)


class TestImprovedAccuracy:
    """Test improved accuracy calculation with citation density."""

    def test_accuracy_with_citation(self):
        """Test accuracy calculation with proper citations."""
        answer = "학칙 제15조에 따라 휴학 신청을 해야 합니다. 신청 기간은 매학기 시작 2주 전입니다."
        score = EvaluationMetrics.calculate_accuracy(answer)
        # Should be higher due to citation
        assert score > 0.5, "Answer with citation should score > 0.5"

    def test_accuracy_with_multiple_citations(self):
        """Test accuracy calculation with multiple citations."""
        answer = (
            "학칙 제15조와 제16조에 따라 휴학 신청을 해야 합니다. "
            "시행세칙 제3조에 의거하여 기간을 준수해야 합니다."
        )
        score = EvaluationMetrics.calculate_accuracy(answer)
        # Should be even higher due to multiple citations
        assert score > 0.6, "Answer with multiple citations should score > 0.6"

    def test_accuracy_without_citation(self):
        """Test accuracy calculation without citations."""
        answer = "휴학 신청을 해야 합니다."
        score = EvaluationMetrics.calculate_accuracy(answer)
        # Should be lower without citation
        assert score < 0.5, "Answer without citation should score < 0.5"

    def test_accuracy_long_answer_without_citation(self):
        """Test that long answers without citations still get some score."""
        answer = (
            "휴학 신청을 위해서는 학부사무실에 방문하여 신청서를 작성하고 제출해야 합니다. "
            * 5
        )
        score = EvaluationMetrics.calculate_accuracy(answer)
        # Should have some score from length but capped at 0.4 without citation
        assert 0.3 <= score <= 0.5, "Long answer without citation should score 0.3-0.5"


class TestImprovedCompleteness:
    """Test improved completeness calculation with practical info."""

    def test_completeness_with_keywords_and_practical(self):
        """Test completeness with keyword coverage and practical info."""
        question = "휴학 신청 방법과 절차를 알려주세요"
        answer = "학부사무실에 방문하여 신청서를 제출해야 합니다. 기한은 학기 시작 2주 전까지입니다."
        score = EvaluationMetrics.calculate_completeness(question, answer)
        # Should have practical info (0.4) + some keyword coverage
        # Keyword overlap is low due to word suffixes ("신청서를" vs "신청")
        # So total should be around 0.4-0.6
        assert score >= 0.4, (
            f"Answer with practical info should score >= 0.4, got {score}"
        )

    def test_completeness_keywords_only(self):
        """Test completeness with only keyword coverage."""
        question = "휴학 신청 방법을 알려주세요"
        answer = "휴학 신청은 중요합니다."
        score = EvaluationMetrics.calculate_completeness(question, answer)
        # Should have some score from keywords but limited
        assert 0.2 <= score <= 0.7, "Answer with keywords only should score 0.2-0.7"

    def test_completeness_practical_info_only(self):
        """Test completeness with only practical info."""
        question = "휴학 신청 방법을 알려주세요"
        answer = "학기 시작 2주 전까지 3.5학점 이상을 신청해야 합니다."
        score = EvaluationMetrics.calculate_completeness(question, answer)
        # Should get bonus from practical info even with limited keyword overlap
        assert score > 0.3, "Answer with practical info should score > 0.3"


class TestImprovedRelevance:
    """Test improved relevance calculation with question type detection."""

    def test_relevance_procedural_with_action(self):
        """Test relevance for procedural questions with action verbs."""
        question = "휴학 어떻게 신청하나요?"
        answer_with_action = "학부사무실에 방문하여 신청서를 제출해야 합니다."
        answer_without_action = "휴학 규정이 있습니다."

        score_with = EvaluationMetrics.calculate_relevance(
            question, answer_with_action, completeness=0.7
        )
        score_without = EvaluationMetrics.calculate_relevance(
            question, answer_without_action, completeness=0.7
        )

        assert score_with > score_without, (
            "Procedural answer with action should score higher"
        )
        assert score_with > 0.8, "Good procedural answer should score > 0.8"

    def test_relevance_eligibility_with_criteria(self):
        """Test relevance for eligibility questions with criteria."""
        question = "성적 장학금 자격 기준이 무엇인가요?"
        answer_with_practical = (
            "3.5학점 이상이어야 합니다. 신청 기간은 다음 주까지입니다."
        )
        answer_without_practical = "장학금 규정이 있습니다."

        score_with = EvaluationMetrics.calculate_relevance(
            question, answer_with_practical, completeness=0.7
        )
        score_without = EvaluationMetrics.calculate_relevance(
            question, answer_without_practical, completeness=0.7
        )

        assert score_with > score_without, (
            "Eligibility answer with criteria should score higher"
        )

    def test_relevance_fact_check_with_citation(self):
        """Test relevance for fact-check questions with citation."""
        question = "휴학 기간이 얼마나 되나요?"
        answer_with_citation = "학칙 제15조에 따라 1년입니다."
        answer_without_citation = "1년입니다."

        score_with = EvaluationMetrics.calculate_relevance(
            question, answer_with_citation, completeness=0.7
        )
        score_without = EvaluationMetrics.calculate_relevance(
            question, answer_without_citation, completeness=0.7
        )

        assert score_with > score_without, (
            "Fact-check answer with citation should score higher"
        )

    def test_relevance_default_weighted(self):
        """Test relevance for general questions (default weighted completeness)."""
        question = "휴학에 대해 알려주세요"
        answer = "휴학은 학기 중 학업을 중단하는 제도입니다."

        score = EvaluationMetrics.calculate_relevance(
            question, answer, completeness=0.7
        )

        # Should use default weighted calculation
        assert 0.7 <= score <= 1.0, "Default relevance should be weighted completeness"


class TestVagueAnswerDetection:
    """Test vague answer detection patterns."""

    def test_vague_yes_only(self):
        """Test detection of yes-only answers."""
        assert AutoFailPatterns.is_vague_answer("네.")
        assert AutoFailPatterns.is_vague_answer("예.")

    def test_vague_no_only(self):
        """Test detection of no-only answers."""
        assert AutoFailPatterns.is_vague_answer("아니요.")

    def test_vague_regulation_reference(self):
        """Test detection of vague regulation references."""
        assert AutoFailPatterns.is_vague_answer("규정에 따릅니다.")
        assert AutoFailPatterns.is_vague_answer("정해진 바가 있습니다.")
        assert AutoFailPatterns.is_vague_answer("별도로 정해진 규정을 참조하세요.")

    def test_not_vague_specific_answer(self):
        """Test that specific answers are not detected as vague."""
        assert not AutoFailPatterns.is_vague_answer(
            "학칙 제15조에 따라 휴학 신청을 해야 합니다."
        )
        assert not AutoFailPatterns.is_vague_answer(
            "학부사무실에 방문하여 신청서를 제출해야 합니다."
        )


class TestGeneralizationDetection:
    """Test generalization phrase detection."""

    def test_generalization_varies_by_university(self):
        """Test detection of 'varies by university' phrases."""
        assert AutoFailPatterns.is_generalization("대학마다 다를 수 있습니다.")
        assert AutoFailPatterns.is_generalization("각 대학의 상황에 따라 다릅니다.")
        assert AutoFailPatterns.is_generalization("대학 상황에 따라 다를 수 있어요.")

    def test_generalization_generally(self):
        """Test detection of 'generally' phrases."""
        assert AutoFailPatterns.is_generalization("일반적으로 그렇습니다.")
        assert AutoFailPatterns.is_generalization("보통은 가능합니다.")

    def test_generalization_contact_department(self):
        """Test detection of 'contact department' phrases."""
        assert AutoFailPatterns.is_generalization("관련 부서에 문의하세요.")
        assert AutoFailPatterns.is_generalization("확인이 필요합니다.")
        assert AutoFailPatterns.is_generalization("해당 규정을 참고하세요.")

    def test_not_generalization_specific_answer(self):
        """Test that specific answers are not detected as generalization."""
        assert not AutoFailPatterns.is_generalization(
            "학칙 제15조에 따라 휴학 신청을 해야 합니다."
        )
        assert not AutoFailPatterns.is_generalization(
            "학부사무실에 방문하여 신청서를 제출해야 합니다."
        )


class TestAutoFailChecker:
    """Test automatic failure condition checking."""

    def test_check_generalization_fails(self):
        """Test that generalization answers fail."""
        should_fail, reason = AutoFailChecker.check_generalization(
            "대학마다 다를 수 있습니다."
        )
        assert should_fail is True
        assert "generalization" in reason.lower()

    def test_check_generalization_passes(self):
        """Test that specific answers pass."""
        should_fail, reason = AutoFailChecker.check_generalization(
            "학칙 제15조에 따라 휴학 신청을 해야 합니다."
        )
        assert should_fail is False
        assert reason == ""

    def test_check_empty_fails(self):
        """Test that empty answers fail."""
        should_fail, reason = AutoFailChecker.check_empty_answer("")
        assert should_fail is True
        assert "empty" in reason.lower() or "insufficient" in reason.lower()

        should_fail, reason = AutoFailChecker.check_empty_answer("짧음")
        assert should_fail is True

    def test_check_empty_passes(self):
        """Test that valid answers pass."""
        should_fail, reason = AutoFailChecker.check_empty_answer(
            "학칙 제15조에 따라 휴학 신청을 해야 합니다."
        )
        assert should_fail is False
        assert reason == ""

    def test_vague_answer_fails_generalization_check(self):
        """Test that vague answers are caught by generalization check."""
        # Vague answers should be detected
        should_fail, reason = AutoFailChecker.check_generalization("네.")
        assert should_fail is True


class TestCitationPatterns:
    """Test citation pattern detection."""

    def test_citation_article_format(self):
        """Test detection of article format citations."""
        assert AutoFailPatterns.has_citation("학칙 제15조에 따라야 합니다.")
        assert AutoFailPatterns.has_citation("제16조 3항에 의거합니다.")
        assert AutoFailPatterns.has_citation("15조에 명시되어 있습니다.")

    def test_citation_regulation_name(self):
        """Test detection of regulation name citations."""
        assert AutoFailPatterns.has_citation("학칙에 따라야 합니다.")
        assert AutoFailPatterns.has_citation("시행세칙을 참조하세요.")
        assert AutoFailPatterns.has_citation("장학 규정에 의거합니다.")

    def test_citation_combined_format(self):
        """Test detection of combined format citations."""
        assert AutoFailPatterns.has_citation(
            "학칙 제15조 및 시행세칙 제3조를 준수해야 합니다."
        )

    def test_no_citation(self):
        """Test that non-citations are not detected."""
        assert not AutoFailPatterns.has_citation("학부사무실에 방문하세요.")
        assert not AutoFailPatterns.has_citation("신청서를 제출해야 합니다.")


class TestPracticalInfoPatterns:
    """Test practical information pattern detection."""

    def test_practical_deadline(self):
        """Test detection of deadline information."""
        assert AutoFailPatterns.has_practical_info("2주 이내에 제출해야 합니다.")
        assert AutoFailPatterns.has_practical_info("5일 전까지 신청하세요.")
        assert AutoFailPatterns.has_practical_info("3시간 이내에 완료해야 합니다.")

    def test_practical_score_requirements(self):
        """Test detection of score/grade requirements."""
        assert AutoFailPatterns.has_practical_info("3.5학점 이상이어야 합니다.")
        assert AutoFailPatterns.has_practical_info("90점 이상 필요합니다.")
        assert AutoFailPatterns.has_practical_info("2회 이상 참석해야 합니다.")

    def test_practical_department_contact(self):
        """Test detection of department/contact information."""
        assert AutoFailPatterns.has_practical_info("학부사무실에 문의하세요.")
        assert AutoFailPatterns.has_practical_info("교학과 담당자에게 연락하세요.")
        assert AutoFailPatterns.has_practical_info("장학팀에서 확인하세요.")

    def test_no_practical_info(self):
        """Test that non-practical answers are not detected."""
        assert not AutoFailPatterns.has_practical_info("휴학은 중요한 제도입니다.")
        assert not AutoFailPatterns.has_practical_info("규정을 준수해야 합니다.")


class TestActionVerbs:
    """Test action verb detection."""

    def test_action_verbs_common(self):
        """Test detection of common action verbs."""
        assert AutoFailPatterns.has_action_verbs("신청서를 제출해야 합니다.")
        assert AutoFailPatterns.has_action_verbs("방문하여 상담을 받으세요.")
        assert AutoFailPatterns.has_action_verbs("연락하여 확인하세요.")

    def test_action_verbs_extended(self):
        """Test detection of extended action verbs."""
        assert AutoFailPatterns.has_action_verbs("등록을 완료해야 합니다.")
        assert AutoFailPatterns.has_action_verbs("발급받아 제출하세요.")
        assert AutoFailPatterns.has_action_verbs("심사를 신청하세요.")

    def test_no_action_verbs(self):
        """Test that answers without action verbs are not detected."""
        assert not AutoFailPatterns.has_action_verbs("휴학 제도가 있습니다.")
        assert not AutoFailPatterns.has_action_verbs("장학금은 중요합니다.")
