"""
Evaluation Helpers for RAG Quality Assessment.

Provides reusable metric calculation functions for quality evaluation.
Separates calculation logic from evaluation orchestration.

Enhanced version with better accuracy, completeness, and relevance calculation.
"""

import logging
from typing import Set

from .evaluation_constants import AutoFailPatterns, ScoringThresholds

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Quality metric calculation helpers.

    Provides static methods for calculating individual quality dimensions.
    All methods return scores normalized to 0.0-1.0 range unless specified.

    Improved version with better accuracy, completeness, and relevance calculation.
    """

    @staticmethod
    def calculate_accuracy(answer: str) -> float:
        """
        Calculate accuracy score based on citation density and structure.

        Longer, well-structured answers with proper regulation citations
        receive higher scores. Score is normalized to 0.0-1.0 range.

        Args:
            answer: The answer text to evaluate

        Returns:
            Accuracy score between 0.0 and 1.0
        """
        # Base score from answer length (0.0 to 0.4)
        length_score = min(
            0.4, len(answer) / (ScoringThresholds.TARGET_ANSWER_LENGTH * 2)
        )

        # Citation score (0.0 to 0.6) - heavily weighted for citations
        has_citation = AutoFailPatterns.has_citation(answer)
        if has_citation:
            # Count citation patterns for bonus
            import re

            citation_count = 0
            for pattern in AutoFailPatterns.CITATION_PATTERNS:
                citation_count += len(re.findall(pattern, answer))
            citation_score = min(0.6, 0.3 + citation_count * 0.1)
        else:
            citation_score = 0.0

        return min(ScoringThresholds.MAX_SCORE, length_score + citation_score)

    @staticmethod
    def calculate_completeness(question: str, answer: str) -> float:
        """
        Calculate completeness score based on question type and coverage.

        Measures how well the answer covers the question's key terms
        and provides practical information.

        Args:
            question: The original question
            answer: The answer text to evaluate

        Returns:
            Completeness score between 0.0 and 1.0
        """
        question_words: Set[str] = set(question.split())
        answer_words: Set[str] = set(answer.split())

        overlap = len(question_words & answer_words)
        denominator = max(len(question_words), 1)

        # Base keyword coverage (0.0 to 0.6)
        keyword_score = min(0.6, overlap / denominator)

        # Practical info bonus (0.0 to 0.4)
        has_practical = AutoFailPatterns.has_practical_info(answer)
        practical_score = 0.4 if has_practical else 0.0

        return min(ScoringThresholds.MAX_SCORE, keyword_score + practical_score)

    @staticmethod
    def calculate_relevance(question: str, answer: str, completeness: float) -> float:
        """
        Calculate relevance score based on intent alignment and question type.

        Checks if answer directly addresses the question's intent.

        Args:
            question: The original question
            answer: The answer text to evaluate
            completeness: The completeness score

        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Procedural questions need action verbs
        if any(
            word in question for word in ["방법", "신청", "절차", "어떻게", "하는 법"]
        ):
            has_action = AutoFailPatterns.has_action_verbs(answer)
            if has_action:
                return min(ScoringThresholds.MAX_SCORE, completeness + 0.2)
            else:
                return completeness * 0.7  # Penalize missing action info

        # Eligibility questions need criteria/deadlines
        elif any(word in question for word in ["자격", "조건", "기준", "누가", "대상"]):
            has_practical = AutoFailPatterns.has_practical_info(answer)
            if has_practical:
                return min(ScoringThresholds.MAX_SCORE, completeness + 0.2)
            else:
                return completeness * 0.7

        # Fact check questions need specific citations
        elif any(word in question for word in ["언제", "면체", "얼마", "기간", "횟수"]):
            has_citation = AutoFailPatterns.has_citation(answer)
            if has_citation:
                return min(ScoringThresholds.MAX_SCORE, completeness + 0.2)
            else:
                return completeness * 0.7

        # Default: weighted completeness
        weighted_score = (
            completeness * ScoringThresholds.KEYWORD_OVERLAP_RELEVANCE_WEIGHT
        )
        return min(
            ScoringThresholds.MAX_SCORE,
            weighted_score + ScoringThresholds.BASE_RELEVANCE_SCORE,
        )

    @staticmethod
    def calculate_source_citation(has_citation: bool) -> float:
        """
        Calculate source citation score.

        Answers with proper regulation references receive higher scores.

        Args:
            has_citation: Whether the answer contains citation patterns

        Returns:
            Source citation score
        """
        if has_citation:
            return ScoringThresholds.MAX_SCORE
        return ScoringThresholds.NO_CITATION_SCORE

    @staticmethod
    def calculate_practicality(has_practical_info: bool) -> float:
        """
        Calculate practicality score.

        Answers with deadlines, requirements, or contact info receive higher scores.

        Args:
            has_practical_info: Whether the answer contains practical information

        Returns:
            Practicality score (max 0.5)
        """
        if has_practical_info:
            return ScoringThresholds.WITH_PRACTICAL_INFO_SCORE
        return ScoringThresholds.NO_PRACTICAL_INFO_SCORE

    @staticmethod
    def calculate_actionability(has_action_verbs: bool) -> float:
        """
        Calculate actionability score.

        Answers with clear action verbs receive higher scores.

        Args:
            has_action_verbs: Whether the answer contains action verbs

        Returns:
            Actionability score (max 0.5)
        """
        if has_action_verbs:
            return ScoringThresholds.WITH_ACTION_VERB_SCORE
        return ScoringThresholds.NO_ACTION_VERB_SCORE


class AutoFailChecker:
    """
    Automatic failure condition detection.

    Identifies answers that should receive automatic failing scores
    due to generalization, emptiness, or other critical issues.
    """

    @staticmethod
    def check_generalization(answer: str) -> tuple[bool, str]:
        """
        Check if answer contains generalization phrases.

        Args:
            answer: The answer text to check

        Returns:
            Tuple of (is_generalization, reason)
        """
        from .evaluation_constants import AutoFailPatterns

        if AutoFailPatterns.is_generalization(answer):
            return True, "Answer contains generalization phrases"

        # Check for vague answers
        if hasattr(AutoFailPatterns, "is_vague_answer"):
            if AutoFailPatterns.is_vague_answer(answer):
                return True, "Answer is vague without specific information"

        return False, ""

    @staticmethod
    def check_empty_answer(answer: str) -> tuple[bool, str]:
        """
        Check if answer is empty or too short.

        Args:
            answer: The answer text to check

        Returns:
            Tuple of (is_empty, reason)
        """
        if not answer or len(answer.strip()) < ScoringThresholds.MIN_ANSWER_LENGTH:
            return True, "Empty or insufficient answer"

        return False, ""

    @staticmethod
    def check_all_fact_checks_pass(fact_checks) -> tuple[bool, str]:
        """
        Check if all fact checks passed.

        Args:
            fact_checks: List of FactCheck objects

        Returns:
            Tuple of (all_passed, reason)
        """
        from ..domain.value_objects import FactCheckStatus

        all_passed = all(fc.status == FactCheckStatus.PASS for fc in fact_checks)

        if not all_passed:
            return False, "Some fact checks failed"

        return True, ""

    @staticmethod
    def check_all_auto_fail_conditions(answer: str, fact_checks) -> tuple[bool, str]:
        """
        Check all automatic failure conditions.

        Args:
            answer: The answer text to check
            fact_checks: List of FactCheck objects

        Returns:
            Tuple of (should_fail, reason)
        """
        # Check generalization
        is_generalization, reason = AutoFailChecker.check_generalization(answer)
        if is_generalization:
            return True, reason

        # Check empty answer
        is_empty, reason = AutoFailChecker.check_empty_answer(answer)
        if is_empty:
            return True, reason

        # Check fact checks
        all_passed, reason = AutoFailChecker.check_all_fact_checks_pass(fact_checks)
        if not all_passed:
            return True, reason

        return False, ""
