"""
Evaluation Helpers for RAG Quality Assessment.

Provides reusable metric calculation functions for quality evaluation.
Separates calculation logic from evaluation orchestration.

Enhanced version with better accuracy, completeness, and relevance calculation.
Includes Hallucination Prevention for phone numbers and other university references.
"""

import logging
import re
from typing import List, Set, Tuple

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


class HallucinationDetector:
    """
    Detects and prevents hallucinations in RAG answers.

    Features:
    - Detects phone numbers (XX-XXXX-XXXX, 02-XXX-XXXX, etc.)
    - Detects other university names (서울대, 연세대, 고려대, etc.)
    - Detects evasive responses ("대학마다 다릅니다", "일반적으로")
    - Sanitizes answers by blocking or filtering detected patterns
    """

    # Phone number patterns (Korean format)
    PHONE_PATTERNS = [
        re.compile(r"\d{2,3}-\d{3,4}-\d{4}"),  # 02-1234-5678, 010-123-4567
        re.compile(r"\d{10,11}"),  # 0212345678, 01012345678
        re.compile(r"\(\d{2,3}\)\s*\d{3,4}-\d{4}"),  # (02) 1234-5678
    ]

    # Other university names to detect
    OTHER_UNIVERSITIES = [
        "서울대학교",
        "서울대",
        "연세대학교",
        "연세대",
        "고려대학교",
        "고려대",
        "카이스트",
        "KAIST",
        "포항공대",
        "POSTECH",
        "서강대학교",
        "서강대",
        "이화여대",
        "한국외대",
        "홍익대학교",
        "홍익대",
        "성균관대학교",
        "성균관대",
        "건국대학교",
        "건국대",
        "동국대학교",
        "동국대",
    ]

    # Evasive response patterns (also check for generalization)
    EVASIVE_PATTERNS = [
        re.compile(r"대학마다\s*다를\s*수"),
        re.compile(r"각\s*대학의\s*상황에\s*따라"),
        re.compile(r"일반적으로"),
        re.compile(r"보통은"),
        re.compile(r"대체로"),
        re.compile(r"기관에\s*따라\s*다르"),
        re.compile(r"상황에\s*따라\s*다르"),
    ]

    @staticmethod
    def detect_phone_numbers(answer: str) -> List[str]:
        """
        Detect phone numbers in the answer.

        Args:
            answer: The answer text to check

        Returns:
            List of detected phone numbers
        """
        detected = []
        for pattern in HallucinationDetector.PHONE_PATTERNS:
            matches = pattern.findall(answer)
            detected.extend(matches)
        return detected

    @staticmethod
    def detect_other_universities(answer: str) -> List[str]:
        """
        Detect references to other universities in the answer.

        Args:
            answer: The answer text to check

        Returns:
            List of detected university references
        """
        detected = []
        for university in HallucinationDetector.OTHER_UNIVERSITIES:
            if university in answer:
                detected.append(university)
        return detected

    @staticmethod
    def detect_evasive_responses(answer: str) -> List[str]:
        """
        Detect evasive responses in the answer.

        Args:
            answer: The answer text to check

        Returns:
            List of detected evasive phrases
        """
        detected = []
        for pattern in HallucinationDetector.EVASIVE_PATTERNS:
            matches = pattern.findall(answer)
            detected.extend(matches)
        return detected

    @staticmethod
    def has_hallucination(answer: str) -> Tuple[bool, List[str]]:
        """
        Check if the answer contains any hallucination patterns.

        Args:
            answer: The answer text to check

        Returns:
            Tuple of (has_hallucination, list_of_issues)
        """
        issues = []

        # Check for phone numbers
        phone_numbers = HallucinationDetector.detect_phone_numbers(answer)
        if phone_numbers:
            issues.append(f"전화번호 포함: {', '.join(phone_numbers)}")

        # Check for other universities
        universities = HallucinationDetector.detect_other_universities(answer)
        if universities:
            issues.append(f"다른 대학교 언급: {', '.join(universities)}")

        # Check for evasive responses
        evasive = HallucinationDetector.detect_evasive_responses(answer)
        if evasive:
            issues.append(f"회피성 답변: {', '.join(evasive)}")

        return (len(issues) > 0, issues)

    @staticmethod
    def sanitize_answer(answer: str) -> Tuple[str, List[str]]:
        """
        Sanitize the answer by removing or blocking detected hallucinations.

        Args:
            answer: The answer text to sanitize

        Returns:
            Tuple of (sanitized_answer, list_of_changes)
        """
        sanitized = answer
        changes = []

        # Remove phone numbers
        for pattern in HallucinationDetector.PHONE_PATTERNS:
            matches = list(pattern.finditer(sanitized))
            for match in reversed(matches):  # Reverse to maintain positions
                phone = match.group(0)
                replacement = "[연락처는 학교 홈페이지를 확인하세요]"
                sanitized = (
                    sanitized[: match.start()] + replacement + sanitized[match.end() :]
                )
                changes.append(f"전화번호 '{phone}' 제거됨")

        # Replace other university references
        for university in HallucinationDetector.OTHER_UNIVERSITIES:
            if university in sanitized:
                sanitized = sanitized.replace(university, "[다른 대학교]")
                changes.append(f"대학교명 '{university}' 수정됨")

        # Replace evasive responses with clearer message
        evasive_found = HallucinationDetector.detect_evasive_responses(sanitized)
        for evasive in evasive_found:
            sanitized = sanitized.replace(
                evasive,
                "[정확한 정보는 동의대학교 규정을 확인하세요]",
            )
            changes.append(f"회피성 답변 '{evasive}' 수정됨")

        return sanitized, changes

    @staticmethod
    def block_if_hallucination(answer: str) -> Tuple[bool, str, List[str]]:
        """
        Block the answer if hallucination is detected.

        Args:
            answer: The answer text to check

        Returns:
            Tuple of (should_block, blocking_reason, list_of_issues)
        """
        has_hallucination, issues = HallucinationDetector.has_hallucination(answer)

        if has_hallucination:
            reason = "답변에 신뢰할 수 없는 정보가 포함되어 있습니다. " + "; ".join(
                issues
            )
            return True, reason, issues

        return False, "", issues
