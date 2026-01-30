"""
Quality Evaluator for RAG Testing.

Infrastructure layer for evaluating answer quality across 6 dimensions.
Scores accuracy, completeness, relevance, source citation, practicality, actionability.

Clean Architecture: Infrastructure implements domain interfaces.
"""

import json
import logging
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ...infrastructure.llm_client import ILLMClient
    from ..domain.entities import QualityTestResult
    from ..domain.value_objects import FactCheck, QualityDimensions, QualityScore

from .evaluation_constants import ScoringThresholds
from .evaluation_helpers import AutoFailChecker, EvaluationMetrics

logger = logging.getLogger(__name__)


class QualityEvaluator:
    """
    Evaluates the quality of RAG answers across multiple dimensions.

    Scoring:
    - Accuracy (1.0): Correctness of factual information
    - Completeness (1.0): Coverage of question aspects
    - Relevance (1.0): Alignment with user intent
    - Source Citation (1.0): Proper regulation references
    - Practicality (0.5): Deadlines, requirements, contact info
    - Actionability (0.5): Clear next steps for user

    Total: 5.0 points maximum
    Passing: >= 4.0 AND all fact checks pass
    """

    # LLM prompt for quality evaluation
    QUALITY_EVALUATION_PROMPT = """당신은 대학 규정 답변의 품질을 평가하는 전문가입니다.

다음 질문과 답변을 평가하세요.

질문: {question}
답변: {answer}

6가지 차원에서 평가하고 점수를 매기세요 (0.0~1.0):

1. **정확성 (Accuracy)**: 규정 내용이 정확한가?
2. **완전성 (Completeness)**: 질문의 모든 측면을 답변했는가?
3. **관련성 (Relevance)**: 질문 의도에 맞는 답변인가?
4. **출처 명시 (Source Citation)**: 규정명/조항을 명시했는가?
5. **실용성 (Practicality)**: 기한/서류/담당부서 정보가 있는가? (최대 0.5)
6. **행동 가능성 (Actionability)**: 사용자가 바로 행동 가능한가? (최대 0.5)

주의사항:
- 일반론 답변("대학마다 다를 수 있습니다")은 자동 0점 처리
- 구체적인 규정 조항이 없으면 정확성 0점
- 출처 인용이 없으면 출처 명시 0점

반드시 JSON 형식으로만 응답하세요:
{{
  "accuracy": {{"score": 0.0~1.0, "reason": "이유"}},
  "completeness": {{"score": 0.0~1.0, "reason": "이유"}},
  "relevance": {{"score": 0.0~1.0, "reason": "이유"}},
  "source_citation": {{"score": 0.0~1.0, "reason": "이유"}},
  "practicality": {{"score": 0.0~0.5, "reason": "이유"}},
  "actionability": {{"score": 0.0~0.5, "reason": "이유"}}
}}"""

    def __init__(self, llm_client: Optional["ILLMClient"] = None):
        """
        Initialize the quality evaluator.

        Args:
            llm_client: Optional LLM client for intelligent evaluation.
        """
        self.llm = llm_client

    def evaluate(
        self,
        test_result: "QualityTestResult",
        fact_checks: List["FactCheck"],
    ) -> "QualityScore":
        """
        Evaluate the quality of an answer.

        Args:
            test_result: Test result with answer to evaluate.
            fact_checks: List of fact check results.

        Returns:
            QualityScore with dimension scores and total.
        """
        # Check for automatic fail conditions
        should_fail, fail_reason = AutoFailChecker.check_all_auto_fail_conditions(
            test_result.answer, fact_checks
        )

        if should_fail:
            logger.warning(f"Auto-fail condition detected: {fail_reason}")
            return self._create_fail_quality_score(fail_reason)

        # Perform quality evaluation
        if self.llm:
            dimensions = self._evaluate_with_llm(test_result.query, test_result.answer)
        else:
            dimensions = self._evaluate_rule_based(
                test_result.query, test_result.answer, test_result.sources
            )

        # Calculate total score
        total_score = self._calculate_total_score(dimensions)

        # Determine pass/fail
        is_pass = total_score >= ScoringThresholds.PASS_THRESHOLD

        from ..domain.value_objects import QualityScore

        return QualityScore(
            dimensions=dimensions,
            total_score=round(total_score, 2),
            is_pass=is_pass,
        )

    def _calculate_total_score(self, dimensions: "QualityDimensions") -> float:
        """
        Calculate total score from all dimensions.

        Args:
            dimensions: Quality dimensions with individual scores

        Returns:
            Total score (sum of all dimensions)
        """
        return (
            dimensions.accuracy
            + dimensions.completeness
            + dimensions.relevance
            + dimensions.source_citation
            + dimensions.practicality
            + dimensions.actionability
        )

    def _create_fail_quality_score(self, reason: str) -> "QualityScore":
        """
        Create a failing quality score.

        Args:
            reason: Reason for failure (for logging)

        Returns:
            QualityScore with all zeros and failed status
        """
        from ..domain.value_objects import QualityDimensions, QualityScore

        dimensions = QualityDimensions(
            accuracy=ScoringThresholds.MIN_SCORE,
            completeness=ScoringThresholds.MIN_SCORE,
            relevance=ScoringThresholds.MIN_SCORE,
            source_citation=ScoringThresholds.MIN_SCORE,
            practicality=ScoringThresholds.MIN_SCORE,
            actionability=ScoringThresholds.MIN_SCORE,
        )

        return QualityScore(
            dimensions=dimensions,
            total_score=ScoringThresholds.MIN_TOTAL_SCORE,
            is_pass=False,
        )

    def _evaluate_with_llm(self, question: str, answer: str) -> "QualityDimensions":
        """
        Evaluate quality using LLM.

        Args:
            question: The user's question
            answer: The RAG system's answer

        Returns:
            QualityDimensions with LLM-evaluated scores
        """
        from ..domain.value_objects import QualityDimensions

        prompt = self.QUALITY_EVALUATION_PROMPT.format(question=question, answer=answer)

        try:
            response = self.llm.generate(
                system_prompt="당신은 대학 규정 답변의 품질을 평가하는 전문가입니다.",
                user_message=prompt,
                temperature=0.0,
            )

            # Parse JSON response
            cleaned = self._extract_json_from_response(response)
            data = json.loads(cleaned.strip())

            return QualityDimensions(
                accuracy=float(data["accuracy"]["score"]),
                completeness=float(data["completeness"]["score"]),
                relevance=float(data["relevance"]["score"]),
                source_citation=float(data["source_citation"]["score"]),
                practicality=float(data["practicality"]["score"]),
                actionability=float(data["actionability"]["score"]),
            )

        except Exception as e:
            logger.warning(f"Failed to parse LLM quality evaluation: {e}")
            # Fallback to rule-based
            return self._evaluate_rule_based(question, answer, [])

    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from LLM response (handles markdown code blocks).

        Args:
            response: Raw LLM response

        Returns:
            Cleaned JSON string
        """
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned

    def _evaluate_rule_based(
        self, question: str, answer: str, sources: List[str]
    ) -> "QualityDimensions":
        """
        Evaluate quality using enhanced rule-based approach.

        Uses improved metrics that consider question types and citation density.

        Args:
            question: The user's question
            answer: The RAG system's answer
            sources: List of source documents used

        Returns:
            QualityDimensions with improved rule-based scores
        """
        from ..domain.value_objects import QualityDimensions
        from .evaluation_constants import AutoFailPatterns

        # Accuracy: Based on citation density and structure (improved)
        accuracy = EvaluationMetrics.calculate_accuracy(answer)

        # Completeness: Based on question coverage and practical info (improved)
        completeness = EvaluationMetrics.calculate_completeness(question, answer)

        # Relevance: Based on question type and intent alignment (improved)
        relevance = EvaluationMetrics.calculate_relevance(
            question, answer, completeness
        )

        # Source Citation: Check for citation patterns
        has_citation = AutoFailPatterns.has_citation(answer)
        source_citation = EvaluationMetrics.calculate_source_citation(has_citation)

        # Practicality: Check for practical info
        has_practical_info = AutoFailPatterns.has_practical_info(answer)
        practicality = EvaluationMetrics.calculate_practicality(has_practical_info)

        # Actionability: Check for action verbs
        has_action_verbs = AutoFailPatterns.has_action_verbs(answer)
        actionability = EvaluationMetrics.calculate_actionability(has_action_verbs)

        return QualityDimensions(
            accuracy=round(accuracy, 2),
            completeness=round(completeness, 2),
            relevance=round(relevance, 2),
            source_citation=round(source_citation, 2),
            practicality=round(practicality, 2),
            actionability=round(actionability, 2),
        )
