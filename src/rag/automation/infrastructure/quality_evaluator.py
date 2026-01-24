"""
Quality Evaluator for RAG Testing.

Infrastructure layer for evaluating answer quality across 6 dimensions.
Scores accuracy, completeness, relevance, source citation, practicality, actionability.

Clean Architecture: Infrastructure implements domain interfaces.
"""

import json
import logging
import re
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ...infrastructure.llm_client import ILLMClient
    from ..domain.entities import TestResult
    from ..domain.value_objects import FactCheck, QualityDimensions, QualityScore

logger = logging.getLogger(__name__)


# Patterns for detecting quality issues
GENERALIZATION_PATTERNS = [
    r"대학마다\s*다를\s*수",
    r"각\s*대학의\s*상황에\s*따라",
    r"일반적으로",
    r"보통은",
    r"대체로",
]

# Patterns for source citations
CITATION_PATTERNS = [
    r"제\d+[조항]",
    r"\d+[조항]\s*.*규정",
    r"[가-힣]+규정",
    r"[가-힣]+학칙",
]

# Patterns for practical information
PRACTICAL_INFO_PATTERNS = [
    r"\d+[년월일시점분]\s*이내",
    r"\d+회\s*이상",
    r"\d+[학점점]",
    r"\d+\.\d+\s*이상",
    r"[가-힣]+\s*부서",
    r"[가-힣]+\s*담당자",
]


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
        test_result: "TestResult",
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
        if self._has_generalization(test_result.answer):
            logger.warning("Generalization detected - automatic fail")

            return self._create_fail_quality_score(
                "Answer contains generalization phrases"
            )

        if self._has_empty_answer(test_result):
            return self._create_fail_quality_score("Empty answer")

        if not self._all_fact_checks_pass(fact_checks):
            return self._create_fail_quality_score("Some fact checks failed")

        # Perform quality evaluation
        if self.llm:
            dimensions = self._evaluate_with_llm(test_result.query, test_result.answer)
        else:
            dimensions = self._evaluate_rule_based(
                test_result.query, test_result.answer, test_result.sources
            )

        # Calculate total score
        total_score = (
            dimensions.accuracy
            + dimensions.completeness
            + dimensions.relevance
            + dimensions.source_citation
            + dimensions.practicality
            + dimensions.actionability
        )

        # Determine pass/fail
        is_pass = total_score >= 4.0

        from ..domain.value_objects import QualityScore

        return QualityScore(
            dimensions=dimensions,
            total_score=round(total_score, 2),
            is_pass=is_pass,
        )

    def _has_generalization(self, answer: str) -> bool:
        """Check if answer contains generalization phrases."""
        for pattern in GENERALIZATION_PATTERNS:
            if re.search(pattern, answer):
                return True
        return False

    def _has_empty_answer(self, test_result: "TestResult") -> bool:
        """Check if answer is empty."""
        return not test_result.answer or len(test_result.answer.strip()) < 10

    def _all_fact_checks_pass(self, fact_checks: List["FactCheck"]) -> bool:
        """Check if all fact checks passed."""
        from ..domain.value_objects import FactCheckStatus

        return all(fc.status == FactCheckStatus.PASS for fc in fact_checks)

    def _create_fail_quality_score(self, reason: str) -> "QualityScore":
        """Create a failing quality score."""
        from ..domain.value_objects import QualityDimensions, QualityScore

        dimensions = QualityDimensions(
            accuracy=0.0,
            completeness=0.0,
            relevance=0.0,
            source_citation=0.0,
            practicality=0.0,
            actionability=0.0,
        )

        return QualityScore(
            dimensions=dimensions,
            total_score=0.0,
            is_pass=False,
        )

    def _evaluate_with_llm(self, question: str, answer: str) -> "QualityDimensions":
        """Evaluate quality using LLM."""
        from ..domain.value_objects import QualityDimensions

        prompt = self.QUALITY_EVALUATION_PROMPT.format(question=question, answer=answer)

        try:
            response = self.llm.generate(
                system_prompt="당신은 대학 규정 답변의 품질을 평가하는 전문가입니다.",
                user_message=prompt,
                temperature=0.0,
            )

            # Parse JSON response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

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

    def _evaluate_rule_based(
        self, question: str, answer: str, sources: List[str]
    ) -> "QualityDimensions":
        """Evaluate quality using rule-based approach."""
        from ..domain.value_objects import QualityDimensions

        # Accuracy: Based on answer length and structure
        accuracy = min(1.0, len(answer) / 200)

        # Completeness: Based on question coverage
        question_words = set(question.split())
        answer_words = set(answer.split())
        overlap = len(question_words & answer_words)
        completeness = min(1.0, overlap / max(len(question_words), 1))

        # Relevance: Based on keyword overlap
        relevance = min(1.0, completeness * 0.8 + 0.2)

        # Source Citation: Check for citation patterns
        has_citation = any(re.search(p, answer) for p in CITATION_PATTERNS)
        source_citation = 1.0 if has_citation else 0.3

        # Practicality: Check for practical info
        has_practical = any(re.search(p, answer) for p in PRACTICAL_INFO_PATTERNS)
        practicality = 0.5 if has_practical else 0.2

        # Actionability: Based on verbs and structure
        action_verbs = ["신청", "제출", "방문", "연락", "확인", "준비"]
        has_action = any(verb in answer for verb in action_verbs)
        actionability = 0.5 if has_action else 0.2

        return QualityDimensions(
            accuracy=round(accuracy, 2),
            completeness=round(completeness, 2),
            relevance=round(relevance, 2),
            source_citation=round(source_citation, 2),
            practicality=round(practicality, 2),
            actionability=round(actionability, 2),
        )
