"""
LLM-as-Judge Evaluation System for RAG Quality Assessment.

Implements the 4-metric evaluation system defined in rag-quality-local skill:
1. Accuracy - Factual correctness without hallucination
2. Completeness - All key information present
3. Citations - Accurate regulation references
4. Context Relevance - Retrieved sources relevance

Phase 1 Integration: Uses improved EvaluationPrompts for better assessment.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.rag.config import get_config
from src.rag.infrastructure.llm_adapter import LLMClientAdapter

logger = logging.getLogger(__name__)

# Phase 1: Import improved evaluation prompts
try:
    from .prompts import EvaluationPrompts
    EVALUATION_PROMPTS_AVAILABLE = True
except ImportError:
    EVALUATION_PROMPTS_AVAILABLE = False
    EvaluationPrompts = None


class QualityLevel(Enum):
    """Quality level classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILING = "failing"


@dataclass
class JudgeResult:
    """Result of LLM-as-Judge evaluation."""

    query: str
    answer: str
    sources: List[Dict[str, Any]]

    # 4 core metrics
    accuracy: float
    completeness: float
    citations: float
    context_relevance: float

    # Overall score
    overall_score: float
    passed: bool

    # Detailed analysis
    reasoning: Dict[str, str] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)

    # Metadata
    evaluation_id: str = ""
    timestamp: str = ""

    def __post_init__(self):
        """Generate evaluation ID and timestamp if not provided."""
        if not self.evaluation_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.evaluation_id = f"eval_{timestamp}"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class LLMJudge:
    """LLM-as-Judge evaluator for RAG responses."""

    # Quality thresholds
    THRESHOLDS = {
        "overall": 0.80,
        "accuracy": 0.85,
        "completeness": 0.75,
        "citations": 0.70,
        "context_relevance": 0.75,
    }

    # Automatic failure indicators
    HALLUCINATION_PATTERNS = [
        r"02-\d{3,4}-\d{4}",  # Fake phone numbers
        r"서울대",  # Wrong university
        r"한국외대",  # Wrong university
        r"연세대",  # Wrong university
        r"고려대",  # Wrong university
    ]

    AVOIDANCE_PHRASES = [
        "대학마다 다릅니다",
        "확인해주세요",
        "일반적으로",
    ]

    def __init__(self, llm_client: Optional[LLMClientAdapter] = None):
        """Initialize the LLM Judge.

        Args:
            llm_client: Optional LLM client for judge evaluation
        """
        if llm_client is None:
            config = get_config()
            llm_client = LLMClientAdapter(
                provider=config.llm_provider,
                model=config.llm_model,
                base_url=config.llm_base_url,
            )

        self.llm_client = llm_client

        # Phase 1: Use improved prompts if available
        self.use_improved_prompts = EVALUATION_PROMPTS_AVAILABLE

    def evaluate_with_llm(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        expected_info: Optional[List[str]] = None,
    ) -> JudgeResult:
        """
        Evaluate a RAG response using LLM with improved prompts (Phase 1).

        This method uses the improved EvaluationPrompts for more accurate assessment.

        Args:
            query: The user's query
            answer: The RAG system's answer
            sources: Retrieved source documents
            expected_info: Optional list of expected information points

        Returns:
            JudgeResult with 4-metric scores and analysis
        """
        if not self.use_improved_prompts or not EvaluationPrompts:
            # Fallback to rule-based evaluation
            logger.warning("Improved prompts not available, using rule-based evaluation")
            return self.evaluate(query, answer, sources, expected_info)

        try:
            # Format prompt with improved EvaluationPrompts
            system_prompt, user_prompt = EvaluationPrompts.format_accuracy_prompt(
                query=query,
                answer=answer,
                context=sources,
                expected_info=expected_info
            )

            # Call LLM for evaluation
            response = self.llm_client.generate(
                system_prompt=system_prompt,
                user_message=user_prompt,
                temperature=0.0,  # Use low temperature for consistent evaluation
            )

            # Parse JSON response
            import json
            try:
                # Extract JSON from response
                json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON without code blocks
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        raise ValueError("No JSON found in LLM response")

                eval_result = json.loads(json_str)

                # Extract scores
                accuracy = eval_result.get("accuracy", 0.5)
                completeness = eval_result.get("completeness", 0.5)
                citations = eval_result.get("citations", 0.5)
                context_relevance = eval_result.get("context_relevance", 0.5)

                # Calculate weighted overall score
                weights = {
                    "accuracy": 0.35,
                    "completeness": 0.25,
                    "citations": 0.20,
                    "context_relevance": 0.20,
                }
                overall_score = (
                    accuracy * weights["accuracy"]
                    + completeness * weights["completeness"]
                    + citations * weights["citations"]
                    + context_relevance * weights["context_relevance"]
                )

                # Determine pass/fail
                passed = self._determine_pass_fail(
                    accuracy, completeness, citations, context_relevance, overall_score
                )

                # Build reasoning from LLM response
                reasoning = {
                    "accuracy": eval_result.get("accuracy_reasoning", ""),
                    "completeness": eval_result.get("completeness_reasoning", ""),
                    "citations": eval_result.get("citations_reasoning", ""),
                    "context_relevance": eval_result.get("context_relevance_reasoning", ""),
                }

                # Extract issues and strengths
                issues = eval_result.get("issues", [])
                strengths = eval_result.get("strengths", [])

                return JudgeResult(
                    query=query,
                    answer=answer,
                    sources=sources,
                    accuracy=round(accuracy, 3),
                    completeness=round(completeness, 3),
                    citations=round(citations, 3),
                    context_relevance=round(context_relevance, 3),
                    overall_score=round(overall_score, 3),
                    passed=passed,
                    reasoning=reasoning,
                    issues=issues,
                    strengths=strengths,
                )

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM evaluation response: {e}")
                # Fallback to rule-based evaluation
                return self.evaluate(query, answer, sources, expected_info)

        except Exception as e:
            logger.warning(f"LLM-based evaluation failed: {e}")
            # Fallback to rule-based evaluation
            return self.evaluate(query, answer, sources, expected_info)

    def evaluate(
        self,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        expected_info: Optional[List[str]] = None,
    ) -> JudgeResult:
        """Evaluate a RAG response using LLM-as-Judge.

        Args:
            query: The user's query
            answer: The RAG system's answer
            sources: Retrieved source documents
            expected_info: Optional list of expected information points

        Returns:
            JudgeResult with 4-metric scores and analysis
        """
        # Calculate individual metrics
        accuracy = self._evaluate_accuracy(query, answer, sources)
        completeness = self._evaluate_completeness(query, answer, expected_info)
        citations = self._evaluate_citations(answer)
        context_relevance = self._evaluate_context_relevance(sources)

        # Calculate weighted overall score
        weights = {
            "accuracy": 0.35,
            "completeness": 0.25,
            "citations": 0.20,
            "context_relevance": 0.20,
        }
        overall_score = (
            accuracy * weights["accuracy"]
            + completeness * weights["completeness"]
            + citations * weights["citations"]
            + context_relevance * weights["context_relevance"]
        )

        # Determine pass/fail
        passed = self._determine_pass_fail(
            accuracy, completeness, citations, context_relevance, overall_score
        )

        # Generate reasoning
        reasoning = {
            "accuracy": self._explain_accuracy(accuracy),
            "completeness": self._explain_completeness(completeness),
            "citations": self._explain_citations(citations),
            "context_relevance": self._explain_context_relevance(context_relevance),
        }

        # Identify issues and strengths
        issues = self._identify_issues(
            accuracy, completeness, citations, context_relevance, answer
        )
        strengths = self._identify_strengths(
            accuracy, completeness, citations, context_relevance, answer
        )

        return JudgeResult(
            query=query,
            answer=answer,
            sources=sources,
            accuracy=round(accuracy, 3),
            completeness=round(completeness, 3),
            citations=round(citations, 3),
            context_relevance=round(context_relevance, 3),
            overall_score=round(overall_score, 3),
            passed=passed,
            reasoning=reasoning,
            issues=issues,
            strengths=strengths,
        )

    def _evaluate_accuracy(
        self, query: str, answer: str, sources: List[Dict[str, Any]]
    ) -> float:
        """Evaluate factual correctness and detect hallucinations.

        Score: 0.0-1.0
        - 1.0: Completely accurate, no hallucinations
        - 0.0: Major hallucinations or completely incorrect
        """
        # Check for empty or minimal answer first
        if not answer or not answer.strip() or answer.strip() in ["...", "", " "]:
            # Empty answer is a major failure
            return 0.0

        # Check for automatic failures
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, answer):
                return 0.0  # Automatic failure

        # Check for avoidance phrases
        for phrase in self.AVOIDANCE_PHRASES:
            if phrase in answer and len(answer) < 100:
                return 0.3  # Generic avoidance without helpful info

        # Base score from context relevance (answer should be grounded in sources)
        if not sources:
            return 0.5

        # Check if answer contains information from sources
        base_score = 0.7
        top_source = sources[0] if sources else {}
        source_score = top_source.get("score", 0.0)

        if source_score > 0.8:
            base_score = 0.95
        elif source_score > 0.6:
            base_score = 0.85
        elif source_score > 0.4:
            base_score = 0.75
        else:
            base_score = 0.65

        return base_score

    def _evaluate_completeness(
        self, query: str, answer: str, expected_info: Optional[List[str]] = None
    ) -> float:
        """Evaluate if all key information is present.

        Score: 0.0-1.0
        - 1.0: All required information present
        - 0.0: Very incomplete
        """
        if not expected_info:
            # No expected info provided, estimate from answer length
            if len(answer) > 300:
                return 0.85
            elif len(answer) > 150:
                return 0.75
            elif len(answer) > 50:
                return 0.60
            else:
                return 0.30

        # Check which expected points are covered
        covered = 0
        for point in expected_info:
            if point.lower() in answer.lower():
                covered += 1

        return covered / len(expected_info) if expected_info else 0.5

    def _evaluate_citations(self, answer: str) -> float:
        """Evaluate regulation reference quality.

        Score: 0.0-1.0
        - 1.0: Perfect "규정명 + 제X조" format
        - 0.0: No citations
        """
        # Perfect: "규정명 + 제X조"
        if re.search(r"\w+규정\s*제\d+조", answer):
            return 1.0

        # Good: Has both regulation and article
        if "규정" in answer and "제" in answer and "조" in answer:
            return 0.85

        # Fair: Has regulation or article
        if "규정" in answer or ("제" in answer and "조" in answer):
            return 0.60

        # Poor: Generic mention
        if "관련" in answer and ("규정" in answer or "조" in answer):
            return 0.30

        # No citation
        return 0.0

    def _evaluate_context_relevance(self, sources: List[Dict[str, Any]]) -> float:
        """Evaluate retrieved source relevance.

        Score: 0.0-1.0
        - 1.0: All sources highly relevant
        - 0.0: No relevant sources

        Note: Returns minimum base score (0.2) for empty sources to avoid
        completely zeroing out the overall score when extraction fails.
        """
        if not sources:
            # Return minimal base score instead of 0.0 to avoid total failure
            # when source extraction fails but other metrics may be good
            return 0.2

        # Weight by position (top sources matter more)
        relevance_scores = []
        for i, source in enumerate(sources):
            score = source.get("score", 0.5)  # Default to 0.5 instead of 0.0
            if score is None or score == 0:
                score = 0.3  # Minimum score for retrieved sources
            # Position weight: top sources matter more
            position_weight = 1.0 / (1 + i * 0.1)
            relevance_scores.append(score * position_weight)

        return sum(relevance_scores) / len(relevance_scores)

    def _determine_pass_fail(
        self,
        accuracy: float,
        completeness: float,
        citations: float,
        context_relevance: float,
        overall: float,
    ) -> bool:
        """Determine if evaluation passes quality gates."""
        return (
            overall >= self.THRESHOLDS["overall"]
            and accuracy >= self.THRESHOLDS["accuracy"]
            and completeness >= self.THRESHOLDS["completeness"]
            and citations >= self.THRESHOLDS["citations"]
            and context_relevance >= self.THRESHOLDS["context_relevance"]
        )

    def _explain_accuracy(self, score: float) -> str:
        """Generate explanation for accuracy score."""
        if score >= 0.95:
            return "완벽한 정확도, 환각 없음"
        elif score >= 0.85:
            return "우수한 정확도, 사소한 부정확함 가능"
        elif score >= 0.75:
            return "양호한 정확도, 일부 부정확한 정보"
        elif score >= 0.50:
            return "낮은 정확도, 여러 오류 포함"
        else:
            return "매우 낮은 정확도, 환각 또는 심각한 오류"

    def _explain_completeness(self, score: float) -> str:
        """Generate explanation for completeness score."""
        if score >= 0.95:
            return "모든 필수 정보 포함"
        elif score >= 0.85:
            return "대부분의 필수 정보 포함, 사소한 누락"
        elif score >= 0.75:
            return "핵심 정보 포함, 일부 누락"
        elif score >= 0.50:
            return "중요한 정보 누락"
        else:
            return "매우 불완전한 답변"

    def _explain_citations(self, score: float) -> str:
        """Generate explanation for citations score."""
        if score >= 0.95:
            return "완벽한 규정 인용 형식"
        elif score >= 0.80:
            return "정확한 규정 인용"
        elif score >= 0.60:
            return "부분적인 규정 인용"
        elif score >= 0.30:
            return "일반적인 규정 언급만"
        else:
            return "규정 인용 없음"

    def _explain_context_relevance(self, score: float) -> str:
        """Generate explanation for context relevance score."""
        if score >= 0.90:
            return "매우 높은 검색 관련성"
        elif score >= 0.80:
            return "높은 검색 관련성"
        elif score >= 0.70:
            return "적절한 검색 관련성"
        elif score >= 0.50:
            return "낮은 검색 관련성"
        else:
            return "매우 낮은 검색 관련성"

    def _identify_issues(
        self,
        accuracy: float,
        completeness: float,
        citations: float,
        context_relevance: float,
        answer: str,
    ) -> List[str]:
        """Identify specific issues from evaluation."""
        issues = []

        if accuracy < 0.5:
            issues.append("환각 또는 심각한 사실 오류")
        elif accuracy < 0.85:
            issues.append("일부 부정확한 정보")

        if completeness < 0.75:
            issues.append("핵심 정보 누락")

        if citations < 0.70:
            issues.append("규정 인용 부족 또는 부정확")

        if context_relevance < 0.75:
            issues.append("검색된 문서 관련성 낮음")

        # Check for specific issues
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, answer):
                issues.append("가짜 연락처 또는 잘못된 대학명")
                break

        return issues

    def _identify_strengths(
        self,
        accuracy: float,
        completeness: float,
        citations: float,
        context_relevance: float,
        answer: str,
    ) -> List[str]:
        """Identify specific strengths from evaluation."""
        strengths = []

        if accuracy >= 0.95:
            strengths.append("완벽한 사실 정확도")
        elif accuracy >= 0.85:
            strengths.append("높은 사실 정확도")

        if completeness >= 0.90:
            strengths.append("포괄적인 정보 제공")

        if citations >= 0.90:
            strengths.append("정확한 규정 인용")
        elif citations >= 0.70:
            strengths.append("적절한 규정 인용")

        if context_relevance >= 0.85:
            strengths.append("높은 검색 품질")

        if len(answer) > 200 and "제" in answer:
            strengths.append("상세하고 구조화된 답변")

        return strengths


@dataclass
class EvaluationSummary:
    """Summary of multiple evaluations."""

    evaluation_id: str
    timestamp: str

    total_queries: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0

    # Average scores
    avg_accuracy: float = 0.0
    avg_completeness: float = 0.0
    avg_citations: float = 0.0
    avg_context_relevance: float = 0.0
    avg_overall_score: float = 0.0

    # By persona
    by_persona: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # By category
    by_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Failure patterns
    failure_patterns: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "evaluation_id": self.evaluation_id,
            "timestamp": self.timestamp,
            "total_queries": self.total_queries,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "avg_accuracy": self.avg_accuracy,
            "avg_completeness": self.avg_completeness,
            "avg_citations": self.avg_citations,
            "avg_context_relevance": self.avg_context_relevance,
            "avg_overall_score": self.avg_overall_score,
            "by_persona": self.by_persona,
            "by_category": self.by_category,
            "failure_patterns": self.failure_patterns,
        }


class EvaluationBatch:
    """Batch evaluator for multiple queries."""

    def __init__(self, judge: Optional[LLMJudge] = None):
        """Initialize batch evaluator.

        Args:
            judge: Optional LLMJudge instance
        """
        self.judge = judge or LLMJudge()
        self.results: List[JudgeResult] = []

    def add_result(self, result: JudgeResult) -> None:
        """Add an evaluation result."""
        self.results.append(result)

    def get_summary(self) -> EvaluationSummary:
        """Generate summary of all evaluations."""
        if not self.results:
            return EvaluationSummary(
                evaluation_id="",
                timestamp=datetime.now().isoformat(),
            )

        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)

        # Calculate averages
        avg_accuracy = sum(r.accuracy for r in self.results) / total
        avg_completeness = sum(r.completeness for r in self.results) / total
        avg_citations = sum(r.citations for r in self.results) / total
        avg_context_relevance = sum(r.context_relevance for r in self.results) / total
        avg_overall_score = sum(r.overall_score for r in self.results) / total

        # Count failure patterns
        failure_patterns = {}
        for result in self.results:
            for issue in result.issues:
                failure_patterns[issue] = failure_patterns.get(issue, 0) + 1

        return EvaluationSummary(
            evaluation_id=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            total_queries=total,
            passed=passed,
            failed=total - passed,
            pass_rate=passed / total,
            avg_accuracy=avg_accuracy,
            avg_completeness=avg_completeness,
            avg_citations=avg_citations,
            avg_context_relevance=avg_context_relevance,
            avg_overall_score=avg_overall_score,
            failure_patterns=failure_patterns,
        )

    def save_to_file(self, filepath: str) -> None:
        """Save evaluation results to JSON file."""
        data = {
            "summary": self.get_summary().to_dict(),
            "results": [
                {
                    "evaluation_id": r.evaluation_id,
                    "query": r.query,
                    "answer": r.answer[:200] + "..."
                    if len(r.answer) > 200
                    else r.answer,
                    "accuracy": r.accuracy,
                    "completeness": r.completeness,
                    "citations": r.citations,
                    "context_relevance": r.context_relevance,
                    "overall_score": r.overall_score,
                    "passed": r.passed,
                    "issues": r.issues,
                    "strengths": r.strengths,
                }
                for r in self.results
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
