"""
Quality Analyzer for automated feedback loop.

Analyzes evaluation results and generates improvement suggestions.

Clean Architecture: Domain layer contains analysis logic and business rules.
"""

import logging
from statistics import mean
from typing import Dict, List

from .models import EvaluationResult, ImprovementSuggestion, QualityIssue

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """
    Quality Analyzer for automated feedback loop.

    Categorizes failures and generates actionable improvement suggestions
    for optimizing RAG system components.
    """

    def __init__(self, min_failure_count: int = 5):
        """
        Initialize the quality analyzer.

        Args:
            min_failure_count: Minimum failures before generating suggestion
        """
        self.min_failure_count = min_failure_count
        logger.info(
            f"Initialized QualityAnalyzer with min_failure_count={min_failure_count}"
        )

    def analyze_failures(
        self,
        results: List[EvaluationResult],
    ) -> List[ImprovementSuggestion]:
        """
        Analyze evaluation results and generate improvement suggestions.

        Args:
            results: List of evaluation results to analyze

        Returns:
            List of improvement suggestions prioritized by impact
        """
        logger.info(f"Analyzing {len(results)} evaluation results")

        # Categorize failures by type
        failures_by_type = self._categorize_failures(results)

        # Generate suggestions for each failure type
        suggestions = []
        for issue_type, failures in failures_by_type.items():
            if len(failures) >= self.min_failure_count:
                suggestion = self._generate_suggestion(issue_type, failures)
                suggestions.append(suggestion)

        # Prioritize by affected count (most impactful first)
        suggestions.sort(key=lambda s: s.affected_count, reverse=True)

        logger.info(f"Generated {len(suggestions)} improvement suggestions")
        return suggestions

    def _categorize_failures(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Categorize failures by issue type.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary mapping issue types to failed results
        """
        categories = {
            QualityIssue.HALLUCINATION.value: [],
            QualityIssue.IRRELEVANT_RETRIEVAL.value: [],
            QualityIssue.INCOMPLETE_ANSWER.value: [],
            QualityIssue.IRRELEVANT_ANSWER.value: [],
        }

        for result in results:
            # Check faithfulness (hallucination)
            if result.faithfulness < 0.90:
                categories[QualityIssue.HALLUCINATION.value].append(result)

            # Check contextual precision (irrelevant retrieval)
            if result.contextual_precision < 0.80:
                categories[QualityIssue.IRRELEVANT_RETRIEVAL.value].append(result)

            # Check contextual recall (incomplete answer)
            if result.contextual_recall < 0.80:
                categories[QualityIssue.INCOMPLETE_ANSWER.value].append(result)

            # Check answer relevancy (irrelevant answer)
            if result.answer_relevancy < 0.85:
                categories[QualityIssue.IRRELEVANT_ANSWER.value].append(result)

        # Log failure counts
        for issue_type, failures in categories.items():
            if failures:
                logger.info(f"{issue_type}: {len(failures)} failures")

        return categories

    def _generate_suggestion(
        self,
        issue_type: str,
        failures: List[EvaluationResult],
    ) -> ImprovementSuggestion:
        """
        Generate specific improvement suggestion for failure type.

        Args:
            issue_type: Type of quality issue
            failures: List of failed evaluation results

        Returns:
            ImprovementSuggestion with actionable recommendations
        """
        avg_score = self._calculate_average_score(issue_type, failures)
        severity = self._calculate_severity(avg_score)

        if issue_type == QualityIssue.HALLUCINATION.value:
            return ImprovementSuggestion(
                issue_type=issue_type,
                component="prompt_engineering",
                recommendation=self._generate_hallucination_suggestion(avg_score),
                expected_impact="+0.15 Faithfulness score",
                implementation_effort="Low (1 hour)",
                affected_count=len(failures),
                severity=severity,
                parameters={
                    "current_avg_score": avg_score,
                    "target_score": 0.90,
                    "suggested_temperature": 0.1,
                },
            )

        elif issue_type == QualityIssue.IRRELEVANT_RETRIEVAL.value:
            optimal_threshold = self._calculate_optimal_threshold(failures)
            return ImprovementSuggestion(
                issue_type=issue_type,
                component="reranking",
                recommendation=self._generate_reranking_suggestion(optimal_threshold),
                expected_impact="+0.20 Contextual Precision score",
                implementation_effort="Medium (2 hours)",
                affected_count=len(failures),
                severity=severity,
                parameters={
                    "current_avg_score": avg_score,
                    "target_score": 0.80,
                    "suggested_threshold": optimal_threshold,
                },
            )

        elif issue_type == QualityIssue.INCOMPLETE_ANSWER.value:
            return ImprovementSuggestion(
                issue_type=issue_type,
                component="retrieval",
                recommendation=self._generate_recall_suggestion(avg_score),
                expected_impact="+0.15 Contextual Recall score",
                implementation_effort="Medium (2-3 hours)",
                affected_count=len(failures),
                severity=severity,
                parameters={
                    "current_avg_score": avg_score,
                    "target_score": 0.80,
                    "suggested_top_k": 10,
                },
            )

        elif issue_type == QualityIssue.IRRELEVANT_ANSWER.value:
            return ImprovementSuggestion(
                issue_type=issue_type,
                component="prompt_engineering",
                recommendation=self._generate_relevancy_suggestion(avg_score),
                expected_impact="+0.10 Answer Relevancy score",
                implementation_effort="Low (1 hour)",
                affected_count=len(failures),
                severity=severity,
                parameters={
                    "current_avg_score": avg_score,
                    "target_score": 0.85,
                },
            )

        else:
            # Fallback for unknown issue types
            return ImprovementSuggestion(
                issue_type=issue_type,
                component="general",
                recommendation=f"Investigate {issue_type} issues",
                expected_impact="Unknown",
                implementation_effort="Unknown",
                affected_count=len(failures),
                severity=severity,
            )

    def _calculate_average_score(
        self, issue_type: str, failures: List[EvaluationResult]
    ) -> float:
        """Calculate average score for failures of specific type."""
        if issue_type == QualityIssue.HALLUCINATION.value:
            scores = [f.faithfulness for f in failures]
        elif issue_type == QualityIssue.IRRELEVANT_RETRIEVAL.value:
            scores = [f.contextual_precision for f in failures]
        elif issue_type == QualityIssue.INCOMPLETE_ANSWER.value:
            scores = [f.contextual_recall for f in failures]
        elif issue_type == QualityIssue.IRRELEVANT_ANSWER.value:
            scores = [f.answer_relevancy for f in failures]
        else:
            return 0.0

        return round(mean(scores), 3) if scores else 0.0

    def _calculate_severity(self, avg_score: float) -> str:
        """Calculate severity level based on average score."""
        if avg_score < 0.70:
            return "high"
        elif avg_score < 0.80:
            return "medium"
        else:
            return "low"

    def _calculate_optimal_threshold(self, failures: List[EvaluationResult]) -> float:
        """
        Calculate optimal reranker threshold from failures.

        Analyzes failure patterns to suggest threshold adjustment.
        """
        # Simplified calculation - in production would use more sophisticated analysis
        # Current assumption: increase threshold by 0.05
        return round(0.30 + 0.05, 2)

    def _generate_hallucination_suggestion(self, avg_score: float) -> str:
        """Generate suggestion for hallucination issues."""
        return (
            "Reduce LLM temperature to 0.1 and add strict context adherence instruction. "
            "Prompt: 'Answer ONLY using the provided context. Do not add information not in context.'"
        )

    def _generate_reranking_suggestion(self, optimal_threshold: float) -> str:
        """Generate suggestion for irrelevant retrieval issues."""
        return f"Increase BGE reranker threshold from current to {optimal_threshold:.2f} to filter irrelevant documents more aggressively."

    def _generate_recall_suggestion(self, avg_score: float) -> str:
        """Generate suggestion for incomplete answer issues."""
        return "Increase top_k retrieval from 5 to 10 documents to capture more relevant information. Consider adding query expansion for related terms."

    def _generate_relevancy_suggestion(self, avg_score: float) -> str:
        """Generate suggestion for irrelevant answer issues."""
        return "Add query intent clarification in system prompt. Include instruction: 'First, understand what the user is asking, then provide a direct answer addressing their specific question.'"

    def generate_summary_report(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, any]:
        """
        Generate summary report from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with summary statistics
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)

        return {
            "total_queries": total,
            "passed_queries": passed,
            "pass_rate": round(passed / total, 3) if total > 0 else 0.0,
            "avg_faithfulness": round(mean(r.faithfulness for r in results), 3),
            "avg_answer_relevancy": round(mean(r.answer_relevancy for r in results), 3),
            "avg_contextual_precision": round(
                mean(r.contextual_precision for r in results), 3
            ),
            "avg_contextual_recall": round(
                mean(r.contextual_recall for r in results), 3
            ),
            "avg_overall_score": round(mean(r.overall_score for r in results), 3),
        }
