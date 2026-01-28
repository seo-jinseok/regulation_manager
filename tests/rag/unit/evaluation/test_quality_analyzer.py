"""
Unit tests for Quality Analyzer domain component.

Tests automated feedback loop and improvement suggestion generation.
"""

import pytest

from src.rag.domain.evaluation.models import (
    EvaluationResult,
    ImprovementSuggestion,
    QualityIssue,
)
from src.rag.domain.evaluation.quality_analyzer import QualityAnalyzer


class TestQualityAnalyzer:
    """Test Quality Analyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create quality analyzer."""
        return QualityAnalyzer(min_failure_count=5)

    @pytest.fixture
    def passing_results(self):
        """Create results that pass all thresholds."""
        return [
            EvaluationResult(
                query=f"Query {i}",
                answer=f"Answer {i}",
                contexts=[f"Context {i}"],
                faithfulness=0.92,
                answer_relevancy=0.88,
                contextual_precision=0.85,
                contextual_recall=0.87,
                overall_score=0.88,
                passed=True,
            )
            for i in range(10)
        ]

    @pytest.fixture
    def hallucination_failures(self):
        """Create results with hallucination failures."""
        return [
            EvaluationResult(
                query=f"Query {i}",
                answer=f"Hallucinated answer {i}",
                contexts=[],
                faithfulness=0.85,  # Below 0.90 threshold
                answer_relevancy=0.90,
                contextual_precision=0.85,
                contextual_recall=0.85,
                overall_score=0.86,
                passed=False,
                failure_reasons=["Faithfulness below threshold"],
            )
            for i in range(10)
        ]

    @pytest.fixture
    def retrieval_failures(self):
        """Create results with retrieval failures."""
        return [
            EvaluationResult(
                query=f"Query {i}",
                answer=f"Answer {i}",
                contexts=["Irrelevant context"],
                faithfulness=0.92,
                answer_relevancy=0.88,
                contextual_precision=0.75,  # Below 0.80 threshold
                contextual_recall=0.82,
                overall_score=0.84,
                passed=False,
                failure_reasons=["Contextual Precision below threshold"],
            )
            for i in range(10)
        ]

    def test_analyzer_initialization(self):
        """WHEN analyzer created, THEN should have correct min_failure_count."""
        analyzer = QualityAnalyzer(min_failure_count=3)
        assert analyzer.min_failure_count == 3

    def test_analyze_passing_results_no_suggestions(self, analyzer, passing_results):
        """WHEN all results pass, THEN should not generate suggestions."""
        suggestions = analyzer.analyze_failures(passing_results)

        assert len(suggestions) == 0

    def test_analyze_hallucination_failures(self, analyzer, hallucination_failures):
        """WHEN hallucination failures detected, THEN should generate suggestion."""
        suggestions = analyzer.analyze_failures(hallucination_failures)

        assert len(suggestions) == 1

        suggestion = suggestions[0]
        assert suggestion.issue_type == QualityIssue.HALLUCINATION.value
        assert suggestion.component == "prompt_engineering"
        assert "temperature" in suggestion.recommendation.lower()
        assert suggestion.affected_count == 10
        assert suggestion.implementation_effort == "Low (1 hour)"

    def test_analyze_retrieval_failures(self, analyzer, retrieval_failures):
        """WHEN retrieval failures detected, THEN should generate suggestion."""
        suggestions = analyzer.analyze_failures(retrieval_failures)

        assert len(suggestions) == 1

        suggestion = suggestions[0]
        assert suggestion.issue_type == QualityIssue.IRRELEVANT_RETRIEVAL.value
        assert suggestion.component == "reranking"
        assert "threshold" in suggestion.recommendation.lower()
        assert suggestion.affected_count == 10

    def test_suggestions_prioritized_by_impact(self, analyzer):
        """WHEN multiple failure types, THEN suggestions sorted by impact."""
        mixed_failures = []

        # Add 15 hallucination failures (highest impact)
        mixed_failures.extend(
            [
                EvaluationResult(
                    query="Query",
                    answer="Answer",
                    contexts=[],
                    faithfulness=0.85,
                    answer_relevancy=0.90,
                    contextual_precision=0.85,
                    contextual_recall=0.85,
                    overall_score=0.0,
                    passed=False,
                )
                for _ in range(15)
            ]
        )

        # Add 8 retrieval failures
        mixed_failures.extend(
            [
                EvaluationResult(
                    query="Query",
                    answer="Answer",
                    contexts=["Context"],
                    faithfulness=0.92,
                    answer_relevancy=0.88,
                    contextual_precision=0.75,
                    contextual_recall=0.82,
                    overall_score=0.0,
                    passed=False,
                )
                for _ in range(8)
            ]
        )

        suggestions = analyzer.analyze_failures(mixed_failures)

        # Hallucination should be first (15 failures > 8 failures)
        assert len(suggestions) == 2
        assert suggestions[0].affected_count == 15
        assert suggestions[0].issue_type == QualityIssue.HALLUCINATION.value
        assert suggestions[1].affected_count == 8

    def test_insufficient_failures_no_suggestions(self, analyzer):
        """WHEN failures below threshold, THEN should not generate suggestions."""
        # Only 3 failures (below min_failure_count of 5)
        few_failures = [
            EvaluationResult(
                query="Query",
                answer="Answer",
                contexts=[],
                faithfulness=0.85,
                answer_relevancy=0.90,
                contextual_precision=0.85,
                contextual_recall=0.85,
                overall_score=0.0,
                passed=False,
            )
            for _ in range(3)
        ]

        suggestions = analyzer.analyze_failures(few_failures)

        assert len(suggestions) == 0

    def test_calculate_severity_high(self, analyzer):
        """WHEN average score very low, THEN severity should be high."""
        severity = analyzer._calculate_severity(0.65)
        assert severity == "high"

    def test_calculate_severity_medium(self, analyzer):
        """WHEN average score medium-low, THEN severity should be medium."""
        severity = analyzer._calculate_severity(0.75)
        assert severity == "medium"

    def test_calculate_severity_low(self, analyzer):
        """WHEN average score near threshold, THEN severity should be low."""
        severity = analyzer._calculate_severity(0.82)
        assert severity == "low"

    def test_generate_summary_report(self, analyzer, passing_results):
        """WHEN generating summary, THEN should include all statistics."""
        summary = analyzer.generate_summary_report(passing_results)

        assert summary["total_queries"] == 10
        assert summary["passed_queries"] == 10
        assert summary["pass_rate"] == 1.0
        assert "avg_faithfulness" in summary
        assert "avg_answer_relevancy" in summary
        assert "avg_contextual_precision" in summary
        assert "avg_contextual_recall" in summary
        assert "avg_overall_score" in summary

    def test_generate_summary_with_failures(self, analyzer):
        """WHEN some results fail, THEN summary should reflect correctly."""
        mixed_results = []

        # 7 passing
        mixed_results.extend(
            [
                EvaluationResult(
                    query="Query",
                    answer="Answer",
                    contexts=["Context"],
                    faithfulness=0.92,
                    answer_relevancy=0.88,
                    contextual_precision=0.85,
                    contextual_recall=0.87,
                    overall_score=0.88,
                    passed=True,
                )
                for _ in range(7)
            ]
        )

        # 3 failing
        mixed_results.extend(
            [
                EvaluationResult(
                    query="Query",
                    answer="Answer",
                    contexts=[],
                    faithfulness=0.85,
                    answer_relevancy=0.90,
                    contextual_precision=0.85,
                    contextual_recall=0.85,
                    overall_score=0.86,
                    passed=False,
                )
                for _ in range(3)
            ]
        )

        summary = analyzer.generate_summary_report(mixed_results)

        assert summary["total_queries"] == 10
        assert summary["passed_queries"] == 7
        assert summary["pass_rate"] == 0.7


class TestImprovementSuggestion:
    """Test ImprovementSuggestion model."""

    def test_suggestion_to_dict(self):
        """WHEN converting to dict, THEN should include all fields."""
        suggestion = ImprovementSuggestion(
            issue_type="hallucination",
            component="prompt_engineering",
            recommendation="Reduce temperature to 0.1",
            expected_impact="+0.15 Faithfulness",
            implementation_effort="Low (1 hour)",
            affected_count=10,
            severity="high",
            parameters={"temperature": 0.1},
        )

        suggestion_dict = suggestion.to_dict()

        assert suggestion_dict["issue_type"] == "hallucination"
        assert suggestion_dict["component"] == "prompt_engineering"
        assert suggestion_dict["affected_count"] == 10
        assert suggestion_dict["severity"] == "high"
        assert "parameters" in suggestion_dict
