"""
Focused coverage tests for component_analyzer.py.

Extends existing test_component_analyzer.py to reach 85% coverage.
Targets uncovered branches in _evaluate_component_contribution,
_evaluate_missing_component, _determine_overall_impact, and
_check_timing_importance.
"""

from src.rag.automation.domain.entities import TestResult
from src.rag.automation.domain.value_objects import (
    ComponentAnalysis,
    ComponentContribution,
    FactCheck,
    FactCheckStatus,
    QualityDimensions,
    QualityScore,
    RAGComponent,
)
from src.rag.automation.infrastructure.component_analyzer import ComponentAnalyzer


class TestComponentContributionEvaluationMissingBranches:
    """Tests for uncovered branches in _evaluate_component_contribution."""

    def test_self_rag_with_low_accuracy(self):
        """
        SPEC: Self-RAG should score 0 when accuracy is not high.
        """
        analyzer = ComponentAnalyzer()

        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.5,  # Low accuracy
                completeness=0.5,
                relevance=0.5,
                source_citation=0.5,
                practicality=0.2,
                actionability=0.3,
            ),
            total_score=2.5,
            is_pass=False,
        )

        test_result = TestResult(
            test_case_id="test_001",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={"self_rag": True},
            quality_score=quality_score,
            passed=False,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.SELF_RAG, test_result
        )

        assert score == 0
        assert "unclear impact" in reason.lower()

    def test_hyde_with_few_sources(self):
        """
        SPEC: HyDE should score 0 when source diversity is limited.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_002",
            query="query",
            answer="answer",
            sources=["only one source"],  # Less than 3 sources
            confidence=0.7,
            execution_time_ms=1000,
            rag_pipeline_log={"hyde": True},
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.HYDE, test_result
        )

        assert score == 0
        assert "limited" in reason.lower()

    def test_corrective_rag_without_filtering(self):
        """
        SPEC: Corrective RAG should score 0 when no filtering occurred.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_003",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.6,
            execution_time_ms=1000,
            rag_pipeline_log={"corrective": True},  # No "filtered" or "is_relevant"
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.CORRECTIVE_RAG, test_result
        )

        assert score == 0
        assert "without" in reason.lower()

    def test_hybrid_search_with_no_sources(self):
        """
        SPEC: Hybrid search should score -1 when no sources retrieved.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_004",
            query="query",
            answer="answer",
            sources=[],  # No sources
            confidence=0.3,
            execution_time_ms=1000,
            rag_pipeline_log={"hybrid_search": True},
            passed=False,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.HYBRID_SEARCH, test_result
        )

        assert score == -1
        assert "failed" in reason.lower()

    def test_hybrid_search_with_adequate_coverage(self):
        """
        SPEC: Hybrid search should score 1 with adequate coverage (1-3 sources).
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_005",
            query="query",
            answer="answer",
            sources=["source1", "source2"],  # 2 sources
            confidence=0.7,
            execution_time_ms=1000,
            rag_pipeline_log={"hybrid_search": True},
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.HYBRID_SEARCH, test_result
        )

        assert score == 1
        assert "adequate" in reason.lower()

    def test_bge_reranker_with_low_confidence(self):
        """
        SPEC: BGE Reranker should score 0 when confidence is low.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_006",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.5,  # Below 0.7
            execution_time_ms=1000,
            rag_pipeline_log={"reranker": "bge"},
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.BGE_RERANKER, test_result
        )

        assert score == 0
        assert "low" in reason.lower()

    def test_query_analyzer_with_low_relevance(self):
        """
        SPEC: Query analyzer should score 0 when relevance is low.
        """
        analyzer = ComponentAnalyzer()

        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.5,
                completeness=0.5,
                relevance=0.5,  # Below 0.7
                source_citation=0.5,
                practicality=0.2,
                actionability=0.3,
            ),
            total_score=2.5,
            is_pass=False,
        )

        test_result = TestResult(
            test_case_id="test_007",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={"query_analyzer": True},
            quality_score=quality_score,
            passed=False,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.QUERY_ANALYZER, test_result
        )

        assert score == 0
        assert "low" in reason.lower()

    def test_dynamic_query_expansion_without_benefit(self):
        """
        SPEC: Query expansion should score 0 when no clear benefit.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_008",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.6,
            execution_time_ms=1000,
            # No "expanded_query" in log
            rag_pipeline_log={"query_expansion": True},
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.DYNAMIC_QUERY_EXPANSION, test_result
        )

        assert score == 0
        assert "without" in reason.lower() or "no clear" in reason.lower()

    def test_fact_check_without_checks(self):
        """
        SPEC: Fact check should score 0 when no checks recorded.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_009",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.7,
            execution_time_ms=1000,
            rag_pipeline_log={"fact_check": True},
            fact_checks=None,  # No checks
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.FACT_CHECK, test_result
        )

        assert score == 0
        assert "no checks" in reason.lower()

    def test_fact_check_partial_pass(self):
        """
        SPEC: Fact check should score 1 when most checks pass.
        """
        analyzer = ComponentAnalyzer()

        fact_checks = [
            FactCheck(
                claim="Claim 1",
                status=FactCheckStatus.PASS,
                source="source",
                confidence=0.9,
            ),
            FactCheck(
                claim="Claim 2",
                status=FactCheckStatus.PASS,
                source="source",
                confidence=0.8,
            ),
            FactCheck(
                claim="Claim 3",
                status=FactCheckStatus.FAIL,
                source="source",
                confidence=0.5,
                correction="Wrong",
            ),
        ]

        test_result = TestResult(
            test_case_id="test_010",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.7,
            execution_time_ms=1000,
            rag_pipeline_log={"fact_check": True},
            fact_checks=fact_checks,
            passed=False,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.FACT_CHECK, test_result
        )

        assert score == 1  # 2/3 passed > 1.5 (half)
        assert "passed" in reason.lower()


class TestEvaluateMissingComponentMissingBranches:
    """Tests for uncovered branches in _evaluate_missing_component."""

    def test_missing_hybrid_search_with_sources(self):
        """
        SPEC: Missing hybrid search should score -1 if sources found via fallback.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_011",
            query="query",
            answer="answer",
            sources=["source1", "source2"],  # Sources found
            confidence=0.7,
            execution_time_ms=1000,
            rag_pipeline_log={},  # No hybrid_search
            passed=True,
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.HYBRID_SEARCH, test_result
        )

        assert score == -1
        assert "fallback" in reason.lower()

    def test_missing_fact_check_with_failed_test(self):
        """
        SPEC: Missing fact check should score -1 when test fails.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_012",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.3,
            execution_time_ms=1000,
            rag_pipeline_log={},  # No fact_check
            passed=False,  # Test failed
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.FACT_CHECK, test_result
        )

        assert score == -1
        assert "missing" in reason.lower()

    def test_missing_optional_component_self_rag(self):
        """
        SPEC: Missing Self-RAG should score 0 (optional component).
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_013",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.7,
            execution_time_ms=1000,
            rag_pipeline_log={},  # No self_rag
            passed=True,
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.SELF_RAG, test_result
        )

        assert score == 0
        assert "optional" in reason.lower()

    def test_missing_optional_component_hyde(self):
        """
        SPEC: Missing HyDE should score 0 (optional component).
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_014",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.7,
            execution_time_ms=1000,
            rag_pipeline_log={},  # No hyde
            passed=True,
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.HYDE, test_result
        )

        assert score == 0
        assert "optional" in reason.lower()

    def test_missing_optional_component_bge_reranker(self):
        """
        SPEC: Missing BGE Reranker should score 0 (optional component).
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_015",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.7,
            execution_time_ms=1000,
            rag_pipeline_log={},  # No bge_reranker
            passed=True,
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.BGE_RERANKER, test_result
        )

        assert score == 0
        assert "optional" in reason.lower()

    def test_missing_non_critical_component(self):
        """
        SPEC: Missing non-critical components should score 0.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_016",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.7,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=True,
        )

        # Test QUERY_ANALYZER (not in critical or optional lists)
        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.QUERY_ANALYZER, test_result
        )

        assert score == 0
        assert "not executed" in reason.lower()


class TestDetermineOverallImpactMissingBranches:
    """Tests for uncovered branches in _determine_overall_impact."""

    def test_strong_positive_impact(self):
        """
        SPEC: Should return strong positive impact when total score >= 4.
        """
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=2,
                reason="",
                was_executed=True,
            ),
            ComponentContribution(
                component=RAGComponent.FACT_CHECK, score=2, reason="", was_executed=True
            ),
        ]

        result = analyzer._determine_overall_impact(contributions)

        assert result == "Strong positive impact from multiple components"

    def test_moderate_positive_impact(self):
        """
        SPEC: Should return moderate positive impact when total score >= 2.
        """
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=1,
                reason="",
                was_executed=True,
            ),
            ComponentContribution(
                component=RAGComponent.FACT_CHECK, score=1, reason="", was_executed=True
            ),
        ]

        result = analyzer._determine_overall_impact(contributions)

        assert result == "Moderate positive impact from components"

    def test_neutral_impact(self):
        """
        SPEC: Should return neutral impact when total score >= 0 but < 2.
        """
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=0,
                reason="",
                was_executed=True,
            ),
            ComponentContribution(
                component=RAGComponent.FACT_CHECK, score=1, reason="", was_executed=True
            ),
        ]

        result = analyzer._determine_overall_impact(contributions)

        assert result == "Neutral impact, components executed without clear benefit"

    def test_multiple_negative_impact(self):
        """
        SPEC: Should return multiple negative impact when 2+ components have negative scores.
        """
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=-1,
                reason="",
                was_executed=True,
            ),
            ComponentContribution(
                component=RAGComponent.FACT_CHECK,
                score=-1,
                reason="",
                was_executed=True,
            ),
        ]

        result = analyzer._determine_overall_impact(contributions)

        assert result == "Multiple components negatively impacted result"

    def test_some_negative_impact(self):
        """
        SPEC: Should return some negative impact when only 1 component has negative score and total is negative.
        """
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=-2,  # More negative to make total < 0
                reason="",
                was_executed=True,
            ),
            ComponentContribution(
                component=RAGComponent.FACT_CHECK, score=0, reason="", was_executed=True
            ),
        ]

        result = analyzer._determine_overall_impact(contributions)

        assert result == "Some components negatively impacted result"


class TestCheckTimingImportanceMissingBranches:
    """Tests for uncovered branches in _check_timing_importance."""

    def test_timing_important_with_slow_execution(self):
        """
        SPEC: Should return True when execution time > 5000ms.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_017",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.7,
            execution_time_ms=6000,  # > 5000
            rag_pipeline_log={},
            passed=True,
        )

        result = analyzer._check_timing_importance(test_result)

        assert result is True

    def test_timing_not_important_with_fast_execution(self):
        """
        SPEC: Should return False when execution time <= 5000ms.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_018",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.7,
            execution_time_ms=1000,  # < 5000
            rag_pipeline_log={},
            passed=True,
        )

        result = analyzer._check_timing_importance(test_result)

        assert result is False

    def test_timing_important_with_timeout_indicator(self):
        """
        SPEC: Should return True when log contains 'timeout'.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_019",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.7,
            execution_time_ms=1000,
            rag_pipeline_log={"error": "timeout occurred"},
            passed=True,
        )

        result = analyzer._check_timing_importance(test_result)

        assert result is True

    def test_timing_important_with_slow_indicator(self):
        """
        SPEC: Should return True when log contains 'slow'.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_020",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.7,
            execution_time_ms=1000,
            rag_pipeline_log={"warning": "slow response"},
            passed=True,
        )

        result = analyzer._check_timing_importance(test_result)

        assert result is True

    def test_timing_not_important_exactly_5000ms(self):
        """
        SPEC: Should return False when execution time is exactly 5000ms.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_021",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.7,
            execution_time_ms=5000,  # Exactly boundary
            rag_pipeline_log={},
            passed=True,
        )

        result = analyzer._check_timing_importance(test_result)

        assert result is False


class TestGetComponentSuggestionsMissingBranches:
    """Tests for uncovered branches in get_component_suggestions."""

    def test_suggestions_for_zero_score_executed_component(self):
        """
        SPEC: Should generate review suggestions for executed components with score 0.
        """
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=0,
                reason="Executed but unclear impact",
                was_executed=True,  # Was executed
            )
        ]

        analysis = ComponentAnalysis(
            test_case_id="test_022",
            contributions=contributions,
            overall_impact="Neutral",
            failure_cause_components=[],
            timestamp_importance=False,
        )

        suggestions = analyzer.get_component_suggestions(analysis)

        assert len(suggestions) > 0
        assert any(
            "review" in s.lower() or "configuration" in s.lower() for s in suggestions
        )

    def test_suggestions_with_timing_importance(self):
        """
        SPEC: Should include timing suggestion when timestamp_importance is True.
        """
        analyzer = ComponentAnalyzer()

        contributions = []

        analysis = ComponentAnalysis(
            test_case_id="test_023",
            contributions=contributions,
            overall_impact="Neutral",
            failure_cause_components=[],
            timestamp_importance=True,  # Timing is important
        )

        suggestions = analyzer.get_component_suggestions(analysis)

        # Actual suggestion text from implementation
        assert len(suggestions) > 0
        assert "execution time" in " ".join(suggestions).lower()

    def test_suggestions_combined_negative_and_zero_score(self):
        """
        SPEC: Should generate suggestions for both negative and zero-score components.
        """
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=-2,
                reason="Failed to retrieve",
                was_executed=True,
            ),
            ComponentContribution(
                component=RAGComponent.QUERY_ANALYZER,
                score=0,
                reason="Executed but unclear",
                was_executed=True,
            ),
        ]

        analysis = ComponentAnalysis(
            test_case_id="test_024",
            contributions=contributions,
            overall_impact="Negative",
            failure_cause_components=[RAGComponent.HYBRID_SEARCH],
            timestamp_importance=False,
        )

        suggestions = analyzer.get_component_suggestions(analysis)

        # Should have suggestions for both components
        assert len(suggestions) >= 2

    def test_suggestions_empty_for_all_positive_scores(self):
        """
        SPEC: Should return empty list when all contributions have positive scores.
        """
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=2,
                reason="Excellent",
                was_executed=True,
            ),
            ComponentContribution(
                component=RAGComponent.FACT_CHECK,
                score=1,
                reason="Good",
                was_executed=True,
            ),
        ]

        analysis = ComponentAnalysis(
            test_case_id="test_025",
            contributions=contributions,
            overall_impact="Positive",
            failure_cause_components=[],
            timestamp_importance=False,
        )

        suggestions = analyzer.get_component_suggestions(analysis)

        # No negative or zero-score executed components
        assert len(suggestions) == 0


class TestAnalyzeComponentsIntegration:
    """Integration tests for analyze_components with various scenarios."""

    def test_analyze_components_with_all_components_executed(self):
        """
        SPEC: Should handle case where all components were executed.
        """
        analyzer = ComponentAnalyzer()

        # Log that triggers all components
        test_result = TestResult(
            test_case_id="test_026",
            query="query",
            answer="answer",
            sources=["s1", "s2", "s3", "s4"],
            confidence=0.8,
            execution_time_ms=1500,
            rag_pipeline_log={
                "self_rag": True,
                "hyde": True,
                "corrective_rag": True,
                "hybrid_search": True,
                "bge_reranker": True,
                "query_analyzer": True,
                "dynamic_query_expansion": True,
                "fact_check": True,
            },
            passed=True,
        )

        analysis = analyzer.analyze_components(test_result)

        # All 8 RAGComponents should be marked as executed
        executed_count = sum(1 for c in analysis.contributions if c.was_executed)
        assert executed_count == 8

    def test_analyze_components_with_no_components_executed(self):
        """
        SPEC: Should handle case where no components were detected.
        """
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="test_027",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.3,
            execution_time_ms=1000,
            rag_pipeline_log={},  # Empty log
            passed=False,
        )

        analysis = analyzer.analyze_components(test_result)

        # Should still create 8 contributions (one per RAGComponent)
        assert len(analysis.contributions) == 8
        # But none marked as executed
        executed_count = sum(1 for c in analysis.contributions if c.was_executed)
        assert executed_count == 0
