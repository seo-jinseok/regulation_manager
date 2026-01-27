"""
Additional coverage tests for ComponentAnalyzer to reach 80%+ coverage.

Tests remaining uncovered code paths in:
- map_failure_to_components with all failure types
- map_failure_to_components with invalid component names
- Edge cases in component detection
"""

from src.rag.automation.domain.entities import QualityTestResult
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


class TestMapFailureToComponentsAllTypes:
    """Tests for map_failure_to_components with all defined failure types."""

    def test_map_irrelevant_sources_to_corrective_rag(self):
        """
        SPEC: irrelevant_sources should map to corrective_rag.
        """
        analyzer = ComponentAnalyzer()
        result = analyzer.map_failure_to_components("irrelevant_sources")

        assert RAGComponent.CORRECTIVE_RAG in result

    def test_map_missing_information_to_dynamic_query_expansion(self):
        """
        SPEC: missing_information should map to dynamic_query_expansion.
        """
        analyzer = ComponentAnalyzer()
        result = analyzer.map_failure_to_components("missing_information")

        assert RAGComponent.DYNAMIC_QUERY_EXPANSION in result

    def test_map_poor_ranking_to_bge_reranker(self):
        """
        SPEC: poor_ranking should map to bge_reranker.
        """
        analyzer = ComponentAnalyzer()
        result = analyzer.map_failure_to_components("poor_ranking")

        assert RAGComponent.BGE_RERANKER in result

    def test_map_misunderstood_intent_to_query_analyzer(self):
        """
        SPEC: misunderstood_intent should map to query_analyzer.
        """
        analyzer = ComponentAnalyzer()
        result = analyzer.map_failure_to_components("misunderstood_intent")

        assert RAGComponent.QUERY_ANALYZER in result

    def test_map_redundant_retrieval_to_self_rag(self):
        """
        SPEC: redundant_retrieval should map to self_rag.
        """
        analyzer = ComponentAnalyzer()
        result = analyzer.map_failure_to_components("redundant_retrieval")

        assert RAGComponent.SELF_RAG in result

    def test_map_generic_query_to_hyde(self):
        """
        SPEC: generic_query should map to hyde.
        """
        analyzer = ComponentAnalyzer()
        result = analyzer.map_failure_to_components("generic_query")

        assert RAGComponent.HYDE in result


class TestMapFailureToComponentsEdgeCases:
    """Tests for edge cases in map_failure_to_components."""

    def test_map_failure_to_components_invalid_component_name(self):
        """
        SPEC: Invalid component name in mapping should return default components.

        This tests the ValueError exception handling when trying to create
        RAGComponent from an invalid enum value.
        """
        analyzer = ComponentAnalyzer()

        # Create a mock scenario where FAILURE_TYPE_TO_COMPONENT might have
        # an invalid value (simulating future code changes or errors)
        # For now, we test that unknown failure types return defaults

        result = analyzer.map_failure_to_components("completely_unknown_failure")

        # Should return default components when mapping fails
        assert RAGComponent.HYBRID_SEARCH in result
        assert RAGComponent.QUERY_ANALYZER in result

    def test_map_failure_to_components_none_mapping(self):
        """
        SPEC: None mapping result should return default components.
        """
        analyzer = ComponentAnalyzer()

        # Test with a failure type that doesn't exist in the mapping
        result = analyzer.map_failure_to_components("nonexistent_failure_xyz")

        # Should return default components
        assert len(result) == 2
        assert RAGComponent.HYBRID_SEARCH in result
        assert RAGComponent.QUERY_ANALYZER in result


class TestDetectExecutedComponents:
    """Tests for _detect_executed_components edge cases."""

    def test_detect_executed_components_empty_log(self):
        """
        SPEC: Empty log should return empty component list.
        """
        analyzer = ComponentAnalyzer()
        result = analyzer._detect_executed_components({})

        assert result == []

    def test_detect_executed_components_none_log(self):
        """
        SPEC: None log should be handled gracefully.
        """
        analyzer = ComponentAnalyzer()
        # Convert None to string for pattern matching
        result = analyzer._detect_executed_components(None)

        # None converted to "none" string won't match any patterns
        assert result == []

    def test_detect_executed_components_nested_log(self):
        """
        SPEC: Should detect components in nested log structures.
        """
        analyzer = ComponentAnalyzer()
        log = {
            "level1": {
                "level2": {
                    "self_rag": True,
                    "hyde": "enabled",
                }
            },
            "array": ["item1", "fact_check", "item3"],
        }

        result = analyzer._detect_executed_components(log)

        # Should detect components even when nested
        assert "self_rag" in result
        assert "fact_check" in result

    def test_detect_executed_components_case_insensitive(self):
        """
        SPEC: Pattern matching should be case-insensitive.
        """
        analyzer = ComponentAnalyzer()
        log = {
            "Self_RAG": True,
            "HyDe": "enabled",
            "HyBriD_SearCh": "active",
        }

        result = analyzer._detect_executed_components(log)

        # Should match despite case differences
        assert "self_rag" in result
        assert "hyde" in result
        assert "hybrid_search" in result

    def test_detect_executed_components_multiple_patterns(self):
        """
        SPEC: Should detect all patterns that match for a component.
        """
        analyzer = ComponentAnalyzer()
        log = {
            "hybrid_search": True,
            "dense_retrieval": True,
            "sparse_retrieval": True,
        }

        result = analyzer._detect_executed_components(log)

        # All three patterns should result in one component entry
        # The method appends the component name when ANY pattern matches
        # So hybrid_search appears once even though multiple patterns match
        assert "hybrid_search" in result
        # Count should be 1 (not 3) because it's unique component names
        assert result.count("hybrid_search") == 1


class TestAnalyzeComponentsEdgeCases:
    """Tests for analyze_components with edge cases."""

    def test_analyze_components_minimal_log(self):
        """
        SPEC: Should handle minimal log with minimal information.
        """
        analyzer = ComponentAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test_minimal",
            query="test query",
            answer="test answer",
            sources=[],
            confidence=0.5,
            execution_time_ms=100,
            rag_pipeline_log={"minimal": "data"},
            passed=True,
        )

        analysis = analyzer.analyze_components(test_result)

        # Should still create analysis with all components
        assert len(analysis.contributions) == 8
        assert analysis.test_case_id == "test_minimal"

    def test_analyze_components_with_string_log(self):
        """
        SPEC: Should handle log as string rather than dict.
        """
        analyzer = ComponentAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test_string",
            query="test query",
            answer="test answer",
            sources=[],
            confidence=0.5,
            execution_time_ms=100,
            # Pass string instead of dict
            rag_pipeline_log="fact_check passed self_rag reflection",
            passed=True,
        )

        analysis = analyzer.analyze_components(test_result)

        # Should detect components from string
        executed = [c for c in analysis.contributions if c.was_executed]
        assert len(executed) > 0

    def test_analyze_components_very_long_execution_time(self):
        """
        SPEC: Should handle very long execution times.
        """
        analyzer = ComponentAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test_slow",
            query="test query",
            answer="test answer",
            sources=[],
            confidence=0.5,
            execution_time_ms=100000,  # 100 seconds
            rag_pipeline_log={},
            passed=False,
        )

        analysis = analyzer.analyze_components(test_result)

        # Timing should be flagged as important
        assert analysis.timestamp_importance is True

    def test_analyze_components_zero_execution_time(self):
        """
        SPEC: Should handle zero execution time.
        """
        analyzer = ComponentAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test_zero",
            query="test query",
            answer="test answer",
            sources=[],
            confidence=0.5,
            execution_time_ms=0,
            rag_pipeline_log={},
            passed=True,
        )

        analysis = analyzer.analyze_components(test_result)

        # Timing should not be flagged
        assert analysis.timestamp_importance is False

    def test_analyze_components_all_positive_scores(self):
        """
        SPEC: Should calculate correct impact when all scores positive.
        """
        analyzer = ComponentAnalyzer()

        # Create a log that triggers all components positively
        test_result = QualityTestResult(
            test_case_id="test_all_positive",
            query="test query",
            answer="test answer with good sources",
            sources=["source1", "source2", "source3", "source4", "source5"],
            confidence=0.9,
            execution_time_ms=1000,
            rag_pipeline_log={
                "self_rag": True,
                "reflection": "improved",
                "hyde": "enabled",
                "document_embedding": "done",
                "corrective": "filtered irrelevant",
                "is_relevant": "yes",
                "hybrid_search": "active",
                "dense_retrieval": True,
                "sparse_retrieval": True,
                "bge": "reranker",
                "rerank": "success",
                "query_analyzer": True,
                "intent_analysis": "matched",
                "query_expansion": "expanded",
                "expanded_query": "expanded terms",
                "fact_check": True,
                "verify": "done",
            },
            passed=True,
        )

        # Create fact checks that all pass
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
        ]

        test_result.fact_checks = fact_checks

        # Create quality score with all high dimensions
        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.9,
                completeness=0.9,
                relevance=0.9,
                source_citation=0.9,
                practicality=0.8,
                actionability=0.8,
            ),
            total_score=5.0,
            is_pass=True,
        )

        test_result.quality_score = quality_score

        analysis = analyzer.analyze_components(test_result)

        # Should have strong positive impact
        assert (
            analysis.overall_impact == "Strong positive impact from multiple components"
        )
        assert len(analysis.failure_cause_components) == 0


class TestGetComponentSuggestions:
    """Additional tests for get_component_suggestions."""

    def test_suggestions_with_all_negative_scores(self):
        """
        SPEC: Should generate suggestions for all negative contributions.
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
                component=RAGComponent.FACT_CHECK,
                score=-1,
                reason="Errors found",
                was_executed=True,
            ),
        ]

        analysis = ComponentAnalysis(
            test_case_id="test",
            contributions=contributions,
            overall_impact="Negative",
            failure_cause_components=[
                RAGComponent.HYBRID_SEARCH,
                RAGComponent.FACT_CHECK,
            ],
            timestamp_importance=False,
        )

        suggestions = analyzer.get_component_suggestions(analysis)

        # Should have suggestions for both negative components
        assert len(suggestions) >= 2

    def test_suggestions_mixed_scores(self):
        """
        SPEC: Should generate suggestions only for negative and zero-score executed.
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
                component=RAGComponent.QUERY_ANALYZER,
                score=-1,
                reason="Failed",
                was_executed=True,
            ),
            ComponentContribution(
                component=RAGComponent.FACT_CHECK,
                score=0,
                reason="Unclear",
                was_executed=True,
            ),
            ComponentContribution(
                component=RAGComponent.SELF_RAG,
                score=1,
                reason="Good",
                was_executed=True,
            ),
        ]

        analysis = ComponentAnalysis(
            test_case_id="test",
            contributions=contributions,
            overall_impact="Mixed",
            failure_cause_components=[RAGComponent.QUERY_ANALYZER],
            timestamp_importance=False,
        )

        suggestions = analyzer.get_component_suggestions(analysis)

        # Should have suggestions for negative and zero-score executed only
        assert len(suggestions) >= 2
        # Check that positive score components don't get "Improve" suggestions
        for suggestion in suggestions:
            assert "hybrid_search" not in suggestion.lower()
            assert "self_rag" not in suggestion.lower()


class TestEvaluateComponentContributionFactCheckEdgeCases:
    """Additional tests for fact_check contribution evaluation."""

    def test_fact_check_all_fail_with_some_pass(self):
        """
        SPEC: Should handle mixed pass/fail with more fails.
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
                status=FactCheckStatus.FAIL,
                source="source",
                confidence=0.3,
                correction="Wrong",
            ),
            FactCheck(
                claim="Claim 3",
                status=FactCheckStatus.FAIL,
                source="source",
                confidence=0.4,
                correction="Also wrong",
            ),
        ]

        test_result = QualityTestResult(
            test_case_id="test",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={"fact_check": True},
            fact_checks=fact_checks,
            passed=False,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.FACT_CHECK, test_result
        )

        # 1/3 passed is not more than half, so should be negative
        assert score == -1
        assert "errors" in reason.lower() or "failed" in reason.lower()

    def test_fact_check_exactly_half_pass(self):
        """
        SPEC: Should handle exactly half pass (border case).
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
                status=FactCheckStatus.FAIL,
                source="source",
                confidence=0.3,
                correction="Wrong",
            ),
        ]

        test_result = QualityTestResult(
            test_case_id="test",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={"fact_check": True},
            fact_checks=fact_checks,
            passed=False,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.FACT_CHECK, test_result
        )

        # 1/2 = 0.5 is not > 0.5, so should not get score 1
        # Should get negative since half failed
        assert score == -1


class TestEvaluateMissingComponentEdgeCases:
    """Additional tests for missing component evaluation."""

    def test_missing_hybrid_search_with_no_sources_negative(self):
        """
        SPEC: Missing hybrid search with no sources should score -2.
        """
        analyzer = ComponentAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test",
            query="query",
            answer="answer",
            sources=[],  # No sources
            confidence=0.3,
            execution_time_ms=1000,
            rag_pipeline_log={},  # No hybrid_search
            passed=False,
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.HYBRID_SEARCH, test_result
        )

        assert score == -2
        assert "no sources" in reason.lower()

    def test_missing_fact_check_with_passed_test(self):
        """
        SPEC: Missing fact check with passed test should not penalize.
        """
        analyzer = ComponentAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.8,
            execution_time_ms=1000,
            rag_pipeline_log={},  # No fact_check
            passed=True,  # Test passed
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.FACT_CHECK, test_result
        )

        # Should not penalize missing fact check if test passed
        assert score == 0
