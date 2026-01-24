"""
Specification tests for ComponentAnalyzer (Phase 4).

Tests define expected behavior for RAG component analysis.
Following DDD PRESERVE phase: specification tests for greenfield development.
"""


import pytest

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


class TestComponentDetection:
    """Specification tests for RAG component detection."""

    def test_detect_executed_components_identifies_hybrid_search(self):
        """
        SPEC: _detect_executed_components should identify hybrid_search in logs.

        Given: Pipeline logs containing hybrid_search pattern
        When: _detect_executed_components is called
        Then: hybrid_search is in the returned list
        """
        # Arrange
        analyzer = ComponentAnalyzer()
        pipeline_log = {
            "search_method": "hybrid_search",
            "dense_retrieval": True,
            "sparse_retrieval": True,
        }

        # Act
        components = analyzer._detect_executed_components(pipeline_log)

        # Assert
        assert "hybrid_search" in components

    def test_detect_executed_components_identifies_fact_check(self):
        """
        SPEC: _detect_executed_components should identify fact_check in logs.

        Given: Pipeline logs containing fact_check pattern
        When: _detect_executed_components is called
        Then: fact_check is in the returned list
        """
        # Arrange
        analyzer = ComponentAnalyzer()
        pipeline_log = {
            "fact_check": True,
            "verification": "passed",
        }

        # Act
        components = analyzer._detect_executed_components(pipeline_log)

        # Assert
        assert "fact_check" in components

    def test_detect_executed_components_identifies_reranker(self):
        """
        SPEC: _detect_executed_components should identify bge_reranker in logs.

        Given: Pipeline logs containing rerank pattern
        When: _detect_executed_components is called
        Then: bge_reranker is in the returned list
        """
        # Arrange
        analyzer = ComponentAnalyzer()
        pipeline_log = {
            "reranker": "bge",
            "reranked_results": True,
        }

        # Act
        components = analyzer._detect_executed_components(pipeline_log)

        # Assert
        assert "bge_reranker" in components


class TestComponentContributionEvaluation:
    """Specification tests for component contribution evaluation."""

    @pytest.fixture
    def successful_test_result(self):
        """Create a successful test result."""
        fact_checks = [
            FactCheck(
                claim="휴학 신청은 학기 개시 30일 전까지 가능",
                status=FactCheckStatus.PASS,
                source="규정 제10조",
                confidence=0.95,
            )
        ]

        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.9,
                completeness=0.85,
                relevance=0.9,
                source_citation=0.95,
                practicality=0.4,
                actionability=0.45,
            ),
            total_score=4.55,
            is_pass=True,
        )

        return TestResult(
            test_case_id="test_001",
            query="휴학 신청 방법",
            answer="휴학 신청은 학기 개시 30일 전까지 가능합니다...",
            sources=["규정 제10조", "규정 제11조"],
            confidence=0.88,
            execution_time_ms=1500,
            rag_pipeline_log={
                "search_method": "hybrid_search",
                "fact_check": True,
                "reranker": "bge",
            },
            fact_checks=fact_checks,
            quality_score=quality_score,
            passed=True,
        )

    def test_evaluate_component_contribution_positive_for_hybrid_search(self, successful_test_result):
        """
        SPEC: _evaluate_component_contribution should give positive score for successful hybrid_search.

        Given: A test result with good source coverage from hybrid_search
        When: _evaluate_component_contribution is called for HYBRID_SEARCH
        Then: Returns positive score (1 or 2)
        """
        # Arrange
        analyzer = ComponentAnalyzer()

        # Act
        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.HYBRID_SEARCH,
            successful_test_result,
        )

        # Assert
        assert score > 0
        assert "coverage" in reason.lower()

    def test_evaluate_component_contribution_positive_for_fact_check(self, successful_test_result):
        """
        SPEC: _evaluate_component_contribution should give positive score for passed fact checks.

        Given: A test result with all fact checks passed
        When: _evaluate_component_contribution is called for FACT_CHECK
        Then: Returns positive score (1 or 2)
        """
        # Arrange
        analyzer = ComponentAnalyzer()

        # Act
        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.FACT_CHECK,
            successful_test_result,
        )

        # Assert
        assert score == 2  # All fact checks passed
        assert "passed" in reason.lower()

    def test_evaluate_component_contribution_negative_for_failed_fact_check(self):
        """
        SPEC: _evaluate_component_contribution should give negative score for failed fact checks.

        Given: A test result with failed fact checks
        When: _evaluate_component_contribution is called for FACT_CHECK
        Then: Returns negative score (-1 or -2)
        """
        # Arrange
        analyzer = ComponentAnalyzer()

        fact_checks = [
            FactCheck(
                claim="Incorrect claim",
                status=FactCheckStatus.FAIL,
                source="규정 제1조",
                confidence=0.5,
                correction="정보가 틀렸습니다",
            ),
            FactCheck(
                claim="Another incorrect claim",
                status=FactCheckStatus.FAIL,
                source="규정 제2조",
                confidence=0.6,
            ),
        ]

        test_result = TestResult(
            test_case_id="test_002",
            query="질문",
            answer="잘못된 답변",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={"fact_check": True},
            fact_checks=fact_checks,
            passed=False,
        )

        # Act
        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.FACT_CHECK,
            test_result,
        )

        # Assert
        assert score < 0
        assert "errors" in reason.lower() or "failed" in reason.lower()


class TestComponentAnalysis:
    """Specification tests for complete component analysis."""

    @pytest.fixture
    def sample_test_result(self):
        """Create a sample test result for analysis."""
        return TestResult(
            test_case_id="test_003",
            query="복학 신청 방법",
            answer="복학 신청은 학기 개시 20일 전까지...",
            sources=["규정 제15조"],
            confidence=0.75,
            execution_time_ms=1200,
            rag_pipeline_log={
                "search_method": "hybrid_search",
                "query_analyzer": True,
            },
            passed=True,
        )

    def test_analyze_components_creates_analysis_for_all_components(self, sample_test_result):
        """
        SPEC: analyze_components should create contribution for all RAG components.

        Given: A test result
        When: analyze_components is called
        Then: ComponentAnalysis has 8 contributions (one for each RAGComponent)
        """
        # Arrange
        analyzer = ComponentAnalyzer()

        # Act
        analysis = analyzer.analyze_components(sample_test_result)

        # Assert
        assert len(analysis.contributions) == 8  # One for each RAGComponent

    def test_analyze_components_identifies_failure_causes(self, sample_test_result):
        """
        SPEC: analyze_components should identify components with negative scores.

        Given: A test result with some failed components
        When: analyze_components is called
        Then: failure_cause_components includes components with score <= -1
        """
        # Arrange
        analyzer = ComponentAnalyzer()

        # Create a test result that would cause failures
        failed_result = TestResult(
            test_case_id="test_004",
            query="질문",
            answer="답변",
            sources=[],  # No sources - should cause hybrid_search failure
            confidence=0.3,
            execution_time_ms=2000,
            rag_pipeline_log={},
            passed=False,
        )

        # Act
        analysis = analyzer.analyze_components(failed_result)

        # Assert
        # At least hybrid_search should be a failure cause (score = -1 or -2)
        failure_cause_names = [c.value for c in analysis.failure_cause_components]
        assert "hybrid_search" in failure_cause_names

    def test_analyze_components_calculates_net_impact_score(self, sample_test_result):
        """
        SPEC: analyze_components should calculate net impact score.

        Given: A test result
        When: analyze_components is called
        Then: net_impact_score equals sum of all contribution scores
        """
        # Arrange
        analyzer = ComponentAnalyzer()

        # Act
        analysis = analyzer.analyze_components(sample_test_result)

        # Assert
        expected_score = sum(c.score for c in analysis.contributions)
        assert analysis.net_impact_score == expected_score


class TestFailureMapping:
    """Specification tests for failure type to component mapping."""

    def test_map_failure_to_components_for_low_relevance(self):
        """
        SPEC: map_failure_to_components should map low_relevance to hybrid_search.

        Given: failure_type is "low_relevance"
        When: map_failure_to_components is called
        Then: Returns list containing HYBRID_SEARCH
        """
        # Arrange
        analyzer = ComponentAnalyzer()

        # Act
        components = analyzer.map_failure_to_components("low_relevance")

        # Assert
        assert RAGComponent.HYBRID_SEARCH in components

    def test_map_failure_to_components_for_incorrect_answer(self):
        """
        SPEC: map_failure_to_components should map incorrect_answer to fact_check.

        Given: failure_type is "incorrect_answer"
        When: map_failure_to_components is called
        Then: Returns list containing FACT_CHECK
        """
        # Arrange
        analyzer = ComponentAnalyzer()

        # Act
        components = analyzer.map_failure_to_components("incorrect_answer")

        # Assert
        assert RAGComponent.FACT_CHECK in components

    def test_map_failure_to_components_defaults_to_common_components(self):
        """
        SPEC: map_failure_to_components should default to common components for unknown types.

        Given: failure_type is unknown
        When: map_failure_to_components is called
        Then: Returns list with HYBRID_SEARCH and QUERY_ANALYZER
        """
        # Arrange
        analyzer = ComponentAnalyzer()

        # Act
        components = analyzer.map_failure_to_components("unknown_failure_type")

        # Assert
        assert RAGComponent.HYBRID_SEARCH in components
        assert RAGComponent.QUERY_ANALYZER in components


class TestSuggestions:
    """Specification tests for component improvement suggestions."""

    def test_get_component_suggestions_for_negative_contributions(self):
        """
        SPEC: get_component_suggestions should generate suggestions for negative contributions.

        Given: ComponentAnalysis with negative contribution scores
        When: get_component_suggestions is called
        Then: Returns suggestions to improve those components
        """
        # Arrange
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=-2,
                reason="Failed to retrieve sources",
                was_executed=True,
            ),
        ]

        analysis = ComponentAnalysis(
            test_case_id="test_005",
            contributions=contributions,
            overall_impact="Negative impact",
            failure_cause_components=[RAGComponent.HYBRID_SEARCH],
            timestamp_importance=False,
        )

        # Act
        suggestions = analyzer.get_component_suggestions(analysis)

        # Assert
        assert len(suggestions) > 0
        assert any("hybrid_search" in s.lower() for s in suggestions)
        assert any("improve" in s.lower() for s in suggestions)
