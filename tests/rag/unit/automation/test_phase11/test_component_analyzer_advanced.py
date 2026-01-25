"""
Advanced Unit Tests for ComponentAnalyzer (Phase 11).

Tests for advanced RAG component scenarios including:
- Self-RAG reflection quality improvement
- HyDE hypothetical embeddings
- Corrective-RAG filtering
- BGE Reranker relevance scoring
- Dynamic query expansion
- Query analyzer intent matching
- Missing component evaluation
- Complex multi-component scenarios

Clean Architecture: Infrastructure layer tests.
"""

import pytest

from src.rag.automation.domain.entities import TestResult
from src.rag.automation.domain.value_objects import (
    ComponentContribution,
    FactCheck,
    FactCheckStatus,
    QualityDimensions,
    QualityScore,
    RAGComponent,
)
from src.rag.automation.infrastructure.component_analyzer import ComponentAnalyzer


class TestSelfRAGScenarios:
    """Test suite for Self-RAG reflection quality scenarios."""

    @pytest.fixture
    def high_accuracy_result(self):
        """Create a test result with high accuracy (Self-RAG reflection helped)."""
        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.92,  # High accuracy from reflection
                completeness=0.85,
                relevance=0.88,
                source_citation=0.90,
                practicality=0.70,
                actionability=0.72,
            ),
            total_score=5.07,
            is_pass=True,
        )

        return TestResult(
            test_case_id="self_rag_001",
            query="복수전공 신청 자격",
            answer="복수전공 신청은 2학년 이상...",
            sources=["규정 제20조", "규정 제21조"],
            confidence=0.85,
            execution_time_ms=1800,
            rag_pipeline_log={
                "self_rag": True,
                "retrieve_feedback": "improved",
            },
            quality_score=quality_score,
            passed=True,
        )

    def test_self_rag_positive_impact_with_high_accuracy(self, high_accuracy_result):
        """WHEN Self-RAG produces high accuracy, THEN should give positive score."""
        analyzer = ComponentAnalyzer()

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.SELF_RAG,
            high_accuracy_result,
        )

        assert score == 1
        assert "reflection improved" in reason.lower()

    def test_self_rag_neutral_impact_with_low_accuracy(self):
        """WHEN Self-RAG executes but accuracy is low, THEN should give neutral score."""
        analyzer = ComponentAnalyzer()

        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.65,  # Low accuracy
                completeness=0.70,
                relevance=0.68,
                source_citation=0.75,
                practicality=0.60,
                actionability=0.62,
            ),
            total_score=4.0,
            is_pass=False,
        )

        test_result = TestResult(
            test_case_id="self_rag_002",
            query="질문",
            answer="답변",
            sources=[],
            confidence=0.60,
            execution_time_ms=1500,
            rag_pipeline_log={"self_rag": True},
            quality_score=quality_score,
            passed=False,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.SELF_RAG,
            test_result,
        )

        assert score == 0
        assert "unclear impact" in reason.lower()

    def test_self_rag_not_executed_optional_component(self):
        """WHEN Self-RAG is not executed, THEN should treat as optional (neutral score)."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="self_rag_003",
            query="질문",
            answer="답변",
            sources=["규정 제1조"],
            confidence=0.70,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=True,
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.SELF_RAG,
            test_result,
        )

        assert score == 0
        assert "optional" in reason.lower()


class TestHyDESCenarios:
    """Test suite for HyDE hypothetical embeddings scenarios."""

    def test_hyde_positive_with_diverse_sources(self):
        """WHEN HyDE produces diverse source retrieval, THEN should give positive score."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="hyde_001",
            query="장학금 신청 방법",
            answer="장학금 신청은 다음과 같습니다...",
            sources=[
                "규정 제30조",
                "규정 제31조",
                "규정 제32조",
                "규정 제33조",
            ],  # 4 diverse sources
            confidence=0.82,
            execution_time_ms=1600,
            rag_pipeline_log={
                "hyde": True,
                "hypothetical": "document_embedding",
            },
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.HYDE,
            test_result,
        )

        assert score == 1
        assert "diverse source" in reason.lower()

    def test_hyde_neutral_with_limited_sources(self):
        """WHEN HyDE executes but limited source diversity, THEN should give neutral score."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="hyde_002",
            query="질문",
            answer="답변",
            sources=["규정 제1조"],  # Only 1 source
            confidence=0.65,
            execution_time_ms=1400,
            rag_pipeline_log={"hyde": True},
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.HYDE,
            test_result,
        )

        assert score == 0
        assert "limited source diversity" in reason.lower()

    def test_hyde_not_executed_optional(self):
        """WHEN HyDE is not executed, THEN should treat as optional."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="hyde_003",
            query="질문",
            answer="답변",
            sources=["규정 제1조"],
            confidence=0.70,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=True,
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.HYDE,
            test_result,
        )

        assert score == 0
        assert "optional" in reason.lower()


class TestCorrectiveRAGScenarios:
    """Test suite for Corrective-RAG filtering scenarios."""

    def test_corrective_rag_positive_with_filtering(self):
        """WHEN Corrective-RAG filters irrelevant content, THEN should give positive score."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="corrective_001",
            query="휴학 신청 방법",
            answer="휴학 신청은 다음과 같습니다...",
            sources=["규정 제10조"],
            confidence=0.80,
            execution_time_ms=1700,
            rag_pipeline_log={
                "corrective": True,
                "is_relevant": True,
                "filtered": 5,  # Filtered 5 irrelevant docs
            },
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.CORRECTIVE_RAG,
            test_result,
        )

        assert score == 1
        assert "filtered" in reason.lower()

    def test_corrective_rag_neutral_without_filtering(self):
        """WHEN Corrective-RAG executes without filtering, THEN should give neutral score."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="corrective_002",
            query="질문",
            answer="답변",
            sources=["규정 제1조"],
            confidence=0.70,
            execution_time_ms=1500,
            rag_pipeline_log={"corrective": True},
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.CORRECTIVE_RAG,
            test_result,
        )

        assert score == 0
        assert "without filtering" in reason.lower()


class TestBGERerankerScenarios:
    """Test suite for BGE Reranker relevance scoring scenarios."""

    def test_reranker_positive_with_high_confidence(self):
        """WHEN Reranking produces high confidence, THEN should give positive score."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="reranker_001",
            query="성적 정정 방법",
            answer="성적 정정은 다음과 같습니다...",
            sources=["규정 제50조"],
            confidence=0.88,  # High confidence from reranking
            execution_time_ms=1600,
            rag_pipeline_log={
                "reranker": "bge",
                "rerank": True,
            },
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.BGE_RERANKER,
            test_result,
        )

        assert score == 1
        assert "confidence" in reason.lower()

    def test_reranker_neutral_with_low_confidence(self):
        """WHEN Reranking produces low confidence, THEN should give neutral score."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="reranker_002",
            query="질문",
            answer="답변",
            sources=["규정 제1조"],
            confidence=0.55,  # Low confidence
            execution_time_ms=1500,
            rag_pipeline_log={"reranker": "bge"},
            passed=False,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.BGE_RERANKER,
            test_result,
        )

        assert score == 0
        assert "confidence low" in reason.lower()

    def test_reranker_not_executed_optional(self):
        """WHEN Reranker is not executed, THEN should treat as optional."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="reranker_003",
            query="질문",
            answer="답변",
            sources=["규정 제1조"],
            confidence=0.70,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=True,
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.BGE_RERANKER,
            test_result,
        )

        assert score == 0
        assert "optional" in reason.lower()


class TestDynamicQueryExpansionScenarios:
    """Test suite for Dynamic Query Expansion scenarios."""

    def test_query_expansion_positive_with_expanded_query(self):
        """WHEN Query expansion improves retrieval, THEN should give positive score."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="expansion_001",
            query="등록금 납부",
            answer="등록금 납부는 다음과 같습니다...",
            sources=["규정 제40조", "규정 제41조"],
            confidence=0.80,
            execution_time_ms=1500,
            rag_pipeline_log={
                "query_expansion": True,
                "expanded_query": "등록금 납부 기간 방법 연체",
            },
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.DYNAMIC_QUERY_EXPANSION,
            test_result,
        )

        assert score == 1
        assert "improved retrieval" in reason.lower()

    def test_query_expansion_neutral_without_clear_benefit(self):
        """WHEN Query expansion executes without clear benefit, THEN should give neutral score."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="expansion_002",
            query="질문",
            answer="답변",
            sources=["규정 제1조"],
            confidence=0.65,
            execution_time_ms=1400,
            rag_pipeline_log={"query_expansion": True},
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.DYNAMIC_QUERY_EXPANSION,
            test_result,
        )

        assert score == 0
        assert "without clear benefit" in reason.lower()


class TestQueryAnalyzerScenarios:
    """Test suite for Query Analyzer intent matching scenarios."""

    def test_query_analyzer_positive_with_high_relevance(self):
        """WHEN Query analysis produces high relevance, THEN should give positive score."""
        analyzer = ComponentAnalyzer()

        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.80,
                completeness=0.82,
                relevance=0.88,  # High relevance
                source_citation=0.85,
                practicality=0.70,
                actionability=0.72,
            ),
            total_score=4.77,
            is_pass=True,
        )

        test_result = TestResult(
            test_case_id="analyzer_001",
            query="전과 신청 자격",
            answer="전과 신청 자격은 다음과 같습니다...",
            sources=["규정 제25조"],
            confidence=0.82,
            execution_time_ms=1300,
            rag_pipeline_log={
                "query_analyzer": True,
                "intent_analysis": "eligibility_query",
            },
            quality_score=quality_score,
            passed=True,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.QUERY_ANALYZER,
            test_result,
        )

        assert score == 1
        assert "relevance" in reason.lower()

    def test_query_analyzer_neutral_with_low_relevance(self):
        """WHEN Query analysis produces low relevance, THEN should give neutral score."""
        analyzer = ComponentAnalyzer()

        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.65,
                completeness=0.68,
                relevance=0.55,  # Low relevance
                source_citation=0.70,
                practicality=0.60,
                actionability=0.62,
            ),
            total_score=3.8,
            is_pass=False,
        )

        test_result = TestResult(
            test_case_id="analyzer_002",
            query="질문",
            answer="답변",
            sources=[],
            confidence=0.55,
            execution_time_ms=1400,
            rag_pipeline_log={"query_analyzer": True},
            quality_score=quality_score,
            passed=False,
        )

        score, reason = analyzer._evaluate_component_contribution(
            RAGComponent.QUERY_ANALYZER,
            test_result,
        )

        assert score == 0
        assert "relevance low" in reason.lower()


class TestMissingComponentEvaluation:
    """Test suite for missing component evaluation logic."""

    def test_hybrid_search_missing_no_sources_negative(self):
        """WHEN Hybrid search is missing and no sources, THEN should give -2 score."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="missing_001",
            query="질문",
            answer="답변",
            sources=[],  # No sources
            confidence=0.2,
            execution_time_ms=500,
            rag_pipeline_log={},
            passed=False,
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.HYBRID_SEARCH,
            test_result,
        )

        assert score == -2
        assert "no sources retrieved" in reason.lower()

    def test_hybrid_search_missing_but_sources_found_negative(self):
        """WHEN Hybrid search is missing but sources found (fallback), THEN should give -1 score."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="missing_002",
            query="질문",
            answer="답변",
            sources=["규정 제1조"],  # Sources found (fallback)
            confidence=0.60,
            execution_time_ms=800,
            rag_pipeline_log={},
            passed=True,
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.HYBRID_SEARCH,
            test_result,
        )

        assert score == -1
        assert "fallback" in reason.lower()

    def test_fact_check_missing_with_test_failed_negative(self):
        """WHEN Fact check is missing and test failed, THEN should give -1 score."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="missing_003",
            query="질문",
            answer="답변",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,  # Test failed
        )

        score, reason = analyzer._evaluate_missing_component(
            RAGComponent.FACT_CHECK,
            test_result,
        )

        assert score == -1
        assert "without verification" in reason.lower()


class TestOverallImpactDetermination:
    """Test suite for overall impact determination logic."""

    def test_strong_positive_impact_from_multiple_components(self):
        """WHEN total score >= 4, THEN should return 'Strong positive impact'."""
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=2,
                reason="Excellent coverage",
                was_executed=True,
            ),
            ComponentContribution(
                component=RAGComponent.FACT_CHECK,
                score=2,
                reason="All checks passed",
                was_executed=True,
            ),
        ]

        impact = analyzer._determine_overall_impact(contributions)

        assert impact == "Strong positive impact from multiple components"

    def test_moderate_positive_impact(self):
        """WHEN total score >= 2, THEN should return 'Moderate positive impact'."""
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=1,
                reason="Adequate coverage",
                was_executed=True,
            ),
            ComponentContribution(
                component=RAGComponent.FACT_CHECK,
                score=1,
                reason="Most checks passed",
                was_executed=True,
            ),
        ]

        impact = analyzer._determine_overall_impact(contributions)

        assert impact == "Moderate positive impact from components"

    def test_neutral_impact(self):
        """WHEN total score >= 0, THEN should return 'Neutral impact'."""
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=0,
                reason="No clear benefit",
                was_executed=True,
            ),
        ]

        impact = analyzer._determine_overall_impact(contributions)

        assert impact == "Neutral impact, components executed without clear benefit"

    def test_multiple_components_negative_impact(self):
        """WHEN 2+ components have negative scores, THEN should return 'Multiple components negative'."""
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=-1,
                reason="Failed",
                was_executed=True,
            ),
            ComponentContribution(
                component=RAGComponent.FACT_CHECK,
                score=-1,
                reason="Failed",
                was_executed=True,
            ),
        ]

        impact = analyzer._determine_overall_impact(contributions)

        assert impact == "Multiple components negatively impacted result"

    def test_some_components_negative_impact(self):
        """WHEN total score < 0 and < 2 negative components, THEN should return 'Some negative'."""
        analyzer = ComponentAnalyzer()

        contributions = [
            ComponentContribution(
                component=RAGComponent.HYBRID_SEARCH,
                score=-1,
                reason="Failed",
                was_executed=True,
            ),
        ]

        impact = analyzer._determine_overall_impact(contributions)

        assert impact == "Some components negatively impacted result"


class TestTimingImportance:
    """Test suite for timing importance detection."""

    def test_timing_important_with_slow_execution(self):
        """WHEN execution time > 5 seconds, THEN should return True."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="timing_001",
            query="질문",
            answer="답변",
            sources=[],
            confidence=0.5,
            execution_time_ms=5500,  # > 5 seconds
            rag_pipeline_log={},
            passed=True,
        )

        is_important = analyzer._check_timing_importance(test_result)

        assert is_important is True

    def test_timing_important_with_timeout_indicator(self):
        """WHEN log contains 'timeout', THEN should return True."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="timing_002",
            query="질문",
            answer="답변",
            sources=[],
            confidence=0.5,
            execution_time_ms=3000,
            rag_pipeline_log={
                "timeout": True,
                "component": "reranker",
            },
            passed=True,
        )

        is_important = analyzer._check_timing_importance(test_result)

        assert is_important is True

    def test_timing_not_important_normal_execution(self):
        """WHEN execution time is normal and no timeout indicators, THEN should return False."""
        analyzer = ComponentAnalyzer()

        test_result = TestResult(
            test_case_id="timing_003",
            query="질문",
            answer="답변",
            sources=["규정 제1조"],
            confidence=0.75,
            execution_time_ms=1200,  # Normal time
            rag_pipeline_log={},
            passed=True,
        )

        is_important = analyzer._check_timing_importance(test_result)

        assert is_important is False


class TestComplexMultiComponentScenarios:
    """Test suite for complex multi-component interaction scenarios."""

    def test_full_rag_pipeline_all_components_positive(self):
        """WHEN full RAG pipeline executes successfully, THEN all components should have positive scores."""
        analyzer = ComponentAnalyzer()

        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.90,
                completeness=0.88,
                relevance=0.92,
                source_citation=0.95,
                practicality=0.85,
                actionability=0.87,
            ),
            total_score=5.37,
            is_pass=True,
        )

        fact_checks = [
            FactCheck(
                claim="휴학 신청은 30일 전까지",
                status=FactCheckStatus.PASS,
                source="규정 제10조",
                confidence=0.95,
            ),
            FactCheck(
                claim="복학 신청은 20일 전까지",
                status=FactCheckStatus.PASS,
                source="규정 제15조",
                confidence=0.93,
            ),
        ]

        test_result = TestResult(
            test_case_id="complex_001",
            query="휴학과 복학 신청 방법",
            answer="휴학과 복학 신청 방법은 다음과 같습니다...",
            sources=[
                "규정 제10조",
                "규정 제11조",
                "규정 제12조",
                "규정 제15조",
                "규정 제16조",
            ],
            confidence=0.90,
            execution_time_ms=2500,
            rag_pipeline_log={
                "self_rag": True,
                "retrieve_feedback": "improved",
                "hyde": True,
                "corrective": True,
                "is_relevant": True,
                "filtered": 3,
                "hybrid_search": True,
                "dense_retrieval": True,
                "sparse_retrieval": True,
                "reranker": "bge",
                "query_analyzer": True,
                "intent_analysis": "procedural",
                "query_expansion": True,
                "expanded_query": "휴학 복학 절차 기간",
                "fact_check": True,
                "verify": True,
            },
            fact_checks=fact_checks,
            quality_score=quality_score,
            passed=True,
        )

        analysis = analyzer.analyze_components(test_result)

        # Check that positive components have positive scores
        hybrid_contrib = next(
            (
                c
                for c in analysis.contributions
                if c.component == RAGComponent.HYBRID_SEARCH
            ),
            None,
        )
        assert hybrid_contrib is not None
        assert hybrid_contrib.score > 0

        fact_check_contrib = next(
            (
                c
                for c in analysis.contributions
                if c.component == RAGComponent.FACT_CHECK
            ),
            None,
        )
        assert fact_check_contrib is not None
        assert fact_check_contrib.score == 2  # All fact checks passed

        # Check overall impact
        assert "Strong positive" in analysis.overall_impact

    def test_degraded_pipeline_multiple_failures(self):
        """WHEN pipeline has multiple component failures, THEN should identify all failure causes."""
        analyzer = ComponentAnalyzer()

        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.50,
                completeness=0.45,
                relevance=0.40,
                source_citation=0.30,
                practicality=0.35,
                actionability=0.30,
            ),
            total_score=2.3,
            is_pass=False,
        )

        fact_checks = [
            FactCheck(
                claim="잘못된 정보",
                status=FactCheckStatus.FAIL,
                source="규정 제1조",
                confidence=0.3,
                correction="정보가 틀렸습니다",
            ),
        ]

        test_result = TestResult(
            test_case_id="complex_002",
            query="질문",
            answer="부정확한 답변",
            sources=[],  # No sources
            confidence=0.25,
            execution_time_ms=1000,
            rag_pipeline_log={},  # Minimal execution
            fact_checks=fact_checks,
            quality_score=quality_score,
            passed=False,
        )

        analysis = analyzer.analyze_components(test_result)

        # Check that failure causes are identified
        assert len(analysis.failure_cause_components) >= 2

        # Hybrid search should be a failure cause (no sources)
        failure_names = [c.value for c in analysis.failure_cause_components]
        assert "hybrid_search" in failure_names

        # Fact check should be a failure cause (failed)
        assert "fact_check" in failure_names

    def test_partial_pipeline_some_missing_optional(self):
        """WHEN optional components are missing, THEN should have neutral impact."""
        analyzer = ComponentAnalyzer()

        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.80,
                completeness=0.78,
                relevance=0.82,
                source_citation=0.85,
                practicality=0.75,
                actionability=0.77,
            ),
            total_score=4.77,
            is_pass=True,
        )

        test_result = TestResult(
            test_case_id="complex_003",
            query="질문",
            answer="답변",
            sources=["규정 제1조", "규정 제2조"],
            confidence=0.78,
            execution_time_ms=1500,
            rag_pipeline_log={
                "hybrid_search": True,
                "fact_check": True,
                # Missing: self_rag, hyde, corrective, reranker, expansion
            },
            quality_score=quality_score,
            passed=True,
        )

        analysis = analyzer.analyze_components(test_result)

        # Check optional components have score 0
        self_rag_contrib = next(
            (c for c in analysis.contributions if c.component == RAGComponent.SELF_RAG),
            None,
        )
        assert self_rag_contrib is not None
        assert self_rag_contrib.score == 0
        assert not self_rag_contrib.was_executed

        hyde_contrib = next(
            (c for c in analysis.contributions if c.component == RAGComponent.HYDE),
            None,
        )
        assert hyde_contrib is not None
        assert hyde_contrib.score == 0
        assert not hyde_contrib.was_executed
