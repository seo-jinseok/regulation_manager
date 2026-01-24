"""
RAG Component Analyzer.

Infrastructure service for analyzing RAG component behavior
and attributing contributions to test results.

Clean Architecture: Infrastructure layer analyzes system behavior.
"""

import logging
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from ..domain.entities import TestResult
    from ..domain.value_objects import (
        ComponentAnalysis,
        ComponentContribution,
        RAGComponent,
    )

logger = logging.getLogger(__name__)


class ComponentAnalyzer:
    """
    Infrastructure service for analyzing RAG component behavior.

    Tracks which RAG components were executed during a test and
    evaluates their contribution to the test outcome.
    """

    # Component detection patterns in RAG pipeline logs
    COMPONENT_PATTERNS = {
        "self_rag": ["self_rag", "reflection", "retrieve_feedback"],
        "hyde": ["hyde", "hypothetical", "document_embedding"],
        "corrective_rag": ["corrective", "retrieval_evaluator", "is_relevant"],
        "hybrid_search": ["hybrid_search", "dense_retrieval", "sparse_retrieval"],
        "bge_reranker": ["reranker", "bge", "rerank"],
        "query_analyzer": ["query_analyzer", "intent_analysis"],
        "dynamic_query_expansion": ["query_expansion", "expanded_query"],
        "fact_check": ["fact_check", "fact_checker", "verify"],
    }

    # Failure type to component mapping
    FAILURE_TYPE_TO_COMPONENT = {
        "low_relevance": "hybrid_search",
        "irrelevant_sources": "corrective_rag",
        "missing_information": "dynamic_query_expansion",
        "incorrect_answer": "fact_check",
        "poor_ranking": "bge_reranker",
        "misunderstood_intent": "query_analyzer",
        "redundant_retrieval": "self_rag",
        "generic_query": "hyde",
    }

    def __init__(self):
        """Initialize the component analyzer."""
        self.logger = logging.getLogger(__name__)

    def analyze_components(
        self,
        test_result: "TestResult",
    ) -> "ComponentAnalysis":
        """
        Analyze which RAG components contributed to the test result.

        Args:
            test_result: Test result with RAG pipeline logs.

        Returns:
            ComponentAnalysis with contribution scores for each component.
        """
        from ..domain.value_objects import ComponentContribution, RAGComponent

        self.logger.info(f"Analyzing components for test: {test_result.test_case_id}")

        # Detect which components were executed
        executed_components = self._detect_executed_components(
            test_result.rag_pipeline_log
        )

        # Evaluate each component's contribution
        contributions: List[ComponentContribution] = []

        for component in RAGComponent:
            was_executed = component.value in executed_components

            if was_executed:
                score, reason = self._evaluate_component_contribution(
                    component=component,
                    test_result=test_result,
                )
            else:
                # Component not executed, score based on whether it should have been
                score, reason = self._evaluate_missing_component(
                    component=component,
                    test_result=test_result,
                )

            contribution = ComponentContribution(
                component=component,
                score=score,
                reason=reason,
                was_executed=was_executed,
            )
            contributions.append(contribution)

        # Identify failure cause components
        failure_cause_components = [c.component for c in contributions if c.score <= -1]

        # Determine overall impact
        overall_impact = self._determine_overall_impact(contributions)

        # Check if timing was a factor
        timestamp_importance = self._check_timing_importance(test_result)

        # Create analysis
        from ..domain.value_objects import ComponentAnalysis

        analysis = ComponentAnalysis(
            test_case_id=test_result.test_case_id,
            contributions=contributions,
            overall_impact=overall_impact,
            failure_cause_components=failure_cause_components,
            timestamp_importance=timestamp_importance,
        )

        self.logger.info(
            f"Component analysis complete: {analysis.net_impact_score} net impact, "
            f"{len(failure_cause_components)} failure causes"
        )

        return analysis

    def _detect_executed_components(self, pipeline_log: Dict) -> List[str]:
        """
        Detect which RAG components were executed based on pipeline logs.

        Args:
            pipeline_log: RAG pipeline execution log.

        Returns:
            List of component names that were executed.
        """
        executed = []

        # Convert log to string for pattern matching
        log_str = str(pipeline_log).lower()

        for component_name, patterns in self.COMPONENT_PATTERNS.items():
            if any(pattern in log_str for pattern in patterns):
                executed.append(component_name)

        self.logger.debug(f"Detected executed components: {executed}")

        return executed

    def _evaluate_component_contribution(
        self,
        component: "RAGComponent",
        test_result: "TestResult",
    ) -> tuple[int, str]:
        """
        Evaluate the contribution of an executed component.

        Args:
            component: RAG component to evaluate.
            test_result: Test result to evaluate against.

        Returns:
            Tuple of (score, reason).
        """
        from ..domain.value_objects import RAGComponent

        score = 0
        reasons = []

        # Component-specific evaluation
        if component == RAGComponent.SELF_RAG:
            # Check if reflection improved quality
            if (
                test_result.quality_score
                and test_result.quality_score.dimensions.accuracy > 0.8
            ):
                score = 1
                reasons.append("Self-RAG reflection improved answer accuracy")
            else:
                score = 0
                reasons.append("Self-RAG executed but unclear impact")

        elif component == RAGComponent.HYDE:
            # Check if hypothetical embeddings helped retrieval
            if test_result.sources and len(test_result.sources) >= 3:
                score = 1
                reasons.append("HyDE contributed to diverse source retrieval")
            else:
                score = 0
                reasons.append("HyDE executed but limited source diversity")

        elif component == RAGComponent.CORRECTIVE_RAG:
            # Check if corrective filtering prevented irrelevant sources
            log_str = str(test_result.rag_pipeline_log).lower()
            if "filtered" in log_str or "is_relevant" in log_str:
                score = 1
                reasons.append("Corrective RAG filtered irrelevant content")
            else:
                score = 0
                reasons.append("Corrective RAG executed without filtering")

        elif component == RAGComponent.HYBRID_SEARCH:
            # Check if hybrid search provided good coverage
            if test_result.sources and len(test_result.sources) >= 4:
                score = 2
                reasons.append("Hybrid search provided excellent source coverage")
            elif test_result.sources:
                score = 1
                reasons.append("Hybrid search provided adequate coverage")
            else:
                score = -1
                reasons.append("Hybrid search failed to retrieve sources")

        elif component == RAGComponent.BGE_RERANKER:
            # Check if reranking improved relevance
            if test_result.confidence > 0.7:
                score = 1
                reasons.append("Reranking improved relevance confidence")
            else:
                score = 0
                reasons.append("Reranking executed but confidence low")

        elif component == RAGComponent.QUERY_ANALYZER:
            # Check if query analysis matched intent
            if (
                test_result.quality_score
                and test_result.quality_score.dimensions.relevance > 0.7
            ):
                score = 1
                reasons.append("Query analysis improved relevance")
            else:
                score = 0
                reasons.append("Query analysis executed but relevance low")

        elif component == RAGComponent.DYNAMIC_QUERY_EXPANSION:
            # Check if expansion helped retrieve more relevant info
            log_str = str(test_result.rag_pipeline_log).lower()
            if "expanded_query" in log_str:
                score = 1
                reasons.append("Query expansion improved retrieval")
            else:
                score = 0
                reasons.append("Query expansion executed without clear benefit")

        elif component == RAGComponent.FACT_CHECK:
            # Check if fact checking caught errors
            if test_result.fact_checks:
                passed_checks = sum(
                    1 for fc in test_result.fact_checks if fc.status.value == "pass"
                )
                if passed_checks == len(test_result.fact_checks):
                    score = 2
                    reasons.append(
                        f"All {len(test_result.fact_checks)} fact checks passed"
                    )
                elif passed_checks > len(test_result.fact_checks) / 2:
                    score = 1
                    reasons.append(
                        f"Most fact checks passed ({passed_checks}/{len(test_result.fact_checks)})"
                    )
                else:
                    score = -1
                    reasons.append(
                        f"Fact checks found errors ({passed_checks}/{len(test_result.fact_checks)} passed)"
                    )
            else:
                score = 0
                reasons.append("Fact checking executed but no checks recorded")

        return score, " ".join(reasons)

    def _evaluate_missing_component(
        self,
        component: "RAGComponent",
        test_result: "TestResult",
    ) -> tuple[int, str]:
        """
        Evaluate the impact of a component not being executed.

        Args:
            component: RAG component that was not executed.
            test_result: Test result to evaluate against.

        Returns:
            Tuple of (score, reason).
        """
        from ..domain.value_objects import RAGComponent

        # Check if the component should have been executed
        score = 0
        reason = f"{component.value} not executed"

        # Critical components that should always run
        if component == RAGComponent.HYBRID_SEARCH:
            if not test_result.sources:
                score = -2
                reason = "Hybrid search missing, no sources retrieved"
            else:
                score = -1
                reason = "Hybrid search not executed but sources found (fallback?)"

        elif component == RAGComponent.FACT_CHECK:
            if test_result.passed is False:
                score = -1
                reason = "Fact check missing, test failed without verification"

        # Optional components
        elif component in [
            RAGComponent.SELF_RAG,
            RAGComponent.HYDE,
            RAGComponent.BGE_RERANKER,
        ]:
            score = 0
            reason = f"{component.value} not executed (optional component)"

        else:
            score = 0
            reason = f"{component.value} not executed"

        return score, reason

    def _determine_overall_impact(
        self, contributions: List["ComponentContribution"]
    ) -> str:
        """
        Determine overall impact assessment from contributions.

        Args:
            contributions: List of component contributions.

        Returns:
            Overall impact description.
        """
        total_score = sum(c.score for c in contributions)
        negative_count = sum(1 for c in contributions if c.score < 0)

        if total_score >= 4:
            return "Strong positive impact from multiple components"
        elif total_score >= 2:
            return "Moderate positive impact from components"
        elif total_score >= 0:
            return "Neutral impact, components executed without clear benefit"
        elif negative_count >= 2:
            return "Multiple components negatively impacted result"
        else:
            return "Some components negatively impacted result"

    def _check_timing_importance(self, test_result: "TestResult") -> bool:
        """
        Check if execution time was a factor in the result.

        Args:
            test_result: Test result to evaluate.

        Returns:
            True if timing was a factor, False otherwise.
        """
        # Consider timing important if execution was very slow
        if test_result.execution_time_ms > 5000:  # 5 seconds
            return True

        # Consider timing important if there were timeout indicators
        log_str = str(test_result.rag_pipeline_log).lower()
        if "timeout" in log_str or "slow" in log_str:
            return True

        return False

    def map_failure_to_components(
        self,
        failure_type: str,
    ) -> List["RAGComponent"]:
        """
        Map a failure type to likely responsible components.

        Args:
            failure_type: Type of failure (e.g., "low_relevance").

        Returns:
            List of RAG components that likely caused the failure.
        """
        from ..domain.value_objects import RAGComponent

        component_name = self.FAILURE_TYPE_TO_COMPONENT.get(failure_type)

        if component_name:
            try:
                return [RAGComponent(component_name)]
            except ValueError:
                pass

        # Default to common components if no specific mapping
        return [RAGComponent.HYBRID_SEARCH, RAGComponent.QUERY_ANALYZER]

    def get_component_suggestions(
        self,
        analysis: "ComponentAnalysis",
    ) -> List[str]:
        """
        Generate suggestions for improving RAG component performance.

        Args:
            analysis: Component analysis result.

        Returns:
            List of improvement suggestions.
        """
        suggestions = []

        for contribution in analysis.contributions:
            if contribution.score <= -1:
                # Component is causing issues
                suggestions.append(
                    f"Improve {contribution.component.value}: {contribution.reason}"
                )

            elif contribution.score == 0 and contribution.was_executed:
                # Component executed but no clear benefit
                suggestions.append(
                    f"Review {contribution.component.value} configuration: {contribution.reason}"
                )

        if analysis.timestamp_importance:
            suggestions.append("Consider optimizing component execution time")

        return suggestions
