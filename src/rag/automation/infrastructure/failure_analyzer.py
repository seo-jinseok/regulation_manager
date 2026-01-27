"""
Failure Analyzer with 5-Why Analysis.

Infrastructure service for automated root cause analysis of test failures
using the 5-Why technique.

Clean Architecture: Infrastructure layer uses external dependencies (LLM).
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..domain.value_objects import ComponentAnalysis, FiveWhyAnalysis

logger = logging.getLogger(__name__)


class FailureAnalyzer:
    """
    Infrastructure service for analyzing test failures.

    Performs automated 5-Why root cause analysis and generates
    actionable improvement suggestions.
    """

    def __init__(self, llm_client: Optional[object] = None):
        """
        Initialize the failure analyzer.

        Args:
            llm_client: Optional LLM client for enhanced analysis.
        """
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)

    def analyze_failure(
        self,
        test_result: "QualityTestResult",
        component_analysis: Optional["ComponentAnalysis"] = None,
    ) -> "FiveWhyAnalysis":
        """
        Perform 5-Why analysis on a failed test.

        Args:
            test_result: Failed test result.
            component_analysis: Optional component analysis for context.

        Returns:
            FiveWhyAnalysis with root cause and suggested fix.
        """
        from ..domain.value_objects import FiveWhyAnalysis

        if test_result.passed:
            self.logger.warning(
                f"Test {test_result.test_case_id} passed, no failure to analyze"
            )
            return FiveWhyAnalysis(
                test_case_id=test_result.test_case_id,
                original_failure="No failure",
                why_chain=[],
                root_cause="N/A",
                suggested_fix="N/A",
            )

        self.logger.info(f"Analyzing failure for test: {test_result.test_case_id}")

        # Determine failure type
        failure_type = self._classify_failure(test_result, component_analysis)

        # Perform 5-Why chain
        why_chain = self._perform_five_why(
            test_result, failure_type, component_analysis
        )

        # Extract root cause (last why)
        root_cause = why_chain[-1] if why_chain else "Unknown"

        # Generate suggested fix
        suggested_fix = self._generate_suggested_fix(
            root_cause, failure_type, test_result
        )

        # Determine what to patch
        component_to_patch = self._determine_patch_target(root_cause, failure_type)

        # Check if code change is required
        code_change_required = self._requires_code_change(root_cause, failure_type)

        analysis = FiveWhyAnalysis(
            test_case_id=test_result.test_case_id,
            original_failure=failure_type,
            why_chain=why_chain,
            root_cause=root_cause,
            suggested_fix=suggested_fix,
            component_to_patch=component_to_patch,
            code_change_required=code_change_required,
        )

        self.logger.info(f"5-Why analysis complete: {analysis.analysis_depth} levels")

        return analysis

    def _classify_failure(
        self,
        test_result: "QualityTestResult",
        component_analysis: Optional["ComponentAnalysis"],
    ) -> str:
        """
        Classify the type of failure.

        Args:
            test_result: Failed test result.
            component_analysis: Optional component analysis.

        Returns:
            Failure type description.
        """
        # Check fact check failures
        if test_result.fact_checks:
            failed_checks = [
                fc for fc in test_result.fact_checks if fc.status.value != "pass"
            ]
            if failed_checks:
                return f"Fact check failure: {failed_checks[0].claim}"

        # Check quality score
        if test_result.quality_score:
            dimensions = test_result.quality_score.dimensions
            issues = []

            if dimensions.accuracy < 0.7:
                issues.append("low_accuracy")
            if dimensions.completeness < 0.7:
                issues.append("incomplete")
            if dimensions.relevance < 0.7:
                issues.append("low_relevance")
            if dimensions.source_citation < 0.7:
                issues.append("poor_citation")

            if issues:
                return f"Quality failure: {', '.join(issues)}"

        # Check component analysis
        if component_analysis and component_analysis.critical_failures:
            critical_components = [
                cf.component.value for cf in component_analysis.critical_failures
            ]
            return f"Component failure: {', '.join(critical_components)}"

        # Check for errors
        if test_result.error_message:
            return f"Execution error: {test_result.error_message}"

        # Check confidence
        if test_result.confidence < 0.5:
            return f"Low confidence: {test_result.confidence:.2f}"

        # Default
        return "Unknown failure type"

    def _perform_five_why(
        self,
        test_result: "QualityTestResult",
        failure_type: str,
        component_analysis: Optional["ComponentAnalysis"],
    ) -> List[str]:
        """
        Perform the 5-Why analysis chain.

        Args:
            test_result: Failed test result.
            failure_type: Type of failure.
            component_analysis: Optional component analysis.

        Returns:
            List of 5 "why" answers (or fewer if root cause found earlier).
        """
        why_chain = []

        # Why 1: What happened?
        why_1 = f"Test failed: {failure_type}"
        why_chain.append(why_1)

        # Why 2: Why did it happen?
        why_2 = self._get_second_why(test_result, failure_type, component_analysis)
        why_chain.append(why_2)

        # Why 3: Why did that happen?
        why_3 = self._get_third_why(why_2, component_analysis)
        why_chain.append(why_3)

        # Why 4: Why did that happen?
        why_4 = self._get_fourth_why(why_3)
        why_chain.append(why_4)

        # Why 5: Why did that happen? (Root cause)
        why_5 = self._get_fifth_why(why_4)
        why_chain.append(why_5)

        return why_chain

    def _get_second_why(
        self,
        test_result: "QualityTestResult",
        failure_type: str,
        component_analysis: Optional["ComponentAnalysis"],
    ) -> str:
        """Get the second why."""
        if "low_relevance" in failure_type:
            return "Retrieved sources did not match query intent"
        elif "incomplete" in failure_type:
            return "Answer did not address all aspects of the question"
        elif "low_accuracy" in failure_type:
            return "Generated answer contained incorrect information"
        elif "Fact check" in failure_type:
            return "Answer made claims not supported by regulations"
        elif component_analysis and component_analysis.critical_failures:
            failed_comp = component_analysis.critical_failures[0]
            return f"RAG component {failed_comp.component.value} failed: {failed_comp.reason}"
        else:
            return "RAG system did not process the query correctly"

    def _get_third_why(
        self, second_why: str, component_analysis: Optional["ComponentAnalysis"]
    ) -> str:
        """Get the third why."""
        if "intent" in second_why.lower():
            return "Query analysis did not correctly identify user intent"
        elif "aspect" in second_why.lower():
            return "Query expansion did not cover all relevant terms"
        elif "incorrect" in second_why.lower():
            return "LLM generation was not grounded in retrieved sources"
        elif "component" in second_why.lower():
            return "Component configuration or parameters are suboptimal"
        else:
            return "Insufficient or irrelevant context was retrieved"

    def _get_fourth_why(self, third_why: str) -> str:
        """Get the fourth why."""
        if "intent" in third_why.lower():
            return "Intent training data or patterns do not cover this query type"
        elif "expansion" in third_why.lower():
            return "Query expansion patterns are incomplete for this domain"
        elif "grounded" in third_why.lower():
            return "LLM prompt does not enforce source-based generation"
        elif "configuration" in third_why.lower():
            return "Component parameters not tuned for current data distribution"
        else:
            return "Retrieval embeddings or indexing strategy need improvement"

    def _get_fifth_why(self, fourth_why: str) -> str:
        """Get the fifth why (root cause)."""
        if "training data" in fourth_why.lower() or "patterns" in fourth_why.lower():
            return "intents.json and synonyms.json need updates for this query pattern"
        elif "prompt" in fourth_why.lower():
            return "LLM prompt engineering needs improvement"
        elif "parameters" in fourth_why.lower():
            return "Component hyperparameters require retuning"
        else:
            return "Knowledge base or embeddings need re-indexing"

    def _generate_suggested_fix(
        self,
        root_cause: str,
        failure_type: str,
        test_result: "QualityTestResult",
    ) -> str:
        """
        Generate suggested fix based on root cause.

        Args:
            root_cause: Identified root cause.
            failure_type: Type of failure.
            test_result: Failed test result.

        Returns:
            Suggested fix description.
        """
        if "intents.json" in root_cause or "synonyms.json" in root_cause:
            return f"Add query pattern '{test_result.query[:50]}...' to intents.json with appropriate intent"

        elif "prompt" in root_cause:
            return "Review and update LLM generation prompt to enforce source-based answers"

        elif "parameters" in root_cause:
            return (
                "Retune RAG component parameters (e.g., top_k, temperature, thresholds)"
            )

        elif "indexing" in root_cause:
            return "Re-index knowledge base with updated embeddings"

        else:
            return f"Investigate {root_cause} and implement appropriate fix"

    def _determine_patch_target(
        self, root_cause: str, failure_type: str
    ) -> Optional[str]:
        """
        Determine which file/component needs patching.

        Args:
            root_cause: Identified root cause.
            failure_type: Type of failure.

        Returns:
            File/component to patch, or None if not applicable.
        """
        if "intents.json" in root_cause:
            return "intents.json"
        elif "synonyms.json" in root_cause:
            return "synonyms.json"
        elif "prompt" in root_cause:
            return "llm_prompt"
        elif "parameters" in root_cause:
            return "config"
        else:
            return None

    def _requires_code_change(self, root_cause: str, failure_type: str) -> bool:
        """
        Determine if the fix requires code changes.

        Args:
            root_cause: Identified root cause.
            failure_type: Type of failure.

        Returns:
            True if code change is required, False otherwise.
        """
        # JSON file updates are not code changes
        if any(
            json_file in root_cause for json_file in ["intents.json", "synonyms.json"]
        ):
            return False

        # Config changes are not code changes
        if "config" in root_cause or "parameters" in root_cause:
            return False

        # Prompt changes are borderline, consider as not code
        if "prompt" in root_cause:
            return False

        # Re-indexing is not a code change
        if "indexing" in root_cause:
            return False

        # Everything else likely requires code changes
        return True

    def generate_patch_suggestion(
        self,
        analysis: "FiveWhyAnalysis",
        test_result: "QualityTestResult",
    ) -> Dict[str, any]:
        """
        Generate a concrete patch suggestion.

        Args:
            analysis: Five-Why analysis result.
            test_result: Original failed test result.

        Returns:
            Dictionary with patch details.
        """
        patch = {
            "target": analysis.component_to_patch,
            "requires_code_change": analysis.code_change_required,
            "suggested_fix": analysis.suggested_fix,
            "query": test_result.query,
        }

        if analysis.component_to_patch == "intents.json":
            patch["patch_content"] = self._generate_intents_patch(test_result.query)
        elif analysis.component_to_patch == "synonyms.json":
            patch["patch_content"] = self._generate_synonyms_patch(test_result.query)

        return patch

    def _generate_intents_patch(self, query: str) -> Dict:
        """Generate intents.json patch content."""
        # Simple extraction (can be enhanced with NLP)
        keywords = query.split()[:5]  # First 5 words as keywords

        return {
            "intent": "extracted_from_query",
            "patterns": [query],
            "keywords": keywords,
            "examples": [query],
        }

    def _generate_synonyms_patch(self, query: str) -> Dict:
        """Generate synonyms.json patch content."""
        keywords = query.split()[:3]  # First 3 words

        return {
            "term": keywords[0] if keywords else "",
            "synonyms": keywords[1:],
            "context": "regulation_query",
        }
