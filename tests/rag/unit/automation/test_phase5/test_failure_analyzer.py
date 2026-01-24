"""
Specification tests for FailureAnalyzer (Phase 5).

Tests define expected behavior for 5-Why root cause analysis.
Following DDD PRESERVE phase: specification tests for greenfield development.
"""

import pytest

from src.rag.automation.domain.entities import TestResult
from src.rag.automation.domain.value_objects import (
    FactCheck,
    FactCheckStatus,
    QualityDimensions,
    QualityScore,
)
from src.rag.automation.infrastructure.failure_analyzer import FailureAnalyzer


class TestFailureClassification:
    """Specification tests for failure type classification."""

    def test_classify_failure_for_fact_check_failure(self):
        """
        SPEC: _classify_failure should identify fact check failures.

        Given: A test result with failed fact checks
        When: _classify_failure is called
        Then: Returns failure type containing "Fact check failure"
        """
        # Arrange
        analyzer = FailureAnalyzer()

        fact_checks = [
            FactCheck(
                claim="잘못된 주장",
                status=FactCheckStatus.FAIL,
                source="규정 제1조",
                confidence=0.3,
                correction="올바른 정보",
            )
        ]

        test_result = TestResult(
            test_case_id="test_001",
            query="질문",
            answer="답변",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            fact_checks=fact_checks,
            passed=False,
        )

        # Act
        failure_type = analyzer._classify_failure(test_result, None)

        # Assert
        assert "Fact check failure" in failure_type

    def test_classify_failure_for_low_quality(self):
        """
        SPEC: _classify_failure should identify quality failures.

        Given: A test result with low quality scores
        When: _classify_failure is called
        Then: Returns failure type containing "Quality failure"
        """
        # Arrange
        analyzer = FailureAnalyzer()

        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.5,  # Low accuracy
                completeness=0.6,
                relevance=0.5,  # Low relevance
                source_citation=0.4,
                practicality=0.3,
                actionability=0.3,
            ),
            total_score=2.6,
            is_pass=False,
        )

        test_result = TestResult(
            test_case_id="test_002",
            query="질문",
            answer="답변",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            quality_score=quality_score,
            passed=False,
        )

        # Act
        failure_type = analyzer._classify_failure(test_result, None)

        # Assert
        assert "Quality failure" in failure_type
        assert "low_accuracy" in failure_type or "low_relevance" in failure_type

    def test_classify_failure_for_execution_error(self):
        """
        SPEC: _classify_failure should identify execution errors.

        Given: A test result with error message
        When: _classify_failure is called
        Then: Returns failure type containing "Execution error"
        """
        # Arrange
        analyzer = FailureAnalyzer()

        test_result = TestResult(
            test_case_id="test_003",
            query="질문",
            answer="",
            sources=[],
            confidence=0.0,
            execution_time_ms=100,
            rag_pipeline_log={},
            error_message="Connection timeout",
            passed=False,
        )

        # Act
        failure_type = analyzer._classify_failure(test_result, None)

        # Assert
        assert "Execution error" in failure_type


class TestFiveWhyChain:
    """Specification tests for 5-Why chain generation."""

    @pytest.fixture
    def failed_test_result(self):
        """Create a failed test result for 5-Why analysis."""
        fact_checks = [
            FactCheck(
                claim="휴학 기간은 1년이다",
                status=FactCheckStatus.FAIL,
                source="규정 제10조",
                confidence=0.4,
                correction="휴학 기간은 최대 2학기까지 가능",
            )
        ]

        return TestResult(
            test_case_id="test_004",
            query="휴학 기간이 얼마나 되나요?",
            answer="휴학 기간은 1년입니다",
            sources=["규정 제10조"],
            confidence=0.7,
            execution_time_ms=1200,
            rag_pipeline_log={"fact_check": True},
            fact_checks=fact_checks,
            passed=False,
        )

    def test_perform_five_why_creates_five_level_chain(self, failed_test_result):
        """
        SPEC: _perform_five_why should create a chain of 5 levels.

        Given: A failed test result
        When: _perform_five_why is called
        Then: Returns a list with exactly 5 elements
        """
        # Arrange
        analyzer = FailureAnalyzer()
        failure_type = "Fact check failure: 휴학 기간은 1년이다"

        # Act
        why_chain = analyzer._perform_five_why(
            failed_test_result,
            failure_type,
            None,
        )

        # Assert
        assert len(why_chain) == 5

    def test_perform_five_why_first_why_is_what_happened(self, failed_test_result):
        """
        SPEC: _perform_five_why should start with "what happened".

        Given: A failed test result
        When: _perform_five_why is called
        Then: First element describes the failure
        """
        # Arrange
        analyzer = FailureAnalyzer()
        failure_type = "Fact check failure: 휴학 기간은 1년이다"

        # Act
        why_chain = analyzer._perform_five_why(
            failed_test_result,
            failure_type,
            None,
        )

        # Assert
        assert "Test failed" in why_chain[0] or failure_type in why_chain[0]

    def test_perform_five_why_last_why_is_root_cause(self, failed_test_result):
        """
        SPEC: _perform_five_why should end with root cause.

        Given: A failed test result
        When: _perform_five_why is called
        Then: Last element identifies root cause (e.g., intents.json, prompt, etc.)
        """
        # Arrange
        analyzer = FailureAnalyzer()
        failure_type = "Fact check failure: 휴학 기간은 1년이다"

        # Act
        why_chain = analyzer._perform_five_why(
            failed_test_result,
            failure_type,
            None,
        )

        # Assert
        root_cause = why_chain[-1]
        # Root cause should mention actionable items
        assert any(
            keyword in root_cause.lower()
            for keyword in ["intents.json", "prompt", "parameters", "indexing", "knowledge base"]
        )


class TestRootCauseIdentification:
    """Specification tests for root cause identification."""

    def test_determine_patch_target_for_intent_issues(self):
        """
        SPEC: _determine_patch_target should return intents.json for intent issues.

        Given: Root cause mentions intent or pattern issues
        When: _determine_patch_target is called
        Then: Returns "intents.json"
        """
        # Arrange
        analyzer = FailureAnalyzer()
        root_cause = "intents.json and synonyms.json need updates for this query pattern"
        failure_type = "Quality failure: low_relevance"

        # Act
        target = analyzer._determine_patch_target(root_cause, failure_type)

        # Assert
        assert target == "intents.json"

    def test_determine_patch_target_for_prompt_issues(self):
        """
        SPEC: _determine_patch_target should return llm_prompt for prompt issues.

        Given: Root cause mentions prompt issues
        When: _determine_patch_target is called
        Then: Returns "llm_prompt"
        """
        # Arrange
        analyzer = FailureAnalyzer()
        root_cause = "LLM prompt engineering needs improvement"
        failure_type = "Quality failure: low_accuracy"

        # Act
        target = analyzer._determine_patch_target(root_cause, failure_type)

        # Assert
        assert target == "llm_prompt"

    def test_requires_code_change_for_json_patches(self):
        """
        SPEC: _requires_code_change should return False for JSON file patches.

        Given: Root cause that requires intents.json or synonyms.json update
        When: _requires_code_change is called
        Then: Returns False (JSON updates are not code changes)
        """
        # Arrange
        analyzer = FailureAnalyzer()
        root_cause = "intents.json and synonyms.json need updates"
        failure_type = "Quality failure"

        # Act
        requires_code = analyzer._requires_code_change(root_cause, failure_type)

        # Assert
        assert requires_code is False

    def test_requires_code_change_for_prompt_changes(self):
        """
        SPEC: _requires_code_change should return False for prompt changes.

        Given: Root cause that requires prompt update
        When: _requires_code_change is called
        Then: Returns False (prompt changes are not code changes)
        """
        # Arrange
        analyzer = FailureAnalyzer()
        root_cause = "LLM prompt engineering needs improvement"
        failure_type = "Quality failure"

        # Act
        requires_code = analyzer._requires_code_change(root_cause, failure_type)

        # Assert
        assert requires_code is False

    def test_requires_code_change_for_other_issues(self):
        """
        SPEC: _requires_code_change should return True for other issues.

        Given: Root cause that doesn't match known non-code patterns
        When: _requires_code_change is called
        Then: Returns True (likely requires code change)
        """
        # Arrange
        analyzer = FailureAnalyzer()
        root_cause = "Algorithm logic error in retrieval component"
        failure_type = "Execution error"

        # Act
        requires_code = analyzer._requires_code_change(root_cause, failure_type)

        # Assert
        assert requires_code is True


class TestCompleteAnalysis:
    """Specification tests for complete 5-Why analysis."""

    @pytest.fixture
    def failed_result(self):
        """Create a failed test result."""
        fact_checks = [
            FactCheck(
                claim="잘못된 정보",
                status=FactCheckStatus.FAIL,
                source="규정 제1조",
                confidence=0.3,
                correction="올바른 정보",
            )
        ]

        return TestResult(
            test_case_id="test_005",
            query="질문",
            answer="잘못된 답변",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={"fact_check": True},
            fact_checks=fact_checks,
            passed=False,
        )

    def test_analyze_failure_creates_complete_analysis(self, failed_result):
        """
        SPEC: analyze_failure should create complete FiveWhyAnalysis.

        Given: A failed test result
        When: analyze_failure is called
        Then: Returns FiveWhyAnalysis with all fields populated
        """
        # Arrange
        analyzer = FailureAnalyzer()

        # Act
        analysis = analyzer.analyze_failure(failed_result)

        # Assert
        assert analysis.test_case_id == "test_005"
        assert analysis.original_failure != "No failure"
        assert len(analysis.why_chain) == 5
        assert analysis.root_cause != "Unknown"
        assert analysis.suggested_fix != "N/A"

    def test_analyze_failure_returns_no_failure_for_passed_tests(self):
        """
        SPEC: analyze_failure should return analysis with no failure for passed tests.

        Given: A test result that passed
        When: analyze_failure is called
        Then: Returns FiveWhyAnalysis with "No failure"
        """
        # Arrange
        analyzer = FailureAnalyzer()

        passed_result = TestResult(
            test_case_id="test_006",
            query="질문",
            answer="답변",
            sources=["규정 제1조"],
            confidence=0.9,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=True,
        )

        # Act
        analysis = analyzer.analyze_failure(passed_result)

        # Assert
        assert analysis.original_failure == "No failure"
        assert analysis.root_cause == "N/A"
        assert analysis.suggested_fix == "N/A"


class TestPatchGeneration:
    """Specification tests for patch suggestion generation."""

    def test_generate_intents_patch_creates_valid_structure(self):
        """
        SPEC: _generate_intents_patch should create valid intent entry structure.

        Given: A query string
        When: _generate_intents_patch is called
        Then: Returns dict with intent, patterns, keywords, and examples
        """
        # Arrange
        analyzer = FailureAnalyzer()
        query = "휴학 신청 기간이 언제인가요?"

        # Act
        patch = analyzer._generate_intents_patch(query)

        # Assert
        assert "intent" in patch
        assert "patterns" in patch
        assert "keywords" in patch
        assert "examples" in patch
        assert query in patch["patterns"]
        assert query in patch["examples"]

    def test_generate_synonyms_patch_creates_valid_structure(self):
        """
        SPEC: _generate_synonyms_patch should create valid synonym entry structure.

        Given: A query string
        When: _generate_synonyms_patch is called
        Then: Returns dict with term, synonyms, and context
        """
        # Arrange
        analyzer = FailureAnalyzer()
        query = "휴학 신청 방법"

        # Act
        patch = analyzer._generate_synonyms_patch(query)

        # Assert
        assert "term" in patch
        assert "synonyms" in patch
        assert "context" in patch
        assert patch["context"] == "regulation_query"
