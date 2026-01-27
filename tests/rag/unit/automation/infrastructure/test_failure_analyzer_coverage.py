"""
Coverage tests for FailureAnalyzer to reach 80%+ coverage.

Tests uncovered code paths in:
- Suggested fix generation for various root causes
- Patch target determination for various scenarios
- 5-Why chain generation for all failure types
- generate_patch_suggestion for various scenarios
- _classify_failure for low confidence cases
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
from src.rag.automation.infrastructure.failure_analyzer import FailureAnalyzer


class TestClassifyFailureLowConfidence:
    """Tests for _classify_failure with low confidence scenarios."""

    def test_classify_failure_low_confidence(self):
        """
        SPEC: _classify_failure should identify low confidence failures.
        """
        # Arrange
        analyzer = FailureAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test_001",
            query="질문",
            answer="답변",
            sources=[],
            confidence=0.3,  # Below 0.5 threshold
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        # Act
        failure_type = analyzer._classify_failure(test_result, None)

        # Assert
        assert "Low confidence" in failure_type
        assert "0.30" in failure_type

    def test_classify_failure_confidence_exactly_threshold(self):
        """
        SPEC: _classify_failure should handle confidence at exact threshold (0.5).
        """
        # Arrange
        analyzer = FailureAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test_002",
            query="질문",
            answer="답변",
            sources=[],
            confidence=0.5,  # Exactly at threshold
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        # Act
        failure_type = analyzer._classify_failure(test_result, None)

        # Assert - Should NOT be classified as low confidence at 0.5
        assert "Low confidence" not in failure_type


class TestClassifyFailureComponentFailures:
    """Tests for _classify_failure with component analysis."""

    def test_classify_failure_with_component_critical_failures(self):
        """
        SPEC: _classify_failure should identify component failures.
        """
        # Arrange
        analyzer = FailureAnalyzer()

        component_analysis = ComponentAnalysis(
            test_case_id="test_003",
            contributions=[
                ComponentContribution(
                    component=RAGComponent.HYBRID_SEARCH,
                    score=-2,
                    reason="Failed",
                    was_executed=True,
                )
            ],
            overall_impact="Negative",
            failure_cause_components=[RAGComponent.HYBRID_SEARCH],
            timestamp_importance=False,
        )

        test_result = QualityTestResult(
            test_case_id="test_003",
            query="질문",
            answer="답변",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        # Act
        failure_type = analyzer._classify_failure(test_result, component_analysis)

        # Assert
        assert "Component failure" in failure_type
        assert "hybrid_search" in failure_type

    def test_classify_failure_default_for_unknown(self):
        """
        SPEC: _classify_failure should return default for unknown failure type.
        """
        # Arrange
        analyzer = FailureAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test_004",
            query="질문",
            answer="답변",
            sources=["source1"],
            confidence=0.8,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,  # Failed but no clear reason
        )

        # Act
        failure_type = analyzer._classify_failure(test_result, None)

        # Assert
        assert "Unknown failure type" in failure_type


class TestGetSecondWhyAllTypes:
    """Tests for _get_second_why with all failure types."""

    def test_get_second_why_low_relevance(self):
        """
        SPEC: _get_second_why for low_relevance should mention sources/intent.
        """
        analyzer = FailureAnalyzer()
        test_result = QualityTestResult(
            test_case_id="test",
            query="q",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer._get_second_why(
            test_result, "Quality failure: low_relevance", None
        )

        assert "intent" in result.lower() or "sources" in result.lower()

    def test_get_second_why_incomplete(self):
        """
        SPEC: _get_second_why for incomplete should mention aspects.
        """
        analyzer = FailureAnalyzer()
        test_result = QualityTestResult(
            test_case_id="test",
            query="q",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer._get_second_why(
            test_result, "Quality failure: incomplete", None
        )

        assert "aspect" in result.lower()

    def test_get_second_why_low_accuracy(self):
        """
        SPEC: _get_second_why for low_accuracy should mention incorrect info.
        """
        analyzer = FailureAnalyzer()
        test_result = QualityTestResult(
            test_case_id="test",
            query="q",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer._get_second_why(
            test_result, "Quality failure: low_accuracy", None
        )

        assert "incorrect" in result.lower()

    def test_get_second_why_poor_citation(self):
        """
        SPEC: _get_second_why for poor_citation should follow default.
        """
        analyzer = FailureAnalyzer()
        test_result = QualityTestResult(
            test_case_id="test",
            query="q",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer._get_second_why(
            test_result, "Quality failure: poor_citation", None
        )

        # Should use default processing
        assert result

    def test_get_second_why_with_component_analysis(self):
        """
        SPEC: _get_second_why with component failure should mention component.
        """
        analyzer = FailureAnalyzer()

        component_analysis = ComponentAnalysis(
            test_case_id="test",
            contributions=[
                ComponentContribution(
                    component=RAGComponent.HYBRID_SEARCH,
                    score=-2,  # Critical failure (-2) to trigger critical_failures
                    reason="Failed",
                    was_executed=True,
                )
            ],
            overall_impact="Negative",
            failure_cause_components=[RAGComponent.HYBRID_SEARCH],
            timestamp_importance=False,
        )

        test_result = QualityTestResult(
            test_case_id="test",
            query="q",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer._get_second_why(
            test_result, "Component failure: hybrid_search", component_analysis
        )

        assert "hybrid_search" in result
        assert "failed" in result.lower()

    def test_get_second_why_default_unknown_failure(self):
        """
        SPEC: _get_second_why for unknown should mention RAG processing.
        """
        analyzer = FailureAnalyzer()
        test_result = QualityTestResult(
            test_case_id="test",
            query="q",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer._get_second_why(test_result, "Unknown failure", None)

        assert "rag" in result.lower() or "process" in result.lower()


class TestGetThirdWhyAllPatterns:
    """Tests for _get_third_why with all second_why patterns."""

    def test_get_third_why_intent_pattern(self):
        """
        SPEC: _get_third_why for intent should mention query analysis.
        """
        analyzer = FailureAnalyzer()
        second_why = "Retrieved sources did not match query intent"

        result = analyzer._get_third_why(second_why, None)

        assert "intent" in result.lower() or "analysis" in result.lower()

    def test_get_third_why_aspect_pattern(self):
        """
        SPEC: _get_third_why for aspect should mention query expansion.
        """
        analyzer = FailureAnalyzer()
        second_why = "Answer did not address all aspects of the question"

        result = analyzer._get_third_why(second_why, None)

        assert "expansion" in result.lower()

    def test_get_third_why_incorrect_pattern(self):
        """
        SPEC: _get_third_why for incorrect should mention grounding.
        """
        analyzer = FailureAnalyzer()
        second_why = "Generated answer contained incorrect information"

        result = analyzer._get_third_why(second_why, None)

        assert "grounded" in result.lower() or "llm" in result.lower()

    def test_get_third_why_component_pattern(self):
        """
        SPEC: _get_third_why for component should mention configuration.
        """
        analyzer = FailureAnalyzer()
        second_why = "RAG component hybrid_search failed: Failed to retrieve"

        result = analyzer._get_third_why(second_why, None)

        assert "configuration" in result.lower() or "parameters" in result.lower()

    def test_get_third_why_default_pattern(self):
        """
        SPEC: _get_third_why for default should mention context.
        """
        analyzer = FailureAnalyzer()
        second_why = "Some other reason"

        result = analyzer._get_third_why(second_why, None)

        assert "context" in result.lower() or "retrieved" in result.lower()


class TestGetFourthWhyAllPatterns:
    """Tests for _get_fourth_why with all third_why patterns."""

    def test_get_fourth_why_intent_pattern(self):
        """
        SPEC: _get_fourth_why for intent should mention training data.
        """
        analyzer = FailureAnalyzer()
        third_why = "Query analysis did not correctly identify user intent"

        result = analyzer._get_fourth_why(third_why)

        assert "training" in result.lower() or "patterns" in result.lower()

    def test_get_fourth_why_expansion_pattern(self):
        """
        SPEC: _get_fourth_why for expansion should mention domain patterns.
        """
        analyzer = FailureAnalyzer()
        third_why = "Query expansion did not cover all relevant terms"

        result = analyzer._get_fourth_why(third_why)

        assert "domain" in result.lower() or "patterns" in result.lower()

    def test_get_fourth_why_grounded_pattern(self):
        """
        SPEC: _get_fourth_why for grounded should mention prompt.
        """
        analyzer = FailureAnalyzer()
        third_why = "LLM generation was not grounded in retrieved sources"

        result = analyzer._get_fourth_why(third_why)

        assert "prompt" in result.lower()

    def test_get_fourth_why_configuration_pattern(self):
        """
        SPEC: _get_fourth_why for configuration should mention parameters.
        """
        analyzer = FailureAnalyzer()
        third_why = "Component configuration or parameters are suboptimal"

        result = analyzer._get_fourth_why(third_why)

        assert "parameters" in result.lower() or "tuned" in result.lower()

    def test_get_fourth_why_default_pattern(self):
        """
        SPEC: _get_fourth_why for default should mention indexing.
        """
        analyzer = FailureAnalyzer()
        third_why = "Insufficient or irrelevant context was retrieved"

        result = analyzer._get_fourth_why(third_why)

        assert "indexing" in result.lower() or "embeddings" in result.lower()


class TestGetFifthWhyAllPatterns:
    """Tests for _get_fifth_why with all fourth_why patterns."""

    def test_get_fifth_why_training_data_pattern(self):
        """
        SPEC: _get_fifth_why for training data should mention JSON updates.
        """
        analyzer = FailureAnalyzer()
        fourth_why = "Intent training data or patterns do not cover this query type"

        result = analyzer._get_fifth_why(fourth_why)

        assert "intents.json" in result or "synonyms.json" in result

    def test_get_fifth_why_prompt_pattern(self):
        """
        SPEC: _get_fifth_why for prompt should mention prompt engineering.
        """
        analyzer = FailureAnalyzer()
        fourth_why = "LLM prompt does not enforce source-based generation"

        result = analyzer._get_fifth_why(fourth_why)

        assert "prompt" in result.lower()

    def test_get_fifth_why_parameters_pattern(self):
        """
        SPEC: _get_fifth_why for parameters should mention retuning.
        """
        analyzer = FailureAnalyzer()
        fourth_why = "Component parameters not tuned for current data distribution"

        result = analyzer._get_fifth_why(fourth_why)

        assert "parameters" in result.lower() or "retuning" in result.lower()

    def test_get_fifth_why_indexing_pattern(self):
        """
        SPEC: _get_fifth_why for indexing should mention re-indexing.
        """
        analyzer = FailureAnalyzer()
        fourth_why = "Retrieval embeddings or indexing strategy need improvement"

        result = analyzer._get_fifth_why(fourth_why)

        assert "indexing" in result.lower() or "embeddings" in result.lower()


class TestGenerateSuggestedFix:
    """Tests for _generate_suggested_fix with various root causes."""

    def test_generate_suggested_fix_intents_json(self):
        """
        SPEC: Suggested fix for intents.json should mention adding pattern.
        """
        analyzer = FailureAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test",
            query="휴학 신청 기간이 언제인가요?",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer._generate_suggested_fix(
            "intents.json and synonyms.json need updates",
            "Quality failure",
            test_result,
        )

        assert "intents.json" in result
        assert "pattern" in result.lower()

    def test_generate_suggested_fix_synonyms_json(self):
        """
        SPEC: Suggested fix for synonyms.json mentions intents.json (same message).
        """
        analyzer = FailureAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test",
            query="휴학 신청 방법",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer._generate_suggested_fix(
            "synonyms.json need updates for this query pattern",
            "Quality failure",
            test_result,
        )

        # Implementation returns intents.json message for both intents.json and synonyms.json
        assert "intents.json" in result

    def test_generate_suggested_fix_prompt(self):
        """
        SPEC: Suggested fix for prompt should mention prompt review.
        """
        analyzer = FailureAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test",
            query="query",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer._generate_suggested_fix(
            "LLM prompt engineering needs improvement", "Quality failure", test_result
        )

        assert "prompt" in result.lower()
        assert "source-based" in result.lower()

    def test_generate_suggested_fix_parameters(self):
        """
        SPEC: Suggested fix for parameters should mention retuning.
        """
        analyzer = FailureAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test",
            query="query",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer._generate_suggested_fix(
            "Component hyperparameters require retuning", "Quality failure", test_result
        )

        assert "retune" in result.lower()
        assert "parameters" in result.lower()

    def test_generate_suggested_fix_indexing(self):
        """
        SPEC: Suggested fix for indexing should mention re-indexing.
        """
        analyzer = FailureAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test",
            query="query",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer._generate_suggested_fix(
            "Knowledge base or embeddings need re-indexing",
            "Quality failure",
            test_result,
        )

        assert "re-index" in result.lower()

    def test_generate_suggested_fix_unknown_root_cause(self):
        """
        SPEC: Suggested fix for unknown should mention investigation.
        """
        analyzer = FailureAnalyzer()

        test_result = QualityTestResult(
            test_case_id="test",
            query="query",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer._generate_suggested_fix(
            "Unknown root cause", "Unknown failure", test_result
        )

        assert "investigate" in result.lower()


class TestDeterminePatchTarget:
    """Tests for _determine_patch_target with various root causes."""

    def test_determine_patch_target_synonyms(self):
        """
        SPEC: Should return synonyms.json for synonyms root cause.
        """
        analyzer = FailureAnalyzer()

        result = analyzer._determine_patch_target(
            "synonyms.json need updates", "Quality failure"
        )

        assert result == "synonyms.json"

    def test_determine_patch_target_config(self):
        """
        SPEC: Should return config for parameters root cause.
        """
        analyzer = FailureAnalyzer()

        result = analyzer._determine_patch_target(
            "Component parameters require retuning", "Quality failure"
        )

        assert result == "config"

    def test_determine_patch_target_unknown(self):
        """
        SPEC: Should return None for unknown root causes.
        """
        analyzer = FailureAnalyzer()

        result = analyzer._determine_patch_target(
            "Unknown algorithm issue", "Execution error"
        )

        assert result is None


class TestRequiresCodeChange:
    """Tests for _requires_code_change with various scenarios."""

    def test_requires_code_change_synonyms(self):
        """
        SPEC: Should return False for synonyms.json updates.
        """
        analyzer = FailureAnalyzer()

        result = analyzer._requires_code_change(
            "synonyms.json needs updates", "Quality failure"
        )

        assert result is False

    def test_requires_code_change_config(self):
        """
        SPEC: Should return False for config changes.
        """
        analyzer = FailureAnalyzer()

        result = analyzer._requires_code_change(
            "config parameters need tuning", "Quality failure"
        )

        assert result is False

    def test_requires_code_change_prompt(self):
        """
        SPEC: Should return False for prompt changes.
        """
        analyzer = FailureAnalyzer()

        result = analyzer._requires_code_change(
            "prompt needs improvement", "Quality failure"
        )

        assert result is False

    def test_requires_code_change_indexing(self):
        """
        SPEC: Should return False for re-indexing.
        """
        analyzer = FailureAnalyzer()

        result = analyzer._requires_code_change(
            "embeddings need re-indexing", "Quality failure"
        )

        assert result is False

    def test_requires_code_change_algorithm_issue(self):
        """
        SPEC: Should return True for algorithm/logic issues.
        """
        analyzer = FailureAnalyzer()

        result = analyzer._requires_code_change(
            "Algorithm has a bug", "Execution error"
        )

        assert result is True


class TestGeneratePatchSuggestion:
    """Tests for generate_patch_suggestion method."""

    def test_generate_patch_suggestion_intents_json(self):
        """
        SPEC: Should generate intents.json patch content.
        """
        analyzer = FailureAnalyzer()

        from src.rag.automation.domain.value_objects import FiveWhyAnalysis

        analysis = FiveWhyAnalysis(
            test_case_id="test_001",
            original_failure="Quality failure",
            why_chain=[],
            root_cause="intents.json needs updates",
            suggested_fix="Add query pattern",
            component_to_patch="intents.json",
            code_change_required=False,
        )

        test_result = QualityTestResult(
            test_case_id="test_001",
            query="휴학 신청 방법",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer.generate_patch_suggestion(analysis, test_result)

        assert result["target"] == "intents.json"
        assert "patch_content" in result
        assert result["patch_content"]["intent"] == "extracted_from_query"
        assert "휴학 신청 방법" in result["patch_content"]["patterns"]

    def test_generate_patch_suggestion_synonyms_json(self):
        """
        SPEC: Should generate synonyms.json patch content.
        """
        analyzer = FailureAnalyzer()

        from src.rag.automation.domain.value_objects import FiveWhyAnalysis

        analysis = FiveWhyAnalysis(
            test_case_id="test_002",
            original_failure="Quality failure",
            why_chain=[],
            root_cause="synonyms.json needs updates",
            suggested_fix="Add synonyms",
            component_to_patch="synonyms.json",
            code_change_required=False,
        )

        test_result = QualityTestResult(
            test_case_id="test_002",
            query="장학금 신청 자격",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer.generate_patch_suggestion(analysis, test_result)

        assert result["target"] == "synonyms.json"
        assert "patch_content" in result
        assert result["patch_content"]["term"] == "장학금"
        assert "신청" in result["patch_content"]["synonyms"]

    def test_generate_patch_suggestion_no_patch_target(self):
        """
        SPEC: Should handle case with no patch target.
        """
        analyzer = FailureAnalyzer()

        from src.rag.automation.domain.value_objects import FiveWhyAnalysis

        analysis = FiveWhyAnalysis(
            test_case_id="test_003",
            original_failure="Execution error",
            why_chain=[],
            root_cause="Unknown issue",
            suggested_fix="Investigate",
            component_to_patch=None,
            code_change_required=True,
        )

        test_result = QualityTestResult(
            test_case_id="test_003",
            query="query",
            answer="a",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            passed=False,
        )

        result = analyzer.generate_patch_suggestion(analysis, test_result)

        assert result["target"] is None
        assert result["requires_code_change"] is True
        assert "patch_content" not in result


class TestGenerateIntentsPatch:
    """Tests for _generate_intents_patch method."""

    def test_generate_intents_patch_structure(self):
        """
        SPEC: Should create valid intent entry structure.
        """
        analyzer = FailureAnalyzer()

        query = "휴학 신청 기간이 언제인가요?"

        result = analyzer._generate_intents_patch(query)

        assert "intent" in result
        assert "patterns" in result
        assert "keywords" in result
        assert "examples" in result
        assert query in result["patterns"]
        assert query in result["examples"]

    def test_generate_intents_patch_keywords_extraction(self):
        """
        SPEC: Should extract first 5 words as keywords.
        """
        analyzer = FailureAnalyzer()

        query = "휴학 신청 기간이 언제인가요?"

        result = analyzer._generate_intents_patch(query)

        assert len(result["keywords"]) <= 5
        assert "휴학" in result["keywords"]

    def test_generate_intents_patch_long_query(self):
        """
        SPEC: Should handle long queries correctly.
        """
        analyzer = FailureAnalyzer()

        query = "대학원 진학을 위한 학부 성적 기준과 연구 실적 요건이 어떻게 되나요?"

        result = analyzer._generate_intents_patch(query)

        assert len(result["keywords"]) == 5
        assert query in result["patterns"]


class TestGenerateSynonymsPatch:
    """Tests for _generate_synonyms_patch method."""

    def test_generate_synonyms_patch_structure(self):
        """
        SPEC: Should create valid synonym entry structure.
        """
        analyzer = FailureAnalyzer()

        query = "휴학 신청 방법"

        result = analyzer._generate_synonyms_patch(query)

        assert "term" in result
        assert "synonyms" in result
        assert "context" in result

    def test_generate_synonyms_patch_context(self):
        """
        SPEC: Should set context to regulation_query.
        """
        analyzer = FailureAnalyzer()

        query = "any query"

        result = analyzer._generate_synonyms_patch(query)

        assert result["context"] == "regulation_query"

    def test_generate_synonyms_patch_first_word_as_term(self):
        """
        SPEC: Should use first word as term.
        """
        analyzer = FailureAnalyzer()

        query = "휴학 신청 방법"

        result = analyzer._generate_synonyms_patch(query)

        assert result["term"] == "휴학"
        assert "신청" in result["synonyms"]
        assert "방법" in result["synonyms"]

    def test_generate_synonyms_patch_short_query(self):
        """
        SPEC: Should handle short queries.
        """
        analyzer = FailureAnalyzer()

        query = "휴학"

        result = analyzer._generate_synonyms_patch(query)

        assert result["term"] == "휴학"
        # May have empty synonyms list for single word
        assert isinstance(result["synonyms"], list)


class TestAnalyzeFailureCompleteFlow:
    """Integration tests for complete analyze_failure flow."""

    def test_analyze_failure_with_all_quality_dimensions_low(self):
        """
        SPEC: Should handle case where all quality dimensions are low.
        """
        analyzer = FailureAnalyzer()

        quality_score = QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.5,
                completeness=0.5,
                relevance=0.5,
                source_citation=0.5,
                practicality=0.3,
                actionability=0.3,
            ),
            total_score=2.6,
            is_pass=False,
        )

        test_result = QualityTestResult(
            test_case_id="test_001",
            query="질문",
            answer="답변",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            quality_score=quality_score,
            passed=False,
        )

        analysis = analyzer.analyze_failure(test_result)

        assert analysis.test_case_id == "test_001"
        assert len(analysis.why_chain) == 5
        assert analysis.root_cause
        assert analysis.suggested_fix

    def test_analyze_failure_with_all_fact_checks_failed(self):
        """
        SPEC: Should handle case where all fact checks failed.
        """
        analyzer = FailureAnalyzer()

        fact_checks = [
            FactCheck(
                claim="Wrong claim 1",
                status=FactCheckStatus.FAIL,
                source="source",
                confidence=0.3,
                correction="Correction 1",
            ),
            FactCheck(
                claim="Wrong claim 2",
                status=FactCheckStatus.FAIL,
                source="source",
                confidence=0.4,
                correction="Correction 2",
            ),
            FactCheck(
                claim="Wrong claim 3",
                status=FactCheckStatus.FAIL,
                source="source",
                confidence=0.3,
                correction="Correction 3",
            ),
        ]

        test_result = QualityTestResult(
            test_case_id="test_002",
            query="질문",
            answer="답변",
            sources=[],
            confidence=0.5,
            execution_time_ms=1000,
            rag_pipeline_log={},
            fact_checks=fact_checks,
            passed=False,
        )

        analysis = analyzer.analyze_failure(test_result)

        assert "Fact check failure" in analysis.original_failure
        assert len(analysis.why_chain) == 5
