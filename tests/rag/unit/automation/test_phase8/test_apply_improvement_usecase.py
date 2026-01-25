"""
Unit Tests for ApplyImprovementUseCase (Phase 8).

Tests for applying automated improvements to RAG system based on
5-Why analysis results.
Clean Architecture: Application layer tests.
"""

import json
from pathlib import Path

import pytest

from src.rag.automation.application.apply_improvement_usecase import (
    ApplyImprovementUseCase,
)
from src.rag.automation.domain.entities import TestResult
from src.rag.automation.domain.value_objects import (
    FiveWhyAnalysis,
    QualityDimensions,
    QualityScore,
)


@pytest.fixture
def temp_intents_file(tmp_path):
    """Create a temporary intents.json file for testing."""
    intents_path = tmp_path / "intents.json"
    intents_data = {
        "intents": [
            {
                "intent": "query_withdrawal",
                "patterns": ["휴학 신청 방법"],
                "keywords": ["휴학", "신청"],
                "examples": ["휴학 신청은 어떻게 하나요?"],
            }
        ]
    }
    intents_path.write_text(json.dumps(intents_data, ensure_ascii=False, indent=2))
    return intents_path


@pytest.fixture
def temp_synonyms_file(tmp_path):
    """Create a temporary synonyms.json file for testing."""
    synonyms_path = tmp_path / "synonyms.json"
    synonyms_data = {
        "synonyms": [
            {
                "term": "휴학",
                "synonyms": ["학기휴학", "등록휴학"],
                "context": "regulation_query",
            }
        ]
    }
    synonyms_path.write_text(json.dumps(synonyms_data, ensure_ascii=False, indent=2))
    return synonyms_path


@pytest.fixture
def sample_test_result():
    """Create a sample test result for testing."""
    return TestResult(
        test_case_id="test_001",
        query="성적 장학금 신청 방법",
        answer="성적 장학금은 학기 성적이 우수한 학생에게 지급됩니다",
        sources=["장학금 규정 제5조"],
        confidence=0.85,
        execution_time_ms=1200,
        rag_pipeline_log={},
        quality_score=QualityScore(
            dimensions=QualityDimensions(
                accuracy=0.9,
                completeness=0.8,
                relevance=0.85,
                source_citation=1.0,
                practicality=0.4,
                actionability=0.45,
            ),
            total_score=4.4,
            is_pass=True,
        ),
        passed=True,
    )


@pytest.fixture
def sample_analysis_intents():
    """Create a sample 5-Why analysis for intents patching."""
    return FiveWhyAnalysis(
        test_case_id="test_001",
        original_failure="Query not matched to correct intent",
        why_chain=[
            "Query not matched",
            "Intent pattern missing",
            "No pattern for this query type",
            "Intents not comprehensive",
            "Manual intent creation incomplete",
        ],
        root_cause="Missing intent pattern for scholarship queries",
        suggested_fix="Add new intent pattern for 성적 장학금 queries",
        component_to_patch="intents.json",
        code_change_required=False,
    )


@pytest.fixture
def sample_analysis_synonyms():
    """Create a sample 5-Why analysis for synonyms patching."""
    return FiveWhyAnalysis(
        test_case_id="test_002",
        original_failure="Query terms not recognized",
        why_chain=[
            "Terms not recognized",
            "Synonym mapping missing",
            "No synonym for this term",
            "Synonyms not comprehensive",
            "Manual synonym creation incomplete",
        ],
        root_cause="Missing synonym for 장학금 terms",
        suggested_fix="Add 장학금 to synonyms",
        component_to_patch="synonyms.json",
        code_change_required=False,
    )


@pytest.fixture
def sample_analysis_config():
    """Create a sample 5-Why analysis for config changes."""
    return FiveWhyAnalysis(
        test_case_id="test_003",
        original_failure="Retrieval quality insufficient",
        why_chain=[
            "Poor retrieval",
            "Not enough documents",
            "Top-k too low",
            "Config not optimized",
            "Manual tuning required",
        ],
        root_cause="Retrieval top_k parameter too low",
        suggested_fix="Increase top_k from 5 to 7",
        component_to_patch="config",
        code_change_required=False,
    )


@pytest.fixture
def sample_analysis_code():
    """Create a sample 5-Why analysis requiring code changes."""
    return FiveWhyAnalysis(
        test_case_id="test_004",
        original_failure="Hallucination in answer",
        why_chain=[
            "Answer hallucinated",
            "No source constraint",
            "Prompt insufficient",
            "LLM not constrained",
            "Code update required",
        ],
        root_cause="LLM prompt allows hallucination",
        suggested_fix="Update prompt to enforce source-based generation",
        component_to_patch=None,
        code_change_required=True,
    )


class TestApplyImprovementUseCaseInitialization:
    """Test suite for use case initialization."""

    def test_initialize_with_default_paths(self):
        """WHEN initializing with defaults, THEN should set default paths."""
        usecase = ApplyImprovementUseCase()

        assert usecase.intents_path == Path("src/rag/data/intents.json")
        assert usecase.synonyms_path == Path("src/rag/data/synonyms.json")

    def test_initialize_with_custom_paths(self, temp_intents_file, temp_synonyms_file):
        """WHEN initializing with custom paths, THEN should use custom paths."""
        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        assert usecase.intents_path == temp_intents_file
        assert usecase.synonyms_path == temp_synonyms_file


class TestApplyImprovements:
    """Test suite for apply_improvements method."""

    def test_apply_improvements_dry_run(
        self,
        temp_intents_file,
        temp_synonyms_file,
        sample_analysis_intents,
        sample_test_result,
    ):
        """WHEN applying improvements in dry_run mode, THEN should not modify files."""
        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        # Get original file content
        original_content = temp_intents_file.read_text()

        results = usecase.apply_improvements(
            analyses=[sample_analysis_intents],
            test_results=[sample_test_result],
            dry_run=True,
        )

        # File should not be modified
        assert temp_intents_file.read_text() == original_content
        assert results["statistics"]["intents_patches"] == 1

    def test_apply_improvements_patches_intents(
        self,
        temp_intents_file,
        temp_synonyms_file,
        sample_analysis_intents,
        sample_test_result,
    ):
        """WHEN applying intents patch, THEN should add to intents_patches list."""
        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        results = usecase.apply_improvements(
            analyses=[sample_analysis_intents],
            test_results=[sample_test_result],
            dry_run=True,
        )

        assert len(results["intents_patches"]) == 1
        assert results["intents_patches"][0]["test_case_id"] == "test_001"
        assert results["intents_patches"][0]["applied"] is False  # dry_run

    def test_apply_improvements_patches_synonyms(
        self,
        temp_intents_file,
        temp_synonyms_file,
        sample_analysis_synonyms,
    ):
        """WHEN applying synonyms patch, THEN should add to synonyms_patches list."""
        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        # Create matching test result
        test_result = TestResult(
            test_case_id="test_002",
            query="장학금 신청",
            answer="장학금은 신청해야 합니다",
            sources=[],
            confidence=0.8,
            execution_time_ms=1000,
            rag_pipeline_log={},
            quality_score=QualityScore(
                dimensions=QualityDimensions(
                    accuracy=0.8,
                    completeness=0.7,
                    relevance=0.75,
                    source_citation=0.5,
                    practicality=0.4,
                    actionability=0.4,
                ),
                total_score=3.55,
                is_pass=False,
            ),
            passed=False,
        )

        results = usecase.apply_improvements(
            analyses=[sample_analysis_synonyms],
            test_results=[test_result],
            dry_run=True,
        )

        assert len(results["synonyms_patches"]) == 1
        assert results["synonyms_patches"][0]["test_case_id"] == "test_002"

    def test_apply_improvements_generates_config_suggestions(
        self,
        temp_intents_file,
        temp_synonyms_file,
        sample_analysis_config,
    ):
        """WHEN analysis targets config, THEN should generate config suggestions."""
        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        # Create matching test result
        test_result = TestResult(
            test_case_id="test_003",
            query="검색 결과",
            answer="검색 결과가 부족합니다",
            sources=[],
            confidence=0.6,
            execution_time_ms=1000,
            rag_pipeline_log={},
            quality_score=QualityScore(
                dimensions=QualityDimensions(
                    accuracy=0.6,
                    completeness=0.6,
                    relevance=0.6,
                    source_citation=0.5,
                    practicality=0.3,
                    actionability=0.3,
                ),
                total_score=2.9,
                is_pass=False,
            ),
            passed=False,
        )

        results = usecase.apply_improvements(
            analyses=[sample_analysis_config],
            test_results=[test_result],
            dry_run=True,
        )

        assert len(results["config_suggestions"]) == 1
        assert results["config_suggestions"][0]["test_case_id"] == "test_003"
        assert results["statistics"]["config_changes"] == 1

    def test_apply_improvements_generates_code_suggestions(
        self,
        temp_intents_file,
        temp_synonyms_file,
        sample_analysis_code,
    ):
        """WHEN code change required, THEN should generate code suggestions."""
        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        # Create matching test result
        test_result = TestResult(
            test_case_id="test_004",
            query="장학금 지급 기준",
            answer="장학금은 학기 성적이 우수한 학생에게 지급됩니다",
            sources=[],
            confidence=0.85,
            execution_time_ms=1000,
            rag_pipeline_log={},
            quality_score=QualityScore(
                dimensions=QualityDimensions(
                    accuracy=0.7,
                    completeness=0.7,
                    relevance=0.7,
                    source_citation=0.5,
                    practicality=0.4,
                    actionability=0.4,
                ),
                total_score=3.4,
                is_pass=False,
            ),
            passed=False,
        )

        results = usecase.apply_improvements(
            analyses=[sample_analysis_code],
            test_results=[test_result],
            dry_run=True,
        )

        assert len(results["code_suggestions"]) == 1
        assert results["code_suggestions"][0]["test_case_id"] == "test_004"
        assert results["statistics"]["code_changes"] == 1

    def test_apply_improvements_skips_missing_component(
        self,
        temp_intents_file,
        temp_synonyms_file,
        sample_test_result,
    ):
        """WHEN component_to_patch is None, THEN should skip processing."""
        analysis_no_target = FiveWhyAnalysis(
            test_case_id="test_005",
            original_failure="Some failure",
            why_chain=["why1", "why2"],
            root_cause="Unknown cause",
            suggested_fix="Fix it",
            component_to_patch=None,
            code_change_required=False,
        )

        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        results = usecase.apply_improvements(
            analyses=[analysis_no_target],
            test_results=[sample_test_result],
            dry_run=True,
        )

        # Should have no patches
        assert len(results["intents_patches"]) == 0
        assert len(results["synonyms_patches"]) == 0

    def test_apply_improvements_handles_multiple_analyses(
        self,
        temp_intents_file,
        temp_synonyms_file,
        sample_analysis_intents,
        sample_analysis_synonyms,
        sample_analysis_config,
        sample_test_result,
    ):
        """WHEN multiple analyses provided, THEN should process all."""
        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        results = usecase.apply_improvements(
            analyses=[
                sample_analysis_intents,
                sample_analysis_synonyms,
                sample_analysis_config,
            ],
            test_results=[sample_test_result] * 3,
            dry_run=True,
        )

        assert results["statistics"]["total_analyses"] == 3
        assert len(results["intents_patches"]) == 1
        assert len(results["synonyms_patches"]) == 1
        assert len(results["config_suggestions"]) == 1


class TestPatchIntents:
    """Test suite for _patch_intents method."""

    def test_patch_intents_creates_new_entry(
        self,
        temp_intents_file,
        sample_analysis_intents,
        sample_test_result,
    ):
        """WHEN patching intents with new intent, THEN should create new entry."""
        usecase = ApplyImprovementUseCase(intents_path=temp_intents_file)

        result = usecase._patch_intents(
            analysis=sample_analysis_intents,
            test_result=sample_test_result,
            dry_run=False,
        )

        assert result is not None
        assert result["test_case_id"] == "test_001"
        assert result["exists"] is False
        assert result["applied"] is True

        # Verify intent was actually added to file
        intents_data = json.loads(temp_intents_file.read_text())
        assert len(intents_data["intents"]) == 2  # Original + new

    def test_patch_intents_detects_existing_intent(
        self,
        temp_intents_file,
        sample_test_result,
    ):
        """WHEN intent already exists, THEN should detect and skip."""
        # Create analysis that would generate existing intent name
        analysis = FiveWhyAnalysis(
            test_case_id="test_001",
            original_failure="Test",
            why_chain=["why1"],
            root_cause="Test",
            suggested_fix="Test",
            component_to_patch="intents.json",
        )

        usecase = ApplyImprovementUseCase(intents_path=temp_intents_file)

        # First patch adds the intent
        result1 = usecase._patch_intents(
            analysis=analysis,
            test_result=sample_test_result,
            dry_run=False,
        )
        assert result1["applied"] is True

        # Second patch should detect it exists
        result2 = usecase._patch_intents(
            analysis=analysis,
            test_result=sample_test_result,
            dry_run=False,
        )
        assert result2["exists"] is True
        assert result2["applied"] is False

    def test_patch_intents_dry_run_skips_write(
        self,
        temp_intents_file,
        sample_analysis_intents,
        sample_test_result,
    ):
        """WHEN dry_run=True, THEN should not write to file."""
        usecase = ApplyImprovementUseCase(intents_path=temp_intents_file)

        original_content = temp_intents_file.read_text()

        result = usecase._patch_intents(
            analysis=sample_analysis_intents,
            test_result=sample_test_result,
            dry_run=True,
        )

        assert result["applied"] is False
        assert temp_intents_file.read_text() == original_content

    def test_patch_intents_handles_missing_file(
        self,
        tmp_path,
        sample_analysis_intents,
        sample_test_result,
    ):
        """WHEN intents file doesn't exist, THEN should return None."""
        non_existent_path = tmp_path / "non_existent_intents.json"
        usecase = ApplyImprovementUseCase(intents_path=non_existent_path)

        result = usecase._patch_intents(
            analysis=sample_analysis_intents,
            test_result=sample_test_result,
            dry_run=False,
        )

        assert result is None

    def test_patch_intents_generates_proper_intent_entry(
        self,
        temp_intents_file,
        sample_analysis_intents,
        sample_test_result,
    ):
        """WHEN generating intent entry, THEN should include all required fields."""
        usecase = ApplyImprovementUseCase(intents_path=temp_intents_file)

        result = usecase._patch_intents(
            analysis=sample_analysis_intents,
            test_result=sample_test_result,
            dry_run=False,
        )

        new_intent = result["new_intent"]
        assert "intent" in new_intent
        assert "patterns" in new_intent
        assert "keywords" in new_intent
        assert "examples" in new_intent
        assert "metadata" in new_intent
        assert new_intent["metadata"]["generated_from"] == "automated_testing"
        assert (
            new_intent["metadata"]["root_cause"] == sample_analysis_intents.root_cause
        )


class TestPatchSynonyms:
    """Test suite for _patch_synonyms method."""

    def test_patch_synonyms_creates_new_entry(
        self,
        temp_synonyms_file,
        sample_analysis_synonyms,
    ):
        """WHEN patching synonyms with new term, THEN should create new entry."""
        usecase = ApplyImprovementUseCase(synonyms_path=temp_synonyms_file)

        # Create matching test result
        test_result = TestResult(
            test_case_id="test_002",
            query="장학금 신청",
            answer="장학금은 신청해야 합니다",
            sources=[],
            confidence=0.8,
            execution_time_ms=1000,
            rag_pipeline_log={},
            quality_score=QualityScore(
                dimensions=QualityDimensions(
                    accuracy=0.8,
                    completeness=0.7,
                    relevance=0.75,
                    source_citation=0.5,
                    practicality=0.4,
                    actionability=0.4,
                ),
                total_score=3.55,
                is_pass=False,
            ),
            passed=False,
        )

        result = usecase._patch_synonyms(
            analysis=sample_analysis_synonyms,
            test_result=test_result,
            dry_run=False,
        )

        assert result is not None
        assert result["test_case_id"] == "test_002"
        assert result["exists"] is False
        assert result["applied"] is True

        # Verify synonym was actually added to file
        synonyms_data = json.loads(temp_synonyms_file.read_text())
        assert len(synonyms_data["synonyms"]) >= 1

    def test_patch_synonyms_detects_existing_synonym(
        self,
        temp_synonyms_file,
        sample_analysis_synonyms,
        sample_test_result,
    ):
        """WHEN synonym already exists, THEN should detect and skip."""
        usecase = ApplyImprovementUseCase(synonyms_path=temp_synonyms_file)

        # First patch adds the synonym
        result1 = usecase._patch_synonyms(
            analysis=sample_analysis_synonyms,
            test_result=sample_test_result,
            dry_run=False,
        )
        assert result1["applied"] is True

        # Second patch should detect it exists (same term)
        result2 = usecase._patch_synonyms(
            analysis=sample_analysis_synonyms,
            test_result=sample_test_result,
            dry_run=False,
        )
        assert result2["exists"] is True
        assert result2["applied"] is False

    def test_patch_synonyms_dry_run_skips_write(
        self,
        temp_synonyms_file,
        sample_analysis_synonyms,
        sample_test_result,
    ):
        """WHEN dry_run=True, THEN should not write to file."""
        usecase = ApplyImprovementUseCase(synonyms_path=temp_synonyms_file)

        original_content = temp_synonyms_file.read_text()

        result = usecase._patch_synonyms(
            analysis=sample_analysis_synonyms,
            test_result=sample_test_result,
            dry_run=True,
        )

        assert result["applied"] is False
        assert temp_synonyms_file.read_text() == original_content

    def test_patch_synonyms_handles_missing_file(
        self,
        tmp_path,
        sample_analysis_synonyms,
        sample_test_result,
    ):
        """WHEN synonyms file doesn't exist, THEN should return None."""
        non_existent_path = tmp_path / "non_existent_synonyms.json"
        usecase = ApplyImprovementUseCase(synonyms_path=non_existent_path)

        result = usecase._patch_synonyms(
            analysis=sample_analysis_synonyms,
            test_result=sample_test_result,
            dry_run=False,
        )

        assert result is None

    def test_patch_synonyms_generates_proper_synonym_entry(
        self,
        temp_synonyms_file,
        sample_analysis_synonyms,
        sample_test_result,
    ):
        """WHEN generating synonym entry, THEN should include all required fields."""
        usecase = ApplyImprovementUseCase(synonyms_path=temp_synonyms_file)

        result = usecase._patch_synonyms(
            analysis=sample_analysis_synonyms,
            test_result=sample_test_result,
            dry_run=False,
        )

        new_synonym = result["new_synonym"]
        assert "term" in new_synonym
        assert "synonyms" in new_synonym
        assert "context" in new_synonym
        assert "metadata" in new_synonym
        assert new_synonym["context"] == "regulation_query"
        assert new_synonym["metadata"]["generated_from"] == "automated_testing"


class TestHelperMethods:
    """Test suite for helper methods."""

    def test_extract_keywords_removes_stop_words(self):
        """WHEN extracting keywords, THEN should remove Korean stop words."""
        usecase = ApplyImprovementUseCase()

        query = "휴학 은 는 어떻게 신청 하나요"
        keywords = usecase._extract_keywords(query)

        # Should not contain stop words
        assert "은" not in keywords
        assert "는" not in keywords
        assert "어떻게" not in keywords or "어떻게" in keywords  # May be kept

        # Should contain meaningful terms
        assert "휴학" in keywords or any("휴학" in k for k in keywords)

    def test_extract_keywords_limits_to_5(self):
        """WHEN extracting keywords, THEN should return at most 5 keywords."""
        usecase = ApplyImprovementUseCase()

        query = "장학금 성적 학점 신청 기간 휴학 복학 등록"
        keywords = usecase._extract_keywords(query)

        assert len(keywords) <= 5

    def test_sanitize_string_removes_special_chars(self):
        """WHEN sanitizing string, THEN should remove special characters."""
        usecase = ApplyImprovementUseCase()

        input_str = "hello@#$%^&*() world"
        sanitized = usecase._sanitize_string(input_str)

        assert "@" not in sanitized
        assert "#" not in sanitized
        assert "_" in sanitized  # Replaces special chars

    def test_sanitize_string_replaces_spaces_with_underscore(self):
        """WHEN sanitizing string, THEN should replace spaces with underscores."""
        usecase = ApplyImprovementUseCase()

        input_str = "hello world test"
        sanitized = usecase._sanitize_string(input_str)

        assert " " not in sanitized
        assert "_" in sanitized

    def test_sanitize_string_limits_length(self):
        """WHEN sanitizing long string, THEN should limit to 30 chars."""
        usecase = ApplyImprovementUseCase()

        long_string = "a" * 50
        sanitized = usecase._sanitize_string(long_string)

        assert len(sanitized) <= 30

    def test_assess_priority_returns_high_for_retrieval(self):
        """WHEN root cause mentions retrieval, THEN should return high priority."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test",
            original_failure="Test",
            why_chain=[],
            root_cause="Retrieval failure in search component",
            suggested_fix="Fix",
        )

        priority = usecase._assess_priority(analysis)
        assert priority == "high"

    def test_assess_priority_returns_medium_for_other(self):
        """WHEN root cause doesn't mention priority keywords, THEN should return medium."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test",
            original_failure="Test",
            why_chain=[],
            root_cause="UI layout issue",
            suggested_fix="Fix",
        )

        priority = usecase._assess_priority(analysis)
        assert priority == "medium"

    def test_suggest_file_to_modify_returns_prompt_file(self):
        """WHEN root cause mentions prompt, THEN should suggest prompt file."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test",
            original_failure="Test",
            why_chain=[],
            root_cause="LLM prompt issue",
            suggested_fix="Fix",
        )

        file_hint = usecase._suggest_file_to_modify(analysis)
        assert "prompt" in file_hint.lower()

    def test_suggest_file_to_modify_returns_retrieval_file(self):
        """WHEN root cause mentions retrieval, THEN should suggest retrieval file."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test",
            original_failure="Test",
            why_chain=[],
            root_cause="Retrieval search failed",
            suggested_fix="Fix",
        )

        file_hint = usecase._suggest_file_to_modify(analysis)
        assert "hybrid_search" in file_hint

    def test_generate_code_hint_for_prompt(self):
        """WHEN root cause is prompt-related, THEN should generate appropriate hint."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test",
            original_failure="Test",
            why_chain=[],
            root_cause="Prompt allows hallucination",
            suggested_fix="Fix",
        )

        hint = usecase._generate_code_hint(analysis)
        assert "prompt" in hint.lower()
        assert "source" in hint.lower()

    def test_suggest_config_changes_returns_structure(self):
        """WHEN suggesting config changes, THEN should return proper structure."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test",
            original_failure="Test",
            why_chain=[],
            root_cause="Config issue",
            suggested_fix="Fix",
        )

        changes = usecase._suggest_config_changes(analysis)
        assert "suggested_changes" in changes
        assert isinstance(changes["suggested_changes"], list)


class TestPreviewImprovements:
    """Test suite for preview_improvements method."""

    def test_preview_generates_markdown_output(
        self,
        sample_analysis_intents,
        sample_test_result,
    ):
        """WHEN generating preview, THEN should return markdown formatted string."""
        usecase = ApplyImprovementUseCase()

        preview = usecase.preview_improvements(
            analyses=[sample_analysis_intents],
            test_results=[sample_test_result],
        )

        assert "# Improvement Preview" in preview
        assert "Total analyses:" in preview
        assert "## Test:" in preview

    def test_preview_includes_test_details(
        self,
        sample_analysis_intents,
        sample_test_result,
    ):
        """WHEN generating preview, THEN should include test case details."""
        usecase = ApplyImprovementUseCase()

        preview = usecase.preview_improvements(
            analyses=[sample_analysis_intents],
            test_results=[sample_test_result],
        )

        assert sample_test_result.test_case_id in preview
        assert sample_test_result.query[:30] in preview

    def test_preview_includes_analysis_details(
        self,
        sample_analysis_intents,
        sample_test_result,
    ):
        """WHEN generating preview, THEN should include analysis details."""
        usecase = ApplyImprovementUseCase()

        preview = usecase.preview_improvements(
            analyses=[sample_analysis_intents],
            test_results=[sample_test_result],
        )

        assert sample_analysis_intents.root_cause in preview
        assert sample_analysis_intents.suggested_fix in preview
        assert sample_analysis_intents.component_to_patch in preview
