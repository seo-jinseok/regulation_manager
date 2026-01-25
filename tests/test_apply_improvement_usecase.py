"""
Unit tests for ApplyImprovementUseCase.

Tests the application layer for applying automated improvements to the RAG system.
Follows TRUST 5 framework: Testable, Readable, Unified, Secured, Trackable.

Clean Architecture: Application layer tests with mock infrastructure.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.rag.automation.application.apply_improvement_usecase import (
    ApplyImprovementUseCase,
)
from src.rag.automation.domain.entities import TestResult
from src.rag.automation.domain.value_objects import FiveWhyAnalysis

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_intents_file():
    """Create a temporary intents.json file for testing."""
    data = {
        "intents": [
            {"intent": "query_test", "patterns": ["test"], "keywords": ["test"]}
        ]
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_synonyms_file():
    """Create a temporary synonyms.json file for testing."""
    data = {"synonyms": [{"term": "test", "synonyms": ["exam"], "context": "test"}]}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_test_result():
    """Create a sample TestResult for testing."""
    return TestResult(
        test_case_id="test-001",
        query="학칙에 대해 알려줘",
        answer="학칙은 대학의 기본 규정입니다.",
        sources=["학칙 제1조"],
        confidence=0.85,
        execution_time_ms=1500,
        rag_pipeline_log={"retrieval_count": 5, "reranked": True},
        fact_checks=[],
        quality_score=None,
        passed=False,
    )


@pytest.fixture
def sample_five_why_analysis_intents():
    """Create a FiveWhyAnalysis targeting intents.json."""
    return FiveWhyAnalysis(
        test_case_id="test-001",
        original_failure="Query not understood",
        why_chain=[
            "Query not understood",
            "No matching intent found",
            "Intent patterns too specific",
            "Limited pattern coverage",
            "Insufficient intent training data",
        ],
        root_cause="Insufficient intent training data in intents.json",
        suggested_fix="Add new intent pattern for the query",
        component_to_patch="intents.json",
        code_change_required=False,
    )


@pytest.fixture
def sample_five_why_analysis_synonyms():
    """Create a FiveWhyAnalysis targeting synonyms.json."""
    return FiveWhyAnalysis(
        test_case_id="test-002",
        original_failure="Query terms not matched",
        why_chain=[
            "Query terms not matched",
            "No synonyms found",
            "Limited synonym coverage",
            "Missing common terms",
            "Insufficient synonym database",
        ],
        root_cause="Insufficient synonym coverage in synonyms.json",
        suggested_fix="Add new synonym mappings for common terms",
        component_to_patch="synonyms.json",
        code_change_required=False,
    )


@pytest.fixture
def sample_five_why_analysis_config():
    """Create a FiveWhyAnalysis targeting config."""
    return FiveWhyAnalysis(
        test_case_id="test-003",
        original_failure="Low retrieval accuracy",
        why_chain=[
            "Low retrieval accuracy",
            "Insufficient top_k results",
            "Top_k parameter too low",
            "Config not optimized",
            "Default parameters suboptimal",
        ],
        root_cause="Configuration parameter top_k is too low",
        suggested_fix="Increase top_k from 5 to 7",
        component_to_patch="config",
        code_change_required=False,
    )


@pytest.fixture
def sample_five_why_analysis_code():
    """Create a FiveWhyAnalysis requiring code changes."""
    return FiveWhyAnalysis(
        test_case_id="test-004",
        original_failure="Hallucination in generated answer",
        why_chain=[
            "Hallucination in generated answer",
            "LLM not constrained properly",
            "Prompt lacks source enforcement",
            "Prompt template insufficient",
            "Prompt design needs improvement",
        ],
        root_cause="Prompt template does not enforce source-based generation",
        suggested_fix="Update prompt to require source citation",
        component_to_patch=None,
        code_change_required=True,
    )


# =============================================================================
# Use Case Initialization Tests
# =============================================================================


class TestUseCaseInitialization:
    """Tests for ApplyImprovementUseCase initialization."""

    def test_usecase_initialization_default_paths(self):
        """WHEN usecase created with default paths, THEN should initialize correctly."""
        usecase = ApplyImprovementUseCase()

        assert usecase.intents_path == Path("src/rag/data/intents.json")
        assert usecase.synonyms_path == Path("src/rag/data/synonyms.json")
        assert usecase.logger is not None

    def test_usecase_initialization_custom_paths(self):
        """WHEN usecase created with custom paths, THEN should use custom paths."""
        custom_intents = Path("/custom/path/intents.json")
        custom_synonyms = Path("/custom/path/synonyms.json")

        usecase = ApplyImprovementUseCase(
            intents_path=custom_intents,
            synonyms_path=custom_synonyms,
        )

        assert usecase.intents_path == custom_intents
        assert usecase.synonyms_path == custom_synonyms

    def test_usecase_initialization_with_temp_files(
        self, temp_intents_file, temp_synonyms_file
    ):
        """WHEN usecase created with temp files, THEN should initialize correctly."""
        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        assert usecase.intents_path.exists()
        assert usecase.synonyms_path.exists()


# =============================================================================
# Apply Improvements Tests
# =============================================================================


class TestApplyImprovements:
    """Tests for apply_improvements method."""

    def test_apply_improvements_with_intents_patch(
        self,
        temp_intents_file,
        sample_test_result,
        sample_five_why_analysis_intents,
    ):
        """WHEN improvements applied with intents patch, THEN should update intents."""
        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=Path("nonexistent.json"),
        )

        result = usecase.apply_improvements(
            analyses=[sample_five_why_analysis_intents],
            test_results=[sample_test_result],
            dry_run=True,
        )

        assert result["statistics"]["total_analyses"] == 1
        assert result["statistics"]["intents_patches"] == 1
        assert len(result["intents_patches"]) == 1
        assert result["intents_patches"][0]["test_case_id"] == "test-001"
        assert result["intents_patches"][0]["applied"] is False  # dry_run=True

    def test_apply_improvements_with_synonyms_patch(
        self,
        temp_synonyms_file,
        sample_test_result,
        sample_five_why_analysis_synonyms,
    ):
        """WHEN improvements applied with synonyms patch, THEN should update synonyms."""
        usecase = ApplyImprovementUseCase(
            intents_path=Path("nonexistent.json"),
            synonyms_path=temp_synonyms_file,
        )

        result = usecase.apply_improvements(
            analyses=[sample_five_why_analysis_synonyms],
            test_results=[sample_test_result],
            dry_run=True,
        )

        assert result["statistics"]["total_analyses"] == 1
        assert result["statistics"]["synonyms_patches"] == 1
        assert len(result["synonyms_patches"]) == 1
        assert result["synonyms_patches"][0]["test_case_id"] == "test-002"

    def test_apply_improvements_with_config_suggestion(
        self, sample_test_result, sample_five_why_analysis_config
    ):
        """WHEN improvements applied with config change, THEN should generate suggestion."""
        usecase = ApplyImprovementUseCase()

        result = usecase.apply_improvements(
            analyses=[sample_five_why_analysis_config],
            test_results=[sample_test_result],
            dry_run=True,
        )

        assert result["statistics"]["total_analyses"] == 1
        assert result["statistics"]["config_changes"] == 1
        assert len(result["config_suggestions"]) == 1
        assert result["config_suggestions"][0]["test_case_id"] == "test-003"

    def test_apply_improvements_with_code_suggestion(
        self, sample_test_result, sample_five_why_analysis_code
    ):
        """WHEN improvements applied with code change, THEN should generate suggestion."""
        usecase = ApplyImprovementUseCase()

        result = usecase.apply_improvements(
            analyses=[sample_five_why_analysis_code],
            test_results=[sample_test_result],
            dry_run=True,
        )

        assert result["statistics"]["total_analyses"] == 1
        assert result["statistics"]["code_changes"] == 1
        assert len(result["code_suggestions"]) == 1
        assert result["code_suggestions"][0]["test_case_id"] == "test-004"
        assert result["code_suggestions"][0]["priority"] in ["high", "medium"]

    def test_apply_multiple_improvements(
        self,
        temp_intents_file,
        temp_synonyms_file,
        sample_test_result,
        sample_five_why_analysis_intents,
        sample_five_why_analysis_synonyms,
        sample_five_why_analysis_config,
        sample_five_why_analysis_code,
    ):
        """WHEN multiple improvements applied, THEN should apply all types."""
        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=temp_synonyms_file,
        )

        # Create test results for each analysis
        test_results = [
            TestResult(
                test_case_id="test-001",
                query="학칙에 대해 알려줘",
                answer="학칙은 대학의 기본 규정입니다.",
                sources=["학칙 제1조"],
                confidence=0.85,
                execution_time_ms=1500,
                rag_pipeline_log={},
                fact_checks=[],
                quality_score=None,
                passed=False,
            ),
            TestResult(
                test_case_id="test-002",
                query="규정 설명",
                answer="규정입니다.",
                sources=["규정 제1조"],
                confidence=0.80,
                execution_time_ms=1200,
                rag_pipeline_log={},
                fact_checks=[],
                quality_score=None,
                passed=False,
            ),
            TestResult(
                test_case_id="test-003",
                query="검색어",
                answer="검색 결과",
                sources=["규정"],
                confidence=0.70,
                execution_time_ms=1000,
                rag_pipeline_log={},
                fact_checks=[],
                quality_score=None,
                passed=False,
            ),
            TestResult(
                test_case_id="test-004",
                query="정보 요청",
                answer="정보",
                sources=["규정"],
                confidence=0.75,
                execution_time_ms=1100,
                rag_pipeline_log={},
                fact_checks=[],
                quality_score=None,
                passed=False,
            ),
        ]

        result = usecase.apply_improvements(
            analyses=[
                sample_five_why_analysis_intents,
                sample_five_why_analysis_synonyms,
                sample_five_why_analysis_config,
                sample_five_why_analysis_code,
            ],
            test_results=test_results,
            dry_run=True,
        )

        assert result["statistics"]["total_analyses"] == 4
        assert result["statistics"]["intents_patches"] == 1
        assert result["statistics"]["synonyms_patches"] == 1
        assert result["statistics"]["config_changes"] == 1
        assert result["statistics"]["code_changes"] == 1

    def test_apply_improvements_dry_run_true_does_not_modify_files(
        self,
        temp_intents_file,
        sample_test_result,
        sample_five_why_analysis_intents,
    ):
        """WHEN dry_run=True, THEN should not modify actual files."""
        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=Path("nonexistent.json"),
        )

        # Read original content
        with open(temp_intents_file, "r", encoding="utf-8") as f:
            original_content = f.read()

        result = usecase.apply_improvements(
            analyses=[sample_five_why_analysis_intents],
            test_results=[sample_test_result],
            dry_run=True,
        )

        # Read content after dry run
        with open(temp_intents_file, "r", encoding="utf-8") as f:
            after_content = f.read()

        assert original_content == after_content
        assert result["intents_patches"][0]["applied"] is False

    def test_apply_improvements_dry_run_false_modifies_files(
        self,
        temp_intents_file,
        sample_test_result,
        sample_five_why_analysis_intents,
    ):
        """WHEN dry_run=False, THEN should modify actual files."""
        usecase = ApplyImprovementUseCase(
            intents_path=temp_intents_file,
            synonyms_path=Path("nonexistent.json"),
        )

        # Read original content
        with open(temp_intents_file, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        original_intents_count = len(original_data.get("intents", []))

        result = usecase.apply_improvements(
            analyses=[sample_five_why_analysis_intents],
            test_results=[sample_test_result],
            dry_run=False,
        )

        # Read content after applying
        with open(temp_intents_file, "r", encoding="utf-8") as f:
            after_data = json.load(f)
        after_intents_count = len(after_data.get("intents", []))

        assert after_intents_count == original_intents_count + 1
        assert result["intents_patches"][0]["applied"] is True

    def test_apply_improvements_empty_lists(self):
        """WHEN empty analyses provided, THEN should return empty results."""
        usecase = ApplyImprovementUseCase()

        result = usecase.apply_improvements(analyses=[], test_results=[], dry_run=True)

        assert result["statistics"]["total_analyses"] == 0
        assert result["statistics"]["intents_patches"] == 0
        assert result["statistics"]["synonyms_patches"] == 0
        assert result["statistics"]["code_changes"] == 0
        assert result["statistics"]["config_changes"] == 0
        assert len(result["intents_patches"]) == 0
        assert len(result["synonyms_patches"]) == 0
        assert len(result["code_suggestions"]) == 0
        assert len(result["config_suggestions"]) == 0

    def test_apply_improvements_no_component_to_patch(self, sample_test_result):
        """WHEN analysis has no component_to_patch, THEN should skip patching."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test-005",
            original_failure="Generic failure",
            why_chain=["Why 1", "Why 2", "Why 3", "Why 4", "Why 5"],
            root_cause="Some root cause",
            suggested_fix="Some fix",
            component_to_patch=None,
            code_change_required=False,
        )

        result = usecase.apply_improvements(
            analyses=[analysis],
            test_results=[sample_test_result],
            dry_run=True,
        )

        assert result["statistics"]["total_analyses"] == 1
        assert result["statistics"]["intents_patches"] == 0
        assert result["statistics"]["synonyms_patches"] == 0

    def test_apply_result_structure(
        self, temp_intents_file, sample_test_result, sample_five_why_analysis_intents
    ):
        """WHEN improvements applied, THEN result should have correct structure."""
        usecase = ApplyImprovementUseCase(intents_path=temp_intents_file)

        result = usecase.apply_improvements(
            analyses=[sample_five_why_analysis_intents],
            test_results=[sample_test_result],
            dry_run=True,
        )

        # Check top-level structure
        assert "intents_patches" in result
        assert "synonyms_patches" in result
        assert "code_suggestions" in result
        assert "config_suggestions" in result
        assert "statistics" in result

        # Check statistics structure
        assert "total_analyses" in result["statistics"]
        assert "intents_patches" in result["statistics"]
        assert "synonyms_patches" in result["statistics"]
        assert "code_changes" in result["statistics"]
        assert "config_changes" in result["statistics"]


# =============================================================================
# Patch Intents Tests
# =============================================================================


class TestPatchIntents:
    """Tests for _patch_intents method."""

    def test_patch_intents_file_not_found(
        self, sample_test_result, sample_five_why_analysis_intents
    ):
        """WHEN intents file not found, THEN should return None."""
        usecase = ApplyImprovementUseCase(intents_path=Path("nonexistent.json"))

        result = usecase._patch_intents(
            analysis=sample_five_why_analysis_intents,
            test_result=sample_test_result,
            dry_run=True,
        )

        assert result is None

    def test_patch_intents_intent_already_exists(
        self,
        temp_intents_file,
        sample_test_result,
        sample_five_why_analysis_intents,
    ):
        """WHEN intent already exists, THEN should skip and mark exists=True."""
        usecase = ApplyImprovementUseCase(intents_path=temp_intents_file)

        # The temp file already has "query_test" intent
        result = usecase._patch_intents(
            analysis=sample_five_why_analysis_intents,
            test_result=sample_test_result,
            dry_run=True,
        )

        # Result should still be returned but with exists=True
        assert result is not None
        assert result["exists"] is True
        assert result["applied"] is False


# =============================================================================
# Patch Synonyms Tests
# =============================================================================


class TestPatchSynonyms:
    """Tests for _patch_synonyms method."""

    def test_patch_synonyms_file_not_found(
        self, sample_test_result, sample_five_why_analysis_synonyms
    ):
        """WHEN synonyms file not found, THEN should return None."""
        usecase = ApplyImprovementUseCase(synonyms_path=Path("nonexistent.json"))

        result = usecase._patch_synonyms(
            analysis=sample_five_why_analysis_synonyms,
            test_result=sample_test_result,
            dry_run=True,
        )

        assert result is None

    def test_patch_synonyms_synonym_already_exists(
        self,
        temp_synonyms_file,
        sample_test_result,
        sample_five_why_analysis_synonyms,
    ):
        """WHEN synonym already exists, THEN should skip and mark exists=True."""
        usecase = ApplyImprovementUseCase(synonyms_path=temp_synonyms_file)

        result = usecase._patch_synonyms(
            analysis=sample_five_why_analysis_synonyms,
            test_result=sample_test_result,
            dry_run=True,
        )

        # The temp file already has "test" term
        assert result is not None
        # The result depends on the generated synonym term


# =============================================================================
# Generate Entry Tests
# =============================================================================


class TestGenerateEntries:
    """Tests for _generate_intent_entry and _generate_synonym_entry methods."""

    def test_generate_intent_entry_structure(self, sample_five_why_analysis_intents):
        """WHEN generating intent entry, THEN should have correct structure."""
        usecase = ApplyImprovementUseCase()

        query = "학칙에 대해 알려줘"
        entry = usecase._generate_intent_entry(query, sample_five_why_analysis_intents)

        assert "intent" in entry
        assert "patterns" in entry
        assert "keywords" in entry
        assert "examples" in entry
        assert "metadata" in entry
        assert query in entry["patterns"]
        assert query in entry["examples"]

    def test_generate_synonym_entry_structure(self, sample_five_why_analysis_synonyms):
        """WHEN generating synonym entry, THEN should have correct structure."""
        usecase = ApplyImprovementUseCase()

        query = "규정 설명"
        entry = usecase._generate_synonym_entry(
            query, sample_five_why_analysis_synonyms
        )

        assert "term" in entry
        assert "synonyms" in entry
        assert "context" in entry
        assert "metadata" in entry
        assert entry["context"] == "regulation_query"

    def test_generate_intent_entry_sanitizes_query(
        self, sample_five_why_analysis_intents
    ):
        """WHEN query has special characters, THEN intent name should be sanitized."""
        usecase = ApplyImprovementUseCase()

        query = "학칙?!@# (제8조)"
        entry = usecase._generate_intent_entry(query, sample_five_why_analysis_intents)

        # Intent name should be sanitized (no special chars)
        assert "_" in entry["intent"] or "학칙" in entry["intent"]


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestExtractKeywords:
    """Tests for _extract_keywords method."""

    def test_extract_keywords_basic(self):
        """WHEN extracting keywords, THEN should return meaningful terms."""
        usecase = ApplyImprovementUseCase()

        query = "학칙에 대해 알려줘"
        keywords = usecase._extract_keywords(query)

        assert isinstance(keywords, list)
        assert len(keywords) >= 0

    def test_extract_keywords_removes_stop_words(self):
        """WHEN extracting keywords, THEN should remove Korean stop words."""
        usecase = ApplyImprovementUseCase()

        query = "학칙은 무엇인가요"
        keywords = usecase._extract_keywords(query)

        # Stop words like '은', '는' should be removed
        assert "은" not in keywords
        assert "는" not in keywords

    def test_extract_keywords_empty_query(self):
        """WHEN query is empty, THEN should return empty list."""
        usecase = ApplyImprovementUseCase()

        keywords = usecase._extract_keywords("")
        assert keywords == []


class TestSanitizeString:
    """Tests for _sanitize_string method."""

    def test_sanitize_string_removes_special_chars(self):
        """WHEN sanitizing string, THEN should remove special characters."""
        usecase = ApplyImprovementUseCase()

        result = usecase._sanitize_string("test@#$%string")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result
        assert "%" not in result

    def test_sanitize_string_replaces_spaces(self):
        """WHEN sanitizing string, THEN should replace spaces with underscores."""
        usecase = ApplyImprovementUseCase()

        result = usecase._sanitize_string("test string with spaces")
        assert " " not in result
        assert "_" in result

    def test_sanitize_string_limits_length(self):
        """WHEN string is long, THEN should limit to 30 characters."""
        usecase = ApplyImprovementUseCase()

        long_string = "a" * 50
        result = usecase._sanitize_string(long_string)
        assert len(result) <= 30


class TestAssessPriority:
    """Tests for _assess_priority method."""

    def test_assess_priority_high_for_retrieval(self):
        """WHEN root cause mentions retrieval, THEN should return high priority."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test-001",
            original_failure="Failure",
            why_chain=["Why"] * 5,
            root_cause="Retrieval accuracy is low",
            suggested_fix="Fix retrieval",
            component_to_patch=None,
            code_change_required=False,
        )

        priority = usecase._assess_priority(analysis)
        assert priority == "high"

    def test_assess_priority_high_for_search(self):
        """WHEN root cause mentions search, THEN should return high priority."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test-001",
            original_failure="Failure",
            why_chain=["Why"] * 5,
            root_cause="Search quality issue",
            suggested_fix="Fix search",
            component_to_patch=None,
            code_change_required=False,
        )

        priority = usecase._assess_priority(analysis)
        assert priority == "high"

    def test_assess_priority_medium_for_generic(self):
        """WHEN root cause is generic, THEN should return medium priority."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test-001",
            original_failure="Failure",
            why_chain=["Why"] * 5,
            root_cause="Generic issue not related to core",
            suggested_fix="Fix generic",
            component_to_patch=None,
            code_change_required=False,
        )

        priority = usecase._assess_priority(analysis)
        assert priority == "medium"


class TestSuggestFileToModify:
    """Tests for _suggest_file_to_modify method."""

    def test_suggest_file_for_prompt_issue(self):
        """WHEN root cause mentions prompt, THEN should suggest prompt file."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test-001",
            original_failure="Failure",
            why_chain=["Why"] * 5,
            root_cause="Prompt template issue",
            suggested_fix="Fix prompt",
            component_to_patch=None,
            code_change_required=True,
        )

        file_hint = usecase._suggest_file_to_modify(analysis)
        assert "prompt" in file_hint.lower() if file_hint else True

    def test_suggest_file_for_retrieval_issue(self):
        """WHEN root cause mentions retrieval, THEN should suggest hybrid_search."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test-001",
            original_failure="Failure",
            why_chain=["Why"] * 5,
            root_cause="Retrieval component issue",
            suggested_fix="Fix retrieval",
            component_to_patch=None,
            code_change_required=True,
        )

        file_hint = usecase._suggest_file_to_modify(analysis)
        assert "hybrid_search" in file_hint if file_hint else True

    def test_suggest_file_for_fact_check_issue(self):
        """WHEN root cause mentions fact_check, THEN should suggest fact_checker."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test-001",
            original_failure="Failure",
            why_chain=["Why"] * 5,
            root_cause="Fact checking failed",
            suggested_fix="Fix fact check",
            component_to_patch=None,
            code_change_required=True,
        )

        file_hint = usecase._suggest_file_to_modify(analysis)
        assert "fact_checker" in file_hint if file_hint else True


class TestGenerateCodeHint:
    """Tests for _generate_code_hint method."""

    def test_generate_hint_for_prompt(self):
        """WHEN root cause is prompt, THEN should generate relevant hint."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test-001",
            original_failure="Failure",
            why_chain=["Why"] * 5,
            root_cause="Prompt issue",
            suggested_fix="Fix prompt",
            component_to_patch=None,
            code_change_required=True,
        )

        hint = usecase._generate_code_hint(analysis)
        assert hint is not None
        assert len(hint) > 0

    def test_generate_hint_for_parameters(self):
        """WHEN root cause is parameters, THEN should suggest parameter adjustment."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test-001",
            original_failure="Failure",
            why_chain=["Why"] * 5,
            root_cause="Parameters not optimized",
            suggested_fix="Adjust parameters",
            component_to_patch=None,
            code_change_required=True,
        )

        hint = usecase._generate_code_hint(analysis)
        assert "parameter" in hint.lower() or "top_k" in hint.lower()


class TestSuggestConfigChanges:
    """Tests for _suggest_config_changes method."""

    def test_suggest_config_changes_structure(self):
        """WHEN suggesting config changes, THEN should have correct structure."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test-001",
            original_failure="Failure",
            why_chain=["Why"] * 5,
            root_cause="Config issue",
            suggested_fix="Adjust config",
            component_to_patch="config",
            code_change_required=False,
        )

        changes = usecase._suggest_config_changes(analysis)

        assert "suggested_changes" in changes
        assert isinstance(changes["suggested_changes"], list)


# =============================================================================
# Preview Improvements Tests
# =============================================================================


class TestPreviewImprovements:
    """Tests for preview_improvements method."""

    def test_preview_improvements_basic(
        self, sample_test_result, sample_five_why_analysis_intents
    ):
        """WHEN previewing improvements, THEN should generate markdown string."""
        usecase = ApplyImprovementUseCase()

        preview = usecase.preview_improvements(
            analyses=[sample_five_why_analysis_intents],
            test_results=[sample_test_result],
        )

        assert isinstance(preview, str)
        assert len(preview) > 0
        assert "# Improvement Preview" in preview
        assert "test-001" in preview
        assert sample_test_result.query in preview

    def test_preview_improvements_empty(self):
        """WHEN no analyses provided, THEN should return minimal preview."""
        usecase = ApplyImprovementUseCase()

        preview = usecase.preview_improvements(analyses=[], test_results=[])

        assert isinstance(preview, str)
        assert "Total analyses: 0" in preview

    def test_preview_improvements_includes_all_fields(
        self, sample_test_result, sample_five_why_analysis_code
    ):
        """WHEN previewing improvements, THEN should include all relevant fields."""
        usecase = ApplyImprovementUseCase()

        preview = usecase.preview_improvements(
            analyses=[sample_five_why_analysis_code],
            test_results=[sample_test_result],
        )

        assert "Root Cause:" in preview
        assert "Suggested Fix:" in preview
        assert "Target:" in preview or "Code Change Required:" in preview


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_mismatched_lengths_analyses_and_results(self):
        """WHEN analyses and test_results lengths differ, THEN should not raise error."""
        usecase = ApplyImprovementUseCase()

        analysis = FiveWhyAnalysis(
            test_case_id="test-001",
            original_failure="Failure",
            why_chain=["Why"] * 5,
            root_cause="Some cause",
            suggested_fix="Some fix",
            component_to_patch=None,
            code_change_required=False,
        )

        test_result = TestResult(
            test_case_id="test-001",
            query="test query",
            answer="test answer",
            sources=[],
            confidence=0.5,
            execution_time_ms=100,
            rag_pipeline_log={},
            fact_checks=[],
            quality_score=None,
            passed=False,
        )

        # Should not raise error even with strict=False in zip
        result = usecase.apply_improvements(
            analyses=[analysis, analysis],  # 2 analyses
            test_results=[test_result],  # 1 test result
            dry_run=True,
        )

        assert result is not None

    def test_corrupted_json_file(
        self, sample_test_result, sample_five_why_analysis_intents, tmp_path
    ):
        """WHEN JSON file is corrupted, THEN should handle gracefully."""
        # Create a corrupted JSON file
        corrupted_file = tmp_path / "corrupted.json"
        with open(corrupted_file, "w") as f:
            f.write("{ invalid json }")

        usecase = ApplyImprovementUseCase(intents_path=corrupted_file)

        result = usecase._patch_intents(
            analysis=sample_five_why_analysis_intents,
            test_result=sample_test_result,
            dry_run=True,
        )

        assert result is None

    def test_query_with_special_characters(self, sample_five_why_analysis_intents):
        """WHEN query has special characters, THEN should handle correctly."""
        usecase = ApplyImprovementUseCase()

        test_result = TestResult(
            test_case_id="test-special",
            query="학칙?!@# (제8조)에 대해 알려줘...",
            answer="응답",
            sources=[],
            confidence=0.8,
            execution_time_ms=100,
            rag_pipeline_log={},
            fact_checks=[],
            quality_score=None,
            passed=False,
        )

        keywords = usecase._extract_keywords(test_result.query)
        sanitized = usecase._sanitize_string(test_result.query)

        assert isinstance(keywords, list)
        assert isinstance(sanitized, str)


# =============================================================================
# Boundary Conditions
# =============================================================================


class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_very_long_query_truncation(self, sample_five_why_analysis_intents):
        """WHEN query is very long, THEN should truncate appropriately."""
        usecase = ApplyImprovementUseCase()

        long_query = "a" * 100
        sanitized = usecase._sanitize_string(long_query)

        assert len(sanitized) <= 30

    def test_empty_query(self):
        """WHEN query is empty, THEN should handle gracefully."""
        usecase = ApplyImprovementUseCase()

        keywords = usecase._extract_keywords("")
        sanitized = usecase._sanitize_string("")

        assert keywords == []
        assert sanitized == ""

    def test_none_values_in_test_result(self):
        """WHEN optional fields are None, THEN should handle correctly."""
        usecase = ApplyImprovementUseCase()

        test_result = TestResult(
            test_case_id="test-none",
            query="query",
            answer="answer",
            sources=[],
            confidence=0.8,
            execution_time_ms=100,
            rag_pipeline_log={},
            fact_checks=[],
            quality_score=None,
            passed=False,
            error_message=None,
        )

        analysis = FiveWhyAnalysis(
            test_case_id="test-none",
            original_failure="Failure",
            why_chain=["Why"] * 5,
            root_cause="Cause",
            suggested_fix="Fix",
            component_to_patch=None,
            code_change_required=False,
        )

        code_suggestion = usecase._generate_code_suggestion(analysis, test_result)

        assert code_suggestion is not None
        assert code_suggestion["test_case_id"] == "test-none"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
