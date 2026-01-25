"""
Tests for EvaluationUseCase.

Covers the evaluation system for measuring RAG quality.
"""

import json

import pytest

from src.rag.application.evaluate import (
    EvaluationResult,
    EvaluationSummary,
    EvaluationUseCase,
)
from src.rag.application.evaluate import TestCase as EvalTestCase


class MockSearchResult:
    """Mock search result for testing."""

    def __init__(self, score, chunk):
        self.score = score
        self.chunk = chunk


class MockChunk:
    """Mock chunk with rule_code."""

    def __init__(self, rule_code):
        self.rule_code = rule_code


class MockQueryRewriteInfo:
    """Mock query rewrite info."""

    def __init__(self, matched_intents=None, rewritten=""):
        self.matched_intents = matched_intents or []
        self.rewritten = rewritten


class MockSearchUseCase:
    """Mock search use case for testing."""

    def __init__(self, results=None, rewrite_info=None, should_raise=False):
        self._results = results or []
        self._rewrite_info = rewrite_info
        self._should_raise = should_raise
        self.last_query = None
        self.last_top_k = None

    def search(self, query_text, top_k=5):
        self.last_query = query_text
        self.last_top_k = top_k
        if self._should_raise:
            raise RuntimeError("Search failed")
        return self._results

    def get_last_query_rewrite(self):
        return self._rewrite_info


@pytest.fixture
def temp_dataset_path(tmp_path):
    """Create a temporary evaluation dataset file."""
    data = {
        "test_cases": [
            {
                "id": "test-001",
                "query": "교원인사규정 내용",
                "category": "인사",
                "expected_intents": ["regulation_search"],
                "expected_keywords": ["교원", "인사"],
                "expected_rule_codes": ["3-1-5"],
                "min_relevance_score": 0.3,
                "notes": "Basic regulation search",
            },
            {
                "id": "test-002",
                "query": "휴가 규정",
                "category": "인사",
                "expected_intents": [],
                "expected_keywords": ["휴가"],
                "expected_rule_codes": [],
                "min_relevance_score": 0.2,
                "notes": "Leave regulation search",
            },
        ]
    }
    dataset_path = tmp_path / "evaluation_dataset.json"
    dataset_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(dataset_path)


class TestEvaluationUseCaseInit:
    """Tests for EvaluationUseCase initialization."""

    def test_init_with_search_usecase_and_path(self, temp_dataset_path):
        """Test initialization with all parameters."""
        search_usecase = MockSearchUseCase()
        usecase = EvaluationUseCase(
            search_usecase=search_usecase, dataset_path=temp_dataset_path
        )
        assert usecase._search_usecase is search_usecase
        assert usecase._dataset_path == temp_dataset_path

    def test_init_without_search_usecase(self, temp_dataset_path):
        """Test initialization without search usecase."""
        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        assert usecase._search_usecase is None
        assert usecase._dataset_path == temp_dataset_path


class TestLoadDataset:
    """Tests for load_dataset method."""

    def test_load_dataset_success(self, temp_dataset_path):
        """Test successful dataset loading."""
        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        test_cases = usecase.load_dataset()

        assert len(test_cases) == 2
        assert test_cases[0].id == "test-001"
        assert test_cases[0].query == "교원인사규정 내용"
        assert test_cases[0].category == "인사"
        assert test_cases[0].expected_intents == ["regulation_search"]
        assert test_cases[0].expected_keywords == ["교원", "인사"]
        assert test_cases[0].expected_rule_codes == ["3-1-5"]
        assert test_cases[0].min_relevance_score == 0.3
        assert test_cases[0].notes == "Basic regulation search"

    def test_load_dataset_file_not_found(self, tmp_path):
        """Test loading non-existent dataset raises error."""
        usecase = EvaluationUseCase(dataset_path=str(tmp_path / "nonexistent.json"))
        with pytest.raises(FileNotFoundError, match="Evaluation dataset not found"):
            usecase.load_dataset()

    def test_load_dataset_stores_test_cases(self, temp_dataset_path):
        """Test that loaded test cases are stored internally."""
        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        test_cases = usecase.load_dataset()
        assert usecase._test_cases is test_cases
        assert len(usecase._test_cases) == 2


class TestEvaluateSingle:
    """Tests for evaluate_single method."""

    def test_evaluate_single_no_search_usecase(self, temp_dataset_path):
        """Test evaluation without search usecase returns error result."""
        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        usecase.load_dataset()

        test_case = EvalTestCase(
            id="test-001", query="test query", expected_keywords=["test"]
        )
        result = usecase.evaluate_single(test_case)

        assert result.passed is False
        assert result.error == "SearchUseCase not provided"
        assert result.intent_matched is False
        assert result.keyword_coverage == 0.0

    def test_evaluate_single_passing_case(self, temp_dataset_path):
        """Test evaluation of a passing test case."""
        chunk = MockChunk(rule_code="3-1-5")
        search_results = [
            MockSearchResult(score=0.8, chunk=chunk),
        ]
        rewrite_info = MockQueryRewriteInfo(
            matched_intents=["regulation_search"], rewritten="교원 인사규정"
        )
        search_usecase = MockSearchUseCase(
            results=search_results, rewrite_info=rewrite_info
        )

        usecase = EvaluationUseCase(search_usecase=search_usecase)
        test_case = EvalTestCase(
            id="test-001",
            query="교원인사규정",
            expected_intents=["regulation_search"],
            expected_keywords=["교원", "인사"],
            expected_rule_codes=["3-1-5"],
            min_relevance_score=0.3,
        )

        result = usecase.evaluate_single(test_case)

        assert result.passed is True
        assert result.intent_matched is True
        assert result.keyword_coverage >= 0.5
        assert result.rule_code_matched is True
        assert result.top_score == 0.8
        assert result.matched_intents == ["regulation_search"]
        assert "교원" in result.found_keywords or "인사" in result.found_keywords
        assert result.error is None

    def test_evaluate_single_failing_low_score(self, temp_dataset_path):
        """Test evaluation fails when relevance score is too low."""
        chunk = MockChunk(rule_code="3-1-5")
        search_results = [
            MockSearchResult(score=0.1, chunk=chunk),
        ]
        rewrite_info = MockQueryRewriteInfo(
            matched_intents=["regulation_search"], rewritten="교원 인사규정"
        )
        search_usecase = MockSearchUseCase(
            results=search_results, rewrite_info=rewrite_info
        )

        usecase = EvaluationUseCase(search_usecase=search_usecase)
        test_case = EvalTestCase(
            id="test-001",
            query="test",
            expected_intents=["regulation_search"],
            expected_keywords=["교원"],
            min_relevance_score=0.5,
        )

        result = usecase.evaluate_single(test_case)

        assert result.passed is False
        assert result.top_score == 0.1

    def test_evaluate_single_keyword_coverage(self, temp_dataset_path):
        """Test keyword coverage calculation."""
        rewrite_info = MockQueryRewriteInfo(rewritten="교원 인사 규정")
        search_usecase = MockSearchUseCase(results=[], rewrite_info=rewrite_info)

        usecase = EvaluationUseCase(search_usecase=search_usecase)
        test_case = EvalTestCase(
            id="test-001",
            query="test",
            expected_keywords=["교원", "인사", "휴가"],
        )

        result = usecase.evaluate_single(test_case)

        # "교원 인사 규정" contains "교원" and "인사" but not "휴가"
        # Coverage should be 2/3 = 0.667
        assert result.keyword_coverage == 2.0 / 3.0

    def test_evaluate_single_no_keywords_expected(self, temp_dataset_path):
        """Test evaluation with no expected keywords returns 1.0 coverage."""
        rewrite_info = MockQueryRewriteInfo(rewritten="test")
        search_usecase = MockSearchUseCase(results=[], rewrite_info=rewrite_info)

        usecase = EvaluationUseCase(search_usecase=search_usecase)
        test_case = EvalTestCase(
            id="test-001",
            query="test",
            expected_keywords=[],
        )

        result = usecase.evaluate_single(test_case)

        assert result.keyword_coverage == 1.0

    def test_evaluate_single_no_results(self, temp_dataset_path):
        """Test evaluation with no search results."""
        rewrite_info = MockQueryRewriteInfo(rewritten="test")
        search_usecase = MockSearchUseCase(results=[], rewrite_info=rewrite_info)

        usecase = EvaluationUseCase(search_usecase=search_usecase)
        test_case = EvalTestCase(
            id="test-001",
            query="test",
            expected_keywords=[],
            min_relevance_score=0.0,
        )

        result = usecase.evaluate_single(test_case)

        assert result.top_score == 0.0

    def test_evaluate_single_exception_handling(self, temp_dataset_path):
        """Test evaluation handles exceptions gracefully."""
        search_usecase = MockSearchUseCase(should_raise=True)

        usecase = EvaluationUseCase(search_usecase=search_usecase)
        test_case = EvalTestCase(id="test-001", query="test")

        result = usecase.evaluate_single(test_case)

        assert result.passed is False
        assert result.error == "Search failed"
        assert result.intent_matched is False
        assert result.keyword_coverage == 0.0

    def test_evaluate_single_intent_matching(self, temp_dataset_path):
        """Test intent matching logic."""
        rewrite_info = MockQueryRewriteInfo(matched_intents=["regulation_search"])
        search_usecase = MockSearchUseCase(results=[], rewrite_info=rewrite_info)

        usecase = EvaluationUseCase(search_usecase=search_usecase)
        test_case = EvalTestCase(
            id="test-001",
            query="test",
            expected_intents=["regulation_search"],
        )

        result = usecase.evaluate_single(test_case)

        assert result.intent_matched is True

    def test_evaluate_single_intent_no_match(self, temp_dataset_path):
        """Test intent not matched."""
        rewrite_info = MockQueryRewriteInfo(matched_intents=["other_intent"])
        search_usecase = MockSearchUseCase(results=[], rewrite_info=rewrite_info)

        usecase = EvaluationUseCase(search_usecase=search_usecase)
        test_case = EvalTestCase(
            id="test-001",
            query="test",
            expected_intents=["regulation_search"],
        )

        result = usecase.evaluate_single(test_case)

        assert result.intent_matched is False

    def test_evaluate_single_intent_case_insensitive(self, temp_dataset_path):
        """Test intent matching is case-insensitive."""
        rewrite_info = MockQueryRewriteInfo(matched_intents=["Regulation_Search"])
        search_usecase = MockSearchUseCase(results=[], rewrite_info=rewrite_info)

        usecase = EvaluationUseCase(search_usecase=search_usecase)
        test_case = EvalTestCase(
            id="test-001",
            query="test",
            expected_intents=["regulation_search"],
        )

        result = usecase.evaluate_single(test_case)

        assert result.intent_matched is True

    def test_evaluate_single_rule_code_matching(self, temp_dataset_path):
        """Test rule code matching."""
        chunk = MockChunk(rule_code="3-1-5")
        search_results = [MockSearchResult(score=0.8, chunk=chunk)]
        search_usecase = MockSearchUseCase(results=search_results)

        usecase = EvaluationUseCase(search_usecase=search_usecase)
        test_case = EvalTestCase(
            id="test-001",
            query="test",
            expected_rule_codes=["3-1-5", "3-1-6"],
        )

        result = usecase.evaluate_single(test_case)

        assert result.rule_code_matched is True
        assert result.found_rule_codes == ["3-1-5"]

    def test_evaluate_single_uses_top_k(self, temp_dataset_path):
        """Test that evaluate_single passes top_k to search."""
        search_results = [
            MockSearchResult(score=0.8, chunk=MockChunk(rule_code="3-1-5")),
        ]
        search_usecase = MockSearchUseCase(results=search_results)

        usecase = EvaluationUseCase(search_usecase=search_usecase)
        test_case = EvalTestCase(id="test-001", query="test")

        usecase.evaluate_single(test_case, top_k=10)

        assert search_usecase.last_top_k == 10


class TestRunEvaluation:
    """Tests for run_evaluation method."""

    def test_run_evaluation_all_cases(self, temp_dataset_path):
        """Test running evaluation on all test cases."""
        chunk = MockChunk(rule_code="3-1-5")
        search_results = [MockSearchResult(score=0.8, chunk=chunk)]
        rewrite_info = MockQueryRewriteInfo(
            matched_intents=["regulation_search"], rewritten="교원 인사"
        )
        search_usecase = MockSearchUseCase(
            results=search_results, rewrite_info=rewrite_info
        )

        usecase = EvaluationUseCase(
            search_usecase=search_usecase, dataset_path=temp_dataset_path
        )
        usecase.load_dataset()

        summary = usecase.run_evaluation(top_k=5)

        assert summary.total_cases == 2
        assert summary.passed_cases + summary.failed_cases == 2
        assert len(summary.results) == 2

    def test_run_evaluation_with_category_filter(self, temp_dataset_path):
        """Test running evaluation with category filter."""
        chunk = MockChunk(rule_code="3-1-5")
        search_results = [MockSearchResult(score=0.8, chunk=chunk)]
        search_usecase = MockSearchUseCase(results=search_results)

        usecase = EvaluationUseCase(
            search_usecase=search_usecase, dataset_path=temp_dataset_path
        )
        usecase.load_dataset()

        summary = usecase.run_evaluation(category="인사")

        assert summary.total_cases == 2  # Both cases have category "인사"

    def test_run_evaluation_auto_loads_dataset(self, temp_dataset_path):
        """Test that run_evaluation auto-loads dataset if not loaded."""
        chunk = MockChunk(rule_code="3-1-5")
        search_results = [MockSearchResult(score=0.8, chunk=chunk)]
        search_usecase = MockSearchUseCase(results=search_results)

        usecase = EvaluationUseCase(
            search_usecase=search_usecase, dataset_path=temp_dataset_path
        )

        # Don't call load_dataset() explicitly
        summary = usecase.run_evaluation()

        # Should auto-load
        assert summary.total_cases == 2

    def test_run_evaluation_calculates_summary_metrics(self, temp_dataset_path):
        """Test that run_evaluation calculates correct summary metrics."""
        chunk = MockChunk(rule_code="3-1-5")
        search_results = [MockSearchResult(score=0.8, chunk=chunk)]
        rewrite_info = MockQueryRewriteInfo(
            matched_intents=["regulation_search"], rewritten="교원 인사"
        )
        search_usecase = MockSearchUseCase(
            results=search_results, rewrite_info=rewrite_info
        )

        usecase = EvaluationUseCase(
            search_usecase=search_usecase, dataset_path=temp_dataset_path
        )
        usecase.load_dataset()

        summary = usecase.run_evaluation()

        # Check pass rate calculation
        expected_rate = summary.passed_cases / summary.total_cases
        assert summary.pass_rate == expected_rate

        # Check intent accuracy
        assert summary.intent_accuracy > 0

        # Check keyword coverage average
        assert summary.keyword_coverage_avg >= 0


class TestFormatMethods:
    """Tests for format_summary and format_details methods."""

    def test_format_summary(self, temp_dataset_path):
        """Test formatting evaluation summary."""
        summary = EvaluationSummary(
            total_cases=10,
            passed_cases=8,
            failed_cases=2,
            pass_rate=0.8,
            intent_accuracy=0.9,
            keyword_coverage_avg=0.75,
            rule_code_accuracy=0.85,
            results=[],
        )

        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        formatted = usecase.format_summary(summary)

        assert "RAG 시스템 평가 결과" in formatted
        assert "총 테스트 케이스: 10" in formatted
        assert "통과: 8 | 실패: 2" in formatted
        assert "통과율: 80.0%" in formatted
        assert "의도 인식 정확도: 90.0%" in formatted
        assert "키워드 커버리지: 75.0%" in formatted
        assert "규정 코드 정확도: 85.0%" in formatted

    def test_format_details(self, temp_dataset_path):
        """Test formatting detailed results."""
        test_case = EvalTestCase(
            id="test-001",
            query="교원인사규정",
            category="인사",
        )
        result = EvaluationResult(
            test_case=test_case,
            passed=True,
            intent_matched=True,
            keyword_coverage=0.8,
            rule_code_matched=True,
            top_score=0.85,
            matched_intents=["regulation_search"],
            found_keywords=["교원", "인사"],
            found_rule_codes=["3-1-5"],
        )

        summary = EvaluationSummary(
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            pass_rate=1.0,
            intent_accuracy=1.0,
            keyword_coverage_avg=0.8,
            rule_code_accuracy=1.0,
            results=[result],
        )

        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        formatted = usecase.format_details(summary)

        assert "[1] test-001: ✅ PASS" in formatted
        assert "쿼리: 교원인사규정" in formatted
        assert "의도 매칭: ✓" in formatted
        assert "키워드 커버리지: 80%" in formatted
        assert "규정 코드: ✓" in formatted
        assert "Top Score: 0.850" in formatted

    def test_format_details_with_error(self, temp_dataset_path):
        """Test formatting details with error result."""
        test_case = EvalTestCase(id="test-001", query="test")
        result = EvaluationResult(
            test_case=test_case,
            passed=False,
            intent_matched=False,
            keyword_coverage=0.0,
            rule_code_matched=False,
            top_score=0.0,
            error="Search failed",
        )

        summary = EvaluationSummary(
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
            pass_rate=0.0,
            intent_accuracy=0.0,
            keyword_coverage_avg=0.0,
            rule_code_accuracy=0.0,
            results=[result],
        )

        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        formatted = usecase.format_details(summary)

        assert "[1] test-001: ❌ FAIL" in formatted
        assert "에러: Search failed" in formatted


class TestHelperMethods:
    """Tests for helper methods."""

    def test_check_intent_match_empty_expected(self, temp_dataset_path):
        """Test _check_intent_match with empty expected returns True."""
        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        result = usecase._check_intent_match([], ["any"])
        assert result is True

    def test_check_intent_match_empty_matched(self, temp_dataset_path):
        """Test _check_intent_match with empty matched."""
        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        result = usecase._check_intent_match(["expected"], [])
        assert result is False

    def test_check_intent_match_intersection(self, temp_dataset_path):
        """Test _check_intent_match finds intersection."""
        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        result = usecase._check_intent_match(
            ["regulation_search", "other"], ["regulation_search", "another"]
        )
        assert result is True

    def test_find_keywords(self, temp_dataset_path):
        """Test _find_keywords returns matching keywords."""
        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        result = usecase._find_keywords(["교원", "인사", "휴가"], "교원인사규정")
        assert "교원" in result
        assert "인사" in result
        assert "휴가" not in result

    def test_find_keywords_case_insensitive(self, temp_dataset_path):
        """Test _find_keywords is case-insensitive."""
        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        result = usecase._find_keywords(["TEST"], "this is a test")
        assert "TEST" in result

    def test_check_rule_code_match_empty_expected(self, temp_dataset_path):
        """Test _check_rule_code_match with empty expected returns True."""
        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        result = usecase._check_rule_code_match([], ["any"])
        assert result is True

    def test_check_rule_code_match_intersection(self, temp_dataset_path):
        """Test _check_rule_code_match finds intersection."""
        usecase = EvaluationUseCase(dataset_path=temp_dataset_path)
        result = usecase._check_rule_code_match(["3-1-5", "3-1-6"], ["3-1-7", "3-1-5"])
        assert result is True
