"""
Evaluation Use Case for RAG System Quality Assessment.

Provides automated evaluation of search quality against a test dataset.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class TestCase:
    """A single test case for evaluation."""

    id: str
    query: str
    category: str = ""
    expected_intents: List[str] = field(default_factory=list)
    expected_keywords: List[str] = field(default_factory=list)
    expected_rule_codes: List[str] = field(default_factory=list)
    min_relevance_score: float = 0.3
    notes: str = ""


@dataclass
class EvaluationResult:
    """Result of evaluating a single test case."""

    test_case: TestCase
    passed: bool
    intent_matched: bool
    keyword_coverage: float  # 0.0 ~ 1.0
    rule_code_matched: bool
    top_score: float
    matched_intents: List[str] = field(default_factory=list)
    found_keywords: List[str] = field(default_factory=list)
    found_rule_codes: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class EvaluationSummary:
    """Summary of all evaluation results."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    intent_accuracy: float
    keyword_coverage_avg: float
    rule_code_accuracy: float
    results: List[EvaluationResult] = field(default_factory=list)


class EvaluationUseCase:
    """
    Use case for evaluating RAG system quality.

    Loads test cases from evaluation_dataset.json and runs them
    against the search system to measure quality metrics.
    """

    def __init__(
        self,
        search_usecase: Optional[Any] = None,
        dataset_path: Optional[str] = None,
    ):
        """
        Initialize evaluation use case.

        Args:
            search_usecase: SearchUseCase instance for running queries.
            dataset_path: Path to evaluation_dataset.json.
        """
        self._search_usecase = search_usecase
        self._dataset_path = dataset_path or self._default_dataset_path()
        self._test_cases: List[TestCase] = []

    def _default_dataset_path(self) -> str:
        """Get default path to evaluation dataset."""
        return str(
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "config"
            / "evaluation_dataset.json"
        )

    def load_dataset(self) -> List[TestCase]:
        """
        Load test cases from the evaluation dataset.

        Returns:
            List of TestCase objects.
        """
        path = Path(self._dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {path}")

        data = json.loads(path.read_text(encoding="utf-8"))
        test_cases = []

        for item in data.get("test_cases", []):
            test_cases.append(
                TestCase(
                    id=item.get("id", ""),
                    query=item.get("query", ""),
                    category=item.get("category", ""),
                    expected_intents=item.get("expected_intents", []),
                    expected_keywords=item.get("expected_keywords", []),
                    expected_rule_codes=item.get("expected_rule_codes", []),
                    min_relevance_score=item.get("min_relevance_score", 0.3),
                    notes=item.get("notes", ""),
                )
            )

        self._test_cases = test_cases
        return test_cases

    def evaluate_single(
        self,
        test_case: TestCase,
        top_k: int = 5,
    ) -> EvaluationResult:
        """
        Evaluate a single test case.

        Args:
            test_case: The test case to evaluate.
            top_k: Number of results to consider.

        Returns:
            EvaluationResult with metrics.
        """
        if self._search_usecase is None:
            return EvaluationResult(
                test_case=test_case,
                passed=False,
                intent_matched=False,
                keyword_coverage=0.0,
                rule_code_matched=False,
                top_score=0.0,
                error="SearchUseCase not provided",
            )

        try:
            # Run search
            results = self._search_usecase.search(
                query_text=test_case.query,
                top_k=top_k,
            )

            # Get query rewrite info for intent matching
            rewrite_info = self._search_usecase.get_last_query_rewrite()
            matched_intents = (
                rewrite_info.matched_intents if rewrite_info else []
            ) or []

            # Check intent matching
            intent_matched = self._check_intent_match(
                test_case.expected_intents, matched_intents
            )

            # Check keyword coverage
            rewritten_query = (
                rewrite_info.rewritten if rewrite_info else test_case.query
            )
            found_keywords = self._find_keywords(
                test_case.expected_keywords, rewritten_query
            )
            keyword_coverage = (
                len(found_keywords) / len(test_case.expected_keywords)
                if test_case.expected_keywords
                else 1.0
            )

            # Check rule code matching
            found_rule_codes = [r.chunk.rule_code for r in results if r.chunk]
            rule_code_matched = self._check_rule_code_match(
                test_case.expected_rule_codes, found_rule_codes
            )

            # Get top score
            top_score = results[0].score if results else 0.0

            # Determine if passed
            passed = (
                (not test_case.expected_intents or intent_matched)
                and keyword_coverage >= 0.5
                and (not test_case.expected_rule_codes or rule_code_matched)
                and top_score >= test_case.min_relevance_score
            )

            return EvaluationResult(
                test_case=test_case,
                passed=passed,
                intent_matched=intent_matched,
                keyword_coverage=keyword_coverage,
                rule_code_matched=rule_code_matched,
                top_score=top_score,
                matched_intents=list(matched_intents),
                found_keywords=found_keywords,
                found_rule_codes=found_rule_codes[:3],
            )

        except Exception as e:
            return EvaluationResult(
                test_case=test_case,
                passed=False,
                intent_matched=False,
                keyword_coverage=0.0,
                rule_code_matched=False,
                top_score=0.0,
                error=str(e),
            )

    def _check_intent_match(self, expected: List[str], matched: List[str]) -> bool:
        """Check if any expected intent is in matched intents."""
        if not expected:
            return True
        expected_lower = {e.lower() for e in expected}
        matched_lower = {m.lower() for m in matched}
        return bool(expected_lower & matched_lower)

    def _find_keywords(self, expected: List[str], text: str) -> List[str]:
        """Find which expected keywords appear in text."""
        text_lower = text.lower()
        return [kw for kw in expected if kw.lower() in text_lower]

    def _check_rule_code_match(self, expected: List[str], found: List[str]) -> bool:
        """Check if any expected rule code is in found codes."""
        if not expected:
            return True
        return bool(set(expected) & set(found))

    def run_evaluation(
        self,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> EvaluationSummary:
        """
        Run evaluation on all test cases.

        Args:
            top_k: Number of results to consider per query.
            category: Optional category filter.

        Returns:
            EvaluationSummary with overall metrics.
        """
        if not self._test_cases:
            self.load_dataset()

        test_cases = self._test_cases
        if category:
            test_cases = [tc for tc in test_cases if tc.category == category]

        results: List[EvaluationResult] = []
        for tc in test_cases:
            result = self.evaluate_single(tc, top_k=top_k)
            results.append(result)

        # Calculate summary metrics
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        intent_correct = sum(1 for r in results if r.intent_matched)
        rule_code_correct = sum(1 for r in results if r.rule_code_matched)
        keyword_coverage_sum = sum(r.keyword_coverage for r in results)

        return EvaluationSummary(
            total_cases=total,
            passed_cases=passed,
            failed_cases=total - passed,
            pass_rate=passed / total if total > 0 else 0.0,
            intent_accuracy=intent_correct / total if total > 0 else 0.0,
            keyword_coverage_avg=keyword_coverage_sum / total if total > 0 else 0.0,
            rule_code_accuracy=rule_code_correct / total if total > 0 else 0.0,
            results=results,
        )

    def format_summary(self, summary: EvaluationSummary) -> str:
        """Format evaluation summary as a readable string."""
        lines = [
            "=" * 60,
            "RAG 시스템 평가 결과",
            "=" * 60,
            f"총 테스트 케이스: {summary.total_cases}",
            f"통과: {summary.passed_cases} | 실패: {summary.failed_cases}",
            f"통과율: {summary.pass_rate:.1%}",
            "-" * 60,
            f"의도 인식 정확도: {summary.intent_accuracy:.1%}",
            f"키워드 커버리지: {summary.keyword_coverage_avg:.1%}",
            f"규정 코드 정확도: {summary.rule_code_accuracy:.1%}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def format_details(self, summary: EvaluationSummary) -> str:
        """Format detailed results for each test case."""
        lines = []
        for i, result in enumerate(summary.results, 1):
            status = "✅ PASS" if result.passed else "❌ FAIL"
            lines.append(f"\n[{i}] {result.test_case.id}: {status}")
            lines.append(f"    쿼리: {result.test_case.query}")
            lines.append(
                f"    의도 매칭: {'✓' if result.intent_matched else '✗'} {result.matched_intents}"
            )
            lines.append(
                f"    키워드 커버리지: {result.keyword_coverage:.0%} {result.found_keywords}"
            )
            lines.append(
                f"    규정 코드: {'✓' if result.rule_code_matched else '✗'} {result.found_rule_codes}"
            )
            lines.append(f"    Top Score: {result.top_score:.3f}")
            if result.error:
                lines.append(f"    에러: {result.error}")
        return "\n".join(lines)
