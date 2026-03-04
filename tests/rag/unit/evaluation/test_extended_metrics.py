"""Unit tests for Extended Metrics."""

import pytest

from src.rag.domain.evaluation.extended_metrics import (
    CitationVerifier,
    CitationVerificationResult,
    ConsistencyChecker,
    ConsistencyResult,
    LatencyTracker,
    LatencySummary,
    ReadabilityScorer,
    ReadabilityResult,
)


class TestLatencyTracker:
    """Test LatencyTracker class."""

    def test_record_and_summary(self):
        tracker = LatencyTracker()
        tracker.record("q1", 100.0)
        tracker.record("q2", 200.0)
        tracker.record("q3", 300.0)

        summary = tracker.get_summary()
        assert summary.total_queries == 3
        assert summary.mean_ms == pytest.approx(200.0)
        assert summary.max_ms == 300.0

    def test_percentiles(self):
        tracker = LatencyTracker()
        for i in range(1, 101):
            tracker.record(f"q{i}", float(i))

        summary = tracker.get_summary()
        assert summary.p50_ms == pytest.approx(50.5, abs=1.0)
        assert summary.p95_ms == pytest.approx(95.05, abs=1.0)
        assert summary.p99_ms == pytest.approx(99.01, abs=1.0)

    def test_slow_query_detection(self):
        tracker = LatencyTracker()
        tracker.record("fast", 100.0)
        tracker.record("slow", 15000.0)

        summary = tracker.get_summary()
        assert len(summary.slow_queries) == 1
        assert summary.slow_queries[0]["query"] == "slow"

    def test_filter_by_persona(self):
        tracker = LatencyTracker()
        tracker.record("q1", 100.0, persona="student")
        tracker.record("q2", 200.0, persona="professor")
        tracker.record("q3", 150.0, persona="student")

        student_summary = tracker.get_summary(persona="student")
        assert student_summary.total_queries == 2

    def test_empty_summary(self):
        tracker = LatencyTracker()
        summary = tracker.get_summary()
        assert summary.total_queries == 0
        assert summary.mean_ms == 0.0

    def test_measure_with_function(self):
        tracker = LatencyTracker()

        def slow_fn(query):
            return f"answer to {query}"

        result = tracker.measure(slow_fn, "test query", persona="student")
        assert result == "answer to test query"
        assert len(tracker.records) == 1
        assert tracker.records[0].response_time_ms > 0


class TestConsistencyChecker:
    """Test ConsistencyChecker class."""

    def test_consistent_responses(self):
        checker = ConsistencyChecker()
        call_count = 0

        def stable_fn(query):
            return "same answer every time"

        result = checker.check("test", stable_fn, runs=3)
        assert result.passed is True
        assert result.min_similarity == 1.0
        assert len(result.responses) == 3

    def test_inconsistent_responses(self):
        responses = iter([
            "The answer is A",
            "Completely different response B",
            "Yet another totally different C",
        ])
        checker = ConsistencyChecker()

        def variable_fn(query):
            return next(responses)

        result = checker.check("test", variable_fn, runs=3)
        assert result.min_similarity < 1.0
        assert len(result.similarity_scores) == 3  # 3 pairwise comparisons

    def test_custom_similarity_fn(self):
        def always_half(a, b):
            return 0.5

        checker = ConsistencyChecker(similarity_fn=always_half)
        result = checker.check("test", lambda q: "answer", runs=3)
        assert result.min_similarity == 0.5
        assert result.passed is False

    def test_minimum_runs(self):
        """Runs should be at least 2."""
        checker = ConsistencyChecker()
        result = checker.check("test", lambda q: "answer", runs=1)
        assert len(result.responses) == 2  # Clamped to minimum

    def test_default_similarity(self):
        sim = ConsistencyChecker._default_similarity(
            "hello world foo", "hello world bar"
        )
        # hello, world shared; foo vs bar different
        # intersection=2, union=4 => 0.5
        assert sim == pytest.approx(0.5)

    def test_default_similarity_empty(self):
        assert ConsistencyChecker._default_similarity("", "") == 1.0
        assert ConsistencyChecker._default_similarity("hello", "") == 0.0


class TestCitationVerifier:
    """Test CitationVerifier class."""

    def test_extract_citations(self):
        verifier = CitationVerifier()
        text = "제3조에 따르면... 제15조의2 제2항에..."
        citations = verifier.extract_citations(text)
        assert len(citations) >= 2
        assert any("제3조" in c for c in citations)

    def test_extract_no_citations(self):
        verifier = CitationVerifier()
        assert verifier.extract_citations("no citations here") == []
        assert verifier.extract_citations("") == []

    def test_verify_all_exist(self):
        verifier = CitationVerifier(db_lookup_fn=lambda c: True)
        result = verifier.verify("query", "제1조와 제2조에 따르면")
        assert result.passed is True
        assert result.citation_existence_rate == 1.0
        assert len(result.hallucinated_citations) == 0

    def test_verify_some_missing(self):
        def lookup(citation):
            return "제1조" in citation

        verifier = CitationVerifier(db_lookup_fn=lookup)
        result = verifier.verify("query", "제1조와 제999조에 따르면")
        assert result.citation_existence_rate < 1.0
        assert len(result.hallucinated_citations) > 0

    def test_verify_no_citations_passes(self):
        """No citations in response means nothing to verify = pass."""
        verifier = CitationVerifier()
        result = verifier.verify("query", "답변입니다.")
        assert result.passed is True
        assert result.total_citations == 0

    def test_verify_without_lookup_fn(self):
        """Without a lookup function, assume all exist."""
        verifier = CitationVerifier()
        result = verifier.verify("query", "제5조에 따르면")
        assert result.passed is True
        assert result.verified_citations == result.total_citations


class TestReadabilityScorer:
    """Test ReadabilityScorer class."""

    def test_well_structured_response(self):
        scorer = ReadabilityScorer()
        response = (
            "## 졸업 요건\n\n"
            "졸업에 필요한 요건은 다음과 같습니다:\n\n"
            "1. 총 130학점 이상 이수\n"
            "2. 평점평균 2.0 이상\n"
            "3. 졸업논문 또는 졸업시험 통과\n\n"
            "자세한 내용은 학칙 제50조를 참고하시기 바랍니다."
        )
        result = scorer.score("졸업 요건", response)
        assert result.overall_score > 0.5
        assert isinstance(result, ReadabilityResult)

    def test_empty_response(self):
        scorer = ReadabilityScorer()
        result = scorer.score("질문", "")
        assert result.overall_score < 0.5
        assert result.passed is False

    def test_short_response(self):
        scorer = ReadabilityScorer()
        result = scorer.score("복잡한 질문", "네.")
        assert result.length_appropriateness < 1.0

    def test_structure_scoring_with_bullets(self):
        scorer = ReadabilityScorer()
        text = "- 항목 1\n- 항목 2\n- 항목 3"
        result = scorer.score("질문", text)
        assert result.structure_score > 0.0

    def test_to_dict(self):
        result = ReadabilityResult(
            query="q",
            structure_score=0.8,
            length_appropriateness=0.7,
            language_quality=0.9,
            overall_score=0.8,
            passed=True,
        )
        d = result.to_dict()
        assert d["query"] == "q"
        assert d["overall_score"] == 0.8
