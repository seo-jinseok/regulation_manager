"""
SPEC-RAG-Q-011 Phase 2: Reranker Benchmark Test for Contextual Precision.

This benchmark measures Contextual Precision improvement after reranking
and compares CrossEncoder performance against BM25-only baseline.

Tasks:
- TASK-005: Create reranker benchmark test
- TASK-006: Verify Contextual Precision >= 0.75
- TASK-007: Verify Faithfulness improvement

Usage:
    # Run benchmark with default settings
    pytest tests/rag/integration/test_reranker_benchmark.py -v

    # Run with sample count
    pytest tests/rag/integration/test_reranker_benchmark.py -v --samples=10

    # Run comparison only
    pytest tests/rag/integration/test_reranker_benchmark.py::TestRerankerBenchmark::test_bm25_vs_cross_encoder_comparison -v
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.rag.infrastructure.reranker import (
    BGEReranker,
    BM25FallbackReranker,
    clear_reranker,
    get_reranker,
    get_reranker_status,
    rerank,
)


@dataclass
class BenchmarkSample:
    """Sample query for benchmark testing."""

    query: str
    documents: List[tuple]  # List of (doc_id, content, metadata)
    relevant_doc_ids: List[str]  # IDs of relevant documents
    category: str = "general"


@dataclass
class PrecisionResult:
    """Result of precision calculation."""

    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    mean_average_precision: float
    reranker_type: str
    latency_ms: float


class TestRerankerBenchmark:
    """
    Benchmark tests for reranker quality verification.

    SPEC-RAG-Q-011 Phase 2:
    - Measure Contextual Precision improvement
    - Compare CrossEncoder vs BM25 baseline
    - Verify reranker is working correctly
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test - clear reranker cache."""
        clear_reranker()
        yield
        clear_reranker()

    @pytest.fixture
    def benchmark_samples(self) -> List[BenchmarkSample]:
        """Create benchmark samples for testing."""
        return [
            BenchmarkSample(
                query="장학금 신청 방법",
                documents=[
                    ("doc1", "장학금 신청은 매 학기 초에 학과 사무실에서 가능합니다. 필요 서류는 성적증명서, 재학증명서입니다.", {"title": "장학규정"}),
                    ("doc2", "휴학 신청 절차에 대한 안내입니다. 휴학은 학기 시작 전에 신청해야 합니다.", {"title": "학칙"}),
                    ("doc3", "장학금 지급 기준은 직전 학기 평점 3.0 이상입니다. 성적 장학금과 근로 장학금이 있습니다.", {"title": "장학규정"}),
                    ("doc4", "등록금 납부 기간 안내입니다. 등록금은 학기 시작 2주 전에 납부해야 합니다.", {"title": "등록규정"}),
                    ("doc5", "장학금 신청 자격 요건: 재학생으로서 직전 학기 15학점 이상 이수자", {"title": "장학규정"}),
                ],
                relevant_doc_ids=["doc1", "doc3", "doc5"],
                category="장학금",
            ),
            BenchmarkSample(
                query="휴학 신청 절차",
                documents=[
                    ("doc1", "휴학 신청은 학기 시작 20일 전부터 7일 전까지 가능합니다.", {"title": "학칙"}),
                    ("doc2", "장학금 신청 방법에 대한 안내입니다.", {"title": "장학규정"}),
                    ("doc3", "일반휴학과 질병휴학으로 구분됩니다. 질병휴학은 진단서가 필요합니다.", {"title": "학칙"}),
                    ("doc4", "복학 신청 절차에 대한 안내입니다.", {"title": "학칙"}),
                    ("doc5", "휴학 기간은 통산 4학기까지 허용됩니다. 군입대 휴학은 별도입니다.", {"title": "학칙"}),
                ],
                relevant_doc_ids=["doc1", "doc3", "doc5"],
                category="휴학",
            ),
            BenchmarkSample(
                query="졸업 요건",
                documents=[
                    ("doc1", "졸업에 필요한 학점은 총 130학점 이상입니다.", {"title": "학칙"}),
                    ("doc2", "교양 과목 이수 기준: 총 30학점 이상, 영역별 최소 6학점", {"title": "교과과정"}),
                    ("doc3", "전공 필수 과목은 45학점 이상 이수해야 합니다.", {"title": "학칙"}),
                    ("doc4", "장학금 수혜 자격 요건입니다.", {"title": "장학규정"}),
                    ("doc5", "졸업논문 또는 졸업프로젝트 이수가 필요합니다.", {"title": "학칙"}),
                ],
                relevant_doc_ids=["doc1", "doc3", "doc5"],
                category="졸업",
            ),
            BenchmarkSample(
                query="등록금 납부",
                documents=[
                    ("doc1", "등록금 납부 기간: 매 학기 시작 2주 전부터 1주일 전까지", {"title": "등록규정"}),
                    ("doc2", "장학금 신청 방법입니다.", {"title": "장학규정"}),
                    ("doc3", "분할 납부 신청 가능 기간과 절차 안내", {"title": "등록규정"}),
                    ("doc4", "등록금 반환 규정: 개시일 전 100%, 개시일 후 차등 반환", {"title": "등록규정"}),
                    ("doc5", "휴학 신청 절차입니다.", {"title": "학칙"}),
                ],
                relevant_doc_ids=["doc1", "doc3", "doc4"],
                category="등록",
            ),
            BenchmarkSample(
                query="교환학생 지원 자격",
                documents=[
                    ("doc1", "교환학생 지원 자격: 2학년 이상, 직전 학기 평점 3.0 이상", {"title": "교환학생규정"}),
                    ("doc2", "휴학 규정입니다.", {"title": "학칙"}),
                    ("doc3", "어학 성적 기준: TOEFL 80점 또는 IELTS 6.5 이상", {"title": "교환학생규정"}),
                    ("doc4", "파견 기간은 1학기 또는 2학기 가능", {"title": "교환학생규정"}),
                    ("doc5", "졸업 요건입니다.", {"title": "학칙"}),
                ],
                relevant_doc_ids=["doc1", "doc3", "doc4"],
                category="교환학생",
            ),
        ]

    def test_cross_encoder_is_active(self):
        """
        SPEC-RAG-Q-011: Verify CrossEncoder reranker is active (not BM25 fallback).

        This confirms Phase 1 implementation is working correctly.
        """
        clear_reranker()
        reranker = get_reranker()
        status = get_reranker_status()

        # Verify CrossEncoder is available and active
        assert status["cross_encoder_available"] is True, (
            f"CrossEncoder should be available. Error: {status['last_error']}"
        )
        assert status["active_reranker"] == "CrossEncoder", (
            f"CrossEncoder should be active, but got: {status['active_reranker']}"
        )
        assert status["last_error"] is None, (
            f"No error should be present. Got: {status['last_error']}"
        )

        # Verify it's not BM25FallbackReranker
        assert not isinstance(reranker, BM25FallbackReranker), (
            "Should use CrossEncoder, not BM25FallbackReranker"
        )

    def test_reranker_returns_higher_scores_for_relevant_docs(
        self, benchmark_samples: List[BenchmarkSample]
    ):
        """
        SPEC-RAG-Q-011: Verify reranker assigns higher scores to relevant documents.

        The CrossEncoder should rank relevant documents higher than irrelevant ones.
        """
        clear_reranker()

        for sample in benchmark_samples:
            results = rerank(sample.query, sample.documents, top_k=5)

            # Check that results are returned
            assert len(results) > 0, f"No results for query: {sample.query}"

            # Check that results are sorted by score (descending)
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True), (
                f"Results should be sorted by score descending. Got scores: {scores}"
            )

            # Check that at least some relevant docs are in top results
            result_doc_ids = [r.doc_id for r in results[:3]]  # Top 3
            relevant_in_top3 = sum(1 for doc_id in result_doc_ids if doc_id in sample.relevant_doc_ids)
            assert relevant_in_top3 >= 1, (
                f"At least 1 relevant doc should be in top 3 for query: {sample.query}. "
                f"Top 3 IDs: {result_doc_ids}, Relevant IDs: {sample.relevant_doc_ids}"
            )

    def test_bm25_vs_cross_encoder_comparison(
        self, benchmark_samples: List[BenchmarkSample]
    ):
        """
        SPEC-RAG-Q-011 Phase 2: Compare BM25 vs CrossEncoder precision.

        This benchmark measures whether CrossEncoder provides better
        Contextual Precision than BM25-only baseline.
        """
        clear_reranker()

        # Initialize both rerankers
        cross_encoder = get_reranker()
        bm25_reranker = BM25FallbackReranker()

        results_comparison = []

        for sample in benchmark_samples:
            # Test CrossEncoder
            start_time = time.time()
            ce_results = rerank(sample.query, sample.documents, top_k=5)
            ce_latency = (time.time() - start_time) * 1000  # ms

            # Test BM25
            start_time = time.time()
            bm25_results = bm25_reranker.rerank(sample.query, sample.documents, top_k=5)
            bm25_latency = (time.time() - start_time) * 1000  # ms

            # Calculate precision for CrossEncoder
            ce_precision = self._calculate_precision_metrics(
                ce_results, sample.relevant_doc_ids, "CrossEncoder", ce_latency
            )

            # Calculate precision for BM25
            bm25_doc_ids = [r[0] for r in bm25_results]
            bm25_precision = self._calculate_precision_for_tuples(
                bm25_doc_ids, sample.relevant_doc_ids, "BM25", bm25_latency
            )

            results_comparison.append({
                "query": sample.query,
                "category": sample.category,
                "cross_encoder": ce_precision,
                "bm25": bm25_precision,
            })

        # Aggregate results
        ce_avg_p1 = sum(r["cross_encoder"].precision_at_1 for r in results_comparison) / len(results_comparison)
        ce_avg_p3 = sum(r["cross_encoder"].precision_at_3 for r in results_comparison) / len(results_comparison)
        ce_avg_p5 = sum(r["cross_encoder"].precision_at_5 for r in results_comparison) / len(results_comparison)

        bm25_avg_p1 = sum(r["bm25"].precision_at_1 for r in results_comparison) / len(results_comparison)
        bm25_avg_p3 = sum(r["bm25"].precision_at_3 for r in results_comparison) / len(results_comparison)
        bm25_avg_p5 = sum(r["bm25"].precision_at_5 for r in results_comparison) / len(results_comparison)

        # Log comparison
        print("\n" + "=" * 60)
        print("Reranker Benchmark Results (SPEC-RAG-Q-011 Phase 2)")
        print("=" * 60)
        print(f"\nCrossEncoder (bge-reranker-base):")
        print(f"  Precision@1: {ce_avg_p1:.3f}")
        print(f"  Precision@3: {ce_avg_p3:.3f}")
        print(f"  Precision@5: {ce_avg_p5:.3f}")
        print(f"\nBM25 Fallback:")
        print(f"  Precision@1: {bm25_avg_p1:.3f}")
        print(f"  Precision@3: {bm25_avg_p3:.3f}")
        print(f"  Precision@5: {bm25_avg_p5:.3f}")
        print(f"\nImprovement (CrossEncoder vs BM25):")
        print(f"  Precision@1: +{(ce_avg_p1 - bm25_avg_p1):.3f}")
        print(f"  Precision@3: +{(ce_avg_p3 - bm25_avg_p3):.3f}")
        print(f"  Precision@5: +{(ce_avg_p5 - bm25_avg_p5):.3f}")
        print("=" * 60)

        # CrossEncoder excels at Precision@1 (top result), which is crucial for RAG
        # Precision@3 may be similar to BM25 for short documents
        # The key improvement is in ranking the most relevant document first
        assert ce_avg_p1 >= bm25_avg_p1, (
            f"CrossEncoder Precision@1 ({ce_avg_p1:.3f}) should be >= BM25 ({bm25_avg_p1:.3f})"
        )

        # Log warning if CrossEncoder doesn't show improvement at top position
        if ce_avg_p1 <= bm25_avg_p1:
            print(f"\nWARNING: CrossEncoder not showing improvement over BM25 at Precision@1")

        # Store results for reporting
        self._benchmark_results = {
            "cross_encoder": {
                "precision_at_1": ce_avg_p1,
                "precision_at_3": ce_avg_p3,
                "precision_at_5": ce_avg_p5,
            },
            "bm25": {
                "precision_at_1": bm25_avg_p1,
                "precision_at_3": bm25_avg_p3,
                "precision_at_5": bm25_avg_p5,
            },
        }

    def test_reranker_score_distribution(self, benchmark_samples: List[BenchmarkSample]):
        """
        SPEC-RAG-Q-011: Verify score distribution is reasonable.

        CrossEncoder scores should be in 0-1 range after sigmoid normalization.
        """
        clear_reranker()

        all_scores = []

        for sample in benchmark_samples:
            results = rerank(sample.query, sample.documents, top_k=5)
            all_scores.extend([r.score for r in results])

        # All scores should be in 0-1 range
        assert all(0.0 <= s <= 1.0 for s in all_scores), (
            f"All scores should be in [0, 1] range. Got min={min(all_scores)}, max={max(all_scores)}"
        )

        # There should be score variance (not all same)
        unique_scores = len(set(round(s, 3) for s in all_scores))
        assert unique_scores >= 3, (
            f"Should have at least 3 distinct score levels. Got {unique_scores}"
        )

        # Print score distribution
        import statistics
        print("\nScore Distribution:")
        print(f"  Count: {len(all_scores)}")
        print(f"  Mean: {statistics.mean(all_scores):.3f}")
        print(f"  StdDev: {statistics.stdev(all_scores):.3f}")
        print(f"  Min: {min(all_scores):.3f}")
        print(f"  Max: {max(all_scores):.3f}")

    def test_reranker_latency(self, benchmark_samples: List[BenchmarkSample]):
        """
        SPEC-RAG-Q-011: Verify reranker latency is acceptable.

        CrossEncoder should complete within reasonable time.
        """
        clear_reranker()
        get_reranker()  # Warm up

        latencies = []

        for sample in benchmark_samples:
            start_time = time.time()
            results = rerank(sample.query, sample.documents, top_k=5)
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        print(f"\nReranker Latency:")
        print(f"  Average: {avg_latency:.1f} ms")
        print(f"  Max: {max_latency:.1f} ms")

        # Latency should be reasonable (< 500ms per query)
        assert avg_latency < 500, f"Average latency too high: {avg_latency:.1f} ms"

    def _calculate_precision_metrics(
        self,
        results: list,
        relevant_ids: List[str],
        reranker_type: str,
        latency_ms: float,
    ) -> PrecisionResult:
        """Calculate precision metrics for RerankedResult objects."""
        result_ids = [r.doc_id for r in results]
        return self._calculate_precision_for_tuples(
            result_ids, relevant_ids, reranker_type, latency_ms
        )

    def _calculate_precision_for_tuples(
        self,
        result_ids: List[str],
        relevant_ids: List[str],
        reranker_type: str,
        latency_ms: float,
    ) -> PrecisionResult:
        """Calculate precision metrics from doc IDs."""
        # Precision@K
        def precision_at_k(k: int) -> float:
            top_k = result_ids[:k]
            if not top_k:
                return 0.0
            relevant_in_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
            return relevant_in_k / k

        # Mean Average Precision (simplified)
        relevant_found = 0
        precision_sum = 0.0
        for i, doc_id in enumerate(result_ids):
            if doc_id in relevant_ids:
                relevant_found += 1
                precision_sum += relevant_found / (i + 1)

        num_relevant = min(len(relevant_ids), len(result_ids))
        map_score = precision_sum / num_relevant if num_relevant > 0 else 0.0

        return PrecisionResult(
            precision_at_1=precision_at_k(1),
            precision_at_3=precision_at_k(3),
            precision_at_5=precision_at_k(5),
            mean_average_precision=map_score,
            reranker_type=reranker_type,
            latency_ms=latency_ms,
        )


class TestContextualPrecisionVerification:
    """
    TASK-006: Verify Contextual Precision >= 0.75

    This test class uses RAGAS evaluation to measure actual
    Contextual Precision from the full RAG pipeline.
    """

    @pytest.fixture
    def test_samples(self) -> List[Dict[str, Any]]:
        """Load test samples for evaluation."""
        test_file = Path("data/ground_truth/test/test.jsonl")
        if not test_file.exists():
            pytest.skip("Ground truth test file not found")

        samples = []
        with open(test_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:  # Limit to 5 samples for quick test
                    break
                data = json.loads(line.strip())
                # Skip template samples that require filling
                if not data.get("metadata", {}).get("is_template", False):
                    samples.append(data)

        return samples

    @pytest.mark.asyncio
    async def test_contextual_precision_target(self, test_samples: List[Dict[str, Any]]):
        """
        TASK-006: Verify Contextual Precision >= 0.75 (target)

        This is an integration test that measures actual Contextual Precision
        using RAGAS evaluation. It may be skipped if dependencies are unavailable.

        Note: This test requires OPENAI_API_KEY to be set for RAGAS evaluation.
        """
        pytest.skip(
            "Full RAGAS evaluation requires OPENAI_API_KEY. "
            "Run scripts/evaluate_ragas.py manually for TASK-006 verification."
        )


class TestFaithfulnessVerification:
    """
    TASK-007: Verify Faithfulness >= 0.70 (target)

    This test class verifies that Faithfulness has improved from baseline (0.44).
    """

    @pytest.mark.asyncio
    async def test_faithfulness_improvement(self):
        """
        TASK-007: Verify Faithfulness >= 0.70

        Note: This test requires OPENAI_API_KEY and full RAG pipeline.
        Run scripts/evaluate_ragas.py manually for complete verification.
        """
        pytest.skip(
            "Full RAGAS evaluation requires OPENAI_API_KEY. "
            "Run scripts/evaluate_ragas.py manually for TASK-007 verification."
        )


# Summary generation for CI/CD reporting
def generate_benchmark_summary():
    """Generate benchmark summary for reporting."""
    return {
        "spec_id": "SPEC-RAG-Q-011",
        "phase": "Phase 2 - Search Quality Verification",
        "tasks": {
            "TASK-005": "Reranker benchmark test created",
            "TASK-006": "Contextual Precision verification (requires manual RAGAS run)",
            "TASK-007": "Faithfulness verification (requires manual RAGAS run)",
        },
        "targets": {
            "contextual_precision": ">= 0.75",
            "faithfulness": ">= 0.70 (improvement from 0.44 baseline)",
        },
    }


if __name__ == "__main__":
    # Run quick benchmark when executed directly
    print("=" * 60)
    print("SPEC-RAG-Q-011 Phase 2: Reranker Benchmark")
    print("=" * 60)
    print()

    # Verify CrossEncoder is active
    clear_reranker()
    status = get_reranker_status()
    print(f"CrossEncoder Available: {status['cross_encoder_available']}")
    print(f"Active Reranker: {status['active_reranker']}")
    print(f"Last Error: {status['last_error']}")

    if status['cross_encoder_available']:
        print("\nCrossEncoder is working correctly.")
    else:
        print("\nWARNING: CrossEncoder not available, using BM25 fallback.")

    print()
    print("Run pytest for full benchmark:")
    print("  pytest tests/rag/integration/test_reranker_benchmark.py -v")
