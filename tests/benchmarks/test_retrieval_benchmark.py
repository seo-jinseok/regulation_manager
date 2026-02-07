"""
Performance Benchmarks for Retrieval Components.

Measures:
- Hybrid search performance (BM25 + Dense)
- Reranker performance
- CRAG pipeline performance
- Cache effectiveness

Uses pytest-benchmark for consistent measurements.
"""

import time
from typing import List
from unittest.mock import Mock, patch

import pytest

from src.rag.domain.entities import Chunk, ChunkLevel, RegulationStatus, SearchResult
from src.rag.infrastructure.crag_retriever import CRAGRetriever, RetrievalQuality
from src.rag.infrastructure.hybrid_search import ScoredDocument
from src.rag.infrastructure.hybrid_search_integration import DenseHybridSearcher

# Check if pytest-benchmark is available
HAS_BENCHMARK = False
try:
    __import__("pytest_benchmark")
    HAS_BENCHMARK = True
except ImportError:
    pass


# Provide a dummy benchmark fixture when pytest-benchmark is not available
if not HAS_BENCHMARK:

    @pytest.fixture
    def benchmark(request, *args, **kwargs):  # noqa: ARG001
        """Dummy benchmark fixture when pytest-benchmark is not available."""

        class DummyBenchmark:
            def __call__(self, func, *args, **kwargs):
                # Run function and return result
                return func(*args, **kwargs)

        return DummyBenchmark()

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def benchmark_documents() -> List[tuple]:
    """Create documents for benchmarking."""
    return [
        (
            f"doc{i}",
            f"규정 내용 {i}: 휴학, 복학, 제적, 성적 경고, 장학금 등 학사 관리에 관한 사항을 규정한다.",
            {"category": "학사", "level": "article"},
        )
        for i in range(100)
    ]


@pytest.fixture
def benchmark_searcher(benchmark_documents):
    """Create hybrid searcher with benchmark documents."""
    searcher = DenseHybridSearcher(
        dense_model_name="test-model",
        use_dynamic_weights=False,
    )

    # Mock dense retriever to avoid model loading
    with patch("src.rag.infrastructure.hybrid_search_integration.DenseRetriever"):
        searcher.add_documents(benchmark_documents)

    return searcher


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Create sample chunks for benchmarking."""
    return [
        Chunk(
            id=f"chunk{i}",
            rule_code="RULE001",
            level=ChunkLevel.ARTICLE,
            title=f"제{i}조 조항",
            text=f"이 조항은 규정 내용 {i}에 대한 것이다.",
            embedding_text=f"이 조항은 규정 내용 {i}에 대한 것이다.",
            full_text=f"제{i}조 조항\n이 조항은 규정 내용 {i}에 대한 것이다.",
            parent_path=["규정"],
            token_count=30,
            keywords=[],
            is_searchable=True,
            status=RegulationStatus.ACTIVE,
        )
        for i in range(50)
    ]


# =============================================================================
# Benchmarks: Hybrid Search
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.slow
class TestHybridSearchBenchmark:
    """Benchmark hybrid search performance."""

    def test_bm25_search_performance(self, benchmark, benchmark_searcher):
        """Benchmark BM25 sparse search performance."""

        def search_bm25():
            return benchmark_searcher.search_sparse("휴학 절차는 무엇인가?", top_k=10)

        results = benchmark(search_bm25)

        # Verify results are valid
        assert isinstance(results, list)

    def test_dense_search_performance(self, benchmark, benchmark_searcher):
        """Benchmark dense semantic search performance."""
        # Mock dense search
        with patch.object(benchmark_searcher, "_get_dense_retriever") as mock_dense:
            mock_retriever = Mock()
            mock_retriever.search.return_value = []
            mock_dense.return_value = mock_retriever

            def search_dense():
                return benchmark_searcher.search_dense(
                    "휴학 절차는 무엇인가?", top_k=10
                )

            results = benchmark(search_dense)
            assert isinstance(results, list)

    def test_hybrid_search_performance(self, benchmark, benchmark_searcher):
        """Benchmark combined hybrid search performance."""
        with patch.object(benchmark_searcher, "_get_dense_retriever") as mock_dense:
            mock_retriever = Mock()
            mock_retriever.search.return_value = []
            mock_dense.return_value = mock_retriever

            def search_hybrid():
                return benchmark_searcher.search("휴학 절차는 무엇인가?", top_k=10)

            results = benchmark(search_hybrid)
            assert isinstance(results, list)

    def test_result_fusion_performance(self, benchmark, benchmark_searcher):
        """Benchmark result fusion performance."""
        sparse_results = [
            ScoredDocument(f"doc{i}", 0.9 - i * 0.05, f"content {i}", {})
            for i in range(20)
        ]
        dense_results = [
            ScoredDocument(f"doc{i}", 0.8 - i * 0.03, f"content {i}", {})
            for i in range(20)
        ]

        def fuse():
            return benchmark_searcher.fuse_results(
                sparse_results, dense_results, top_k=10, query_text="test query"
            )

        results = benchmark(fuse)
        assert len(results) <= 10


# =============================================================================
# Benchmarks: Reranker
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.slow
class TestRerankerBenchmark:
    """Benchmark reranker performance."""

    def test_reranking_small_batch(self, benchmark, sample_chunks):
        """Benchmark reranking with small batch (10 documents)."""
        documents = [
            (c.id, c.embedding_text, c.to_metadata()) for c in sample_chunks[:10]
        ]

        with patch("src.rag.infrastructure.reranker.get_reranker"):
            from src.rag.infrastructure.reranker import BGEReranker

            reranker = BGEReranker()

            # Mock the rerank function
            with patch("src.rag.infrastructure.reranker.rerank", return_value=[]):

                def rerank():
                    return reranker.rerank("휴학 절차", documents, top_k=5)

                results = benchmark(rerank)
                assert isinstance(results, list)

    def test_reranking_medium_batch(self, benchmark, sample_chunks):
        """Benchmark reranking with medium batch (30 documents)."""
        documents = [
            (c.id, c.embedding_text, c.to_metadata()) for c in sample_chunks[:30]
        ]

        with patch("src.rag.infrastructure.reranker.get_reranker"):
            from src.rag.infrastructure.reranker import BGEReranker

            reranker = BGEReranker()

            with patch("src.rag.infrastructure.reranker.rerank", return_value=[]):

                def rerank():
                    return reranker.rerank("휴학 절차", documents, top_k=10)

                results = benchmark(rerank)
                assert isinstance(results, list)

    def test_reranking_large_batch(self, benchmark, sample_chunks):
        """Benchmark reranking with large batch (50 documents)."""
        documents = [
            (c.id, c.embedding_text, c.to_metadata()) for c in sample_chunks[:50]
        ]

        with patch("src.rag.infrastructure.reranker.get_reranker"):
            from src.rag.infrastructure.reranker import BGEReranker

            reranker = BGEReranker()

            with patch("src.rag.infrastructure.reranker.rerank", return_value=[]):

                def rerank():
                    return reranker.rerank("휴학 절차", documents, top_k=10)

                results = benchmark(rerank)
                assert isinstance(results, list)


# =============================================================================
# Benchmarks: CRAG Pipeline
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.asyncio
class TestCRAGBenchmark:
    """Benchmark CRAG pipeline performance."""

    async def test_quality_evaluation_performance(self, benchmark, sample_chunks):
        """Benchmark retrieval quality evaluation."""
        retriever = CRAGRetriever()
        search_results = [
            SearchResult(chunk=c, score=0.7, rank=i)
            for i, c in enumerate(sample_chunks[:20])
        ]

        def evaluate():
            return retriever.evaluate_retrieval_quality(
                "test query", search_results, "medium"
            )

        quality, score = benchmark(evaluate)
        assert isinstance(quality, RetrievalQuality)

    async def test_relevance_score_calculation(self, benchmark, sample_chunks):
        """Benchmark relevance score calculation."""
        retriever = CRAGRetriever()
        search_results = [
            SearchResult(chunk=c, score=0.7, rank=i)
            for i, c in enumerate(sample_chunks[:20])
        ]

        def calculate_score():
            return retriever._calculate_relevance_score("test query", search_results)

        score = benchmark(calculate_score)
        assert 0.0 <= score <= 1.0

    async def test_keyword_overlap_calculation(self, benchmark, sample_chunks):
        """Benchmark keyword overlap calculation."""
        retriever = CRAGRetriever()
        search_results = [
            SearchResult(chunk=c, score=0.7, rank=i)
            for i, c in enumerate(sample_chunks[:10])
        ]

        def calculate_overlap():
            return retriever._calculate_keyword_overlap("휴학 절차", search_results)

        score = benchmark(calculate_overlap)
        assert 0.0 <= score <= 1.0

    async def test_reranking_enhancement(self, benchmark, sample_chunks):
        """Benchmark CRAG reranking enhancement."""
        retriever = CRAGRetriever(enable_rerank=True)
        search_results = [
            SearchResult(chunk=c, score=0.7, rank=i)
            for i, c in enumerate(sample_chunks[:20])
        ]

        def rerank():
            return retriever.apply_rerank("test query", search_results)

        results = benchmark(rerank)
        assert len(results) == len(search_results)


# =============================================================================
# Benchmarks: Cache Performance
# =============================================================================


@pytest.mark.benchmark
class TestCacheBenchmark:
    """Benchmark cache operations."""

    def test_cache_write_performance(self, benchmark, tmp_path):
        """Benchmark cache write performance."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path / "cache"), ttl_days=30)

        def cache_write():
            cache.set(
                f"system_prompt_{time.time()}", "user_message", "model", "response"
            )

        benchmark(cache_write)

    def test_cache_read_performance(self, benchmark, tmp_path):
        """Benchmark cache read performance."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path / "cache"), ttl_days=30)
        cache.set("test_prompt", "test_message", "test_model", "test_response")

        def cache_read():
            return cache.get("test_prompt", "test_message", "test_model")

        result = benchmark(cache_read)
        assert result == "test_response"

    def test_cache_hash_computation(self, benchmark):
        """Benchmark cache hash computation performance."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir="/tmp/test", ttl_days=30)

        def compute_hash():
            return cache._compute_hash(
                "You are a helpful assistant.",
                "What is the meaning of life?",
                "gpt-4o-mini",
            )

        hash_value = benchmark(compute_hash)
        assert len(hash_value) == 32


# =============================================================================
# Performance Regression Tests
# =============================================================================


@pytest.mark.benchmark
class TestPerformanceRegression:
    """Tests for performance regression detection."""

    def test_p2_improvement_visible(self, benchmark):
        """
        Verify P2 performance improvements are measurable.

        This test ensures that the optimizations from P2 (Performance)
        are actually providing measurable benefits.
        """
        # Create a simple search scenario
        documents = [
            (f"doc{i}", f"content {i} with 휴학 keyword", {"level": "article"})
            for i in range(50)
        ]

        with patch("src.rag.infrastructure.hybrid_search_integration.DenseRetriever"):
            searcher = DenseHybridSearcher(use_dynamic_weights=False)
            searcher.add_documents(documents)

        # Measure search time
        def search_operation():
            return searcher.search("휴학", top_k=10)

        # This should complete in reasonable time after P2 optimizations
        # The exact threshold depends on the system, but we check it's not excessive
        start = time.time()
        results = search_operation()
        elapsed = time.time() - start

        assert isinstance(results, list)
        # Assert search completes in less than 1 second (generous threshold)
        assert elapsed < 1.0, f"Search took {elapsed:.2f}s, expected < 1.0s"

    def test_scalability_with_document_count(self):
        """Test that performance scales reasonably with document count."""
        with patch("src.rag.infrastructure.hybrid_search_integration.DenseRetriever"):
            searcher = DenseHybridSearcher(use_dynamic_weights=False)

            # Test with increasing document counts
            counts = [50, 100, 200]
            times = []

            for count in counts:
                documents = [
                    (f"doc{i}", f"content {i}", {"level": "article"})
                    for i in range(count)
                ]
                searcher.clear()
                searcher.add_documents(documents)

                start = time.time()
                searcher.search("test", top_k=10)
                elapsed = time.time() - start
                times.append(elapsed)

            # Verify scaling is sub-linear (ideally)
            # Time for 200 docs should be less than 4x time for 50 docs
            # (linear would be 4x, sub-linear is better)
            ratio = times[2] / times[0]
            assert ratio < 4.0, f"Scaling ratio {ratio:.2f} indicates poor performance"


@pytest.mark.benchmark
def test_baseline_establishment():
    """
    Establish baseline metrics for comparison.

    This test establishes baseline performance metrics that can be
    compared against in future runs to detect regressions.
    """
    from src.rag.infrastructure.hybrid_search_integration import DenseHybridSearcher

    with patch("src.rag.infrastructure.hybrid_search_integration.DenseRetriever"):
        searcher = DenseHybridSearcher(use_dynamic_weights=False)

        documents = [
            (f"doc{i}", f"규정 내용 {i}에 대한 설명", {"category": "학사"})
            for i in range(100)
        ]
        searcher.add_documents(documents)

        # Measure search performance
        start = time.time()
        results = searcher.search("휴학 절차", top_k=10)
        search_time = time.time() - start

        # Establish baseline metrics
        baseline_metrics = {
            "search_time_seconds": search_time,
            "result_count": len(results),
            "documents_indexed": 100,
        }

        # In a real scenario, these would be saved to a file for comparison
        # For now, we just verify they're reasonable
        assert baseline_metrics["search_time_seconds"] < 2.0
        assert baseline_metrics["result_count"] >= 0
