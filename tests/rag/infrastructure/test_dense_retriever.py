"""
Tests for Dense Retriever and Korean-optimized semantic search.

Tests cover:
- Dense Retriever initialization and indexing
- Korean embedding model loading
- Cosine similarity search
- Caching performance
- Batch processing
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from src.rag.infrastructure.dense_retriever import (
    DenseRetriever,
    create_dense_retriever,
)


@pytest.fixture
def sample_documents() -> List[Tuple[str, str, Dict]]:
    """Sample Korean documents for testing."""
    return [
        (
            "doc1",
            "휴학 신청은 학기 시작 14일 전까지 가능합니다. 휴학 신청서를 제출하고 승인을 받아야 합니다.",
            {"category": "학적", "regulation": "학칙"},
        ),
        (
            "doc2",
            "장학금은 성적 우수자, 저소득 가구, 근로 장학 등 다양한 종류가 있습니다.",
            {"category": "장학", "regulation": "장학금 지급 규정"},
        ),
        (
            "doc3",
            "졸업 요건은 총 140학점 이상, 전공 60학점, 교양 30학점 이수입니다.",
            {"category": "졸업", "regulation": "학칙"},
        ),
        (
            "doc4",
            "교원은 5년 근무 후 연구년 휴직을 신청할 수 있으며 승인이 필요합니다.",
            {"category": "교원", "regulation": "교원 인사 규정"},
        ),
        (
            "doc5",
            "수강신청은 매학기 지정된 기간에 하며 복수전공, 부전공도 함께 신청합니다.",
            {"category": "수강", "regulation": "수업 규정"},
        ),
    ]


@pytest.fixture
def dense_retriever():
    """Initialize Dense Retriever with default model."""
    return create_dense_retriever(model_name="jhgan/ko-sbert-sts")


class TestDenseRetriever:
    """Test suite for Dense Retriever functionality."""

    def test_initialization(self):
        """Test retriever initialization with default config."""
        retriever = create_dense_retriever()
        assert retriever is not None
        assert retriever.model_name == "jhgan/ko-sbert-sts"
        assert retriever.config.cache_embeddings is True

    def test_initialization_custom_model(self):
        """Test retriever initialization with custom model."""
        retriever = create_dense_retriever(model_name="jhgan/ko-sbert-sts")
        assert retriever.model_name == "jhgan/ko-sbert-sts"

    def test_embedding_dimension(self, dense_retriever):
        """Test embedding dimension property."""
        # Load model first
        dense_retriever._load_model()
        assert dense_retriever.embedding_dim > 0
        assert dense_retriever.embedding_dim == 768  # ko-sbert-sts dimension

    def test_add_documents(self, dense_retriever, sample_documents):
        """Test adding documents to index."""
        dense_retriever.add_documents(sample_documents)
        assert len(dense_retriever._doc_embeddings) == len(sample_documents)
        assert len(dense_retriever._doc_texts) == len(sample_documents)
        assert len(dense_retriever._doc_metadata) == len(sample_documents)

    def test_search_single_query(self, dense_retriever, sample_documents):
        """Test searching with a single query."""
        dense_retriever.add_documents(sample_documents)
        results = dense_retriever.search("휴학 절차", top_k=3)

        assert len(results) > 0
        assert len(results) <= 3

        # Check result structure
        doc_id, score, content, metadata = results[0]
        assert isinstance(doc_id, str)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # Cosine similarity range
        assert isinstance(content, str)
        assert isinstance(metadata, dict)

    def test_search_relevance(self, dense_retriever, sample_documents):
        """Test that search returns relevant results."""
        dense_retriever.add_documents(sample_documents)

        # Query about 휴학 should return relevant content in top results
        results = dense_retriever.search("휴학", top_k=3)
        assert len(results) > 0
        # Check if any result contains relevant keywords
        found_relevant = any(
            "휴학" in content or "신청" in content or "학기" in content
            for _, _, content, _ in results
        )
        assert found_relevant, (
            "Expected relevant content for '휴학', got results without relevant keywords"
        )

        # Query about 장학금 should return relevant content in top results
        results = dense_retriever.search("장학금", top_k=3)
        assert len(results) > 0
        found_relevant = any(
            "장학금" in content or "장학" in content or "성적" in content
            for _, _, content, _ in results
        )
        assert found_relevant, (
            "Expected relevant content for '장학금', got results without relevant keywords"
        )

        # Query about 졸업 should return relevant content in top results
        results = dense_retriever.search("졸업", top_k=3)
        assert len(results) > 0
        found_relevant = any(
            "졸업" in content or "학점" in content or "요건" in content
            for _, _, content, _ in results
        )
        assert found_relevant, (
            "Expected relevant content for '졸업', got results without relevant keywords"
        )

    def test_search_empty_index(self, dense_retriever):
        """Test searching with empty index."""
        results = dense_retriever.search("test query", top_k=5)
        assert len(results) == 0

    def test_search_with_threshold(self, dense_retriever, sample_documents):
        """Test search with similarity threshold."""
        dense_retriever.add_documents(sample_documents)

        # High threshold should return fewer results
        results_high = dense_retriever.search("휴학", top_k=5, score_threshold=0.8)
        results_low = dense_retriever.search("휴학", top_k=5, score_threshold=0.3)

        assert len(results_high) <= len(results_low)

    def test_batch_search(self, dense_retriever, sample_documents):
        """Test batch search with multiple queries."""
        dense_retriever.add_documents(sample_documents)

        queries = ["휴학", "장학금", "졸업"]
        results_list = dense_retriever.search_batch(queries, top_k=2)

        assert len(results_list) == len(queries)

        for results in results_list:
            assert len(results) <= 2
            if results:
                doc_id, score, content, metadata = results[0]
                assert isinstance(score, float)

    def test_caching(self, dense_retriever, sample_documents):
        """Test embedding cache functionality."""
        # Disable index caching to test query cache
        dense_retriever.config.cache_embeddings = True
        dense_retriever.add_documents(sample_documents)

        # First search (cache miss)
        results1 = dense_retriever.search("휴학")
        cache_stats = dense_retriever.get_cache_stats()
        initial_misses = cache_stats["cache_misses"]

        # Second search with same query (cache hit)
        results2 = dense_retriever.search("휴학")
        cache_stats = dense_retriever.get_cache_stats()
        final_misses = cache_stats["cache_misses"]

        # Results should be identical
        assert len(results1) == len(results2)

        # Cache misses should not increase on second search (cached)
        assert initial_misses == final_misses

        # Cache stats should reflect hits
        assert cache_stats["cache_hits"] > 0

    def test_cache_stats(self, dense_retriever, sample_documents):
        """Test cache statistics reporting."""
        dense_retriever.add_documents(sample_documents)
        dense_retriever.search("test query")

        stats = dense_retriever.get_cache_stats()
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_size" in stats
        assert "indexed_docs" in stats

        assert stats["indexed_docs"] == len(sample_documents)

    def test_save_and_load_index(self, dense_retriever, sample_documents):
        """Test saving and loading vector index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index.pkl"

            # Add documents and save
            dense_retriever.add_documents(sample_documents)
            dense_retriever.save_index(str(index_path))
            assert index_path.exists()

            # Create new retriever and load
            new_retriever = create_dense_retriever(
                model_name=dense_retriever.model_name
            )
            success = new_retriever.load_index(str(index_path))
            assert success is True

            # Verify loaded data
            assert len(new_retriever._doc_embeddings) == len(sample_documents)
            assert new_retriever.embedding_dim == dense_retriever.embedding_dim

    def test_clear(self, dense_retriever, sample_documents):
        """Test clearing index and cache."""
        dense_retriever.add_documents(sample_documents)
        assert len(dense_retriever._doc_embeddings) > 0

        dense_retriever.clear()
        assert len(dense_retriever._doc_embeddings) == 0
        assert len(dense_retriever._embedding_cache) == 0

    def test_list_models(self):
        """Test listing available models."""
        models = DenseRetriever.list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "jhgan/ko-sbert-sts" in models

    def test_get_model_info(self):
        """Test getting model information."""
        info = DenseRetriever.get_model_info("jhgan/ko-sbert-sts")
        assert isinstance(info, dict)
        assert "dims" in info
        assert "language" in info
        assert info["language"] == "ko"

    def test_korean_semantic_understanding(self, dense_retriever, sample_documents):
        """Test Korean semantic understanding with related queries."""
        dense_retriever.add_documents(sample_documents)

        # Related queries that should find the same document
        queries = ["휴학", "휴학 신청", "휴학 절차", "학교 쉬고 싶어"]

        results_list = [dense_retriever.search(q, top_k=1) for q in queries]

        # All queries should return doc1 (휴학 관련)
        for results in results_list:
            if results:
                assert "휴학" in results[0][2] or "신청" in results[0][2]

    def test_cosine_similarity_range(self, dense_retriever, sample_documents):
        """Test that cosine similarity scores are in valid range."""
        dense_retriever.add_documents(sample_documents)

        results = dense_retriever.search("휴학 장학금 졸업", top_k=10)

        for _doc_id, score, _content, _metadata in results:
            assert 0 <= score <= 1, f"Score {score} not in [0, 1] range"


class TestDenseRetrieverIntegration:
    """Integration tests for Dense Retriever."""

    def test_end_to_end_workflow(self):
        """Test complete workflow: index -> search -> save -> load -> search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test_index.pkl"

            # Create and index
            retriever = create_dense_retriever("jhgan/ko-sbert-sts")
            docs = [
                ("doc1", "휴학은 학기 시작 전에 신청해야 합니다.", {"type": "학적"}),
                ("doc2", "장학금은 성적에 따라 지급됩니다.", {"type": "장학"}),
            ]

            retriever.add_documents(docs)
            results = retriever.search("휴학")
            assert len(results) > 0

            # Save and load
            retriever.save_index(str(index_path))

            new_retriever = create_dense_retriever("jhgan/ko-sbert-sts")
            new_retriever.load_index(str(index_path))
            new_results = new_retriever.search("휴학")

            assert len(new_results) > 0

    def test_performance_with_large_index(self):
        """Test performance with larger document set."""
        retriever = create_dense_retriever("jhgan/ko-sbert-sts")

        # Generate 100 documents
        docs = []
        for i in range(100):
            docs.append(
                (
                    f"doc{i}",
                    f"문서 {i}번의 내용입니다. 휴학, 장학금, 졸업에 관한 규정이 포함되어 있습니다.",
                    {"index": i},
                )
            )

        import time

        start = time.time()
        retriever.add_documents(docs)
        index_time = time.time() - start

        start = time.time()
        results = retriever.search("휴학 규정", top_k=10)
        search_time = time.time() - start

        # Performance assertions
        assert len(results) == 10
        assert index_time < 30  # Should index in under 30 seconds
        assert search_time < 5  # Should search in under 5 seconds

        print(f"\nIndexed 100 docs in {index_time:.2f}s")
        print(f"Searched in {search_time:.4f}s")


@pytest.mark.benchmark
class TestDenseRetrieverBenchmark:
    """Benchmark tests for Dense Retriever performance."""

    def test_indexing_throughput(self):
        """Measure indexing throughput (documents/second)."""
        retriever = create_dense_retriever("jhgan/ko-sbert-sts")

        docs = [
            (f"doc{i}", f"문서 {i}번 내용입니다.", {})
            for i in range(50)  # 50 documents
        ]

        import time

        start = time.time()
        retriever.add_documents(docs)
        elapsed = time.time() - start

        throughput = len(docs) / elapsed
        print(f"\nIndexing throughput: {throughput:.2f} docs/sec")

        assert throughput > 1  # At least 1 doc/sec

    def test_search_latency(self):
        """Measure search latency (queries/second)."""
        retriever = create_dense_retriever("jhgan/ko-sbert-sts")

        docs = [
            (f"doc{i}", f"문서 {i}번 내용입니다. 휴학, 장학금, 졸업 규정.", {})
            for i in range(50)
        ]
        retriever.add_documents(docs)

        queries = ["휴학", "장학금", "졸업", "수강", "등록"]

        import time

        start = time.time()
        for query in queries:
            retriever.search(query, top_k=10)
        elapsed = time.time() - start

        qps = len(queries) / elapsed
        print(f"\nSearch throughput: {qps:.2f} queries/sec")

        assert qps > 1  # At least 1 query/sec
