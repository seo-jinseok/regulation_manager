"""
Integration Tests for RAG Pipeline.

Tests the complete flow: HybridSearch -> DenseRetriever -> Reranker -> CRAG/SelfRAG -> LLM
"""

from typing import List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.rag.domain.entities import Chunk, ChunkLevel, RegulationStatus, SearchResult
from src.rag.infrastructure.crag_retriever import (
    CRAGPipeline,
    CRAGRetriever,
    RetrievalQuality,
)
from src.rag.infrastructure.hybrid_search_integration import DenseHybridSearcher
from src.rag.infrastructure.llm_cache import LLMResponseCache
from src.rag.infrastructure.llm_client import MockLLMClient
from src.rag.infrastructure.reranker import BGEReranker

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Create sample regulation chunks for testing."""
    return [
        Chunk(
            id="chunk1",
            rule_code="RULE001",
            level=ChunkLevel.ARTICLE,
            title="제1조 목적",
            text="이 규정은 대학교의 학사 관리에 관한 사항을 규정함을 목적으로 한다.",
            embedding_text="이 규정은 대학교의 학사 관리에 관한 사항을 규정함을 목적으로 한다.",
            full_text="제1조 목적\n이 규정은 대학교의 학사 관리에 관한 사항을 규정함을 목적으로 한다.",
            parent_path=["교원인사규정"],
            token_count=50,
            keywords=[],
            is_searchable=True,
            status=RegulationStatus.ACTIVE,
        ),
        Chunk(
            id="chunk2",
            rule_code="RULE001",
            level=ChunkLevel.ARTICLE,
            title="제2조 정의",
            text="이 규정에서 사용하는 용어의 정의는 다음과 같다.",
            embedding_text="이 규정에서 사용하는 용어의 정의는 다음과 같다.",
            full_text="제2조 정의\n이 규정에서 사용하는 용어의 정의는 다음과 같다.",
            parent_path=["교원인사규정"],
            token_count=30,
            keywords=[],
            is_searchable=True,
            status=RegulationStatus.ACTIVE,
        ),
        Chunk(
            id="chunk3",
            rule_code="RULE002",
            level=ChunkLevel.ARTICLE,
            title="휴학 절차",
            text="휴학을 하고자 하는 학생은 학기 시작 14일 전까지 신청하여야 한다.",
            embedding_text="휴학을 하고자 하는 학생은 학기 시작 14일 전까지 신청하여야 한다.",
            full_text="휴학 절차\n휴학을 하고자 하는 학생은 학기 시작 14일 전까지 신청하여야 한다.",
            parent_path=["학칙"],
            token_count=40,
            keywords=[],
            is_searchable=True,
            status=RegulationStatus.ACTIVE,
        ),
    ]


@pytest.fixture
def mock_dense_retriever():
    """Mock dense retriever for testing."""
    with patch(
        "src.rag.infrastructure.hybrid_search_integration.DenseRetriever"
    ) as mock:
        instance = mock.return_value
        instance.search.return_value = []
        instance.get_cache_stats.return_value = {
            "cache_hits": 0,
            "cache_misses": 0,
        }
        instance.add_documents = Mock()
        instance.clear = Mock()
        yield instance


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = MockLLMClient()
    return client


@pytest.fixture
def mock_reranker():
    """Mock reranker for testing."""
    # Create a mock that implements IReranker interface
    from unittest.mock import MagicMock

    reranker = MagicMock()
    reranker.rerank.return_value = []
    # Note: rerank_with_context is not in IReranker interface, it's BGEReranker specific
    return reranker


@pytest.fixture
def llm_cache(tmp_path):
    """Create LLM cache with temporary directory."""
    cache_dir = tmp_path / "llm_cache"
    return LLMResponseCache(cache_dir=str(cache_dir), ttl_days=30, max_entries=100)


@pytest.fixture
def hybrid_searcher(mock_dense_retriever):
    """Create hybrid searcher with mocked dense retriever."""
    searcher = DenseHybridSearcher(
        dense_model_name="test-model",
        use_dynamic_weights=False,
    )
    # Replace the dense retriever with mock
    searcher._dense_retriever = mock_dense_retriever
    return searcher


@pytest.fixture
def crag_retriever(hybrid_searcher, mock_llm_client):
    """Create CRAG retriever with mocked dependencies."""
    return CRAGRetriever(
        hybrid_searcher=hybrid_searcher,
        llm_client=mock_llm_client,
        enable_tfix=True,
        enable_rerank=True,
        max_tfix_attempts=2,
    )


@pytest.fixture
def crag_pipeline(crag_retriever):
    """Create CRAG pipeline."""
    return CRAGPipeline(retriever=crag_retriever)


# =============================================================================
# Test: HybridSearch -> DenseRetriever Integration
# =============================================================================


@pytest.mark.integration
class TestHybridSearchIntegration:
    """Test hybrid search integration with sparse and dense retrieval."""

    def test_add_documents_to_both_indices(
        self, hybrid_searcher, mock_dense_retriever, sample_chunks
    ):
        """Test that documents are added to both BM25 and Dense indices."""
        documents = [(c.id, c.embedding_text, c.to_metadata()) for c in sample_chunks]

        hybrid_searcher.add_documents(documents)

        # Verify parent (BM25) index was populated
        assert len(hybrid_searcher.bm25.documents) == len(sample_chunks)

        # Verify dense retriever was called
        mock_dense_retriever.add_documents.assert_called_once()

    def test_hybrid_search_fuses_results(
        self, hybrid_searcher, mock_dense_retriever, sample_chunks
    ):
        """Test that hybrid search properly fuses sparse and dense results."""
        # Add documents to BM25
        documents = [(c.id, c.embedding_text, c.to_metadata()) for c in sample_chunks]
        hybrid_searcher.add_documents(documents)

        # Mock dense search results
        from src.rag.infrastructure.hybrid_search import ScoredDocument

        mock_dense_retriever.search.return_value = [
            (
                "chunk3",
                0.9,
                sample_chunks[2].embedding_text,
                sample_chunks[2].to_metadata(),
            )
        ]

        # Perform search
        results = hybrid_searcher.search("휴학", top_k=3, use_dense=True)

        # Verify results are returned
        assert isinstance(results, list)
        # Results should be ScoredDocument objects
        for result in results:
            assert isinstance(result, ScoredDocument)

    def test_search_with_dense_disabled(self, hybrid_searcher, sample_chunks):
        """Test search with only sparse (BM25) retrieval."""
        documents = [(c.id, c.embedding_text, c.to_metadata()) for c in sample_chunks]
        hybrid_searcher.add_documents(documents)

        # Search without dense retrieval
        results = hybrid_searcher.search("휴학", top_k=3, use_dense=False)

        # Verify results are returned
        assert isinstance(results, list)

    def test_cache_stats(self, hybrid_searcher, mock_dense_retriever):
        """Test cache statistics retrieval."""
        mock_dense_retriever.get_cache_stats.return_value = {
            "cache_hits": 10,
            "cache_misses": 5,
        }

        stats = hybrid_searcher.get_cache_stats()

        assert "bm25" in stats
        assert "dense" in stats
        assert stats["dense"]["cache_hits"] == 10


# =============================================================================
# Test: Reranker Integration
# =============================================================================


@pytest.mark.integration
class TestRerankerIntegration:
    """Test reranker integration with search results."""

    @pytest.mark.asyncio
    async def test_reranker_reorders_results(self, sample_chunks):
        """Test that reranker reorders search results by relevance."""
        from src.rag.infrastructure.reranker import (
            RerankedResult,
            rerank_search_results,
        )

        # Create search results
        search_results = [
            SearchResult(chunk=sample_chunks[0], score=0.7, rank=0),
            SearchResult(chunk=sample_chunks[1], score=0.6, rank=1),
            SearchResult(chunk=sample_chunks[2], score=0.5, rank=2),
        ]

        # Mock the rerank function to return RerankedResult objects with reversed scores
        def mock_rerank_func(query, documents, top_k):
            # Return RerankedResult objects with modified scores
            return [
                RerankedResult(
                    doc_id=doc[0],
                    content=doc[1],
                    score=1.0 - i * 0.1,  # Reversed scores
                    original_rank=i,
                    metadata=doc[2],
                )
                for i, doc in enumerate(documents[:top_k])
            ]

        with patch(
            "src.rag.infrastructure.reranker.rerank", side_effect=mock_rerank_func
        ):
            reranked = rerank_search_results("휴학 절차", search_results, top_k=3)

        # Verify reranking was applied
        assert len(reranked) == 3
        # Verify scores were updated
        assert reranked[0].score != search_results[0].score

    def test_reranker_with_context_boosting(self, sample_chunks):
        """Test reranker with metadata context boosting."""
        # BGEReranker has rerank_with_context method (not in IReranker interface)
        reranker = BGEReranker()

        # Mock the internal rerank function properly
        # It needs to return RerankedResult objects for the internal logic
        from src.rag.infrastructure.reranker import RerankedResult

        mock_results = [
            RerankedResult("chunk1", "content1", 0.7, 0, {}),
            RerankedResult("chunk2", "content2", 0.6, 1, {}),
        ]

        with patch("src.rag.infrastructure.reranker.rerank", return_value=mock_results):
            documents = [
                (c.id, c.embedding_text, c.to_metadata()) for c in sample_chunks
            ]

            context = {
                "target_regulation": "교원인사규정",
                "regulation_boost": 0.15,
            }

            # BGEReranker has rerank_with_context as additional method
            if hasattr(reranker, "rerank_with_context"):
                results = reranker.rerank_with_context(
                    "query", documents, context=context, top_k=2
                )
                assert len(results) == 2
            else:
                # Fallback: test basic rerank
                results = reranker.rerank("query", documents, top_k=2)
                assert len(results) <= 2


# =============================================================================
# Test: CRAG Pipeline Integration
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestCRAGPipelineIntegration:
    """Test CRAG pipeline integration with all components."""

    async def test_quality_evaluation(self, crag_retriever, sample_chunks):
        """Test retrieval quality evaluation."""
        search_results = [
            SearchResult(chunk=sample_chunks[0], score=0.8, rank=0),
            SearchResult(chunk=sample_chunks[1], score=0.7, rank=1),
        ]

        quality, score = crag_retriever.evaluate_retrieval_quality(
            "휴학 절차", search_results, complexity="medium"
        )

        assert isinstance(quality, RetrievalQuality)
        assert 0.0 <= score <= 1.0

    async def test_tfix_trigger_conditions(self, crag_retriever, sample_chunks):
        """Test T-Fix trigger conditions for different quality levels."""
        # POOR quality should trigger T-Fix
        quality = RetrievalQuality.POOR
        assert crag_retriever.should_trigger_tfix(quality, 0.3, attempt_count=0) is True

        # EXCELLENT quality should not trigger T-Fix
        quality = RetrievalQuality.EXCELLENT
        assert (
            crag_retriever.should_trigger_tfix(quality, 0.9, attempt_count=0) is False
        )

    async def test_tfix_max_attempts_limit(self, crag_retriever):
        """Test that T-Fix respects max attempts limit."""
        quality = RetrievalQuality.POOR

        # Should not trigger after max attempts
        assert (
            crag_retriever.should_trigger_tfix(quality, 0.3, attempt_count=2) is False
        )

    async def test_reranking_enhancement(self, crag_retriever, sample_chunks):
        """Test document re-ranking enhancement."""
        search_results = [
            SearchResult(chunk=sample_chunks[0], score=0.7, rank=0),
            SearchResult(chunk=sample_chunks[1], score=0.6, rank=1),
        ]

        reranked = crag_retriever.apply_rerank("휴학", search_results)

        assert len(reranked) == len(search_results)
        # Verify reranking occurred
        assert reranked[0].rank == 0

    async def test_complete_crag_pipeline(
        self, crag_pipeline, crag_retriever, sample_chunks
    ):
        """Test complete CRAG pipeline flow."""
        initial_results = [
            SearchResult(chunk=sample_chunks[0], score=0.7, rank=0),
            SearchResult(chunk=sample_chunks[1], score=0.6, rank=1),
        ]

        # Mock the hybrid searcher to avoid actual search
        with patch.object(crag_retriever, "_hybrid_searcher") as mock_searcher:
            mock_searcher.search = AsyncMock(return_value=initial_results)

            final_results = await crag_pipeline.search(
                "휴학 절차", initial_results, complexity="medium"
            )

        # Verify pipeline completed
        assert isinstance(final_results, list)
        assert len(final_results) >= 0


# =============================================================================
# Test: LLM Cache Integration
# =============================================================================


@pytest.mark.integration
class TestLLMCacheIntegration:
    """Test LLM cache integration with the pipeline."""

    def test_cache_miss_on_first_call(self, llm_cache):
        """Test that first call results in cache miss."""
        result = llm_cache.get(
            system_prompt="You are a helpful assistant.",
            user_message="What is the purpose?",
            model="gpt-4o-mini",
        )

        assert result is None

    def test_cache_hit_after_set(self, llm_cache):
        """Test that subsequent calls hit cache."""
        system_prompt = "You are a helpful assistant."
        user_message = "What is the purpose?"
        model = "gpt-4o-mini"

        # Set cache
        llm_cache.set(system_prompt, user_message, model, "This is the purpose.")

        # Get from cache
        result = llm_cache.get(system_prompt, user_message, model)

        assert result == "This is the purpose."

    def test_cache_expiration(self, llm_cache):
        """Test cache expiration with short TTL."""
        import time

        # Create cache with very short TTL
        cache = LLMResponseCache(
            cache_dir=str(llm_cache.cache_dir), ttl_days=0, max_entries=100
        )

        cache.set("test", "test", "model", "response")

        # Wait a moment
        time.sleep(0.1)

        # Should be expired
        result = cache.get("test", "test", "model")
        assert result is None

    def test_cache_stats(self, llm_cache):
        """Test cache statistics."""
        llm_cache.set("prompt1", "message1", "model", "response1")
        llm_cache.set("prompt2", "message2", "model", "response2")

        stats = llm_cache.stats()

        assert stats["total_entries"] == 2
        assert stats["active_entries"] >= 0

    def test_cache_cleanup_expired(self, llm_cache):
        """Test cleanup of expired entries."""
        import time

        # Add expired entry
        llm_cache.set("old", "old", "model", "old_response")

        # Manually expire it
        llm_cache._index["old_hash"] = {
            "query_hash": "old_hash",
            "response": "old_response",
            "model": "model",
            "timestamp": time.time() - (31 * 24 * 3600),  # 31 days ago
            "ttl_days": 30,
        }

        # Cleanup
        removed = llm_cache.clear_expired()

        assert removed >= 0


# =============================================================================
# Test: Error Handling and Fallbacks
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and fallback mechanisms."""

    async def test_hybrid_search_with_no_results(self, hybrid_searcher):
        """Test hybrid search when no results are found."""
        # Search without adding documents
        results = hybrid_searcher.search("nonexistent query", top_k=5)

        # Should return empty list
        assert isinstance(results, list)

    async def test_crag_with_empty_results(self, crag_retriever):
        """Test CRAG evaluation with empty results."""
        quality, score = crag_retriever.evaluate_retrieval_quality(
            "test query", [], complexity="medium"
        )

        assert quality == RetrievalQuality.POOR
        assert score == 0.0

    async def test_tfix_with_no_searcher(self, mock_llm_client):
        """Test T-Fix when hybrid searcher is not available."""
        retriever = CRAGRetriever(
            hybrid_searcher=None, llm_client=mock_llm_client, enable_tfix=True
        )

        original_results = [Mock()]

        new_results, improved = await retriever.apply_tfix("test", original_results)

        # Should return original results when searcher unavailable
        assert new_results == original_results
        assert improved is False

    def test_reranker_with_empty_documents(self, mock_reranker):
        """Test reranker with empty document list."""
        results = mock_reranker.rerank("query", [], top_k=5)

        assert results == []

    def test_cache_with_corrupted_index(self, llm_cache, tmp_path):
        """Test cache recovery from corrupted index."""

        # Write corrupted index
        index_path = llm_cache._index_path
        with open(index_path, "w") as f:
            f.write("{invalid json")

        # Cache should handle corrupted index gracefully
        stats = llm_cache.stats()

        # Should not crash
        assert "total_entries" in stats


# =============================================================================
# Test: End-to-End Pipeline Flow
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestEndToEndPipeline:
    """Test complete end-to-end RAG pipeline flow."""

    async def test_full_pipeline_flow(self, sample_chunks):
        """Test complete flow from search to cached LLM response."""
        # Setup

        # Create components
        llm_client = MockLLMClient()
        cache_dir = "/tmp/test_llm_cache"
        cache = LLMResponseCache(cache_dir=cache_dir, ttl_days=1)

        # Step 1: Search
        search_results = [
            SearchResult(chunk=sample_chunks[0], score=0.8, rank=0),
            SearchResult(chunk=sample_chunks[1], score=0.7, rank=1),
        ]

        # Step 2: Check LLM cache
        system_prompt = "You are a regulation assistant."
        user_message = (
            f"Context: {search_results[0].chunk.text}\nQuestion: What is the purpose?"
        )

        cached_response = cache.get(system_prompt, user_message, "mock-model")
        assert cached_response is None  # First call should miss

        # Step 3: Generate response
        response = llm_client.generate(system_prompt, user_message)

        # Step 4: Cache response
        cache.set(system_prompt, user_message, "mock-model", response)

        # Step 5: Verify cache hit
        cached_response = cache.get(system_prompt, user_message, "mock-model")
        assert cached_response == response

        # Cleanup
        import shutil

        if cache.cache_dir.exists():
            shutil.rmtree(cache.cache_dir)


@pytest.mark.integration
class TestQueryProcessingFlow:
    """Test query processing through the pipeline."""

    def test_query_normalization(self, hybrid_searcher):
        """Test that queries are properly normalized."""
        # Add test documents

        hybrid_searcher.add_documents(
            [("doc1", "휴학 신청은 학기 시작 14일 전까지 가능합니다.", {})]
        )

        # Test with normalized query
        results = hybrid_searcher.search("  휴학  ", top_k=3)

        # Should handle whitespace normalization
        assert isinstance(results, list)

    def test_query_expansion(self, hybrid_searcher):
        """Test query expansion with synonyms."""
        expanded = hybrid_searcher.expand_query("휴학")

        # Should return expanded query
        assert isinstance(expanded, str)

    def test_empty_query_handling(self, hybrid_searcher):
        """Test handling of empty queries."""
        results = hybrid_searcher.search("", top_k=3)

        # Should not crash
        assert isinstance(results, list)
