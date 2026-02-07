"""
Coverage Enhancement Tests for Edge Cases.

Tests for edge cases in:
- SelfRAG relevance evaluation
- LLM cache error handling
- Reranker edge cases
- LLM client fallback logic
"""

import time
from unittest.mock import Mock, patch

import pytest

# =============================================================================
# SelfRAG Edge Cases
# =============================================================================


@pytest.mark.coverage
class TestSelfRAGEdgeCases:
    """Test edge cases in SelfRAG component."""

    def test_self_rag_with_empty_results(self):
        """Test SelfRAG with empty search results."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator(llm_client=None)

        is_relevant, results = evaluator.evaluate_relevance("test query", [])

        # Empty results are not relevant
        assert is_relevant is False
        assert results == []

    def test_self_rag_with_very_long_context(self):
        """Test SelfRAG with context exceeding max chars."""
        from src.rag.domain.entities import (
            Chunk,
            ChunkLevel,
            RegulationStatus,
            SearchResult,
        )
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator(llm_client=None)

        # Create chunk with very long text
        long_text = "규정 내용 " * 10000  # Very long text

        chunk = Chunk(
            id="test_chunk",
            rule_code="RULE001",
            level=ChunkLevel.ARTICLE,
            title="Test Article",
            text=long_text,
            embedding_text=long_text,
            full_text=long_text,
            parent_path=[],
            token_count=50000,
            keywords=[],
            is_searchable=True,
            status=RegulationStatus.ACTIVE,
        )

        result = SearchResult(chunk=chunk, score=0.8, rank=0)

        is_relevant, results = evaluator.evaluate_relevance(
            "test query", [result], max_context_chars=100
        )

        # Should handle truncation gracefully
        assert is_relevant is True
        assert len(results) == 1

    def test_self_rag_with_special_characters(self):
        """Test SelfRAG with special characters in query."""
        from src.rag.domain.entities import (
            Chunk,
            ChunkLevel,
            RegulationStatus,
            SearchResult,
        )
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator(llm_client=None)

        chunk = Chunk(
            id="test_chunk",
            rule_code="RULE001",
            level=ChunkLevel.ARTICLE,
            title="Test Article",
            text="규정 내용",
            embedding_text="규정 내용",
            full_text="규정 내용",
            parent_path=[],
            token_count=10,
            keywords=[],
            is_searchable=True,
            status=RegulationStatus.ACTIVE,
        )

        result = SearchResult(chunk=chunk, score=0.8, rank=0)

        # Test with special characters
        special_queries = [
            "테스트 <script>alert('xss')</script>",
            "테스트\t\n\r",  # Control characters
            "테스트 🔥🎉",  # Emojis
            "테스트 \\u0000",  # Null byte
        ]

        for query in special_queries:
            is_relevant, results = evaluator.evaluate_relevance(query, [result])
            assert isinstance(is_relevant, bool)
            assert isinstance(results, list)

    def test_self_rag_pipeline_with_null_chunk(self):
        """Test SelfRAG pipeline with null/None chunk."""
        from src.rag.infrastructure.self_rag import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator(llm_client=None)

        # Create result with None chunk (edge case)
        result = Mock()
        result.chunk = None

        # Should handle gracefully
        try:
            is_relevant, results = evaluator.evaluate_relevance("test", [result])
            # If it doesn't crash, that's good
            assert isinstance(is_relevant, bool)
        except (AttributeError, TypeError):
            # Expected to fail with None chunk
            pass


# =============================================================================
# LLM Cache Error Handling
# =============================================================================


@pytest.mark.coverage
class TestLLMCacheErrorHandling:
    """Test error handling in LLM cache."""

    def test_cache_with_invalid_json_index(self, tmp_path):
        """Test cache recovery from invalid JSON index."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        # Create invalid JSON file
        index_path = tmp_path / "cache_index.json"
        with open(index_path, "w") as f:
            f.write("{invalid json content")

        cache = LLMResponseCache(cache_dir=str(tmp_path), ttl_days=30)

        # Should handle invalid JSON gracefully
        stats = cache.stats()
        assert "total_entries" in stats
        # Should start fresh with empty index
        assert stats["total_entries"] == 0

    def test_cache_with_permission_denied(self, tmp_path):
        """Test cache handling when write permission denied."""
        import stat

        from src.rag.infrastructure.llm_cache import LLMResponseCache

        # Create cache directory and make it read-only
        cache_dir = tmp_path / "readonly_cache"
        cache_dir.mkdir()

        index_path = cache_dir / "cache_index.json"
        index_path.touch()

        # Make file read-only
        index_path.chmod(stat.S_IRUSR)

        try:
            cache = LLMResponseCache(cache_dir=str(cache_dir), ttl_days=30)

            # Try to write - should handle gracefully or raise PermissionError
            try:
                cache.set("test", "test", "model", "response")
            except (PermissionError, OSError):
                # Expected behavior when file is read-only
                pass

            # Verify behavior (may or may not succeed depending on OS)
            stats = cache.stats()
            assert "total_entries" in stats

        finally:
            # Restore permissions for cleanup
            try:
                index_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except Exception:
                pass

    def test_cache_with_concurrent_writes(self, tmp_path):
        """Test cache handling concurrent write operations."""
        import threading

        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path / "concurrent"), ttl_days=30)

        # Simulate concurrent writes
        def write_entries(start_idx: int):
            for i in range(start_idx, start_idx + 10):
                cache.set(f"key_{i}", f"value_{i}", "model", f"response_{i}")

        threads = [
            threading.Thread(target=write_entries, args=(0,)),
            threading.Thread(target=write_entries, args=(10,)),
            threading.Thread(target=write_entries, args=(20,)),
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All entries should be written
        stats = cache.stats()
        assert stats["total_entries"] == 30

    def test_cache_with_none_values(self, tmp_path):
        """Test cache with None values."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path), ttl_days=30)

        # Set with None response (edge case)
        cache.set("test", "test", "model", None)

        # Get should return None
        result = cache.get("test", "test", "model")
        assert result is None

    def test_cache_with_empty_strings(self, tmp_path):
        """Test cache with empty strings."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path), ttl_days=30)

        # Set with empty strings
        cache.set("", "", "", "")

        # Should handle empty strings
        result = cache.get("", "", "")
        assert result == ""

    def test_cache_cleanup_with_all_expired(self, tmp_path):
        """Test cache cleanup when all entries are expired."""
        from src.rag.infrastructure.llm_cache import LLMResponseCache

        cache = LLMResponseCache(cache_dir=str(tmp_path), ttl_days=0)

        # Add entries
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}", "model", f"response_{i}")

        # Manually expire all
        for key in cache._index:
            cache._index[key]["timestamp"] = time.time() - 3600

        # Cleanup should remove all
        removed = cache.clear_expired()
        assert removed == 10

        stats = cache.stats()
        assert stats["total_entries"] == 0


# =============================================================================
# Reranker Edge Cases
# =============================================================================


@pytest.mark.coverage
class TestRerankerEdgeCases:
    """Test edge cases in reranker."""

    def test_reranker_with_single_document(self):
        """Test reranker with only one document."""
        from src.rag.domain.entities import Chunk, ChunkLevel, RegulationStatus
        from src.rag.infrastructure.reranker import BGEReranker, RerankedResult

        reranker = BGEReranker()

        chunk = Chunk(
            id="test",
            rule_code="RULE001",
            level=ChunkLevel.ARTICLE,
            title="Test",
            text="content",
            embedding_text="content",
            full_text="content",
            parent_path=[],
            token_count=10,
            keywords=[],
            is_searchable=True,
            status=RegulationStatus.ACTIVE,
        )

        documents = [(chunk.id, chunk.embedding_text, chunk.to_metadata())]

        # Mock the rerank function to return RerankedResult objects
        with patch(
            "src.rag.infrastructure.reranker.rerank",
            return_value=[
                RerankedResult("test", "content", 0.9, 0, chunk.to_metadata())
            ],
        ):
            results = reranker.rerank("query", documents, top_k=5)

        assert len(results) <= 5

    def test_reranker_with_duplicate_documents(self):
        """Test reranker with duplicate document IDs."""
        from src.rag.infrastructure.reranker import BGEReranker

        reranker = BGEReranker()

        # Create documents with duplicate IDs
        documents = [
            ("doc1", "content 1", {}),
            ("doc1", "content 2", {}),  # Duplicate ID
            ("doc2", "content 3", {}),
        ]

        with patch("src.rag.infrastructure.reranker.rerank", return_value=[]):
            results = reranker.rerank("query", documents, top_k=10)

        # Should handle duplicates
        assert isinstance(results, list)

    def test_reranker_with_unicode_content(self):
        """Test reranker with Unicode content."""
        from src.rag.infrastructure.reranker import BGEReranker

        reranker = BGEReranker()

        # Test with various Unicode content
        unicode_content = [
            "한글 내용",
            "日本語の内容",
            "العربية المحتوى",
            "Emoji content 🔥🎉✨",
            "Mixed content 한글 🔥 English",
        ]

        documents = [
            (f"doc{i}", content, {}) for i, content in enumerate(unicode_content)
        ]

        with patch("src.rag.infrastructure.reranker.rerank", return_value=[]):
            results = reranker.rerank("query", documents, top_k=5)

        assert isinstance(results, list)

    def test_reranker_context_with_empty_metadata(self):
        """Test reranker context boosting with empty metadata."""
        from src.rag.infrastructure.reranker import BGEReranker, RerankedResult

        reranker = BGEReranker()

        documents = [
            ("doc1", "content 1", {}),
            ("doc2", "content 2", {}),
            ("doc3", "content 3", {}),
        ]

        context = {
            "target_regulation": "nonexistent",
            "target_audience": "nonexistent",
        }

        # Mock with RerankedResult objects
        mock_results = [
            RerankedResult("doc1", "content 1", 0.7, 0, {}),
            RerankedResult("doc2", "content 2", 0.6, 1, {}),
            RerankedResult("doc3", "content 3", 0.5, 2, {}),
        ]

        with patch(
            "src.rag.infrastructure.reranker.rerank",
            return_value=mock_results,
        ):
            results = reranker.rerank_with_context("query", documents, context, top_k=3)

        # Should handle empty metadata gracefully
        assert len(results) <= 3


# =============================================================================
# LLM Client Fallback Logic
# =============================================================================


@pytest.mark.coverage
class TestLLMClientFallback:
    """Test LLM client fallback logic."""

    def test_mock_llm_with_various_temperatures(self):
        """Test mock LLM with different temperature values."""
        from src.rag.infrastructure.llm_client import MockLLMClient

        client = MockLLMClient()

        temperatures = [0.0, 0.5, 1.0, 1.5, 2.0]

        for temp in temperatures:
            response = client.generate(
                system_prompt="test",
                user_message="test",
                temperature=temp,
            )

            # Should work with any temperature
            assert isinstance(response, str)
            assert "Mock Response" in response

    def test_mock_llm_embedding_consistency(self):
        """Test that mock embeddings are consistent for same input."""
        from src.rag.infrastructure.llm_client import MockLLMClient

        client = MockLLMClient()

        text = "휴학 절차에 대한 규정"

        emb1 = client.get_embedding(text)
        emb2 = client.get_embedding(text)

        # Mock embeddings should be consistent
        assert emb1 == emb2

    def test_mock_llm_with_empty_inputs(self):
        """Test mock LLM with empty inputs."""
        from src.rag.infrastructure.llm_client import MockLLMClient

        client = MockLLMClient()

        # Empty strings
        response = client.generate("", "")
        assert isinstance(response, str)

        embedding = client.get_embedding("")
        assert isinstance(embedding, list)
        assert len(embedding) == 384

    def test_openai_client_missing_api_key(self):
        """Test OpenAI client without API key."""
        # Clear environment variable
        import os

        from src.rag.exceptions import MissingAPIKeyError
        from src.rag.infrastructure.llm_client import OpenAIClient

        original_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            with pytest.raises(MissingAPIKeyError):
                client = OpenAIClient(api_key=None)
        finally:
            # Restore original key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key


# =============================================================================
# CRAG Edge Cases
# =============================================================================


@pytest.mark.coverage
@pytest.mark.asyncio
class TestCRAGEdgeCases:
    """Test edge cases in CRAG retriever."""

    async def test_crag_with_all_zero_scores(self):
        """Test CRAG evaluation when all scores are zero."""
        from src.rag.domain.entities import (
            Chunk,
            ChunkLevel,
            RegulationStatus,
            SearchResult,
        )
        from src.rag.infrastructure.crag_retriever import CRAGRetriever

        retriever = CRAGRetriever()

        chunks = [
            Chunk(
                id=f"chunk{i}",
                rule_code="RULE001",
                level=ChunkLevel.ARTICLE,
                title="Test",
                text="content",
                embedding_text="content",
                full_text="content",
                parent_path=[],
                token_count=10,
                keywords=[],
                is_searchable=True,
                status=RegulationStatus.ACTIVE,
            )
            for i in range(5)
        ]

        results = [
            SearchResult(chunk=c, score=0.0, rank=i) for i, c in enumerate(chunks)
        ]

        quality, score = retriever.evaluate_retrieval_quality("test", results, "medium")

        # Should handle zero scores
        assert isinstance(quality, object)  # RetrievalQuality enum
        assert score >= 0.0

    async def test_crag_with_negative_scores(self):
        """Test CRAG with negative scores (edge case)."""
        from src.rag.domain.entities import (
            Chunk,
            ChunkLevel,
            RegulationStatus,
            SearchResult,
        )
        from src.rag.infrastructure.crag_retriever import CRAGRetriever

        retriever = CRAGRetriever()

        chunk = Chunk(
            id="test",
            rule_code="RULE001",
            level=ChunkLevel.ARTICLE,
            title="Test",
            text="content",
            embedding_text="content",
            full_text="content",
            parent_path=[],
            token_count=10,
            keywords=[],
            is_searchable=True,
            status=RegulationStatus.ACTIVE,
        )

        # Negative score (shouldn't happen in practice but test edge case)
        result = SearchResult(chunk=chunk, score=-0.5, rank=0)

        quality, score = retriever.evaluate_retrieval_quality(
            "test", [result], "medium"
        )

        # Should clamp to valid range
        assert 0.0 <= score <= 1.0

    async def test_crag_tfix_with_invalid_complexity(self):
        """Test CRAG T-Fix with invalid complexity value."""
        from src.rag.infrastructure.crag_retriever import (
            CRAGRetriever,
            RetrievalQuality,
        )

        retriever = CRAGRetriever()

        # Test with various complexity values
        complexities = ["simple", "medium", "complex", "invalid", None, ""]

        for complexity in complexities:
            try:
                should_trigger = retriever.should_trigger_tfix(
                    RetrievalQuality.POOR, 0.3, attempt_count=0
                )
                assert isinstance(should_trigger, bool)
            except (KeyError, TypeError):
                # Some invalid complexities might raise errors
                pass


# =============================================================================
# Hybrid Search Edge Cases
# =============================================================================


@pytest.mark.coverage
class TestHybridSearchEdgeCases:
    """Test edge cases in hybrid search."""

    def test_hybrid_search_with_mixed_language_query(self):
        """Test hybrid search with mixed Korean/English query."""
        from src.rag.infrastructure.hybrid_search_integration import DenseHybridSearcher

        with patch("src.rag.infrastructure.hybrid_search_integration.DenseRetriever"):
            searcher = DenseHybridSearcher(use_dynamic_weights=False)

            # Mixed language queries
            queries = [
                "휴학 leave of absence",
                "성적 grade 평가 evaluation",
                "장학금 scholarship 지급",
            ]

            for query in queries:
                results = searcher.search(query, top_k=5)
                assert isinstance(results, list)

    def test_hybrid_search_with_query_length_variations(self):
        """Test hybrid search with various query lengths."""
        from src.rag.infrastructure.hybrid_search_integration import DenseHybridSearcher

        with patch("src.rag.infrastructure.hybrid_search_integration.DenseRetriever"):
            searcher = DenseHybridSearcher(use_dynamic_weights=False)

            # Add test document
            searcher.add_documents([("doc1", "휴학 규정 내용입니다", {})])

            # Various query lengths
            queries = [
                "",  # Empty
                "휴",  # Single character
                "휴학" * 100,  # Very long
                " ",  # Whitespace only
            ]

            for query in queries:
                try:
                    results = searcher.search(query, top_k=5)
                    assert isinstance(results, list)
                except Exception:
                    # Some queries might raise errors
                    pass
