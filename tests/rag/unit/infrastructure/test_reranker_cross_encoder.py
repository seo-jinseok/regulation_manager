"""
SPEC-RAG-Q-011: Test CrossEncoder reranker integration.

Tests verify that:
- CrossEncoder can be imported without FlagEmbedding errors
- Reranker initializes successfully with CrossEncoder
- Reranking produces correct results
- BM25FallbackReranker is available as backup
"""

import pytest
from unittest.mock import MagicMock, patch

from src.rag.infrastructure.reranker import (
    BGEReranker,
    BM25FallbackReranker,
    RerankedResult,
    clear_reranker,
    get_reranker,
    get_reranker_status,
    rerank,
    warmup_reranker,
)


class TestCrossEncoderIntegration:
    """Test CrossEncoder reranker integration (SPEC-RAG-Q-011)."""

    def test_reranker_initializes_without_flagembedding_error(self):
        """SPEC-RAG-Q-011: Verify reranker initializes without FlagEmbedding import error."""
        clear_reranker()

        # This should NOT raise ImportError about is_torch_fx_available
        reranker = get_reranker()

        # Verify it's not a BM25FallbackReranker (which means CrossEncoder failed)
        # Note: This test uses the actual model, so it may take time on first run
        status = get_reranker_status()
        assert status["cross_encoder_available"] is True
        assert status["last_error"] is None
        assert status["active_reranker"] == "CrossEncoder"

        clear_reranker()

    def test_reranker_returns_valid_results(self):
        """SPEC-RAG-Q-011: Verify reranker returns valid results with CrossEncoder."""
        clear_reranker()

        docs = [
            ("doc1", "장학금 신청 방법에 대한 안내입니다.", {"title": "장학규정"}),
            ("doc2", "휴학 규정과 절차입니다.", {"title": "학칙"}),
            ("doc3", "장학금 지급 기준입니다.", {"title": "장학규정"}),
        ]

        results = rerank("장학금 신청", docs, top_k=3)

        # Verify results
        assert len(results) >= 1  # At least one result
        assert all(isinstance(r, RerankedResult) for r in results)
        assert all(0.0 <= r.score <= 1.0 for r in results)

        # Verify results are sorted by score (descending)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        clear_reranker()

    def test_reranker_with_empty_documents(self):
        """SPEC-RAG-Q-011: Verify reranker handles empty input."""
        results = rerank("query", [], top_k=10)
        assert results == []

    def test_reranker_with_single_document(self):
        """SPEC-RAG-Q-011: Verify reranker handles single document."""
        docs = [("doc1", "장학금 신청 방법", {"title": "장학규정"})]
        results = rerank("장학금", docs, top_k=1)

        assert len(results) == 1
        assert results[0].doc_id == "doc1"

    def test_bm25_fallback_is_available(self):
        """SPEC-RAG-Q-011: Verify BM25FallbackReranker is available as backup."""
        fallback = BM25FallbackReranker()

        docs = [
            ("doc1", "장학금 신청 방법", {"title": "장학규정"}),
            ("doc2", "휴학 규정", {"title": "학칙"}),
        ]

        results = fallback.rerank("장학금", docs, top_k=2)

        assert len(results) == 2
        assert all(len(r) == 4 for r in results)  # (doc_id, content, score, metadata)

    def test_fallback_to_bm25_on_crossencoder_failure(self):
        """SPEC-RAG-Q-011: Verify fallback to BM25 when CrossEncoder fails."""
        clear_reranker()

        # Mock CrossEncoder to fail
        mock_st = MagicMock()
        mock_st.CrossEncoder.side_effect = ImportError("Mock CrossEncoder failure")

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            reranker = get_reranker()

            # Should fall back to BM25
            assert isinstance(reranker, BM25FallbackReranker)

            status = get_reranker_status()
            assert status["cross_encoder_available"] is False
            assert "CrossEncoder" in status["last_error"]

        clear_reranker()


class TestRerankerScoreNormalization:
    """Test score normalization for CrossEncoder output."""

    def test_scores_are_normalized_to_0_1_range(self):
        """SPEC-RAG-Q-011: Verify CrossEncoder scores are normalized to 0-1 range."""
        clear_reranker()

        docs = [
            ("doc1", "완전히 무관한 내용입니다.", {}),
            ("doc2", "장학금 신청 방법과 절차 안내", {}),
            ("doc3", "장학금 지급 기준에 대한 규정", {}),
        ]

        results = rerank("장학금 신청", docs, top_k=3)

        # All scores should be in 0-1 range (after sigmoid normalization)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} is out of range"

        clear_reranker()

    def test_relevance_threshold_filtering(self):
        """SPEC-RAG-Q-011: Verify low-relevance documents are filtered."""
        clear_reranker()

        docs = [
            ("doc1", "완전히 무관한 내용입니다.", {}),
            ("doc2", "장학금 신청 방법", {}),
        ]

        # Use higher threshold to filter low-relevance docs
        results = rerank("휴학 절차", docs, top_k=2, min_relevance=0.8)

        # Low-relevance documents should be filtered out
        # (exact behavior depends on model output)
        for r in results:
            assert r.score >= 0.8

        clear_reranker()


class TestBGERerankerWithCrossEncoder:
    """Test BGEReranker class with CrossEncoder backend."""

    def test_bge_reranker_uses_cross_encoder(self):
        """SPEC-RAG-Q-011: Verify BGEReranker works with CrossEncoder backend."""
        clear_reranker()

        reranker = BGEReranker()
        docs = [
            ("doc1", "장학금 신청 방법", {"title": "장학규정"}),
            ("doc2", "휴학 규정", {"title": "학칙"}),
        ]

        results = reranker.rerank("장학금", docs, top_k=2)

        assert len(results) >= 1
        assert all(len(r) == 4 for r in results)

        clear_reranker()

    def test_bge_reranker_with_context(self):
        """SPEC-RAG-Q-011: Verify BGEReranker context boosting with CrossEncoder."""
        clear_reranker()

        reranker = BGEReranker()
        docs = [
            ("doc1", "휴학 신청 방법", {"regulation_title": "학적규정"}),
            ("doc2", "휴학 관련 내용", {"regulation_title": "장학규정"}),
        ]

        results = reranker.rerank_with_context(
            "휴학",
            docs,
            context={"target_regulation": "학적규정"},
            top_k=2,
        )

        # Results should be returned (context boosting is applied)
        assert len(results) >= 1

        clear_reranker()
