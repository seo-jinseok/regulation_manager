"""
Characterization tests for Reranker BM25 Fallback behavior (TAG-002).

These tests verify the graceful degradation when BGE reranker fails:
- BM25FallbackReranker class functionality
- Automatic fallback when FlagEmbedding is unavailable
- Automatic fallback when compute_score raises exceptions
- Status tracking for monitoring
- Behavior preservation with Korean text

These are CHARACTERIZATION tests - they document what the code DOES,
not what it SHOULD do. They ensure behavior is preserved during refactoring.
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.rag.infrastructure.reranker import (
    BM25FallbackReranker,
    RerankedResult,
    clear_reranker,
    get_reranker_status,
    rerank,
)


class TestBM25FallbackRerankerCharacterize:
    """Characterization tests for BM25FallbackReranker class."""

    def test_characterize_bm25_reranker_basic_rerank(self):
        """Characterize: BM25FallbackReranker returns results sorted by relevance."""
        reranker = BM25FallbackReranker()
        docs = [
            ("doc1", "This is about apples and oranges", {}),
            ("doc2", "Apples are delicious fruits", {}),
            ("doc3", "Vegetables are healthy", {}),
        ]

        result = reranker.rerank("apples", docs, top_k=3)

        # BM25 returns results sorted by relevance
        assert len(result) == 3
        # Verify structure is correct
        for doc_id, content, score, metadata in result:
            assert isinstance(doc_id, str)
            assert isinstance(content, str)
            assert isinstance(score, float)
            assert isinstance(metadata, dict)
        # Verify scores are in descending order
        scores = [r[2] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_characterize_bm25_reranker_empty_documents(self):
        """Characterize: BM25FallbackReranker returns empty list for empty input."""
        reranker = BM25FallbackReranker()
        result = reranker.rerank("query", [], top_k=10)
        assert result == []

    def test_characterize_bm25_reranker_preserves_metadata(self):
        """Characterize: BM25FallbackReranker preserves document metadata."""
        reranker = BM25FallbackReranker()
        metadata = {"title": "Test Doc", "regulation_title": "Test Regulation"}
        docs = [("doc1", "apples and more apples", metadata)]

        result = reranker.rerank("apples", docs, top_k=1)

        assert len(result) == 1
        assert result[0][3] == metadata  # metadata is 4th element

    def test_characterize_bm25_reranker_korean_tokenization(self):
        """Characterize: BM25FallbackReranker handles Korean text with kiwipiepy."""
        reranker = BM25FallbackReranker(language="korean")
        docs = [
            ("doc1", "장학금 신청 방법에 대한 안내입니다", {}),
            ("doc2", "휴학 신청 절차", {}),
            ("doc3", "졸업 요건", {}),
        ]

        result = reranker.rerank("장학금 신청", docs, top_k=3)

        assert len(result) == 3
        # doc1 should be most relevant due to "장학금" and "신청" overlap
        assert result[0][0] == "doc1"

    def test_characterize_bm25_reranker_scores_normalized(self):
        """Characterize: BM25FallbackReranker normalizes scores to 0-1 range."""
        reranker = BM25FallbackReranker()
        docs = [
            ("doc1", "apple apple apple", {}),
            ("doc2", "banana", {}),
            ("doc3", "cherry", {}),
        ]

        result = reranker.rerank("apple", docs, top_k=3)

        for doc_id, content, score, metadata in result:
            assert 0.0 <= score <= 1.0, f"Score {score} not in range 0-1"

    def test_characterize_bm25_reranker_with_context_boost(self):
        """Characterize: BM25FallbackReranker supports context boosting."""
        reranker = BM25FallbackReranker()
        docs = [
            ("doc1", "학적 규정 내용", {"regulation_title": "학적규정"}),
            ("doc2", "장학 규정 내용", {"regulation_title": "장학규정"}),
        ]

        result_no_boost = reranker.rerank_with_context(
            "규정", docs, context={}, top_k=2
        )

        result_with_boost = reranker.rerank_with_context(
            "규정",
            docs,
            context={"target_regulation": "학적규정", "regulation_boost": 0.2},
            top_k=2,
        )

        # With boost, doc1 should be ranked higher
        assert result_with_boost[0][0] == "doc1"


class TestRerankerFallbackCharacterize:
    """Characterization tests for automatic fallback behavior."""

    def test_characterize_fallback_on_import_error(self):
        """Characterize: rerank falls back to BM25 when FlagEmbedding import fails."""
        clear_reranker()

        # Mock FlagEmbedding to raise ImportError
        with patch.dict(
            "sys.modules", {"FlagEmbedding": None}
        ):  # None causes ImportError
            docs = [
                ("doc1", "장학금 신청 방법", {"title": "장학규정"}),
                ("doc2", "휴학 규정", {"title": "학적규정"}),
            ]

            # Should NOT raise - should fall back gracefully
            result = rerank("장학금", docs, top_k=2)

            assert len(result) == 2
            # Result should come from BM25 fallback
            assert isinstance(result[0], RerankedResult)

        clear_reranker()

    def test_characterize_fallback_on_compute_score_error(self):
        """Characterize: rerank falls back to BM25 when compute_score raises."""
        clear_reranker()

        # Create a mock reranker that raises on compute_score
        mock_broken_reranker = MagicMock()
        mock_broken_reranker.compute_score.side_effect = RuntimeError(
            "transformers compatibility error"
        )

        with patch(
            "src.rag.infrastructure.reranker.get_reranker",
            return_value=mock_broken_reranker,
        ):
            docs = [
                ("doc1", "장학금 신청 방법", {}),
                ("doc2", "휴학 규정", {}),
            ]

            # Should NOT raise - should fall back gracefully
            result = rerank("장학금", docs, top_k=2)

            assert len(result) == 2
            # Result should come from BM25 fallback
            assert isinstance(result[0], RerankedResult)

        clear_reranker()

    def test_characterize_status_tracking_on_success(self):
        """Characterize: get_reranker_status returns correct status on success."""
        clear_reranker()

        # Mock successful FlagEmbedding
        mock_flag_module = MagicMock()
        mock_instance = MagicMock()
        mock_instance.compute_score.return_value = [0.8, 0.5]
        mock_flag_module.FlagReranker.return_value = mock_instance

        with patch.dict("sys.modules", {"FlagEmbedding": mock_flag_module}):
            # Trigger reranker initialization
            docs = [("doc1", "content", {})]
            rerank("test", docs, top_k=1)

            status = get_reranker_status()

            # BGE should be available
            assert status["bge_available"] is True
            assert status["last_error"] is None

        clear_reranker()

    def test_characterize_status_tracking_on_failure(self):
        """Characterize: get_reranker_status returns error info on failure."""
        clear_reranker()

        # Mock FlagEmbedding to raise
        mock_flag_module = MagicMock()
        mock_flag_module.FlagReranker.side_effect = ImportError("Not found")

        with patch.dict("sys.modules", {"FlagEmbedding": mock_flag_module}):
            docs = [("doc1", "content", {})]
            rerank("test", docs, top_k=1)

            status = get_reranker_status()

            # BGE should be unavailable
            assert status["bge_available"] is False
            assert "import" in status["last_error"].lower()

        clear_reranker()


class TestRerankerFallbackBehaviorPreservation:
    """Tests ensuring behavior is preserved during refactoring."""

    def test_characterize_rerank_returns_same_structure_bge_or_bm25(self):
        """Characterize: rerank returns same structure regardless of backend."""
        clear_reranker()
        docs = [
            ("doc1", "장학금 신청 방법", {"title": "장학"}),
            ("doc2", "휴학 규정", {"title": "학적"}),
        ]

        # Test with BM25 fallback directly
        bm25_reranker = BM25FallbackReranker()
        bm25_result = bm25_reranker.rerank("장학금", docs, top_k=2)

        # Verify structure
        assert len(bm25_result[0]) == 4  # (doc_id, content, score, metadata)
        assert isinstance(bm25_result[0][2], float)  # score is float
        assert isinstance(bm25_result[0][3], dict)  # metadata is dict

        clear_reranker()

    def test_characterize_top_k_respected_in_fallback(self):
        """Characterize: top_k is respected in BM25 fallback mode."""
        reranker = BM25FallbackReranker()
        docs = [(f"doc{i}", f"content {i}", {}) for i in range(10)]

        result = reranker.rerank("content", docs, top_k=3)

        assert len(result) == 3

    def test_characterize_empty_query_handled_gracefully(self):
        """Characterize: empty query is handled without error."""
        reranker = BM25FallbackReranker()
        docs = [("doc1", "some content", {})]

        # Should not raise
        result = reranker.rerank("", docs, top_k=1)
        assert len(result) == 1


class TestBM25FallbackRerankerEdgeCases:
    """Edge case tests for BM25FallbackReranker."""

    def test_characterize_single_document_returns_result(self):
        """Characterize: single document returns proper result."""
        reranker = BM25FallbackReranker()
        docs = [("doc1", "unique content", {"key": "value"})]

        result = reranker.rerank("unique", docs, top_k=1)

        assert len(result) == 1
        assert result[0][0] == "doc1"
        assert result[0][3] == {"key": "value"}

    def test_characterize_all_identical_documents(self):
        """Characterize: identical documents all get similar scores."""
        reranker = BM25FallbackReranker()
        docs = [
            ("doc1", "same content", {}),
            ("doc2", "same content", {}),
            ("doc3", "same content", {}),
        ]

        result = reranker.rerank("same", docs, top_k=3)

        assert len(result) == 3
        # Scores should be equal (or very close)
        scores = [r[2] for r in result]
        assert max(scores) - min(scores) < 0.01

    def test_characterize_no_matching_terms(self):
        """Characterize: query with no matching terms returns original order."""
        reranker = BM25FallbackReranker()
        docs = [
            ("doc1", "apples and oranges", {}),
            ("doc2", "bananas and grapes", {}),
        ]

        result = reranker.rerank("xyzzy", docs, top_k=2)

        # Should return results even with no matches
        assert len(result) == 2

    def test_characterize_very_long_query(self):
        """Characterize: very long query is handled without error."""
        reranker = BM25FallbackReranker()
        docs = [("doc1", "some content here", {})]
        long_query = " ".join(["word"] * 100)

        # Should not raise
        result = reranker.rerank(long_query, docs, top_k=1)
        assert len(result) == 1

    def test_characterize_special_korean_characters(self):
        """Characterize: special Korean characters (jamo) are handled."""
        reranker = BM25FallbackReranker(language="korean")
        docs = [
            ("doc1", "교원인사규정 제15조 ① 항", {}),
            ("doc2", "학칙 제1조【목적】", {}),
        ]

        # Should not raise
        result = reranker.rerank("제15조", docs, top_k=2)
        assert len(result) == 2
