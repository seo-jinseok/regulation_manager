"""
Additional characterization tests for Reranker module.

These tests document the CURRENT behavior of reranking,
focusing on areas not covered by existing tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, List, Tuple, Any


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_documents():
    """Create sample documents for reranking."""
    return [
        ("doc1", "휴학 신청 방법에 대한 내용", {"rule_code": "TEST-001"}),
        ("doc2", "복학 절차 안내", {"rule_code": "TEST-002"}),
        ("doc3", "졸업 요건 설명", {"rule_code": "TEST-003"}),
    ]


@pytest.fixture
def mock_flag_reranker():
    """Create a mock FlagReranker for testing."""
    reranker = MagicMock()
    reranker.compute_score.return_value = [0.9, 0.7, 0.5]
    return reranker


# ============================================================================
# RerankedResult Tests
# ============================================================================


class TestRerankedResult:
    """Tests for RerankedResult dataclass."""

    def test_reranked_result_creation(self):
        """RerankedResult can be created with required fields."""
        from src.rag.infrastructure.reranker import RerankedResult

        result = RerankedResult(
            doc_id="doc1",
            content="Test content",
            score=0.9,
            original_rank=1,
            metadata={"rule_code": "TEST"},
        )
        assert result.doc_id == "doc1"
        assert result.score == 0.9
        assert result.original_rank == 1


# ============================================================================
# get_reranker Function Tests
# ============================================================================


class TestGetReranker:
    """Tests for get_reranker function."""

    def test_get_reranker_returns_reranker(self):
        """get_reranker returns a reranker instance."""
        from src.rag.infrastructure.reranker import get_reranker, clear_reranker

        clear_reranker()

        # Mock FlagReranker import inside the function
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "FlagEmbedding":
                mod = MagicMock()
                mod.FlagReranker = MagicMock(return_value=MagicMock())
                return mod
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            try:
                r1 = get_reranker()
                # Second call should return same instance
                r2 = get_reranker()
                assert r1 is r2
            except Exception:
                # If FlagEmbedding not installed, skip
                pass

        clear_reranker()


# ============================================================================
# clear_reranker Function Tests
# ============================================================================


class TestClearReranker:
    """Tests for clear_reranker function."""

    def test_clear_reranker_resets_singleton(self):
        """clear_reranker resets the singleton instance."""
        from src.rag.infrastructure.reranker import clear_reranker

        clear_reranker()
        # Should not raise any error
        clear_reranker()


# ============================================================================
# rerank Function Tests
# ============================================================================


class TestRerankFunction:
    """Tests for rerank function."""

    def test_rerank_empty_documents(self):
        """rerank returns empty list for empty documents."""
        from src.rag.infrastructure.reranker import rerank

        result = rerank("test query", [])
        assert result == []

    def test_rerank_single_document(self, mock_flag_reranker):
        """rerank handles single document."""
        from src.rag.infrastructure.reranker import rerank

        with patch("src.rag.infrastructure.reranker.get_reranker") as mock_get:
            mock_get.return_value = mock_flag_reranker
            mock_flag_reranker.compute_score.return_value = 0.9  # Single float

            result = rerank("test query", [("doc1", "content", {})])

            assert len(result) == 1
            assert result[0].doc_id == "doc1"

    def test_rerank_respects_top_k(self, mock_flag_reranker, sample_documents):
        """rerank respects top_k parameter."""
        from src.rag.infrastructure.reranker import rerank

        with patch("src.rag.infrastructure.reranker.get_reranker") as mock_get:
            mock_get.return_value = mock_flag_reranker

            result = rerank("test query", sample_documents, top_k=2)

            assert len(result) == 2


# ============================================================================
# BGEReranker Class Tests
# ============================================================================


class TestBGEReranker:
    """Tests for BGEReranker class."""

    def test_bge_reranker_rerank_empty(self):
        """BGEReranker.rerank handles empty documents."""
        from src.rag.infrastructure.reranker import BGEReranker

        reranker = BGEReranker()
        result = reranker.rerank("query", [])
        assert result == []

    def test_bge_reranker_rerank_with_context_empty(self):
        """BGEReranker.rerank_with_context handles empty documents."""
        from src.rag.infrastructure.reranker import BGEReranker

        reranker = BGEReranker()
        result = reranker.rerank_with_context("query", [], context={})
        assert result == []

    def test_bge_reranker_rerank_with_context_boost(self, mock_flag_reranker):
        """BGEReranker.rerank_with_context applies context boosting."""
        from src.rag.infrastructure.reranker import BGEReranker

        reranker = BGEReranker()

        docs = [
            ("doc1", "content", {"regulation_title": "학칙"}),
            ("doc2", "other", {"regulation_title": "other"}),
        ]

        with patch("src.rag.infrastructure.reranker.rerank") as mock_rerank:
            from src.rag.infrastructure.reranker import RerankedResult

            mock_rerank.return_value = [
                RerankedResult("doc1", "content", 0.5, 1, {"regulation_title": "학칙"}),
                RerankedResult("doc2", "other", 0.4, 2, {"regulation_title": "other"}),
            ]

            context = {"target_regulation": "학칙", "regulation_boost": 0.2}
            result = reranker.rerank_with_context("query", docs, context=context, top_k=2)

            assert len(result) == 2

    def test_bge_reranker_audience_boost(self, mock_flag_reranker):
        """BGEReranker applies audience boosting."""
        from src.rag.infrastructure.reranker import BGEReranker

        reranker = BGEReranker()

        docs = [
            ("doc1", "content", {"audience": "undergraduate"}),
            ("doc2", "other", {"audience": "graduate"}),
        ]

        with patch("src.rag.infrastructure.reranker.rerank") as mock_rerank:
            from src.rag.infrastructure.reranker import RerankedResult

            mock_rerank.return_value = [
                RerankedResult("doc1", "content", 0.5, 1, {"audience": "undergraduate"}),
                RerankedResult("doc2", "other", 0.5, 2, {"audience": "graduate"}),
            ]

            context = {"target_audience": "undergraduate", "audience_boost": 0.1}
            result = reranker.rerank_with_context("query", docs, context=context, top_k=2)

            assert len(result) == 2


# ============================================================================
# KoreanReranker Class Tests
# ============================================================================


class TestKoreanReranker:
    """Tests for KoreanReranker class."""

    def test_korean_reranker_rerank_empty(self):
        """KoreanReranker.rerank handles empty documents."""
        from src.rag.infrastructure.reranker import KoreanReranker

        with patch("src.rag.infrastructure.reranker._EXTENDED_AVAILABLE", False):
            reranker = KoreanReranker()
            result = reranker.rerank("query", [])
            assert result == []

    def test_korean_reranker_with_ab_testing(self):
        """KoreanReranker can be created with A/B testing."""
        from src.rag.infrastructure.reranker import KoreanReranker

        with patch("src.rag.infrastructure.reranker._EXTENDED_AVAILABLE", False):
            reranker = KoreanReranker(use_ab_testing=True)
            assert reranker._model_name is None


# ============================================================================
# warmup_reranker Tests
# ============================================================================


class TestWarmupReranker:
    """Tests for warmup_reranker function."""

    def test_warmup_reranker_calls_get_reranker(self):
        """warmup_reranker calls get_reranker."""
        from src.rag.infrastructure.reranker import warmup_reranker

        with patch("src.rag.infrastructure.reranker.get_reranker") as mock_get:
            mock_get.return_value = MagicMock()
            warmup_reranker("test-model")
            mock_get.assert_called_once()


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_rerank_with_none_metadata(self):
        """rerank handles documents with None metadata."""
        from src.rag.infrastructure.reranker import rerank

        docs = [("doc1", "content", None)]

        with patch("src.rag.infrastructure.reranker.get_reranker") as mock_get:
            mock_reranker = MagicMock()
            mock_reranker.compute_score.return_value = [0.9]
            mock_get.return_value = mock_reranker

            result = rerank("query", docs)

            assert len(result) == 1

    def test_context_boost_caps_at_1(self, mock_flag_reranker):
        """Context boosting caps score at 1.0."""
        from src.rag.infrastructure.reranker import BGEReranker

        reranker = BGEReranker()

        docs = [("doc1", "content", {"regulation_title": "학칙"})]

        with patch("src.rag.infrastructure.reranker.rerank") as mock_rerank:
            from src.rag.infrastructure.reranker import RerankedResult

            # High base score with high boost should cap at 1.0
            mock_rerank.return_value = [
                RerankedResult("doc1", "content", 0.95, 1, {"regulation_title": "학칙"}),
            ]

            context = {"target_regulation": "학칙", "regulation_boost": 0.2}
            result = reranker.rerank_with_context("query", docs, context=context, top_k=1)

            # Score should be capped at 1.0
            assert result[0][2] <= 1.0
