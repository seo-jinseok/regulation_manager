"""
Focused tests for reranker module to improve coverage from 76% toward 90%.
Tests target high-value, testable code paths.

SPEC-RAG-Q-011: Updated FakeReranker to use predict() for CrossEncoder.
"""

from unittest.mock import MagicMock, patch

from src.rag.infrastructure.reranker import (
    BGEReranker,
    clear_reranker,
)


class FakeReranker:
    """
    Fake reranker for testing without loading the actual model.

    SPEC-RAG-Q-011: Uses predict() method to match CrossEncoder API.
    """

    def __init__(self, return_float=False):
        self.return_float = return_float
        self.call_count = 0

    def predict(self, pairs):
        """
        Return fake scores based on keyword matching.
        CrossEncoder uses predict() instead of compute_score().
        """
        self.call_count += 1
        if self.return_float and len(pairs) == 1:
            # Return a single score for single document
            # Use a value that maps to ~0.75 after sigmoid: sigmoid(1.1) ≈ 0.75
            import numpy as np
            return np.array([1.1])  # sigmoid(1.1) ≈ 0.75
        # Return raw scores that will be normalized via sigmoid
        import numpy as np
        return np.array([0.5 * (i + 1) for i in range(len(pairs))])


# Test single document returns float (line 102)
def test_single_document_float_score():
    """Test that single document returns float score."""
    fake = FakeReranker(return_float=True)
    with patch("src.rag.infrastructure.reranker.get_reranker", return_value=fake):
        from src.rag.infrastructure.reranker import rerank

        docs = [("doc1", "content", {})]
        result = rerank("query", docs, top_k=1)

        assert len(result) == 1
        # sigmoid(1.1) ≈ 0.750, allow small floating point difference
        assert abs(result[0].score - 0.750) < 0.001


# Test rerank_search_results (lines 139-169)
def test_rerank_search_results_empty():
    """Test rerank_search_results with empty results."""
    from src.rag.infrastructure.reranker import rerank_search_results

    result = rerank_search_results("query", [], top_k=10)
    assert result == []


def test_rerank_search_results_preserves_attributes():
    """Test that rerank preserves chunk attributes."""
    from src.rag.domain.entities import Chunk, ChunkLevel, SearchResult
    from src.rag.infrastructure.reranker import rerank_search_results

    chunk = Chunk(
        id="chunk1",
        text="content",
        title="test_title",
        parent_path=["path1", "path2"],
        rule_code="1-1-1",
        level=ChunkLevel.ARTICLE,
        embedding_text="content",
        full_text="content",
        token_count=10,
        keywords=[],
        is_searchable=True,
    )
    original = SearchResult(chunk=chunk, score=0.5, rank=1)

    fake = FakeReranker()
    with patch("src.rag.infrastructure.reranker.get_reranker", return_value=fake):
        results = rerank_search_results("query", [original], top_k=1)

        assert len(results) == 1
        assert results[0].chunk.id == "chunk1"
        assert results[0].chunk.title == "test_title"
        assert results[0].chunk.rule_code == "1-1-1"


# Test rerank_with_context empty documents (line 252)
def test_rerank_with_context_empty():
    """Test rerank_with_context with empty documents."""
    reranker = BGEReranker()
    result = reranker.rerank_with_context("query", [], context={}, top_k=10)
    assert result == []


def test_rerank_with_context_none_context():
    """Test rerank_with_context with None context."""
    reranker = BGEReranker()
    docs = [("doc1", "content", {})]

    result = reranker.rerank_with_context("query", docs, context=None, top_k=1)

    assert len(result) == 1


def test_reranker_custom_model_name():
    """Test BGEReranker with custom model name."""
    reranker = BGEReranker(model_name="custom/model")
    assert reranker._model_name == "custom/model"


def test_clear_reranker():
    """Test clear_reranker removes global instance."""
    import src.rag.infrastructure.reranker as reranker_module

    reranker_module._reranker = MagicMock()
    assert reranker_module._reranker is not None

    clear_reranker()

    assert reranker_module._reranker is None
