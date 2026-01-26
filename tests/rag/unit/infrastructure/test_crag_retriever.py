"""
Unit tests for CRAG Retriever (Cycle 9).

Tests cover:
- Relevance score calculation
- T-Fix trigger conditions
- Document re-ranking
- CRAG metrics tracking
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.rag.domain.entities import Chunk, ChunkLevel, SearchResult
from src.rag.infrastructure.crag_retriever import (
    CRAGMetrics,
    CRAGPipeline,
    CRAGRetriever,
    RetrievalQuality,
)


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    chunks = [
        Chunk(
            id="chunk1",
            rule_code="3-1-24",
            level=ChunkLevel.ARTICLE,
            title="교원 인사 규정 제8조",
            text="교원의 승진은 교원인사위원회의 심의를 거쳐 총장이 행한다.",
            embedding_text="교원의 승진은 교원인사위원회의 심의를 거쳐 총장이 행한다.",
            full_text="제8조(승진) 교원의 승진은 교원인사위원회의 심의를 거쳐 총장이 행한다.",
            parent_path=["교원인사규정"],
            token_count=50,
            keywords=[],
            is_searchable=True,
        ),
        Chunk(
            id="chunk2",
            rule_code="3-2-10",
            level=ChunkLevel.ARTICLE,
            title="학사 규정 제15조",
            text="졸업 학점은 140학점 이상이어야 한다.",
            embedding_text="졸업 학점은 140학점 이상이어야 한다.",
            full_text="제15조(졸업학점) 졸업 학점은 140학점 이상이어야 한다.",
            parent_path=["학사규정"],
            token_count=30,
            keywords=[],
            is_searchable=True,
        ),
        Chunk(
            id="chunk3",
            rule_code="3-1-25",
            level=ChunkLevel.ARTICLE,
            title="교원 인사 규정 제9조",
            text="정년은 65세로 한다.",
            embedding_text="정년은 65세로 한다.",
            full_text="제9조(정년) 정년은 65세로 한다.",
            parent_path=["교원인사규정"],
            token_count=20,
            keywords=[],
            is_searchable=True,
        ),
    ]
    return chunks


@pytest.fixture
def sample_results(sample_chunks):
    """Create sample search results."""
    return [
        SearchResult(chunk=sample_chunks[0], score=0.85, rank=0),
        SearchResult(chunk=sample_chunks[1], score=0.65, rank=1),
        SearchResult(chunk=sample_chunks[2], score=0.45, rank=2),
    ]


class TestCRAGMetrics:
    """Tests for CRAGMetrics dataclass."""

    def test_initial_metrics(self):
        """Test that metrics initialize with zeros."""
        metrics = CRAGMetrics()
        assert metrics.total_evaluations == 0
        assert metrics.excellent_count == 0
        assert metrics.poor_rate == 0.0

    def test_record_evaluation_excellent(self):
        """Test recording an excellent evaluation."""
        metrics = CRAGMetrics()
        metrics.record_evaluation(RetrievalQuality.EXCELLENT, 10.0)

        assert metrics.total_evaluations == 1
        assert metrics.excellent_count == 1
        assert metrics.avg_evaluation_time_ms == 10.0

    def test_record_tfix(self):
        """Test recording T-Fix attempts."""
        metrics = CRAGMetrics()
        metrics.record_tfix(successful=True, tfix_time_ms=100.0)
        metrics.record_tfix(successful=False, tfix_time_ms=50.0)

        assert metrics.tfix_triggered == 2
        assert metrics.tfix_successful == 1
        assert metrics.tfix_failed == 1
        assert metrics.tfix_success_rate == 0.5


class TestCRAGRetriever:
    """Tests for CRAGRetriever class."""

    @pytest.fixture
    def retriever(self):
        """Create a CRAGRetriever instance."""
        return CRAGRetriever(
            hybrid_searcher=None,
            query_analyzer=None,
            llm_client=None,
            enable_tfix=True,
            enable_rerank=True,
        )

    def test_initialization(self, retriever):
        """Test retriever initialization."""
        assert retriever.enable_tfix is True
        assert retriever.enable_rerank is True
        assert retriever.metrics is not None

    def test_evaluate_retrieval_quality_empty_results(self, retriever):
        """Test evaluation with empty results."""
        quality, score = retriever.evaluate_retrieval_quality("test query", [], "medium")

        assert quality == RetrievalQuality.POOR
        assert score == 0.0
        assert retriever.metrics.total_evaluations == 1

    def test_evaluate_retrieval_quality_high_score(self, retriever, sample_results):
        """Test evaluation with high-scoring results."""
        high_score_results = [
            SearchResult(chunk=sample_results[0].chunk, score=0.90, rank=0),
            SearchResult(chunk=sample_results[1].chunk, score=0.80, rank=1),
        ]

        quality, score = retriever.evaluate_retrieval_quality(
            "교원 승진 절차", high_score_results, "medium"
        )

        assert quality == RetrievalQuality.EXCELLENT
        assert score > 0.7

    def test_should_trigger_tfix_poor_quality(self, retriever):
        """Test T-Fix trigger for poor quality."""
        assert retriever.should_trigger_tfix(RetrievalQuality.POOR, 0.3) is True

    def test_should_trigger_tfix_excellent_quality(self, retriever):
        """Test T-Fix trigger for excellent quality."""
        assert retriever.should_trigger_tfix(RetrievalQuality.EXCELLENT, 0.9) is False

    def test_tokenize_korean(self, retriever):
        """Test Korean tokenization."""
        tokens = retriever._tokenize("교원의 승진 절차는 어떻게 되나요?")
        assert len(tokens) > 0
        assert "교원" in tokens or "승진" in tokens

    def test_tokenize_filters_stopwords(self, retriever):
        """Test that stopwords are filtered out."""
        tokens = retriever._tokenize("교원은 승진이 됩니다")
        assert "은" not in tokens
        assert "이" not in tokens


class TestCRAGPipeline:
    """Tests for CRAGPipeline class."""

    @pytest.fixture
    def pipeline(self):
        """Create a CRAGPipeline instance."""
        retriever = CRAGRetriever(
            hybrid_searcher=None,
            query_analyzer=None,
            llm_client=None,
            enable_tfix=True,
            enable_rerank=True,
        )
        return CRAGPipeline(retriever)

    @pytest.fixture
    def sample_results(self, sample_chunks):
        """Create sample search results."""
        return [
            SearchResult(chunk=sample_chunks[0], score=0.85, rank=0),
            SearchResult(chunk=sample_chunks[1], score=0.65, rank=1),
        ]

    @pytest.mark.asyncio
    async def test_search_with_excellent_results(self, pipeline, sample_results):
        """Test pipeline with excellent initial results."""
        with patch.object(
            pipeline.retriever,
            "evaluate_retrieval_quality",
            return_value=(RetrievalQuality.EXCELLENT, 0.9),
        ):
            result = await pipeline.search("query", sample_results, "medium")

        assert len(result) > 0
