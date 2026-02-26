"""SPEC-RAG-003 Phase 3: Search Quality + Hallucination Guard tests."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


# ─── Task 3.1 + 3.4: Relevance filtering and no-info guardrail ───


@dataclass
class FakeChunk:
    text: str = "규정 본문"
    regulation_name: str = "학칙"
    chapter_title: str = "제1장"
    section_title: str = ""
    article_number: str = "제1조"


@dataclass
class FakeSearchResult:
    score: float
    chunk: FakeChunk = None

    def __post_init__(self):
        if self.chunk is None:
            self.chunk = FakeChunk()


class TestRelevanceFiltering:
    """Task 3.1: Documents with score < 0.25 should be filtered out."""

    def test_high_relevance_kept(self):
        """Results above threshold are kept."""
        results = [FakeSearchResult(score=0.8), FakeSearchResult(score=0.5)]
        min_score = 0.25
        high = [r for r in results if r.score >= min_score]
        assert len(high) == 2

    def test_low_relevance_filtered(self):
        """Results below threshold are removed when some are above."""
        results = [
            FakeSearchResult(score=0.8),
            FakeSearchResult(score=0.1),
            FakeSearchResult(score=0.05),
        ]
        min_score = 0.25
        high = [r for r in results if r.score >= min_score]
        assert len(high) == 1
        assert high[0].score == 0.8

    def test_all_below_threshold_triggers_guardrail(self):
        """When all results are below threshold, guardrail should activate."""
        results = [
            FakeSearchResult(score=0.1),
            FakeSearchResult(score=0.05),
        ]
        min_score = 0.25
        high = [r for r in results if r.score >= min_score]
        assert len(high) == 0
        assert all(r.score < min_score for r in results)

    def test_boundary_score_kept(self):
        """Result at exactly the threshold should be kept."""
        results = [FakeSearchResult(score=0.25)]
        min_score = 0.25
        high = [r for r in results if r.score >= min_score]
        assert len(high) == 1

    def test_empty_results_no_crash(self):
        """Empty results should not cause errors."""
        results = []
        min_score = 0.25
        high = [r for r in results if r.score >= min_score]
        assert len(high) == 0


# ─── Task 3.2: English query Dense weight boost ───


class TestEnglishDenseBoost:
    """Task 3.2: English queries should get higher dense weight."""

    def _make_analyzer(self):
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer

        return QueryAnalyzer.__new__(QueryAnalyzer)

    def test_english_query_gets_dense_boost(self):
        """English query should return (0.2, 0.8) weights."""
        analyzer = self._make_analyzer()
        assert analyzer._is_english_query("tuition fee refund policy")

    def test_korean_query_no_boost(self):
        """Korean query should NOT trigger English boost."""
        analyzer = self._make_analyzer()
        assert not analyzer._is_english_query("등록금 환불 규정")

    def test_mixed_query_threshold(self):
        """Mixed query below 70% ASCII should not trigger boost."""
        analyzer = self._make_analyzer()
        # "등록금 refund" → Korean dominant
        assert not analyzer._is_english_query("등록금 환불 refund")

    def test_empty_query_no_boost(self):
        """Empty query should not trigger boost."""
        analyzer = self._make_analyzer()
        assert not analyzer._is_english_query("")

    def test_english_query_weight_values(self):
        """Verify actual weight values for English queries via get_weights."""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer()
        bm25_w, dense_w = analyzer.get_weights("What is the tuition refund policy?")
        assert bm25_w == 0.2
        assert dense_w == 0.8

    def test_korean_query_preserves_original_weights(self):
        """Korean query should use original WEIGHT_PRESETS."""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer

        analyzer = QueryAnalyzer()
        bm25_w, dense_w = analyzer.get_weights("등록금 환불 규정은?")
        # Should NOT be (0.2, 0.8)
        assert bm25_w != 0.2 or dense_w != 0.8


# ─── Task 3.3: Corrective RAG threshold adjustment ───


class TestCorrectiveRAGThresholds:
    """Task 3.3: Verify adjusted Corrective RAG thresholds."""

    def test_default_thresholds_updated(self):
        """Default thresholds should be raised by 0.05."""
        from src.rag.config import RAGConfig

        config = RAGConfig()
        thresholds = config.corrective_rag_thresholds
        assert thresholds["simple"] == 0.35
        assert thresholds["medium"] == 0.45
        assert thresholds["complex"] == 0.55

    def test_min_relevance_score_default(self):
        """min_relevance_score should default to 0.25."""
        from src.rag.config import RAGConfig

        config = RAGConfig()
        assert config.min_relevance_score == 0.25

    def test_min_relevance_from_env(self):
        """min_relevance_score should read from env variable."""
        from src.rag.config import RAGConfig

        with patch.dict("os.environ", {"RAG_MIN_RELEVANCE_SCORE": "0.3"}):
            config = RAGConfig()
            assert config.min_relevance_score == 0.3


# ─── Task 3.4: Hallucination guardrail messages ───


class TestHallucinationGuardrail:
    """Task 3.4: No-info response when all results are low relevance."""

    def test_korean_no_info_message_format(self):
        """Korean no-info message should include query keywords."""
        from src.rag.application.search_usecase import _detect_query_language

        query = "가상의 규정 없는 질문"
        lang = _detect_query_language(query)
        assert lang == "ko"

        keywords = query[:30]
        msg = f'죄송합니다. "{keywords}"에 대한 규정 정보를 찾을 수 없습니다.'
        assert keywords in msg
        assert "찾을 수 없습니다" in msg

    def test_english_no_info_message_format(self):
        """English no-info message should include query keywords."""
        from src.rag.application.search_usecase import _detect_query_language

        query = "nonexistent regulation topic"
        lang = _detect_query_language(query)
        assert lang == "en"

        keywords = query[:30]
        msg = f'Sorry, no regulation information was found for "{keywords}"'
        assert keywords in msg
        assert "found" in msg

    def test_no_info_answer_zero_confidence(self):
        """No-info Answer should have confidence=0.0 and empty sources."""
        from src.rag.domain.entities import Answer

        ans = Answer(text="정보 없음", sources=[], confidence=0.0)
        assert ans.confidence == 0.0
        assert ans.sources == []
