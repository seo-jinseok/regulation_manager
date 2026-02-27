"""
Implementation tests for SPEC-RAG-QUALITY-012.

Validates new behavior after modifications:
- REQ-001: Temperature-scaled sigmoid + batch normalization
- REQ-002: Adaptive threshold with top-1 fallback
- REQ-004: Extended faculty keyword coverage
- REQ-005: International student audience detection
- REQ-006: Enhanced CoT pattern stripping
"""

import math
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# --- REQ-001: Score Calibration Tests ---


class TestCalibrateScores:
    """REQ-001: Temperature-scaled sigmoid + batch min-max normalization."""

    def test_empty_list_returns_empty(self):
        from src.rag.infrastructure.reranker import _calibrate_scores

        assert _calibrate_scores([]) == []

    def test_single_score_applies_sigmoid(self):
        from src.rag.infrastructure.reranker import (
            SIGMOID_TEMPERATURE,
            _calibrate_scores,
        )

        result = _calibrate_scores([0.0])
        # sigmoid(0 / T) = 0.5
        assert result[0] == pytest.approx(0.5, abs=0.01)

    def test_batch_normalization_spreads_scores(self):
        """Batch of close logits should produce well-spread normalized scores."""
        from src.rag.infrastructure.reranker import _calibrate_scores

        logits = [-0.1, -0.05, 0.0, 0.05, 0.1]
        scores = _calibrate_scores(logits)

        # After batch normalization, std_dev should be >= 0.15
        std_dev = float(np.std(scores))
        assert std_dev >= 0.15, f"std_dev={std_dev}, expected >= 0.15"

    def test_highest_logit_gets_highest_score(self):
        """Ordering should be preserved."""
        from src.rag.infrastructure.reranker import _calibrate_scores

        logits = [1.0, -0.5, 0.3, -1.0, 0.7]
        scores = _calibrate_scores(logits)

        # The max logit (1.0) should produce the max score
        assert scores[0] == max(scores)
        # The min logit (-1.0) should produce the min score
        assert scores[3] == min(scores)

    def test_score_range_within_bounds(self):
        """All normalized scores should be in [0.1, 0.95]."""
        from src.rag.infrastructure.reranker import _calibrate_scores

        logits = [-5.0, -2.0, 0.0, 2.0, 5.0]
        scores = _calibrate_scores(logits)

        for s in scores:
            assert 0.1 <= s <= 0.95, f"Score {s} outside [0.1, 0.95]"

    def test_identical_logits_return_uniform(self):
        """All same logits → uniform 0.5."""
        from src.rag.infrastructure.reranker import _calibrate_scores

        scores = _calibrate_scores([0.5, 0.5, 0.5])
        for s in scores:
            assert s == pytest.approx(0.5, abs=0.01)

    def test_two_logits_discriminated(self):
        """Even two logits with small difference should be discriminated."""
        from src.rag.infrastructure.reranker import _calibrate_scores

        scores = _calibrate_scores([0.0, 0.2])
        assert scores[1] > scores[0]
        assert abs(scores[1] - scores[0]) > 0.1


# --- REQ-002: Adaptive Threshold Tests ---


class TestAdaptiveThreshold:
    """REQ-002: Complexity-adaptive filtering with top-1 fallback."""

    def test_get_adaptive_threshold_simple(self):
        from src.rag.infrastructure.reranker import get_adaptive_threshold

        assert get_adaptive_threshold("simple") == 0.50

    def test_get_adaptive_threshold_medium(self):
        from src.rag.infrastructure.reranker import get_adaptive_threshold

        assert get_adaptive_threshold("medium") == 0.40

    def test_get_adaptive_threshold_complex(self):
        from src.rag.infrastructure.reranker import get_adaptive_threshold

        assert get_adaptive_threshold("complex") == 0.35

    def test_get_adaptive_threshold_unknown_falls_to_medium(self):
        from src.rag.infrastructure.reranker import get_adaptive_threshold

        assert get_adaptive_threshold("unknown") == 0.40

    def test_simple_threshold_higher_than_complex(self):
        """Simple queries should have stricter threshold."""
        from src.rag.infrastructure.reranker import ADAPTIVE_THRESHOLDS

        assert ADAPTIVE_THRESHOLDS["simple"] > ADAPTIVE_THRESHOLDS["complex"]


class TestApplyAdaptiveFilter:
    """REQ-002: Filter documents with adaptive threshold + top-1 fallback."""

    def _make_result(self, score: float):
        from src.rag.infrastructure.reranker import RerankedResult

        return RerankedResult(
            doc_id=f"doc_{score}",
            content="test",
            score=score,
            original_rank=1,
            metadata={},
        )

    def test_empty_list_returns_empty(self):
        from src.rag.infrastructure.reranker import _apply_adaptive_filter

        assert _apply_adaptive_filter([], 0.5) == []

    def test_all_above_threshold_kept(self):
        from src.rag.infrastructure.reranker import _apply_adaptive_filter

        results = [self._make_result(0.9), self._make_result(0.7), self._make_result(0.6)]
        filtered = _apply_adaptive_filter(results, 0.5)
        assert len(filtered) == 3

    def test_below_threshold_filtered(self):
        from src.rag.infrastructure.reranker import _apply_adaptive_filter

        results = [self._make_result(0.9), self._make_result(0.7), self._make_result(0.2)]
        filtered = _apply_adaptive_filter(results, 0.5)
        assert len(filtered) == 2
        assert all(r.score >= 0.5 for r in filtered)

    def test_all_below_threshold_keeps_top1(self):
        """AC-002-5: Always return at least 1 document."""
        from src.rag.infrastructure.reranker import _apply_adaptive_filter

        results = [self._make_result(0.3), self._make_result(0.2), self._make_result(0.1)]
        filtered = _apply_adaptive_filter(results, 0.5)
        assert len(filtered) == 1
        assert filtered[0].score == 0.3  # Top-1


# --- REQ-004: Faculty Keyword Extension Tests ---


class TestFacultyKeywordExtension:
    """REQ-004: Extended faculty keyword coverage."""

    def test_existing_faculty_keywords_still_detected(self):
        """Backward-compat: existing keywords still work."""
        from src.rag.infrastructure.query_analyzer import Audience, QueryAnalyzer

        qa = QueryAnalyzer()
        for query in ["교수 연봉", "교원 인사규정", "전임 교수"]:
            assert qa.detect_audience(query) == Audience.FACULTY, f"Failed: {query}"

    def test_new_faculty_keywords_detected(self):
        """New keywords: 겸임, 초빙, 연봉, 호봉, 임용, 승진, 재임용."""
        from src.rag.infrastructure.query_analyzer import Audience, QueryAnalyzer

        qa = QueryAnalyzer()
        new_keywords = ["겸임교수", "초빙교수", "호봉 산정", "임용 절차", "재임용 심사"]
        for query in new_keywords:
            assert qa.detect_audience(query) == Audience.FACULTY, f"Failed: {query}"

    def test_faculty_synonym_expansion(self):
        """Synonyms: 봉급→연봉→호봉, 임용→채용→발령."""
        from src.rag.infrastructure.query_analyzer import Audience, QueryAnalyzer

        qa = QueryAnalyzer()
        synonym_queries = ["교원 봉급 체계", "교수 발령 절차"]
        for query in synonym_queries:
            assert qa.detect_audience(query) == Audience.FACULTY, f"Failed: {query}"


# --- REQ-005: International Student Audience Tests ---


class TestInternationalAudience:
    """REQ-005: International student audience detection."""

    def test_international_audience_type_exists(self):
        """INTERNATIONAL enum value must exist."""
        from src.rag.infrastructure.query_analyzer import Audience

        assert hasattr(Audience, "INTERNATIONAL")

    def test_korean_international_keywords_detected(self):
        """Korean international student keywords trigger INTERNATIONAL."""
        from src.rag.infrastructure.query_analyzer import Audience, QueryAnalyzer

        qa = QueryAnalyzer()
        queries = ["유학생 비자 연장", "외국인 등록", "체류 자격 변경"]
        for query in queries:
            assert qa.detect_audience(query) == Audience.INTERNATIONAL, f"Failed: {query}"

    def test_english_query_detected_as_international(self):
        """English queries auto-detect as international audience."""
        from src.rag.infrastructure.query_analyzer import Audience, QueryAnalyzer

        qa = QueryAnalyzer()
        queries = ["How do I apply for D-2 visa?", "Where is the international office?"]
        for query in queries:
            audience = qa.detect_audience(query)
            assert audience == Audience.INTERNATIONAL, f"Failed: {query}"

    def test_international_weight_preset(self):
        """International audience uses dense-heavy weights (0.20, 0.80)."""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer

        qa = QueryAnalyzer()
        # English query should trigger international weight preset
        bm25_w, dense_w = qa.get_weights("How to extend my visa?")
        assert bm25_w == pytest.approx(0.20, abs=0.05)
        assert dense_w == pytest.approx(0.80, abs=0.05)


# --- REQ-006: CoT Hardening Tests ---


class TestCoTHardening:
    """REQ-006: Enhanced CoT pattern stripping."""

    def test_strips_think_tags(self):
        """<think>...</think> must be stripped."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = "<think>내부 분석 내용: 이 질문은 학사 규정에 관한 것으로...</think>\n\n실제 답변입니다."
        result = _strip_cot_from_answer(text)
        assert "<think>" not in result
        assert "내부 분석 내용" not in result
        assert "실제 답변입니다" in result

    def test_strips_analysis_tags(self):
        """<analysis>...</analysis> must be stripped."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = "<analysis>쿼리 분석: 학생 신분 관련</analysis>\n\n답변입니다."
        result = _strip_cot_from_answer(text)
        assert "<analysis>" not in result
        assert "쿼리 분석" not in result

    def test_strips_internal_analysis_header(self):
        """## 내부 분석 sections must be stripped."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = "## 내부 분석\n이것은 내부 분석입니다.\n\n## 답변\n실제 답변입니다."
        result = _strip_cot_from_answer(text)
        assert "내부 분석" not in result
        assert "실제 답변입니다" in result

    def test_strips_search_strategy_marker(self):
        """[검색 전략] must be stripped."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = "[검색 전략]: BM25 + Dense 혼합 검색\n\n실제 답변입니다."
        result = _strip_cot_from_answer(text)
        assert "검색 전략" not in result
        assert "실제 답변입니다" in result

    def test_strips_confidence_score(self):
        """신뢰도: 0.X must be stripped."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = "답변 내용입니다.\n\n신뢰도: 0.85"
        result = _strip_cot_from_answer(text)
        assert "신뢰도" not in result
        assert "답변 내용" in result

    def test_strips_multiple_cot_patterns_combined(self):
        """All patterns stripped when combined in one response."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = (
            "<think>분석 중...</think>\n"
            "[검색 전략]: hybrid\n"
            "실제 답변입니다.\n"
            "신뢰도: 0.92"
        )
        result = _strip_cot_from_answer(text)
        assert "<think>" not in result
        assert "검색 전략" not in result
        assert "신뢰도" not in result
        assert "실제 답변" in result

    def test_preserves_clean_answer(self):
        """Clean answers without CoT markers should not be damaged."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = "학칙 제10조에 따르면 휴학 기간은 최대 4학기입니다."
        result = _strip_cot_from_answer(text)
        assert result.strip() == text
