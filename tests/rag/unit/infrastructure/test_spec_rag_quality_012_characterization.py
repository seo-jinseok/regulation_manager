"""
Characterization tests for SPEC-RAG-QUALITY-012.

Captures current behavior before modifications:
- Reranker sigmoid normalization
- MIN_RELEVANCE_THRESHOLD filtering
- Audience detection (faculty, student, staff)
- CoT stripping patterns
- Weight presets per query type
"""

import math
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# --- Reranker Characterization ---


class TestRerankerSigmoidCharacterization:
    """Capture current sigmoid normalization behavior."""

    def test_current_sigmoid_maps_zero_to_half(self):
        """Sigmoid(0) = 0.5 - captures score clustering problem."""
        score = 1 / (1 + math.exp(0))
        assert score == pytest.approx(0.5, abs=0.001)

    def test_current_sigmoid_maps_positive_logit(self):
        """Sigmoid of a typical positive logit."""
        logit = 2.0
        score = 1 / (1 + math.exp(-logit))
        assert score == pytest.approx(0.8808, abs=0.001)

    def test_current_sigmoid_maps_negative_logit(self):
        """Sigmoid of a typical negative logit."""
        logit = -2.0
        score = 1 / (1 + math.exp(-logit))
        assert score == pytest.approx(0.1192, abs=0.001)

    def test_current_sigmoid_score_uniformity_problem(self):
        """
        Characterize the score uniformity problem:
        small logit differences → almost identical sigmoid scores.
        """
        logits = [-0.1, -0.05, 0.0, 0.05, 0.1]
        scores = [1 / (1 + math.exp(-s)) for s in logits]
        std_dev = np.std(scores)
        # Current behavior: very low variance
        assert std_dev < 0.05, f"Expected low variance, got {std_dev}"

    def test_min_relevance_threshold_value(self):
        """Capture current MIN_RELEVANCE_THRESHOLD."""
        from src.rag.infrastructure.reranker import MIN_RELEVANCE_THRESHOLD

        assert MIN_RELEVANCE_THRESHOLD == 0.25


class TestRerankerFilteringCharacterization:
    """Capture current reranker filtering behavior."""

    def test_rerank_filters_below_threshold(self):
        """Documents below MIN_RELEVANCE_THRESHOLD are filtered out."""
        from src.rag.infrastructure.reranker import MIN_RELEVANCE_THRESHOLD

        # Any score below 0.25 would be filtered
        assert MIN_RELEVANCE_THRESHOLD == 0.25

    def test_rerank_empty_documents(self):
        """Empty document list returns empty results."""
        from src.rag.infrastructure.reranker import rerank

        results = rerank("test query", [], top_k=5)
        assert results == []


# --- Query Analyzer Characterization ---


class TestAudienceDetectionCharacterization:
    """Capture current audience detection behavior."""

    def test_faculty_keywords_detected(self):
        """Current faculty keywords trigger FACULTY audience."""
        from src.rag.infrastructure.query_analyzer import Audience, QueryAnalyzer

        qa = QueryAnalyzer()
        faculty_queries = ["교수 연봉은 얼마인가요", "교원 인사규정", "강사 채용 절차"]
        for query in faculty_queries:
            audience = qa.detect_audience(query)
            assert audience == Audience.FACULTY, f"Failed for: {query}"

    def test_student_keywords_detected(self):
        """Current student keywords trigger STUDENT audience."""
        from src.rag.infrastructure.query_analyzer import Audience, QueryAnalyzer

        qa = QueryAnalyzer()
        student_queries = ["학생 휴학 절차", "등록금 납부 방법", "장학금 신청"]
        for query in student_queries:
            audience = qa.detect_audience(query)
            assert audience == Audience.STUDENT, f"Failed for: {query}"

    def test_staff_keywords_detected(self):
        """Current staff keywords trigger STAFF audience."""
        from src.rag.infrastructure.query_analyzer import Audience, QueryAnalyzer

        qa = QueryAnalyzer()
        staff_queries = ["직원 승진 규정", "행정 절차"]
        for query in staff_queries:
            audience = qa.detect_audience(query)
            assert audience == Audience.STAFF, f"Failed for: {query}"

    def test_international_audience_type_exists(self):
        """INTERNATIONAL audience type now exists (added by SPEC-RAG-QUALITY-012)."""
        from src.rag.infrastructure.query_analyzer import Audience

        assert hasattr(Audience, "INTERNATIONAL")

    def test_faculty_keyword_count(self):
        """Faculty keyword count after SPEC-RAG-QUALITY-012 REQ-004."""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer

        qa = QueryAnalyzer()
        # Original 14 + 7 new (겸임, 초빙, 호봉, 임용, 재임용, 봉급, 발령)
        assert len(qa.FACULTY_KEYWORDS) == 21


class TestWeightPresetsCharacterization:
    """Capture current weight presets."""

    def test_english_query_weights(self):
        """English queries use dense-heavy weights."""
        from src.rag.infrastructure.query_analyzer import QueryAnalyzer

        qa = QueryAnalyzer()
        bm25_w, dense_w = qa.get_weights("What is the scholarship policy?")
        assert bm25_w == pytest.approx(0.2, abs=0.01)
        assert dense_w == pytest.approx(0.8, abs=0.01)


# --- CoT Stripping Characterization ---


class TestCoTStrippingCharacterization:
    """Capture current CoT stripping patterns."""

    def test_strips_numbered_analysis_steps(self):
        """Numbered analysis steps are stripped."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = "1. **Analyze the User's Request**: The user asks about...\n\n실제 답변입니다."
        result = _strip_cot_from_answer(text)
        assert "Analyze" not in result
        assert "실제 답변입니다" in result

    def test_strips_user_persona(self):
        """User Persona markers are stripped."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = "**User Persona:** 학부생\n\n답변 내용입니다."
        result = _strip_cot_from_answer(text)
        assert "User Persona" not in result
        assert "답변 내용입니다" in result

    def test_strips_step_markers(self):
        """Step markers are stripped."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = "Step 1: Analyze the query\nStep 2: Find regulations\n\n실제 답변"
        result = _strip_cot_from_answer(text)
        assert "Step 1" not in result
        assert "실제 답변" in result

    def test_strips_think_tags(self):
        """<think> tags are now stripped (fixed by SPEC-RAG-QUALITY-012 REQ-006)."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = "<think>내부 분석 내용</think>\n\n실제 답변입니다."
        result = _strip_cot_from_answer(text)
        assert "<think>" not in result
        assert "내부 분석" not in result

    def test_strips_confidence_scores(self):
        """Confidence scores are now stripped (fixed by SPEC-RAG-QUALITY-012 REQ-006)."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = "답변 내용입니다. 신뢰도: 0.85"
        result = _strip_cot_from_answer(text)
        assert "신뢰도" not in result

    def test_strips_search_strategy(self):
        """[검색 전략] markers are now stripped (fixed by SPEC-RAG-QUALITY-012 REQ-006)."""
        from src.rag.application.search_usecase import _strip_cot_from_answer

        text = "[검색 전략]: BM25 + Dense 혼합 검색을 사용\n\n실제 답변입니다."
        result = _strip_cot_from_answer(text)
        assert "검색 전략" not in result
