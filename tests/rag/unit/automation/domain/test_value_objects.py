"""
Unit tests for automation domain value objects.

These tests specify the intended behavior of IntentAnalysis, FactCheck, and QualityScore.
Following Test-First DDD approach for greenfield development.
"""

import pytest

from src.rag.automation.domain.value_objects import (
    FactCheck,
    FactCheckStatus,
    IntentAnalysis,
    QualityDimensions,
    QualityScore,
)


class TestIntentAnalysis:
    """Test IntentAnalysis value object behavior."""

    def test_intent_analysis_creation(self):
        """WHEN creating intent analysis, THEN it should have 3-level intent."""
        intent = IntentAnalysis(
            surface_intent="휴학 절차 문의",
            hidden_intent="휴학 신청 방법과 필요 서류 확인",
            behavioral_intent="휴학 신청서 제출",
        )

        assert intent.surface_intent == "휴학 절차 문의"
        assert intent.hidden_intent == "휴학 신청 방법과 필요 서류 확인"
        assert intent.behavioral_intent == "휴학 신청서 제출"

    def test_intent_analysis_immutability(self):
        """WHEN creating intent analysis, THEN it should be immutable."""
        intent = IntentAnalysis(
            surface_intent="휴학 절차 문의",
            hidden_intent="휴학 신청 방법과 필요 서류 확인",
            behavioral_intent="휴학 신청서 제출",
        )

        # Should be frozen (immutable)
        with pytest.raises(Exception):  # FrozenInstanceError
            intent.surface_intent = "다른 의도"

    def test_intent_analysis_equality(self):
        """WHEN comparing two intent analyses, THEN they should be equal if all attributes match."""
        intent1 = IntentAnalysis(
            surface_intent="휴학 절차 문의",
            hidden_intent="휴학 신청 방법과 필요 서류 확인",
            behavioral_intent="휴학 신청서 제출",
        )

        intent2 = IntentAnalysis(
            surface_intent="휴학 절차 문의",
            hidden_intent="휴학 신청 방법과 필요 서류 확인",
            behavioral_intent="휴학 신청서 제출",
        )

        assert intent1 == intent2


class TestFactCheck:
    """Test FactCheck value object behavior."""

    def test_fact_check_creation_pass(self):
        """WHEN creating a passing fact check, THEN it should have correct status."""
        fact_check = FactCheck(
            claim="휴학 신청은 학기 시작 2주 전까지 가능합니다",
            status=FactCheckStatus.PASS,
            source="교육과정규정 제13조",
            confidence=0.95,
        )

        assert fact_check.status == FactCheckStatus.PASS
        assert fact_check.source == "교육과정규정 제13조"
        assert fact_check.confidence == 0.95

    def test_fact_check_creation_fail(self):
        """WHEN creating a failing fact check, THEN it should have correct status."""
        fact_check = FactCheck(
            claim="휴학 신청은 언제든지 가능합니다",
            status=FactCheckStatus.FAIL,
            source="교육과정규정 제13조",
            confidence=0.98,
            correction="휴학 신청은 학기 시작 2주 전까지 가능함",
        )

        assert fact_check.status == FactCheckStatus.FAIL
        assert fact_check.correction == "휴학 신청은 학기 시작 2주 전까지 가능함"

    def test_fact_check_all_statuses(self):
        """THEN all fact check statuses should be defined."""
        expected_statuses = {
            FactCheckStatus.PASS,
            FactCheckStatus.FAIL,
            FactCheckStatus.UNCERTAIN,
        }

        assert len(FactCheckStatus) == 3
        assert set(FactCheckStatus) == expected_statuses

    def test_fact_check_confidence_range(self):
        """WHEN creating fact check, THEN confidence should be between 0 and 1."""
        fact_check = FactCheck(
            claim="휴학 신청은 학기 시작 2주 전까지 가능합니다",
            status=FactCheckStatus.PASS,
            source="교육과정규정 제13조",
            confidence=0.95,
        )

        assert 0.0 <= fact_check.confidence <= 1.0

    def test_fact_check_immutability(self):
        """WHEN creating fact check, THEN it should be immutable."""
        fact_check = FactCheck(
            claim="휴학 신청은 학기 시작 2주 전까지 가능합니다",
            status=FactCheckStatus.PASS,
            source="교육과정규정 제13조",
            confidence=0.95,
        )

        # Should be frozen (immutable)
        with pytest.raises(Exception):  # FrozenInstanceError
            fact_check.status = FactCheckStatus.FAIL


class TestQualityScore:
    """Test QualityScore value object behavior."""

    def test_quality_score_creation_perfect(self):
        """WHEN creating perfect quality score, THEN it should have 5.0 total."""
        dimensions = QualityDimensions(
            accuracy=1.0,
            completeness=1.0,
            relevance=1.0,
            source_citation=1.0,
            practicality=0.5,
            actionability=0.5,
        )

        score = QualityScore(
            dimensions=dimensions,
            total_score=5.0,
            is_pass=True,
        )

        assert score.total_score == 5.0
        assert score.is_pass is True
        assert score.dimensions.accuracy == 1.0

    def test_quality_score_creation_failing(self):
        """WHEN creating failing quality score, THEN it should be below threshold."""
        dimensions = QualityDimensions(
            accuracy=0.5,
            completeness=0.5,
            relevance=0.5,
            source_citation=0.5,
            practicality=0.0,
            actionability=0.0,
        )

        score = QualityScore(
            dimensions=dimensions,
            total_score=2.5,
            is_pass=False,
        )

        assert score.total_score == 2.5
        assert score.is_pass is False

    def test_quality_score_threshold(self):
        """WHEN total score >= 4.0, THEN it should pass."""
        dimensions_pass = QualityDimensions(
            accuracy=1.0,
            completeness=1.0,
            relevance=1.0,
            source_citation=1.0,
            practicality=0.5,
            actionability=0.5,
        )

        score_pass = QualityScore(
            dimensions=dimensions_pass,
            total_score=5.0,
            is_pass=True,
        )

        assert score_pass.is_pass is True
        assert score_pass.total_score >= 4.0

    def test_quality_score_partial_success(self):
        """WHEN total score is 3.0~3.9, THEN it should be partial success (= fail)."""
        dimensions_partial = QualityDimensions(
            accuracy=0.7,
            completeness=0.7,
            relevance=0.7,
            source_citation=0.7,
            practicality=0.2,
            actionability=0.0,
        )

        score_partial = QualityScore(
            dimensions=dimensions_partial,
            total_score=3.0,
            is_pass=False,
        )

        assert score_partial.is_pass is False
        assert 3.0 <= score_partial.total_score < 4.0

    def test_quality_score_immutability(self):
        """WHEN creating quality score, THEN it should be immutable."""
        dimensions = QualityDimensions(
            accuracy=1.0,
            completeness=1.0,
            relevance=1.0,
            source_citation=1.0,
            practicality=0.5,
            actionability=0.5,
        )

        score = QualityScore(
            dimensions=dimensions,
            total_score=5.0,
            is_pass=True,
        )

        # Should be frozen (immutable)
        with pytest.raises(Exception):  # FrozenInstanceError
            score.total_score = 4.0
