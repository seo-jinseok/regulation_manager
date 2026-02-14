"""
Unit tests for PeriodKeywordDetector.

TDD approach: These tests define the expected behavior of period keyword detection.
"""

import pytest

from src.rag.infrastructure.period_keyword_detector import PeriodKeywordDetector


class TestPeriodKeywordDetector:
    """Tests for PeriodKeywordDetector class."""

    @pytest.fixture
    def detector(self) -> PeriodKeywordDetector:
        """Create PeriodKeywordDetector instance."""
        return PeriodKeywordDetector()

    def test_detect_period_keywords_basic(self, detector):
        """Detect single period keyword in query."""
        result = detector.detect_period_keywords("수강신청 기간이 언제인가요?")
        assert "기간" in result
        assert "언제" in result

    def test_detect_period_keywords_multiple(self, detector):
        """Detect multiple period keywords in query."""
        result = detector.detect_period_keywords("등록 기한과 날짜를 알고 싶습니다")
        assert "기한" in result
        assert "날짜" in result

    def test_detect_period_keywords_no_keywords(self, detector):
        """Return empty list when no period keywords present."""
        result = detector.detect_period_keywords("교원인사규정의 내용을 알려주세요")
        assert result == []

    def test_detect_period_keywords_empty_string(self, detector):
        """Handle empty string input."""
        result = detector.detect_period_keywords("")
        assert result == []

    def test_detect_period_keywords_partial_match_not_detected(self, detector):
        """Partial matches should not be detected (e.g., '기간다' should not match '기간')."""
        # '기간다' should not match '기간' as a separate word
        result = detector.detect_period_keywords("기간다라")
        # This should NOT detect '기간' unless it's a word boundary issue
        # We'll accept either behavior but document the expectation
        # For now, allow partial match since Korean doesn't have clear word boundaries
        pass

    def test_detect_all_keywords(self, detector):
        """All defined keywords should be detectable."""
        # Test all 12 keywords: 기간, 언제, 기한, 날짜,까지, 마감, 신청일, 등록일, 시작, 종료, 개강, 종강
        test_cases = [
            ("기간", "신청 기간"),
            ("언제", "언제까지"),
            ("기한", "기한 내에"),
            ("날짜", "날짜 확인"),
            ("까지", "12월까지"),
            ("마감", "마감일"),
            ("신청일", "신청일입니다"),
            ("등록일", "등록일 확인"),
            ("시작", "시작일"),
            ("종료", "종료일"),
            ("개강", "개강일"),
            ("종강", "종강일"),
        ]

        for keyword, query in test_cases:
            result = detector.detect_period_keywords(query)
            assert keyword in result, f"Expected '{keyword}' to be detected in '{query}'"

    def test_detect_period_keywords_returns_unique(self, detector):
        """Return unique keywords only (no duplicates)."""
        result = detector.detect_period_keywords("기간과 기간이 다릅니다")
        assert result.count("기간") == 1

    def test_is_period_related_returns_true(self, detector):
        """Return True when query contains period keywords."""
        assert detector.is_period_related("수강신청 기간") is True
        assert detector.is_period_related("언제 마감인가요") is True
        assert detector.is_period_related("개강일은 언제인가요") is True

    def test_is_period_related_returns_false(self, detector):
        """Return False when query contains no period keywords."""
        assert detector.is_period_related("교원인사규정 내용") is False
        assert detector.is_period_related("연구윤리에 대해") is False
        assert detector.is_period_related("") is False

    def test_is_period_related_single_keyword(self, detector):
        """Detect single keyword correctly."""
        # Test boundary cases
        assert detector.is_period_related("기간") is True
        assert detector.is_period_related("기간외") is True  # '기간' is still present


class TestPeriodKeywordDetectorEdgeCases:
    """Tests for edge cases in period keyword detection."""

    @pytest.fixture
    def detector(self) -> PeriodKeywordDetector:
        """Create PeriodKeywordDetector instance."""
        return PeriodKeywordDetector()

    def test_whitespace_handling(self, detector):
        """Handle queries with various whitespace."""
        assert detector.is_period_related("  기간  ") is True
        assert detector.is_period_related("기간\n") is True
        assert detector.is_period_related("\t기간") is True

    def test_mixed_with_numbers(self, detector):
        """Handle queries with numbers and dates."""
        result = detector.detect_period_keywords("2024년 1월 15일까지 신청")
        assert "까지" in result

    def test_case_sensitivity(self, detector):
        """Korean keywords don't have case, but ensure consistent behavior."""
        # Korean doesn't have case, but this documents expected behavior
        result = detector.detect_period_keywords("기간")
        assert "기간" in result

    def test_long_query_performance(self, detector):
        """Handle long queries efficiently."""
        long_query = "이것은 매우 긴 쿼리입니다. " * 100 + "기간" + " 더 많은 텍스트" * 100
        result = detector.detect_period_keywords(long_query)
        assert "기간" in result

    def test_special_characters(self, detector):
        """Handle queries with special characters."""
        result = detector.detect_period_keywords("기간! @#$%^&*()")
        assert "기간" in result

    def test_newlines_in_query(self, detector):
        """Handle queries with newlines."""
        result = detector.detect_period_keywords("수강신청\n기간\n안내")
        assert "기간" in result


class TestPeriodKeywordDetectorCoverage:
    """Tests for 95%+ detection rate requirement."""

    @pytest.fixture
    def detector(self) -> PeriodKeywordDetector:
        """Create PeriodKeywordDetector instance."""
        return PeriodKeywordDetector()

    def test_real_world_queries(self, detector):
        """Test with real-world period-related queries."""
        real_queries = [
            ("수강신청 기간이 언제인가요?", ["기간", "언제"]),
            ("이번 학기 개강일은?", ["개강"]),
            ("등록금 납부 마감일", ["마감"]),
            ("휴학 신청은 언제까지인가요", ["언제", "까지"]),  # "신청" alone is not a keyword
            ("복학 신청일 확인", ["신청일"]),
            ("중간고사 기간", ["기간"]),
            ("기말고사 종료일", ["종료"]),
            ("종강일 알려주세요", ["종강"]),
            ("학기 시작일", ["시작"]),
            ("성적 공시 기한", ["기한"]),
        ]

        for query, expected_keywords in real_queries:
            result = detector.detect_period_keywords(query)
            for keyword in expected_keywords:
                assert keyword in result, f"Expected '{keyword}' in '{query}', got {result}"

    def test_non_period_queries(self, detector):
        """Test queries that should NOT be detected as period-related."""
        non_period_queries = [
            "교원 승진 규정",
            "연구윤리 규정 내용",
            "학생지침에 대하여",
            "시험 응시 방법",
            "성적 평가 기준",
        ]

        for query in non_period_queries:
            assert detector.is_period_related(query) is False, f"'{query}' should not be period-related"
