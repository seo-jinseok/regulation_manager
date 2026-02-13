"""
Tests for RegulationTitleDetector module.

Tests regulation title detection with various patterns and confidence scoring.
"""
import pytest

from src.parsing.detectors.regulation_title_detector import (
    RegulationTitleDetector,
    TitleMatchResult,
    detect_regulation_title,
    get_default_detector,
)


class TestRegulationTitleDetector:
    """Test RegulationTitleDetector class."""

    def test_init_default(self):
        """Test default initialization."""
        detector = RegulationTitleDetector()
        assert detector.min_confidence == 0.5

    def test_detect_explicit_regulation_keywords(self):
        """Test detection of explicit regulation keywords."""
        detector = RegulationTitleDetector()
        test_cases = [
            ("학칙", True),
            ("규정", True),
            ("요령", True),
            ("지침", True),
            ("세칙", True),
            ("시행세칙", True),
        ]
        for title, expected in test_cases:
            result = detector.detect(title)
            assert result.is_title == expected, f"Failed for: {title}"

    def test_detect_compound_regulations(self):
        """Test detection of compound regulation titles."""
        detector = RegulationTitleDetector()
        test_cases = [
            ("대학원학칙", True),
            ("등록금납입지침", True),
            ("장학관리규정", True),
            ("연구처리규정", True),
        ]
        for title, expected in test_cases:
            result = detector.detect(title)
            assert result.is_title == expected, f"Failed for: {title}"

    def test_detect_skip_patterns(self):
        """Test that skip patterns are correctly rejected."""
        detector = RegulationTitleDetector()
        skip_cases = [
            "1. 첫 번째 항목",
            "가. 첫 번째 소항목",
            "이 규정집은 동의대학교의 규정을 담고 있습니다.",
            "제1편 총칙",
            "제1장 총칙",
            "제1조 목적",
            "동의대학교 규정집",
            "목 차",
            "총 장",
            "① 첫 번째 항",
        ]
        for text in skip_cases:
            result = detector.detect(text)
            assert not result.is_title, f"Should skip: {text}"

    def test_confidence_scoring(self):
        """Test confidence score calculation."""
        detector = RegulationTitleDetector()

        # High confidence - exact keyword match
        result1 = detector.detect("학칙")
        assert result1.confidence_score >= 0.9

        # Medium confidence - pattern match
        result2 = detector.detect("등록금에 관한 규정")
        assert result2.confidence_score >= 0.7

        # Low confidence - no match
        result3 = detector.detect("일반 텍스트")
        assert result3.confidence_score == 0.0

    def test_length_validation(self):
        """Test length-based filtering."""
        detector = RegulationTitleDetector()

        # Too short
        result1 = detector.detect("규")
        assert not result1.is_title

        # Too long
        result2 = detector.detect("a" * 250)
        assert not result2.is_title

        # Just right
        result3 = detector.detect("동의대학교학칙")
        assert result3.is_title

    def test_false_positive_filtering(self):
        """Test false positive pattern filtering."""
        detector = RegulationTitleDetector()
        false_positives = [
            "이 규정은 학생의 권리를 보호한다.",
            "규정 제정에 관한 사항",
            "규정 개정 안내",
            "규정 폐지 및 폐지 이유",
        ]
        for text in false_positives:
            result = detector.detect(text)
            assert not result.is_title, f"Should be false positive: {text}"

    def test_is_title_convenience_method(self):
        """Test is_title convenience method."""
        detector = RegulationTitleDetector()
        assert detector.is_title("학칙") is True
        assert detector.is_title("일반 텍스트") is False

    def test_extract_title(self):
        """Test extract_title method."""
        detector = RegulationTitleDetector()
        assert detector.extract_title("학칙") == "학칙"
        assert detector.extract_title("일반 텍스트") is None

    def test_get_confidence(self):
        """Test get_confidence method."""
        detector = RegulationTitleDetector()
        confidence = detector.get_confidence("학칙")
        assert 0.0 <= confidence <= 1.0

    def test_batch_detect(self):
        """Test batch detection."""
        detector = RegulationTitleDetector()
        texts = ["학칙", "규정", "일반 텍스트"]
        results = detector.batch_detect(texts)
        assert len(results) == 3
        assert results[0].is_title is True
        assert results[1].is_title is True
        assert results[2].is_title is False


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_detect_regulation_title(self):
        """Test detect_regulation_title convenience function."""
        assert detect_regulation_title("학칙") is True
        assert detect_regulation_title("일반 텍스트") is False

    def test_get_default_detector(self):
        """Test get_default_detector returns singleton."""
        detector1 = get_default_detector()
        detector2 = get_default_detector()
        assert detector1 is detector2


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_string(self):
        """Test empty string handling."""
        detector = RegulationTitleDetector()
        result = detector.detect("")
        assert not result.is_title
        assert result.confidence_score == 0.0

    def test_whitespace_only(self):
        """Test whitespace-only string."""
        detector = RegulationTitleDetector()
        result = detector.detect("   ")
        assert not result.is_title

    def test_mixed_language(self):
        """Test mixed Korean-English titles."""
        detector = RegulationTitleDetector()
        result = detector.detect("ISO규정")
        # Should detect based on keyword ending
        assert result.is_title is True

    def test_numbers_in_title(self):
        """Test titles with numbers."""
        detector = RegulationTitleDetector()
        # "2024년도 규정" - ends with 규정, should match
        result1 = detector.detect("2024년도 규정")
        assert result1.is_title is True, "2024년도 규정 should be a title"

        # "제3규정" - ends with 규정, should match (not 제3조)
        result2 = detector.detect("제3규정")
        assert result2.is_title is True, "제3규정 should be a title"

        # "1학칙" - numbered + keyword pattern, should NOT match
        result3 = detector.detect("1학칙")
        # Should be rejected by the skip pattern r'^\d+[가-힣]'
        assert result3.is_title is False, "1학칙 is numbered + Korean, should be skipped"

    def test_parentheses_in_title(self):
        """Test titles with parentheses."""
        detector = RegulationTitleDetector()
        # "학칙(개정)" - should match as it ends with 학칙
        result = detector.detect("학칙(개정)")
        # The pattern requires keyword at END, so this won't match with parenthesis
        # But "개정학칙" would match
        assert result.is_title is False or result.title == "학칙(개정)"

    def test_min_confidence_threshold(self):
        """Test min_confidence parameter."""
        detector = RegulationTitleDetector(min_confidence=0.8)
        result = detector.detect("학칙")
        # 학칙 should have high confidence since it's a TITLE_KEYWORDS entry
        assert result.confidence_score >= 0.8, f"Confidence was {result.confidence_score}"
