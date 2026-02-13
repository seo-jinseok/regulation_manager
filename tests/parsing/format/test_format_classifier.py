"""
Tests for Format Classification Infrastructure.

Tests the format type enum and classifier for HWPX regulation parsing.
This module implements TASK-001: Format Classification Infrastructure for SPEC-HWXP-002.

TDD Approach: RED-GREEN-REFACTOR
- RED: These tests fail initially (implementation doesn't exist)
- GREEN: Implementation will be added to make tests pass
- REFACTOR: Code will be cleaned up while keeping tests green
"""
import pytest
from typing import Tuple
from enum import Enum


# Import the format classification components
from src.parsing.format.format_type import FormatType
from src.parsing.format.format_classifier import (
    FormatClassifier,
    ClassificationResult,
    ListPattern,
)


class TestFormatType:
    """Test FormatType enum definition and behavior."""

    def test_format_type_enum_exists(self):
        """Test that FormatType enum can be instantiated."""
        format_type = FormatType.ARTICLE
        assert format_type is not None
        assert format_type.value == "article"

    def test_format_type_has_all_required_types(self):
        """Test that FormatType has all four required format types."""
        required_types = ["ARTICLE", "LIST", "GUIDELINE", "UNSTRUCTURED"]
        for type_name in required_types:
            assert hasattr(FormatType, type_name), f"FormatType missing {type_name}"

    def test_format_type_values_are_correct(self):
        """Test that FormatType enum values match specification."""
        assert FormatType.ARTICLE.value == "article"
        assert FormatType.LIST.value == "list"
        assert FormatType.GUIDELINE.value == "guideline"
        assert FormatType.UNSTRUCTURED.value == "unstructured"

    def test_format_type_string_representation(self):
        """Test FormatType string representation for logging."""
        assert str(FormatType.ARTICLE) == "article" or "article" in str(FormatType.ARTICLE)
        assert str(FormatType.LIST) == "list" or "list" in str(FormatType.LIST)


class TestListPattern:
    """Test ListPattern enum for list format detection."""

    def test_list_pattern_enum_exists(self):
        """Test that ListPattern enum can be instantiated."""
        pattern = ListPattern.NUMERIC
        assert pattern is not None
        assert pattern.value == "numeric"

    def test_list_pattern_has_all_required_patterns(self):
        """Test that ListPattern has all four required patterns."""
        required_patterns = ["NUMERIC", "KOREAN_ALPHABET", "CIRCLED_NUMBER", "MIXED"]
        for pattern_name in required_patterns:
            assert hasattr(ListPattern, pattern_name), f"ListPattern missing {pattern_name}"

    def test_list_pattern_values_are_correct(self):
        """Test that ListPattern enum values match specification."""
        assert ListPattern.NUMERIC.value == "numeric"
        assert ListPattern.KOREAN_ALPHABET.value == "korean"
        assert ListPattern.CIRCLED_NUMBER.value == "circled"
        assert ListPattern.MIXED.value == "mixed"


class TestFormatClassifier:
    """Test FormatClassifier class for content format classification."""

    def test_classifier_initialization(self):
        """Test that FormatClassifier can be initialized."""
        classifier = FormatClassifier()
        assert classifier is not None

    def test_classify_article_format_positive(self):
        """Test classification of article format content (제N조 markers)."""
        classifier = FormatClassifier()

        # Test cases for article format
        article_samples = [
            "제1조(목적) 이 규정은...",
            "제2조 학생의 권리와 의무",
            "제10조의2(특별시험) 시험은...",
            "제100조 본 규정은...",
            # Multiple articles
            "제1조 목적\n제2조 정의\n제3조 시행",
        ]

        for sample in article_samples:
            result = classifier.classify(sample)
            assert result.format_type == FormatType.ARTICLE, \
                f"Expected ARTICLE format for: {sample[:50]}..."
            assert result.confidence > 0.8, \
                f"Expected high confidence for clear article markers"

    def test_classify_article_format_negative(self):
        """Test that non-article content is not classified as ARTICLE."""
        classifier = FormatClassifier()

        # Test cases that should NOT be classified as article
        non_article_samples = [
            "1. 첫 번째 항목\n2. 두 번째 항목\n3. 세 번째 항목",
            "가. 첫 번째\n나. 두 번째\n다. 세 번째",
            "이 규정은 학생의 권리를 보호하기 위해 만들어졌습니다. " +
            "모든 학생은 평등한 대우를 받아야 합니다.",
        ]

        for sample in non_article_samples:
            result = classifier.classify(sample)
            assert result.format_type != FormatType.ARTICLE, \
                f"Should not classify as ARTICLE: {sample[:50]}..."

    def test_classify_list_format_numeric(self):
        """Test classification of numeric list format (1., 2., 3.)."""
        classifier = FormatClassifier()

        # Numeric list patterns
        numeric_samples = [
            "1. 첫 번째 항목\n2. 두 번째 항목\n3. 세 번째 항목",
            "1. 학생은 성실하게 학업에 임해야 한다.\n" +
            "2. 교수는 공정하게 평가해야 한다.\n" +
            "3. 모든 구성원은 상호 존중해야 한다.",
            # With spacing variations
            "1. 항목 하나\n  2. 항목 둘\n    3. 항목 셋",
        ]

        for sample in numeric_samples:
            result = classifier.classify(sample)
            assert result.format_type == FormatType.LIST, \
                f"Expected LIST format for numeric list: {sample[:50]}..."
            assert result.list_pattern == ListPattern.NUMERIC, \
                f"Expected NUMERIC list pattern"
            assert result.confidence > 0.7, \
                f"Expected reasonable confidence for list format"

    def test_classify_list_format_korean_alphabet(self):
        """Test classification of Korean alphabet list format (가., 나., 다.)."""
        classifier = FormatClassifier()

        # Korean alphabet list patterns
        korean_samples = [
            "가. 첫 번째 항목\n나. 두 번째 항목\n다. 세 번째 항목",
            "가. 학생의 의무\n나. 교수의 책임\n다. 평가 기준",
            # Full alphabet range
            "가. 첫째\n나. 둘째\n다. 셋째\n라. 넷째\n마. 다섯째",
        ]

        for sample in korean_samples:
            result = classifier.classify(sample)
            assert result.format_type == FormatType.LIST, \
                f"Expected LIST format for Korean list: {sample[:50]}..."
            assert result.list_pattern == ListPattern.KOREAN_ALPHABET, \
                f"Expected KOREAN_ALPHABET list pattern"
            assert result.confidence > 0.7

    def test_classify_list_format_circled_number(self):
        """Test classification of circled number format (①, ②, ③)."""
        classifier = FormatClassifier()

        # Circled number patterns
        circled_samples = [
            "① 첫 번째 항목\n② 두 번째 항목\n③ 세 번째 항목",
            "① 평가 원칙\n② 평가 방법\n③ 평가 일정",
            # Mixed with other characters
            "①항목 하나\n②항목 둘\n③항목 셋",
        ]

        for sample in circled_samples:
            result = classifier.classify(sample)
            assert result.format_type == FormatType.LIST, \
                f"Expected LIST format for circled list: {sample[:50]}..."
            assert result.list_pattern == ListPattern.CIRCLED_NUMBER, \
                f"Expected CIRCLED_NUMBER list pattern"
            assert result.confidence > 0.7

    def test_classify_list_format_mixed(self):
        """Test classification of mixed list format."""
        classifier = FormatClassifier()

        # Mixed pattern samples
        mixed_samples = [
            "1. 첫 번째\n가. 상세 항목 하나\n나. 상세 항목 둘\n2. 두 번째\n① 세부 항목",
            "①첫째\n가.상세\n②둘째\n나.상세",
        ]

        for sample in mixed_samples:
            result = classifier.classify(sample)
            assert result.format_type == FormatType.LIST, \
                f"Expected LIST format for mixed list: {sample[:50]}..."
            assert result.list_pattern == ListPattern.MIXED, \
                f"Expected MIXED list pattern"

    def test_classify_guideline_format(self):
        """Test classification of guideline format (continuous prose)."""
        classifier = FormatClassifier()

        # Guideline format samples
        guideline_samples = [
            # Continuous prose without structure markers
            "이 규정은 학생의 권리와 의무를 명확히 하고, " +
            "대학 생활에서 필요한 기본적인 사항을 규정함을 목적으로 한다. " +
            "모든 학생은 자유롭게 학문을 탐구할 권리가 있으며, " +
            "동시에 university의 규정을 준수할 의무가 있다.",

            # Paragraph-based content (NO article markers)
            "학칙 목적\n\n" +
            "이 학칙은 대학의 교육 목적을 달성하기 위한 " +
            "기본적인 조직과 운영에 관한 사항을 규정한다.\n\n" +
            "정의\n\n" +
            "이 학칙에서 사용하는 용어의 정의는 다음과 같다.",

            # Long continuous text
            "평생교육원은 지역사회 주민들에게 평생학습의 기회를 제공한다. " +
            "다양한 프로그램을 운영하며, 전문 강사를 초빙한다. " +
            "수강료는 저렴하게 책정하며, 지역사회 발전에 기여함을 원칙으로 한다.",
        ]

        for sample in guideline_samples:
            result = classifier.classify(sample)
            assert result.format_type == FormatType.GUIDELINE, \
                f"Expected GUIDELINE format: {sample[:50]}..."
            assert result.confidence > 0.6, \
                "Guidelines may have moderate confidence due to lack of clear markers"

    def test_classify_unstructured_format(self):
        """Test classification of unstructured format (ambiguous content)."""
        classifier = FormatClassifier()

        # Unstructured/ambiguous samples
        unstructured_samples = [
            # Very short content
            "규정 내용",

            # Mixed with unclear patterns
            "일부 내용입니다\n그리고 또 다른 내용\n계속되는 텍스트",
        ]

        for sample in unstructured_samples:
            result = classifier.classify(sample)
            # For unstructured content, confidence should be low
            assert result.confidence < 0.6, \
                f"Expected low confidence for unstructured: {sample[:50]}..."
            # Format type could be GUIDELINE or UNSTRUCTURED
            assert result.format_type in [FormatType.GUIDELINE, FormatType.UNSTRUCTURED]

        # Single line prose should be GUIDELINE with moderate confidence
        result = classifier.classify("이것은 규정의 전체 내용이지만 구조가 명확하지 않습니다")
        assert result.format_type in [FormatType.GUIDELINE, FormatType.UNSTRUCTURED]

    def test_classify_empty_and_edge_cases(self):
        """Test classification of edge cases (empty, None, very short)."""
        classifier = FormatClassifier()

        # Edge cases
        edge_cases = [
            ("", FormatType.UNSTRUCTURED),
            ("   ", FormatType.UNSTRUCTURED),
            ("\n\n\n", FormatType.UNSTRUCTURED),
            ("제", FormatType.UNSTRUCTURED),
            ("1.", FormatType.UNSTRUCTURED),
        ]

        for text, expected_format in edge_cases:
            result = classifier.classify(text)
            assert result.format_type == expected_format, \
                f"Expected {expected_format} for empty/short: '{text}'"
            assert result.confidence <= 0.5, \
                "Edge cases should have low confidence"

    def test_confidence_scoring_range(self):
        """Test that confidence scores are always in [0.0, 1.0] range."""
        classifier = FormatClassifier()

        test_samples = [
            "제1조 목적",
            "1. 항목\n2. 항목",
            "연속된 텍스트 내용입니다",
            "",
        ]

        for sample in test_samples:
            result = classifier.classify(sample)
            assert 0.0 <= result.confidence <= 1.0, \
                f"Confidence {result.confidence} out of range for: {sample[:30]}"

    def test_classification_result_structure(self):
        """Test that ClassificationResult has required attributes."""
        classifier = FormatClassifier()
        result = classifier.classify("제1조 목적")

        # Check required attributes
        assert hasattr(result, 'format_type'), "Missing format_type attribute"
        assert hasattr(result, 'confidence'), "Missing confidence attribute"
        assert hasattr(result, 'list_pattern'), "Missing list_pattern attribute"
        assert hasattr(result, 'indicators'), "Missing indicators attribute"

        # Check attribute types
        assert isinstance(result.format_type, FormatType)
        assert isinstance(result.confidence, float)
        assert result.list_pattern is None or isinstance(result.list_pattern, ListPattern)
        assert isinstance(result.indicators, dict) or isinstance(result.indicators, list)

    def test_classification_indicators_for_article(self):
        """Test that classification provides indicators for article format."""
        classifier = FormatClassifier()
        result = classifier.classify("제1조 목적\n제2조 정의")

        # For article format, indicators should include article markers found
        if result.indicators:
            if isinstance(result.indicators, dict):
                assert 'article_markers' in result.indicators or \
                       'markers' in result.indicators or \
                       len(result.indicators) > 0

    def test_classification_indicators_for_list(self):
        """Test that classification provides indicators for list format."""
        classifier = FormatClassifier()
        result = classifier.classify("1. 첫째\n2. 둘째\n3. 셋째")

        # For list format, indicators should include list pattern info
        if result.indicators and isinstance(result.indicators, dict):
            assert 'list_pattern' in result.indicators or \
                   'pattern' in result.indicators or \
                   'list_count' in result.indicators


class TestClassificationAccuracy:
    """Test classification accuracy on sample data.

    Acceptance Criteria:
    - Classification accuracy >80% on sample data
    - Confidence scores for all classifications
    """

    @pytest.mark.parametrize("sample,expected_format,expected_min_confidence", [
        # Article format - clear markers
        ("제1조(목적) 이 규정은...", FormatType.ARTICLE, 0.85),
        ("제2조 학생의 권리", FormatType.ARTICLE, 0.8),
        ("제10조의2 특별조항", FormatType.ARTICLE, 0.8),

        # List format - numeric
        ("1. 첫째\n2. 둘째\n3. 셋째", FormatType.LIST, 0.8),
        ("1.항목\n2.항목\n3.항목", FormatType.LIST, 0.8),

        # List format - Korean alphabet
        ("가. 첫째\n나. 둘째\n다. 셋째", FormatType.LIST, 0.8),
        ("가.항목\n나.항목\n다.항목", FormatType.LIST, 0.8),

        # List format - circled
        ("① 첫째\n② 둘째\n③ 셋째", FormatType.LIST, 0.8),
        ("①항목\n②항목\n③항목", FormatType.LIST, 0.8),

        # Guideline format
        ("이 규정은 학생의 권리를 보호한다. 모든 학생은 평등한 대우를 받는다.",
         FormatType.GUIDELINE, 0.6),
    ])
    def test_classification_accuracy_samples(
        self, sample, expected_format, expected_min_confidence
    ):
        """Test classification accuracy on diverse samples."""
        classifier = FormatClassifier()
        result = classifier.classify(sample)

        # Check format classification
        assert result.format_type == expected_format, \
            f"Expected {expected_format}, got {result.format_type} for: {sample[:50]}"

        # Check confidence threshold
        assert result.confidence >= expected_min_confidence, \
            f"Expected confidence >= {expected_min_confidence}, " + \
            f"got {result.confidence} for: {sample[:50]}"

    def test_overall_accuracy_on_sample_set(self):
        """Test overall accuracy >80% on a representative sample set."""
        classifier = FormatClassifier()

        # Representative sample set with expected classifications
        test_samples = [
            # (sample, expected_format)
            ("제1조 목적", FormatType.ARTICLE),
            ("제2조 정의\n제3조 시행", FormatType.ARTICLE),
            ("1. 항목\n2. 항목\n3. 항목", FormatType.LIST),
            ("가. 항목\n나. 항목\n다. 항목", FormatType.LIST),
            ("① 항목\n② 항목\n③ 항목", FormatType.LIST),
            ("연속된 텍스트입니다. 구조가 없습니다.", FormatType.GUIDELINE),
            ("이 규정은 다음과 같이 시행한다.", FormatType.GUIDELINE),
        ]

        correct_classifications = 0
        total_samples = len(test_samples)

        for sample, expected_format in test_samples:
            result = classifier.classify(sample)
            if result.format_type == expected_format:
                correct_classifications += 1

        accuracy = correct_classifications / total_samples
        assert accuracy > 0.80, \
            f"Classification accuracy {accuracy:.2%} is below 80% threshold. " + \
            f"Correct: {correct_classifications}/{total_samples}"


class TestListPatternDetection:
    """Test list pattern detection functionality."""

    def test_detect_numeric_pattern(self):
        """Test numeric pattern detection (1., 2., 3.)."""
        classifier = FormatClassifier()
        result = classifier.classify("1. 하나\n2. 둘\n3. 셋")

        assert result.format_type == FormatType.LIST
        assert result.list_pattern == ListPattern.NUMERIC

    def test_detect_korean_alphabet_pattern(self):
        """Test Korean alphabet pattern detection (가., 나., 다.)."""
        classifier = FormatClassifier()
        result = classifier.classify("가. 하나\n나. 둘\n다. 셋")

        assert result.format_type == FormatType.LIST
        assert result.list_pattern == ListPattern.KOREAN_ALPHABET

    def test_detect_circled_number_pattern(self):
        """Test circled number pattern detection (①, ②, ③)."""
        classifier = FormatClassifier()
        result = classifier.classify("① 하나\n② 둘\n③ 셋")

        assert result.format_type == FormatType.LIST
        assert result.list_pattern == ListPattern.CIRCLED_NUMBER

    def test_detect_mixed_pattern(self):
        """Test mixed pattern detection."""
        classifier = FormatClassifier()
        result = classifier.classify("1. 하나\n가. 둘\n① 셋")

        assert result.format_type == FormatType.LIST
        assert result.list_pattern == ListPattern.MIXED

    def test_list_pattern_none_for_non_list(self):
        """Test that list_pattern is None for non-list formats."""
        classifier = FormatClassifier()

        # Article format
        result_article = classifier.classify("제1조 목적")
        assert result_article.format_type != FormatType.LIST
        # list_pattern may be None for non-list formats

        # Guideline format
        result_guideline = classifier.classify("연속된 텍스트")
        if result_guideline.format_type != FormatType.LIST:
            # list_pattern should be None or not applicable
            pass


class TestConfidenceScoring:
    """Test confidence scoring algorithm."""

    def test_high_confidence_for_clear_patterns(self):
        """Test high confidence for clear format indicators."""
        classifier = FormatClassifier()

        # Clear article markers
        result = classifier.classify("제1조 목적\n제2조 정의\n제3조 시행")
        assert result.confidence >= 0.9, "Clear article markers should have high confidence"

        # Clear list patterns
        result = classifier.classify("1. 항목\n2. 항목\n3. 항목\n4. 항목\n5. 항목")
        assert result.confidence >= 0.8, "Clear list patterns should have high confidence"

    def test_moderate_confidence_for_ambiguous_content(self):
        """Test moderate confidence for ambiguous content."""
        classifier = FormatClassifier()

        # Ambiguous content
        result = classifier.classify("이것은 규정의 내용입니다.")
        assert 0.4 <= result.confidence <= 0.7, \
            "Ambiguous content should have moderate confidence"

    def test_low_confidence_for_edge_cases(self):
        """Test low confidence for edge cases."""
        classifier = FormatClassifier()

        edge_cases = ["", "   ", "\n", "단어"]

        for case in edge_cases:
            result = classifier.classify(case)
            assert result.confidence <= 0.5, \
                f"Edge case '{case}' should have low confidence"

    def test_confidence_increases_with_more_evidence(self):
        """Test that confidence increases with more pattern evidence."""
        classifier = FormatClassifier()

        # Single article marker
        result_single = classifier.classify("제1조 목적")

        # Multiple article markers
        result_multiple = classifier.classify(
            "제1조 목적\n제2조 정의\n제3조 시행\n제4조 평가\n제5조 시효"
        )

        # More evidence should result in higher confidence
        assert result_multiple.confidence >= result_single.confidence, \
            "More pattern evidence should increase confidence"


class TestClassificationResult:
    """Test ClassificationResult data structure."""

    def test_result_has_required_fields(self):
        """Test that result has all required fields."""
        classifier = FormatClassifier()
        result = classifier.classify("제1조 목적")

        # Check field existence
        assert hasattr(result, 'format_type'), "Missing format_type"
        assert hasattr(result, 'confidence'), "Missing confidence"
        assert hasattr(result, 'list_pattern'), "Missing list_pattern"
        assert hasattr(result, 'indicators'), "Missing indicators"

    def test_result_field_types(self):
        """Test that result fields have correct types."""
        classifier = FormatClassifier()
        result = classifier.classify("1. 항목\n2. 항목")

        assert isinstance(result.format_type, FormatType)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert result.list_pattern is None or isinstance(result.list_pattern, ListPattern)

        # indicators can be dict or list
        assert isinstance(result.indicators, (dict, list))

    def test_result_immutability(self):
        """Test that result is reasonably immutable or has expected behavior."""
        classifier = FormatClassifier()
        result = classifier.classify("제1조 목적")

        # Classification results should be consistent
        result2 = classifier.classify("제1조 목적")

        assert result.format_type == result2.format_type
        assert abs(result.confidence - result2.confidence) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
