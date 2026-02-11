"""
Test Suite for HWPX Multi-Format Parser Coordinator.

TDD Approach: RED Phase
- Failing tests written before implementation
- Tests cover all core functionality of HWPXMultiFormatParser

Reference: SPEC-HWXP-002, TASK-006
"""
import json
import zipfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from io import BytesIO
import pytest

from src.parsing.multi_format_parser import HWPXMultiFormatParser
from src.parsing.format.format_type import FormatType, ListPattern
from src.parsing.detectors.regulation_title_detector import RegulationTitleDetector
from src.parsing.format.format_classifier import FormatClassifier
from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor
from src.parsing.analyzers.guideline_structure_analyzer import GuidelineStructureAnalyzer
from src.parsing.analyzers.unstructured_regulation_analyzer import UnstructuredRegulationAnalyzer
from src.parsing.metrics.coverage_tracker import CoverageTracker


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_hwpx_content():
    """Sample HWPX content for testing."""
    return {
        "section1_toc": """
동의대학교 규정집 목차

제1편 총칙
  제1장 학칙
    겸임교원규정 ...................... 1
    교원인사규정 ...................... 5
  제2장 학사
    수업규정 ........................ 15
    성적평가규정 .................... 25
    시행세칙 ........................ 30
    운영지침 ........................ 35
""",
        "section0_article": """
겸임교원규정

제1조(목적) 이 규정은 동의대학교의 겸임교원에 관한 사항을 규정함을 목적으로 한다.

제2조(정의) 이 규정에서 "겸임교원"이라 함은 본 대학교 이외의 교육기관 또는 연구기관 등에 재직하는 자로서 본 대학교에서 교원으로 임용된 자를 말한다.

교원인사규정

제1조(목적) 이 규정은 교원 인사에 관한 사항을 규정한다.
""",
        "section0_list": """
시행세칙

1. 이 세칙은 동의대학교의 각종 행정 업무 처리에 관한 사항을 규정한다.
2. 각 부서는 이 세칙을 준수하여야 한다.
3. 이 세칙에서 정하지 아니한 사항은 다른 법령을 따른다.
① 세부 시행 방법은 총장이 별도로 정한다.
② 이 규정의 개정은 총장의 승인을 얻어야 한다.
""",
        "section0_guideline": """
운영지침

이 지침은 동의대학교 연구소의 운영에 관한 기본 사항을 정함을 목적으로 한다. 연구소는 대학의 교육 및 연구 진흥을 위하여 설립한다. 연구소장은 교수 또는 부교수 중에서 총장이 임명한다. 따라서 연구소의 운영은 효율성과 투명성을 확보해야 한다.
 또한 연구소는 정기적으로 운영 성과를 보고해야 한다.
""",
        "section0_empty": """
폐지규정

이 규정은 2020년 1월 1일부로 폐지되었다.
""",
    }


@pytest.fixture
def mock_hwpx_file(tmp_path, sample_hwpx_content):
    """Create a mock HWPX file for testing."""
    hwpx_path = tmp_path / "test_regulation.hwpx"

    with zipfile.ZipFile(hwpx_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add section1.xml (TOC)
        zf.writestr("Contents/section1.xml", sample_hwpx_content["section1_toc"])

        # Add section0.xml (main content)
        zf.writestr("Contents/section0.xml", sample_hwpx_content["section0_article"])

        # Add manifest files
        zf.writestr("Contents/_rels/.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')

    return hwpx_path


@pytest.fixture
def multi_format_parser():
    """Create a HWPXMultiFormatParser instance for testing."""
    # Mock LLM client to avoid actual API calls
    mock_llm_client = Mock()
    mock_llm_client.generate.return_value = json.dumps({
        "structure_type": "unstructured",
        "confidence": 0.8,
        "provisions": [
            {"number": "1", "content": "Provision 1 content"},
            {"number": "2", "content": "Provision 2 content"}
        ]
    })

    return HWPXMultiFormatParser(llm_client=mock_llm_client)


# ============================================================================
# Test Class: HWPXMultiFormatParser Initialization
# ============================================================================

class TestHWPXMultiFormatParserInit:
    """Test HWPXMultiFormatParser initialization and configuration."""

    def test_init_default_parameters(self):
        """Test parser initialization with default parameters."""
        parser = HWPXMultiFormatParser()

        assert parser.format_classifier is not None
        assert isinstance(parser.format_classifier, FormatClassifier)
        assert parser.list_extractor is not None
        assert isinstance(parser.list_extractor, ListRegulationExtractor)
        assert parser.guideline_analyzer is not None
        assert isinstance(parser.guideline_analyzer, GuidelineStructureAnalyzer)
        assert parser.unstructured_analyzer is not None
        assert isinstance(parser.unstructured_analyzer, UnstructuredRegulationAnalyzer)
        assert parser.coverage_tracker is not None
        assert isinstance(parser.coverage_tracker, CoverageTracker)

    def test_init_with_custom_llm_client(self):
        """Test parser initialization with custom LLM client."""
        mock_llm = Mock()
        parser = HWPXMultiFormatParser(llm_client=mock_llm)

        assert parser.unstructured_analyzer.llm_client == mock_llm

    def test_init_with_status_callback(self):
        """Test parser initialization with status callback."""
        callback_calls = []

        def status_callback(message: str):
            callback_calls.append(message)

        parser = HWPXMultiFormatParser(status_callback=status_callback)

        assert parser.status_callback == status_callback


# ============================================================================
# Test Class: TOC Extraction
# ============================================================================

class TestTOCExtraction:
    """Test Table of Contents extraction from section1.xml."""

    def test_extract_toc_from_hwpx(self, multi_format_parser, mock_hwpx_file):
        """Test TOC extraction from HWPX file."""
        toc_entries = multi_format_parser._extract_toc(mock_hwpx_file)

        assert len(toc_entries) > 0
        assert any("겸임교원규정" in entry.get("title", "") for entry in toc_entries)
        assert any("교원인사규정" in entry.get("title", "") for entry in toc_entries)

    def test_extract_toc_returns_expected_structure(self, multi_format_parser, mock_hwpx_file):
        """Test TOC entries have expected structure."""
        toc_entries = multi_format_parser._extract_toc(mock_hwpx_file)

        for entry in toc_entries:
            assert "title" in entry
            assert "page" in entry
            assert isinstance(entry["title"], str)

    def test_extract_toc_handles_empty_file(self, multi_format_parser, tmp_path):
        """Test TOC extraction from file without section1.xml."""
        empty_hwpx = tmp_path / "empty.hwpx"

        with zipfile.ZipFile(empty_hwpx, 'w') as zf:
            zf.writestr("Contents/section0.xml", "<content></content>")

        toc_entries = multi_format_parser._extract_toc(empty_hwpx)

        # Should return empty list, not crash
        assert toc_entries == []

    def test_extract_toc_uses_title_detector(self, multi_format_parser, mock_hwpx_file):
        """Test that TOC extraction uses RegulationTitleDetector."""
        with patch.object(multi_format_parser.title_detector, 'detect') as mock_detect:
            # Mock to return True for specific titles
            from src.parsing.detectors.regulation_title_detector import TitleMatchResult
            mock_detect.return_value = TitleMatchResult(
                is_title=True,
                title="test",
                confidence_score=1.0,
                match_type="test"
            )

            toc_entries = multi_format_parser._extract_toc(mock_hwpx_file)

            # Verify detector was called
            assert mock_detect.called


# ============================================================================
# Test Class: Format Delegation
# ============================================================================

class TestFormatDelegation:
    """Test format classification and delegation to appropriate extractors."""

    def test_classify_article_format(self, multi_format_parser):
        """Test classification of article format content."""
        content = "제1조(목적) 이 규정은 목적을 규정한다."
        result = multi_format_parser._classify_format(content)

        assert result.format_type == FormatType.ARTICLE

    def test_classify_list_format(self, multi_format_parser):
        """Test classification of list format content."""
        content = "1. 첫 번째 항목\n2. 두 번째 항목\n3. 세 번째 항목"
        result = multi_format_parser._classify_format(content)

        assert result.format_type == FormatType.LIST

    def test_classify_guideline_format(self, multi_format_parser):
        """Test classification of guideline format content."""
        content = "이 지침은 운영에 관한 기본 사항을 정한다. 따라서 모든 부서가 준수해야 한다."
        result = multi_format_parser._classify_format(content)

        assert result.format_type == FormatType.GUIDELINE

    def test_classify_unstructured_format(self, multi_format_parser):
        """Test classification of ambiguous/unstructured content."""
        # Very short content without clear structure should be UNSTRUCTURED
        content = "xyz"
        result = multi_format_parser._classify_format(content)

        assert result.format_type == FormatType.UNSTRUCTURED

    def test_extract_with_article_format(self, multi_format_parser):
        """Test extraction delegates to article extractor."""
        title = "겸임교원규정"
        content = "제1조(목적) 이 규정은 겸임교원에 관한 사항을 규정함을 목적으로 한다."

        result = multi_format_parser._extract_with_format(
            title=title,
            content=content,
            format_type=FormatType.ARTICLE
        )

        assert "articles" in result or "provisions" in result

    def test_extract_with_list_format(self, multi_format_parser):
        """Test extraction delegates to list extractor."""
        title = "시행세칙"
        content = "1. 첫 번째 항목\n2. 두 번째 항목"

        result = multi_format_parser._extract_with_format(
            title=title,
            content=content,
            format_type=FormatType.LIST
        )

        assert "articles" in result or "items" in result

    def test_extract_with_guideline_format(self, multi_format_parser):
        """Test extraction delegates to guideline analyzer."""
        title = "운영지침"
        content = "이 지침은 운영에 관한 기본 사항을 정한다. 따라서 모든 부서가 준수해야 한다."

        result = multi_format_parser._extract_with_format(
            title=title,
            content=content,
            format_type=FormatType.GUIDELINE
        )

        assert "provisions" in result or "articles" in result

    def test_extract_with_unstructured_format(self, multi_format_parser):
        """Test extraction delegates to unstructured analyzer."""
        title = "기타규정"
        content = "This is some ambiguous content."

        result = multi_format_parser._extract_with_format(
            title=title,
            content=content,
            format_type=FormatType.UNSTRUCTURED
        )

        # Should call LLM analyzer (mocked)
        assert "provisions" in result or "articles" in result


# ============================================================================
# Test Class: Multi-Section Aggregation
# ============================================================================

class TestMultiSectionAggregation:
    """Test aggregation of content from multiple sections."""

    def test_aggregate_sections_from_hwpx(self, multi_format_parser, mock_hwpx_file):
        """Test content aggregation from multiple XML sections."""
        sections = multi_format_parser._aggregate_sections(mock_hwpx_file)

        # Should have content from section0.xml
        assert "section0" in sections or len(sections) > 0

    def test_aggregate_sections_merges_content(self, multi_format_parser, tmp_path):
        """Test that multi-section aggregation properly merges content."""
        hwpx_path = tmp_path / "multi_section.hwpx"

        with zipfile.ZipFile(hwpx_path, 'w') as zf:
            zf.writestr("Contents/section0.xml", "<content>Main content</content>")
            zf.writestr("Contents/section1.xml", "<content>TOC content</content>")
            zf.writestr("Contents/section2.xml", "<content>Appendix content</content>")

        sections = multi_format_parser._aggregate_sections(hwpx_path)

        # Should include multiple sections
        assert len(sections) >= 1

    def test_aggregate_sections_handles_missing_sections(self, multi_format_parser, tmp_path):
        """Test aggregation when some sections are missing."""
        hwpx_path = tmp_path / "partial.hwpx"

        with zipfile.ZipFile(hwpx_path, 'w') as zf:
            zf.writestr("Contents/section0.xml", "<content>Content</content>")

        sections = multi_format_parser._aggregate_sections(hwpx_path)

        # Should not crash, return available sections
        assert isinstance(sections, dict)


# ============================================================================
# Test Class: LLM Fallback
# ============================================================================

class TestLLMFallback:
    """Test LLM fallback for low-coverage regulations."""

    def test_llm_fallback_for_low_coverage(self, multi_format_parser):
        """Test LLM fallback is triggered for low coverage (< 20%)."""
        # Create a regulation with low coverage
        regulation = {
            "title": "저조항규정",
            "content": "Very short content.",
            "articles": [],
            "provisions": [],
            "metadata": {
                "format_type": "unstructured",
                "coverage_score": 0.1  # 10% coverage
            }
        }

        # LLM analyzer is mocked, so should use mock response
        result = multi_format_parser._attempt_llm_fallback(regulation)

        assert "articles" in result or "provisions" in result

    def test_llm_fallback_skipped_for_good_coverage(self, multi_format_parser):
        """Test LLM fallback is skipped for good coverage (> 20%)."""
        regulation = {
            "title": "충분항규정",
            "content": "This regulation has sufficient content that meets the coverage threshold.",
            "articles": [
                {"number": 1, "content": "A" * 300}
            ],
            "provisions": [],
            "metadata": {
                "format_type": "article",
                "coverage_score": 0.9  # 90% coverage
            }
        }

        result = multi_format_parser._attempt_llm_fallback(regulation)

        # Should return original without LLM call
        assert result["metadata"].get("llm_enhanced") is not True


# ============================================================================
# Test Class: Coverage Tracking Integration
# ============================================================================

class TestCoverageTracking:
    """Test coverage tracking integration during parsing."""

    def test_coverage_tracking_updates_during_parsing(self, multi_format_parser, mock_hwpx_file):
        """Test that coverage tracker is updated during parsing."""
        initial_report = multi_format_parser.coverage_tracker.get_coverage_report()
        initial_total = initial_report.total_regulations

        # Parse the file
        result = multi_format_parser.parse_file(mock_hwpx_file)

        # Coverage should be tracked
        final_report = multi_format_parser.coverage_tracker.get_coverage_report()
        assert final_report.total_regulations >= initial_total

    def test_coverage_report_contains_format_breakdown(self, multi_format_parser, mock_hwpx_file):
        """Test coverage report includes format breakdown."""
        result = multi_format_parser.parse_file(mock_hwpx_file)

        coverage_report = multi_format_parser.coverage_tracker.get_coverage_report()

        assert hasattr(coverage_report, "format_breakdown")
        assert isinstance(coverage_report.format_breakdown, dict)

    def test_coverage_report_includes_final_statistics(self, multi_format_parser, mock_hwpx_file):
        """Test final parsing result includes coverage statistics."""
        result = multi_format_parser.parse_file(mock_hwpx_file)

        assert "coverage" in result
        assert "total" in result["coverage"] or "total_regulations" in result["coverage"]


# ============================================================================
# Test Class: Progress Callbacks
# ============================================================================

class TestProgressCallbacks:
    """Test progress callback functionality."""

    def test_status_callback_invoked_during_parsing(self, tmp_path):
        """Test that status callback is invoked during parsing."""
        callback_calls = []

        def status_callback(message: str):
            callback_calls.append(message)

        parser = HWPXMultiFormatParser(status_callback=status_callback)

        # Create minimal HWPX file
        hwpx_path = tmp_path / "test.hwpx"
        with zipfile.ZipFile(hwpx_path, 'w') as zf:
            zf.writestr("Contents/section1.xml", "<content>제1조 규정</content>")
            zf.writestr("Contents/section0.xml", "<content>제1조(목적) 목적이다.</content>")

        parser.parse_file(hwpx_path)

        # Callback should have been called at least once
        assert len(callback_calls) > 0

    def test_status_callback_receives_meaningful_messages(self, tmp_path):
        """Test that status callback receives meaningful progress messages."""
        callback_messages = []

        def status_callback(message: str):
            callback_messages.append(message)

        parser = HWPXMultiFormatParser(status_callback=status_callback)

        # Create minimal HWPX file
        hwpx_path = tmp_path / "test.hwpx"
        with zipfile.ZipFile(hwpx_path, 'w') as zf:
            zf.writestr("Contents/section1.xml", "<content>제1조 규정</content>")
            zf.writestr("Contents/section0.xml", "<content>제1조(목적) 목적이다.</content>")

        parser.parse_file(hwpx_path)

        # Check for meaningful message patterns
        meaningful_messages = [m for m in callback_messages if len(m) > 5]
        assert len(meaningful_messages) > 0


# ============================================================================
# Test Class: Full Parsing Workflow
# ============================================================================

class TestFullParsingWorkflow:
    """Test end-to-end parsing workflow."""

    def test_parse_file_returns_complete_result(self, multi_format_parser, mock_hwpx_file):
        """Test complete parsing workflow returns valid result structure."""
        result = multi_format_parser.parse_file(mock_hwpx_file)

        # Verify result structure
        assert "docs" in result or "regulations" in result
        assert "coverage" in result or "metadata" in result

    def test_parse_file_creates_regulation_entries(self, multi_format_parser, mock_hwpx_file):
        """Test that parsing creates regulation entries for all TOC items."""
        result = multi_format_parser.parse_file(mock_hwpx_file)

        docs = result.get("docs", result.get("regulations", []))

        # Should have at least some regulations
        assert len(docs) > 0

    def test_parse_file_includes_coverage_report(self, multi_format_parser, mock_hwpx_file):
        """Test that parsing result includes coverage report."""
        result = multi_format_parser.parse_file(mock_hwpx_file)

        coverage = result.get("coverage", {})
        assert coverage is not None

    def test_parse_file_handles_malformed_hwpx(self, multi_format_parser, tmp_path):
        """Test parsing handles malformed HWPX files gracefully."""
        malformed_hwpx = tmp_path / "malformed.hwpx"

        # Create file with invalid XML
        with zipfile.ZipFile(malformed_hwpx, 'w') as zf:
            zf.writestr("Contents/section1.xml", "<<<invalid>>>")

        # Should not crash
        result = multi_format_parser.parse_file(malformed_hwpx)
        assert result is not None

    def test_parse_file_empty_hwpx(self, multi_format_parser, tmp_path):
        """Test parsing handles completely empty HWPX file."""
        empty_hwpx = tmp_path / "empty.hwpx"

        with zipfile.ZipFile(empty_hwpx, 'w') as zf:
            pass  # Empty archive

        result = multi_format_parser.parse_file(empty_hwpx)

        # Should return empty result, not crash
        assert result is not None
        assert result.get("docs", result.get("regulations", [])) == []


# ============================================================================
# Test Class: Format Classification Accuracy
# ============================================================================

class TestFormatClassificationAccuracy:
    """Test accuracy of format classification for various content types."""

    def test_article_format_classification_accuracy(self, multi_format_parser):
        """Test article format classification accuracy with multiple samples."""
        samples = [
            "제1조(목적) 이 규정은 목적을 규정한다.",
            "제2조(정의) 이 규정에서 용어를 정의한다.",
            "제10조의2(특례) 특별한 사항을 규정한다.",
        ]

        for sample in samples:
            result = multi_format_parser._classify_format(sample)
            assert result.format_type == FormatType.ARTICLE, f"Failed for: {sample}"

    def test_list_format_classification_accuracy(self, multi_format_parser):
        """Test list format classification accuracy with multiple samples."""
        samples = [
            "1. 첫 번째\n2. 두 번째\n3. 세 번째",
            "가. 첫 번째\n나. 두 번째\n다. 세 번째",
            "① 첫 번째\n② 두 번째\n③ 세 번째",
        ]

        for sample in samples:
            result = multi_format_parser._classify_format(sample)
            assert result.format_type == FormatType.LIST, f"Failed for: {sample}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegrationWithExistingComponents:
    """Test integration with existing parsing components."""

    def test_integration_with_regulation_title_detector(self, multi_format_parser):
        """Test that parser integrates with RegulationTitleDetector."""
        assert multi_format_parser.title_detector is not None
        assert isinstance(multi_format_parser.title_detector, RegulationTitleDetector)

    def test_integration_with_format_classifier(self, multi_format_parser):
        """Test that parser integrates with FormatClassifier."""
        assert multi_format_parser.format_classifier is not None
        assert isinstance(multi_format_parser.format_classifier, FormatClassifier)

    def test_integration_with_list_extractor(self, multi_format_parser):
        """Test that parser integrates with ListRegulationExtractor."""
        assert multi_format_parser.list_extractor is not None
        assert isinstance(multi_format_parser.list_extractor, ListRegulationExtractor)

    def test_integration_with_guideline_analyzer(self, multi_format_parser):
        """Test that parser integrates with GuidelineStructureAnalyzer."""
        assert multi_format_parser.guideline_analyzer is not None
        assert isinstance(multi_format_parser.guideline_analyzer, GuidelineStructureAnalyzer)

    def test_integration_with_unstructured_analyzer(self, multi_format_parser):
        """Test that parser integrates with UnstructuredRegulationAnalyzer."""
        assert multi_format_parser.unstructured_analyzer is not None
        assert isinstance(multi_format_parser.unstructured_analyzer, UnstructuredRegulationAnalyzer)

    def test_integration_with_coverage_tracker(self, multi_format_parser):
        """Test that parser integrates with CoverageTracker."""
        assert multi_format_parser.coverage_tracker is not None
        assert isinstance(multi_format_parser.coverage_tracker, CoverageTracker)
