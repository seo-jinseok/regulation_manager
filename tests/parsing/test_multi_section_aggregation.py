"""
Test Suite for TASK-007: Multi-Section Aggregation

Tests for enhanced multi-section content aggregation with:
- Duplicate elimination
- TOC completeness validation
- Edge case handling

Reference: SPEC-HWXP-002, TASK-007
TDD Approach: RED Phase - Tests written before implementation
"""
import zipfile
from pathlib import Path
from unittest.mock import Mock
import pytest

from src.parsing.multi_format_parser import HWPXMultiFormatParser


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def multi_section_hwpx(tmp_path):
    """Create HWPX file with multiple sections for testing."""
    hwpx_path = tmp_path / "multi_section.hwpx"

    with zipfile.ZipFile(hwpx_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # section0.xml - Main content
        zf.writestr("Contents/section0.xml", """
겸임교원규정

제1조(목적) 이 규정은 동의대학교의 겸임교원에 관한 사항을 규정함을 목적으로 한다.

제2조(정의) 이 규정에서 "겸임교원"이라 함은 본 대학교 이외의 교육기관 또는 연구기관 등에 재직하는 자로서 본 대학교에서 교원으로 임용된 자를 말한다.

교원인사규정

제1조(목적) 이 규정은 교원 인사에 관한 사항을 규정한다.
""")

        # section1.xml - TOC
        zf.writestr("Contents/section1.xml", """
동의대학교 규정집 목차

제1편 총칙
  제1장 학칙
    겸임교원규정 ...................... 1
    교원인사규정 ...................... 5
  제2장 학사
    수업규정 ........................ 15
    성적평가규정 .................... 25
""")

        # section2.xml - Additional content (appendices)
        zf.writestr("Contents/section2.xml", """
부록

별표1 서식
별표2 예규
""")

        # section3.xml - More additional content
        zf.writestr("Contents/section3.xml", """
연혁

2020년 1월 1일: 규정 제정
2024년 3월 1일: 규정 개정
""")

    return hwpx_path


@pytest.fixture
def duplicate_content_hwpx(tmp_path):
    """Create HWPX file with duplicate content across sections."""
    hwpx_path = tmp_path / "duplicate.hwpx"

    with zipfile.ZipFile(hwpx_path, 'w') as zf:
        # section0.xml
        zf.writestr("Contents/section0.xml", """
겸임교원규정

제1조(목적) 이 규정은 목적을 규정한다.
제2조(정의) 이 규정에서 정의를 규정한다.
""")

        # section2.xml - Contains duplicate content
        zf.writestr("Contents/section2.xml", """
겸임교원규정

제1조(목적) 이 규정은 목적을 규정한다.
제2조(정의) 이 규정에서 정의를 규정한다.
""")

    return hwpx_path


@pytest.fixture
def incomplete_toc_hwpx(tmp_path):
    """Create HWPX file where TOC lists regulations not in content."""
    hwpx_path = tmp_path / "incomplete.hwpx"

    with zipfile.ZipFile(hwpx_path, 'w') as zf:
        # TOC lists 4 regulations
        zf.writestr("Contents/section1.xml", """
동의대학교 규정집 목차

제1편 총칙
    겸임교원규정 ...................... 1
    교원인사규정 ...................... 5
    수업규정 ........................ 15
    성적평가규정 .................... 25
""")

        # Content only has 2 regulations
        zf.writestr("Contents/section0.xml", """
겸임교원규정

제1조(목적) 이 규정은 목적을 규정한다.

교원인사규정

제1조(목적) 이 규정은 교원 인사에 관한 사항을 규정한다.
""")

    return hwpx_path


@pytest.fixture
def parser():
    """Create HWPXMultiFormatParser instance for testing."""
    mock_llm = Mock()
    return HWPXMultiFormatParser(llm_client=mock_llm)


# ============================================================================
# Test Class: Multi-Section Aggregation
# ============================================================================

class TestMultiSectionAggregation:
    """Test aggregation of content from multiple sections."""

    def test_aggregate_sections_collects_all_sections(self, parser, multi_section_hwpx):
        """Test that aggregation collects content from all sections."""
        sections = parser._aggregate_sections(multi_section_hwpx)

        # Should have collected all 4 sections
        assert len(sections) == 4
        assert "Contents/section0.xml" in sections
        assert "Contents/section1.xml" in sections
        assert "Contents/section2.xml" in sections
        assert "Contents/section3.xml" in sections

    def test_aggregate_sections_preserves_content_order(self, parser, multi_section_hwpx):
        """Test that aggregation maintains consistent section processing order."""
        sections = parser._aggregate_sections(multi_section_hwpx)

        # Check that sections are sorted properly
        section_names = list(sections.keys())
        assert section_names[0] == "Contents/section0.xml" or "section0" in section_names[0]

    def test_aggregate_sections_logs_section_info(self, parser, multi_section_hwpx, caplog):
        """Test that aggregation logs section count and sizes."""
        import logging
        caplog.set_level(logging.DEBUG)

        sections = parser._aggregate_sections(multi_section_hwpx)

        # Check for log messages
        assert any("sections" in record.message.lower() for record in caplog.records)

    def test_aggregate_sections_handles_empty_sections(self, parser, tmp_path):
        """Test aggregation with empty section content."""
        hwpx_path = tmp_path / "empty_sections.hwpx"

        with zipfile.ZipFile(hwpx_path, 'w') as zf:
            zf.writestr("Contents/section0.xml", "")
            zf.writestr("Contents/section1.xml", "")

        sections = parser._aggregate_sections(hwpx_path)

        # Should still return dictionary with empty content
        assert isinstance(sections, dict)
        assert len(sections) == 2

    def test_aggregate_sections_handles_corrupted_sections(self, parser, tmp_path):
        """Test aggregation handles unreadable sections gracefully."""
        hwpx_path = tmp_path / "corrupt.hwpx"

        with zipfile.ZipFile(hwpx_path, 'w') as zf:
            zf.writestr("Contents/section0.xml", "Valid content")
            # Create a file that will cause issues (empty name)
            zf.writestr("Contents/section", "")

        # Should not crash
        sections = parser._aggregate_sections(hwpx_path)
        assert isinstance(sections, dict)


# ============================================================================
# Test Class: Content Merging and Duplicate Elimination
# ============================================================================

class TestContentMerging:
    """Test content merging with duplicate elimination."""

    def test_merge_section_contents_removes_duplicates(self, parser):
        """Test that merging removes duplicate content blocks."""
        sections = {
            "Contents/section0.xml": "Line 1\nLine 2\nLine 3",
            "Contents/section2.xml": "Line 1\nLine 2\nLine 4"
        }

        merged = parser._merge_section_contents(sections)

        # Should have 4 unique lines (Line 1, 2 from first section, 3, 4 from both)
        lines = merged.split('\n')
        assert len(lines) == 4
        assert lines.count("Line 1") == 1  # Only one occurrence
        assert lines.count("Line 2") == 1

    def test_merge_section_contents_preserves_order(self, parser):
        """Test that merging preserves content order."""
        sections = {
            "Contents/section0.xml": "First\nSecond",
            "Contents/section1.xml": "Third\nFourth"
        }

        merged = parser._merge_section_contents(sections)

        # section0 should come before section1
        lines = merged.split('\n')
        first_index = lines.index("First")
        third_index = lines.index("Third")
        assert first_index < third_index

    def test_merge_section_contents_handles_empty_sections(self, parser):
        """Test merging with empty section content."""
        sections = {
            "Contents/section0.xml": "Content 1",
            "Contents/section1.xml": ""
        }

        merged = parser._merge_section_contents(sections)

        # Should skip empty lines
        assert "Content 1" in merged
        assert merged.count('\n') <= 1

    def test_merge_section_contents_returns_empty_for_no_sections(self, parser):
        """Test merging with no sections."""
        merged = parser._merge_section_contents({})
        assert merged == ""

    def test_merge_section_contents_normalizes_whitespace(self, parser):
        """Test that merging normalizes whitespace for duplicate detection."""
        sections = {
            "Contents/section0.xml": "  Line 1  \nLine 2",
            "Contents/section2.xml": "Line 1\nLine 2"
        }

        merged = parser._merge_section_contents(sections)

        # Should detect normalized versions as duplicates
        lines = merged.split('\n')
        # After stripping, "  Line 1  " and "Line 1" are the same
        assert len([l for l in lines if l.strip() == "Line 1"]) == 1


# ============================================================================
# Test Class: TOC Completeness Validation
# ============================================================================

class TestTOCCompleteness:
    """Test TOC completeness validation."""

    def test_validate_toc_completeness_all_found(self, parser):
        """Test validation when all TOC entries have content."""
        toc_entries = [
            {"title": "규정1"},
            {"title": "규정2"}
        ]
        sections = {
            "Contents/section0.xml": "규정1\n제1조 내용\n규정2\n제1조 내용"
        }

        is_complete, missing = parser._validate_toc_completeness(toc_entries, sections)

        assert is_complete is True
        assert len(missing) == 0

    def test_validate_toc_completeness_with_missing(self, parser, incomplete_toc_hwpx):
        """Test validation detects missing regulations."""
        # Parse file to get TOC and sections
        toc_entries = parser._extract_toc(incomplete_toc_hwpx)
        sections = parser._aggregate_sections(incomplete_toc_hwpx)

        is_complete, missing = parser._validate_toc_completeness(toc_entries, sections)

        assert is_complete is False
        assert len(missing) > 0
        assert "수업규정" in missing or "성적평가규정" in missing

    def test_validate_toc_completeness_returns_missing_titles(self, parser):
        """Test validation returns list of missing titles."""
        toc_entries = [
            {"title": "규정1"},
            {"title": "규정2"},
            {"title": "규정3"}
        ]
        sections = {
            "Contents/section0.xml": "규정1\n제1조 내용"
        }

        is_complete, missing = parser._validate_toc_completeness(toc_entries, sections)

        assert is_complete is False
        assert "규정2" in missing
        assert "규정3" in missing
        assert len(missing) == 2

    def test_validate_toc_completeness_handles_empty_toc(self, parser):
        """Test validation with empty TOC."""
        is_complete, missing = parser._validate_toc_completeness([], {})

        assert is_complete is True
        assert len(missing) == 0

    def test_validate_toc_completeness_logs_completeness_rate(self, parser, caplog):
        """Test validation logs completeness percentage."""
        import logging
        caplog.set_level(logging.INFO)

        toc_entries = [
            {"title": "규정1"},
            {"title": "규정2"},
            {"title": "규정3"},
            {"title": "규정4"}
        ]
        sections = {
            "Contents/section0.xml": "규정1\n규정2"
        }

        parser._validate_toc_completeness(toc_entries, sections)

        # Check for completeness rate in logs
        log_messages = [record.message for record in caplog.records]
        assert any("50.0%" in msg or "50%" in msg for msg in log_messages)


# ============================================================================
# Test Class: Enhanced Content Finding
# ============================================================================

class TestEnhancedContentFinding:
    """Test enhanced content finding across multiple sections."""

    def test_find_content_prioritizes_section0(self, parser, multi_section_hwpx):
        """Test content finding prioritizes section0.xml."""
        sections = parser._aggregate_sections(multi_section_hwpx)

        # Should find content in section0 first
        content = parser._find_content_for_title("겸임교원규정", sections)

        assert "제1조" in content
        assert len(content) > 0

    def test_find_content_searches_multiple_sections(self, parser, multi_section_hwpx):
        """Test content finding searches across all sections."""
        sections = parser._aggregate_sections(multi_section_hwpx)

        # Should find content even if it's in later sections
        content = parser._find_content_for_title("부록", sections)

        # "부록" is in section2.xml
        assert content is not None

    def test_find_content_returns_empty_for_missing(self, parser):
        """Test content finding returns empty string for missing titles."""
        sections = {
            "Contents/section0.xml": "Some content here"
        }

        content = parser._find_content_for_title("NonExistent Regulation", sections)

        assert content == ""


# ============================================================================
# Test Class: Integration Tests
# ============================================================================

class TestMultiSectionIntegration:
    """Integration tests for multi-section aggregation."""

    def test_parse_file_with_multi_section_hwpx(self, parser, multi_section_hwpx):
        """Test full parsing workflow with multi-section HWPX file."""
        result = parser.parse_file(multi_section_hwpx)

        # Should successfully parse
        assert result is not None
        assert "docs" in result
        assert "coverage" in result
        assert "metadata" in result

    def test_parse_file_includes_toc_completeness_in_metadata(self, parser, multi_section_hwpx):
        """Test that parsing result includes TOC completeness info."""
        result = parser.parse_file(multi_section_hwpx)

        metadata = result.get("metadata", {})
        assert "toc_complete" in metadata
        assert "missing_regulations" in metadata
        assert isinstance(metadata["toc_complete"], bool)

    def test_parse_file_includes_missing_titles_when_incomplete(self, parser, incomplete_toc_hwpx):
        """Test parsing includes missing titles when TOC is incomplete."""
        result = parser.parse_file(incomplete_toc_hwpx)

        metadata = result.get("metadata", {})
        assert metadata.get("toc_complete") is False
        assert metadata.get("missing_regulations", 0) > 0
        assert "missing_titles" in metadata
        assert len(metadata.get("missing_titles", [])) > 0

    def test_parse_file_handles_duplicate_content(self, parser, duplicate_content_hwpx):
        """Test parsing handles duplicate content across sections."""
        result = parser.parse_file(duplicate_content_hwpx)

        # Should not crash with duplicates
        assert result is not None
        assert "docs" in result

    def test_parse_file_logs_section_aggregation(self, parser, multi_section_hwpx, caplog):
        """Test parsing logs section aggregation progress."""
        import logging
        caplog.set_level(logging.INFO)

        parser.parse_file(multi_section_hwpx)

        # Check for aggregation-related log messages
        log_messages = [record.message for record in caplog.records]
        assert any("section" in msg.lower() for msg in log_messages)


# ============================================================================
# Test Class: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases for multi-section aggregation."""

    def test_aggregate_sections_with_single_section(self, parser, tmp_path):
        """Test aggregation with only one section."""
        hwpx_path = tmp_path / "single.hwpx"

        with zipfile.ZipFile(hwpx_path, 'w') as zf:
            zf.writestr("Contents/section0.xml", "Single section content")

        sections = parser._aggregate_sections(hwpx_path)

        assert len(sections) == 1
        assert "Contents/section0.xml" in sections

    def test_aggregate_sections_with_many_sections(self, parser, tmp_path):
        """Test aggregation with many sections (10+)."""
        hwpx_path = tmp_path / "many.hwpx"

        with zipfile.ZipFile(hwpx_path, 'w') as zf:
            for i in range(15):
                zf.writestr(f"Contents/section{i}.xml", f"Section {i} content")

        sections = parser._aggregate_sections(hwpx_path)

        assert len(sections) == 15

    def test_merge_with_large_duplicate_content(self, parser):
        """Test merging handles large content with many duplicates."""
        # Create sections with repeated patterns
        base_content = "\n".join([f"Line {i}" for i in range(100)])
        sections = {
            "Contents/section0.xml": base_content,
            "Contents/section2.xml": base_content  # Exact duplicate
        }

        merged = parser._merge_section_contents(sections)

        # Should only have 100 unique lines
        lines = [l for l in merged.split('\n') if l.strip()]
        assert len(lines) == 100

    def test_validate_completeness_with_unicode_titles(self, parser):
        """Test validation handles Unicode regulation titles."""
        toc_entries = [
            {"title": "연구윤리규정"},
            {"title": "연구장려규정"},
            {"title": "연구지원규정"}
        ]
        sections = {
            "Contents/section0.xml": "연구윤리규정\n연구지원규정"
        }

        is_complete, missing = parser._validate_toc_completeness(toc_entries, sections)

        assert is_complete is False
        assert "연구장려규정" in missing

    def test_find_content_with_partial_title_match(self, parser):
        """Test content finding with partial title matches."""
        sections = {
            "Contents/section0.xml": "동의대학교겸임교원규정\n제1조 내용"
        }

        # Search for shorter version of title
        content = parser._find_content_for_title("겸임교원규정", sections)

        # Should still find it
        assert content is not None
        assert len(content) > 0
