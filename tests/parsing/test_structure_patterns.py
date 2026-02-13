"""
Unit tests for Structure Patterns and Structure Analyzer

Tests the authority-based pattern detection and structure analysis for
different regulation types.
"""
import pytest
from pathlib import Path

from src.parsing.structure_analyzer import (
    RegulationAuthority,
    StructureAnalyzer,
    StructureInfo,
    StructurePatterns,
    get_authority_display_name,
    get_structure_summary,
    detect_authority_from_text,
)


class TestRegulationAuthority:
    """Test regulation authority detection."""

    def test_detect_university_council(self):
        """Test detection of university council regulations."""
        text = "대학본부규정"
        authority = detect_authority_from_text(text)
        assert authority == RegulationAuthority.UNIVERSITY_COUNCIL

    def test_detect_graduate_school(self):
        """Test detection of graduate school regulations."""
        text = "대학원학칙"
        authority = detect_authority_from_text(text)
        assert authority == RegulationAuthority.GRADUATE_SCHOOL

    def test_detect_presidential_directive(self):
        """Test detection of presidential directive."""
        text = "직제행정"
        authority = detect_authority_from_text(text)
        assert authority == RegulationAuthority.PRESIDENTIAL_DIRECTIVE

    def test_detect_directive_regulation(self):
        """Test detection of directive regulation."""
        text = "직제규정"
        authority = detect_authority_from_text(text)
        assert authority == RegulationAuthority.DIRECTIVE_REGULATION

    def test_detect_other_regulations(self):
        """Test detection of other regulations."""
        text = "기타규정"
        authority = detect_authority_from_text(text)
        assert authority == RegulationAuthority.OTHER


class TestStructurePatterns:
    """Test structure pattern definitions."""

    def test_university_council_pattern(self):
        """Test university council structure pattern."""
        pattern = StructurePatterns.UNIVERSITY_COUNCIL
        assert pattern.authority == RegulationAuthority.UNIVERSITY_COUNCIL
        assert pattern.structure_type == "university_council"

    def test_graduate_school_pattern(self):
        """Test graduate school structure pattern."""
        pattern = StructurePatterns.GRADUATE_SCHOOL
        assert pattern.authority == RegulationAuthority.GRADUATE_SCHOOL
        assert pattern.structure_type == "graduate_school"

    def test_presidential_directive_pattern(self):
        """Test presidential directive pattern (no parts/chapters)."""
        pattern = StructurePatterns.PRESIDENTIAL_DIRECTIVE
        assert pattern.authority == RegulationAuthority.PRESIDENTIAL_DIRECTIVE
        assert pattern.structure_type == "presidential_directive"

    def test_directive_regulation_pattern(self):
        """Test directive regulation pattern (no parts/chapters)."""
        pattern = StructurePatterns.DIRECTIVE_REGULATION
        assert pattern.authority == RegulationAuthority.DIRECTIVE_REGULATION
        assert pattern.structure_type == "directive_regulation"

    def test_other_regulations_pattern(self):
        """Test other regulations pattern."""
        pattern = StructurePatterns.OTHER_REGULATIONS
        assert pattern.authority == RegulationAuthority.OTHER
        assert pattern.structure_type == "other_regulations"

    def test_pattern_matching(self):
        """Test pattern title matching."""
        patterns = StructurePatterns.get_all_patterns()

        # University council pattern
        assert patterns[0].matches_title("대학본부규정")
        assert patterns[0].matches_title("대학 본부 규정")

        # Graduate school pattern
        assert patterns[1].matches_title("대학원학칙")
        assert patterns[1].matches_title("대학원 학칙")

        # Presidential directive
        assert patterns[2].matches_title("직제행정")
        assert patterns[2].matches_title("대학 직제 행정")

        # Directive regulation
        assert patterns[3].matches_title("직제규정")

        # Other regulations
        assert patterns[4].matches_title("기타규정")

    def test_part_number_extraction(self):
        """Test part number extraction from patterns."""
        # University council: [N-1]장
        pattern = StructurePatterns.UNIVERSITY_COUNCIL
        match = pattern.part_pattern.search("[1-1]장")
        assert match is not None
        assert match.group(1) == "1-1"

        # Other regulations: [기-6]권
        pattern = StructurePatterns.OTHER_REGULATIONS
        match = pattern.part_pattern.search("[기-6]권")
        assert match is not None
        assert match.group(1) == "6"


class TestStructureAnalyzer:
    """Test structure analyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StructureAnalyzer()

    def test_analyze_university_council_title(self):
        """Test analysis of university council regulation title."""
        title = "대학본부규정"
        info = self.analyzer.analyze_title(title)

        assert info.authority == RegulationAuthority.UNIVERSITY_COUNCIL
        assert info.structure_type == "university_council"
        assert info.has_parts is True
        assert info.has_chapters is True
        assert info.part_format == "[N-1]장"
        assert info.chapter_format == "[N-1]장"

    def test_analyze_graduate_school_title(self):
        """Test analysis of graduate school regulation title."""
        title = "대학원학칙"
        info = self.analyzer.analyze_title(title)

        assert info.authority == RegulationAuthority.GRADUATE_SCHOOL
        assert info.structure_type == "graduate_school"
        assert info.has_parts is True
        assert info.has_chapters is True
        assert info.part_format == "[N-2]장"
        assert info.chapter_format == "[N-2]장"

    def test_analyze_presidential_directive_title(self):
        """Test analysis of presidential directive."""
        title = "직제행정"
        info = self.analyzer.analyze_title(title)

        assert info.authority == RegulationAuthority.PRESIDENTIAL_DIRECTIVE
        assert info.structure_type == "presidential_directive"
        assert info.has_parts is False
        assert info.has_chapters is False

    def test_analyze_directive_regulation_title(self):
        """Test analysis of directive regulation."""
        title = "직제규정"
        info = self.analyzer.analyze_title(title)

        assert info.authority == RegulationAuthority.DIRECTIVE_REGULATION
        assert info.structure_type == "directive_regulation"
        assert info.has_parts is False
        assert info.has_chapters is False

    def test_analyze_other_regulations_title(self):
        """Test analysis of other regulations."""
        title = "기타규정"
        info = self.analyzer.analyze_title(title)

        assert info.authority == RegulationAuthority.OTHER
        assert info.structure_type == "other_regulations"
        assert info.has_parts is True
        assert info.has_chapters is True
        assert info.part_format == "[기-N]권"
        assert info.chapter_format == "[기-N]권"

    def test_analyze_unknown_title(self):
        """Test analysis with unknown title (fallback)."""
        title = "알 수 없는 규정"
        info = self.analyzer.analyze_title(title)

        assert info.authority == RegulationAuthority.OTHER
        assert info.structure_type == "unknown"

    def test_structure_info_to_dict(self):
        """Test StructureInfo serialization."""
        info = StructureInfo(
            authority=RegulationAuthority.UNIVERSITY_COUNCIL,
            structure_type="university_council",
            has_parts=True,
            has_chapters=True,
            part_format="[N-1]장",
            chapter_format="[N-1]장",
        )

        result = info.to_dict()
        assert result["authority"] == "대학본부"
        assert result["structure_type"] == "university_council"
        assert result["has_parts"] is True
        assert result["has_chapters"] is True
        assert result["part_format"] == "[N-1]장"

    def test_classify_regulation(self):
        """Test regulation classification."""
        analyzer = StructureAnalyzer()

        # Test university council
        title = "대학본부규정"
        content = "제1장 총칙"
        authority, classification = analyzer.classify_regulation(title, content)

        assert authority == RegulationAuthority.UNIVERSITY_COUNCIL
        assert classification == "university_council"

        # Test graduate school
        title = "대학원학칙"
        content = "제2장 학칙"
        authority, classification = analyzer.classify_regulation(title, content)

        assert authority == RegulationAuthority.GRADUATE_SCHOOL
        assert classification == "graduate_school"

    def test_get_authority_display_name(self):
        """Test authority display name conversion."""
        assert get_authority_display_name(RegulationAuthority.UNIVERSITY_COUNCIL) == "대학본부"
        assert get_authority_display_name(RegulationAuthority.GRADUATE_SCHOOL) == "대학원학칙"
        assert get_authority_display_name(RegulationAuthority.PRESIDENTIAL_DIRECTIVE) == "직제행정"
        assert get_authority_display_name(RegulationAuthority.DIRECTIVE_REGULATION) == "직제규정"
        assert get_authority_display_name(RegulationAuthority.OTHER) == "기타규정"

    def test_get_structure_summary(self):
        """Test structure summary generation."""
        info = StructureInfo(
            authority=RegulationAuthority.UNIVERSITY_COUNCIL,
            structure_type="university_council",
            has_parts=True,
            has_chapters=True,
            part_format="[N-1]장",
            chapter_format="[N-1]장",
        )

        summary = get_structure_summary(
            RegulationAuthority.UNIVERSITY_COUNCIL,
            info
        )

        assert "대학본부" in summary
        assert "[N-1]장" in summary
        assert "파트 구조" in summary or "챕터 구조" in summary
