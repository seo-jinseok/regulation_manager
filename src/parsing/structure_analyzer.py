"""
Structure Analyzer for University Regulations

This module provides structure analysis capabilities for different regulation types,
detecting authority-based patterns and extracting hierarchical structure.
"""
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .structure_patterns import (
    RegulationAuthority,
    StructurePattern,
    StructurePatterns,
    detect_authority_from_text,
)

logger = logging.getLogger(__name__)


@dataclass
class StructureInfo:
    """
    Information about document structure.

    Attributes:
        authority: The regulation authority type
        structure_type: String describing the structure type
        has_parts: Whether the document has part markers
        has_chapters: Whether the document has chapter markers
        part_format: Format of part markers (e.g., "[N-1]장")
        chapter_format: Format of chapter markers
    """
    authority: RegulationAuthority
    structure_type: str
    has_parts: bool = False
    has_chapters: bool = False
    part_format: Optional[str] = None
    chapter_format: Optional[str] = None

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "authority": self.authority.value,
            "structure_type": self.structure_type,
            "has_parts": self.has_parts,
            "has_chapters": self.has_chapters,
            "part_format": self.part_format,
            "chapter_format": self.chapter_format,
        }


class StructureAnalyzer:
    """
    Analyze regulation document structure based on authority type.

    Detects and extracts structural patterns from regulation text,
    enabling proper parsing of different document types.
    """

    def __init__(self):
        """Initialize the structure analyzer."""
        self.patterns = StructurePatterns

    def analyze_title(self, title: str) -> StructureInfo:
        """
        Analyze a regulation title to determine structure type.

        Args:
            title: The regulation title text

        Returns:
            StructureInfo with detected pattern information
        """
        # Detect authority from title
        authority = detect_authority_from_text(title)

        # Get matching pattern
        pattern = self.patterns.identify_pattern(title)

        if not pattern:
            # Default fallback
            return StructureInfo(
                authority=RegulationAuthority.OTHER,
                structure_type="unknown",
                has_parts=False,
                has_chapters=False,
            )

        return StructureInfo(
            authority=authority,
            structure_type=pattern.structure_type,
            has_parts=self._check_for_part_markers(pattern.part_pattern),
            has_chapters=self._check_for_chapter_markers(pattern.chapter_pattern),
            part_format=self._get_part_format(pattern),
            chapter_format=self._get_chapter_format(pattern),
        )

    def analyze_document(
        self,
        content_lines: List[str],
        title: str
    ) -> Tuple[StructureInfo, List[Dict[str, any]]]:
        """
        Analyze full document to extract structure markers.

        Args:
            content_lines: List of content lines
            title: Document title

        Returns:
            Tuple of (StructureInfo, list_of_markers)
        """
        structure_info = self.analyze_title(title)
        markers = []

        for line in content_lines:
            line = line.strip()
            if not line:
                continue

            # Check for part markers
            if structure_info.part_format:
                part_match = self.patterns.identify_pattern(title).part_pattern.search(line)
                if part_match:
                    markers.append({
                        "type": "part",
                        "text": line,
                        "number": part_match.group(1),
                    })
                    structure_info.has_parts = True

            # Check for chapter markers
            if structure_info.chapter_format:
                chapter_match = self.patterns.identify_pattern(title).chapter_pattern.search(line)
                if chapter_match:
                    markers.append({
                        "type": "chapter",
                        "text": line,
                        "number": chapter_match.group(1),
                    })
                    structure_info.has_chapters = True

        return structure_info, markers

    def extract_structure_outline(
        self,
        content_lines: List[str],
        structure_info: StructureInfo
    ) -> List[Dict[str, any]]:
        """
        Extract structural outline from document content.

        Args:
            content_lines: List of content lines
            structure_info: Document structure information

        Returns:
            List of outline items with hierarchy
        """
        outline = []
        current_level = 0
        current_part = None
        current_chapter = None

        for line in content_lines:
            line = line.strip()
            if not line:
                continue

            # Check for part markers
            if structure_info.part_format:
                part_pattern = self.patterns.identify_pattern(
                    structure_info.authority.value
                ).part_pattern
                part_match = part_pattern.search(line)
                if part_match:
                    current_part = part_match.group(1)
                    current_level = 0
                    outline.append({
                        "level": current_level,
                        "type": "part",
                        "number": current_part,
                        "text": line,
                    })
                    continue

            # Check for chapter markers
            if structure_info.chapter_format:
                chapter_pattern = self.patterns.identify_pattern(
                    structure_info.authority.value
                ).chapter_pattern
                chapter_match = chapter_pattern.search(line)
                if chapter_match:
                    current_chapter = chapter_match.group(1)
                    current_level = 1
                    outline.append({
                        "level": current_level,
                        "type": "chapter",
                        "number": current_chapter,
                        "text": line,
                    })
                    continue

            # Check for article markers
            if self._is_article_marker(line):
                current_level = 2
                outline.append({
                    "level": current_level,
                    "type": "article",
                    "text": line,
                })
                continue

        return outline

    def classify_regulation(
        self,
        title: str,
        content: str
    ) -> Tuple[RegulationAuthority, str]:
        """
        Classify regulation based on title and content.

        Args:
            title: Regulation title
            content: Regulation content

        Returns:
            Tuple of (authority, classification_string)
        """
        authority = detect_authority_from_text(title + " " + content[:500] if len(content) > 500 else content)

        # Generate classification string
        if authority == RegulationAuthority.UNIVERSITY_COUNCIL:
            classification = "university_council"
        elif authority == RegulationAuthority.GRADUATE_SCHOOL:
            classification = "graduate_school"
        elif authority == RegulationAuthority.PRESIDENTIAL_DIRECTIVE:
            classification = "presidential_directive"
        elif authority == RegulationAuthority.DIRECTIVE_REGULATION:
            classification = "directive_regulation"
        else:
            classification = "other_regulations"

        return authority, classification

    def _check_for_part_markers(self, pattern: re.Pattern) -> bool:
        """Check if pattern has valid part markers."""
        return pattern.pattern != re.compile(r'$^')

    def _check_for_chapter_markers(self, pattern: re.Pattern) -> bool:
        """Check if pattern has valid chapter markers."""
        return pattern.pattern != re.compile(r'$^')

    def _get_part_format(self, pattern: StructurePattern) -> Optional[str]:
        """Get part format description."""
        if pattern.part_pattern.pattern == re.compile(r'$^').pattern:
            return None
        # Extract format from pattern
        pattern_str = pattern.part_pattern.pattern
        if r'\[(\d+-\d+)\]장' in pattern_str:
            return "[N-N]장"
        elif r'\[기-(\d+)\]권' in pattern_str:
            return "[기-N]권"
        elif r'\[(\d+)\]장' in pattern_str:
            return "[N]장"
        return None

    def _get_chapter_format(self, pattern: StructurePattern) -> Optional[str]:
        """Get chapter format description."""
        if pattern.chapter_pattern.pattern == re.compile(r'$^').pattern:
            return None
        pattern_str = pattern.chapter_pattern.pattern
        if r'\[(\d+-\d+)\]장' in pattern_str:
            return "[N-N]장"
        elif r'\[(\d+)\]장' in pattern_str:
            return "[N]장"
        elif r'\[기-(\d+)\]권' in pattern_str:
            return "[기-N]권"
        return None

    def _is_article_marker(self, text: str) -> bool:
        """Detect if text is an article (조) marker."""
        # Handle both "제N조" and "## 제N조" formats
        cleaned_text = text.strip()
        if cleaned_text.startswith("##"):
            cleaned_text = cleaned_text[2:].strip()

        return bool(re.match(r'^제\s*\d+[조의]*\d*\s*(\(.+\))?', cleaned_text))


def get_authority_display_name(authority: RegulationAuthority) -> str:
    """
    Get display name for authority type.

    Args:
        authority: RegulationAuthority enum

    Returns:
        Korean display name
    """
    display_names = {
        RegulationAuthority.UNIVERSITY_COUNCIL: "대학본부",
        RegulationAuthority.GRADUATE_SCHOOL: "대학원학칙",
        RegulationAuthority.PRESIDENTIAL_DIRECTIVE: "직제행정",
        RegulationAuthority.DIRECTIVE_REGULATION: "직제규정",
        RegulationAuthority.OTHER: "기타규정",
    }
    return display_names.get(authority, "기타")


def get_structure_summary(
    authority: RegulationAuthority,
    structure_info: StructureInfo
) -> str:
    """
    Get human-readable summary of structure.

    Args:
        authority: Regulation authority type
        structure_info: Structure information

    Returns:
        Summary string in Korean
    """
    authority_name = get_authority_display_name(authority)

    parts_desc = []
    if structure_info.has_parts:
        if structure_info.part_format:
            parts_desc.append(f"파트 구조: {structure_info.part_format}")
        else:
            parts_desc.append("파트 구조 있음")
    if structure_info.has_chapters:
        if structure_info.chapter_format:
            parts_desc.append(f"챕터 구조: {structure_info.chapter_format}")
        else:
            parts_desc.append("챕터 구조 있음")

    if parts_desc:
        return f"{authority_name} ({', '.join(parts_desc)})"
    return authority_name
