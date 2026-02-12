"""
Structure Pattern Definitions for University Regulations

This module defines pattern matching rules for different regulation types
based on their issuing authority and document structure.
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RegulationAuthority(Enum):
    """University regulation authority types."""
    UNIVERSITY_COUNCIL = "대학본부"  # University Council
    GRADUATE_SCHOOL = "대학원학칙"  # Graduate School Regulations
    PRESIDENTIAL_DIRECTIVE = "직제행정"  # Presidential Directive
    DIRECTIVE_REGULATION = "직제규정"  # Directive Regulations
    OTHER = "기타규정"  # Other Regulations


@dataclass
class StructurePattern:
    """
    Pattern definition for a specific regulation authority.

    Attributes:
        authority: The regulation authority type
        title_pattern: Regex pattern to match regulation titles
        part_pattern: Regex pattern to match part markers (편)
        chapter_pattern: Regex pattern to match chapter markers (장)
        structure_type: String describing the structure type
    """
    authority: RegulationAuthority
    title_pattern: re.Pattern
    part_pattern: re.Pattern
    chapter_pattern: re.Pattern
    structure_type: str

    def matches_title(self, text: str) -> bool:
        """Check if text matches this pattern's title."""
        return bool(self.title_pattern.search(text))

    def extract_part_number(self, text: str) -> Optional[str]:
        """Extract part number from text."""
        match = self.part_pattern.search(text)
        return match.group(1) if match else None

    def extract_chapter_number(self, text: str) -> Optional[str]:
        """Extract chapter number from text."""
        match = self.chapter_pattern.search(text)
        return match.group(1) if match else None


class StructurePatterns:
    """
    Collection of structure patterns for different regulation types.

    Patterns:
    1. University Council (대학본부): Part [N-1] Chapter
    2. Graduate School (대학원학칙): Part [N-2] Chapter
    3. Presidential Directive (직제/행정): No part/chapter markers
    4. Directive Regulations (직제규정): No part/chapter markers
    5. Other Regulations (기타규정): Part [기-6] Authority
    """

    # University Council Pattern: 대학본부 → [N-1]장
    UNIVERSITY_COUNCIL = StructurePattern(
        authority=RegulationAuthority.UNIVERSITY_COUNCIL,
        title_pattern=re.compile(r'대학\s*본부\s*.*규정'),
        part_pattern=re.compile(r'\[(\d+-\d+)\]장'),  # [N-1]장
        chapter_pattern=re.compile(r'\[(\d+-\d+)\]장'),  # [N-1]장
        structure_type="university_council"
    )

    # Graduate School Pattern: 대학원학칙 → [N-2]장
    GRADUATE_SCHOOL = StructurePattern(
        authority=RegulationAuthority.GRADUATE_SCHOOL,
        title_pattern=re.compile(r'대학\s*원\s*학칙'),
        part_pattern=re.compile(r'\[(\d+-\d+)\]장'),  # [N-2]장
        chapter_pattern=re.compile(r'\[(\d+-\d+)\]장'),  # [N-2]장
        structure_type="graduate_school"
    )

    # Presidential Directive Pattern: 직제/행정 → No structure markers
    PRESIDENTIAL_DIRECTIVE = StructurePattern(
        authority=RegulationAuthority.PRESIDENTIAL_DIRECTIVE,
        title_pattern=re.compile(r'직제\s*행정|대학\s*직제\s*행정'),
        part_pattern=re.compile(r'$^'),  # No part markers
        chapter_pattern=re.compile(r'$^'),  # No chapter markers
        structure_type="presidential_directive"
    )

    # Directive Regulation Pattern: 직제규정 → No structure markers
    DIRECTIVE_REGULATION = StructurePattern(
        authority=RegulationAuthority.DIRECTIVE_REGULATION,
        title_pattern=re.compile(r'직제\s*규정'),
        part_pattern=re.compile(r'$^'),  # No part markers
        chapter_pattern=re.compile(r'$^'),  # No chapter markers
        structure_type="directive_regulation"
    )

    # Other Regulations Pattern: 기타규정 → [기-6]권
    OTHER_REGULATIONS = StructurePattern(
        authority=RegulationAuthority.OTHER,
        title_pattern=re.compile(r'기타\s*규정'),
        part_pattern=re.compile(r'\[기-(\d+)\]권'),  # [기-6]권
        chapter_pattern=re.compile(r'\[기-(\d+)\]권'),  # [기-6]권
        structure_type="other_regulations"
    )

    # Default/Fallback Pattern: 그 외 → Simple numbering
    DEFAULT = StructurePattern(
        authority=RegulationAuthority.OTHER,
        title_pattern=re.compile(r'.+'),  # Matches any title
        part_pattern=re.compile(r'\[(\d+)\]장'),  # Simple [N]장
        chapter_pattern=re.compile(r'\[(\d+)\]장'),  # Simple [N]장
        structure_type="default"
    )

    @classmethod
    def identify_pattern(
        cls, title: str, document_type: Optional[str] = None
    ) -> Optional[StructurePattern]:
        """
        Identify the appropriate structure pattern based on regulation title.

        Args:
            title: The regulation title text
            document_type: Optional document type hint

        Returns:
            Matching StructurePattern or None
        """
        # Check each pattern in priority order
        patterns_to_check = [
            cls.UNIVERSITY_COUNCIL,
            cls.GRADUATE_SCHOOL,
            cls.PRESIDENTIAL_DIRECTIVE,
            cls.DIRECTIVE_REGULATION,
            cls.OTHER_REGULATIONS,
        ]

        for pattern in patterns_to_check:
            if pattern.matches_title(title):
                return pattern

        # Fallback to default pattern
        return cls.DEFAULT

    @classmethod
    def get_all_patterns(cls) -> list[StructurePattern]:
        """Return all available patterns."""
        return [
            cls.UNIVERSITY_COUNCIL,
            cls.GRADUATE_SCHOOL,
            cls.PRESIDENTIAL_DIRECTIVE,
            cls.DIRECTIVE_REGULATION,
            cls.OTHER_REGULATIONS,
            cls.DEFAULT,
        ]


def detect_authority_from_text(text: str) -> RegulationAuthority:
    """
    Detect regulation authority from text content.

    Args:
        text: Text to analyze

    Returns:
        Detected RegulationAuthority
    """
    text_lower = text.lower()

    if '대학본부' in text_lower or '대학 본부' in text_lower:
        return RegulationAuthority.UNIVERSITY_COUNCIL
    elif '대학원학칙' in text_lower or '대학원 학칙' in text_lower:
        return RegulationAuthority.GRADUATE_SCHOOL
    elif '직제행정' in text_lower or '직제 행정' in text_lower:
        return RegulationAuthority.PRESIDENTIAL_DIRECTIVE
    elif '직제규정' in text_lower or '직제 규정' in text_lower:
        return RegulationAuthority.DIRECTIVE_REGULATION
    elif '기타규정' in text_lower or '기타 규정' in text_lower:
        return RegulationAuthority.OTHER
    else:
        return RegulationAuthority.OTHER
