"""
Format Type Definitions for HWPX Regulation Parsing.

This module defines the format types and list patterns used in
format classification for HWPX regulation parsing (SPEC-HWXP-002).

TDD Approach: GREEN Phase
- Implementation created to make failing tests pass
- Minimal implementation focused on test requirements
"""
from enum import Enum


class FormatType(Enum):
    """
    Regulation format types for classification.

    HWPX regulations can be in one of four formats:
    - ARTICLE: Has clear article markers (제N조)
    - LIST: Bullet/numbered lists without articles
    - GUIDELINE: Continuous prose without clear structure
    - UNSTRUCTURED: Ambiguous or mixed content
    """

    ARTICLE = "article"
    """Clear article markers like 제1조, 제2조, etc."""

    LIST = "list"
    """Numbered or bulleted lists without articles."""

    GUIDELINE = "guideline"
    """Continuous prose without clear structural markers."""

    UNSTRUCTURED = "unstructured"
    """Ambiguous content that doesn't fit other categories."""

    def __str__(self) -> str:
        """Return string representation of the format type."""
        return self.value


class ListPattern(Enum):
    """
    List pattern types for list format classification.

    List-format regulations can use different numbering styles:
    - NUMERIC: 1., 2., 3. (Western numerals)
    - KOREAN_ALPHABET: 가., 나., 다. (Korean alphabet)
    - CIRCLED_NUMBER: ①, ②, ③ (Circled numbers)
    - MIXED: Combination of multiple patterns
    """

    NUMERIC = "numeric"
    """Western numeric list markers (1., 2., 3.)"""

    KOREAN_ALPHABET = "korean"
    """Korean alphabet list markers (가., 나., 다.)"""

    CIRCLED_NUMBER = "circled"
    """Circled number markers (①, ②, ③)"""

    MIXED = "mixed"
    """Combination of multiple list patterns"""

    def __str__(self) -> str:
        """Return string representation of the list pattern."""
        return self.value
