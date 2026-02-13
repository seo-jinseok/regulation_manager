"""
Text Normalizer for HWPX Regulation Parsing

Provides comprehensive text cleaning and normalization for HWPX extracted text,
addressing common artifacts that prevent accurate regulation parsing.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Normalize and clean HWPX extracted text.

    Handles:
    - Page header contamination (e.g., "겸임교원규정 3—1—10～")
    - Unicode filler characters (U+F0800-U+F0FFF)
    - Title duplication (e.g., "규정명 페이지헤더 규정명")
    - Horizontal rules and separator characters
    - Whitespace normalization
    """

    # Page header pattern: digit(s) + dash + digit(s) + dash + digit(s) + tilde
    # Dashes: - (U+002D), ― (U+2015), － (U+FF0D), — (U+2014)
    # Tildes: ~ (U+007E), ～ (U+FF5E), 〜 (U+301C)
    # Pattern matches: "3-1-10～" or variations with different dash types
    PAGE_HEADER_PATTERN = re.compile(
        r'\s*[\d]+[\-―－—][\d]+[\-―－—][\d]+[~～〜].*?$'
    )

    # Unicode filler character range used by HWPX for layout
    FILLER_CHARS_PATTERN = re.compile(r'[\U000f0800-\U000f0fff]+')

    # Horizontal rules/separator characters
    HORIZONTAL_RULE_PATTERN = re.compile(r'[─＿─]+')

    # Whitespace normalization (multiple spaces/tabs to single space)
    WHITESPACE_PATTERN = re.compile(r'\s+')

    # Regulation title keywords for duplicate detection
    REGULATION_KEYWORDS = [
        '규정', '요령', '지침', '세칙', '내규', '학칙', '헌장',
        '기준', '수칙', '준칙', '요강', '운영', '정관',
        '시행세칙', '시행규칙', '운영규정', '관리규정', '처리규정',
    ]

    def __init__(
        self,
        remove_page_headers: bool = True,
        remove_filler_chars: bool = True,
        normalize_whitespace: bool = True,
        detect_duplicate_titles: bool = True,
    ):
        """
        Initialize the text normalizer.

        Args:
            remove_page_headers: Remove page header patterns.
            remove_filler_chars: Remove Unicode filler characters.
            normalize_whitespace: Normalize whitespace.
            detect_duplicate_titles: Detect and clean duplicate titles.
        """
        self.remove_page_headers = remove_page_headers
        self.remove_filler_chars = remove_filler_chars
        self.normalize_whitespace = normalize_whitespace
        self.detect_duplicate_titles = detect_duplicate_titles

    def clean(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text to clean.

        Returns:
            Cleaned text string.
        """
        if not text:
            return ""

        # Apply cleaning steps in order
        if self.remove_filler_chars:
            text = self.FILLER_CHARS_PATTERN.sub('', text)

        if self.remove_page_headers:
            text = self.PAGE_HEADER_PATTERN.sub('', text)

        if self.detect_duplicate_titles:
            text = self._clean_duplicate_titles(text)

        # Remove horizontal rules
        text = self.HORIZONTAL_RULE_PATTERN.sub('', text)

        if self.normalize_whitespace:
            text = self.WHITESPACE_PATTERN.sub(' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _clean_duplicate_titles(self, text: str) -> str:
        """
        Clean duplicate regulation titles caused by page header contamination.

        Examples:
            "겸임교원규정 3—1—10～ 겸임교원규정" -> "겸임교원규정"
            "학칙 1-2-5～ 학칙" -> "학칙"

        Args:
            text: Text that may contain duplicate titles.

        Returns:
            Cleaned text with duplicates removed.
        """
        # Check for pattern: title + page_header + title
        words = text.split()
        if len(words) < 2:
            return text

        # Look for duplicate title pattern
        # First word might be a title keyword
        first_word = words[0]

        # Check if first word ends with regulation keyword
        if any(first_word.endswith(kw) for kw in self.REGULATION_KEYWORDS):
            # Look for duplicate at end
            if len(words) >= 2 and words[-1] == first_word:
                # Found duplicate: return only the title
                return first_word

        return text

    def is_meaningful(self, text: str) -> bool:
        """
        Check if text is meaningful (not empty/whitespace after cleaning).

        Args:
            text: Text to check.

        Returns:
            True if text has meaningful content.
        """
        if not text:
            return False

        cleaned = self.clean(text)
        # Check if cleaned text has at least one Korean or alphanumeric character
        has_content = bool(re.search(r'[가-힣a-zA-Z0-9]', cleaned))
        return has_content

    def clean_line(self, line: str) -> Optional[str]:
        """
        Clean a single line, returning None if line becomes empty.

        Args:
            line: Line to clean.

        Returns:
            Cleaned line or None if empty.
        """
        cleaned = self.clean(line)
        return cleaned if cleaned else None

    def get_title_hash(self, title: str) -> str:
        """
        Get a normalized hash of a title for duplicate detection.

        Args:
            title: Title to hash.

        Returns:
            Normalized title string for comparison.
        """
        # Clean and normalize for comparison
        normalized = self.clean(title)
        # Remove spaces for more aggressive matching
        return normalized.replace(' ', '').lower()

    def titles_match(self, title1: str, title2: str) -> bool:
        """
        Check if two titles match after normalization.

        Args:
            title1: First title.
            title2: Second title.

        Returns:
            True if titles match after normalization.
        """
        return self.get_title_hash(title1) == self.get_title_hash(title2)


# Singleton instance for convenience
_default_normalizer: Optional[TextNormalizer] = None


def get_default_normalizer() -> TextNormalizer:
    """Get or create the default normalizer instance."""
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = TextNormalizer()
    return _default_normalizer


def normalize_regulation_text(text: str) -> str:
    """
    Convenience function to normalize regulation text.

    Args:
        text: Raw text to normalize.

    Returns:
            Normalized text.
    """
    return get_default_normalizer().clean(text)
