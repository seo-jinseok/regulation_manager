"""
Citation Patterns Value Object.

Defines standard citation patterns for Korean regulation references.
Provides pattern matching and validation for citation format standardization.
"""

import re
from enum import Enum
from typing import Optional


class CitationFormat(Enum):
    """Citation format types."""

    STANDARD = "standard"  # 「규정명」 제X조
    WITH_PARAGRAPH = "with_paragraph"  # 「규정명」 제X조 제X항
    WITH_SUB_ARTICLE = "with_sub_article"  # 「규정명」 제X조의Y
    INCOMPLETE = "incomplete"  # Missing required elements


class CitationPatterns:
    """
    Value object for citation pattern definitions and matching.

    Standard Citation Formats:
    - 「규정명」 제X조 (standard)
    - 「규정명」 제X조 제X항 (with paragraph)
    - 「규정명」 제X조의Y (with sub-article)

    Examples:
        >>> patterns = CitationPatterns()
        >>> match = patterns.match_citation("「학칙」 제25조에 따르면")
        >>> match.group(1)
        '학칙'
        >>> patterns.is_valid_format("「학칙」 제25조")
        True
    """

    # Standard citation pattern: 「규정명」 제X조 [제X항]
    # Group 1: regulation name
    # Group 2: article number
    # Group 3: paragraph number (optional)
    STANDARD_PATTERN = re.compile(r"「([^」]+)」\s*제(\d+)조(?:\s*제(\d+)항)?")

    # Sub-article pattern: 「규정명」 제X조의Y
    # Group 1: regulation name
    # Group 2: article number
    # Group 3: sub-article number
    SUB_ARTICLE_PATTERN = re.compile(r"「([^」]+)」\s*제(\d+)조의(\d+)")

    # Combined pattern for all valid formats (ordered by specificity)
    ALL_PATTERNS = [
        # Sub-article first (more specific)
        (CitationFormat.WITH_SUB_ARTICLE, SUB_ARTICLE_PATTERN),
        # Then standard with optional paragraph
        (CitationFormat.STANDARD, STANDARD_PATTERN),
    ]

    def match_citation(self, text: str) -> Optional[re.Match]:
        """
        Match citation pattern in text.

        Args:
            text: Text to search for citation pattern

        Returns:
            Match object if found, None otherwise

        Note:
            Returns first match from combined pattern.
            For format-specific matching, use match_with_format().
        """
        return self.STANDARD_PATTERN.search(text)

    def match_with_format(self, text: str) -> Optional[tuple[CitationFormat, re.Match]]:
        """
        Match citation with format type information.

        Args:
            text: Text to search for citation pattern

        Returns:
            Tuple of (format_type, match) if found, None otherwise
        """
        # Try sub-article pattern first (more specific)
        match = self.SUB_ARTICLE_PATTERN.search(text)
        if match:
            return (CitationFormat.WITH_SUB_ARTICLE, match)

        # Then try standard pattern
        match = self.STANDARD_PATTERN.search(text)
        if match:
            # Check if has paragraph
            if match.group(3):
                return (CitationFormat.WITH_PARAGRAPH, match)
            return (CitationFormat.STANDARD, match)

        return None

    def is_valid_format(self, citation: str) -> bool:
        """
        Check if citation follows standard format.

        Args:
            citation: Citation text to validate

        Returns:
            True if citation follows standard format, False otherwise
        """
        # Must have guillemets, regulation name, and article number
        for _, pattern in self.ALL_PATTERNS:
            if pattern.fullmatch(citation.strip()):
                return True
        return False

    def standardize(self, citation: str) -> Optional[str]:
        """
        Standardize citation format.

        Normalizes whitespace and ensures consistent formatting.

        Args:
            citation: Citation text to standardize

        Returns:
            Standardized citation if valid, None otherwise
        """
        result = self.match_with_format(citation)
        if not result:
            return None

        format_type, match = result
        regulation = match.group(1)
        article = match.group(2)

        if format_type == CitationFormat.WITH_SUB_ARTICLE:
            sub_article = match.group(3)
            return f"「{regulation}」 제{article}조의{sub_article}"
        elif format_type == CitationFormat.WITH_PARAGRAPH:
            paragraph = match.group(3)
            return f"「{regulation}」 제{article}조 제{paragraph}항"
        else:
            return f"「{regulation}」 제{article}조"

    def find_all(self, text: str) -> list[tuple[CitationFormat, re.Match]]:
        """
        Find all citations in text.

        Args:
            text: Text to search for citations

        Returns:
            List of (format_type, match) tuples in order of appearance
        """
        results = []

        # Find all sub-article matches
        for match in self.SUB_ARTICLE_PATTERN.finditer(text):
            results.append((CitationFormat.WITH_SUB_ARTICLE, match))

        # Find all standard matches (excluding those already matched as sub-article)
        for match in self.STANDARD_PATTERN.finditer(text):
            # Skip if this is actually a sub-article (overlapping match)
            is_sub_article = False
            for fmt, sub_match in results:
                if (
                    fmt == CitationFormat.WITH_SUB_ARTICLE
                    and sub_match.start() == match.start()
                ):
                    is_sub_article = True
                    break

            if not is_sub_article:
                if match.group(3):
                    results.append((CitationFormat.WITH_PARAGRAPH, match))
                else:
                    results.append((CitationFormat.STANDARD, match))

        # Sort by position in text
        results.sort(key=lambda x: x[1].start())

        return results
