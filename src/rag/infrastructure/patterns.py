"""
Common regex patterns for Korean regulation text parsing.

Provides centralized patterns for article references, regulation names,
and other common text patterns used across the RAG system.
"""

import re
from typing import Pattern

# ====================
# Article Reference Patterns
# ====================

# Matches: 제N조, 제N조의M, 제N항, 제N호
ARTICLE_PATTERN: Pattern[str] = re.compile(
    r"제\s*\d+\s*조(?:\s*의\s*\d+)?|제\s*\d+\s*항|제\s*\d+\s*호"
)

# Matches article numbers more loosely (with optional 제)
ARTICLE_LOOSE_PATTERN: Pattern[str] = re.compile(
    r"(?:제)?\s*(\d+)\s*조"
)

# Matches any hierarchical reference: 제N조, 제N항, 제N장, 제N절
HIERARCHY_PATTERN: Pattern[str] = re.compile(
    r"(?:제)?\s*\d+\s*[조항장절]"
)

# ====================
# Regulation Name Patterns
# ====================

# Regulation name suffixes
REGULATION_SUFFIXES = ("규정", "학칙", "내규", "세칙", "지침", "정관")
REGULATION_SUFFIX_PATTERN = r"(?:규정|학칙|내규|세칙|지침|정관)"

# Matches regulation name only (e.g., "교원인사규정", "학칙")
REGULATION_ONLY_PATTERN: Pattern[str] = re.compile(
    rf"^\s*([가-힣]+{REGULATION_SUFFIX_PATTERN})\s*$"
)

# Matches regulation name + article reference (e.g., "교원인사규정 제8조")
REGULATION_ARTICLE_PATTERN: Pattern[str] = re.compile(
    rf"([가-힣]+{REGULATION_SUFFIX_PATTERN})\s*"
    r"(제\s*\d+\s*조(?:\s*의\s*\d+)?(?:\s*제?\s*\d+\s*항)?(?:\s*제?\s*\d+\s*호)?)"
)

# ====================
# Rule Code Patterns
# ====================

# Matches rule codes like "3-1-24"
RULE_CODE_PATTERN: Pattern[str] = re.compile(r"^\d+(?:-\d+){2,}$")

# ====================
# Content Patterns
# ====================

# Heading-only pattern (e.g., "(목적)")
HEADING_ONLY_PATTERN: Pattern[str] = re.compile(r"^\([^)]*\)\s*$")

# ====================
# Hierarchy Level Patterns (for chapter/section/subsection)
# ====================

# Pattern for chapter (장/편)
CHAPTER_PATTERN: Pattern[str] = re.compile(r"^(제\s*(\d+)\s*[장편])\s*(.*)")

# Pattern for section (절)
SECTION_PATTERN: Pattern[str] = re.compile(r"^(제\s*(\d+)\s*절)\s*(.*)")

# Pattern for subsection (관)
SUBSECTION_PATTERN: Pattern[str] = re.compile(r"^(제\s*(\d+)\s*관)\s*(.*)")


# ====================
# Utility Functions
# ====================

def normalize_article_token(token: str) -> str:
    """Remove all whitespace from article reference token.

    Args:
        token: Article token like "제 8 조" or "제8조의2"

    Returns:
        Normalized token like "제8조" or "제8조의2"
    """
    return re.sub(r"\s+", "", token)


def extract_article_references(text: str) -> set:
    """Extract all article references from text.

    Args:
        text: Text containing article references.

    Returns:
        Set of normalized article reference strings.
    """
    return {
        normalize_article_token(match)
        for match in ARTICLE_PATTERN.findall(text)
    }


def remove_article_references(text: str) -> str:
    """Remove article reference patterns from text.

    Args:
        text: Text containing article references.

    Returns:
        Text with article references removed.
    """
    return re.sub(r"제\d+조(의\d+)?", "", text)


def is_regulation_name(text: str) -> bool:
    """Check if text is a regulation name.

    Args:
        text: Text to check.

    Returns:
        True if text ends with a regulation suffix.
    """
    return text.strip().endswith(REGULATION_SUFFIXES)
