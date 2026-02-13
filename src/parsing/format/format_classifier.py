"""
Format Classifier for HWPX Regulation Parsing.

This module implements format classification for HWPX regulations,
detecting whether content is in article, list, guideline, or unstructured format.

TDD Approach: GREEN Phase
- Implementation created to make failing tests pass
- Pattern matching for article markers (제N조)
- Pattern matching for list patterns (1., 2., 가., 나., ①, ②)
- Detection for guideline format (continuous prose)
- Confidence scoring algorithm

Reference: SPEC-HWXP-002, TASK-001
"""
import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from src.parsing.format.format_type import FormatType, ListPattern

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """
    Result of format classification.

    Attributes:
        format_type: The detected format type
        confidence: Confidence score in [0.0, 1.0]
        list_pattern: Detected list pattern (only for LIST format)
        indicators: Dictionary of detected indicators and their counts
    """

    format_type: FormatType
    confidence: float
    list_pattern: Optional[ListPattern] = None
    indicators: Dict[str, Union[int, List[str]]] = None

    def __post_init__(self):
        """Initialize indicators dict if not provided."""
        if self.indicators is None:
            self.indicators = {}


class FormatClassifier:
    """
    Classify regulation content into format types.

    Uses pattern matching to detect:
    - Article markers (제N조)
    - List patterns (numeric, Korean alphabet, circled numbers)
    - Guideline format (continuous prose)
    - Unstructured content (ambiguous or empty)
    """

    # Article marker pattern: 제N조, 제N조의M (with optional space)
    ARTICLE_PATTERN = re.compile(r'제\s*\d+조(?:의\s*\d+)?')

    # Numeric list pattern: 1. 2. 3. (at start of line, space optional after dot)
    NUMERIC_LIST_PATTERN = re.compile(r'^\s*\d+\.\s*', re.MULTILINE)

    # Korean alphabet list pattern: 가. 나. 다. (at start of line, space optional)
    KOREAN_LIST_PATTERN = re.compile(
        r'^\s*[가-하][\.\)]\s*',
        re.MULTILINE
    )

    # Circled number pattern: ① ② ③ (at start of line, space optional)
    CIRCLED_NUMBER_PATTERN = re.compile(
        r'^\s*[①-⑮]\s*',
        re.MULTILINE
    )

    # Minimum list items required to classify as LIST format
    MIN_LIST_ITEMS = 2

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.9
    MEDIUM_CONFIDENCE_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self):
        """Initialize the format classifier."""
        logger.debug("FormatClassifier initialized")

    def classify(self, content: str) -> ClassificationResult:
        """
        Classify regulation content into a format type.

        Classification Logic:
        1. Check for article markers (제N조) -> ARTICLE
        2. Check for list patterns -> LIST
        3. Check for continuous prose -> GUIDELINE
        4. Default to UNSTRUCTURED

        Args:
            content: Regulation text content to classify

        Returns:
            ClassificationResult with format type, confidence, and indicators
        """
        if not content or not content.strip():
            return ClassificationResult(
                format_type=FormatType.UNSTRUCTURED,
                confidence=0.0,
                indicators={"reason": "empty_content"}
            )

        # Check for article format first (highest priority)
        article_result = self._check_article_format(content)
        if article_result is not None:
            return article_result

        # Check for list format
        list_result = self._check_list_format(content)
        if list_result is not None:
            return list_result

        # Check for guideline format
        guideline_result = self._check_guideline_format(content)
        if guideline_result is not None:
            return guideline_result

        # Default to unstructured
        return ClassificationResult(
            format_type=FormatType.UNSTRUCTURED,
            confidence=0.4,
            indicators={"reason": "no_clear_pattern"}
        )

    def _check_article_format(self, content: str) -> Optional[ClassificationResult]:
        """
        Check if content is in article format.

        Article format has markers like 제1조, 제2조, etc.

        Args:
            content: Content to check

        Returns:
            ClassificationResult if article format detected, None otherwise
        """
        article_matches = self.ARTICLE_PATTERN.findall(content)

        if not article_matches:
            return None

        # Calculate confidence based on number of article markers
        article_count = len(article_matches)

        # Base confidence starts higher for article format
        confidence = 0.7 + min(article_count * 0.1, 0.25)

        # Check for article-specific patterns
        has_article_structure = self._has_article_structure(content)

        if has_article_structure:
            confidence = min(confidence + 0.15, 1.0)
        else:
            # Still good confidence if we have article markers
            confidence = min(confidence + 0.05, 1.0)

        return ClassificationResult(
            format_type=FormatType.ARTICLE,
            confidence=confidence,
            indicators={
                "article_markers": article_matches,
                "article_count": article_count,
                "has_structure": has_article_structure
            }
        )

    def _has_article_structure(self, content: str) -> bool:
        """
        Check if content has typical article structure.

        Article structure includes:
        - Article numbers followed by titles in parentheses
        - Hierarchical markers (항, 호, 목)

        Args:
            content: Content to check

        Returns:
            True if article structure detected
        """
        # Check for article title pattern: 제N조(제목)
        article_title_pattern = re.compile(r'제\s*\d+조\s*\([^)]+\)')
        has_titles = bool(article_title_pattern.search(content))

        return has_titles

    def _check_list_format(self, content: str) -> Optional[ClassificationResult]:
        """
        Check if content is in list format.

        List format has numbered or bulleted items.

        Args:
            content: Content to check

        Returns:
            ClassificationResult if list format detected, None otherwise
        """
        lines = content.split('\n')

        # Count different list patterns
        numeric_count = len(self.NUMERIC_LIST_PATTERN.findall(content))
        korean_count = len(self.KOREAN_LIST_PATTERN.findall(content))
        circled_count = len(self.CIRCLED_NUMBER_PATTERN.findall(content))

        # Determine dominant pattern
        pattern_counts = {
            ListPattern.NUMERIC: numeric_count,
            ListPattern.KOREAN_ALPHABET: korean_count,
            ListPattern.CIRCLED_NUMBER: circled_count,
        }

        total_list_items = sum(pattern_counts.values())

        # Need minimum number of list items to classify as LIST format
        if total_list_items < self.MIN_LIST_ITEMS:
            return None

        # Determine the dominant pattern
        max_count = max(pattern_counts.values())

        # Check for mixed patterns (more lenient - multiple patterns present)
        pattern_types_present = sum(1 for count in pattern_counts.values() if count > 0)
        is_mixed = pattern_types_present > 1

        if is_mixed:
            list_pattern = ListPattern.MIXED
        else:
            # Get the pattern with maximum count
            list_pattern = max(pattern_counts, key=pattern_counts.get)

        # Calculate confidence based on list item density
        confidence = 0.6 + min(total_list_items * 0.05, 0.2)

        # Add bonus for clear patterns
        if not is_mixed and total_list_items >= 3:
            confidence += 0.05

        confidence = min(confidence, self.HIGH_CONFIDENCE_THRESHOLD)

        return ClassificationResult(
            format_type=FormatType.LIST,
            confidence=confidence,
            list_pattern=list_pattern,
            indicators={
                "list_pattern": list_pattern.value,
                "numeric_count": numeric_count,
                "korean_count": korean_count,
                "circled_count": circled_count,
                "total_items": total_list_items
            }
        )

    def _check_guideline_format(self, content: str) -> Optional[ClassificationResult]:
        """
        Check if content is in guideline format.

        Guideline format is continuous prose without clear structural markers.

        Args:
            content: Content to check

        Returns:
            ClassificationResult if guideline format detected, None otherwise
        """
        # Remove whitespace and check content length
        cleaned_content = content.strip()

        if len(cleaned_content) < 20:
            # Too short to be guideline - need more content
            return None

        # Check for characteristics of continuous prose
        lines = [line.strip() for line in content.split('\n') if line.strip()]

        # Count lines that look like prose (not lists or articles)
        prose_lines = 0
        for line in lines:
            if self._is_prose_line(line):
                prose_lines += 1

        prose_ratio = prose_lines / len(lines) if lines else 0

        # If most lines are prose, classify as guideline
        if prose_ratio >= 0.5 or len(lines) <= 2:  # Also handle short multi-line content
            # Calculate confidence based on content characteristics
            confidence = self.LOW_CONFIDENCE_THRESHOLD + (prose_ratio * 0.35)
            confidence = min(confidence, self.MEDIUM_CONFIDENCE_THRESHOLD + 0.1)

            return ClassificationResult(
                format_type=FormatType.GUIDELINE,
                confidence=confidence,
                indicators={
                    "prose_ratio": prose_ratio,
                    "total_lines": len(lines),
                    "prose_lines": prose_lines
                }
            )

        return None

    def _is_prose_line(self, line: str) -> bool:
        """
        Check if a line looks like prose (not a list or article marker).

        Args:
            line: Line to check

        Returns:
            True if line appears to be prose
        """
        # Skip empty lines
        if not line:
            return False

        # Check if line starts with article or list markers
        if self.ARTICLE_PATTERN.match(line):
            return False

        if self.NUMERIC_LIST_PATTERN.match(line):
            return False

        if self.KOREAN_LIST_PATTERN.match(line):
            return False

        if self.CIRCLED_NUMBER_PATTERN.match(line):
            return False

        # Check line length - prose lines are typically longer
        return len(line) > 15
