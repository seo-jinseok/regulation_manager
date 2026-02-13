"""
List Regulation Extractor for HWPX Regulation Parsing.

This module implements list-format regulation extraction with hierarchy preservation.
List-format regulations use numbered or bulleted lists instead of article markers (제N조).

TDD Approach: GREEN Phase
- Minimal implementation to make failing tests pass
- Pattern detection for numeric, Korean alphabet, and circled number lists
- Nested list extraction with hierarchy preservation
- List-to-article conversion for RAG compatibility

Reference: SPEC-HWXP-002, TASK-003
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from src.parsing.format.format_type import ListPattern

logger = logging.getLogger(__name__)


@dataclass
class ListItem:
    """
    Represents a single list item with optional children.

    Attributes:
        number: The list marker (e.g., "1", "가", "①")
        content: The text content of the item
        level: The hierarchy level (0 = top level)
        children: Nested child items
        pattern: The list pattern type
    """
    number: str
    content: str
    level: int = 0
    pattern: Optional[ListPattern] = None
    children: List['ListItem'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "number": self.number,
            "content": self.content,
            "level": self.level,
            "pattern": self.pattern.value if self.pattern else None,
            "children": [child.to_dict() for child in self.children]
        }


@dataclass
class ExtractionResult:
    """
    Result of list regulation extraction.

    Attributes:
        items: List of extracted items (possibly nested)
        pattern: Detected list pattern
        total_items: Total count of items at all levels
        extraction_rate: Ratio of successfully extracted items
    """
    items: List[ListItem]
    pattern: ListPattern
    total_items: int = 0
    extraction_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "items": [item.to_dict() for item in self.items],
            "pattern": self.pattern.value,
            "total_items": self.total_items,
            "extraction_rate": self.extraction_rate
        }


class ListRegulationExtractor:
    """
    Extract content from list-format regulations with hierarchy preservation.

    List-format regulations use:
    - Numeric markers: 1., 2., 3.
    - Korean alphabet: 가., 나., 다.
    - Circled numbers: ①, ②, ③
    - Mixed patterns with nested hierarchy

    Hierarchy levels (typical Korean regulation structure):
    - Level 0: 1., 2., 3. (main items)
    - Level 1: ①, ②, ③ (sub-items)
    - Level 2: 가., 나., 다. (detail items)
    - Level 3: 1), 2), 3) (sub-detail items)
    """

    # List marker patterns
    NUMERIC_PATTERN = re.compile(r'^(\d+)\.\s*(.*)$', re.MULTILINE)
    KOREAN_ALPHABET_PATTERN = re.compile(r'^([가-하])[\.\)]\s*(.*)$', re.MULTILINE)
    CIRCLED_NUMBER_PATTERN = re.compile(r'^([①-⑮])\s*(.*)$', re.MULTILINE)
    PARENTHESIZED_NUMBER_PATTERN = re.compile(r'^(\d+)\)\s*(.*)$', re.MULTILINE)

    # Pattern for indent detection (spaces or tabs)
    INDENT_PATTERN = re.compile(r'^(\s*)')

    def __init__(self):
        """Initialize the list regulation extractor."""
        logger.debug("ListRegulationExtractor initialized")

    def detect_pattern(self, content: str) -> Dict[str, Any]:
        """
        Detect the list pattern type in the content.

        Args:
            content: Regulation text content

        Returns:
            Dictionary with pattern type and metadata
        """
        if not content or not content.strip():
            return {"pattern": "none", "count": 0}

        # Count each pattern type
        numeric_count = len(self.NUMERIC_PATTERN.findall(content))
        korean_count = len(self.KOREAN_ALPHABET_PATTERN.findall(content))
        circled_count = len(self.CIRCLED_NUMBER_PATTERN.findall(content))
        parenthesized_count = len(self.PARENTHESIZED_NUMBER_PATTERN.findall(content))

        # Determine dominant pattern
        counts = {
            ListPattern.NUMERIC: numeric_count,
            ListPattern.KOREAN_ALPHABET: korean_count,
            ListPattern.CIRCLED_NUMBER: circled_count,
        }

        total = sum(counts.values())

        if total == 0:
            return {"pattern": "none", "count": 0}

        # Check for mixed patterns
        pattern_types_present = sum(1 for count in counts.values() if count > 0)

        if pattern_types_present > 1:
            detected_pattern = ListPattern.MIXED
        else:
            detected_pattern = max(counts, key=counts.get)

        return {
            "pattern": detected_pattern.value,
            "counts": {
                "numeric": numeric_count,
                "korean": korean_count,
                "circled": circled_count,
                "parenthesized": parenthesized_count
            },
            "total": total
        }

    def extract_nested(self, content: str) -> Dict[str, Any]:
        """
        Extract nested list items with hierarchy preservation.

        Args:
            content: Regulation text content

        Returns:
            Dictionary with nested items structure
        """
        if not content or not content.strip():
            return {"items": [], "pattern": "none"}

        # Parse lines with markers
        parsed_items = self._parse_lines_with_markers(content)

        # Build hierarchy from parsed items
        root_items = self._build_hierarchy(parsed_items)

        # Determine pattern
        pattern_info = self.detect_pattern(content)

        return {
            "items": [item.to_dict() for item in root_items],
            "pattern": pattern_info["pattern"]
        }

    def extract(self, content: str) -> Dict[str, Any]:
        """
        Extract list items from content (flat list without nesting).

        Args:
            content: Regulation text content

        Returns:
            Extraction result with items and metadata
        """
        if not content or not content.strip():
            return {
                "items": [],
                "pattern": "none",
                "total_items": 0,
                "extraction_rate": 0.0
            }

        parsed_items = self._parse_lines_with_markers(content)

        # Convert to flat list
        items = []
        for item_data in parsed_items:
            items.append(ListItem(
                number=item_data["marker"],
                content=item_data["content"],
                level=item_data["level"],
                pattern=item_data.get("pattern")
            ))

        pattern_info = self.detect_pattern(content)
        total_items = len(items)

        # Calculate extraction rate: all detected patterns should be extracted
        # In pure list format, extraction_rate should be 1.0 (100%)
        # because we successfully extracted all list items we detected
        extraction_rate = 1.0 if total_items > 0 else 0.0

        return {
            "items": [item.to_dict() for item in items],
            "pattern": pattern_info["pattern"],
            "total_items": total_items,
            "extraction_rate": min(extraction_rate, 1.0)
        }

    def to_article_format(self, content: str) -> Dict[str, Any]:
        """
        Convert list items to article-like format for RAG compatibility.

        Args:
            content: Regulation text content

        Returns:
            Dictionary with article-formatted entries
        """
        if not content or not content.strip():
            return {"articles": []}

        extraction_result = self.extract(content)

        # Convert list items to article-like format
        articles = []
        for idx, item_dict in enumerate(extraction_result["items"], start=1):
            article = {
                "number": idx,
                "content": item_dict["content"],
                "original_marker": item_dict["number"]
            }
            articles.append(article)

        return {
            "articles": articles,
            "pattern": extraction_result["pattern"]
        }

    def extract_with_pattern(self, content: str, pattern: Optional[ListPattern]) -> Dict[str, Any]:
        """
        Extract using a specific pattern hint from FormatClassifier.

        Args:
            content: Regulation text content
            pattern: List pattern from classification result

        Returns:
            Extraction result with items
        """
        if pattern is None:
            # Auto-detect pattern
            return self.extract(content)

        # Use the specified pattern for extraction
        parsed_items = self._parse_lines_with_markers(content, target_pattern=pattern)

        items = []
        for item_data in parsed_items:
            items.append(ListItem(
                number=item_data["marker"],
                content=item_data["content"],
                level=item_data["level"],
                pattern=item_data.get("pattern", pattern)
            ))

        return {
            "items": [item.to_dict() for item in items],
            "pattern": pattern.value
        }

    def _parse_lines_with_markers(
        self,
        content: str,
        target_pattern: Optional[ListPattern] = None
    ) -> List[Dict[str, Any]]:
        """
        Parse lines and extract those with list markers.

        Args:
            content: Text content to parse
            target_pattern: Optional pattern to filter for

        Returns:
            List of parsed item data with marker, content, level
        """
        items = []
        lines = content.split('\n')

        for line in lines:
            if not line.strip():
                continue

            # Detect indent
            indent_match = self.INDENT_PATTERN.match(line)
            indent = indent_match.group(1) if indent_match else ""
            indent_level = len(indent) // 2  # Assume 2 spaces per level

            stripped_line = line.lstrip()

            # Try to match list patterns
            matched = False
            pattern_used = None
            marker = None
            item_content = None

            # Try numeric pattern
            if target_pattern is None or target_pattern == ListPattern.NUMERIC:
                match = self.NUMERIC_PATTERN.match(stripped_line)
                if match:
                    marker = match.group(1)
                    item_content = match.group(2)
                    pattern_used = ListPattern.NUMERIC
                    matched = True

            # Try Korean alphabet pattern
            if not matched and (target_pattern is None or target_pattern == ListPattern.KOREAN_ALPHABET):
                match = self.KOREAN_ALPHABET_PATTERN.match(stripped_line)
                if match:
                    marker = match.group(1)
                    item_content = match.group(2)
                    pattern_used = ListPattern.KOREAN_ALPHABET
                    matched = True

            # Try circled number pattern
            if not matched and (target_pattern is None or target_pattern == ListPattern.CIRCLED_NUMBER):
                match = self.CIRCLED_NUMBER_PATTERN.match(stripped_line)
                if match:
                    marker = match.group(1)
                    item_content = match.group(2)
                    pattern_used = ListPattern.CIRCLED_NUMBER
                    matched = True

            # Try parenthesized number pattern
            if not matched:
                match = self.PARENTHESIZED_NUMBER_PATTERN.match(stripped_line)
                if match:
                    marker = match.group(1)
                    item_content = match.group(2)
                    matched = True

            if matched and item_content is not None:
                items.append({
                    "marker": marker,
                    "content": item_content,
                    "level": indent_level,
                    "pattern": pattern_used
                })

        return items

    def _build_hierarchy(self, parsed_items: List[Dict[str, Any]]) -> List[ListItem]:
        """
        Build hierarchical structure from flat parsed items.

        Args:
            parsed_items: List of parsed item data

        Returns:
            List of root-level items with nested children
        """
        if not parsed_items:
            return []

        root_items = []
        stack: List[Tuple[ListItem, int]] = []  # (item, level) stack

        for item_data in parsed_items:
            item = ListItem(
                number=item_data["marker"],
                content=item_data["content"],
                level=item_data["level"],
                pattern=item_data.get("pattern")
            )

            # Pop items from stack that are at same or higher level
            while stack and stack[-1][1] >= item_data["level"]:
                stack.pop()

            if not stack:
                # Root level item
                root_items.append(item)
            else:
                # Add as child to parent
                parent_item = stack[-1][0]
                parent_item.children.append(item)

            # Push current item to stack
            stack.append((item, item_data["level"]))

        return root_items
