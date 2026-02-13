"""
Regulation Article Extractor

Extract individual articles with full hierarchy preservation from
regulation text, including paragraphs (①), items (1.), and subitems (가.).

This module handles the complex hierarchical structure of Korean regulations,
preserving the nested article, paragraph, item, and subitem relationships.
"""
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RegulationArticleExtractor:
    """
    Extract regulation articles with full hierarchy preservation.

    Handles:
    - Article identification (제N조 patterns)
    - Article title and content extraction
    - Paragraph structures (①-⑮)
    - Item lists (1., 2., 3.)
    - Subitem lists (가., 나., 다.)

    Hierarchy:
    Article (조)
      ├─ Paragraphs (①, ②, ③...)
      ├─ Items (1., 2., 3...)
      │   └─ Subitems (가., 나., 다...)
      └─ Content body
    """

    # Korean circled numbers for paragraphs
    PARAGRAPH_NUMBERS = [
        '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩',
        '⑪', '⑫', '⑬', '⑭', '⑮',
    ]

    # Korean alphabet for subitems
    SUBITEM_PREFIXES = [
        '가.', '나.', '다.', '라.', '마.', '바.', '사.', '아.', '자.',
        '차.', '카.', '타.', '파.', '하.',
    ]

    def __init__(self):
        """Initialize the article extractor."""
        # Pattern: 제N조[의M] [(title)] [inline_content]
        # Original pattern for standard format
        self.article_pattern = re.compile(
            r'^제\s*(\d+)조(의(\d+))?\s*(?:\((.*?)\))?\s*(.*)$'
        )
        # Additional pattern for markdown format: ## 제N조...
        self.article_pattern_markdown = re.compile(
            r'^##\s+제\s*(\d+)조(의(\d+))?\s*(?:\((.*?)\))?\s*(.*)$'
        )
        self.paragraph_pattern = re.compile(r'^([①-⑮])\s*(.*)$')
        self.item_pattern = re.compile(r'^(\d+)\.\s*(.*)$')
        # Build subitem pattern from class constant
        subitem_chars = ''.join([p[0] for p in self.SUBITEM_PREFIXES])
        self.subitem_pattern = re.compile(
            r'^([' + subitem_chars + r'])\)\s*(.*)$'
        )

    def extract_article(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract article structure from text.

        Args:
            text: Article heading and content text.

        Returns:
            Article dictionary with parsed structure.
        """
        # First line should be article heading
        lines = text.strip().split('\n')
        if not lines:
            return None

        heading = lines[0].strip()
        content = '\n'.join(lines[1:]) if len(lines) > 1 else ""

        # Parse article heading - try markdown pattern first, then standard
        match = self.article_pattern_markdown.match(heading)
        if not match:
            match = self.article_pattern.match(heading)

        if not match:
            return None

        article_no_main = match.group(1)  # Main number
        article_no_suffix = match.group(3) or ""  # Suffix (2 from 의2)
        article_title = match.group(4) or ""  # Title in parentheses
        article_inline = match.group(5) or ""  # Inline content

        # Format article number
        article_no = f"제{article_no_main}조"
        if article_no_suffix:
            article_no += f"의{article_no_suffix}"

        # Parse content for hierarchy
        paragraphs, items, subitems, content_body = self._parse_content(content)

        # Determine title (prefer parenthesized title, fallback to inline)
        # If both are empty, use empty string
        title = article_title if article_title else article_inline

        return {
            "article_no": article_no,
            "title": title,
            "content": content_body,
            "paragraphs": paragraphs,
            "items": items,
            "subitems": subitems,
        }

    def _parse_content(
        self, content: str
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        Parse article content for hierarchical structure.

        Args:
            content: Article content text.

        Returns:
            Tuple of (paragraphs, items, subitems, content_body).
        """
        paragraphs: List[Dict[str, Any]] = []
        items: List[Dict[str, Any]] = []
        subitems: List[Dict[str, Any]] = []
        content_lines: List[str] = []

        lines = content.strip().split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # Check for paragraph (①)
            para_match = self.paragraph_pattern.match(line)
            if para_match:
                para_num = para_match.group(1)
                para_text = para_match.group(2)

                # Look ahead for nested items/subitems
                para_content_lines = []
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()

                    # Stop if new paragraph
                    if self.paragraph_pattern.match(next_line):
                        break

                    # Check for nested item
                    item_match = self.item_pattern.match(next_line)
                    if item_match:
                        # Parse nested items
                        para_items, i = self._parse_items(lines, i)
                        if para_items:
                            para_content_lines.append(f"[{len(para_items)} items]")
                        continue

                    para_content_lines.append(next_line)
                    i += 1

                paragraphs.append({
                    "number": para_num,
                    "text": para_text,
                    "content": '\n'.join(para_content_lines),
                })
                continue

            # Check for item (1.)
            item_match = self.item_pattern.match(line)
            if item_match:
                parsed_items, i = self._parse_items(lines, i)
                items.extend(parsed_items)
                continue

            # Regular content line
            content_lines.append(line)
            i += 1

        content_body = '\n'.join(content_lines).strip()

        return paragraphs, items, subitems, content_body

    def _parse_items(
        self, lines: List[str], start_idx: int
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Parse items and nested subitems.

        Args:
            lines: All content lines.
            start_idx: Starting index.

        Returns:
            Tuple of (items_list, new_index).
        """
        items: List[Dict[str, Any]] = []
        i = start_idx

        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # Stop if not an item
            item_match = self.item_pattern.match(line)
            if not item_match:
                break

            item_num = item_match.group(1)
            item_text = item_match.group(2)

            # Look ahead for subitems
            subitems: List[Dict[str, Any]] = []
            item_content_lines = []
            i += 1

            while i < len(lines):
                next_line = lines[i].strip()

                # Stop if new item
                if self.item_pattern.match(next_line):
                    break

                # Check for subitem (가.)
                subitem_match = self.subitem_pattern.match(next_line)
                if subitem_match:
                    subitem_num = subitem_match.group(1)
                    subitem_text = subitem_match.group(2)

                    subitems.append({
                        "number": subitem_num,
                        "text": subitem_text,
                    })
                    i += 1
                    continue

                item_content_lines.append(next_line)
                i += 1

            items.append({
                "number": item_num,
                "text": item_text,
                "content": '\n'.join(item_content_lines),
                "subitems": subitems,
            })

        return items, i

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw text string.

        Returns:
            Cleaned text string.
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common artifacts
        text = text.replace('|', '')

        # Trim whitespace
        text = text.strip()

        return text

    def parse_paragraphs(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse paragraphs from content.

        Args:
            content: Content text.

        Returns:
            List of paragraph dictionaries.
        """
        paragraphs: List[Dict[str, Any]] = []

        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            match = self.paragraph_pattern.match(line)
            if match:
                paragraphs.append({
                    "number": match.group(1),
                    "text": match.group(2),
                })

        return paragraphs

    def parse_items(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse items from content.

        Args:
            content: Content text.

        Returns:
            List of item dictionaries.
        """
        items: List[Dict[str, Any]] = []

        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            match = self.item_pattern.match(line)
            if match:
                items.append({
                    "number": match.group(1),
                    "text": match.group(2),
                })

        return items

    def parse_subitems(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse subitems from content.

        Args:
            content: Content text.

        Returns:
            List of subitem dictionaries.
        """
        subitems: List[Dict[str, Any]] = []

        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            match = self.subitem_pattern.match(line)
            if match:
                subitems.append({
                    "number": match.group(1),
                    "text": match.group(2),
                })

        return subitems


class ParsingReportGenerator:
    """Generate detailed parsing reports for quality validation."""

    def __init__(self):
        """Initialize the report generator."""
        self.success_count = 0
        self.failure_count = 0
        self.failures: List[Dict[str, Any]] = []

    def track_success(self, regulation_id: str, article_count: int) -> None:
        """
        Track successful parsing.

        Args:
            regulation_id: Regulation identifier.
            article_count: Number of articles parsed.
        """
        self.success_count += 1
        logger.info(
            f"Parsed {regulation_id} with {article_count} articles"
        )

    def track_failure(
        self, regulation_id: str, error: str, context: Dict[str, Any]
    ) -> None:
        """
        Track parsing failure.

        Args:
            regulation_id: Regulation identifier.
            error: Error message.
            context: Additional context for debugging.
        """
        self.failure_count += 1
        self.failures.append({
            "regulation_id": regulation_id,
            "error": error,
            "context": context,
        })
        logger.error(f"Failed to parse {regulation_id}: {error}")

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate summary report.

        Returns:
            Report dictionary with statistics.
        """
        total = self.success_count + self.failure_count
        success_rate = (
            self.success_count / total * 100 if total > 0 else 0
        )

        return {
            "total_regulations": total,
            "successfully_parsed": self.success_count,
            "failed_regulations": self.failure_count,
            "success_rate": success_rate,
            "failures": self.failures[:10],  # Limit to 10 for readability
        }

    def validate_completeness(
        self, toc_count: int, parsed_count: int
    ) -> tuple[bool, str]:
        """
        Validate completeness against TOC.

        Args:
            toc_count: Expected count from TOC.
            parsed_count: Actual parsed count.

        Returns:
            Tuple of (is_complete, message).
        """
        if parsed_count == toc_count:
            return True, f"Complete: {parsed_count}/{toc_count} regulations"

        missing = toc_count - parsed_count
        return False, f"Incomplete: {parsed_count}/{toc_count} ({missing} missing)"
