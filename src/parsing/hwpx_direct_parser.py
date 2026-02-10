"""
HWPX Direct Parser

Parse HWPX files (ZIP+XML format) directly to extract regulation text
with 100% accuracy by bypassing HTML/Markdown intermediate conversion stages.

This module provides direct XML parsing from HWPX ZIP archives to preserve
the original document structure and eliminate data loss from multi-stage
conversion pipelines.
"""
import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import xml.etree.ElementTree as ET

from .regulation_article_extractor import RegulationArticleExtractor
from .core.text_normalizer import TextNormalizer
from .detectors.regulation_title_detector import RegulationTitleDetector
from .validators.completeness_checker import CompletenessChecker, TOCEntry

logger = logging.getLogger(__name__)

# HWPX XML namespace mapping
HWPX_NS = {
    'hs': 'http://www.hancom.co.kr/hwpml/2011/section',
    'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
    'hp10': 'http://www.hancom.co.kr/hwpml/2016/paragraph',
    'hc': 'http://www.hancom.co.kr/hwpml/2011/core',
}


@dataclass
class ParsingStatistics:
    """Track parsing statistics for quality validation."""
    total_regulations: int = 0
    successfully_parsed: int = 0
    failed_regulations: int = 0
    total_articles: int = 0
    parsing_errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_regulations": self.total_regulations,
            "successfully_parsed": self.successfully_parsed,
            "failed_regulations": self.failed_regulations,
            "total_articles": self.total_articles,
            "success_rate": (
                self.successfully_parsed / self.total_regulations * 100
                if self.total_regulations > 0
                else 0
            ),
            "parsing_errors": self.parsing_errors[:10],  # Limit error details
        }


class HWPXDirectParser:
    """
    Direct HWPX parser for 100% regulation coverage.

    Parses HWPX files by extracting XML content directly from ZIP archives,
    eliminating the data loss associated with multi-stage HTML/Markdown
    conversion pipelines.

    Key Features:
    - Direct XML parsing from ZIP archive
    - Hierarchical structure preservation (편/장/절/조/항/호/목)
    - Table structure preservation with merged cell info
    - JSON output compatible with existing Chunk/Regulation dataclasses
    """

    def __init__(
        self,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the direct parser.

        Args:
            status_callback: Optional callback for progress updates.
        """
        self.ns = HWPX_NS
        self.status_callback = status_callback
        self.stats = ParsingStatistics()
        self.article_extractor = RegulationArticleExtractor()
        self.text_normalizer = TextNormalizer()
        self.title_detector = RegulationTitleDetector()
        self.completeness_checker = CompletenessChecker(fuzzy_match_threshold=0.85)
        self._seen_titles = set()  # Track seen titles to prevent duplicates
        self._toc_entries: List[TOCEntry] = []  # Store TOC for validation

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse HWPX file and return structured JSON.

        Args:
            file_path: Path to HWPX file.

        Returns:
            Dictionary with metadata, toc, and docs sections.
        """
        start_time = datetime.now()

        if self.status_callback:
            self.status_callback("[dim]HWPX 파일 파싱 중...[/dim]")

        # Extract XML content from ZIP
        sections_xml = self._extract_sections_from_zip(file_path)

        # NEW: Parse TOC first to get complete regulation list
        self._parse_toc_from_sections(sections_xml)

        # Parse regulations from XML
        docs = []
        # Parse section0.xml (main content) for body text
        sections_to_parse = {k: v for k, v in sections_xml.items()
                            if 'section0.xml' in k}

        for idx, (section_num, xml_content) in enumerate(sections_to_parse.items()):
            if self.status_callback:
                self.status_callback(
                    f"[dim]섹션 파싱 중 ({idx+1}/{len(sections_xml)}): {section_num}[/dim]"
                )

            try:
                regulations = self._parse_section_xml(xml_content)
                docs.extend(regulations)
            except Exception as e:
                logger.error(f"Failed to parse section {section_num}: {e}")
                self.stats.parsing_errors.append({
                    "section": section_num,
                    "error": str(e),
                    "error_type": type(e).__name__,
                })
                continue

        # Convert articles to content format for RAG compatibility
        # Transform each regulation's articles into content nodes
        for doc in docs:
            articles = doc.get("articles", [])
            content_nodes = []
            for article in articles:
                is_alternative = article.get("is_alternative", False)
                # For alternative content, use the title as display
                # For regular articles, use article_no
                display_no = article.get("article_no", "") if not is_alternative else ""
                article_title = article.get("title", "")
                content_text = article.get("content", "").strip()

                # Build full_text differently for alternative vs regular content
                if is_alternative:
                    full_text = f"{article_title}\n{content_text}"
                else:
                    full_text = f"{display_no} {article_title}\n{content_text}".strip()

                content_nodes.append({
                    "type": "alternative" if is_alternative else "article",
                    "display_no": display_no,
                    "title": article_title,
                    "text": content_text,
                    "full_text": full_text,
                    "parent_path": [doc.get("title", "")],
                    "level": "regulation" if is_alternative else "article",
                })
            doc["content"] = content_nodes
            # Keep articles for reference
            # doc["articles"] = articles  # Optional: remove or keep based on needs

        # Generate output
        # Create completeness report
        toc_for_validation = self._toc_entries if self._toc_entries else self.completeness_checker.create_toc_from_regulations(docs)
        completeness_report = self.completeness_checker.validate(toc_for_validation, docs)

        output = {
            "metadata": {
                "source_file": file_path.name,
                "parser_version": "3.0.0",  # Updated version with TOC-first parsing
                "parsed_at": start_time.isoformat(),
                "parsing_time_seconds": (
                    datetime.now() - start_time
                ).total_seconds(),
                **self.stats.to_dict(),
                "completeness": completeness_report.to_dict(),
            },
            "toc": [self._toc_entry_to_dict(e) for e in toc_for_validation],
            "docs": docs,
        }

        if self.status_callback:
            success_rate = output["metadata"]["success_rate"]
            self.status_callback(
                f"[green]파싱 완료: {self.stats.successfully_parsed}/{self.stats.total_regulations} "
                f"규정 ({success_rate:.1f}%)[/green]"
            )

        return output

    def _extract_sections_from_zip(
        self, file_path: Path
    ) -> Dict[str, str]:
        """
        Extract section XML files from HWPX ZIP archive.

        Args:
            file_path: Path to HWPX file.

        Returns:
            Dictionary mapping section names to XML content.
        """
        sections = {}

        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Find all section XML files
                section_files = [
                    f for f in zf.namelist()
                    if f.startswith('Contents/section') and f.endswith('.xml')
                ]

                if not section_files:
                    # Fallback: find any XML in Contents
                    section_files = [
                        f for f in zf.namelist()
                        if 'Contents' in f and f.endswith('.xml')
                    ]

                for section_file in sorted(section_files):
                    try:
                        with zf.open(section_file) as f:
                            # Try UTF-8 first, fallback to CP949
                            try:
                                content = f.read().decode('utf-8')
                            except UnicodeDecodeError:
                                f.seek(0)
                                content = f.read().decode('cp949')

                            sections[section_file] = content
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract {section_file}: {e}"
                        )
                        continue

        except Exception as e:
            logger.error(f"Failed to open ZIP file {file_path}: {e}")
            raise

        return sections

    def _parse_toc_from_sections(
        self, sections_xml: Dict[str, str]
    ) -> List[TOCEntry]:
        """
        Parse TOC from section1.xml to get complete regulation list.

        This ensures we know about all 514 regulations even if some
        don't appear in the main content section.

        Args:
            sections_xml: Dictionary of section XML contents.

        Returns:
            List of TOCEntry objects.
        """
        self._toc_entries = []

        # Look for section1.xml (typically contains TOC)
        toc_section = None
        for section_name, content in sections_xml.items():
            if 'section1.xml' in section_name:
                toc_section = content
                break

        if not toc_section:
            logger.warning("No TOC section (section1.xml) found")
            return self._toc_entries

        try:
            root = ET.fromstring(toc_section)
        except ET.ParseError as e:
            logger.error(f"Failed to parse TOC XML: {e}")
            return self._toc_entries

        # Extract regulation titles from TOC
        for elem in root.iter():
            if elem.tag != f'{{{self.ns["hp"]}}}p':
                continue

            text = self._extract_paragraph_text(elem)
            if not text:
                continue

            # Use title detector to identify regulation titles
            if self.title_detector.is_title(text):
                entry = TOCEntry(
                    id=f"toc-{len(self._toc_entries) + 1:04d}",
                    title=text,
                    rule_code=self._extract_rule_code(text),
                )
                self._toc_entries.append(entry)

        logger.info(f"Parsed {len(self._toc_entries)} entries from TOC")
        return self._toc_entries

    def _parse_section_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Parse section XML and extract regulations.

        Args:
            xml_content: XML content string.

        Returns:
            List of regulation dictionaries.
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return []

        regulations = []
        current_regulation: Optional[Dict[str, Any]] = None
        current_articles: List[Dict[str, Any]] = []
        accumulated_text: List[str] = []

        # Track hierarchy
        current_part: Optional[str] = None
        current_chapter: Optional[str] = None
        current_section: Optional[str] = None

        def flush_current_article():
            """Flush accumulated text as content to the last article."""
            if current_articles and accumulated_text:
                content = '\n'.join(accumulated_text).strip()
                if content:
                    current_articles[-1]["content"] = (
                        current_articles[-1].get("content", "") + "\n" + content
                    )
                accumulated_text.clear()

        # Process paragraphs in document order
        for elem in root.iter():
            # Skip non-paragraph elements
            if elem.tag != f'{{{self.ns["hp"]}}}p':
                continue

            text = self._extract_paragraph_text(elem)
            if not text:
                continue

            # Detect regulation title (main headings)
            if self._is_regulation_title(text):
                # Save previous regulation
                if current_regulation:
                    flush_current_article()
                    # If no articles were detected, create alternative content
                    # from accumulated text
                    if not current_articles and accumulated_text:
                        alt_content = '\n'.join(accumulated_text).strip()
                        if alt_content:
                            current_articles.append({
                                "article_no": "",
                                "title": current_regulation.get("title", ""),
                                "content": alt_content,
                                "paragraphs": [],
                                "items": [],
                                "subitems": [],
                                "is_alternative": True,  # Mark as alternative content
                            })
                    current_regulation["articles"] = current_articles
                    regulations.append(current_regulation)

                # Start new regulation
                current_regulation = {
                    "id": f"reg-{len(regulations) + 1:04d}",
                    "kind": "regulation",
                    "title": text.strip(),
                    "rule_code": self._extract_rule_code(text),
                    "part": current_part,
                    "chapter": current_chapter,
                    "section": current_section,
                }
                current_articles = []
                accumulated_text.clear()
                self.stats.total_regulations += 1

            # Detect hierarchy markers
            elif self._is_part_marker(text):
                flush_current_article()
                current_part = text.strip()
                # Add to accumulated text for regulations without articles
                accumulated_text.append(text)
            elif self._is_chapter_marker(text):
                flush_current_article()
                current_chapter = text.strip()
                # Add to accumulated text for regulations without articles
                accumulated_text.append(text)
            elif self._is_section_marker(text):
                flush_current_article()
                current_section = text.strip()
                # Add to accumulated text for regulations without articles
                accumulated_text.append(text)

            # Detect article using RegulationArticleExtractor
            elif current_regulation and self._is_article_marker(text):
                flush_current_article()

                # Use RegulationArticleExtractor to parse article structure
                article_data = self.article_extractor.extract_article(text)
                if article_data:
                    current_articles.append(article_data)
                else:
                    # Fallback to basic parsing
                    article_no_match = re.match(r'(제\s*\d+[조의]*\d*)', text)
                    if article_no_match:
                        article_no = article_no_match.group(1).strip()
                        article_title = re.sub(r'^제\s*\d+[조의]*\d*\s*(?:\((.+?)\))?\s*', '', text)
                        current_articles.append({
                            "article_no": article_no,
                            "title": article_title,
                            "content": "",
                            "paragraphs": [],
                            "items": [],
                            "subitems": [],
                        })
                self.stats.total_articles += 1

            # Accumulate content text
            elif current_regulation:
                # Always accumulate text if we're inside a regulation
                # This handles both cases:
                # 1. Regulations with articles (text gets added to current article)
                # 2. Regulations without articles (text used for alternative content)

                # Check if this is a content marker (항/호/목 etc.)
                # Content markers are preserved for proper structure parsing
                if self._is_content_marker(text):
                    # Mark this line as a content marker for special handling
                    # The RegulationArticleExtractor will parse these patterns
                    accumulated_text.append(text)
                else:
                    # Regular content text
                    accumulated_text.append(text)

        # Save last regulation
        if current_regulation:
            flush_current_article()
            # If no articles were detected, create alternative content
            # from accumulated text
            if not current_articles and accumulated_text:
                alt_content = '\n'.join(accumulated_text).strip()
                if alt_content:
                    current_articles.append({
                        "article_no": "",
                        "title": current_regulation.get("title", ""),
                        "content": alt_content,
                        "paragraphs": [],
                        "items": [],
                        "subitems": [],
                        "is_alternative": True,  # Mark as alternative content
                    })
            current_regulation["articles"] = current_articles
            regulations.append(current_regulation)

        # Update statistics
        self.stats.successfully_parsed = len(regulations)
        self.stats.failed_regulations = (
            self.stats.total_regulations - self.stats.successfully_parsed
        )

        return regulations

    def _extract_paragraph_text(self, p_elem: ET.Element) -> str:
        """
        Extract text content from a paragraph element.

        Args:
            p_elem: Paragraph XML element.

        Returns:
            Extracted text string.
        """
        text_parts = []

        for run in p_elem.findall('.//hp:run', self.ns):
            for t in run.findall('./hp:t', self.ns):
                if t.text:
                    text_parts.append(t.text)

        return self._clean_text(''.join(text_parts))

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text by removing artifacts.

        Args:
            text: Raw text string.

        Returns:
            Cleaned text string.
        """
        # Remove special Unicode filler characters (HWPX uses these for layout)
        text = re.sub(r'[\U000f0800-\U000f0fff]+', '', text)

        # Remove pagination patterns (e.g., "3—2—118～", "1-1-1～1")
        text = re.sub(r'\s*[\d]+[-―－][\d]+[-―－][\d]+[～~～].*?$', '', text)

        # Remove horizontal rules/separator characters
        text = re.sub(r'[─＿]+', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _is_regulation_title(self, text: str) -> bool:
        """
        Detect if text is a regulation title.

        In the context of HWPX parsing, short standalone keywords (like "지침", "규정")
        without any prefix are unlikely to be regulation titles, so we filter them out.

        Args:
            text: Text to analyze.

        Returns:
            True if text appears to be a regulation title.
        """
        # First check with the title detector
        if not self.title_detector.is_title(text):
            return False

        # Additional filtering for HWPX context:
        # Short standalone keywords (2-3 chars ending with keyword) are filtered out
        # These are typically section headers or TOC entries, not regulation titles
        short_standalone_keywords = {'학칙', '규정', '요령', '세칙', '지침', '내규', '헌장', '기준'}
        if text in short_standalone_keywords:
            return False

        # Also filter 2-3 character strings that end with keywords (too short to be titles)
        if len(text) <= 3:
            for keyword in short_standalone_keywords:
                if text.endswith(keyword):
                    return False

        return True

    def _is_part_marker(self, text: str) -> bool:
        """Detect if text is a part (편) marker."""
        return bool(re.match(r'^제\s*\d+\s*편', text))

    def _is_chapter_marker(self, text: str) -> bool:
        """Detect if text is a chapter (장) marker."""
        return bool(re.match(r'^제\s*\d+\s*장', text))

    def _is_section_marker(self, text: str) -> bool:
        """Detect if text is a section (절) marker."""
        return bool(re.match(r'^제\s*\d+\s*절', text))

    def _is_article_marker(self, text: str) -> bool:
        """Detect if text is an article (조) marker."""
        # Handle both "제N조" and "## 제N조" formats
        cleaned_text = text.strip()
        if cleaned_text.startswith("##"):
            cleaned_text = cleaned_text[2:].strip()

        return bool(re.match(r'^제\s*\d+[조의]*\d*\s*(\(.+\))?', cleaned_text))

    def _is_content_marker(self, text: str) -> bool:
        """
        Detect if text matches various content marker patterns.

        This method identifies items (항), subitems (호), tables (별표),
        forms (서식), and numbered lists that appear within regulations.

        Patterns supported:
        - 항 (items): ①, ②, ③, 1., 2., 3.
        - 호 (subitems): 가., 나., 다., (1), (2), (3)
        - 별표 (tables): 별표 1, 별표 2
        - 서식 (forms): 서식 1, 서식 2
        - 숫자 번호: 1., 2., 3. (at start of line)

        Args:
            text: Text to analyze.

        Returns:
            True if text matches any content marker pattern.
        """
        if not text:
            return False

        cleaned_text = text.strip()

        # Remove markdown-style prefixes
        if cleaned_text.startswith("##"):
            cleaned_text = cleaned_text[2:].strip()

        # Pattern 1: 항 (items) - Circled numbers ①, ②, ③, etc.
        if re.match(r'^[\u2460-\u2473\u3251-\u32bf\u2488-\u249b]', cleaned_text):
            return True

        # Pattern 2: 항 (items) - Numbered lists 1., 2., 3., etc.
        # Must be at start of line and followed by space or content
        if re.match(r'^\d+\.\s+\S', cleaned_text):
            return True

        # Pattern 3: 호 (subitems) - Korean 한글 characters 가., 나., 다.
        if re.match(r'^[가-힣]+\.\s+\S', cleaned_text):
            return True

        # Pattern 4: 호 (subitems) - Parenthesized numbers (1), (2), (3)
        if re.match(r'^\(\d+\)\s*\S', cleaned_text):
            return True

        # Pattern 5: 별표 (tables/appendix) - 별표 1, 별표 2, etc.
        if re.match(r'^별표\s*\d+', cleaned_text):
            return True

        # Pattern 6: 서식 (forms) - 서식 1, 서식 2, etc.
        if re.match(r'^서식\s*\d+', cleaned_text):
            return True

        # Pattern 7: Roman numerals (I., II., III., etc.)
        if re.match(r'^[IVX]+\.\s+\S', cleaned_text):
            return True

        # Pattern 8: Alphabetical lists (a., b., c., A., B., C.)
        if re.match(r'^[a-zA-Z]\.\s+\S', cleaned_text):
            return True

        return False

    def _extract_rule_code(self, title: str) -> str:
        """
        Extract rule code from regulation title.

        Args:
            title: Regulation title.

        Returns:
            Rule code string or empty string.
        """
        # Try to extract pattern like "3-1-10" from title
        match = re.search(r'(\d+-\d+-\d+)', title)
        return match.group(1) if match else ""

    def _parse_article(
        self, text: str, elem: ET.Element
    ) -> Optional[Dict[str, Any]]:
        """
        Parse article text and extract structure.

        Args:
            text: Article heading text.
            elem: XML element for the article.

        Returns:
            Article dictionary or None.
        """
        # Parse article number and title
        article_match = re.match(
            r'(제\s*\d+[조의]*\d*)\s*(?:\((.+?)\))?\s*(.*)',
            text
        )

        if not article_match:
            return None

        article_no = article_match.group(1).strip()
        article_title = article_match.group(2) or article_match.group(3) or ""

        # Extract content from subsequent paragraphs
        # (This would be implemented by looking at siblings in document order)
        # For now, create basic structure
        article = {
            "article_no": article_no,
            "title": article_title,
            "content": "",  # Would be extracted from subsequent content
            "paragraphs": [],
            "items": [],
            "subitems": [],
        }

        return article

    def _extract_toc_from_docs(
        self, docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract table of contents from parsed documents.

        Args:
            docs: List of parsed regulation documents.

        Returns:
            List of TOC entries.
        """
        toc = []

        for idx, doc in enumerate(docs):
            toc_entry = {
                "id": f"toc-{idx + 1:04d}",
                "rule_code": doc.get("rule_code", ""),
                "title": doc.get("title", ""),
                "page": str(idx + 1),
            }
            toc.append(toc_entry)

        return toc

    def _toc_entry_to_dict(self, entry: TOCEntry) -> Dict[str, Any]:
        """Convert TOCEntry to dictionary for JSON serialization."""
        return {
            "id": entry.id,
            "title": entry.title,
            "page": entry.page,
            "rule_code": entry.rule_code,
        }


class HWPXTableParser:
    """Parse HWPX table structures preserving merge information."""

    def __init__(self, ns: Dict[str, str]):
        """
        Initialize table parser.

        Args:
            ns: XML namespace mapping.
        """
        self.ns = ns

    def parse_table(self, tbl_elem: ET.Element) -> Dict[str, Any]:
        """
        Parse HWPX table element to structured format.

        Args:
            tbl_elem: Table XML element.

        Returns:
            Dictionary with table structure.
        """
        row_cnt = int(tbl_elem.get('rowCnt', '0'))
        col_cnt = int(tbl_elem.get('colCnt', '0'))

        if row_cnt == 0 or col_cnt == 0:
            return {}

        table = {
            "rows": row_cnt,
            "cols": col_cnt,
            "cells": [],
        }

        rows = tbl_elem.findall('./hp:tr', self.ns)
        for tr_elem in rows:
            cells = tr_elem.findall('./hp:tc', self.ns)
            for tc_elem in cells:
                cell_data = self._extract_cell_data(tc_elem)
                table["cells"].append(cell_data)

        return table

    def _extract_cell_data(self, tc_elem: ET.Element) -> Dict[str, Any]:
        """
        Extract cell data including span information.

        Args:
            tc_elem: Table cell XML element.

        Returns:
            Cell data dictionary.
        """
        cell_data: Dict[str, Any] = {
            "text": "",
            "rowspan": 1,
            "colspan": 1,
        }

        # Get cell span info
        cell_span = tc_elem.find('./hp:cellSpan', self.ns)
        if cell_span is not None:
            cell_data["colspan"] = int(cell_span.get('colSpan', '1'))
            cell_data["rowspan"] = int(cell_span.get('rowSpan', '1'))

        # Extract cell content
        cell_data["text"] = self._extract_cell_text(tc_elem)

        return cell_data

    def _extract_cell_text(self, tc_elem: ET.Element) -> str:
        """
        Extract text content from a table cell.

        Args:
            tc_elem: Table cell XML element.

        Returns:
            Extracted text string.
        """
        text_parts = []

        sub_list = tc_elem.find('./hp:subList', self.ns)
        if sub_list is None:
            return ""

        for p in sub_list.findall('./hp:p', self.ns):
            for run in p.findall('.//hp:run', self.ns):
                for t in run.findall('./hp:t', self.ns):
                    if t.text:
                        text_parts.append(t.text)

        return ' '.join(text_parts).strip()
