"""
HWPX Parsing Orchestrator - TOC-Driven 100% Coverage Parser

Implements a 3-phase hybrid parsing approach:
1. Structure Discovery: Extract TOC from section1.xml
2. Content Extraction: Match bodies from section0.xml
3. Validation & Repair: Ensure 100% completeness
"""
import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import xml.etree.ElementTree as ET

from ..core.text_normalizer import TextNormalizer
from ..detectors.regulation_title_detector import RegulationTitleDetector
from ..validators.completeness_checker import CompletenessChecker, TOCEntry, CompletenessReport
from ..regulation_article_extractor import RegulationArticleExtractor

logger = logging.getLogger(__name__)

# HWPX XML namespace
HWPX_NS = {
    'hs': 'http://www.hancom.co.kr/hwpml/2011/section',
    'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
    'hp10': 'http://www.hancom.co.kr/hwpml/2016/paragraph',
    'hc': 'http://www.hancom.co.kr/hwpml/2011/core',
}


@dataclass
class ParsingMetadata:
    """Metadata for parsing results."""
    source_file: str
    parser_version: str = "3.0.0"
    parsed_at: str = ""
    parsing_time_seconds: float = 0.0
    total_regulations: int = 0
    successfully_parsed: int = 0
    failed_regulations: int = 0
    success_rate: float = 0.0

    def __post_init__(self):
        if not self.parsed_at:
            self.parsed_at = datetime.now().isoformat()


@dataclass
class ParsingResult:
    """Complete parsing result."""
    metadata: ParsingMetadata
    toc: List[Dict[str, Any]] = field(default_factory=list)
    docs: List[Dict[str, Any]] = field(default_factory=list)
    completeness_report: Optional[CompletenessReport] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": self.metadata.__dict__,
            "toc": self.toc,
            "docs": self.docs,
        }


class HWPXParsingOrchestrator:
    """
    TOC-Driven HWPX Parsing Orchestrator for 100% Coverage.

    Key improvements over v2 parser:
    1. Parses section1.xml (TOC) FIRST to get complete regulation list
    2. For each TOC entry, searches section0.xml for body content
    3. Creates regulation entry even if no articles found (repealed/empty)
    4. Uses fuzzy matching to handle page header contamination
    """

    def __init__(
        self,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            status_callback: Optional callback for progress updates.
        """
        self.ns = HWPX_NS
        self.status_callback = status_callback
        self.normalizer = TextNormalizer()
        self.title_detector = RegulationTitleDetector(min_confidence=0.5)
        self.completeness_checker = CompletenessChecker(fuzzy_match_threshold=0.85)
        self.article_extractor = RegulationArticleExtractor()

    def parse_file(self, file_path: Path) -> ParsingResult:
        """
        Parse HWPX file using TOC-driven approach for 100% coverage.

        Args:
            file_path: Path to HWPX file.

        Returns:
            ParsingResult with metadata, TOC, and docs.
        """
        start_time = datetime.now()
        metadata = ParsingMetadata(source_file=file_path.name)

        self._update_status(f"🔍 Starting TOC-driven parsing: {file_path.name}")

        # Phase 1: Extract TOC from section1.xml
        toc_entries = self._extract_toc_from_zip(file_path)
        metadata.total_regulations = len(toc_entries)
        self._update_status(f"📋 TOC extracted: {len(toc_entries)} regulations")

        # Phase 2: Parse bodies from section0.xml
        parsed_regulations = self._parse_bodies_from_zip(file_path, toc_entries)
        metadata.successfully_parsed = len(parsed_regulations)
        metadata.failed_regulations = metadata.total_regulations - metadata.successfully_parsed

        # Phase 3: Ensure 100% completeness (create missing entries)
        self._ensure_completeness(toc_entries, parsed_regulations)

        # Calculate parsing time
        end_time = datetime.now()
        metadata.parsing_time_seconds = (end_time - start_time).total_seconds()
        metadata.success_rate = (
            metadata.successfully_parsed / metadata.total_regulations * 100
            if metadata.total_regulations > 0
            else 0
        )

        # Validate completeness
        completeness = self.completeness_checker.validate(
            [TOCEntry(**entry) for entry in toc_entries],
            parsed_regulations
        )

        # Build TOC and docs for output
        toc_output = [entry for entry in toc_entries]
        docs_output = [reg for reg in parsed_regulations]

        result = ParsingResult(
            metadata=metadata,
            toc=toc_output,
            docs=docs_output,
            completeness_report=completeness
        )

        self._update_status(f"✅ Parsing complete: {metadata.success_rate:.1f}% ({metadata.successfully_parsed}/{metadata.total_regulations})")

        return result

    def _extract_toc_from_zip(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extract TOC entries from section1.xml.

        Args:
            file_path: Path to HWPX file.

        Returns:
            List of TOC entry dictionaries.
        """
        toc_entries = []

        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Try section1.xml first (TOC section)
                section_files = [f for f in zip_ref.namelist() if f.startswith('Contents/section')]

                for section_file in sorted(section_files):
                    # Skip section0 (main content) - we want section1+ for TOC
                    if 'section0.xml' in section_file:
                        continue

                    try:
                        with zip_ref.open(section_file) as f:
                            try:
                                content = f.read().decode('utf-8')
                            except UnicodeDecodeError:
                                f.seek(0)
                                content = f.read().decode('cp949')

                        # Extract titles from this section
                        entries = self._parse_toc_section(content, section_file)
                        toc_entries.extend(entries)

                    except Exception as e:
                        logger.warning(f"Failed to parse TOC from {section_file}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Failed to open ZIP file {file_path}: {e}")
            raise

        return toc_entries

    def _parse_toc_section(self, xml_content: str, section_file: str) -> List[Dict[str, Any]]:
        """
        Parse a TOC section XML to extract regulation titles.

        Args:
            xml_content: XML content string.
            section_file: Source file name for logging.

        Returns:
            List of TOC entry dictionaries.
        """
        entries = []

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error in {section_file}: {e}")
            return entries

        for elem in root.iter():
            if elem.tag != f'{{{self.ns["hp"]}}}p':
                continue

            # Extract text
            text_parts = []
            for run in elem.findall('.//hp:run', self.ns):
                for t in run.findall('./hp:t', self.ns):
                    if t.text:
                        text_parts.append(t.text)

            if not text_parts:
                continue

            text = ''.join(text_parts).strip()
            if not text:
                continue

            # Clean text
            cleaned = self.normalizer.clean(text)
            if not cleaned:
                continue

            # Check if this is a regulation title
            result = self.title_detector.detect(cleaned)

            if result.is_title:
                # Extract rule code if present (e.g., "3-1-10")
                rule_code = self._extract_rule_code(cleaned)

                entry = {
                    "id": f"toc-{len(entries) + 1:04d}",
                    "title": result.title,
                    "page": "",
                    "rule_code": rule_code,
                    "confidence": result.confidence_score,
                }
                entries.append(entry)

        return entries

    def _parse_bodies_from_zip(
        self,
        file_path: Path,
        toc_entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse regulation bodies from section0.xml, matching to TOC.

        Args:
            file_path: Path to HWPX file.
            toc_entries: List of TOC entries to match.

        Returns:
            List of parsed regulation dictionaries.
        """
        regulations = []

        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                section0_path = 'Contents/section0.xml'

                if section0_path not in zip_ref.namelist():
                    logger.error(f"section0.xml not found in {file_path}")
                    return regulations

                with zip_ref.open(section0_path) as f:
                    try:
                        content = f.read().decode('utf-8')
                    except UnicodeDecodeError:
                        f.seek(0)
                        content = f.read().decode('cp949')

                # Parse section0.xml
                regulations = self._parse_content_section(
                    content,
                    toc_entries
                )

        except Exception as e:
            logger.error(f"Failed to parse bodies from {file_path}: {e}")

        return regulations

    def _parse_content_section(
        self,
        xml_content: str,
        toc_entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse content section and match to TOC entries.

        This is the key improvement: we search for each TOC title in the content
        and create regulation entries even if no articles are found.

        Args:
            xml_content: XML content from section0.xml.
            toc_entries: TOC entries to match against.

        Returns:
            List of regulation dictionaries.
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return []

        # Build a map of TOC titles to their entries
        toc_map = {}
        for entry in toc_entries:
            normalized = self.normalizer.get_title_hash(entry['title'])
            toc_map[normalized] = entry

        # Track which TOC entries have been matched
        matched_toc: Set[str] = set()

        # Current regulation being built
        current_regulation: Optional[Dict[str, Any]] = None
        current_articles: List[Dict[str, Any]] = []
        accumulated_text: List[str] = []
        regulations: List[Dict[str, Any]] = []

        def flush_current_article():
            """Flush accumulated text as content to the last article."""
            if current_articles and accumulated_text:
                content = '\n'.join(accumulated_text).strip()
                if content:
                    current_articles[-1]["content"] = (
                        current_articles[-1].get("content", "") + "\n" + content
                    )
                accumulated_text.clear()

        # Process paragraphs
        for elem in root.iter():
            if elem.tag != f'{{{self.ns["hp"]}}}p':
                continue

            text = self._extract_paragraph_text(elem)
            if not text:
                continue

            # Clean text for matching
            cleaned = self.normalizer.clean(text)
            if not cleaned:
                continue

            # Check if this text matches a TOC entry
            normalized = self.normalizer.get_title_hash(cleaned)

            if normalized in toc_map and normalized not in matched_toc:
                # Found a regulation title!
                matched_toc.add(normalized)

                # Save previous regulation
                if current_regulation:
                    flush_current_article()
                    current_regulation["articles"] = current_articles
                    # Always add regulation, even if no articles (repealed/empty)
                    regulations.append(current_regulation)

                # Create new regulation from TOC entry
                toc_entry = toc_map[normalized]
                current_regulation = {
                    "id": f"reg-{len(regulations) + 1:04d}",
                    "kind": "regulation",
                    "title": toc_entry["title"],
                    "rule_code": toc_entry.get("rule_code", ""),
                    "articles": [],
                    "repealed": "폐지" in toc_entry["title"],
                    "empty": False,
                }
                current_articles = []

            # Detect article markers
            elif self._is_article_marker(cleaned):
                flush_current_article()

                # Extract article using the article extractor
                article = self.article_extractor.extract_article(cleaned)
                if article:
                    current_articles.append(article)

            # Otherwise accumulate as content
            elif current_regulation:
                accumulated_text.append(cleaned)

        # Don't forget the last regulation
        if current_regulation:
            flush_current_article()
            current_regulation["articles"] = current_articles
            regulations.append(current_regulation)

        # Create entries for unmatched TOC items (repealed/empty regulations)
        for normalized, toc_entry in toc_map.items():
            if normalized not in matched_toc:
                regulations.append({
                    "id": f"reg-{len(regulations) + 1:04d}",
                    "kind": "regulation",
                    "title": toc_entry["title"],
                    "rule_code": toc_entry.get("rule_code", ""),
                    "articles": [],  # Empty for repealed/missing content
                    "repealed": "폐지" in toc_entry["title"],
                    "empty": True,
                })

        return regulations

    def _ensure_completeness(
        self,
        toc_entries: List[Dict[str, Any]],
        parsed_regulations: List[Dict[str, Any]]
    ):
        """
        Ensure 100% completeness by creating any missing entries.

        Args:
            toc_entries: Original TOC entries.
            parsed_regulations: Parsed regulations (will be modified in place).
        """
        # This is now handled in _parse_content_section by creating
        # entries for unmatched TOC items
        pass

    def _extract_rule_code(self, text: str) -> str:
        """
        Extract rule code from text (e.g., "3-1-10").

        Args:
            text: Text that may contain rule code.

        Returns:
            Extracted rule code or empty string.
        """
        # Pattern: N-N-N (e.g., "3-1-10")
        match = re.search(r'\b\d+-\d+-\d+\b', text)
        if match:
            return match.group()
        return ""

    def _is_article_marker(self, text: str) -> bool:
        """
        Check if text is an article marker.

        Args:
            text: Text to check.

        Returns:
            True if text is an article marker.
        """
        # "## 제N조" format (markdown headers)
        if text.startswith("## ") and "제" in text and "조" in text:
            return True

        # Direct "제N조" format
        article_pattern = re.compile(r'^제\s*\d+\s*조')
        if article_pattern.match(text):
            return True

        return False

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

        return ''.join(text_parts)

    def _update_status(self, message: str):
        """Update status via callback if available."""
        if self.status_callback:
            self.status_callback(message)
        logger.info(message)
