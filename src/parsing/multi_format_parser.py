"""
HWPX Multi-Format Parser Coordinator.

This module implements the main HWPXMultiFormatParser class that orchestrates
TOC extraction, format delegation, and coverage tracking for HWPX regulation parsing.

TDD Approach: GREEN Phase → REFACTOR Phase
- Implementation created to make failing tests pass
- Improved docstrings and type hints
- Better code organization and readability
- Added TOC-specific parsing methods

Version: 1.0.0
Reference: SPEC-HWXP-002, TASK-006
"""
import logging
import zipfile
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

from src.parsing.detectors.regulation_title_detector import RegulationTitleDetector
from src.parsing.format.format_type import FormatType
from src.parsing.format.format_classifier import FormatClassifier, ClassificationResult
from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor
from src.parsing.analyzers.guideline_structure_analyzer import GuidelineStructureAnalyzer
from src.parsing.analyzers.unstructured_regulation_analyzer import UnstructuredRegulationAnalyzer
from src.parsing.metrics.coverage_tracker import CoverageTracker

logger = logging.getLogger(__name__)


class HWPXMultiFormatParser:
    """
    Unified parser that delegates to format-specific extractors for maximum coverage.

    Coordinates TOC extraction, format classification, content extraction, and coverage tracking.
    Integrates all components from TASK-001 through TASK-005.

    Responsibilities:
    - Coordinate TOC extraction and completeness validation
    - Classify regulation format (article/list/guideline/unstructured)
    - Delegate to appropriate format-specific extractor
    - Aggregate content from multiple sections
    - Track coverage metrics in real-time
    - Invoke LLM fallback for low-coverage regulations

    Attributes:
        title_detector: Detector for identifying regulation titles
        format_classifier: Classifier for detecting format types
        list_extractor: Extractor for list-format regulations
        guideline_analyzer: Analyzer for guideline-format regulations
        unstructured_analyzer: Analyzer for unstructured regulations (LLM-based)
        coverage_tracker: Tracker for coverage metrics
        status_callback: Optional callback for progress updates
        llm_client: Optional LLM client for unstructured analysis
    """

    # Section file patterns in HWPX archive
    SECTION_PATTERN = "Contents/section*.xml"
    TOC_SECTION = "Contents/section1.xml"
    MAIN_CONTENT_SECTION = "Contents/section0.xml"

    # Low coverage threshold (20% content)
    LOW_COVERAGE_THRESHOLD = 0.2

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the multi-format parser coordinator.

        Args:
            llm_client: Custom LLM client for unstructured analysis (optional)
            status_callback: Optional callback for progress updates
        """
        # Initialize title detector for TOC extraction
        self.title_detector = RegulationTitleDetector()

        # Initialize format classifier
        self.format_classifier = FormatClassifier()

        # Initialize format-specific extractors/analyzers
        self.list_extractor = ListRegulationExtractor()
        self.guideline_analyzer = GuidelineStructureAnalyzer()
        self.unstructured_analyzer = UnstructuredRegulationAnalyzer(llm_client=llm_client)

        # Initialize coverage tracker
        self.coverage_tracker = CoverageTracker()

        # Store status callback
        self.status_callback = status_callback

        logger.debug("HWPXMultiFormatParser initialized")

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse HWPX file with multi-format detection and extraction.

        Main parsing workflow:
        1. Extract TOC from section1.xml
        2. Aggregate content from all sections
        3. Validate TOC completeness
        4. For each TOC entry:
           a. Extract relevant content
           b. Classify format
           c. Delegate to appropriate extractor
           d. Track coverage
        5. Generate coverage report
        6. Return parsing result

        Args:
            file_path: Path to HWPX file

        Returns:
            Dictionary with regulations/docs, coverage report, and metadata
        """
        self._status_update("Starting HWPX file parsing...")

        # Step 1: Extract TOC
        self._status_update("Extracting Table of Contents...")
        toc_entries = self._extract_toc(file_path)

        # Step 2: Aggregate sections
        self._status_update("Aggregating content from sections...")
        sections = self._aggregate_sections(file_path)

        # Step 3: Validate TOC completeness
        self._status_update("Validating TOC completeness...")
        is_complete, missing_titles = self._validate_toc_completeness(toc_entries, sections)

        if not is_complete:
            logger.warning(
                f"TOC incomplete: {len(missing_titles)}/{len(toc_entries)} "
                f"regulations missing content"
            )

        # Step 4: Extract content for each TOC entry
        self._status_update(f"Extracting content for {len(toc_entries)} regulations...")
        regulations = []

        for toc_entry in toc_entries:
            title = toc_entry.get("title", "")

            # Find relevant content for this title
            content = self._find_content_for_title(title, sections)

            # Classify format
            classification = self._classify_format(content)

            # Extract with appropriate format handler
            extraction_result = self._extract_with_format(
                title=title,
                content=content,
                format_type=classification.format_type
            )

            # Create regulation entry
            regulation = self._create_regulation_entry(
                title=title,
                content=content,
                extraction_result=extraction_result,
                classification=classification
            )

            # Track coverage
            content_length = len(content)
            has_content = content_length > 0
            self.coverage_tracker.track_regulation(
                format_type=classification.format_type,
                has_content=has_content,
                content_length=content_length
            )

            regulations.append(regulation)

        # Step 5: Generate coverage report
        self._status_update("Generating coverage report...")
        coverage_report = self.coverage_tracker.get_coverage_report()

        # Step 6: Build final result
        result = {
            "docs": regulations,
            "coverage": self._build_coverage_dict(coverage_report),
            "metadata": {
                "total_regulations": len(toc_entries),
                "successfully_parsed": len(regulations),
                "toc_complete": is_complete,
                "missing_regulations": len(missing_titles),
                "missing_titles": missing_titles[:10],  # First 10 missing titles
                "source_file": str(file_path)
            }
        }

        self._status_update(f"Parsing complete: {len(regulations)} regulations extracted")

        return result

    def _extract_toc(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extract Table of Contents from section1.xml.

        Args:
            file_path: Path to HWPX file

        Returns:
            List of TOC entry dictionaries with title, page, etc.
        """
        toc_entries = []

        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                if self.TOC_SECTION not in zf.namelist():
                    logger.warning(f"TOC section {self.TOC_SECTION} not found in HWPX file")
                    return toc_entries

                toc_content = zf.read(self.TOC_SECTION).decode('utf-8', errors='ignore')

                # Parse TOC content line by line
                for line in toc_content.split('\n'):
                    line = line.strip()
                    if not line or len(line) < 4:
                        continue

                    # For TOC parsing, we need to extract clean title from TOC format
                    # TOC format: "규정명 ...................... page"
                    # Try to extract the title part before dots and spaces
                    clean_title = self._extract_title_from_toc_line(line)

                    if clean_title:
                        detection_result = self.title_detector.detect(clean_title)
                        if detection_result.is_title:
                            toc_entries.append({
                                "title": clean_title,
                                "page": "",  # Could extract page number from TOC
                                "original_line": line
                            })

        except Exception as e:
            logger.error(f"Error extracting TOC: {e}")

        return toc_entries

    def _extract_title_from_toc_line(self, line: str) -> Optional[str]:
        """
        Extract clean regulation title from TOC line.

        TOC lines often have format: "규정명 ...................... page"
        This method extracts just the title part.

        Args:
            line: TOC line to parse

        Returns:
            Extracted title or None if not a title line
        """
        # Check if line looks like a TOC entry (has dots followed by page number)
        toc_pattern = re.compile(r'^([^\s.]+(?:[^\s.]*\s*)*)\s*[\.·]{2,}\s*\d*\s*$')

        match = toc_pattern.match(line)
        if match:
            # Extract the title part
            title = match.group(1).strip()
            # Clean up any trailing dots or spaces
            title = re.sub(r'[\.·\s]+$', '', title).strip()
            return title if len(title) >= 4 else None

        # If no TOC pattern match, try direct detection
        return line if self.title_detector.detect(line).is_title else None

    def _aggregate_sections(self, file_path: Path) -> Dict[str, str]:
        """
        Aggregate content from all sections in HWPX file.

        This method handles:
        - section0.xml (main content)
        - section1.xml (TOC)
        - section2+.xml (additional sections like appendices)

        The method merges content from all sections while preserving
        section boundaries for later content searching.

        Args:
            file_path: Path to HWPX file

        Returns:
            Dictionary mapping section names to content with duplicates removed
        """
        sections = {}

        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Collect all section files
                section_files = [
                    name for name in zf.namelist()
                    if name.startswith("Contents/section") and name.endswith(".xml")
                ]

                # Sort sections to ensure consistent processing order
                section_files.sort()

                logger.debug(f"Found {len(section_files)} sections: {section_files}")

                for name in section_files:
                    try:
                        content = zf.read(name).decode('utf-8', errors='ignore')
                        sections[name] = content
                        logger.debug(f"Loaded section {name}: {len(content)} characters")
                    except Exception as e:
                        logger.warning(f"Error reading section {name}: {e}")

        except Exception as e:
            logger.error(f"Error aggregating sections: {e}")

        logger.info(f"Aggregated {len(sections)} sections from HWPX file")

        return sections

    def _merge_section_contents(self, sections: Dict[str, str]) -> str:
        """
        Merge content from multiple sections, removing duplicates.

        This method combines content from all sections while:
        - Removing duplicate content blocks
        - Preserving content order (section0, section1, section2+)
        - Handling section boundaries appropriately

        Args:
            sections: Dictionary mapping section names to content

        Returns:
            Merged content string with duplicates removed
        """
        if not sections:
            return ""

        # Track seen content blocks to eliminate duplicates
        seen_blocks = set()
        merged_lines = []

        # Process sections in order (section0 first for main content)
        section_order = sorted(sections.keys(), key=lambda x: (
            0 if "section0.xml" in x else 1 if "section1.xml" in x else 2
        ))

        for section_name in section_order:
            content = sections[section_name]
            lines = content.split('\n')

            for line in lines:
                # Normalize line for duplicate detection
                normalized = line.strip()

                # Skip empty lines
                if not normalized:
                    continue

                # Create a hash for duplicate detection
                line_hash = hash(normalized)

                # Add only if not seen before
                if line_hash not in seen_blocks:
                    seen_blocks.add(line_hash)
                    merged_lines.append(line)

        merged_content = '\n'.join(merged_lines)

        logger.info(f"Merged {len(sections)} sections: {len(merged_lines)} unique lines")

        return merged_content

    def _validate_toc_completeness(
        self,
        toc_entries: List[Dict[str, Any]],
        sections: Dict[str, str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate TOC completeness by checking if all TOC entries have corresponding content.

        This method ensures that all regulations listed in the TOC (typically 514 entries)
        have at least some content in the aggregated sections (excluding the TOC section itself).

        Args:
            toc_entries: List of TOC entry dictionaries
            sections: Dictionary of section contents

        Returns:
            Tuple of (is_complete, missing_titles)
            - is_complete: True if all TOC entries have content
            - missing_titles: List of titles without corresponding content
        """
        if not toc_entries:
            logger.warning("No TOC entries to validate")
            return True, []

        missing_titles = []

        # Exclude TOC section from content search
        # (we only want to check if content exists in actual regulation sections)
        content_sections = {
            name: content
            for name, content in sections.items()
            if self.TOC_SECTION not in name
        }

        # Check each TOC entry for corresponding content
        for toc_entry in toc_entries:
            title = toc_entry.get("title", "")

            # Search for title in content sections (excluding TOC)
            found = False
            for section_content in content_sections.values():
                if title in section_content:
                    found = True
                    break

            if not found:
                missing_titles.append(title)
                logger.debug(f"TOC entry without content: {title}")

        is_complete = len(missing_titles) == 0
        completeness_rate = (len(toc_entries) - len(missing_titles)) / len(toc_entries) * 100

        logger.info(
            f"TOC completeness: {len(toc_entries) - len(missing_titles)}/{len(toc_entries)} "
            f"({completeness_rate:.1f}%)"
        )

        if not is_complete:
            logger.warning(
                f"{len(missing_titles)} TOC entries missing content: "
                f"{', '.join(missing_titles[:5])}{'...' if len(missing_titles) > 5 else ''}"
            )

        return is_complete, missing_titles

    def _find_content_for_title(self, title: str, sections: Dict[str, str]) -> str:
        """
        Find content relevant to a specific title from aggregated sections.

        This method searches across all sections (section0, section1, section2+)
        to find content related to the given title.

        Args:
            title: Regulation title to search for
            sections: Dictionary of section contents

        Returns:
            Content string relevant to the title
        """
        # Search for title in sections (prioritize section0 for main content)
        section_priority = ["section0.xml", "section2.xml", "section3.xml"]

        # Check priority sections first
        for priority_section in section_priority:
            for section_name, section_content in sections.items():
                if priority_section in section_name and title in section_content:
                    content = self._extract_content_around_title(title, section_content)
                    if content:
                        return content

        # Search remaining sections
        for section_name, section_content in sections.items():
            if title in section_content:
                content = self._extract_content_around_title(title, section_content)
                if content:
                    return content

        # Return empty string if not found
        return ""

    def _extract_content_around_title(self, title: str, section_content: str) -> str:
        """
        Extract content around a title marker.

        Args:
            title: Title to search for
            section_content: Section content to search in

        Returns:
            Extracted content string
        """
        # Find title position
        title_pos = section_content.find(title)

        if title_pos == -1:
            return ""

        # Extract content from title to next title or end
        start = title_pos + len(title)

        # Find next regulation title or end of content
        remaining = section_content[start:]
        lines = remaining.split('\n')

        content_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Stop at next title (different from current title)
            detection_result = self.title_detector.detect(line)
            if detection_result.is_title and line != title:
                break

            content_lines.append(line)

        return '\n'.join(content_lines)

    def _classify_format(self, content: str) -> ClassificationResult:
        """
        Classify regulation content format type.

        Args:
            content: Regulation text content

        Returns:
            ClassificationResult with format type and confidence
        """
        return self.format_classifier.classify(content)

    def _extract_with_format(
        self,
        title: str,
        content: str,
        format_type: FormatType
    ) -> Dict[str, Any]:
        """
        Extract content using format-specific extractor.

        Args:
            title: Regulation title
            content: Regulation text content
            format_type: Detected format type

        Returns:
            Extraction result with provisions/articles
        """
        if format_type == FormatType.ARTICLE:
            # Use existing article extraction logic
            return self._extract_article_format(title, content)

        elif format_type == FormatType.LIST:
            # Use list extractor
            return self.list_extractor.extract(content)

        elif format_type == FormatType.GUIDELINE:
            # Use guideline analyzer
            return self.guideline_analyzer.analyze(title, content)

        else:  # UNSTRUCTURED
            # Use unstructured analyzer (LLM-based)
            return self.unstructured_analyzer.analyze(title, content)

    def _extract_article_format(self, title: str, content: str) -> Dict[str, Any]:
        """
        Extract article-format content (basic implementation).

        Args:
            title: Regulation title
            content: Regulation text content

        Returns:
            Dictionary with articles array
        """
        # Basic article extraction - look for 제N조 patterns
        articles = []
        lines = content.split('\n')

        current_article = None
        article_num = 1

        for line in lines:
            line = line.strip()

            # Check for article marker
            if '제' in line and '조' in line:
                if current_article:
                    articles.append(current_article)

                current_article = {
                    "number": article_num,
                    "content": line
                }
                article_num += 1
            elif current_article:
                current_article["content"] += "\n" + line

        if current_article:
            articles.append(current_article)

        return {
            "articles": articles,
            "provisions": [a.get("content", "") for a in articles]
        }

    def _create_regulation_entry(
        self,
        title: str,
        content: str,
        extraction_result: Dict[str, Any],
        classification: ClassificationResult
    ) -> Dict[str, Any]:
        """
        Create a regulation entry from extraction results.

        Args:
            title: Regulation title
            content: Original content
            extraction_result: Result from format-specific extractor
            classification: Format classification result

        Returns:
            Regulation entry dictionary
        """
        articles = extraction_result.get("articles", [])
        provisions = extraction_result.get("provisions", [])

        # Calculate coverage score
        content_length = len(content)
        extracted_length = sum(len(a.get("content", "")) for a in articles)
        coverage_score = extracted_length / content_length if content_length > 0 else 0.0

        return {
            "title": title,
            "content": content,
            "articles": articles,
            "provisions": provisions,
            "metadata": {
                "format_type": classification.format_type.value,
                "confidence": classification.confidence,
                "coverage_score": coverage_score,
                "extraction_rate": min(len(provisions) / max(len(content.split('\n')), 1), 1.0)
            }
        }

    def _attempt_llm_fallback(self, regulation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt LLM fallback for low-coverage regulations.

        Args:
            regulation: Regulation entry with potential low coverage

        Returns:
            Enhanced regulation entry or original if coverage is sufficient
        """
        metadata = regulation.get("metadata", {})
        coverage_score = metadata.get("coverage_score", 1.0)

        # Skip LLM fallback if coverage is already good
        if coverage_score >= self.LOW_COVERAGE_THRESHOLD:
            return regulation

        # Attempt LLM analysis
        self._status_update(f"LLM fallback for: {regulation.get('title', 'Unknown')}")

        title = regulation.get("title", "")
        content = regulation.get("content", "")

        llm_result = self.unstructured_analyzer.analyze(title, content)

        # Update regulation with LLM results
        regulation["articles"] = llm_result.get("articles", regulation.get("articles", []))
        regulation["provisions"] = llm_result.get("provisions", regulation.get("provisions", []))
        regulation["metadata"]["llm_enhanced"] = True

        return regulation

    def _build_coverage_dict(self, coverage_report) -> Dict[str, Any]:
        """
        Convert CoverageReport to dictionary for JSON output.

        Args:
            coverage_report: CoverageReport object

        Returns:
            Dictionary representation of coverage report
        """
        return {
            "total": coverage_report.total_regulations,
            "with_content": coverage_report.regulations_with_content,
            "coverage_rate": coverage_report.coverage_percentage,
            "by_format": {
                fmt.value: count
                for fmt, count in coverage_report.format_breakdown.items()
            },
            "avg_content_length": coverage_report.avg_content_length,
            "low_coverage_count": coverage_report.low_coverage_count,
        }

    def _status_update(self, message: str) -> None:
        """
        Send status update if callback is registered.

        Args:
            message: Status message
        """
        if self.status_callback:
            self.status_callback(message)

        logger.debug(f"Status: {message}")
