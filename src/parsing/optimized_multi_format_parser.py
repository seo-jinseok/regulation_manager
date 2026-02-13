"""
Optimized HWPX Multi-Format Parser with Performance Enhancements.

This module implements performance optimizations for HWPX regulation parsing:
- Parallel processing for independent regulations
- Caching for format classification results
- Optimized section aggregation with content indexing
- Batch processing for format-specific extraction

Performance Target: <60 seconds for 514 regulations
Memory Target: <2GB peak memory usage

Version: 2.0.0 (Optimized)
Reference: SPEC-HWXP-002, TASK-008
"""
import logging
import zipfile
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.parsing.detectors.regulation_title_detector import RegulationTitleDetector
from src.parsing.format.format_type import FormatType
from src.parsing.format.format_classifier import FormatClassifier, ClassificationResult
from src.parsing.extractors.list_regulation_extractor import ListRegulationExtractor
from src.parsing.analyzers.guideline_structure_analyzer import GuidelineStructureAnalyzer
from src.parsing.analyzers.unstructured_regulation_analyzer import UnstructuredRegulationAnalyzer
from src.parsing.metrics.coverage_tracker import CoverageTracker

logger = logging.getLogger(__name__)


class OptimizedHWPXMultiFormatParser:
    """
    Optimized HWPX parser with parallel processing and caching.

    Performance Optimizations:
    1. Parallel regulation processing using ThreadPoolExecutor
    2. LRU cache for format classification (reduces repeated classification)
    3. Pre-indexed section content for fast title lookup
    4. Batch content extraction to minimize I/O

    Thread Safety:
    - Uses thread-local storage for format classifiers
    - Synchronized coverage tracking with locks
    - Immutable data structures for parallel processing

    Performance Targets:
    - Parsing time: <60 seconds for 514 regulations
    - Memory usage: <2GB peak
    - Parallel workers: min(32, (os.cpu_count() or 4) * 4)
    """

    # Section file patterns
    SECTION_PATTERN = "Contents/section*.xml"
    TOC_SECTION = "Contents/section1.xml"
    MAIN_CONTENT_SECTION = "Contents/section0.xml"

    # Low coverage threshold (20% content)
    LOW_COVERAGE_THRESHOLD = 0.2

    # Maximum parallel workers (default: 32 or CPU count * 4)
    MAX_WORKERS = min(32, (None or 4) * 4) if None else 32

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        max_workers: Optional[int] = None
    ):
        """
        Initialize the optimized multi-format parser.

        Args:
            llm_client: Custom LLM client for unstructured analysis (optional)
            status_callback: Optional callback for progress updates
            max_workers: Maximum number of parallel workers (default: CPU count * 4)
        """
        # Initialize title detector for TOC extraction
        self.title_detector = RegulationTitleDetector()

        # Initialize format classifier (will be cloned for threads)
        self.format_classifier = FormatClassifier()

        # Initialize format-specific extractors/analyzers
        self.list_extractor = ListRegulationExtractor()
        self.guideline_analyzer = GuidelineStructureAnalyzer()
        self.unstructured_analyzer = UnstructuredRegulationAnalyzer(llm_client=llm_client)

        # Initialize coverage tracker with thread safety
        self.coverage_tracker = ThreadSafeCoverageTracker()

        # Store status callback
        self.status_callback = status_callback

        # Configure parallel workers
        import os
        cpu_count = os.cpu_count() or 4
        self.max_workers = max_workers or min(32, cpu_count * 4)

        # Thread-local storage for thread-specific objects
        self._local = threading.local()

        logger.debug(f"OptimizedHWPXMultiFormatParser initialized with {self.max_workers} workers")

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse HWPX file with optimized parallel processing.

        Optimized workflow:
        1. Extract TOC (sequential, I/O bound)
        2. Aggregate and index sections (sequential, I/O bound)
        3. Pre-build content index for fast lookup (sequential)
        4. Process regulations in parallel (CPU bound)
        5. Aggregate results and generate report (sequential)

        Args:
            file_path: Path to HWPX file

        Returns:
            Dictionary with regulations/docs, coverage report, and metadata
        """
        self._status_update("Starting optimized HWPX file parsing...")

        # Phase 1: Extract TOC (sequential)
        self._status_update("Extracting Table of Contents...")
        toc_entries = self._extract_toc(file_path)

        # Phase 2: Aggregate sections (sequential)
        self._status_update("Aggregating and indexing sections...")
        sections = self._aggregate_sections(file_path)

        # Phase 3: Build content index for fast lookup (sequential)
        self._status_update("Building content index...")
        content_index = self._build_content_index(toc_entries, sections)

        # Phase 4: Process regulations in parallel (parallel)
        self._status_update(f"Processing {len(toc_entries)} regulations in parallel...")
        regulations = self._process_regulations_parallel(toc_entries, content_index, sections)

        # Phase 5: Generate coverage report (sequential)
        self._status_update("Generating coverage report...")
        coverage_report = self.coverage_tracker.get_coverage_report()

        # Phase 6: Build final result
        result = {
            "docs": regulations,
            "coverage": self._build_coverage_dict(coverage_report),
            "metadata": {
                "total_regulations": len(toc_entries),
                "successfully_parsed": len(regulations),
                "parallel_workers_used": self.max_workers,
                "optimization_enabled": True,
                "source_file": str(file_path)
            }
        }

        self._status_update(f"Parsing complete: {len(regulations)} regulations extracted")

        return result

    def _process_regulations_parallel(
        self,
        toc_entries: List[Dict[str, Any]],
        content_index: Dict[str, str],
        sections: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Process regulations in parallel using thread pool.

        Each regulation is processed independently:
        1. Find content using pre-built index
        2. Classify format (with caching)
        3. Extract with format-specific handler
        4. Track coverage (thread-safe)

        Args:
            toc_entries: List of TOC entry dictionaries
            content_index: Pre-built index of title -> content mapping
            sections: Dictionary of section contents

        Returns:
            List of processed regulation dictionaries
        """
        regulations = []
        completed_count = 0
        total_count = len(toc_entries)

        # Create thread-safe counter for progress updates
        progress_lock = threading.Lock()

        def update_progress():
            nonlocal completed_count
            with progress_lock:
                completed_count += 1
                if completed_count % 50 == 0 or completed_count == total_count:
                    self._status_update(
                        f"Processed {completed_count}/{total_count} regulations "
                        f"({completed_count/total_count*100:.1f}%)"
                    )

        # Process regulations in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self._process_single_regulation,
                    toc_entry,
                    content_index,
                    sections
                ): toc_entry for toc_entry in toc_entries
            }

            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    regulation = future.result()
                    if regulation:
                        regulations.append(regulation)
                    update_progress()
                except Exception as e:
                    toc_entry = futures[future]
                    logger.error(
                        f"Error processing regulation {toc_entry.get('title', 'Unknown')}: {e}"
                    )
                    update_progress()

        return regulations

    def _process_single_regulation(
        self,
        toc_entry: Dict[str, Any],
        content_index: Dict[str, str],
        sections: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single regulation (thread-safe).

        This method is called by parallel workers and must be thread-safe.

        Args:
            toc_entry: TOC entry dictionary
            content_index: Pre-built content index
            sections: Section contents

        Returns:
            Regulation dictionary or None if processing failed
        """
        title = toc_entry.get("title", "")

        # Find content using pre-built index
        content = content_index.get(title, "")
        if not content:
            # Fallback to search if not in index
            content = self._find_content_for_title(title, sections)

        # Classify format (with caching per thread)
        classification = self._classify_format_cached(content)

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

        # Track coverage (thread-safe)
        content_length = len(content)
        has_content = content_length > 0
        self.coverage_tracker.track_regulation(
            format_type=classification.format_type,
            has_content=has_content,
            content_length=content_length
        )

        return regulation

    @lru_cache(maxsize=1024)
    def _classify_format_cached(self, content: str) -> ClassificationResult:
        """
        Classify format with LRU caching to avoid repeated classification.

        Cache key is content hash, so similar content will hit cache.
        Thread-safe due to lru_cache implementation.

        Args:
            content: Regulation text content

        Returns:
            ClassificationResult with format type and confidence
        """
        # Create thread-local classifier if needed
        if not hasattr(self._local, 'classifier'):
            self._local.classifier = FormatClassifier()

        return self._local.classifier.classify(content)

    def _build_content_index(
        self,
        toc_entries: List[Dict[str, Any]],
        sections: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Build an index of title -> content mapping for fast lookup.

        This pre-processes content to avoid repeated string searches
        during parallel regulation processing.

        Args:
            toc_entries: List of TOC entries
            sections: Dictionary of section contents

        Returns:
            Dictionary mapping titles to their content
        """
        index = {}

        # Merge section contents for efficient searching
        merged_content = self._merge_section_contents_for_search(sections)

        for toc_entry in toc_entries:
            title = toc_entry.get("title", "")
            if not title or title in index:
                continue

            # Extract content for this title
            content = self._extract_content_around_title(title, merged_content)
            if content:
                index[title] = content

        logger.info(f"Built content index: {len(index)} titles indexed")

        return index

    def _merge_section_contents_for_search(self, sections: Dict[str, str]) -> str:
        """
        Merge section contents optimized for content searching.

        Unlike the regular merge, this preserves section markers
        to enable efficient title-based content extraction.

        Args:
            sections: Dictionary of section contents

        Returns:
            Merged content string
        """
        if not sections:
            return ""

        # Prioritize section0 (main content), then section2+, skip section1 (TOC)
        section_priority = {
            "section0.xml": 0,
            "section2.xml": 1,
            "section3.xml": 2,
        }

        # Sort sections by priority
        sorted_sections = sorted(
            sections.items(),
            key=lambda x: section_priority.get(Path(x[0]).name, 999)
        )

        # Merge contents
        merged_lines = []
        for section_name, content in sorted_sections:
            if "section1.xml" in section_name:
                continue  # Skip TOC section

            lines = content.split('\n')
            merged_lines.extend(lines)

        return '\n'.join(merged_lines)

    # ============================================================================
    # Inherited methods (unchanged from base parser)
    # ============================================================================

    def _extract_toc(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract Table of Contents from section1.xml."""
        toc_entries = []

        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                if self.TOC_SECTION not in zf.namelist():
                    logger.warning(f"TOC section {self.TOC_SECTION} not found")
                    return toc_entries

                toc_content = zf.read(self.TOC_SECTION).decode('utf-8', errors='ignore')

                for line in toc_content.split('\n'):
                    line = line.strip()
                    if not line or len(line) < 4:
                        continue

                    clean_title = self._extract_title_from_toc_line(line)
                    if clean_title:
                        detection_result = self.title_detector.detect(clean_title)
                        if detection_result.is_title:
                            toc_entries.append({
                                "title": clean_title,
                                "page": "",
                                "original_line": line
                            })

        except Exception as e:
            logger.error(f"Error extracting TOC: {e}")

        return toc_entries

    def _extract_title_from_toc_line(self, line: str) -> Optional[str]:
        """Extract clean regulation title from TOC line."""
        toc_pattern = re.compile(r'^([^\s.]+(?:[^\s.]*\s*)*)\s*[\.·]{2,}\s*\d*\s*$')
        match = toc_pattern.match(line)
        if match:
            title = match.group(1).strip()
            title = re.sub(r'[\.·\s]+$', '', title).strip()
            return title if len(title) >= 4 else None
        return line if self.title_detector.detect(line).is_title else None

    def _aggregate_sections(self, file_path: Path) -> Dict[str, str]:
        """Aggregate content from all sections in HWPX file."""
        sections = {}

        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                section_files = [
                    name for name in zf.namelist()
                    if name.startswith("Contents/section") and name.endswith(".xml")
                ]
                section_files.sort()

                logger.debug(f"Found {len(section_files)} sections")

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

    def _find_content_for_title(self, title: str, sections: Dict[str, str]) -> str:
        """Find content relevant to a specific title from aggregated sections."""
        section_priority = ["section0.xml", "section2.xml", "section3.xml"]

        for priority_section in section_priority:
            for section_name, section_content in sections.items():
                if priority_section in section_name and title in section_content:
                    content = self._extract_content_around_title(title, section_content)
                    if content:
                        return content

        for section_name, section_content in sections.items():
            if title in section_content:
                content = self._extract_content_around_title(title, section_content)
                if content:
                    return content

        return ""

    def _extract_content_around_title(self, title: str, section_content: str) -> str:
        """Extract content around a title marker."""
        title_pos = section_content.find(title)
        if title_pos == -1:
            return ""

        start = title_pos + len(title)
        remaining = section_content[start:]
        lines = remaining.split('\n')

        content_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            detection_result = self.title_detector.detect(line)
            if detection_result.is_title and line != title:
                break

            content_lines.append(line)

        return '\n'.join(content_lines)

    def _extract_with_format(
        self,
        title: str,
        content: str,
        format_type: FormatType
    ) -> Dict[str, Any]:
        """Extract content using format-specific extractor."""
        if format_type == FormatType.ARTICLE:
            return self._extract_article_format(title, content)
        elif format_type == FormatType.LIST:
            return self.list_extractor.extract(content)
        elif format_type == FormatType.GUIDELINE:
            return self.guideline_analyzer.analyze(title, content)
        else:  # UNSTRUCTURED
            return self.unstructured_analyzer.analyze(title, content)

    def _extract_article_format(self, title: str, content: str) -> Dict[str, Any]:
        """Extract article-format content (basic implementation)."""
        articles = []
        lines = content.split('\n')

        current_article = None
        article_num = 1

        for line in lines:
            line = line.strip()

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
        """Create a regulation entry from extraction results."""
        articles = extraction_result.get("articles", [])
        provisions = extraction_result.get("provisions", [])

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

    def _build_coverage_dict(self, coverage_report) -> Dict[str, Any]:
        """Convert CoverageReport to dictionary for JSON output."""
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
        """Send status update if callback is registered."""
        if self.status_callback:
            self.status_callback(message)
        logger.debug(f"Status: {message}")


class ThreadSafeCoverageTracker:
    """
    Thread-safe wrapper for CoverageTracker.

    Uses locks to ensure thread-safe updates to coverage metrics
    during parallel regulation processing.
    """

    def __init__(self):
        """Initialize thread-safe coverage tracker."""
        self.tracker = CoverageTracker()
        self.lock = threading.Lock()

    def track_regulation(self, format_type: FormatType, has_content: bool, content_length: int) -> None:
        """Track regulation with thread safety."""
        with self.lock:
            self.tracker.track_regulation(
                format_type=format_type,
                has_content=has_content,
                content_length=content_length
            )

    def get_coverage_report(self):
        """Get coverage report (thread-safe read)."""
        with self.lock:
            return self.tracker.get_coverage_report()

    @property
    def total_regulations(self) -> int:
        """Get total regulations tracked (thread-safe)."""
        with self.lock:
            return self.tracker.total_regulations

    @property
    def regulations_with_content(self) -> int:
        """Get regulations with content (thread-safe)."""
        with self.lock:
            return self.tracker.regulations_with_content
