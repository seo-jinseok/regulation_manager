"""
Completeness Checker for HWPX Regulation Parsing

Validates parsing completeness by comparing parsed regulations
against the table of contents (TOC) to identify missing entries.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional, Tuple
from difflib import SequenceMatcher

from ..core.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)


@dataclass
class TOCEntry:
    """Table of Contents entry."""
    id: str
    title: str
    page: str = ""
    rule_code: str = ""
    normalized_title: str = ""

    def __post_init__(self):
        """Normalize title after initialization."""
        if not self.normalized_title:
            normalizer = TextNormalizer()
            self.normalized_title = normalizer.get_title_hash(self.title)


@dataclass
class CompletenessReport:
    """Report on parsing completeness."""
    total_toc_entries: int = 0
    total_parsed: int = 0
    matched_entries: int = 0
    missing_entries: int = 0
    extra_entries: int = 0
    is_complete: bool = False
    missing_titles: List[str] = field(default_factory=list)
    extra_titles: List[str] = field(default_factory=list)
    matched_titles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "total_toc_entries": self.total_toc_entries,
            "total_parsed": self.total_parsed,
            "matched_entries": self.matched_entries,
            "missing_entries": self.missing_entries,
            "extra_entries": self.extra_entries,
            "is_complete": self.is_complete,
            "completion_rate": (
                self.matched_entries / self.total_toc_entries * 100
                if self.total_toc_entries > 0
                else 0
            ),
            "missing_titles": self.missing_titles[:10],  # Limit output
            "extra_titles": self.extra_titles[:10],
            "matched_titles_sample": self.matched_titles[:10],
        }


class CompletenessChecker:
    """
    Validate parsing completeness against TOC.

    Uses fuzzy matching to account for text normalization differences
    between TOC entries and parsed regulations.
    """

    def __init__(
        self,
        fuzzy_match_threshold: float = 0.85,
        require_exact_match: bool = False,
    ):
        """
        Initialize the completeness checker.

        Args:
            fuzzy_match_threshold: Similarity threshold for fuzzy matching (0-1).
            require_exact_match: If True, only exact normalized matches count.
        """
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.require_exact_match = require_exact_match
        self.normalizer = TextNormalizer()

    def validate(
        self,
        toc_entries: List[TOCEntry],
        parsed_regulations: List[Dict[str, Any]],
    ) -> CompletenessReport:
        """
        Validate completeness by comparing TOC to parsed regulations.

        Args:
            toc_entries: List of TOC entries.
            parsed_regulations: List of parsed regulation dictionaries.

        Returns:
            CompletenessReport with detailed results.
        """
        report = CompletenessReport()
        report.total_toc_entries = len(toc_entries)
        report.total_parsed = len(parsed_regulations)

        # Build sets of normalized titles
        toc_titles = self._build_title_set(toc_entries)
        parsed_titles = self._build_parsed_title_set(parsed_regulations)

        # Match titles
        matched_toc: Set[str] = set()
        matched_parsed: Set[str] = set()

        # Try exact normalized matches first
        for toc_key, toc_title in toc_titles.items():
            if toc_key in parsed_titles:
                matched_toc.add(toc_key)
                matched_parsed.add(toc_key)
                report.matched_titles.append(toc_title)
                report.matched_entries += 1

        # Try fuzzy matching for remaining
        if not self.require_exact_match:
            remaining_toc = {k: v for k, v in toc_titles.items()
                           if k not in matched_toc}
            remaining_parsed = {k: v for k, v in parsed_titles.items()
                             if k not in matched_parsed}

            for toc_key, toc_title in remaining_toc.items():
                matched = False
                for parsed_key, parsed_title in remaining_parsed.items():
                    if self._titles_similar(toc_key, parsed_key):
                        matched_toc.add(toc_key)
                        matched_parsed.add(parsed_key)
                        report.matched_titles.append(toc_title)
                        report.matched_entries += 1
                        matched = True
                        break

                if not matched:
                    report.missing_titles.append(toc_title)
                    report.missing_entries += 1

        # Find extra entries (parsed but not in TOC)
        for parsed_key, parsed_title in parsed_titles.items():
            if parsed_key not in matched_parsed:
                report.extra_titles.append(parsed_title)
                report.extra_entries += 1

        # Determine completeness
        report.is_complete = (
            report.missing_entries == 0 and
            report.matched_entries == report.total_toc_entries
        )

        return report

    def _build_title_set(self, toc_entries: List[TOCEntry]) -> Dict[str, str]:
        """Build normalized title set from TOC entries."""
        return {entry.normalized_title: entry.title for entry in toc_entries}

    def _build_parsed_title_set(
        self, parsed_regulations: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Build normalized title set from parsed regulations."""
        title_map = {}
        for reg in parsed_regulations:
            title = reg.get("title", "")
            if title:
                normalized = self.normalizer.get_title_hash(title)
                title_map[normalized] = title
        return title_map

    def _titles_similar(self, title1: str, title2: str) -> bool:
        """
        Check if two normalized titles are similar using fuzzy matching.

        Args:
            title1: First normalized title.
            title2: Second normalized title.

        Returns:
            True if titles are similar above threshold.
        """
        # Use SequenceMatcher for similarity ratio
        ratio = SequenceMatcher(None, title1, title2).ratio()
        return ratio >= self.fuzzy_match_threshold

    def create_toc_from_regulations(
        self, regulations: List[Dict[str, Any]]
    ) -> List[TOCEntry]:
        """
        Create TOC entries from parsed regulations for validation.

        Args:
            regulations: List of parsed regulation dictionaries.

        Returns:
            List of TOCEntry objects.
        """
        toc_entries = []
        for idx, reg in enumerate(regulations):
            entry = TOCEntry(
                id=f"toc-{idx + 1:04d}",
                title=reg.get("title", ""),
                page=str(idx + 1),
                rule_code=reg.get("rule_code", ""),
            )
            toc_entries.append(entry)
        return toc_entries

    def find_best_match(
        self,
        title: str,
        candidates: List[str],
    ) -> Tuple[Optional[str], float]:
        """
        Find the best matching candidate for a title.

        Args:
            title: Title to match.
            candidates: List of candidate titles.

        Returns:
            Tuple of (best_match, similarity_score). Returns (None, 0.0) if no match.
        """
        if not candidates:
            return None, 0.0

        normalized_title = self.normalizer.get_title_hash(title)
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            normalized_candidate = self.normalizer.get_title_hash(candidate)
            score = SequenceMatcher(
                None, normalized_title, normalized_candidate
            ).ratio()

            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score >= self.fuzzy_match_threshold:
            return best_match, best_score

        return None, 0.0

    def generate_missing_report(
        self, report: CompletenessReport
    ) -> str:
        """
        Generate human-readable missing entries report.

        Args:
            report: Completeness report.

        Returns:
            Formatted report string.
        """
        lines = [
            "=== Parsing Completeness Report ===",
            f"Total TOC Entries: {report.total_toc_entries}",
            f"Total Parsed: {report.total_parsed}",
            f"Matched: {report.matched_entries}",
            f"Missing: {report.missing_entries}",
            f"Extra: {report.extra_entries}",
            f"Completion Rate: {report.to_dict()['completion_rate']:.1f}%",
            f"Status: {'✓ COMPLETE' if report.is_complete else '✗ INCOMPLETE'}",
            "",
        ]

        if report.missing_entries > 0:
            lines.append("Missing Regulations:")
            for title in report.missing_titles[:20]:
                lines.append(f"  - {title}")
            if len(report.missing_titles) > 20:
                lines.append(f"  ... and {len(report.missing_titles) - 20} more")
            lines.append("")

        if report.extra_entries > 0:
            lines.append("Extra Regulations (not in TOC):")
            for title in report.extra_titles[:20]:
                lines.append(f"  - {title}")
            if len(report.extra_titles) > 20:
                lines.append(f"  ... and {len(report.extra_titles) - 20} more")

        return "\n".join(lines)
