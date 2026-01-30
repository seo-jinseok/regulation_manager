"""
Citation Validator for RAG System.

Validates and enforces proper citation format in RAG answers.
Ensures "규정명 제N조" format compliance with post-processing validation.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class CitationMatch:
    """A single citation match found in text."""

    text: str
    regulation_name: str
    article_number: str
    start_pos: int
    end_pos: int
    is_valid: bool


@dataclass
class CitationValidationResult:
    """Result of citation validation."""

    is_valid: bool
    citation_count: int
    citations: List[CitationMatch]
    missing_regulation_names: List[str]
    citation_density: float
    issues: List[str] = field(default_factory=list)


@dataclass
class CitationEnrichmentResult:
    """Result of citation enrichment."""

    enriched_answer: str
    added_citations: List[str]
    original_answer: str


class CitationValidator:
    """
    Validates and enforces proper citation format in RAG answers.

    Features:
    - Validates "규정명 제N조" format
    - Calculates citation density
    - Detects missing regulation names
    - Enriches answers with missing citations when context permits
    """

    # Citation patterns
    CITATION_PATTERNS = [
        # 규정명 제N조 pattern (preferred) - use \S* to match any non-whitespace before keyword
        re.compile(
            r"(\S*(?:규정|학칙|시행세칙|요령|지침|규칙))\s*제(\d+)(?:조|항)(?:\s*제(\d+)(?:조|항))?"
        ),
        # 제N조 pattern (requires regulation name in context)
        re.compile(r"제(\d+)(?:조|항)(?:\s*제(\d+)(?:조|항))?"),
    ]

    # Regulation name patterns (must appear before citation)
    REGULATION_PATTERNS = [
        re.compile(r"((?:[\w가-힣]+\s+)?(?:규정|학칙|시행세칙|요령|지침|규칙))"),
        re.compile(r"((?:[\w가-힣]+\s+)?학칙)"),
    ]

    # Minimum citation density (citations per 500 characters)
    MIN_CITATION_DENSITY = 0.5  # At least 1 citation per 1000 chars

    def __init__(self, strict_mode: bool = False):
        """
        Initialize CitationValidator.

        Args:
            strict_mode: If True, enforces strict citation format compliance
        """
        self.strict_mode = strict_mode

    def validate_citation(
        self, answer: str, context_sources: List[str]
    ) -> CitationValidationResult:
        """
        Validate citation format in answer.

        Args:
            answer: The RAG-generated answer
            context_sources: List of source regulation names from context

        Returns:
            CitationValidationResult with validation details
        """
        # Extract all citations
        citations = self._extract_citations(answer)

        # Check for missing regulation names
        missing_regulation_names = self._check_missing_regulation_names(
            answer, citations, context_sources
        )

        # Calculate citation density
        citation_density = self._calculate_citation_density(answer, citations)

        # Validate each citation
        issues = []
        valid_citations = 0

        for citation in citations:
            if not citation.is_valid:
                issues.append(
                    f"출처 '{citation.text}'에 규정명이 누락되었습니다. "
                    f"'{{규정명}} 제{citation.article_number}조' 형식을 사용하세요."
                )
            else:
                valid_citations += 1

        # Check overall validity
        is_valid = True
        if self.strict_mode:
            # Strict mode: all citations must have regulation names
            is_valid = len(missing_regulation_names) == 0 and valid_citations > 0
        else:
            # Lenient mode: at least some citations with regulation names
            is_valid = (
                valid_citations > 0 or citation_density >= self.MIN_CITATION_DENSITY
            )

        # Add density warning if too low
        if citation_density < self.MIN_CITATION_DENSITY:
            issues.append(
                f"출처 인용 빈도가 낮습니다 (현재: {citation_density:.2f}, "
                f"권장: {self.MIN_CITATION_DENSITY:.2f} 이상). "
                f"각 규정 조항 인용 시 반드시 규정명을 함께 명시하세요."
            )

        return CitationValidationResult(
            is_valid=is_valid,
            citation_count=len(citations),
            citations=citations,
            missing_regulation_names=missing_regulation_names,
            citation_density=citation_density,
            issues=issues,
        )

    def enrich_citation(
        self, answer: str, context_sources: List[str]
    ) -> CitationEnrichmentResult:
        """
        Enrich answer with missing citations when context permits.

        Args:
            answer: The RAG-generated answer
            context_sources: List of source regulation names from context

        Returns:
            CitationEnrichmentResult with enriched answer
        """
        citations = self._extract_citations(answer)
        enriched_answer = answer
        added_citations = []

        # Find citations without regulation names
        for citation in citations:
            if not citation.is_valid and context_sources:
                # Use the first available regulation name from context
                regulation_name = context_sources[0]
                new_citation = f"{regulation_name} 제{citation.article_number}조"

                # Replace in answer
                enriched_answer = enriched_answer.replace(
                    citation.text, new_citation, 1
                )
                added_citations.append(f"'{citation.text}' -> '{new_citation}'")

        return CitationEnrichmentResult(
            enriched_answer=enriched_answer,
            added_citations=added_citations,
            original_answer=answer,
        )

    def _extract_citations(self, text: str) -> List[CitationMatch]:
        """
        Extract all citation matches from text.

        Args:
            text: The text to search for citations

        Returns:
            List of CitationMatch objects
        """
        citations = []

        # Try each pattern
        for pattern in self.CITATION_PATTERNS:
            for match in pattern.finditer(text):
                groups = match.groups()

                # Determine if this is the preferred format (with regulation name)
                is_valid = False
                regulation_name = ""
                article_number = ""

                if len(groups) >= 2:
                    # Check if groups[0] looks like a regulation name (Pattern 1)
                    if groups[0] and any(
                        suffix in groups[0]
                        for suffix in [
                            "규정",
                            "학칙",
                            "시행세칙",
                            "요령",
                            "지침",
                            "규칙",
                        ]
                    ):
                        # Pattern 1: (regulation, article, subsection)
                        is_valid = True
                        regulation_name = groups[0]
                        article_number = groups[1]
                    # Pattern 2: (article, subsection) - need to check for regulation name in context
                    elif groups[0]:
                        article_number = groups[0]
                        # Look for regulation name before the citation
                        preceding_text = text[: match.start()]
                        if self._has_regulation_name_nearby(preceding_text):
                            is_valid = True
                            regulation_name = self._extract_nearby_regulation_name(
                                preceding_text
                            )

                if article_number:  # Only add if we found an article number
                    citations.append(
                        CitationMatch(
                            text=match.group(0),
                            regulation_name=regulation_name,
                            article_number=article_number,
                            start_pos=match.start(),
                            end_pos=match.end(),
                            is_valid=is_valid,
                        )
                    )

        # Remove duplicate citations (those with overlapping positions)
        citations = self._deduplicate_citations(citations)

        return citations

    def _deduplicate_citations(
        self, citations: List[CitationMatch]
    ) -> List[CitationMatch]:
        """
        Remove duplicate citations based on overlapping positions and normalized content.

        Prefers citations with regulation names over those without.

        Args:
            citations: List of extracted citations

        Returns:
            Deduplicated list of citations
        """
        if not citations:
            return citations

        # Sort by: valid citations first, then by start position
        citations.sort(key=lambda c: (not c.is_valid, c.start_pos))

        # Track normalized citation keys and position ranges
        seen_keys = set()
        seen_ranges = []
        deduplicated = []

        for citation in citations:
            # Normalize the citation text for comparison
            # "제15조" and "15조" should both normalize to "15"
            # "학칙 제15조" should also normalize to "15"
            normalized_key = self._normalize_citation_key(citation)

            # Check for key-based duplicates (same article number)
            is_key_duplicate = normalized_key in seen_keys

            # Check for position-based duplicates (overlapping text)
            overlaps = False
            for existing_start, existing_end in seen_ranges:
                if not (
                    citation.end_pos <= existing_start
                    or existing_end <= citation.start_pos
                ):
                    overlaps = True
                    break

            # Only add citation if not a duplicate by key or position
            if not is_key_duplicate and not overlaps:
                seen_keys.add(normalized_key)
                seen_ranges.append((citation.start_pos, citation.end_pos))
                deduplicated.append(citation)

        return deduplicated

    def _normalize_citation_key(self, citation: CitationMatch) -> str:
        """
        Normalize citation to a unique key for deduplication.

        Normalizes "제15조", "15조", "학칙 제15조" all to the same key.

        Args:
            citation: The citation to normalize

        Returns:
            Normalized key string
        """
        # The article_number is the core identifier
        # Add regulation_name if present for distinct citations
        if citation.regulation_name:
            return f"{citation.regulation_name}:{citation.article_number}"
        return citation.article_number

    def _citations_overlap(
        self, citation1: CitationMatch, citation2: CitationMatch
    ) -> bool:
        """
        Check if two citations overlap in position.

        Args:
            citation1: First citation
            citation2: Second citation

        Returns:
            True if citations overlap
        """
        return not (
            citation1.end_pos <= citation2.start_pos
            or citation2.end_pos <= citation1.start_pos
        )

    def _check_missing_regulation_names(
        self,
        answer: str,
        citations: List[CitationMatch],
        context_sources: List[str],
    ) -> List[str]:
        """
        Check for citations missing regulation names.

        Args:
            answer: The answer text
            citations: List of extracted citations
            context_sources: List of available regulation names from context

        Returns:
            List of citations without regulation names
        """
        missing = []

        for citation in citations:
            if not citation.is_valid and not citation.regulation_name:
                missing.append(citation.text)

        return missing

    def _calculate_citation_density(
        self, answer: str, citations: List[CitationMatch]
    ) -> float:
        """
        Calculate citation density (citations per character).

        Args:
            answer: The answer text
            citations: List of extracted citations

        Returns:
            Citation density (citations per 1000 characters)
        """
        if not answer:
            return 0.0

        # Normalize to citations per 1000 characters
        return (len(citations) * 1000) / max(len(answer), 1)

    def _has_regulation_name_nearby(self, text: str) -> bool:
        """
        Check if there's a regulation name in the preceding text.

        Args:
            text: The text to search

        Returns:
            True if a regulation name pattern is found
        """
        # Look in the last 200 characters
        search_text = text[-200:] if len(text) > 200 else text

        for pattern in self.REGULATION_PATTERNS:
            if pattern.search(search_text):
                return True

        return False

    def _extract_nearby_regulation_name(self, text: str) -> str:
        """
        Extract the most recent regulation name from text.

        Args:
            text: The text to search

        Returns:
            Regulation name if found, empty string otherwise
        """
        # Look in the last 200 characters
        search_text = text[-200:] if len(text) > 200 else text

        # Find all regulation name matches
        matches = []
        for pattern in self.REGULATION_PATTERNS:
            for match in pattern.finditer(search_text):
                matches.append(match)

        # Return the last (most recent) match
        if matches:
            return matches[-1].group(1)

        return ""

    def enforce_citation_format(self, answer: str) -> Tuple[str, List[str]]:
        """
        Enforce proper citation format by adding regulation names where missing.

        Args:
            answer: The answer to enforce citation format on

        Returns:
            Tuple of (enforced_answer, list_of_changes)
        """
        citations = self._extract_citations(answer)
        enforced_answer = answer
        changes = []

        for citation in reversed(citations):  # Reverse to maintain positions
            if not citation.is_valid:
                # Look for nearby regulation name
                preceding = enforced_answer[: citation.start_pos]
                if self._has_regulation_name_nearby(preceding):
                    reg_name = self._extract_nearby_regulation_name(preceding)
                    if reg_name:
                        new_citation = f"{reg_name} 제{citation.article_number}조"
                        old_citation = citation.text

                        # Replace
                        enforced_answer = (
                            enforced_answer[: citation.start_pos]
                            + new_citation
                            + enforced_answer[citation.end_pos :]
                        )

                        changes.append(
                            f"Replaced '{old_citation}' with '{new_citation}'"
                        )

        return enforced_answer, changes


# Convenience functions for backward compatibility
def validate_answer_citation(
    answer: str, context_sources: List[str], strict_mode: bool = False
) -> CitationValidationResult:
    """
    Validate citation format in answer (convenience function).

    Args:
        answer: The RAG-generated answer
        context_sources: List of source regulation names from context
        strict_mode: If True, enforces strict citation format compliance

    Returns:
        CitationValidationResult with validation details
    """
    validator = CitationValidator(strict_mode=strict_mode)
    return validator.validate_citation(answer, context_sources)


def enforce_citations(
    answer: str, context_sources: List[str]
) -> CitationEnrichmentResult:
    """
    Enforce citation format in answer (convenience function).

    Args:
        answer: The RAG-generated answer
        context_sources: List of source regulation names from context

    Returns:
        CitationEnrichmentResult with enriched answer
    """
    validator = CitationValidator()
    return validator.enrich_citation(answer, context_sources)
