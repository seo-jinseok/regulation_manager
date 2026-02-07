"""
Entity type definitions for Regulation RAG System.

Defines the 6 new entity types introduced in SPEC-RAG-SEARCH-001
for enhanced entity recognition.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class EntityType(Enum):
    """
    Entity types for regulation search.

    Part of SPEC-RAG-SEARCH-001 TAG-001: Enhanced Entity Recognition.

    Types:
        SECTION: Regulation sections (조, 항, 호) - e.g., "제15조"
        PROCEDURE: Procedure-related terms - e.g., "신청", "절차", "방법"
        REQUIREMENT: Requirement-related terms - e.g., "자격", "요건", "조건"
        BENEFIT: Benefit-related terms - e.g., "혜택", "지급", "지원"
        DEADLINE: Deadline-related terms - e.g., "기한", "마감", "날짜"
        HYPERNYM: Hierarchical expansion terms - e.g., "등록금" → "학사" → "행정"
    """

    SECTION = "section"  # Regulation sections (조, 항, 호)
    PROCEDURE = "procedure"  # Procedure terms (신청, 절차, 방법)
    REQUIREMENT = "requirement"  # Requirement terms (자격, 요건, 조건)
    BENEFIT = "benefit"  # Benefit terms (혜택, 지급, 지원)
    DEADLINE = "deadline"  # Deadline terms (기한, 마감, 날짜)
    HYPERNYM = "hypernym"  # Hierarchical expansion (상위어)


@dataclass
class EntityMatch:
    """
    Result of entity recognition in a query.

    Attributes:
        entity_type: The type of entity recognized
        text: The matched text from the query
        start: Start position in query (character offset)
        end: End position in query (character offset)
        confidence: Recognition confidence (0.0 to 1.0)
        expanded_terms: Related terms for query expansion
    """

    entity_type: EntityType
    text: str
    start: int
    end: int
    confidence: float
    expanded_terms: List[str]

    def __post_init__(self):
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")

    @property
    def is_high_confidence(self) -> bool:
        """Check if match has high confidence (>= 0.8)."""
        return self.confidence >= 0.8

    @property
    def is_medium_confidence(self) -> bool:
        """Check if match has medium confidence (0.5-0.8)."""
        return 0.5 <= self.confidence < 0.8


@dataclass
class EntityRecognitionResult:
    """
    Complete result of entity recognition on a query.

    Attributes:
        original_query: The input query
        matches: All entity matches found (may be empty)
        has_entities: Whether any entities were recognized
        primary_entity: The highest-confidence entity (if any)
        total_expanded_terms: All unique expanded terms from all matches
    """

    original_query: str
    matches: List[EntityMatch]
    has_entities: bool
    primary_entity: Optional[EntityMatch]
    total_expanded_terms: List[str]

    @classmethod
    def from_matches(cls, original_query: str, matches: List[EntityMatch]):
        """Create result from match list."""
        if not matches:
            return cls(
                original_query=original_query,
                matches=[],
                has_entities=False,
                primary_entity=None,
                total_expanded_terms=[],
            )

        # Sort by confidence descending
        sorted_matches = sorted(matches, key=lambda m: m.confidence, reverse=True)

        # Collect all expanded terms (unique, preserve order)
        all_terms = []
        seen = set()
        for match in sorted_matches:
            for term in match.expanded_terms:
                if term not in seen:
                    all_terms.append(term)
                    seen.add(term)

        return cls(
            original_query=original_query,
            matches=sorted_matches,
            has_entities=True,
            primary_entity=sorted_matches[0],
            total_expanded_terms=all_terms,
        )

    def get_expanded_query(self, max_terms: int = 10) -> str:
        """
        Build expanded query with entity terms.

        Args:
            max_terms: Maximum number of expanded terms to include

        Returns:
            Expanded query string
        """
        if not self.total_expanded_terms:
            return self.original_query

        # Add original query terms that aren't already in expanded terms
        original_words = set(self.original_query.split())
        new_terms = [
            t for t in self.total_expanded_terms[:max_terms] if t not in original_words
        ]

        if new_terms:
            return f"{self.original_query} {' '.join(new_terms)}"
        return self.original_query
