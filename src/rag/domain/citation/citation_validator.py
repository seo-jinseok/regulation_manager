"""
Citation Validator Service.

Validates citations against actual source documents to detect
hallucinated or incorrect references in LLM responses.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from ..entities import Chunk
    from ..repositories import IVectorStore

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status of citation validation."""

    VALID = "valid"  # Citation matches source chunk
    HALUCINATED = "hallucinated"  # Citation doesn't exist in database
    MISMATCH = "mismatch"  # Article number doesn't match
    AMBIGUOUS = "ambiguous"  # Multiple possible matches
    MISSING_REGULATION = "missing_regulation"  # Regulation name not found


@dataclass
class ValidationResult:
    """
    Result of validating a citation against source chunks.

    Attributes:
        citation_text: Original citation text (e.g., "「교원인사규정」 제26조")
        status: Validation status
        matched_chunk: Matched chunk if found
        confidence: Confidence score (0.0 to 1.0)
        error_message: Detailed error message if validation failed
    """

    citation_text: str
    status: ValidationStatus
    matched_chunk: Optional["Chunk"] = None
    confidence: float = 0.0
    error_message: str = ""

    @property
    def is_valid(self) -> bool:
        """Check if citation is valid."""
        return self.status == ValidationStatus.VALID

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "citation_text": self.citation_text,
            "status": self.status.value,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "error_message": self.error_message,
            "matched_chunk_id": self.matched_chunk.id if self.matched_chunk else None,
        }


class CitationValidator:
    """
    Validates citations against source documents.

    Features:
    - Check if cited articles exist in database
    - Validate article numbers match source chunks
    - Detect hallucinated citations
    - Support for article, sub-article, table, form references
    """

    # Regulation suffixes for name extraction
    REGULATION_SUFFIXES = [
        "규정",
        "규칙",
        "세칙",
        "지침",
        "요령",
        "내규",
        "학칙",
    ]

    def __init__(self, store: "IVectorStore"):
        """
        Initialize citation validator.

        Args:
            store: Vector store for source document queries
        """
        from .article_number_extractor import ArticleNumberExtractor

        self.store = store
        self._extractor = ArticleNumberExtractor()

    def validate_citation(
        self, regulation: str, article_number: str, top_k: int = 10
    ) -> ValidationResult:
        """
        Validate a single citation against source documents.

        Args:
            regulation: Regulation name (e.g., "교원인사규정")
            article_number: Article number (e.g., "제26조", "제10조의2", "별표1")
            top_k: Number of search results to check

        Returns:
            ValidationResult with validation status
        """
        from ..value_objects import Query

        # Build search query
        query_text = f"{regulation} {article_number}"
        query = Query(text=query_text)

        # Search for matching chunks
        results = self.store.search(query, top_k=top_k)

        if not results:
            return ValidationResult(
                citation_text=f"「{regulation}」 {article_number}",
                status=ValidationStatus.HALUCINATED,
                confidence=0.0,
                error_message=f"No matches found for {query_text}",
            )

        # Check for exact article number match
        for result in results:
            chunk = result.chunk

            # Check article_number field first (enhanced validation)
            if chunk.article_number and chunk.article_number == article_number:
                # Verify regulation name matches
                if self._regulation_matches(chunk, regulation):
                    return ValidationResult(
                        citation_text=f"「{regulation}」 {article_number}",
                        status=ValidationStatus.VALID,
                        matched_chunk=chunk,
                        confidence=result.score,
                    )

            # Fallback: check title for article number
            if article_number in chunk.title:
                if self._regulation_matches(chunk, regulation):
                    return ValidationResult(
                        citation_text=f"「{regulation}」 {article_number}",
                        status=ValidationStatus.VALID,
                        matched_chunk=chunk,
                        confidence=result.score,
                    )

        # No exact match found
        best_match = results[0].chunk
        return ValidationResult(
            citation_text=f"「{regulation}」 {article_number}",
            status=ValidationStatus.MISMATCH,
            matched_chunk=best_match,
            confidence=results[0].score,
            error_message=f"Article {article_number} not found in {regulation}",
        )

    def validate_answer(
        self, answer_text: str, source_chunks: List["Chunk"]
    ) -> tuple[List[ValidationResult], List[str]]:
        """
        Validate all citations in an answer against source chunks.

        Args:
            answer_text: LLM-generated answer text
            source_chunks: Source chunks used for answer generation

        Returns:
            Tuple of (validation_results, citation_texts)
        """
        from ..infrastructure.fact_checker import FactChecker

        # Extract citations from answer
        fact_checker = FactChecker(self.store)
        citations = fact_checker.extract_citations(answer_text)

        if not citations:
            return [], []

        # Validate each citation
        results = []
        citation_texts = []

        for citation in citations:
            # Determine article number
            article_number = f"제{citation.article}조"
            if citation.article_sub:
                article_number = f"제{citation.article}조의{citation.article_sub}"

            result = self.validate_citation(citation.regulation, article_number)
            results.append(result)
            citation_texts.append(result.citation_text)

        return results, citation_texts

    def detect_hallucinations(
        self, answer_text: str, source_chunks: List["Chunk"]
    ) -> List[ValidationResult]:
        """
        Detect hallucinated citations in answer.

        Args:
            answer_text: LLM-generated answer text
            source_chunks: Source chunks used for answer generation

        Returns:
            List of ValidationResult for hallucinated citations
        """
        results, _ = self.validate_answer(answer_text, source_chunks)

        # Filter for hallucinated and mismatched citations
        hallucinations = [
            r
            for r in results
            if r.status in (ValidationStatus.HALUCINATED, ValidationStatus.MISMATCH)
        ]

        return hallucinations

    def _regulation_matches(self, chunk: "Chunk", regulation: str) -> bool:
        """
        Check if regulation name matches chunk.

        Args:
            chunk: Source chunk
            regulation: Regulation name to match

        Returns:
            True if regulation matches chunk's parent_path
        """
        if not chunk.parent_path:
            return False

        # Direct match
        if regulation in chunk.parent_path:
            return True

        # Normalize and check (remove spaces, dots)
        reg_normalized = regulation.replace(" ", "").replace("·", "")
        for path_item in chunk.parent_path:
            path_normalized = path_item.replace(" ", "").replace("·", "")
            if reg_normalized in path_normalized or path_normalized in reg_normalized:
                return True

        return False

    def get_valid_article_numbers(
        self, regulation: str, chunks: List["Chunk"]
    ) -> Set[str]:
        """
        Get all valid article numbers for a regulation.

        Args:
            regulation: Regulation name
            chunks: List of chunks to filter

        Returns:
            Set of valid article numbers
        """
        valid_articles = set()

        for chunk in chunks:
            # Check if chunk belongs to regulation
            if not self._regulation_matches(chunk, regulation):
                continue

            # Extract article number
            if chunk.article_number:
                valid_articles.add(chunk.article_number)
            elif chunk.title:
                result = self._extractor.extract(chunk.title)
                if result:
                    valid_articles.add(result.to_citation_format())

        return valid_articles

    def build_validation_report(self, results: List[ValidationResult]) -> str:
        """
        Build human-readable validation report.

        Args:
            results: List of validation results

        Returns:
            Formatted report string
        """
        if not results:
            return "No citations to validate."

        lines = ["📋 Citation Validation Report", ""]

        valid_count = sum(1 for r in results if r.is_valid)
        total_count = len(results)

        lines.append(f"Total citations: {total_count}")
        lines.append(f"Valid: {valid_count}")
        lines.append(f"Invalid: {total_count - valid_count}")
        lines.append("")

        # Group by status
        by_status: dict[ValidationStatus, List[ValidationResult]] = {}
        for result in results:
            status = result.status
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(result)

        # List valid citations
        if ValidationStatus.VALID in by_status:
            lines.append("✅ Valid Citations:")
            for result in by_status[ValidationStatus.VALID]:
                lines.append(
                    f"  - {result.citation_text} (confidence: {result.confidence:.2f})"
                )
            lines.append("")

        # List invalid citations
        invalid_statuses = [
            ValidationStatus.HALUCINATED,
            ValidationStatus.MISMATCH,
            ValidationStatus.MISSING_REGULATION,
        ]

        for status in invalid_statuses:
            if status in by_status:
                status_label = status.value.replace("_", " ").title()
                lines.append(f"❌ {status_label}:")
                for result in by_status[status]:
                    lines.append(f"  - {result.citation_text}")
                    if result.error_message:
                        lines.append(f"    Reason: {result.error_message}")
                lines.append("")

        return "\n".join(lines)
