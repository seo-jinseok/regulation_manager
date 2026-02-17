"""
Citation Enhancer Service.

Enhances citations with article numbers and validates against source chunks.
Provides improved citation formatting and validation for RAG answers.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..entities import Chunk

logger = logging.getLogger(__name__)


@dataclass
class EnhancedCitation:
    """
    Enhanced citation with structured article information.

    Attributes:
        regulation: Regulation name
        article_number: Article number (e.g., "제26조", "제10조의2", "별표1")
        chunk_id: ID of source chunk
        confidence: Confidence score (0.0 to 1.0)
        title: Chunk title
        text: Chunk text excerpt
    """

    regulation: str
    article_number: str
    chunk_id: str
    confidence: float
    title: str = ""
    text: str = ""

    def format(self) -> str:
        """
        Format citation for display in responses.

        Examples:
            - "「직원복무규정」 제26조"
            - "「학칙」 제10조의2"
            - "별표1 (직원급별 봉급표)"
        """
        if self.article_number.startswith(("별표", "서식")):
            # Tables and forms don't use regulation quotes
            return (
                f"{self.article_number} ({self.title})"
                if self.title
                else self.article_number
            )
        return f"「{self.regulation}」 {self.article_number}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "regulation": self.regulation,
            "article_number": self.article_number,
            "chunk_id": self.chunk_id,
            "confidence": self.confidence,
            "title": self.title,
            "text": self.text[:200] if self.text else "",  # Truncate long text
        }


class CitationEnhancer:
    """
    Enhances and validates citations for RAG responses.

    Features:
    - Article number extraction from chunks
    - Citation formatting with regulation names
    - Validation against source chunks
    - Support for 별표, 서식 references
    - Regulation title extraction for enhanced formatting
    - Rule code validation for accuracy
    - Confidence scoring based on pattern match quality
    """

    # Regulation suffixes for name extraction and validation
    REGULATION_SUFFIXES = [
        "규정",
        "규칙",
        "세칙",
        "지침",
        "요령",
        "내규",
        "학칙",
    ]

    # Minimum confidence threshold for valid citations
    MIN_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self):
        """Initialize citation enhancer."""
        from .article_number_extractor import ArticleNumberExtractor

        self._extractor = ArticleNumberExtractor()

    def calculate_confidence(
        self, chunk: "Chunk", article_number: str
    ) -> float:
        """
        Calculate confidence score for a citation based on multiple factors.

        Confidence is based on:
        - Pattern match quality (0.4): Is article in title/text?
        - Metadata presence (0.3): Is article_number field set?
        - Source validation (0.3): Does article exist in chunk text?

        Args:
            chunk: Source chunk
            article_number: Article number to validate

        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.0

        # Factor 1: Article in title (0.2)
        if chunk.title and article_number in chunk.title:
            confidence += 0.2

        # Factor 2: Article in text (0.2)
        if chunk.text and article_number in chunk.text:
            confidence += 0.2

        # Factor 3: Metadata field set (0.3)
        if chunk.article_number and chunk.article_number == article_number:
            confidence += 0.3

        # Factor 4: Valid rule_code pattern (0.15)
        if chunk.rule_code and self.validate_rule_code(chunk):
            confidence += 0.15

        # Factor 5: Regulation name present (0.15)
        if chunk.parent_path and len(chunk.parent_path) > 0:
            confidence += 0.15

        return min(confidence, 1.0)

    def calculate_confidence_with_context(
        self, chunk: "Chunk", article_number: str, query: str
    ) -> float:
        """
        Calculate confidence score with query context relevance.

        Extends base confidence by considering query-chunk relevance.

        Args:
            chunk: Source chunk
            article_number: Article number to validate
            query: User query for context relevance

        Returns:
            Context-aware confidence score between 0.0 and 1.0
        """
        base_confidence = self.calculate_confidence(chunk, article_number)

        # Boost confidence if query terms appear in chunk
        if query and chunk.text:
            # Simple keyword overlap check
            query_words = set(query.split())
            chunk_words = set(chunk.text.split())
            overlap = len(query_words & chunk_words)

            # Boost by up to 0.1 based on overlap (max 3 significant words)
            boost = min(overlap / 30.0, 0.1)
            base_confidence = min(base_confidence + boost, 1.0)

        return base_confidence

    def validate_citation_in_source(self, citation: EnhancedCitation) -> bool:
        """
        Validate that citation exists in source text.

        Args:
            citation: Enhanced citation to validate

        Returns:
            True if citation article_number found in text, False otherwise
        """
        if not citation.text:
            return False

        # Check if article number appears in text
        article_number = citation.article_number

        # Handle combined citations (제26조제1항 -> check for 제26조 and ①)
        if "제" in article_number and "항" in article_number:
            # Extract base article and paragraph
            import re
            match = re.match(r"제(\d+)조제(\d+)항", article_number)
            if match:
                base_article = f"제{match.group(1)}조"
                para_num = match.group(2)
                # Check for base article and paragraph marker (①, ②, etc.)
                para_markers = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
                para_idx = int(para_num) - 1 if int(para_num) <= 10 else -1

                has_article = base_article in citation.text
                has_para = (
                    para_idx >= 0
                    and para_idx < len(para_markers)
                    and para_markers[para_idx] in citation.text
                )
                return has_article and has_para

        # Simple check: article number in text
        return article_number in citation.text

    def filter_low_confidence(
        self, citations: List[EnhancedCitation]
    ) -> List[EnhancedCitation]:
        """
        Filter out citations with confidence below threshold.

        Args:
            citations: List of enhanced citations

        Returns:
            Filtered list with only high-confidence citations
        """
        return [
            c for c in citations if c.confidence >= self.MIN_CONFIDENCE_THRESHOLD
        ]

    def enhance_citation(
        self, chunk: "Chunk", confidence: float = 1.0
    ) -> Optional[EnhancedCitation]:
        """
        Enhance a single chunk with citation information.

        Args:
            chunk: Source chunk
            confidence: Confidence score for the citation

        Returns:
            EnhancedCitation if chunk has article info, None otherwise
        """
        # Get regulation name from parent_path
        regulation = chunk.parent_path[0] if chunk.parent_path else ""

        # Get article number
        article_number = chunk.article_number

        # If no article_number in field, try to extract from title
        if not article_number and chunk.title:
            result = self._extractor.extract(chunk.title)
            if result:
                article_number = result.to_citation_format()

        # Require both regulation and article number for valid citation
        if not regulation or not article_number:
            logger.debug(
                f"Cannot enhance citation: regulation={regulation}, "
                f"article_number={article_number}, chunk_id={chunk.id}"
            )
            return None

        return EnhancedCitation(
            regulation=regulation,
            article_number=article_number,
            chunk_id=chunk.id,
            confidence=confidence,
            title=chunk.title,
            text=chunk.text,
        )

    def enhance_citations(
        self, chunks: List["Chunk"], confidences: Optional[List[float]] = None
    ) -> List[EnhancedCitation]:
        """
        Enhance multiple chunks with citation information.

        Args:
            chunks: List of source chunks
            confidences: Optional confidence scores for each chunk

        Returns:
            List of EnhancedCitation objects (may be shorter than input)
        """
        if confidences is None:
            confidences = [1.0] * len(chunks)

        if len(chunks) != len(confidences):
            logger.warning(
                "chunks and confidences length mismatch, using default confidence"
            )
            confidences = [1.0] * len(chunks)

        enhanced = []
        for chunk, conf in zip(chunks, confidences, strict=True):
            citation = self.enhance_citation(chunk, confidence=conf)
            if citation is not None:
                enhanced.append(citation)

        return enhanced

    def format_citations(self, citations: List[EnhancedCitation]) -> str:
        """
        Format list of citations for response text.

        Args:
            citations: List of enhanced citations

        Returns:
            Formatted citation string

        Examples:
            Single citation: "「직원복무규정」 제26조"
            Multiple citations: "「직원복무규정」 제26조, 「학칙」 제15조"
        """
        if not citations:
            return ""

        formatted = [c.format() for c in citations]
        return ", ".join(formatted)

    def group_by_regulation(
        self, citations: List[EnhancedCitation]
    ) -> dict[str, List[EnhancedCitation]]:
        """
        Group citations by regulation name.

        Args:
            citations: List of enhanced citations

        Returns:
            Dict mapping regulation name to list of citations
        """
        grouped: dict[str, List[EnhancedCitation]] = {}
        for citation in citations:
            reg = citation.regulation
            if reg not in grouped:
                grouped[reg] = []
            grouped[reg].append(citation)
        return grouped

    def sort_by_article_number(
        self, citations: List[EnhancedCitation]
    ) -> List[EnhancedCitation]:
        """
        Sort citations by article number.

        Args:
            citations: List of enhanced citations

        Returns:
            Sorted list of citations
        """

        def extract_number(citation: EnhancedCitation) -> int:
            """Extract numeric part from article number for sorting."""
            try:
                # Extract first number from article_number
                import re

                match = re.search(r"\d+", citation.article_number)
                if match:
                    return int(match.group())
            except (ValueError, AttributeError):
                pass
            return 0

        return sorted(citations, key=extract_number)

    def deduplicate_citations(
        self, citations: List[EnhancedCitation]
    ) -> List[EnhancedCitation]:
        """
        Remove duplicate citations (same regulation + article_number).

        Args:
            citations: List of enhanced citations

        Returns:
            Deduplicated list (first occurrence kept)
        """
        seen = set()
        unique = []

        for citation in citations:
            key = (citation.regulation, citation.article_number)
            if key not in seen:
                seen.add(key)
                unique.append(citation)

        return unique

    def extract_regulation_title(self, chunk: "Chunk") -> str:
        """
        Extract regulation title from chunk's parent_path.

        Args:
            chunk: Source chunk

        Returns:
            Regulation title (e.g., "교원인사규정")

        Examples:
            >>> chunk.parent_path = ["교원인사규정", "제2장", "제26조"]
            >>> enhancer.extract_regulation_title(chunk)
            "교원인사규정"
        """
        if not chunk.parent_path:
            return ""

        # Return first element of parent_path as regulation name
        return chunk.parent_path[0] if chunk.parent_path else ""

    def validate_rule_code(self, chunk: "Chunk") -> bool:
        """
        Validate that chunk has a valid rule_code.

        A valid rule_code should:
        - Not be empty
        - Follow the pattern: regulation_article_number (e.g., "교원인사규정_제26조")

        Args:
            chunk: Source chunk

        Returns:
            True if rule_code is valid, False otherwise
        """
        if not chunk.rule_code or not chunk.rule_code.strip():
            return False

        # Check if rule_code contains regulation and article info
        rule_code = chunk.rule_code.strip()

        # Valid patterns:
        # - "규정명_제X조"
        # - "규정명_별표X"
        # - "규정명_서식X"
        has_underscore = "_" in rule_code
        has_article = "제" in rule_code and "조" in rule_code
        has_table = any(prefix in rule_code for prefix in ["별표", "서식"])

        return has_underscore and (has_article or has_table)

    def extract_citations(self, chunks: List["Chunk"]) -> List[EnhancedCitation]:
        """
        Extract enhanced citations from chunks with validation.

        This method extends enhance_citations by:
        1. Validating rule codes before creating citations
        2. Extracting regulation titles from parent_path
        3. Filtering out chunks with invalid citations

        Args:
            chunks: List of source chunks

        Returns:
            List of EnhancedCitation objects (only valid citations)

        Examples:
            >>> chunks = [chunk1, chunk2, chunk3]
            >>> citations = enhancer.extract_citations(chunks)
            >>> len(citations)
            2  # Only chunks with valid rule_codes
        """
        valid_citations = []

        for chunk in chunks:
            # Validate rule code first
            if not self.validate_rule_code(chunk):
                logger.debug(
                    f"Skipping chunk {chunk.id} due to invalid rule_code: {chunk.rule_code}"
                )
                continue

            # Try to enhance citation
            citation = self.enhance_citation(chunk, confidence=1.0)
            if citation is not None:
                # Ensure regulation title is properly extracted
                if not citation.regulation:
                    regulation = self.extract_regulation_title(chunk)
                    if regulation:
                        citation.regulation = regulation

                valid_citations.append(citation)

        return valid_citations

    def format_citation_with_validation(
        self, citation: EnhancedCitation, validate: bool = True
    ) -> str:
        """
        Format citation with optional validation.

        Args:
            citation: Enhanced citation to format
            validate: Whether to validate before formatting

        Returns:
            Formatted citation string

        Examples:
            >>> citation = EnhancedCitation(
            ...     regulation="교원인사규정",
            ...     article_number="제26조",
            ...     chunk_id="c1",
            ...     confidence=0.9
            ... )
            >>> enhancer.format_citation_with_validation(citation)
            "「교원인사규정」 제26조"
        """
        if validate:
            # Validate citation has required fields
            if not citation.regulation or not citation.article_number:
                logger.warning(
                    f"Invalid citation: regulation={citation.regulation}, "
                    f"article_number={citation.article_number}"
                )
                return ""

        return citation.format()
