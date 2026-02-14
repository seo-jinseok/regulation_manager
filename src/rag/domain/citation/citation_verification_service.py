"""
Citation Verification Service.

Provides citation extraction and standardization for Korean regulation references.
Implements pattern detection for citation verification requirements.
"""

from dataclasses import dataclass, replace
from typing import Optional

from .citation_patterns import CitationFormat, CitationPatterns


@dataclass
class ExtractedCitation:
    """
    Extracted citation with structured information.

    Attributes:
        regulation_name: Name of the regulation (e.g., "학칙")
        article: Article number (e.g., 25 from "제25조")
        paragraph: Paragraph number if present (e.g., 2 from "제2항")
        sub_article: Sub-article number if present (e.g., 2 from "제10조의2")
        original_text: Original citation text as it appeared
        content: Extracted content associated with citation (optional)
        is_verified: Whether citation has been verified against source
        format_type: Type of citation format detected
    """

    regulation_name: str
    article: int
    original_text: str
    paragraph: Optional[int] = None
    sub_article: Optional[int] = None
    content: str = ""
    is_verified: bool = False
    format_type: CitationFormat = CitationFormat.STANDARD

    def to_standard_format(self) -> str:
        """
        Format citation in standard format.

        Returns:
            Standardized citation string

        Examples:
            >>> citation = ExtractedCitation("학칙", 25, "「학칙」 제25조")
            >>> citation.to_standard_format()
            '「학칙」 제25조'
        """
        if self.sub_article is not None:
            return f"「{self.regulation_name}」 제{self.article}조의{self.sub_article}"
        elif self.paragraph is not None:
            return f"「{self.regulation_name}」 제{self.article}조 제{self.paragraph}항"
        else:
            return f"「{self.regulation_name}」 제{self.article}조"

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of citation
        """
        return {
            "regulation_name": self.regulation_name,
            "article": self.article,
            "paragraph": self.paragraph,
            "sub_article": self.sub_article,
            "original_text": self.original_text,
            "standard_format": self.to_standard_format(),
            "content": self.content,
            "is_verified": self.is_verified,
            "format_type": self.format_type.value,
        }


class CitationExtractor:
    """
    Extract and standardize citations from response text.

    Provides pattern detection for Korean regulation citations following
    the standard format: 「규정명」 제X조 [제X항]

    Features:
    - Extract multiple citations from text
    - Standardize citation formats
    - Handle sub-article references (제X조의Y)
    - Validate citation format compliance

    Examples:
        >>> extractor = CitationExtractor()
        >>> citations = extractor.extract("「학칙」 제25조에 따르면...")
        >>> len(citations)
        1
        >>> citations[0].regulation_name
        '학칙'
    """

    def __init__(self):
        """Initialize citation extractor."""
        self._patterns = CitationPatterns()

    def extract(self, text: str) -> list[ExtractedCitation]:
        """
        Extract all citations from text.

        Args:
            text: Response text to extract citations from

        Returns:
            List of ExtractedCitation objects

        Examples:
            >>> extractor = CitationExtractor()
            >>> citations = extractor.extract("「학칙」 제25조에 따르면...")
            >>> citations[0].regulation_name
            '학칙'
        """
        if not text:
            return []

        citations = []
        matches = self._patterns.find_all(text)

        for format_type, match in matches:
            citation = self._create_citation_from_match(match, format_type)
            if citation:
                citations.append(citation)

        return citations

    def standardize_format(self, citation_text: str) -> Optional[str]:
        """
        Standardize citation format.

        Args:
            citation_text: Citation text to standardize

        Returns:
            Standardized citation if valid, None otherwise

        Examples:
            >>> extractor = CitationExtractor()
            >>> extractor.standardize_format("「학칙」  제25조")
            '「학칙」 제25조'
        """
        return self._patterns.standardize(citation_text)

    def _create_citation_from_match(
        self, match: object, format_type: CitationFormat
    ) -> Optional[ExtractedCitation]:
        """
        Create ExtractedCitation from regex match.

        Args:
            match: Regex match object
            format_type: Type of citation format

        Returns:
            ExtractedCitation if valid, None otherwise
        """
        try:
            regulation_name = match.group(1)
            article = int(match.group(2))

            # Get original matched text
            original_text = match.group(0)

            # Extract additional components based on format
            paragraph = None
            sub_article = None

            if format_type == CitationFormat.WITH_SUB_ARTICLE:
                sub_article = int(match.group(3))
            elif format_type == CitationFormat.WITH_PARAGRAPH:
                paragraph = int(match.group(3))

            return ExtractedCitation(
                regulation_name=regulation_name,
                article=article,
                paragraph=paragraph,
                sub_article=sub_article,
                original_text=original_text,
                format_type=format_type,
            )
        except (ValueError, IndexError, AttributeError):
            return None


class CitationVerificationService:
    """
    Service for verifying and enhancing citations in RAG responses.

    Provides citation grounding verification, content inclusion,
    and sanitization for unverifiable citations.

    Features:
    - Verify citation grounding against source chunks
    - Include content from source chunks into citations
    - Sanitize unverifiable citations with fallback phrases

    Examples:
        >>> service = CitationVerificationService()
        >>> citation = ExtractedCitation("학칙", 25, "「학칙」 제25조")
        >>> chunks = [{"text": "...", "metadata": {"regulation_name": "학칙", "article": 25}}]
        >>> service.verify_grounding(citation, chunks)
        True
    """

    # Fallback phrases for unverifiable citations
    FALLBACK_PHRASES = [
        "관련 규정에 따르면",
        "해당 규정의 구체적 조항 확인이 필요합니다",
    ]

    def __init__(self):
        """Initialize citation verification service."""
        self._extractor = CitationExtractor()

    def extract_citations(self, text: str) -> list[ExtractedCitation]:
        """
        Extract all citations from text.

        Args:
            text: Response text to extract citations from

        Returns:
            List of ExtractedCitation objects
        """
        return self._extractor.extract(text)

    def verify_grounding(
        self, citation: ExtractedCitation, source_chunks: list[dict]
    ) -> bool:
        """
        Verify if citation exists in source chunks.

        Checks if the citation's regulation_name and article match
        any chunk in the source_chunks list.

        Args:
            citation: Citation to verify
            source_chunks: List of source chunks with metadata

        Returns:
            True if citation is found in sources, False otherwise

        Examples:
            >>> service = CitationVerificationService()
            >>> citation = ExtractedCitation("학칙", 25, "「학칙」 제25조")
            >>> chunks = [{"metadata": {"regulation_name": "학칙", "article": 25}}]
            >>> service.verify_grounding(citation, chunks)
            True
        """
        return self._find_matching_chunk(citation, source_chunks) is not None

    def include_content(
        self, citation: ExtractedCitation, source_chunks: list[dict]
    ) -> ExtractedCitation:
        """
        Include content from matching source chunk into citation.

        Finds the matching chunk and extracts the first sentence
        as key content for the citation.

        Args:
            citation: Citation to enhance with content
            source_chunks: List of source chunks with text and metadata

        Returns:
            ExtractedCitation with content field populated

        Examples:
            >>> service = CitationVerificationService()
            >>> citation = ExtractedCitation("학칙", 25, "「학칙」 제25조")
            >>> chunks = [{"text": "내용입니다.", "metadata": {"regulation_name": "학칙", "article": 25}}]
            >>> result = service.include_content(citation, chunks)
            >>> result.content
            '내용입니다.'
        """
        chunk = self._find_matching_chunk(citation, source_chunks)
        if chunk is None:
            return citation

        text = chunk.get("text", "")
        content = self._extract_first_sentence(text)
        if content:
            return replace(citation, content=content)

        return citation

    def sanitize_unverifiable(self, citation: ExtractedCitation) -> str:
        """
        Handle citations that cannot be verified.

        Returns a generalized phrase or fallback message for
        citations that could not be verified against sources.

        Args:
            citation: Unverifiable citation

        Returns:
            Fallback phrase or message string

        Examples:
            >>> service = CitationVerificationService()
            >>> citation = ExtractedCitation("학칙", 99, "「학칙」 제99조", is_verified=False)
            >>> service.sanitize_unverifiable(citation)
            '관련 규정에 따르면'
        """
        if citation.is_verified and citation.content:
            return citation.content

        # Return fallback phrase
        return self.FALLBACK_PHRASES[0]

    def _find_matching_chunk(
        self, citation: ExtractedCitation, source_chunks: list[dict]
    ) -> Optional[dict]:
        """
        Find chunk matching the citation.

        Args:
            citation: Citation to match
            source_chunks: List of source chunks with metadata

        Returns:
            Matching chunk dict or None if not found
        """
        if not source_chunks:
            return None

        for chunk in source_chunks:
            metadata = chunk.get("metadata", {})
            if (
                metadata.get("regulation_name") == citation.regulation_name
                and metadata.get("article") == citation.article
            ):
                return chunk

        return None

    def _extract_first_sentence(self, text: str) -> str:
        """
        Extract the first sentence from text.

        Args:
            text: Full text content

        Returns:
            First sentence or empty string
        """
        if not text:
            return ""

        # Korean sentence endings
        sentence_endings = ["。", ".", "！", "!", "？", "?"]
        first_sentence_end = -1

        for ending in sentence_endings:
            idx = text.find(ending)
            if idx != -1:
                if first_sentence_end == -1 or idx < first_sentence_end:
                    first_sentence_end = idx

        if first_sentence_end != -1:
            return text[: first_sentence_end + 1].strip()

        # No sentence ending found, return full text if reasonable length
        if len(text) <= 100:
            return text.strip()

        # Truncate at reasonable length
        return text[:100].strip() + "..."
