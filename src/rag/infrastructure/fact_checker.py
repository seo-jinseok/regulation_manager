"""
Fact Checker for LLM Answer Verification.

Extracts regulation/article citations from LLM answers and verifies
them against the actual database. Supports iterative correction.

Enhanced with source_chunks verification for RAG citation grounding.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..domain.repositories import IVectorStore

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A citation extracted from LLM answer."""

    regulation: str
    article: str
    article_sub: Optional[str] = None
    paragraph: Optional[str] = None
    original_text: str = ""  # The original matched text


@dataclass
class VerificationResult:
    """Result of verifying a single citation."""

    citation: Citation
    verified: bool
    matched_content: Optional[str] = None
    confidence: float = 0.0


@dataclass
class FactCheckResult:
    """Overall fact check result for an answer."""

    verified_count: int
    total_count: int
    results: List[VerificationResult] = field(default_factory=list)
    all_verified: bool = False

    @property
    def failed_citations(self) -> List[Citation]:
        """Return citations that failed verification."""
        return [r.citation for r in self.results if not r.verified]

    @property
    def success_rate(self) -> float:
        """Return success rate as percentage."""
        if self.total_count == 0:
            return 100.0
        return (self.verified_count / self.total_count) * 100


class FactChecker:
    """
    Verifies regulation citations in LLM answers.

    Extracts citations like 「교원인사규정」 제36조 and verifies
    they exist in the vector store.

    Enhanced with source_chunks verification for RAG citation grounding.
    When source_chunks is provided, uses CitationVerificationService for
    direct verification against retrieved source chunks.

    Backward Compatibility:
        - verify_citation(citation) - uses vector store search (original behavior)
        - verify_citation(citation, source_chunks) - uses CitationVerificationService
        - check(answer_text) - uses vector store search (original behavior)
        - check(answer_text, source_chunks) - uses CitationVerificationService
    """

    # Patterns for extracting citations
    CITATION_PATTERNS = [
        # 「규정명」 제N조 제M항 N호
        r"「([^」]+)」\s*제?(\d+)조(?:의(\d+))?\s*(?:제?(\d+)항)?(?:\s*(\d+)호)?",
        # 규정명 제N조 (without quotes) - includes 학칙
        r"([가-힣]+(?:규정|규칙|세칙|지침|학칙|요령|내규))\s*제?(\d+)조(?:의(\d+))?\s*(?:제?(\d+)항)?",
        # 학칙만 단독으로 (학칙 제N조)
        r"(학칙)\s+제?(\d+)조(?:의(\d+))?\s*(?:제?(\d+)항)?",
    ]

    def __init__(self, store: "IVectorStore"):
        """
        Initialize fact checker.

        Args:
            store: Vector store for verification queries.
        """
        self.store = store
        self._verification_service = None  # Lazy initialization

    def _get_verification_service(self):
        """
        Get or create CitationVerificationService (lazy initialization).

        Returns:
            CitationVerificationService instance.
        """
        if self._verification_service is None:
            from ..domain.citation.citation_verification_service import (
                CitationVerificationService,
            )

            self._verification_service = CitationVerificationService()
        return self._verification_service

    def extract_citations(self, text: str) -> List[Citation]:
        """
        Extract regulation/article citations from text.

        Args:
            text: LLM answer text.

        Returns:
            List of Citation objects.
        """
        citations = []
        seen = set()

        for pattern in self.CITATION_PATTERNS:
            for match in re.finditer(pattern, text):
                groups = match.groups()
                reg_name = groups[0]
                article = groups[1]
                article_sub = groups[2] if len(groups) > 2 else None
                paragraph = groups[3] if len(groups) > 3 else None

                # Deduplication key
                key = (reg_name, article, article_sub)
                if key in seen:
                    continue
                seen.add(key)

                citations.append(
                    Citation(
                        regulation=reg_name,
                        article=article,
                        article_sub=article_sub,
                        paragraph=paragraph,
                        original_text=match.group(0),
                    )
                )

        return citations

    def verify_citation(
        self, citation: Citation, source_chunks: Optional[List[dict]] = None
    ) -> VerificationResult:
        """
        Verify a single citation against the database or source chunks.

        When source_chunks is provided, uses CitationVerificationService
        for direct verification against the retrieved source chunks.
        Otherwise, falls back to vector store search (original behavior).

        Args:
            citation: Citation to verify.
            source_chunks: Optional list of source chunks with metadata.
                Each chunk should have:
                - "text": The chunk text content
                - "metadata": Dict with "regulation_name" and "article" fields

        Returns:
            VerificationResult with verification status.

        Examples:
            # Original behavior (vector store search)
            >>> result = checker.verify_citation(citation)

            # Enhanced behavior (source chunks verification)
            >>> chunks = [{"text": "...", "metadata": {"regulation_name": "학칙", "article": 25}}]
            >>> result = checker.verify_citation(citation, source_chunks=chunks)
        """
        # If source_chunks provided, use CitationVerificationService
        if source_chunks is not None:
            return self._verify_with_source_chunks(citation, source_chunks)

        # Fallback to original vector store verification
        return self._verify_with_store(citation)

    def _verify_with_source_chunks(
        self, citation: Citation, source_chunks: List[dict]
    ) -> VerificationResult:
        """
        Verify citation using CitationVerificationService against source chunks.

        Args:
            citation: Citation to verify.
            source_chunks: List of source chunks from RAG retrieval.

        Returns:
            VerificationResult with verification status.
        """

        service = self._get_verification_service()

        # Convert Citation to ExtractedCitation for service
        extracted = self._citation_to_extracted(citation)

        # Verify using the service
        is_verified = service.verify_grounding(extracted, source_chunks)

        if is_verified:
            # Get matching chunk for content
            enhanced = service.include_content(extracted, source_chunks)
            return VerificationResult(
                citation=citation,
                verified=True,
                matched_content=enhanced.content[:200] if enhanced.content else None,
                confidence=1.0,  # High confidence for direct chunk match
            )

        return VerificationResult(
            citation=citation,
            verified=False,
            matched_content=None,
            confidence=0.0,
        )

    def _verify_with_store(self, citation: Citation) -> VerificationResult:
        """
        Verify citation using vector store search (original behavior).

        Args:
            citation: Citation to verify.

        Returns:
            VerificationResult with verification status.
        """
        from ..domain.value_objects import Query

        # Build search query
        if citation.article_sub:
            query_text = (
                f"{citation.regulation} 제{citation.article}조의{citation.article_sub}"
            )
        else:
            query_text = f"{citation.regulation} 제{citation.article}조"

        # Search in vector store
        results = self.store.search(Query(text=query_text), top_k=10)

        # Normalize regulation name for matching
        reg_normalized = citation.regulation.replace(" ", "").replace("·", "")
        article_pattern = f"제{citation.article}조"

        for result in results:
            chunk = result.chunk
            # Build searchable text from chunk
            chunk_text = f"{' '.join(chunk.parent_path)} {chunk.title} {chunk.text}"
            chunk_normalized = chunk_text.replace(" ", "").replace("·", "")

            # Check regulation name match
            reg_match = (
                reg_normalized in chunk_normalized or citation.regulation in chunk_text
            )

            # Check article number match
            article_match = article_pattern in chunk_text

            if reg_match and article_match:
                return VerificationResult(
                    citation=citation,
                    verified=True,
                    matched_content=chunk.text[:200] if chunk.text else None,
                    confidence=result.score,
                )

        return VerificationResult(
            citation=citation,
            verified=False,
            matched_content=None,
            confidence=0.0,
        )

    def _citation_to_extracted(self, citation: Citation):
        """
        Convert Citation to ExtractedCitation for service compatibility.

        Args:
            citation: Citation object to convert.

        Returns:
            ExtractedCitation object.
        """
        from ..domain.citation.citation_verification_service import ExtractedCitation

        # Parse article number (handle string to int conversion)
        try:
            article_num = int(citation.article)
        except ValueError:
            article_num = 0

        # Parse paragraph if present
        paragraph_num = None
        if citation.paragraph:
            try:
                paragraph_num = int(citation.paragraph)
            except ValueError:
                pass

        # Parse sub_article if present
        sub_article_num = None
        if citation.article_sub:
            try:
                sub_article_num = int(citation.article_sub)
            except ValueError:
                pass

        return ExtractedCitation(
            regulation_name=citation.regulation,
            article=article_num,
            paragraph=paragraph_num,
            sub_article=sub_article_num,
            original_text=citation.original_text,
        )

    def check(
        self, answer_text: str, source_chunks: Optional[List[dict]] = None
    ) -> FactCheckResult:
        """
        Perform full fact check on an answer.

        When source_chunks is provided, uses CitationVerificationService
        for direct verification against the retrieved source chunks.
        Otherwise, falls back to vector store search (original behavior).

        Args:
            answer_text: LLM-generated answer text.
            source_chunks: Optional list of source chunks with metadata.
                Each chunk should have:
                - "text": The chunk text content
                - "metadata": Dict with "regulation_name" and "article" fields

        Returns:
            FactCheckResult with all verification results.

        Examples:
            # Original behavior (vector store search)
            >>> result = checker.check(answer_text)

            # Enhanced behavior (source chunks verification)
            >>> chunks = [{"text": "...", "metadata": {"regulation_name": "학칙", "article": 25}}]
            >>> result = checker.check(answer_text, source_chunks=chunks)
        """
        citations = self.extract_citations(answer_text)

        if not citations:
            return FactCheckResult(
                verified_count=0,
                total_count=0,
                results=[],
                all_verified=True,  # No citations = nothing to fail
            )

        results = []
        verified_count = 0

        for citation in citations:
            result = self.verify_citation(citation, source_chunks=source_chunks)
            results.append(result)
            if result.verified:
                verified_count += 1

        return FactCheckResult(
            verified_count=verified_count,
            total_count=len(citations),
            results=results,
            all_verified=(verified_count == len(citations)),
        )

    def build_correction_feedback(self, fact_check_result: FactCheckResult) -> str:
        """
        Build feedback message for LLM to correct failed citations.

        Args:
            fact_check_result: Result from fact check.

        Returns:
            Feedback string for LLM correction prompt.
        """
        if fact_check_result.all_verified:
            return ""

        failed = fact_check_result.failed_citations
        feedback_parts = [
            "⚠️ 다음 인용은 데이터베이스에서 확인되지 않았습니다. "
            "해당 조항을 언급하지 말거나, 정확한 조항으로 수정해주세요:\n"
        ]

        for citation in failed:
            feedback_parts.append(f"  - {citation.original_text}")

        feedback_parts.append(
            "\n규정에 명시되지 않은 조항 번호를 만들어내지 마세요. "
            "확인된 정보만 사용하세요."
        )

        return "\n".join(feedback_parts)
