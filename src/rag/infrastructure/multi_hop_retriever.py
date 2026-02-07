"""
Multi-Hop Retrieval for Regulation RAG System.

Implements SPEC-RAG-SEARCH-001 TAG-004: Basic Multi-Hop Retrieval.

Follows citation references in regulation documents to retrieve related content:
- Detects "제X조" references
- Follows "관련 규정" links
- Filters by relevance threshold
- Maximum 2-hop depth to prevent infinite loops

Features:
- Citation pattern extraction from document text
- Reference following to related documents
- Deduplication of visited documents
- Relevance-based filtering
- Cycle detection
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Set

from ..domain.entities import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """
    Citation reference extracted from document content.

    Attributes:
        source_doc_id: ID of the document containing the citation
        target_section: Section number referenced (e.g., "15" for "제15조")
        target_text: Text of the citation (e.g., "제15조")
        citation_type: Type of citation ("section_ref", "regulation_ref")
        start_pos: Start position of citation in source text
        end_pos: End position of citation in source text
    """

    source_doc_id: str
    target_section: Optional[str]
    target_text: str
    citation_type: str
    start_pos: int
    end_pos: int

    def __post_init__(self):
        """Validate citation data."""
        if self.citation_type not in ("section_ref", "regulation_ref"):
            raise ValueError(f"Invalid citation type: {self.citation_type}")


@dataclass
class HopRetrievalResult:
    """
    Result of multi-hop retrieval operation.

    Attributes:
        original_results: Search results from initial query (before hopping)
        hop_results: Additional results retrieved by following citations
        all_results: Combined and deduplicated results
        hops_performed: Number of hops actually performed (0, 1, or 2)
        citations_found: List of citations that were found and followed
        visited_docs: Set of document IDs visited during hopping
    """

    original_results: List[SearchResult]
    hop_results: List[SearchResult]
    all_results: List[SearchResult]
    hops_performed: int
    citations_found: List[Citation]
    visited_docs: Set[str]


class MultiHopRetriever:
    """
    Basic multi-hop retriever for citation following.

    Part of SPEC-RAG-SEARCH-001 TAG-004: Basic Multi-Hop Retrieval.

    This class implements citation-based multi-hop retrieval to find
    related regulation documents by following section references.

    Citation patterns (REQ-MH-001 ~ REQ-MH-003):
    - Section references: "제15조", "제15항", "제15조의2"
    - Regulation references: "관련 규정", "상세 내용은 XX규정"

    Example:
        retriever = MultiHopRetriever(vector_store)
        results = retriever.retrieve(
            query="장학금 신청",
            initial_results=[...],  # From first search
            top_k=10
        )
        # Returns results with cited sections included
    """

    # Maximum hops to prevent infinite loops (REQ-MH-003, REQ-MH-006)
    MAX_HOPS = 2

    # Relevance threshold for filtering citations (REQ-MH-008)
    RELEVANCE_THRESHOLD = 0.5

    # Pattern for section references (REQ-MH-001)
    SECTION_PATTERNS = [
        re.compile(r"제(\d+)조"),  # 제15조
        re.compile(r"제(\d+)항"),  # 제2항
        re.compile(r"제(\d+)호"),  # 제1호
        re.compile(r"제\s*(\d+)\s*조"),  # With spaces: 제 15 조
        re.compile(r"제\s*(\d+)\s*조의(\d+)"),  # 제15조의2
    ]

    # Pattern for regulation references (REQ-MH-002)
    REGULATION_REF_PATTERNS = [
        re.compile(r"관련\s*규정"),
        re.compile(r"상세\s*내용은\s*[^.]*?규정"),
        re.compile(r"[^.]*?규정\s*참조"),
        re.compile(r"[^.]*?규정\s*에\s*따름"),
    ]

    def __init__(self, vector_store, relevance_threshold: float = RELEVANCE_THRESHOLD):
        """
        Initialize multi-hop retriever.

        Args:
            vector_store: Vector store for searching cited documents
            relevance_threshold: Minimum relevance score for citations (default: 0.5)
        """
        self._vector_store = vector_store
        self._relevance_threshold = relevance_threshold

    def retrieve(
        self,
        query: str,
        initial_results: List[SearchResult],
        top_k: int = 10,
        max_hops: int = MAX_HOPS,
    ) -> HopRetrievalResult:
        """
        Perform multi-hop retrieval by following citations.

        This is the main entry point for multi-hop retrieval.

        Args:
            query: Original search query (for relevance checking)
            initial_results: Initial search results from first query
            top_k: Maximum total results to return
            max_hops: Maximum number of hops to perform (default: 2)

        Returns:
            HopRetrievalResult with original + hop results

        Example:
            retriever = MultiHopRetriever(vector_store)
            initial = vector_store.search("장학금", top_k=5)
            result = retriever.retrieve("장학금 신청", initial, top_k=10)
            # result.all_results includes original + cited sections
        """
        if not initial_results:
            return HopRetrievalResult(
                original_results=[],
                hop_results=[],
                all_results=[],
                hops_performed=0,
                citations_found=[],
                visited_docs=set(),
            )

        # Track visited documents to prevent cycles (REQ-MH-009)
        visited_docs = {r.chunk.id for r in initial_results}
        all_hop_results: List[SearchResult] = []
        all_citations: List[Citation] = []

        # Perform hops up to max_hops
        for hop_num in range(max_hops):
            # Determine which documents to extract citations from
            source_results = initial_results if hop_num == 0 else all_hop_results

            # Extract citations from current results
            citations = self._extract_citations(source_results)
            if not citations:
                # No more citations to follow
                break

            all_citations.extend(citations)

            # Follow citations to retrieve new documents
            hop_results = self._follow_citations(query, citations, visited_docs)
            if not hop_results:
                # No new documents found
                break

            all_hop_results.extend(hop_results)

        # Combine and deduplicate results
        combined_results = self._deduplicate_results(initial_results + all_hop_results)

        # Limit to top_k
        final_results = combined_results[:top_k]

        return HopRetrievalResult(
            original_results=initial_results,
            hop_results=all_hop_results,
            all_results=final_results,
            hops_performed=len(all_hop_results) > 0 or len(all_citations) > 0,
            citations_found=all_citations,
            visited_docs=visited_docs,
        )

    def _extract_citations(self, results: List[SearchResult]) -> List[Citation]:
        """
        Extract citations from search results.

        Detects section references (제X조) and regulation references (REQ-MH-004).

        Args:
            results: Search results to extract citations from

        Returns:
            List of extracted citations (deduplicated)
        """
        citations = []

        for result in results:
            doc_text = result.chunk.content
            doc_id = result.chunk.id

            # Track positions to avoid duplicates from overlapping patterns
            seen_positions: set[tuple[str, int, int]] = set()

            # Extract section references
            for pattern in self.SECTION_PATTERNS:
                for match in pattern.finditer(doc_text):
                    pos_key = (doc_id, match.start(), match.end())
                    if pos_key in seen_positions:
                        continue
                    seen_positions.add(pos_key)

                    citation = Citation(
                        source_doc_id=doc_id,
                        target_section=match.group(1) if match.lastindex >= 1 else None,
                        target_text=match.group(),
                        citation_type="section_ref",
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )
                    citations.append(citation)

            # Extract regulation references
            for pattern in self.REGULATION_REF_PATTERNS:
                for match in pattern.finditer(doc_text):
                    pos_key = (doc_id, match.start(), match.end())
                    if pos_key in seen_positions:
                        continue
                    seen_positions.add(pos_key)

                    citation = Citation(
                        source_doc_id=doc_id,
                        target_section=None,
                        target_text=match.group(),
                        citation_type="regulation_ref",
                        start_pos=match.start(),
                        end_pos=match.end(),
                    )
                    citations.append(citation)

        return citations

    def _follow_citations(
        self, query: str, citations: List[Citation], visited_docs: Set[str]
    ) -> List[SearchResult]:
        """
        Follow citations to retrieve related documents.

        Filters by relevance threshold and prevents duplicates (REQ-MH-008, REQ-MH-010).

        Args:
            query: Original query for relevance checking
            citations: Citations to follow
            visited_docs: Set of already visited document IDs

        Returns:
            List of new search results from cited documents
        """
        new_results = []

        for citation in citations:
            # Skip if already visited (REQ-MH-010)
            if citation.source_doc_id in visited_docs:
                continue

            # Check relevance before following (REQ-MH-008)
            if not self._is_citation_relevant(query, citation):
                continue

            # Search for cited document
            results = self._search_cited_document(citation)
            if results:
                # Mark as visited
                for result in results:
                    if result.chunk.id not in visited_docs:
                        visited_docs.add(result.chunk.id)
                        new_results.append(result)

        return new_results

    def _search_cited_document(
        self, citation: Citation
    ) -> Optional[List[SearchResult]]:
        """
        Search for the document referenced by citation.

        Args:
            citation: Citation to search for

        Returns:
            Search results for cited document, or None if not found
        """
        if citation.citation_type == "section_ref":
            # Search for section by number
            section_num = citation.target_section
            if section_num:
                search_query = f"제{section_num}조"
                results = self._vector_store.search(search_query, top_k=1)
                return results if results else None

        elif citation.citation_type == "regulation_ref":
            # Search by reference text
            search_query = citation.target_text
            results = self._vector_store.search(search_query, top_k=1)
            return results if results else None

        return None

    def _is_citation_relevant(self, query: str, citation: Citation) -> bool:
        """
        Check if citation is relevant to the query.

        Simple relevance check based on citation type and text overlap (REQ-MH-008).

        Args:
            query: Original search query
            citation: Citation to check

        Returns:
            True if citation is relevant enough to follow
        """
        # Section references are generally relevant
        if citation.citation_type == "section_ref":
            return True

        # For regulation references, check keyword overlap
        if citation.citation_type == "regulation_ref":
            query_terms = set(query.split())
            citation_terms = set(citation.target_text.split())
            overlap = len(query_terms & citation_terms)
            return overlap >= 1

        return False

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Deduplicate search results by document ID.

        Args:
            results: Results to deduplicate

        Returns:
            Deduplicated results (first occurrence kept)
        """
        seen_ids = set()
        deduplicated = []

        for result in results:
            doc_id = result.chunk.id
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                deduplicated.append(result)

        return deduplicated
