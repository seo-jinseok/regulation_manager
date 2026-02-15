"""
Query Expansion Service for RAG System.

Implements synonym-based query expansion to improve retrieval coverage.
Supports English-Korean mixed queries for international students.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..domain.entities import SearchResult
from ..domain.repositories import ILLMClient, IVectorStore
from ..domain.value_objects import Query
from .synonym_generator_service import SynonymGeneratorService

logger = logging.getLogger(__name__)


@dataclass
class ExpandedQuery:
    """
    An expanded query variant.

    Attributes:
        original_query: The original user query
        expanded_text: The expanded query text
        expansion_method: How the expansion was generated (synonym, translation, etc.)
        confidence: Confidence score for the expansion (0.0-1.0)
        language: Language of the expanded query (ko, en, mixed)
    """

    original_query: str
    expanded_text: str
    expansion_method: str
    confidence: float = 1.0
    language: str = "ko"

    def to_query(self) -> Query:
        """Convert to Query value object."""
        return Query(text=self.expanded_text)


class QueryExpansionService:
    """
    Query expansion service using synonyms and translation.

    Features:
    - Synonym-based expansion for academic terms
    - English-Korean mixed query support
    - Parallel query execution and result merging
    - Reciprocal Rank Fusion (RRF) for result aggregation

    Usage:
        ```python
        expansion_service = QueryExpansionService(
            store=vector_store,
            synonym_service=synonym_service
        )

        # Expand query
        expanded_queries = expansion_service.expand_query("휴학 방법 알려줘")

        # Search with expanded queries
        results = expansion_service.search_with_expansion(
            query="휴학 방법",
            top_k=5
        )
        ```
    """

    # Academic term mappings for common university regulation concepts
    # Note: These synonyms support bidirectional lookup via _expand_with_synonyms()
    ACADEMIC_SYNONYMS = {
        "휴학": ["휴학원", "학업 중단", "학교 쉬다", "일반휴학", "군휴학"],
        "복학": ["복학원", "복학 신청", "학업 재개"],
        "등록금": ["학비", "수업료", "등록금 납부", " tuition"],
        "장학금": ["장학", "奖学金", "scholarship", "재정 지원"],
        "성적": ["학점", "grades", "성적표", "학업 성취"],
        "수강신청": ["수강 신청", "course registration", "강의 신청"],
        "졸업": ["졸업요건", "graduation", "학위 취득"],
        "제적": ["자퇴", "expulsion", "학적 제적"],
        "전공": ["major", "전공 과정", "전공 변경"],
        "교양": ["교양 과목", "general education", "liberal arts"],
        # Bidirectional synonyms for Korean academic terms (TAG-004)
        "복무": ["근무", "복무규정", "출근", "근태"],
        "근무": ["복무", "출근", "근태", "근무시간"],
        "교원": ["교수", "교직원", "교육공무원"],
        "교수": ["교원", "교직원"],
        "승진": ["진급", "승급", "인사상승"],
        "진급": ["승진", "승급"],
        # Note: "규정" and "조례" are in STOPWORDS (cleaned before expansion).
        # Not added as keys to avoid false positives in has_synonyms().
    }

    # English-Korean mappings for international students
    ENGLISH_KOREAN_MAPPINGS = {
        "leave of absence": "휴학",
        "tuition": "등록금",
        "scholarship": "장학금",
        "grades": "성적",
        "course registration": "수강신청",
        "graduation": "졸업",
        "major": "전공",
        "withdrawal": "자퇴",
        "readmission": "복학",
        "academic probation": "학사 경고",
    }

    def __init__(
        self,
        store: IVectorStore,
        synonym_service: Optional[SynonymGeneratorService] = None,
        llm_client: Optional[ILLMClient] = None,
    ):
        """
        Initialize query expansion service.

        Args:
            store: Vector store for searching
            synonym_service: Optional synonym generator service
            llm_client: Optional LLM client for dynamic expansion
        """
        self.store = store
        self.synonym_service = synonym_service
        self.llm_client = llm_client

    def expand_query(
        self,
        query: str,
        max_variants: int = 3,
        method: str = "synonym"
    ) -> List[ExpandedQuery]:
        """
        Expand a query using various methods.

        Args:
            query: Original query text
            max_variants: Maximum number of query variants to generate
            method: Expansion method ("synonym", "translation", "mixed", "llm")

        Returns:
            List of expanded query variants

        Examples:
            >>> service.expand_query("휴학 방법")
            [
                ExpandedQuery("휴학 방법", "휴학 방법", "original", 1.0),
                ExpandedQuery("휴학 방법", "휴학원 방법", "synonym", 0.9),
                ExpandedQuery("휴학 방법", "학업 중단 방법", "synonym", 0.8)
            ]
        """
        expanded = []

        # Always include original query
        expanded.append(
            ExpandedQuery(
                original_query=query,
                expanded_text=query,
                expansion_method="original",
                confidence=1.0,
                language=self._detect_language(query)
            )
        )

        if method == "synonym":
            expanded.extend(self._expand_with_synonyms(query, max_variants - 1))
        elif method == "translation":
            expanded.extend(self._expand_with_translation(query, max_variants - 1))
        elif method == "mixed":
            # Combine synonym and translation expansion
            expanded.extend(self._expand_with_synonyms(query, max_variants // 2))
            expanded.extend(self._expand_with_translation(query, max_variants // 2))
        elif method == "llm" and self.llm_client:
            expanded.extend(self._expand_with_llm(query, max_variants - 1))

        # Remove duplicates and limit
        unique = self._deduplicate_expanded(expanded)
        return unique[:max_variants]

    def search_with_expansion(
        self,
        query: str,
        top_k: int = 5,
        expansion_method: str = "synonym",
        rrf_k: int = 60
    ) -> List[SearchResult]:
        """
        Search using query expansion with Reciprocal Rank Fusion.

        Args:
            query: Original query
            top_k: Number of results to return
            expansion_method: Method for query expansion
            rrf_k: RRF constant (default 60)

        Returns:
            Merged and ranked search results

        Examples:
            >>> results = service.search_with_expansion("휴학 방법", top_k=5)
            >>> len(results)
            5
        """
        # Generate expanded queries
        expanded_queries = self.expand_query(query, max_variants=3, method=expansion_method)

        # Execute parallel searches
        all_results = {}
        for expanded_query in expanded_queries:
            query_obj = expanded_query.to_query()
            results = self.store.search(query_obj, top_k=top_k * 2)

            # Collect results with RRF scoring
            for rank, result in enumerate(results, 1):
                chunk_id = result.chunk.id
                if chunk_id not in all_results:
                    all_results[chunk_id] = {
                        "result": result,
                        "scores": []
                    }
                # Calculate RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (rrf_k + rank)
                all_results[chunk_id]["scores"].append(rrf_score)

        # Aggregate RRF scores
        aggregated = []
        for _chunk_id, data in all_results.items():
            # Sum all RRF scores for this chunk
            total_score = sum(data["scores"])
            result = data["result"]

            # Create new SearchResult with aggregated score
            aggregated.append(
                SearchResult(
                    chunk=result.chunk,
                    score=total_score,
                    query=result.query
                )
            )

        # Sort by aggregated score and return top_k
        aggregated.sort(key=lambda x: x.score, reverse=True)
        return aggregated[:top_k]

    def _expand_with_synonyms(self, query: str, max_variants: int) -> List[ExpandedQuery]:
        """Expand query using synonym database with bidirectional lookup.

        Bidirectional lookup ensures that:
        - If "복무": ["근무"] exists, "복무" queries expand to "근무"
        - And "근무" queries also expand to "복무" (reverse direction)
        """
        expanded = []
        seen_expansions = set()  # Track unique expansions

        # Find terms in query that have synonyms
        for term, synonyms in self.ACADEMIC_SYNONYMS.items():
            if term in query:
                for synonym in synonyms[:max_variants]:
                    expanded_query = query.replace(term, synonym, 1)
                    if expanded_query not in seen_expansions:
                        seen_expansions.add(expanded_query)
                        expanded.append(
                            ExpandedQuery(
                                original_query=query,
                                expanded_text=expanded_query,
                                expansion_method="synonym",
                                confidence=0.9,
                                language="ko"
                            )
                        )

        # Bidirectional lookup: Also expand when a SYNONYM appears in the query
        # E.g., "근무" should expand to "복무" if "복무": ["근무"] exists
        for key_term, synonyms in self.ACADEMIC_SYNONYMS.items():
            for synonym in synonyms:
                if synonym in query and key_term not in query:
                    expanded_query = query.replace(synonym, key_term, 1)
                    if expanded_query not in seen_expansions:
                        seen_expansions.add(expanded_query)
                        expanded.append(
                            ExpandedQuery(
                                original_query=query,
                                expanded_text=expanded_query,
                                expansion_method="synonym_bidirectional",
                                confidence=0.85,  # Slightly lower confidence for bidirectional
                                language="ko"
                            )
                        )

        return expanded

    def _expand_with_translation(self, query: str, max_variants: int) -> List[ExpandedQuery]:
        """Expand query using English-Korean translation."""
        expanded = []

        # Korean to English
        for ko_term, en_term in self.ENGLISH_KOREAN_MAPPINGS.items():
            if ko_term in query:
                expanded_query = query.replace(ko_term, en_term, 1)
                expanded.append(
                    ExpandedQuery(
                        original_query=query,
                        expanded_text=expanded_query,
                        expansion_method="translation",
                        confidence=0.85,
                        language="en"
                    )
                )

        # English to Korean
        for en_term, ko_term in self.ENGLISH_KOREAN_MAPPINGS.items():
            if en_term.lower() in query.lower():
                expanded_query = query.replace(en_term, ko_term, 1)
                expanded.append(
                    ExpandedQuery(
                        original_query=query,
                        expanded_text=expanded_query,
                        expansion_method="translation",
                        confidence=0.85,
                        language="ko"
                    )
                )

        return expanded

    def _expand_with_llm(self, query: str, max_variants: int) -> List[ExpandedQuery]:
        """Expand query using LLM for dynamic synonym generation."""
        if not self.synonym_service or not self.synonym_service.llm_client:
            logger.warning("LLM client not available for query expansion")
            return []

        # Extract key terms from query
        key_terms = self._extract_key_terms(query)
        expanded = []

        for term in key_terms[:3]:  # Limit to top 3 terms
            try:
                synonyms = self.synonym_service.generate_synonyms(
                    term=term,
                    context="대학 규정 검색",
                    exclude_existing=True
                )

                for synonym in synonyms[:max_variants]:
                    expanded_query = query.replace(term, synonym, 1)
                    expanded.append(
                        ExpandedQuery(
                            original_query=query,
                            expanded_text=expanded_query,
                            expansion_method="llm_synonym",
                            confidence=0.8,
                            language="ko"
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to generate synonyms for {term}: {e}")

        return expanded

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for expansion."""
        # Simple extraction: filter common stopwords and keep meaningful terms
        stopwords = {"방법", "알려줘", "어떻게", "하는", "것", "등", "및", "또는"}

        terms = []
        for term in self.ACADEMIC_SYNONYMS.keys():
            if term in query and term not in stopwords:
                terms.append(term)

        return terms

    def _detect_language(self, text: str) -> str:
        """Detect language of text (ko, en, mixed)."""
        korean_chars = sum(1 for c in text if ord('가') <= ord(c) <= ord('힣'))
        english_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)

        if korean_chars > 0 and english_chars > 0:
            return "mixed"
        elif korean_chars > 0:
            return "ko"
        elif english_chars > 0:
            return "en"
        else:
            return "unknown"

    def _deduplicate_expanded(self, expanded: List[ExpandedQuery]) -> List[ExpandedQuery]:
        """Remove duplicate expanded queries."""
        seen = set()
        unique = []

        for exp in expanded:
            # Normalize for comparison
            normalized = exp.expanded_text.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(exp)

        return unique

    def get_expansion_statistics(
        self,
        queries: List[str],
        method: str = "synonym"
    ) -> Dict[str, Any]:
        """
        Get statistics about query expansion effectiveness.

        Args:
            queries: List of test queries
            method: Expansion method to test

        Returns:
            Dictionary with expansion statistics
        """
        stats = {
            "total_queries": len(queries),
            "expanded_queries": 0,
            "avg_variants_per_query": 0.0,
            "language_distribution": {"ko": 0, "en": 0, "mixed": 0},
            "method_distribution": {}
        }

        total_variants = 0

        for query in queries:
            expanded = self.expand_query(query, max_variants=5, method=method)
            if len(expanded) > 1:  # More than just original
                stats["expanded_queries"] += 1
                total_variants += len(expanded)

            # Count languages
            for exp in expanded:
                lang = exp.language
                if lang in stats["language_distribution"]:
                    stats["language_distribution"][lang] += 1

            # Count methods
            for exp in expanded:
                method = exp.expansion_method
                stats["method_distribution"][method] = (
                    stats["method_distribution"].get(method, 0) + 1
                )

        if stats["total_queries"] > 0:
            stats["avg_variants_per_query"] = total_variants / stats["total_queries"]

        return stats
