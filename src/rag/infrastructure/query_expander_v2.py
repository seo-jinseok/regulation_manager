"""
Multi-Stage Query Expansion for Regulation RAG System.

Implements SPEC-RAG-SEARCH-001 TAG-002: Multi-Stage Query Expansion.

This module provides 3-stage query expansion pipeline:
1. Stage 1: Synonym expansion (장학금 ↔ 장학금 지원 ↔ 재정 지원)
2. Stage 2: Hypernym expansion (등록금 → 학사 → 행정)
3. Stage 3: Procedure expansion (신청 → 절차 → 서류 → 제출)

The expander integrates with RegulationEntityRecognizer (TAG-001) for
enhanced entity recognition and expansion term generation.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from ..domain.entity import RegulationEntityRecognizer

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    """
    Result of multi-stage query expansion.

    Attributes:
        original_query: The input query
        stage1_synonyms: Synonym expansion results (Stage 1)
        stage2_hypernyms: Hypernym expansion results (Stage 2)
        stage3_procedures: Procedure expansion results (Stage 3)
        final_expanded: The final expanded query string
        confidence: Overall confidence in expansion quality (0.0 to 1.0)
        method: How expansion was achieved ("multi_stage", "formal_skip", etc.)
    """

    original_query: str
    stage1_synonyms: List[str]
    stage2_hypernyms: List[str]
    stage3_procedures: List[str]
    final_expanded: str
    confidence: float
    method: str

    @property
    def total_terms(self) -> int:
        """Total number of expansion terms generated."""
        return (
            len(self.stage1_synonyms)
            + len(self.stage2_hypernyms)
            + len(self.stage3_procedures)
        )

    @property
    def has_expansions(self) -> bool:
        """Whether any expansions were generated."""
        return self.total_terms > 0


class MultiStageQueryExpander:
    """
    Multi-stage query expander with 3-stage pipeline.

    Part of SPEC-RAG-SEARCH-001 TAG-002: Multi-Stage Query Expansion.

    This class provides structured query expansion through:
    - Stage 1: Synonym expansion (장학금 ↔ 장학금 지원 ↔ 재정 지원)
    - Stage 2: Hypernym expansion (등록금 → 학사 → 행정)
    - Stage 3: Procedure expansion (신청 → 절차 → 서류 → 제출)

    The expander integrates with RegulationEntityRecognizer for enhanced
    entity recognition and uses entity-specific expansion mappings.

    Example:
        expander = MultiStageQueryExpander(entity_recognizer)
        result = expander.expand("장학금 신청 방법")

        # result.stage1_synonyms = ["장학금 지급", "장학"]
        # result.stage2_hypernyms = ["재정", "지원"]
        # result.stage3_procedures = ["신청서", "제출", "절차"]
        # result.final_expanded = "장학금 신청 방법 장학금 지급 장학 재정 지원 신청서 제출 절차"
    """

    # ============ SYNONYM MAPPINGS (Stage 1) ============
    # Core synonym mappings for Stage 1 expansion
    SYNONYM_MAPPINGS: Dict[str, List[str]] = {
        "장학금": ["장학금", "장학금 지급", "장학금 지원", "재정 지원"],
        "연구년": ["연구년", "안식년", "교원연구년", "교원휴직"],
        "휴학": ["휴학", "휴학원", "학적휴식", "휴학신청"],
        "복학": ["복학", "복학원", "복학신청", "재입학"],
        "제적": ["제적", "학적상실", "자퇴원", "자퇴"],
        "자퇴": ["자퇴", "자퇴원", "자퇴신청", "퇴학"],
        "전과": ["전과", "전공", "전과신청", "학사전공"],
        "편입": ["편입", "편입학", "학사편입", "3학년편입"],
        "조교": ["조교", "교육조교", "연구조교", "TA"],
        "교수": ["교수", "교원", "교직원", "강사"],
        # Staff vocabulary additions (SPEC-RAG-QUALITY-005 Phase 2)
        "복무": ["복무", "근무", "재직", "출근"],
        "연차": ["연차", "휴가", "휴직", "휴무"],
        "급여": ["급여", "봉급", "월급", "보수"],
        "연수": ["연수", "교육", "훈련", "연교육"],
        "사무용품": ["사무용품", "비품", "물품", "용품"],
        "입찰": ["입찰", "계약", "발주", "조달"],
    }

    # Maximum expansion limits per stage (REQ-QE-005, REQ-QE-006)
    MAX_SYNONYMS_PER_STAGE = 3
    MAX_HYPERNYMS_PER_STAGE = 2
    MAX_PROCEDURES_PER_STAGE = 2
    MAX_TOTAL_EXPANSIONS = 10  # REQ-QE-008

    # Confidence thresholds (REQ-QE-009)
    MIN_EXPANSION_CONFIDENCE = 0.6

    # Formal query indicators (REQ-QE-010)
    FORMAL_QUERY_INDICATORS = [
        "규정",
        "학칙",
        "세칙",
        "지침",
        "조",
        "항",
        "호",
        "세칙",
        "내규",
        "지침",
    ]

    def __init__(self, entity_recognizer: RegulationEntityRecognizer):
        """
        Initialize multi-stage query expander.

        Args:
            entity_recognizer: Entity recognizer for expansion term generation
        """
        self._entity_recognizer = entity_recognizer

    def expand(self, query: str) -> ExpansionResult:
        """
        Apply 3-stage expansion to query.

        Pipeline stages:
        1. Check if formal query (skip if yes)
        2. Stage 1: Synonym expansion
        3. Stage 2: Hypernym expansion
        4. Stage 3: Procedure expansion
        5. Combine and limit to MAX_TOTAL_EXPANSIONS

        Args:
            query: The search query to expand

        Returns:
            ExpansionResult with stage-by-stage results and final expanded query

        Example:
            expander = MultiStageQueryExpander(entity_recognizer)
            result = expander.expand("장학금 신청 방법")

            # result.stage1_synonyms contains ["장학금 지급", "장학"]
            # result.stage2_hypernyms contains ["재정", "지원"]
            # result.stage3_procedures contains ["신청서", "제출", "절차"]
        """
        if not query or not query.strip():
            return ExpansionResult(
                original_query=query,
                stage1_synonyms=[],
                stage2_hypernyms=[],
                stage3_procedures=[],
                final_expanded=query or "",
                confidence=1.0,
                method="empty",
            )

        # Check if formal regulation query (REQ-QE-010)
        if self._is_formal_query(query):
            logger.debug(
                f"Formal query detected, skipping expansion: '{query[:30]}...'"
            )
            return ExpansionResult(
                original_query=query,
                stage1_synonyms=[],
                stage2_hypernyms=[],
                stage3_procedures=[],
                final_expanded=query,
                confidence=1.0,
                method="formal_skip",
            )

        # Stage 1: Synonym expansion
        stage1_synonyms = self._expand_synonyms(query)

        # Stage 2: Hypernym expansion
        stage2_hypernyms = self._expand_hypernyms(query)

        # Stage 3: Procedure expansion
        stage3_procedures = self._expand_procedures(query)

        # Combine all expansions (deduplicate, preserve order)
        all_terms = self._combine_expansions(
            query, stage1_synonyms, stage2_hypernyms, stage3_procedures
        )

        # Build final expanded query
        if all_terms:
            final_expanded = " ".join(all_terms)
        else:
            final_expanded = query

        # Calculate confidence
        confidence = self._calculate_confidence(all_terms)

        return ExpansionResult(
            original_query=query,
            stage1_synonyms=stage1_synonyms,
            stage2_hypernyms=stage2_hypernyms,
            stage3_procedures=stage3_procedures,
            final_expanded=final_expanded,
            confidence=confidence,
            method="multi_stage",
        )

    def _is_formal_query(self, query: str) -> bool:
        """
        Check if query is already formal regulation language (REQ-QE-010).

        Formal queries contain regulation-specific terms and don't need expansion.

        Args:
            query: The search query

        Returns:
            True if query is formal regulation language, False otherwise
        """
        query_lower = query.lower()

        # Check for formal indicators
        for indicator in self.FORMAL_QUERY_INDICATORS:
            if indicator in query_lower:
                return True

        return False

    def _expand_synonyms(self, query: str) -> List[str]:
        """
        Stage 1: Synonym expansion (REQ-QE-004, REQ-QE-005).

        Expands query with up to 3 related synonyms per matched term.

        Args:
            query: The search query

        Returns:
            List of synonym expansion terms
        """
        synonyms = []

        for term in self.SYNONYM_MAPPINGS:
            if term in query:
                expansion = self.SYNONYM_MAPPINGS[term][
                    1 : self.MAX_SYNONYMS_PER_STAGE + 1
                ]  # Skip original term, keep expansions
                synonyms.extend(expansion)

        return synonyms[: self.MAX_SYNONYMS_PER_STAGE]

    def _expand_hypernyms(self, query: str) -> List[str]:
        """
        Stage 2: Hypernym expansion (REQ-QE-006).

        Expands query with hierarchical terms using entity recognizer.

        Args:
            query: The search query

        Returns:
            List of hypernym expansion terms
        """
        hypernyms = []

        # Use entity recognizer for hypernym detection
        result = self._entity_recognizer.recognize(query)

        # Extract hypernym entity matches
        for match in result.matches:
            if match.entity_type.value == "hypernym":
                # Skip original term, keep expanded terms
                expanded = [t for t in match.expanded_terms if t != match.text]
                hypernyms.extend(expanded)

        return hypernyms[: self.MAX_HYPERNYMS_PER_STAGE]

    def _expand_procedures(self, query: str) -> List[str]:
        """
        Stage 3: Procedure expansion (REQ-QE-007).

        Expands query with procedure chain terms using entity recognizer.

        Args:
            query: The search query

        Returns:
            List of procedure expansion terms
        """
        procedures = []

        # Use entity recognizer for procedure detection
        result = self._entity_recognizer.recognize(query)

        # Extract procedure entity matches
        for match in result.matches:
            if match.entity_type.value == "procedure":
                # Skip original term, keep expanded terms
                expanded = [t for t in match.expanded_terms if t != match.text]
                procedures.extend(expanded)

        return procedures[: self.MAX_PROCEDURES_PER_STAGE]

    def _combine_expansions(
        self,
        query: str,
        stage1: List[str],
        stage2: List[str],
        stage3: List[str],
    ) -> List[str]:
        """
        Combine expansions from all stages with deduplication.

        Args:
            query: Original query
            stage1: Stage 1 results
            stage2: Stage 2 results
            stage3: Stage 3 results

        Returns:
            Combined list of unique terms, preserving insertion order

        Note:
            Deduplicates at word level to handle multi-word phrases from
            synonym mappings. Each phrase is split into words and each
            word is checked individually to prevent duplicates in the
            final expanded query.
        """
        # Start with original query terms
        query_terms = set(query.split())
        combined = []
        seen = set(query_terms)  # Track seen words (including query words)

        # Add expansions in stage order (synonyms → hypernyms → procedures)
        for stage_expansions in [stage1, stage2, stage3]:
            for term in stage_expansions:
                # Split multi-word phrases into individual words
                words = term.split()
                # Filter out words we've already seen
                new_words = [w for w in words if w not in seen]
                # Add only new words to combined list
                for word in new_words:
                    if word not in seen:
                        seen.add(word)
                        combined.append(word)

        # Limit to MAX_TOTAL_EXPANSIONS (REQ-QE-008)
        return combined[: self.MAX_TOTAL_EXPANSIONS]

    def _calculate_confidence(self, expanded_terms: List[str]) -> float:
        """
        Calculate expansion confidence based on relevance (REQ-QE-012).

        More relevant terms = higher confidence.

        Args:
            expanded_terms: All expanded terms

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not expanded_terms:
            return 0.5

        # Higher confidence with more (but not too many) terms
        term_count = len(expanded_terms)

        if term_count == 0:
            return 0.5
        elif term_count <= 3:
            return 0.8
        elif term_count <= 7:
            return 0.7
        else:
            # Too many terms might introduce noise
            return 0.6

    def get_expanded_query(self, query: str, use_expansion: bool = True) -> str:
        """
        Convenience method to get expanded query.

        Args:
            query: Original query
            use_expansion: Whether to apply expansion (default: True)

        Returns:
            Expanded query string, or original if use_expansion=False
        """
        if not use_expansion:
            return query

        result = self.expand(query)
        return result.final_expanded


@dataclass
class QueryExpansionPipeline:
    """
    Pipeline that combines MultiStageQueryExpander with existing components.

    Integration point for TAG-005: Integration & Testing.

    This class will integrate the multi-stage expander with:
    - QueryAnalyzer (for query type classification)
    - HybridSearcher (for search execution)
    - DynamicQueryExpander (for LLM fallback)

    Usage:
        entity_recognizer = RegulationEntityRecognizer()
        multi_stage_expander = MultiStageQueryExpander(entity_recognizer)
        pipeline = QueryExpansionPipeline(multi_stage_expander)

        result = pipeline.process_query("장학금 신청 방법")
        # Returns: ExpansionResult with multi-stage results
    """

    def __init__(
        self,
        expander: MultiStageQueryExpander,
        enable_cache: bool = True,
    ):
        """
        Initialize expansion pipeline.

        Args:
            expander: Multi-stage query expander instance
            enable_cache: Whether to enable expansion result caching
        """
        self._expander = expander
        self._enable_cache = enable_cache
        self._cache: Dict[str, ExpansionResult] = {}

    def process_query(self, query: str) -> ExpansionResult:
        """
        Process query through expansion pipeline.

        This is the main entry point for query expansion.

        Args:
            query: Original user query

        Returns:
            ExpansionResult with all expansion details

        Example:
            pipeline = QueryExpansionPipeline(expander)
            result = pipeline.process_query("장학금 신청")

            # Returns ExpansionResult with:
            # - stage1_synonyms = ["장학금 지급", "장학"]
            # - stage2_hypernyms = ["재정", "지원"]
            # - stage3_procedures = ["신청서", "제출", "절차"]
            # - final_expanded = combined query
            # - confidence = 0.8
        """
        if not query or not query.strip():
            return ExpansionResult(
                original_query=query,
                stage1_synonyms=[],
                stage2_hypernyms=[],
                stage3_procedures=[],
                final_expanded=query or "",
                confidence=0.5,
                method="empty",
            )

        # Check cache if enabled
        if self._enable_cache and query in self._cache:
            logger.debug(f"Cache hit for query: '{query[:30]}...'")
            return self._cache[query]

        # Process through multi-stage expander
        result = self._expander.expand(query)

        # Cache result if enabled
        if self._enable_cache:
            self._cache[query] = result

        return result

    def clear_cache(self) -> None:
        """Clear expansion cache."""
        self._cache.clear()
        logger.debug("Expansion cache cleared")
