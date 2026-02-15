"""
Korean Morphological Expansion for Query Enhancement.

Implements SPEC-RAG-QUALITY-003 Phase 2: Korean Morphological Expansion.

This module provides morphological analysis and expansion for Korean queries
to improve retrieval recall by including conjugation variants.

Uses KiwiPiePy for high-quality Korean morpheme analysis without Java dependency.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Lazy-loaded Kiwi tokenizer (singleton with thread-safe initialization)
_kiwi: Optional[Any] = None
_kiwi_lock = threading.Lock()

# Performance metrics
_kiwi_metrics: Dict[str, Any] = {
    "init_time_ms": 0,
    "call_count": 0,
    "errors": 0,
}


def _get_kiwi() -> Optional[Any]:
    """
    Lazy-load Kiwi tokenizer (singleton with thread-safe initialization).

    Returns:
        Kiwi tokenizer instance or None if unavailable
    """
    global _kiwi

    if _kiwi is not None:
        _kiwi_metrics["call_count"] += 1
        return _kiwi

    with _kiwi_lock:
        if _kiwi is not None:
            _kiwi_metrics["call_count"] += 1
            return _kiwi

        start_time = time.perf_counter()
        try:
            from kiwipiepy import Kiwi

            _kiwi = Kiwi()
            init_time_ms = (time.perf_counter() - start_time) * 1000

            _kiwi_metrics.update(
                {
                    "init_time_ms": init_time_ms,
                    "initialized": True,
                    "version": getattr(Kiwi, "__version__", "unknown"),
                }
            )

            logger.info(
                f"KiwiPiePy initialized for morphological expansion in {init_time_ms:.2f}ms"
            )

        except ImportError as e:
            _kiwi_metrics["errors"] += 1
            logger.warning(
                f"KiwiPiePy not installed. Install with: pip install kiwipiepy>=0.20.0. Error: {e}"
            )
            return None

        except Exception as e:
            _kiwi_metrics["errors"] += 1
            logger.error(f"Failed to initialize Kiwi: {type(e).__name__}. Error: {e}")
            return None

    _kiwi_metrics["call_count"] += 1
    return _kiwi


class ExpansionMode(Enum):
    """Mode for morphological expansion."""

    NOUN_ONLY = "noun_only"  # Extract nouns only (faster, less noise)
    FULL = "full"  # Full morpheme analysis (more terms, may add noise)
    HYBRID = "hybrid"  # Nouns + key verbs/adjectives


@dataclass
class MorphologicalExpansionResult:
    """Result of morphological expansion.

    Attributes:
        original_query: The input query before expansion
        expanded_terms: List of expanded morphological variants
        final_expanded: The final expanded query string
        nouns: Extracted nouns from the query
        verbs: Extracted verbs from the query
        confidence: Confidence score (0.0 to 1.0)
        mode: Expansion mode used
        processing_time_ms: Time taken for expansion
    """

    original_query: str
    expanded_terms: List[str] = field(default_factory=list)
    final_expanded: str = ""
    nouns: List[str] = field(default_factory=list)
    verbs: List[str] = field(default_factory=list)
    confidence: float = 0.8
    mode: str = "hybrid"
    processing_time_ms: float = 0.0

    @property
    def has_expansions(self) -> bool:
        """Whether any expansions were generated."""
        return len(self.expanded_terms) > 0

    @property
    def total_terms(self) -> int:
        """Total number of expansion terms."""
        return len(self.expanded_terms)


class MorphologicalExpander:
    """
    Expands Korean queries with morphological variants.

    Part of SPEC-RAG-QUALITY-003 Phase 2: Korean Morphological Expansion.

    This class provides:
    - Noun extraction for key terms
    - Verb/adjective stem extraction
    - Conjugation variant generation
    - Caching for performance optimization

    Example:
        expander = MorphologicalExpander()
        result = expander.expand("휴학 신청 방법")
        # result.nouns = ["휴학", "신청", "방법"]
        # result.expanded_terms = ["휴학하다", "신청하다"]
    """

    # Kiwi POS tags for meaningful morphemes
    NOUN_TAGS = {
        "NNG",  # General noun (일반명사)
        "NNP",  # Proper noun (고유명사)
        "NNB",  # Dependent noun (의존명사)
        "NNM",  # Unit noun (단위명사)
        "NR",  # Numeral (수사)
        "NP",  # Pronoun (대명사)
    }

    VERB_TAGS = {
        "VV",  # Verb (동사)
        "VA",  # Adjective (형용사)
        "VX",  # Auxiliary verb (보조용언)
        "VCP",  # Positive copula (긍정 지정사)
        "VCN",  # Negative copula (부정 지정사)
    }

    # Common conjugation endings to generate variants
    CONJUGATION_ENDINGS = [
        ("하다", ["한", "할", "하여", "해서", "함", "하는"]),  # Standard verb
        ("되다", ["된", "될", "되어", "돼", "됨", "되는"]),  # Passive verb
        ("이다", ["인", "일", "이어", "임", "이는"]),  # Copula
    ]

    # Performance constraint: Maximum processing time (ms)
    MAX_PROCESSING_TIME_MS = 30

    # Maximum expansion terms to prevent noise
    MAX_EXPANSION_TERMS = 5

    def __init__(
        self,
        mode: ExpansionMode = ExpansionMode.HYBRID,
        enable_cache: bool = True,
        max_cache_size: int = 1000,
    ):
        """
        Initialize the morphological expander.

        Args:
            mode: Expansion mode (NOUN_ONLY, FULL, or HYBRID)
            enable_cache: Whether to enable expansion caching
            max_cache_size: Maximum cache size
        """
        self._mode = mode
        self._enable_cache = enable_cache
        self._max_cache_size = max_cache_size
        self._cache: Dict[str, MorphologicalExpansionResult] = {}

    @property
    def mode(self) -> ExpansionMode:
        """Get the expansion mode."""
        return self._mode

    def expand(self, query: str) -> MorphologicalExpansionResult:
        """
        Expand a query with morphological variants.

        Pipeline:
        1. Check cache for existing result
        2. Analyze query with KiwiPiePy
        3. Extract nouns and verbs based on mode
        4. Generate conjugation variants
        5. Combine and deduplicate terms

        Args:
            query: The search query to expand

        Returns:
            MorphologicalExpansionResult with expanded terms

        Example:
            >>> expander = MorphologicalExpander()
            >>> result = expander.expand("휴학 신청 방법")
            >>> print(result.nouns)
            ['휴학', '신청', '방법']
        """
        start_time = time.perf_counter()

        if not query or not query.strip():
            return MorphologicalExpansionResult(
                original_query=query,
                expanded_terms=[],
                final_expanded=query or "",
                mode=self._mode.value,
            )

        query = query.strip()

        # Check cache
        if self._enable_cache and query in self._cache:
            return self._cache[query]

        # Get Kiwi tokenizer
        kiwi = _get_kiwi()
        if kiwi is None:
            # Fallback: simple tokenization
            return self._fallback_expand(query)

        try:
            # Analyze morphemes
            tokens = kiwi.tokenize(query)

            # Extract morphemes based on mode
            nouns: List[str] = []
            verbs: List[str] = []
            all_terms: Set[str] = set()

            for token in tokens:
                form = token.form
                pos = token.tag

                # Skip short terms
                if len(form) < 2:
                    continue

                # Extract nouns
                if pos in self.NOUN_TAGS:
                    nouns.append(form)
                    all_terms.add(form)

                # Extract verbs/adjectives based on mode
                if pos in self.VERB_TAGS:
                    if self._mode in (ExpansionMode.FULL, ExpansionMode.HYBRID):
                        verbs.append(form)
                        all_terms.add(form)

            # Generate conjugation variants
            expanded_terms: List[str] = []
            if self._mode in (ExpansionMode.FULL, ExpansionMode.HYBRID):
                for verb in verbs:
                    variants = self._generate_conjugation_variants(verb)
                    for variant in variants:
                        if variant not in all_terms and variant not in query:
                            expanded_terms.append(variant)
                            all_terms.add(variant)

            # Limit expansion terms
            expanded_terms = expanded_terms[: self.MAX_EXPANSION_TERMS]

            # Build final expanded query
            if expanded_terms:
                final_expanded = query + " " + " ".join(expanded_terms)
            else:
                final_expanded = query

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            result = MorphologicalExpansionResult(
                original_query=query,
                expanded_terms=expanded_terms,
                final_expanded=final_expanded,
                nouns=nouns,
                verbs=verbs,
                confidence=self._calculate_confidence(len(expanded_terms)),
                mode=self._mode.value,
                processing_time_ms=processing_time_ms,
            )

            # Cache result
            if self._enable_cache:
                self._manage_cache()
                self._cache[query] = result

            return result

        except Exception as e:
            logger.warning(f"Morphological analysis failed: {e}")
            return self._fallback_expand(query)

    def extract_nouns(self, query: str) -> List[str]:
        """
        Extract nouns from a query.

        Args:
            query: The query to analyze

        Returns:
            List of extracted nouns
        """
        result = self.expand(query)
        return result.nouns

    def extract_verbs(self, query: str) -> List[str]:
        """
        Extract verbs/adjectives from a query.

        Args:
            query: The query to analyze

        Returns:
            List of extracted verbs
        """
        result = self.expand(query)
        return result.verbs

    def _generate_conjugation_variants(self, verb: str) -> List[str]:
        """
        Generate conjugation variants for a verb.

        Args:
            verb: The verb stem

        Returns:
            List of conjugation variants
        """
        variants: List[str] = []

        for base_ending, conjugations in self.CONJUGATION_ENDINGS:
            # Check if verb ends with the base ending
            if verb.endswith(base_ending) or verb + "다" == base_ending:
                # Generate variants
                stem = verb.rstrip(base_ending) if verb.endswith(base_ending) else verb
                for conj in conjugations:
                    variant = stem + conj
                    if variant != verb:
                        variants.append(variant)

        # Also try adding common endings if verb doesn't match patterns
        if not variants:
            # Try common patterns: ~하다, ~되다
            for suffix in ["하", "되", "이"]:
                if verb.endswith(suffix):
                    for conj in ["다", "은", "는", "을", "음"]:
                        variants.append(verb + conj)

        return variants

    def _calculate_confidence(self, num_expansions: int) -> float:
        """
        Calculate confidence based on number of expansions.

        Args:
            num_expansions: Number of expansion terms generated

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if num_expansions == 0:
            return 0.5
        elif num_expansions <= 2:
            return 0.95
        elif num_expansions <= 4:
            return 0.85
        else:
            return 0.75  # More terms may introduce noise

    def _fallback_expand(self, query: str) -> MorphologicalExpansionResult:
        """
        Fallback expansion when KiwiPiePy is unavailable.

        Uses simple regex-based tokenization.

        Args:
            query: The query to expand

        Returns:
            MorphologicalExpansionResult with basic expansion
        """
        import re

        # Simple Korean tokenization
        tokens = re.findall(r"[가-힣]+", query)
        nouns = [t for t in tokens if len(t) >= 2]

        result = MorphologicalExpansionResult(
            original_query=query,
            expanded_terms=[],
            final_expanded=query,
            nouns=nouns,
            verbs=[],
            confidence=0.5,
            mode="fallback",
        )

        # Cache result even in fallback mode
        if self._enable_cache:
            self._manage_cache()
            self._cache[query] = result

        return result

    def _manage_cache(self) -> None:
        """Manage cache size by removing old entries."""
        if len(self._cache) >= self._max_cache_size:
            # Remove half of the cache (simple LRU approximation)
            keys_to_remove = list(self._cache.keys())[: self._max_cache_size // 2]
            for key in keys_to_remove:
                del self._cache[key]
            logger.debug(f"Cache cleanup: removed {len(keys_to_remove)} entries")

    def clear_cache(self) -> None:
        """Clear the expansion cache."""
        self._cache.clear()
        logger.debug("Morphological expansion cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get expander statistics.

        Returns:
            Dictionary with expander statistics
        """
        return {
            "mode": self._mode.value,
            "cache_enabled": self._enable_cache,
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size,
            "kiwi_metrics": _kiwi_metrics.copy(),
        }


def create_morphological_expander(
    mode: str = "hybrid",
    enable_cache: bool = True,
) -> MorphologicalExpander:
    """
    Factory function to create a MorphologicalExpander.

    Args:
        mode: Expansion mode ("noun_only", "full", or "hybrid")
        enable_cache: Whether to enable caching

    Returns:
        Configured MorphologicalExpander instance
    """
    mode_map = {
        "noun_only": ExpansionMode.NOUN_ONLY,
        "full": ExpansionMode.FULL,
        "hybrid": ExpansionMode.HYBRID,
    }

    expansion_mode = mode_map.get(mode, ExpansionMode.HYBRID)
    return MorphologicalExpander(mode=expansion_mode, enable_cache=enable_cache)
