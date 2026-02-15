"""
Hybrid Weight Optimizer for Dynamic Search Weight Adjustment.

Implements SPEC-RAG-QUALITY-003 Phase 5: Hybrid Weight Optimization.

This module dynamically adjusts BM25 and vector search weights based on
query formality level to improve retrieval quality for both formal and
colloquial Korean queries.

Key Features:
- Formal/informal query detection using ColloquialTransformer
- Dynamic weight calculation based on query characteristics
- Integration with existing HybridSearcher
- Comprehensive logging for analysis
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.rag.domain.query.colloquial_transformer import ColloquialTransformer

logger = logging.getLogger(__name__)


@dataclass
class WeightDecision:
    """Result of weight optimization decision.

    Attributes:
        bm25_weight: Weight for BM25 (keyword) search (0.0 to 1.0)
        vector_weight: Weight for vector (dense) search (0.0 to 1.0)
        is_colloquial: Whether the query was detected as colloquial
        formality_score: Formality score (0.0 = colloquial, 1.0 = formal)
        detected_patterns: List of colloquial patterns detected
        reasoning: Explanation for the weight decision
    """

    bm25_weight: float
    vector_weight: float
    is_colloquial: bool
    formality_score: float
    detected_patterns: List[str] = field(default_factory=list)
    reasoning: str = ""

    @property
    def total_weight(self) -> float:
        """Sum of weights (should always be 1.0)."""
        return self.bm25_weight + self.vector_weight

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
            "is_colloquial": self.is_colloquial,
            "formality_score": self.formality_score,
            "detected_patterns": self.detected_patterns,
            "reasoning": self.reasoning,
        }


class HybridWeightOptimizer:
    """
    Optimizes hybrid search weights based on query formality.

    Phase 5 (SPEC-RAG-QUALITY-003): Dynamic weight adjustment.

    This class provides:
    - Formal/informal query detection using ColloquialTransformer
    - Dynamic weight calculation based on query characteristics
    - Logging of weight decisions for analysis
    - Manual weight override capability

    Weight Strategy:
    - Formal queries: Balanced weights (0.5 BM25, 0.5 vector)
      - Formal Korean typically has exact keyword matches
    - Colloquial queries: Higher vector weight (0.3 BM25, 0.7 vector)
      - Colloquial queries benefit from semantic matching

    Example:
        optimizer = HybridWeightOptimizer()
        decision = optimizer.optimize("휴학 어떻게 해?")
        # decision.vector_weight = 0.7 (colloquial query)
        # decision.bm25_weight = 0.3
    """

    # Default weight configurations
    FORMAL_WEIGHTS = (0.5, 0.5)  # (bm25, vector) - balanced
    COLLOQUIAL_WEIGHTS = (0.3, 0.7)  # (bm25, vector) - favor semantic

    # Formality thresholds
    FORMALITY_THRESHOLD = 0.5  # Below this is colloquial

    # Performance constraint: Maximum processing time (ms)
    MAX_PROCESSING_TIME_MS = 30

    def __init__(
        self,
        colloquial_transformer: Optional["ColloquialTransformer"] = None,
        enable_logging: bool = True,
        formal_weights: Optional[Tuple[float, float]] = None,
        colloquial_weights: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize the weight optimizer.

        Args:
            colloquial_transformer: Optional ColloquialTransformer for formality detection
            enable_logging: Whether to log weight decisions
            formal_weights: Custom weights for formal queries (bm25, vector)
            colloquial_weights: Custom weights for colloquial queries (bm25, vector)
        """
        self._enable_logging = enable_logging

        # Set weight configurations
        self._formal_weights = formal_weights or self.FORMAL_WEIGHTS
        self._colloquial_weights = colloquial_weights or self.COLLOQUIAL_WEIGHTS

        # Validate weights
        self._validate_weights(self._formal_weights, "formal")
        self._validate_weights(self._colloquial_weights, "colloquial")

        # Initialize colloquial transformer
        self._transformer = colloquial_transformer
        self._transformer_available = colloquial_transformer is not None

        # Statistics tracking
        self._stats = {
            "total_queries": 0,
            "formal_queries": 0,
            "colloquial_queries": 0,
            "pattern_detections": 0,
        }

    def _validate_weights(
        self, weights: Tuple[float, float], name: str
    ) -> None:
        """
        Validate weight configuration.

        Args:
            weights: Tuple of (bm25_weight, vector_weight)
            name: Name of weight set for error messages

        Raises:
            ValueError: If weights are invalid
        """
        bm25_w, vector_w = weights

        if not (0.0 <= bm25_w <= 1.0 and 0.0 <= vector_w <= 1.0):
            raise ValueError(
                f"Invalid {name} weights: weights must be between 0.0 and 1.0"
            )

        if abs(bm25_w + vector_w - 1.0) > 0.01:
            raise ValueError(
                f"Invalid {name} weights: sum must be 1.0, got {bm25_w + vector_w}"
            )

    def _get_transformer(self) -> Optional["ColloquialTransformer"]:
        """
        Get or create colloquial transformer.

        Returns:
            ColloquialTransformer instance or None if unavailable
        """
        if self._transformer is not None:
            return self._transformer

        if not self._transformer_available:
            try:
                from src.rag.domain.query.colloquial_transformer import (
                    ColloquialTransformer,
                )

                self._transformer = ColloquialTransformer()
                self._transformer_available = True
                return self._transformer
            except ImportError:
                logger.warning(
                    "ColloquialTransformer not available, using rule-based detection"
                )
                self._transformer_available = False
                return None

        return None

    def optimize(
        self,
        query: str,
        override_weights: Optional[Tuple[float, float]] = None,
    ) -> WeightDecision:
        """
        Optimize hybrid search weights based on query formality.

        Args:
            query: The search query
            override_weights: Optional manual weight override (bm25, vector)

        Returns:
            WeightDecision with optimized weights and reasoning

        Example:
            >>> optimizer = HybridWeightOptimizer()
            >>> decision = optimizer.optimize("휴학 신청 방법")
            >>> print(f"BM25: {decision.bm25_weight}, Vector: {decision.vector_weight}")
            BM25: 0.5, Vector: 0.5  # Formal query
        """
        self._stats["total_queries"] += 1

        # Handle manual override
        if override_weights is not None:
            self._validate_weights(override_weights, "override")
            bm25_w, vector_w = override_weights
            return WeightDecision(
                bm25_weight=bm25_w,
                vector_weight=vector_w,
                is_colloquial=False,
                formality_score=0.5,
                reasoning="Manual weight override applied",
            )

        # Detect formality
        is_colloquial, formality_score, patterns = self._detect_formality(query)

        # Select weights based on formality
        if is_colloquial:
            bm25_w, vector_w = self._colloquial_weights
            self._stats["colloquial_queries"] += 1
            reasoning = (
                f"Colloquial query detected (score: {formality_score:.2f}). "
                f"Using higher vector weight for semantic matching."
            )
            if patterns:
                reasoning += f" Patterns: {', '.join(patterns[:3])}"
        else:
            bm25_w, vector_w = self._formal_weights
            self._stats["formal_queries"] += 1
            reasoning = (
                f"Formal query detected (score: {formality_score:.2f}). "
                f"Using balanced weights for keyword matching."
            )

        # Log decision
        if self._enable_logging:
            logger.info(
                f"Weight optimization: '{query[:30]}...' -> "
                f"BM25={bm25_w:.1f}, Vector={vector_w:.1f} "
                f"(formality={formality_score:.2f})"
            )

        return WeightDecision(
            bm25_weight=bm25_w,
            vector_weight=vector_w,
            is_colloquial=is_colloquial,
            formality_score=formality_score,
            detected_patterns=patterns,
            reasoning=reasoning,
        )

    def _detect_formality(
        self, query: str
    ) -> Tuple[bool, float, List[str]]:
        """
        Detect formality level of the query.

        Uses ColloquialTransformer if available, otherwise falls back
        to rule-based detection.

        Args:
            query: The search query

        Returns:
            Tuple of (is_colloquial, formality_score, detected_patterns)
            - is_colloquial: True if query is colloquial
            - formality_score: 0.0 (colloquial) to 1.0 (formal)
            - detected_patterns: List of colloquial patterns found
        """
        transformer = self._get_transformer()

        if transformer is not None:
            # Use ColloquialTransformer for accurate detection
            result = transformer.transform(query)

            # Formality score: inverse of whether transformation was applied
            # If transformed -> colloquial (low formality)
            # If not transformed -> formal (high formality)
            formality_score = 1.0 if not result.was_transformed else 0.3

            # Adjust score based on confidence
            if result.was_transformed:
                formality_score = 1.0 - result.confidence

            is_colloquial = formality_score < self.FORMALITY_THRESHOLD
            patterns = result.patterns_matched

            if patterns:
                self._stats["pattern_detections"] += len(patterns)

            return is_colloquial, formality_score, patterns

        # Fallback to rule-based detection
        return self._rule_based_formality_detection(query)

    def _rule_based_formality_detection(
        self, query: str
    ) -> Tuple[bool, float, List[str]]:
        """
        Rule-based formality detection as fallback.

        Detects common colloquial patterns in Korean queries.

        Args:
            query: The search query

        Returns:
            Tuple of (is_colloquial, formality_score, detected_patterns)
        """
        detected_patterns = []
        colloquial_score = 0.0

        # Common colloquial patterns
        colloquial_patterns = {
            # Question endings
            "어떻게 해": 0.8,
            "뭐야": 0.9,
            "뭐에요": 0.8,
            "어디야": 0.8,
            "언제야": 0.8,
            # Request patterns
            "알려줘": 0.7,
            "가르쳐줘": 0.7,
            "해줘": 0.7,
            "해주세요": 0.6,
            # Informal endings
            "이에요": 0.4,
            "예요": 0.4,
            "인가요": 0.4,
            # Casual question markers
            "노": 0.6,
            "나": 0.5,
            "니": 0.5,
        }

        query_lower = query.lower()

        for pattern, score in colloquial_patterns.items():
            if pattern in query_lower:
                detected_patterns.append(pattern)
                colloquial_score = max(colloquial_score, score)

        # Check for formal indicators
        formal_indicators = [
            "규정",
            "제",
            "조",
            "항",
            "신청",
            "방법",
            "절차",
            "기준",
            "요건",
            "안내",
        ]

        formal_count = sum(1 for indicator in formal_indicators if indicator in query)

        # Adjust score based on formal indicators
        if formal_count >= 2:
            colloquial_score = max(0, colloquial_score - 0.4)
        elif formal_count >= 1:
            colloquial_score = max(0, colloquial_score - 0.2)

        # Calculate formality score (inverse of colloquial score)
        formality_score = 1.0 - colloquial_score
        is_colloquial = formality_score < self.FORMALITY_THRESHOLD

        if detected_patterns:
            self._stats["pattern_detections"] += len(detected_patterns)

        return is_colloquial, formality_score, detected_patterns

    def get_weights(self, query: str) -> Tuple[float, float]:
        """
        Get optimized weights for a query.

        Convenience method that returns just the weights without metadata.

        Args:
            query: The search query

        Returns:
            Tuple of (bm25_weight, vector_weight)
        """
        decision = self.optimize(query)
        return decision.bm25_weight, decision.vector_weight

    def get_stats(self) -> Dict[str, any]:
        """
        Get optimizer statistics.

        Returns:
            Dictionary with optimization statistics
        """
        stats = self._stats.copy()

        if stats["total_queries"] > 0:
            stats["colloquial_rate"] = (
                stats["colloquial_queries"] / stats["total_queries"]
            )
            stats["formal_rate"] = (
                stats["formal_queries"] / stats["total_queries"]
            )
        else:
            stats["colloquial_rate"] = 0.0
            stats["formal_rate"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "total_queries": 0,
            "formal_queries": 0,
            "colloquial_queries": 0,
            "pattern_detections": 0,
        }

    def set_formal_weights(self, bm25: float, vector: float) -> None:
        """
        Set weights for formal queries.

        Args:
            bm25: BM25 weight
            vector: Vector search weight
        """
        self._validate_weights((bm25, vector), "formal")
        self._formal_weights = (bm25, vector)
        logger.info(f"Updated formal weights: BM25={bm25}, Vector={vector}")

    def set_colloquial_weights(self, bm25: float, vector: float) -> None:
        """
        Set weights for colloquial queries.

        Args:
            bm25: BM25 weight
            vector: Vector search weight
        """
        self._validate_weights((bm25, vector), "colloquial")
        self._colloquial_weights = (bm25, vector)
        logger.info(f"Updated colloquial weights: BM25={bm25}, Vector={vector}")


def create_hybrid_weight_optimizer(
    colloquial_transformer: Optional["ColloquialTransformer"] = None,
    enable_logging: bool = True,
) -> HybridWeightOptimizer:
    """
    Factory function to create a HybridWeightOptimizer.

    Args:
        colloquial_transformer: Optional ColloquialTransformer instance
        enable_logging: Whether to enable logging

    Returns:
        Configured HybridWeightOptimizer instance
    """
    return HybridWeightOptimizer(
        colloquial_transformer=colloquial_transformer,
        enable_logging=enable_logging,
    )
