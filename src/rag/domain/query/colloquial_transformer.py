"""
Colloquial-to-Formal Query Transformer for Korean RAG System.

Implements SPEC-RAG-QUALITY-003 Phase 1: Colloquial Query Transformation.

This module transforms colloquial Korean queries (e.g., "휴학 어떻게 해?", "이거 뭐야?")
into formal Korean (e.g., "휴학 방법", "이거 정의") to improve retrieval quality.

The transformation preserves the original query intent while using vocabulary
that matches formal document text in the regulation database.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ColloquialPattern:
    """Represents a colloquial-to-formal pattern mapping."""

    pattern: str
    formal: str
    context: str = "general"

    def __post_init__(self):
        """Validate pattern after initialization."""
        if not self.pattern or not self.formal:
            raise ValueError("Pattern and formal must be non-empty strings")


@dataclass
class TransformResult:
    """Result of colloquial query transformation.

    Attributes:
        original_query: The input query before transformation
        transformed_query: The formal Korean query after transformation
        patterns_matched: List of patterns that were matched
        confidence: Confidence score (0.0 to 1.0) for transformation quality
        was_transformed: Whether any transformation was applied
        context_hints: Context hints detected from the query
        method: How transformation was achieved ("dictionary", "regex", "combined", "none")
    """

    original_query: str
    transformed_query: str
    patterns_matched: List[str] = field(default_factory=list)
    confidence: float = 1.0
    was_transformed: bool = False
    context_hints: List[str] = field(default_factory=list)
    method: str = "none"

    @property
    def transformation_applied(self) -> bool:
        """Alias for was_transformed for backward compatibility."""
        return self.was_transformed


class ColloquialTransformer:
    """
    Transforms colloquial Korean queries to formal Korean.

    Part of SPEC-RAG-QUALITY-003 Phase 1: Colloquial-to-Formal Query Transformation.

    This class provides:
    - Dictionary-based pattern matching for common colloquial expressions
    - Regex-based pattern transformation for structural patterns
    - Formal query detection to skip already-formal queries
    - Context hint extraction for query understanding

    Example:
        transformer = ColloquialTransformer()
        result = transformer.transform("휴학 어떻게 해?")
        # result.transformed_query = "휴학 방법"
        # result.was_transformed = True
    """

    # Default confidence levels
    HIGH_CONFIDENCE = 0.95  # Direct dictionary match
    MEDIUM_CONFIDENCE = 0.85  # Regex pattern match
    LOW_CONFIDENCE = 0.70  # Partial match

    # Performance constraint: Maximum processing time (ms)
    MAX_PROCESSING_TIME_MS = 50

    def __init__(
        self,
        patterns_path: Optional[str] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize the colloquial transformer.

        Args:
            patterns_path: Path to colloquial_patterns.json configuration file.
                          If None, uses default path.
            enable_logging: Whether to log transformation decisions for debugging.
        """
        self._enable_logging = enable_logging

        # Load patterns from configuration file
        if patterns_path is None:
            # Default path relative to this module
            base_dir = Path(__file__).parent.parent.parent.parent.parent
            patterns_path = str(base_dir / "data" / "config" / "colloquial_patterns.json")

        self._patterns_path = patterns_path
        self._mappings: List[ColloquialPattern] = []
        self._regex_patterns: List[Dict] = []
        self._formal_indicators: List[str] = []

        self._load_patterns()

        # Cache for transformation results
        self._cache: Dict[str, TransformResult] = {}
        self._cache_enabled = True

        # Unknown patterns queue for dictionary expansion
        self._unknown_patterns: List[str] = []

    def _load_patterns(self) -> None:
        """Load colloquial patterns from configuration file."""
        try:
            with open(self._patterns_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load mapping patterns
            for mapping in data.get("mappings", []):
                try:
                    pattern = ColloquialPattern(
                        pattern=mapping["pattern"],
                        formal=mapping["formal"],
                        context=mapping.get("context", "general"),
                    )
                    self._mappings.append(pattern)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Invalid mapping pattern: {mapping}, error: {e}")

            # Load regex patterns
            self._regex_patterns = data.get("regex_patterns", [])

            # Load formal indicators
            self._formal_indicators = data.get("formal_indicators", [])

            logger.info(
                f"Loaded {len(self._mappings)} mapping patterns, "
                f"{len(self._regex_patterns)} regex patterns, "
                f"{len(self._formal_indicators)} formal indicators"
            )

        except FileNotFoundError:
            logger.warning(
                f"Colloquial patterns file not found: {self._patterns_path}. "
                "Using empty patterns."
            )
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in patterns file: {e}")
            raise

    def transform(self, query: str) -> TransformResult:
        """
        Transform a colloquial query to formal Korean.

        Pipeline:
        1. Check if query is already formal (skip if yes)
        2. Apply dictionary-based pattern matching
        3. Apply regex-based pattern transformation
        4. Return transformed query with confidence score

        Args:
            query: The search query to transform

        Returns:
            TransformResult with transformed query and metadata

        Example:
            >>> transformer = ColloquialTransformer()
            >>> result = transformer.transform("휴학 어떻게 해?")
            >>> print(result.transformed_query)
            '휴학 방법'
        """
        if not query or not query.strip():
            return TransformResult(
                original_query=query,
                transformed_query=query or "",
                confidence=1.0,
                method="empty",
            )

        query = query.strip()

        # Check cache
        if self._cache_enabled and query in self._cache:
            return self._cache[query]

        # Check if query is already formal
        if self._is_formal_query(query):
            if self._enable_logging:
                logger.debug(f"Formal query detected, skipping transformation: '{query[:30]}...'")
            return TransformResult(
                original_query=query,
                transformed_query=query,
                confidence=1.0,
                method="formal_skip",
            )

        # Apply transformations
        transformed = query
        patterns_matched: List[str] = []
        context_hints: List[str] = []
        confidence = 1.0
        method = "none"

        # Step 1: Dictionary-based transformation
        dict_result, dict_matched = self._apply_dictionary_transform(query)
        if dict_matched:
            transformed = dict_result
            patterns_matched.extend(dict_matched)
            method = "dictionary"
            confidence = self.HIGH_CONFIDENCE

        # Step 2: Regex-based transformation
        regex_result, regex_matched = self._apply_regex_transform(transformed)
        if regex_matched:
            transformed = regex_result
            patterns_matched.extend(regex_matched)
            method = "regex" if method == "none" else "combined"
            confidence = min(confidence, self.MEDIUM_CONFIDENCE)

        # Extract context hints
        context_hints = self._extract_context_hints(query)

        # Determine if transformation was applied
        was_transformed = transformed != query

        result = TransformResult(
            original_query=query,
            transformed_query=transformed,
            patterns_matched=patterns_matched,
            confidence=confidence if was_transformed else 1.0,
            was_transformed=was_transformed,
            context_hints=context_hints,
            method=method,
        )

        # Log transformation decision
        if was_transformed and self._enable_logging:
            logger.info(
                f"Transformed: '{query}' -> '{transformed}' "
                f"(patterns: {patterns_matched}, confidence: {confidence:.2f})"
            )

        # Cache result
        if self._cache_enabled:
            self._cache[query] = result

        return result

    def detect_patterns(self, query: str) -> List[ColloquialPattern]:
        """
        Detect colloquial patterns in a query without transforming.

        Args:
            query: The query to analyze

        Returns:
            List of ColloquialPattern objects that match the query
        """
        if not query or not query.strip():
            return []

        query = query.strip()
        detected: List[ColloquialPattern] = []

        for pattern in self._mappings:
            if pattern.pattern in query:
                detected.append(pattern)

        return detected

    def _is_formal_query(self, query: str) -> bool:
        """
        Check if query is already formal regulation language.

        Formal queries contain regulation-specific terms and don't need transformation.

        Args:
            query: The search query

        Returns:
            True if query is formal regulation language, False otherwise
        """
        query_lower = query.lower()

        # Check for formal indicators (some are regex patterns)
        for indicator in self._formal_indicators:
            # Check if indicator looks like a regex pattern
            if '\\' in indicator or any(c in indicator for c in ['\\d', '+', '*']):
                try:
                    if re.search(indicator, query_lower):
                        return True
                except re.error:
                    # If regex fails, try simple string match
                    if indicator in query_lower:
                        return True
            else:
                # Simple string match for non-regex indicators
                if indicator in query_lower:
                    return True

        return False

    def _apply_dictionary_transform(self, query: str) -> tuple[str, List[str]]:
        """
        Apply dictionary-based pattern transformation.

        Args:
            query: The query to transform

        Returns:
            Tuple of (transformed_query, list of matched patterns)
        """
        transformed = query
        matched: List[str] = []

        for pattern in self._mappings:
            if pattern.pattern in transformed:
                # Replace pattern with formal equivalent
                transformed = transformed.replace(pattern.pattern, pattern.formal)
                matched.append(f"{pattern.pattern}->{pattern.formal}")

        return transformed, matched

    def _apply_regex_transform(self, query: str) -> tuple[str, List[str]]:
        """
        Apply regex-based pattern transformation.

        Args:
            query: The query to transform

        Returns:
            Tuple of (transformed_query, list of matched patterns)
        """
        transformed = query
        matched: List[str] = []

        for regex_pattern in self._regex_patterns:
            pattern_str = regex_pattern.get("pattern", "")
            replacement = regex_pattern.get("replacement", "")

            if not pattern_str or not replacement:
                continue

            try:
                pattern = re.compile(pattern_str)
                new_transformed = pattern.sub(replacement, transformed)

                if new_transformed != transformed:
                    matched.append(f"regex:{pattern_str}")
                    transformed = new_transformed

            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern_str}': {e}")
                continue

        return transformed, matched

    def _extract_context_hints(self, query: str) -> List[str]:
        """
        Extract context hints from the query.

        Context hints help identify the type of information the user is seeking.

        Args:
            query: The search query

        Returns:
            List of context hint strings
        """
        hints: List[str] = []

        # Check for context patterns
        context_keywords = {
            "procedure": ["어떻게", "하는법", "절차", "방법", "신청"],
            "definition": ["뭐야", "뭐에요", "정의", "무엇"],
            "deadline": ["언제", "기한", "까지"],
            "location": ["어디", "위치"],
            "eligibility": ["가능", "되나", "할수"],
            "requirements": ["필요", "서류", "조건"],
        }

        query_lower = query.lower()

        for context, keywords in context_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    hints.append(context)
                    break

        return list(set(hints))  # Remove duplicates

    def get_unknown_patterns(self) -> List[str]:
        """
        Get list of patterns that weren't recognized.

        Returns:
            List of unknown pattern strings queued for dictionary expansion
        """
        return self._unknown_patterns.copy()

    def queue_unknown_pattern(self, pattern: str) -> None:
        """
        Queue an unrecognized pattern for dictionary expansion.

        Args:
            pattern: The unrecognized pattern to queue
        """
        if pattern and pattern not in self._unknown_patterns:
            self._unknown_patterns.append(pattern)
            logger.debug(f"Queued unknown pattern for expansion: '{pattern}'")

    def clear_cache(self) -> None:
        """Clear the transformation cache."""
        self._cache.clear()
        logger.debug("Transformation cache cleared")

    def get_stats(self) -> Dict:
        """
        Get transformer statistics.

        Returns:
            Dictionary with transformer statistics
        """
        return {
            "total_mappings": len(self._mappings),
            "total_regex_patterns": len(self._regex_patterns),
            "total_formal_indicators": len(self._formal_indicators),
            "cache_size": len(self._cache),
            "unknown_patterns_queued": len(self._unknown_patterns),
        }


def create_colloquial_transformer(
    patterns_path: Optional[str] = None,
) -> ColloquialTransformer:
    """
    Factory function to create a ColloquialTransformer.

    Args:
        patterns_path: Optional path to patterns configuration file

    Returns:
        Configured ColloquialTransformer instance
    """
    return ColloquialTransformer(patterns_path=patterns_path)
