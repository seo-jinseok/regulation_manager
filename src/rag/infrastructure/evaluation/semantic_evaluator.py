"""
Semantic Similarity Evaluator for RAG Quality Assessment.

Implements SPEC-RAG-QUALITY-003 Phase 3: Semantic Similarity Evaluation.

This module provides embedding-based semantic similarity evaluation
to replace keyword-based evaluation for better Korean language support.

Uses BGE-M3 embeddings (1024 dimensions) for high-quality semantic matching.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SemanticEvaluationResult:
    """Result of semantic similarity evaluation.

    Attributes:
        query: The original query
        answer: The generated answer
        expected: The expected answer/reference
        similarity_score: Cosine similarity score (0.0 to 1.0)
        is_relevant: Whether the answer is semantically relevant
        threshold: The similarity threshold used
        details: Additional evaluation details
    """

    query: str
    answer: str
    expected: str
    similarity_score: float
    is_relevant: bool
    threshold: float = 0.75
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Alias for is_relevant for consistency with other evaluators."""
        return self.is_relevant


@dataclass
class BatchEvaluationResult:
    """Result of batch semantic evaluation.

    Attributes:
        results: List of individual evaluation results
        average_score: Average similarity score
        pass_count: Number of passing evaluations
        fail_count: Number of failing evaluations
        pass_rate: Pass rate (0.0 to 1.0)
    """

    results: List[SemanticEvaluationResult] = field(default_factory=list)
    average_score: float = 0.0
    pass_count: int = 0
    fail_count: int = 0
    pass_rate: float = 0.0

    def __post_init__(self):
        """Calculate statistics after initialization."""
        if self.results:
            self.average_score = sum(r.similarity_score for r in self.results) / len(
                self.results
            )
            self.pass_count = sum(1 for r in self.results if r.is_relevant)
            self.fail_count = len(self.results) - self.pass_count
            self.pass_rate = self.pass_count / len(self.results)


class SemanticEvaluator:
    """
    Evaluates RAG responses using semantic similarity.

    Part of SPEC-RAG-QUALITY-003 Phase 3: Semantic Similarity Evaluation.

    This class provides:
    - Embedding-based semantic similarity calculation
    - Configurable similarity thresholds
    - Batch evaluation support
    - Caching for performance optimization

    Example:
        evaluator = SemanticEvaluator(embedding_model="BAAI/bge-m3")
        result = evaluator.evaluate_similarity(
            answer="휴학은 학기 시작 전에 신청할 수 있습니다.",
            expected="휴학 신청 기한은 학기 시작 14일 전까지입니다."
        )
        # result.similarity_score ≈ 0.85 (semantic match)
        # result.is_relevant = True (above threshold)
    """

    # Default similarity threshold
    DEFAULT_THRESHOLD = 0.75

    # Performance constraint: Maximum processing time (ms)
    MAX_PROCESSING_TIME_MS = 200

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-m3",
        similarity_threshold: float = DEFAULT_THRESHOLD,
        enable_cache: bool = True,
        max_cache_size: int = 500,
    ):
        """
        Initialize the semantic evaluator.

        Args:
            embedding_model: Name of the embedding model to use
            similarity_threshold: Threshold for considering answers relevant (0.0-1.0)
            enable_cache: Whether to enable embedding caching
            max_cache_size: Maximum number of embeddings to cache
        """
        self._embedding_model = embedding_model
        self._similarity_threshold = similarity_threshold
        self._enable_cache = enable_cache
        self._max_cache_size = max_cache_size

        # Embedding cache: text -> embedding vector
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Lazy-loaded embedding model
        self._model: Optional[Any] = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """Load the embedding model lazily."""
        if self._model_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._embedding_model)
            self._model_loaded = True
            logger.info(
                f"Loaded embedding model: {self._embedding_model} "
                f"(dimension: {self._model.get_sentence_embedding_dimension()})"
            )

        except ImportError as e:
            logger.warning(
                f"sentence-transformers not installed. "
                f"Install with: pip install sentence-transformers. Error: {e}"
            )
            self._model = None
            self._model_loaded = True

        except Exception as e:
            logger.error(
                f"Failed to load embedding model {self._embedding_model}: {e}"
            )
            self._model = None
            self._model_loaded = True

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text, using cache if available.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if unavailable
        """
        if not text or not text.strip():
            return None

        # Check cache
        if self._enable_cache and text in self._embedding_cache:
            return self._embedding_cache[text]

        # Load model if needed
        if not self._model_loaded:
            self._load_model()

        if self._model is None:
            return None

        try:
            # Generate embedding
            embedding = self._model.encode(text, convert_to_numpy=True)

            # Cache it
            if self._enable_cache:
                self._manage_cache()
                self._embedding_cache[text] = embedding

            return embedding

        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    def _manage_cache(self) -> None:
        """Manage cache size by removing old entries."""
        if len(self._embedding_cache) >= self._max_cache_size:
            # Remove half of the cache (simple LRU approximation)
            keys_to_remove = list(self._embedding_cache.keys())[
                : self._max_cache_size // 2
            ]
            for key in keys_to_remove:
                del self._embedding_cache[key]
            logger.debug(f"Cache cleanup: removed {len(keys_to_remove)} embeddings")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def evaluate_similarity(
        self,
        answer: str,
        expected: str,
        threshold: Optional[float] = None,
    ) -> SemanticEvaluationResult:
        """
        Evaluate semantic similarity between answer and expected.

        Args:
            answer: The generated answer
            expected: The expected answer/reference
            threshold: Override threshold (uses default if None)

        Returns:
            SemanticEvaluationResult with similarity score

        Example:
            >>> evaluator = SemanticEvaluator()
            >>> result = evaluator.evaluate_similarity(
            ...     answer="휴학은 학기 전에 신청 가능",
            ...     expected="휴학 신청은 학기 시작 전까지 가능합니다"
            ... )
            >>> print(result.is_relevant)
            True
        """
        threshold = threshold or self._similarity_threshold

        # Handle edge cases
        if not answer or not answer.strip():
            return SemanticEvaluationResult(
                query="",
                answer=answer or "",
                expected=expected,
                similarity_score=0.0,
                is_relevant=False,
                threshold=threshold,
                details={"error": "Empty answer"},
            )

        if not expected or not expected.strip():
            return SemanticEvaluationResult(
                query="",
                answer=answer,
                expected=expected or "",
                similarity_score=0.0,
                is_relevant=False,
                threshold=threshold,
                details={"error": "Empty expected"},
            )

        # Get embeddings
        answer_embedding = self._get_embedding(answer)
        expected_embedding = self._get_embedding(expected)

        if answer_embedding is None or expected_embedding is None:
            # Fallback to keyword-based similarity if embeddings unavailable
            return self._fallback_evaluate(answer, expected, threshold)

        # Calculate similarity
        similarity_score = self._cosine_similarity(answer_embedding, expected_embedding)

        # Determine relevance
        is_relevant = similarity_score >= threshold

        return SemanticEvaluationResult(
            query="",
            answer=answer,
            expected=expected,
            similarity_score=round(similarity_score, 4),
            is_relevant=is_relevant,
            threshold=threshold,
            details={
                "embedding_model": self._embedding_model,
                "method": "semantic",
            },
        )

    def evaluate_with_query(
        self,
        query: str,
        answer: str,
        expected: str,
        threshold: Optional[float] = None,
    ) -> SemanticEvaluationResult:
        """
        Evaluate semantic similarity with query context.

        This method considers the query when evaluating similarity,
        which can provide more nuanced results for RAG evaluation.

        Args:
            query: The original query
            answer: The generated answer
            expected: The expected answer/reference
            threshold: Override threshold

        Returns:
            SemanticEvaluationResult with similarity score
        """
        result = self.evaluate_similarity(answer, expected, threshold)
        result.query = query
        return result

    def batch_evaluate(
        self,
        answers: List[str],
        expected: List[str],
        threshold: Optional[float] = None,
    ) -> BatchEvaluationResult:
        """
        Batch evaluate multiple answer-expected pairs.

        Args:
            answers: List of generated answers
            expected: List of expected answers
            threshold: Override threshold

        Returns:
            BatchEvaluationResult with all results and statistics

        Example:
            >>> evaluator = SemanticEvaluator()
            >>> result = evaluator.batch_evaluate(
            ...     answers=["답변1", "답변2"],
            ...     expected=["정답1", "정답2"]
            ... )
            >>> print(f"Pass rate: {result.pass_rate:.2%}")
        """
        if len(answers) != len(expected):
            logger.warning(
                f"Length mismatch: {len(answers)} answers, {len(expected)} expected"
            )

        results: List[SemanticEvaluationResult] = []
        min_len = min(len(answers), len(expected))

        for i in range(min_len):
            result = self.evaluate_similarity(answers[i], expected[i], threshold)
            results.append(result)

        return BatchEvaluationResult(results=results)

    def _fallback_evaluate(
        self,
        answer: str,
        expected: str,
        threshold: float,
    ) -> SemanticEvaluationResult:
        """
        Fallback evaluation using keyword overlap when embeddings unavailable.

        Args:
            answer: The generated answer
            expected: The expected answer
            threshold: Similarity threshold

        Returns:
            SemanticEvaluationResult with keyword-based score
        """
        # Simple word overlap calculation
        answer_words = set(answer.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            similarity_score = 0.0
        else:
            overlap = len(answer_words & expected_words)
            similarity_score = overlap / len(expected_words)

        is_relevant = similarity_score >= threshold

        return SemanticEvaluationResult(
            query="",
            answer=answer,
            expected=expected,
            similarity_score=round(similarity_score, 4),
            is_relevant=is_relevant,
            threshold=threshold,
            details={
                "method": "keyword_fallback",
                "reason": "Embedding model unavailable",
            },
        )

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.debug("Embedding cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get evaluator statistics.

        Returns:
            Dictionary with evaluator statistics
        """
        return {
            "embedding_model": self._embedding_model,
            "similarity_threshold": self._similarity_threshold,
            "cache_enabled": self._enable_cache,
            "cache_size": len(self._embedding_cache),
            "max_cache_size": self._max_cache_size,
            "model_loaded": self._model_loaded,
        }

    def set_threshold(self, threshold: float) -> None:
        """
        Update the similarity threshold.

        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self._similarity_threshold = threshold
        else:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")


def create_semantic_evaluator(
    embedding_model: str = "BAAI/bge-m3",
    similarity_threshold: float = 0.75,
) -> SemanticEvaluator:
    """
    Factory function to create a SemanticEvaluator.

    Args:
        embedding_model: Name of the embedding model
        similarity_threshold: Similarity threshold for relevance

    Returns:
        Configured SemanticEvaluator instance
    """
    return SemanticEvaluator(
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
    )
