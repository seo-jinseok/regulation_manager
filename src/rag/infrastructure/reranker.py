"""
BGE Reranker for Regulation RAG System.

Provides cross-encoder based reranking to improve search result quality.
Uses BAAI/bge-reranker-v2-m3 for multilingual support (including Korean).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..domain.repositories import IReranker

# Import extended functionality for Korean models and A/B testing
try:
    from . import reranker_extended as _extended
    _EXTENDED_AVAILABLE = True
except ImportError:
    _EXTENDED_AVAILABLE = False

if TYPE_CHECKING:
    from ..domain.entities import SearchResult

# Lazy loading to avoid slow import on startup
_reranker = None
_model_name = "BAAI/bge-reranker-v2-m3"


@dataclass
class RerankedResult:
    """A reranked document with its relevance score."""

    doc_id: str
    content: str
    score: float
    original_rank: int
    metadata: dict


def get_reranker(model_name: Optional[str] = None):
    """
    Get or initialize the BGE reranker (singleton pattern).

    Args:
        model_name: Optional model name. Defaults to bge-reranker-v2-m3.

    Returns:
        FlagReranker instance.
    """
    global _reranker, _model_name

    if model_name:
        _model_name = model_name

    if _reranker is None:
        try:
            from FlagEmbedding import FlagReranker

            _reranker = FlagReranker(
                _model_name,
                use_fp16=True,  # Use FP16 for faster inference on Apple Silicon
            )
        except ImportError as e:
            from ..exceptions import RerankerError

            raise RerankerError(
                "FlagEmbedding is required. Install with: uv add FlagEmbedding",
                model=_model_name,
            ) from e
        except Exception as e:
            from ..exceptions import RerankerError

            raise RerankerError(
                f"Failed to initialize reranker: {e}",
                model=_model_name,
            ) from e

    return _reranker


def rerank(
    query: str,
    documents: List[Tuple[str, str, dict]],
    top_k: int = 10,
) -> List[RerankedResult]:
    """
    Rerank documents using BGE cross-encoder.

    Args:
        query: The search query.
        documents: List of (doc_id, content, metadata) tuples.
        top_k: Maximum number of results to return.

    Returns:
        List of RerankedResult sorted by relevance score.
    """
    if not documents:
        return []

    reranker = get_reranker()

    # Prepare query-document pairs
    pairs = [[query, doc[1]] for doc in documents]

    # Compute relevance scores
    scores = reranker.compute_score(pairs, normalize=True)

    # Handle single document case (returns float instead of list)
    if isinstance(scores, float):
        scores = [scores]

    # Create results with scores
    results = []
    for i, (doc_id, content, metadata) in enumerate(documents):
        results.append(
            RerankedResult(
                doc_id=doc_id,
                content=content,
                score=scores[i],
                original_rank=i + 1,
                metadata=metadata,
            )
        )

    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)

    return results[:top_k]


def rerank_search_results(
    query: str,
    search_results: List["SearchResult"],
    top_k: int = 10,
) -> List["SearchResult"]:
    """
    Rerank SearchResult objects from the RAG system.

    Args:
        query: The search query.
        search_results: List of SearchResult objects.
        top_k: Maximum number of results to return.

    Returns:
        List of SearchResult objects, reranked by relevance.
    """
    if not search_results:
        return []

    # Import from domain layer to avoid circular imports
    from ..domain.entities import SearchResult

    # Convert SearchResult to tuples for reranking
    documents = [
        (r.chunk.id, r.chunk.text, r.chunk.to_metadata()) for r in search_results
    ]

    # Rerank
    reranked = rerank(query, documents, top_k=top_k)

    # Map back to SearchResult objects
    id_to_result = {r.chunk.id: r for r in search_results}

    reranked_results = []
    for i, rr in enumerate(reranked):
        original = id_to_result.get(rr.doc_id)
        if original:
            # Create new SearchResult with reranked score
            reranked_results.append(
                SearchResult(
                    chunk=original.chunk,
                    score=rr.score,
                    rank=i + 1,
                )
            )

    return reranked_results


# Optional: Provide a way to release memory
def clear_reranker():
    """Release the reranker model from memory."""
    global _reranker
    _reranker = None


def warmup_reranker(model_name: Optional[str] = None) -> None:
    """
    Pre-load the reranker model for faster first query.

    Args:
        model_name: Optional model name. Defaults to bge-reranker-v2-m3.
    """
    get_reranker(model_name)


class BGEReranker(IReranker):
    """
    BGE cross-encoder reranker implementation.

    Implements IReranker interface for dependency injection.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize BGE reranker.

        Args:
            model_name: Optional model name. Defaults to bge-reranker-v2-m3.
        """
        self._model_name = model_name

    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, str, dict]],
        top_k: int = 10,
    ) -> List[tuple]:
        """
        Rerank documents using BGE cross-encoder.

        Args:
            query: The search query.
            documents: List of (doc_id, content, metadata) tuples.
            top_k: Maximum number of results to return.

        Returns:
            List of (doc_id, content, score, metadata) tuples sorted by relevance.
        """
        if not documents:
            return []

        results = rerank(query, documents, top_k)
        return [(r.doc_id, r.content, r.score, r.metadata) for r in results]

    def rerank_with_context(
        self,
        query: str,
        documents: List[Tuple[str, str, dict]],
        context: Optional[dict] = None,
        top_k: int = 10,
    ) -> List[tuple]:
        """
        Rerank documents using BGE cross-encoder with metadata context boosting.

        Args:
            query: The search query.
            documents: List of (doc_id, content, metadata) tuples.
            context: Optional context dict with:
                - target_regulation: Boost documents from this regulation.
                - target_audience: Boost documents for this audience.
                - regulation_boost: Boost factor for matching regulation (default: 0.15).
                - audience_boost: Boost factor for matching audience (default: 0.1).
            top_k: Maximum number of results to return.

        Returns:
            List of (doc_id, content, score, metadata) tuples sorted by relevance.
        """
        if not documents:
            return []

        context = context or {}

        # Get base reranked results
        results = rerank(query, documents, top_k=len(documents))  # Get all, then filter

        # Apply context-based boosting
        target_regulation = context.get("target_regulation")
        target_audience = context.get("target_audience")
        regulation_boost = context.get("regulation_boost", 0.15)
        audience_boost = context.get("audience_boost", 0.1)

        boosted_results = []
        for r in results:
            boosted_score = r.score
            metadata = r.metadata

            # Boost matching regulation
            if target_regulation:
                doc_regulation = metadata.get("regulation_title") or metadata.get(
                    "규정명", ""
                )
                if target_regulation.lower() in doc_regulation.lower():
                    boosted_score = min(1.0, boosted_score + regulation_boost)

            # Boost matching audience
            if target_audience:
                doc_audience = metadata.get("audience", "all")
                if doc_audience == target_audience or doc_audience == "all":
                    boosted_score = min(1.0, boosted_score + audience_boost)

            boosted_results.append(
                RerankedResult(
                    doc_id=r.doc_id,
                    content=r.content,
                    score=boosted_score,
                    original_rank=r.original_rank,
                    metadata=r.metadata,
                )
            )

        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x.score, reverse=True)

        return [
            (r.doc_id, r.content, r.score, r.metadata) for r in boosted_results[:top_k]
        ]

class KoreanReranker(IReranker):
    """
    Korean-specific reranker with automatic model selection (Cycle 5).
    
    Supports:
    - Korean-specific models (Dongjin-kr/kr-reranker, NLPai/ko-reranker)
    - A/B testing framework
    - Automatic fallback to multilingual model
    
    This is a convenience wrapper around the extended reranker functionality.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_ab_testing: bool = True,
    ):
        """
        Initialize Korean reranker.
        
        Args:
            model_name: Specific model to use (None for auto-selection).
            use_ab_testing: Whether to use A/B testing framework.
        """
        if _EXTENDED_AVAILABLE:
            self._impl = _extended.KoreanReranker(model_name, use_ab_testing)
        else:
            # Fallback to BGE reranker if extended module not available
            logger.warning("Extended reranker not available, using BGEReranker")
            self._impl = BGEReranker(model_name)
        self._model_name = model_name

    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, str, dict]],
        top_k: int = 10,
    ) -> List[tuple]:
        """Rerank Korean documents."""
        return self._impl.rerank(query, documents, top_k)

    def rerank_with_context(
        self,
        query: str,
        documents: List[Tuple[str, str, dict]],
        context: Optional[dict] = None,
        top_k: int = 10,
    ) -> List[tuple]:
        """Rerank with metadata context boosting."""
        if hasattr(self._impl, 'rerank_with_context'):
            return self._impl.rerank_with_context(query, documents, context, top_k)
        else:
            # Fallback for BGEReranker
            return self._impl.rerank_with_context(query, documents, context, top_k)


# Convenience functions for A/B testing framework
def get_ab_manager():
    """Get the A/B test manager instance."""
    if _EXTENDED_AVAILABLE:
        return _extended.get_ab_manager()
    else:
        raise ImportError("Extended reranker module not available")


def get_model_performance_summary() -> Dict:
    """Get performance summary of all reranker models."""
    if _EXTENDED_AVAILABLE:
        return _extended.get_model_performance_summary()
    else:
        raise ImportError("Extended reranker module not available")


def warmup_all_models() -> None:
    """Pre-load all configured reranker models."""
    if _EXTENDED_AVAILABLE:
        _extended.warmup_reranker()
    else:
        _extended.warmup_reranker()
