"""
Enhanced BGE Reranker for Regulation RAG System (Cycle 5).

Features:
- Korean-specific reranker model support
- A/B testing framework for model comparison
- Automatic model selection based on configuration
- Performance metrics tracking

Supported Korean Models:
- Dongjin-kr/kr-reranker: Korean-specific cross-encoder
- NLPai/ko-reranker: NLPai Korean reranker

Supported Multilingual Models:
- BAAI/bge-reranker-v2-m3: Primary multilingual model
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..config import get_config
from .ab_test_framework import (
    ABTestManager,
    create_ab_manager,
)

if TYPE_CHECKING:
    from ..domain.entities import SearchResult

logger = logging.getLogger(__name__)

# Model storage
_rerankers: Dict[str, "FlagReranker"] = {}
_ab_manager: Optional[ABTestManager] = None


@dataclass
class RerankedResult:
    """A reranked document with its relevance score."""

    doc_id: str
    content: str
    score: float
    original_rank: int
    metadata: dict
    model_used: str = ""  # Track which model produced this result


def get_ab_manager() -> ABTestManager:
    """
    Get or initialize the A/B test manager.
    
    Returns:
        ABTestManager instance.
    """
    global _ab_manager

    if _ab_manager is None:
        config = get_config()
        reranker_config = config.reranker

        if reranker_config.enable_ab_testing:
            _ab_manager = create_ab_manager(
                control_model=reranker_config.primary_model,
                test_models=reranker_config.korean_models,
                test_ratio=reranker_config.ab_test_ratio,
            )
            logger.info(
                f"Initialized A/B test manager: "
                f"control={reranker_config.primary_model}, "
                f"test={reranker_config.korean_models}, "
                f"ratio={reranker_config.ab_test_ratio}"
            )
        else:
            # A/B testing disabled, use primary model only
            _ab_manager = create_ab_manager(
                control_model=reranker_config.primary_model,
                test_models=[],
                test_ratio=0.0,
            )

    return _ab_manager


def load_model(model_name: str, use_fp16: bool = True) -> "FlagReranker":
    """
    Load a reranker model.
    
    Args:
        model_name: HuggingFace model identifier.
        use_fp16: Whether to use FP16 precision.
        
    Returns:
        FlagReranker instance.
        
    Raises:
        RerankerError: If model fails to load.
    """
    global _rerankers

    if model_name in _rerankers:
        return _rerankers[model_name]

    try:
        from FlagEmbedding import FlagReranker

        logger.info(f"Loading reranker model: {model_name}")
        start_time = time.time()

        _rerankers[model_name] = FlagReranker(
            model_name,
            use_fp16=use_fp16,
        )

        load_time = time.time() - start_time
        logger.info(f"Model {model_name} loaded in {load_time:.2f}s")

        return _rerankers[model_name]

    except ImportError as e:
        from ..exceptions import RerankerError
        raise RerankerError(
            "FlagEmbedding is required. Install with: uv add FlagEmbedding",
            model=model_name,
        ) from e
    except Exception as e:
        from ..exceptions import RerankerError
        raise RerankerError(
            f"Failed to load reranker model {model_name}: {e}",
            model=model_name,
        ) from e


def select_model_for_query(
    query: str,
    strategy: str = "ab_test",
    korean_models: Optional[List[str]] = None,
) -> str:
    """
    Select appropriate reranker model for the query.
    
    Args:
        query: Search query text.
        strategy: Model selection strategy.
        korean_models: List of Korean model names.
        
    Returns:
        Selected model name.
    """
    config = get_config()
    primary_model = config.reranker.primary_model
    korean_models = korean_models or config.reranker.korean_models

    if strategy == "korean_only" and korean_models:
        return korean_models[0]
    elif strategy == "multilingual_only":
        return primary_model
    elif strategy == "ab_test":
        ab_manager = get_ab_manager()
        return ab_manager.select_model()
    else:
        # Default: use A/B testing
        ab_manager = get_ab_manager()
        return ab_manager.select_model()


def compute_scores(
    model_name: str,
    pairs: List[Tuple[str, str]],
    normalize: bool = True,
) -> List[float]:
    """
    Compute relevance scores using specified model.
    
    Args:
        model_name: Model to use for scoring.
        pairs: List of (query, document) pairs.
        normalize: Whether to normalize scores.
        
    Returns:
        List of relevance scores.
    """
    config = get_config()
    model = load_model(model_name, use_fp16=config.reranker.use_fp16)

    scores = model.compute_score(pairs, normalize=normalize)

    # Handle single document case
    if isinstance(scores, float):
        scores = [scores]

    return scores


def rerank(
    query: str,
    documents: List[Tuple[str, str, dict]],
    top_k: int = 10,
    model_name: Optional[str] = None,
) -> List[RerankedResult]:
    """
    Rerank documents using cross-encoder model.
    
    Args:
        query: The search query.
        documents: List of (doc_id, content, metadata) tuples.
        top_k: Maximum number of results to return.
        model_name: Optional specific model to use.
        
    Returns:
        List of RerankedResult sorted by relevance score.
    """
    if not documents:
        return []

    config = get_config()

    # Select model
    if model_name is None:
        model_name = select_model_for_query(
            query,
            strategy=config.reranker.model_selection_strategy,
        )

    # Compute scores with timing
    start_time = time.time()

    try:
        scores = compute_scores(
            model_name,
            [(query, doc[1]) for doc in documents],
            normalize=True,
        )
        success = True
        error = None
    except Exception as e:
        logger.error(f"Reranking failed with {model_name}: {e}")

        # Fallback to primary model if enabled
        if config.reranker.fallback_to_multilingual and model_name != config.reranker.primary_model:
            logger.info(f"Falling back to {config.reranker.primary_model}")
            model_name = config.reranker.primary_model
            scores = compute_scores(
                model_name,
                [(query, doc[1]) for doc in documents],
                normalize=True,
            )
            success = True
            error = str(e)
        else:
            success = False
            error = str(e)
            scores = [0.0] * len(documents)

    latency_ms = (time.time() - start_time) * 1000

    # Record metrics
    ab_manager = get_ab_manager()
    ab_manager.record_result(model_name, latency_ms, success)

    # Create results
    results = []
    for i, (doc_id, content, metadata) in enumerate(documents):
        results.append(
            RerankedResult(
                doc_id=doc_id,
                content=content,
                score=scores[i],
                original_rank=i + 1,
                metadata=metadata,
                model_used=model_name,
            )
        )

    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)

    return results[:top_k]


def rerank_search_results(
    query: str,
    search_results: List["SearchResult"],
    top_k: int = 10,
    model_name: Optional[str] = None,
) -> List["SearchResult"]:
    """
    Rerank SearchResult objects from the RAG system.
    
    Args:
        query: The search query.
        search_results: List of SearchResult objects.
        top_k: Maximum number of results to return.
        model_name: Optional specific model to use.
        
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
    reranked = rerank(query, documents, top_k=top_k, model_name=model_name)

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


def clear_reranker():
    """Release all reranker models from memory."""
    global _rerankers, _ab_manager
    _rerankers = {}
    _ab_manager = None
    logger.info("All reranker models cleared from memory")


def warmup_reranker(model_name: Optional[str] = None) -> None:
    """
    Pre-load reranker models for faster first query.
    
    Args:
        model_name: Optional specific model to warmup.
                   If None, warms up all configured models.
    """
    config = get_config()

    if model_name:
        models = [model_name]
    else:
        models = [config.reranker.primary_model] + config.reranker.korean_models

    for model in models:
        try:
            load_model(model, use_fp16=config.reranker.use_fp16)
            logger.info(f"Warmed up reranker: {model}")
        except Exception as e:
            logger.warning(f"Failed to warmup {model}: {e}")


def get_model_performance_summary() -> Dict:
    """
    Get performance summary of all reranker models.
    
    Returns:
        Dictionary with model performance metrics.
    """
    ab_manager = get_ab_manager()
    return ab_manager.get_summary()


class KoreanReranker:
    """
    Korean-specific reranker with automatic model selection.
    
    This class provides a simple interface for Korean document reranking
    with automatic model selection and performance tracking.
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
        self._model_name = model_name
        self._use_ab_testing = use_ab_testing

    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, str, dict]],
        top_k: int = 10,
    ) -> List[tuple]:
        """
        Rerank Korean documents.
        
        Args:
            query: Search query in Korean.
            documents: List of (doc_id, content, metadata) tuples.
            top_k: Maximum number of results.
            
        Returns:
            List of (doc_id, content, score, metadata) tuples.
        """
        if not self._use_ab_testing and self._model_name:
            # Direct mode: use specified model
            results = rerank(query, documents, top_k=top_k, model_name=self._model_name)
        else:
            # A/B testing mode: let framework select model
            results = rerank(query, documents, top_k=top_k, model_name=self._model_name)

        return [
            (r.doc_id, r.content, r.score, r.metadata)
            for r in results
        ]

    def rerank_with_context(
        self,
        query: str,
        documents: List[Tuple[str, str, dict]],
        context: Optional[dict] = None,
        top_k: int = 10,
    ) -> List[tuple]:
        """
        Rerank with metadata context boosting.
        
        Args:
            query: Search query.
            documents: List of (doc_id, content, metadata) tuples.
            context: Optional context dict for boosting.
            top_k: Maximum number of results.
            
        Returns:
            List of (doc_id, content, score, metadata) tuples.
        """
        if not documents:
            return []

        context = context or {}

        # Get base reranked results
        results = rerank(query, documents, top_k=len(documents))

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
                doc_regulation = metadata.get("regulation_title") or metadata.get("규정명", "")
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
                    model_used=r.model_used,
                )
            )

        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x.score, reverse=True)

        return [
            (r.doc_id, r.content, r.score, r.metadata)
            for r in boosted_results[:top_k]
        ]
