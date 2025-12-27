"""
BGE Reranker for Regulation RAG System.

Provides cross-encoder based reranking to improve search result quality.
Uses BAAI/bge-reranker-v2-m3 for multilingual support (including Korean).
"""

from typing import List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

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
        except ImportError:
            raise ImportError(
                "FlagEmbedding is required for reranking. "
                "Install with: uv add FlagEmbedding"
            )
    
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
        results.append(RerankedResult(
            doc_id=doc_id,
            content=content,
            score=scores[i],
            original_rank=i + 1,
            metadata=metadata,
        ))
    
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
        (r.chunk.id, r.chunk.text, r.chunk.to_metadata())
        for r in search_results
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
            reranked_results.append(SearchResult(
                chunk=original.chunk,
                score=rr.score,
                rank=i + 1,
            ))
    
    return reranked_results


# Optional: Provide a way to release memory
def clear_reranker():
    """Release the reranker model from memory."""
    global _reranker
    _reranker = None
