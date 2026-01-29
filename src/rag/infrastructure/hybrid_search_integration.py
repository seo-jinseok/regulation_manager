"""
Hybrid Search Integration for BM25 + Dense Retrieval.

Integrates Korean-optimized Dense Retriever with existing BM25 system
for hybrid semantic search with improved recall.
"""

import logging
from typing import Dict, List, Optional, Tuple

from .dense_retriever import DenseRetriever, DenseRetrieverConfig
from .hybrid_search import HybridSearcher, ScoredDocument

logger = logging.getLogger(__name__)


class DenseHybridSearcher(HybridSearcher):
    """
    Enhanced Hybrid Searcher with Korean-optimized Dense Retrieval.

    Combines:
    - BM25 sparse retrieval (keyword matching)
    - Korean-optimized dense retrieval (semantic understanding)
    - Dynamic weight adjustment based on query type
    - Reciprocal Rank Fusion (RRF) for result fusion

    Query Type Weights:
    - ARTICLE_REFERENCE: BM25 only (1.0, 0.0) - 정확한 조호 참조
    - REGULATION_NAME: BM25 + Dense (0.7, 0.3) - 규정명 검색
    - NATURAL_QUESTION: BM25 + Dense (0.6, 0.4) - 자연어 질문
    - INTENT: BM25 + Dense (0.5, 0.5) - 의도 기반 검색
    - GENERAL: BM25 + Dense (0.6, 0.4) - 기본 검색
    """

    def __init__(
        self,
        bm25_weight: float = 0.6,
        dense_weight: float = 0.4,
        rrf_k: int = 60,
        use_dynamic_weights: bool = True,
        use_dynamic_rrf_k: bool = False,
        synonyms_path: Optional[str] = None,
        intents_path: Optional[str] = None,
        index_cache_path: Optional[str] = None,
        tokenize_mode: Optional[str] = None,
        dense_model_name: str = "jhgan/ko-sbert-multinli",
        dense_config: Optional[DenseRetrieverConfig] = None,
    ):
        """
        Initialize Hybrid Searcher with Dense Retrieval.

        Args:
            bm25_weight: Default weight for BM25 scores.
            dense_weight: Default weight for dense scores.
            rrf_k: RRF ranking constant.
            use_dynamic_weights: Enable query-based dynamic weighting.
            use_dynamic_rrf_k: Enable query-based dynamic RRF k value.
            synonyms_path: Path to synonyms JSON file.
            intents_path: Path to intents JSON file.
            index_cache_path: Path to cache BM25 index.
            tokenize_mode: BM25 tokenizer mode.
            dense_model_name: Korean embedding model name.
            dense_config: Optional Dense Retriever configuration.
        """
        # Initialize parent HybridSearcher (BM25)
        super().__init__(
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            rrf_k=rrf_k,
            use_dynamic_weights=use_dynamic_weights,
            use_dynamic_rrf_k=use_dynamic_rrf_k,
            synonyms_path=synonyms_path,
            intents_path=intents_path,
            index_cache_path=index_cache_path,
            tokenize_mode=tokenize_mode,
        )

        # Initialize Dense Retriever
        self.dense_model_name = dense_model_name
        self.dense_config = dense_config or DenseRetrieverConfig(
            batch_size=32, cache_embeddings=True
        )
        self._dense_retriever: Optional[DenseRetriever] = None

    def _get_dense_retriever(self) -> DenseRetriever:
        """Lazy-initialize Dense Retriever."""
        if self._dense_retriever is None:
            self._dense_retriever = DenseRetriever(
                model_name=self.dense_model_name, config=self.dense_config
            )
            logger.info(f"Dense Retriever initialized: {self.dense_model_name}")
        return self._dense_retriever

    def add_documents(self, documents: List[Tuple[str, str, Dict]]) -> None:
        """
        Add documents to both BM25 and Dense indices.

        Args:
            documents: List of (doc_id, content, metadata) tuples.
        """
        # Add to BM25 index (parent method)
        super().add_documents(documents)

        # Add to Dense index
        dense_retriever = self._get_dense_retriever()
        dense_retriever.add_documents(documents)

        logger.info(
            f"Added {len(documents)} documents to hybrid index "
            f"(BM25 + Dense: {self.dense_model_name})"
        )

    def search_dense(
        self,
        query: str,
        top_k: int = 20,
        score_threshold: Optional[float] = None,
    ) -> List[ScoredDocument]:
        """
        Perform dense semantic search.

        Args:
            query: The search query.
            top_k: Maximum number of results.
            score_threshold: Optional minimum similarity score.

        Returns:
            List of ScoredDocument sorted by similarity.
        """
        dense_retriever = self._get_dense_retriever()
        results = dense_retriever.search(
            query, top_k=top_k, score_threshold=score_threshold
        )

        # Convert to ScoredDocument format
        return [
            ScoredDocument(
                doc_id=doc_id,
                score=score,
                content=content,
                metadata=metadata,
            )
            for doc_id, score, content, metadata in results
        ]

    def fuse_results(
        self,
        sparse_results: List[ScoredDocument],
        dense_results: List[ScoredDocument],
        top_k: int = 10,
        query_text: Optional[str] = None,
    ) -> List[ScoredDocument]:
        """
        Fuse sparse and dense results using weighted RRF.

        Overrides parent method to add Dense results when not provided.

        Args:
            sparse_results: Results from BM25 search.
            dense_results: Results from dense/embedding search (optional, auto-generated if None).
            top_k: Maximum number of results.
            query_text: Optional query for dynamic weight and k calculation.

        Returns:
            Fused results sorted by combined score.
        """
        # Auto-generate dense results if not provided
        if dense_results is None and query_text:
            dense_results = self.search_dense(query_text, top_k=top_k * 2)

        return super().fuse_results(
            sparse_results=sparse_results,
            dense_results=dense_results,
            top_k=top_k,
            query_text=query_text,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_dense: bool = True,
        score_threshold: Optional[float] = None,
    ) -> List[ScoredDocument]:
        """
        Perform hybrid search combining BM25 and Dense retrieval.

        Args:
            query: The search query.
            top_k: Maximum number of results.
            use_dense: Whether to use dense retrieval (default: True).
            score_threshold: Optional minimum similarity score for dense results.

        Returns:
            Fused results sorted by combined score.
        """
        # Always perform sparse search
        sparse_results = self.search_sparse(query, top_k=top_k * 2)

        # Perform dense search if enabled
        dense_results = None
        if use_dense:
            dense_results = self.search_dense(
                query, top_k=top_k * 2, score_threshold=score_threshold
            )

        # Fuse results with dynamic weights
        return self.fuse_results(
            sparse_results=sparse_results,
            dense_results=dense_results,
            top_k=top_k,
            query_text=query,
        )

    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms (delegates to parent).

        Args:
            query: The original search query.

        Returns:
            Expanded query with synonyms appended.
        """
        return super().expand_query(query)

    def clear(self) -> None:
        """Clear both BM25 and Dense indices."""
        super().clear()  # Clear BM25

        if self._dense_retriever is not None:
            self._dense_retriever.clear()
            logger.info("Dense retriever cleared")

    def get_cache_stats(self) -> Dict[str, Dict]:
        """
        Get cache statistics for both BM25 and Dense retrieval.

        Returns:
            Dictionary with BM25 and Dense cache stats.
        """
        stats = {"bm25": {"indexed_docs": len(self.bm25.documents)}}

        if self._dense_retriever is not None:
            stats["dense"] = self._dense_retriever.get_cache_stats()

        return stats

    def save_dense_index(self, path: str) -> None:
        """
        Save dense vector index to disk.

        Args:
            path: File path to save the index.
        """
        dense_retriever = self._get_dense_retriever()
        dense_retriever.save_index(path)
        logger.info(f"Dense index saved to {path}")

    def load_dense_index(self, path: str) -> bool:
        """
        Load dense vector index from disk.

        Args:
            path: File path to load the index from.

        Returns:
            True if loaded successfully.
        """
        dense_retriever = self._get_dense_retriever()
        success = dense_retriever.load_index(path)

        if success:
            logger.info(f"Dense index loaded from {path}")
        else:
            logger.warning(f"Failed to load dense index from {path}")

        return success

    @classmethod
    def get_recommended_models(cls) -> Dict[str, str]:
        """
        Get recommended Korean embedding models.

        Returns:
            Dictionary mapping use cases to model names.
        """
        return {
            "accuracy": "BAAI/bge-m3",  # Highest accuracy, slower
            "balanced": "jhgan/ko-sbert-multinli",  # Good balance of speed/accuracy
            "speed": "jhgan/ko-sbert-sts",  # Fastest, decent accuracy
        }


def create_hybrid_searcher(
    dense_model_name: str = "jhgan/ko-sbert-multinli",
    use_dynamic_weights: bool = True,
    **kwargs,
) -> DenseHybridSearcher:
    """
    Factory function to create a Hybrid Searcher with Dense Retrieval.

    Args:
        dense_model_name: Korean embedding model name.
        use_dynamic_weights: Enable query-based dynamic weighting.
        **kwargs: Additional arguments passed to DenseHybridSearcher.

    Returns:
        Configured DenseHybridSearcher instance.

    Example:
        >>> from src.rag.infrastructure.hybrid_search_integration import create_hybrid_searcher
        >>> searcher = create_hybrid_searcher()
        >>> searcher.add_documents([("doc1", "휴학 규정", {})])
        >>> results = searcher.search("휴학 절차", top_k=5)
    """
    return DenseHybridSearcher(
        dense_model_name=dense_model_name,
        use_dynamic_weights=use_dynamic_weights,
        **kwargs,
    )


if __name__ == "__main__":
    # Example usage

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create sample documents
    sample_docs = [
        ("doc1", "휴학 신청은 학기 시작 14일 전까지 가능합니다.", {"category": "학적"}),
        (
            "doc2",
            "장학금은 성적 우수자, 저소득 가구에게 지급됩니다.",
            {"category": "장학"},
        ),
        ("doc3", "졸업 요건은 총 140학점 이상 이수해야 합니다.", {"category": "졸업"}),
    ]

    # Create hybrid searcher
    searcher = create_hybrid_searcher()
    searcher.add_documents(sample_docs)

    # Test queries
    test_queries = [
        "휴학",
        "장학금",
        "졸업",
        "학교 쉬고 싶어",  # Intent query
        "제1조",  # Article reference
    ]

    print("\n" + "=" * 80)
    print("Hybrid Search Test Results")
    print("=" * 80)

    for query in test_queries:
        results = searcher.search(query, top_k=3)
        print(f"\nQuery: {query}")
        print(f"Results: {len(results)} documents")

        for i, doc in enumerate(results, 1):
            print(
                f"  {i}. [{doc.metadata.get('category', 'N/A')}] {doc.score:.3f}: {doc.content[:50]}..."
            )

    # Print cache stats
    print("\n" + "=" * 80)
    print("Cache Statistics")
    print("=" * 80)
    stats = searcher.get_cache_stats()
    print(f"BM25 indexed docs: {stats['bm25']['indexed_docs']}")
    if "dense" in stats:
        print(f"Dense cache hits: {stats['dense']['cache_hits']}")
        print(f"Dense cache misses: {stats['dense']['cache_misses']}")
