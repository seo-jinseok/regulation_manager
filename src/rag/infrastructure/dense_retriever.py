"""
Dense Retriever for Korean-optimized semantic search.

Implements vector-based retrieval using Korean-optimized embedding models
with caching and batch processing for performance optimization.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Global model cache
_embedding_models = {}
_embedding_dims = {}


@dataclass
class DenseRetrieverConfig:
    """Configuration for Dense Retriever."""

    model_name: str = "jhgan/ko-sbert-sts"  # Default: Korean SBERT
    batch_size: int = 32
    cache_embeddings: bool = True
    normalize_embeddings: bool = True
    max_cache_size: int = 10000


class DenseRetriever:
    """
    Dense retriever using Korean-optimized embedding models.

    Supported models:
    - jhgan/ko-sbert-multinli: Korean SBERT (768 dims)
    - BAAI/bge-m3: Multilingual BGE-M3 (1024 dims)
    - jhgan/ko-sbert-sts: Korean STS (768 dims)

    Features:
    - Automatic model download from HuggingFace
    - Embedding caching for performance
    - Batch processing support
    - Cosine similarity search
    """

    # Korean embedding model configurations
    MODEL_CONFIGS = {
        "jhgan/ko-sbert-multinli": {
            "dims": 768,
            "max_length": 512,
            "description": "Korean SBERT trained on NLI data",
            "language": "ko",
            "speed": "medium",
            "accuracy": "high",
        },
        "BAAI/bge-m3": {
            "dims": 1024,
            "max_length": 8192,
            "description": "Multilingual BGE-M3 with dense, sparse, and colbert",
            "language": "multilingual",
            "speed": "slow",
            "accuracy": "very_high",
        },
        "jhgan/ko-sbert-sts": {
            "dims": 768,
            "max_length": 512,
            "description": "Korean SBERT trained on STS data",
            "language": "ko",
            "speed": "fast",
            "accuracy": "medium",
        },
    }

    def __init__(
        self,
        model_name: str = "jhgan/ko-sbert-multinli",
        config: Optional[DenseRetrieverConfig] = None,
    ):
        """
        Initialize Dense Retriever.

        Args:
            model_name: HuggingFace model name for Korean embeddings.
            config: Optional configuration for retriever behavior.
        """
        self.model_name = model_name
        self.config = config or DenseRetrieverConfig()

        # Validate model name
        if model_name not in self.MODEL_CONFIGS:
            logger.warning(
                f"Unknown model '{model_name}'. "
                f"Known models: {list(self.MODEL_CONFIGS.keys())}"
            )

        # Initialize model and cache
        self._model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Vector storage (in-memory for now, can be extended to FAISS/ChromaDB)
        self._doc_embeddings: Dict[str, np.ndarray] = {}
        self._doc_texts: Dict[str, str] = {}
        self._doc_metadata: Dict[str, Dict] = {}

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        global _embedding_models, _embedding_dims

        # Check cache first
        if self.model_name in _embedding_models:
            self._model = _embedding_models[self.model_name]
            logger.debug(f"Using cached embedding model: {self.model_name}")
            return

        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            from sentence_transformers import SentenceTransformer

            # Download model from HuggingFace
            self._model = SentenceTransformer(self.model_name)
            _embedding_models[self.model_name] = self._model
            _embedding_dims[self.model_name] = (
                self._model.get_sentence_embedding_dimension()
            )

            logger.info(
                f"Embedding model loaded: {self.model_name} "
                f"(dims={_embedding_dims[self.model_name]})"
            )
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is required for dense retrieval. "
                "Install with: uv add sentence-transformers"
            ) from err
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension of the current model."""
        global _embedding_dims
        if self.model_name in _embedding_dims:
            return _embedding_dims[self.model_name]

        # Load model to get dimension
        self._load_model()
        return _embedding_dims.get(self.model_name, 768)  # Default fallback

    def _get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Get embedding for a single text with caching.

        Args:
            text: Input text to embed.
            use_cache: Whether to use embedding cache.

        Returns:
            Embedding vector as numpy array.
        """
        # Check cache
        if use_cache and text in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[text]

        self._cache_misses += 1

        # Generate embedding
        self._load_model()
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.config.normalize_embeddings,
        )

        # Cache if enabled
        if use_cache and self.config.cache_embeddings:
            self._cache_embedding(text, embedding)

        return embedding

    def _cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """
        Cache embedding with size limit.

        Args:
            text: Text key for cache.
            embedding: Embedding vector to cache.
        """
        # Enforce cache size limit
        if len(self._embedding_cache) >= self.config.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]

        self._embedding_cache[text] = embedding

    def add_documents(self, documents: List[Tuple[str, str, Dict]]) -> None:
        """
        Add documents to the vector index.

        Args:
            documents: List of (doc_id, content, metadata) tuples.
        """
        if not documents:
            return

        logger.info(f"Indexing {len(documents)} documents with dense retriever")

        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [content for _, content, _ in batch]

            # Generate embeddings in batch
            self._load_model()
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize_embeddings,
                batch_size=batch_size,
            )

            # Store embeddings
            for (doc_id, content, metadata), embedding in zip(batch, embeddings):
                self._doc_embeddings[doc_id] = embedding
                self._doc_texts[doc_id] = content
                self._doc_metadata[doc_id] = metadata

        logger.info(f"Indexed {len(self._doc_embeddings)} documents")

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[str, float, str, Dict]]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query: Search query text.
            top_k: Maximum number of results.
            score_threshold: Optional minimum similarity score (0-1).

        Returns:
            List of (doc_id, score, content, metadata) tuples sorted by score.
        """
        if not self._doc_embeddings:
            logger.warning("No documents indexed for dense search")
            return []

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Calculate cosine similarity with all documents
        doc_ids = list(self._doc_embeddings.keys())
        doc_embeddings = np.array([self._doc_embeddings[doc_id] for doc_id in doc_ids])

        # Cosine similarity (dot product since embeddings are normalized)
        similarities = np.dot(doc_embeddings, query_embedding)

        # Apply threshold if specified
        if score_threshold is not None:
            mask = similarities >= score_threshold
            doc_ids = [doc_ids[i] for i in range(len(doc_ids)) if mask[i]]
            similarities = similarities[mask]

        # Sort by similarity score
        sorted_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in sorted_indices:
            doc_id = doc_ids[idx]
            score = float(similarities[idx])
            content = self._doc_texts[doc_id]
            metadata = self._doc_metadata[doc_id]
            results.append((doc_id, score, content, metadata))

        return results

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[List[Tuple[str, float, str, Dict]]]:
        """
        Search multiple queries in batch for performance.

        Args:
            queries: List of search query texts.
            top_k: Maximum number of results per query.
            score_threshold: Optional minimum similarity score.

        Returns:
            List of result lists, one per query.
        """
        if not queries:
            return []

        self._load_model()

        # Generate query embeddings in batch
        query_embeddings = self._model.encode(
            queries,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.config.normalize_embeddings,
            batch_size=self.config.batch_size,
        )

        # Search for each query
        results = []
        for query_embedding in query_embeddings:
            doc_ids = list(self._doc_embeddings.keys())
            doc_embeddings = np.array(
                [self._doc_embeddings[doc_id] for doc_id in doc_ids]
            )

            similarities = np.dot(doc_embeddings, query_embedding)

            if score_threshold is not None:
                mask = similarities >= score_threshold
                doc_ids = [doc_ids[i] for i in range(len(doc_ids)) if mask[i]]
                similarities = similarities[mask]

            sorted_indices = np.argsort(similarities)[::-1][:top_k]

            query_results = []
            for idx in sorted_indices:
                doc_id = doc_ids[idx]
                score = float(similarities[idx])
                content = self._doc_texts[doc_id]
                metadata = self._doc_metadata[doc_id]
                query_results.append((doc_id, score, content, metadata))

            results.append(query_results)

        return results

    def save_index(self, path: str) -> None:
        """
        Save vector index to disk.

        Args:
            path: File path to save the index.
        """
        import pickle

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        index_data = {
            "model_name": self.model_name,
            "doc_embeddings": {
                doc_id: emb.astype(np.float16)
                for doc_id, emb in self._doc_embeddings.items()
            },
            "doc_texts": self._doc_texts,
            "doc_metadata": self._doc_metadata,
            "embedding_dim": self.embedding_dim,
        }

        with open(path, "wb") as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"Saved dense index with {len(self._doc_embeddings)} documents to {path}"
        )

    def load_index(self, path: str) -> bool:
        """
        Load vector index from disk.

        Args:
            path: File path to load the index from.

        Returns:
            True if loaded successfully, False otherwise.
        """
        import pickle

        if not Path(path).exists():
            return False

        try:
            with open(path, "rb") as f:
                index_data = pickle.load(f)

            # Verify model compatibility
            if index_data.get("model_name") != self.model_name:
                logger.warning(
                    f"Model mismatch: index uses '{index_data['model_name']}' "
                    f"but retriever uses '{self.model_name}'"
                )

            self._doc_embeddings = {
                doc_id: emb.astype(np.float32)
                for doc_id, emb in index_data["doc_embeddings"].items()
            }
            self._doc_texts = index_data["doc_texts"]
            self._doc_metadata = index_data["doc_metadata"]

            logger.info(
                f"Loaded dense index with {len(self._doc_embeddings)} documents from {path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load dense index from {path}: {e}")
            return False

    def clear(self) -> None:
        """Clear all indexed documents and cache."""
        self._doc_embeddings.clear()
        self._doc_texts.clear()
        self._doc_metadata.clear()
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Dense retriever cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hits, misses, and size.
        """
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._embedding_cache),
            "indexed_docs": len(self._doc_embeddings),
        }

    @classmethod
    def get_model_info(cls, model_name: str) -> Dict:
        """
        Get information about a supported model.

        Args:
            model_name: Model name to query.

        Returns:
            Dictionary with model configuration.
        """
        return cls.MODEL_CONFIGS.get(model_name, {})

    @classmethod
    def list_models(cls) -> List[str]:
        """Get list of supported Korean embedding models."""
        return list(cls.MODEL_CONFIGS.keys())


def create_dense_retriever(
    model_name: str = "jhgan/ko-sbert-sts",
    config: Optional[DenseRetrieverConfig] = None,
) -> DenseRetriever:
    """
    Factory function to create a Dense Retriever.

    Args:
        model_name: HuggingFace model name for Korean embeddings.
        config: Optional configuration for retriever behavior.

    Returns:
        Configured DenseRetriever instance.

    Example:
        >>> from src.rag.infrastructure.dense_retriever import create_dense_retriever
        >>> retriever = create_dense_retriever("jhgan/ko-sbert-sts")
        >>> retriever.add_documents([("doc1", "휴학 규정", {})])
        >>> results = retriever.search("휴학 절차", top_k=5)
    """
    return DenseRetriever(model_name=model_name, config=config)


# Convenience function for backward compatibility
def get_embedding_function(model_name: str = "jhgan/ko-sbert-sts"):
    """
    Get sentence-transformers embedding function.

    DEPRECATED: Use create_dense_retriever() instead.
    This function is kept for backward compatibility.
    """
    import warnings

    warnings.warn(
        "get_embedding_function is deprecated. Use create_dense_retriever instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    retriever = create_dense_retriever(model_name)
    retriever._load_model()

    def embedding_function(texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = retriever._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    return embedding_function
