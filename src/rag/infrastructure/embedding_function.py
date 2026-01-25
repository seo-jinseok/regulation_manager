"""
Embedding function factory for ChromaDB integration.

Provides sentence-transformers based embedding functions
for Korean-specific semantic search.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Singleton pattern for embedding models
_embedding_models = {}


def get_embedding_function(model_name: str = "jhgan/ko-sbert-sts"):
    """
    Get or create a sentence-transformers embedding function for ChromaDB.

    Args:
        model_name: HuggingFace model name for sentence-transformers.
                   Default is "jhgan/ko-sbert-sts" for Korean semantic search.

    Returns:
        ChromaDB-compatible embedding function.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    global _embedding_models

    # Check if sentence-transformers is available
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embedding functions. "
            "Install with: uv add sentence-transformers"
        )

    # Return cached model if available
    if model_name in _embedding_models:
        logger.debug(f"Using cached embedding model: {model_name}")
        return _embedding_models[model_name]

    logger.info(f"Loading embedding model: {model_name}")
    try:
        # Load the model
        model = SentenceTransformer(model_name)

        # Create ChromaDB-compatible embedding function
        def embedding_function(texts: List[str]) -> List[List[float]]:
            """Generate embeddings for a list of texts.

            Args:
                texts: List of text strings to embed.

            Returns:
                List of embedding vectors.
            """
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,  # Normalize for cosine similarity
            )
            return embeddings.tolist()

        # Cache the embedding function
        _embedding_models[model_name] = embedding_function
        logger.info(f"Embedding model loaded successfully: {model_name}")

        return embedding_function

    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise


def clear_embedding_cache():
    """Clear cached embedding models to free memory."""
    global _embedding_models
    _embedding_models.clear()
    logger.debug("Embedding model cache cleared")


def get_default_embedding_function():
    """
    Get the default embedding function based on RAG configuration.

    Returns:
        ChromaDB-compatible embedding function using the configured model.

    Raises:
        ImportError: If sentence-transformers is not installed.
    """
    try:
        from ..config import get_config
        config = get_config()
        model_name = config.get_embedding_model_name()
        logger.info(f"Using configured embedding model: {model_name}")
    except Exception:
        # Fallback to ko-sbert if config is unavailable
        model_name = "jhgan/ko-sbert-sts"
        logger.warning(f"Config unavailable, using fallback model: {model_name}")

    return get_embedding_function(model_name)


class EmbeddingFunctionWrapper:
    """
    Wrapper class for sentence-transformers embedding functions.

    Provides a clean interface compatible with ChromaDB's expectations.
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding function wrapper.

        Args:
            model_name: HuggingFace model name. If None, uses config default.
        """
        if model_name is None:
            self._model_name = None  # Will be resolved on first call
        else:
            self._model_name = model_name
        self._embedding_func = None

    def _get_function(self):
        """Lazy-load the embedding function."""
        if self._embedding_func is None:
            if self._model_name is None:
                self._embedding_func = get_default_embedding_function()
            else:
                self._embedding_func = get_embedding_function(self._model_name)
        return self._embedding_func

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors.
        """
        return self._get_function()(texts)


# Convenience function for backward compatibility
def create_embedding_function(model_name: Optional[str] = None):
    """
    Create an embedding function for ChromaDB.

    Args:
        model_name: HuggingFace model name. If None, uses config default.

    Returns:
        Callable embedding function compatible with ChromaDB.

    Example:
        >>> from src.rag.infrastructure.embedding_function import create_embedding_function
        >>> ef = create_embedding_function()
        >>> embeddings = ef(["Hello world", "안녕하세요"])
    """
    wrapper = EmbeddingFunctionWrapper(model_name)
    return wrapper
