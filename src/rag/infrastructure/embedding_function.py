"""
Embedding function factory for ChromaDB integration.

Provides sentence-transformers based embedding functions
for Korean-specific semantic search.
"""

import logging
from typing import List, Optional

import numpy as np

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
    except ImportError as err:
        raise ImportError(
            "sentence-transformers is required for embedding functions. "
            "Install with: uv add sentence-transformers"
        ) from err

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
        ChromaDB-compatible EmbeddingFunctionWrapper instance using the configured model.

    Raises:
        ImportError: If RAG configuration module is not available.
        RuntimeError: If configuration loading fails.
    """
    try:
        from ..config import get_config
    except ImportError as e:
        raise ImportError(
            "RAG configuration unavailable. Ensure src.rag.config is importable. "
            f"Original error: {e}"
        ) from e

    try:
        config = get_config()
        model_name = config.get_embedding_model_name()
        logger.info(f"Using configured embedding model: {model_name}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model configuration: {e}. "
            "Check that RAG_EMBEDDING_MODEL environment variable is set correctly, "
            "or ensure the default model 'jhgan/ko-sbert-sts' is available."
        ) from e

    # Return EmbeddingFunctionWrapper instance instead of plain function
    return EmbeddingFunctionWrapper(model_name)


class EmbeddingFunctionWrapper:
    """
    Wrapper class for sentence-transformers embedding functions.

    Provides a clean interface compatible with ChromaDB's expectations.
    Implements the ChromaDB 0.4.16+ EmbeddingFunction interface.
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
        self._model = None  # Will load the actual SentenceTransformer model

    def name(self) -> str:
        """Return the model name for ChromaDB identification."""
        return self._model_name or "default"

    def is_legacy(self) -> bool:
        """ChromaDB compatibility: mark as non-legacy embedding function."""
        return False

    def _get_model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for embedding functions. "
                    "Install with: uv add sentence-transformers"
                ) from e

            if self._model_name is None:
                # Try to get from config
                try:
                    from ..config import get_config

                    self._model_name = get_config().get_embedding_model_name()
                except Exception as e:
                    raise RuntimeError(
                        "No embedding model specified and configuration unavailable. "
                        "Either provide model_name parameter or ensure RAG config is accessible. "
                        f"Original error: {e}"
                    ) from e

            try:
                self._model = SentenceTransformer(self._model_name)
                logger.info(f"Loaded SentenceTransformer model: {self._model_name}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        return self._model

    def __call__(self, input: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for texts (ChromaDB 0.4.16+ interface).

        Args:
            input: List of text strings to embed.

        Returns:
            List of embedding vectors as numpy arrays.
        """
        model = self._get_model()
        embeddings = model.encode(
            input,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        # Handle both single and multiple inputs
        if embeddings.ndim == 1:
            # Single input, convert to 2D array
            embeddings = embeddings.reshape(1, -1)
        # Convert to list of 1D numpy arrays
        return [embeddings[i].astype(np.float32) for i in range(len(embeddings))]

    def embed_query(self, input) -> List[np.ndarray]:
        """
        Embed a single query string (ChromaDB query interface).

        Args:
            input: Query text to embed (can be string or list).

        Returns:
            List of embedding vectors as numpy arrays.
        """
        # For queries, just call __call__ with the input
        # ChromaDB passes either a string or list of strings
        if isinstance(input, str):
            return self([input])
        elif isinstance(input, list):
            return self(input)
        else:
            return self([str(input)])

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        """
        Embed multiple documents (ChromaDB document interface).

        Args:
            input: List of document texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self(input)


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
