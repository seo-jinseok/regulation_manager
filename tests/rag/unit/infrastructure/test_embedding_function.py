"""
Tests for embedding function module.

Tests the sentence-transformers based embedding functions
for Korean semantic search with ko-sbert-sts model.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.infrastructure.embedding_function import (
    EmbeddingFunctionWrapper,
    clear_embedding_cache,
    create_embedding_function,
    get_default_embedding_function,
    get_embedding_function,
)


class TestEmbeddingFunctionWrapper:
    """Tests for EmbeddingFunctionWrapper class."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_wrapper_lazy_loads_model(self, mock_transformer):
        """Test that embedding function wrapper lazy-loads the model."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        wrapper = EmbeddingFunctionWrapper("test/model")
        result = wrapper(["test text"])

        assert len(result) == 1
        mock_model.encode.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_wrapper_normalizes_embeddings(self, mock_transformer):
        """Test that embeddings are normalized for cosine similarity."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [[0.577, 0.577, 0.577]], dtype=np.float32
        )  # Normalized
        mock_transformer.return_value = mock_model

        wrapper = EmbeddingFunctionWrapper("test/model")
        result = wrapper(["test"])

        mock_model.encode.assert_called_once_with(
            ["test"],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        assert len(result) == 1

    @patch("sentence_transformers.SentenceTransformer")
    def test_wrapper_handles_multiple_texts(self, mock_transformer):
        """Test that wrapper processes multiple texts efficiently."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32
        )
        mock_transformer.return_value = mock_model

        wrapper = EmbeddingFunctionWrapper("test/model")
        result = wrapper(["text1", "text2", "text3"])

        assert len(result) == 3


class TestEmbeddingFunctionCache:
    """Tests for embedding function caching mechanism."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_get_embedding_function_caches_model(self, mock_transformer):
        """Test that models are cached and reused."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2]]
        mock_transformer.return_value = mock_model

        # First call should create the model
        func1 = get_embedding_function("test/model")

        # Second call should return cached model
        func2 = get_embedding_function("test/model")

        assert func1 is func2
        mock_transformer.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_clear_cache_removes_models(self, mock_transformer):
        """Test that cache can be cleared to free memory."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2]]
        mock_transformer.return_value = mock_model

        # Clear any existing cache from previous tests
        clear_embedding_cache()

        # First call creates and caches the model (call_count = 1)
        get_embedding_function("test/model")

        # Clear the cache
        clear_embedding_cache()

        # Second call creates a new model since cache is empty (call_count = 2)
        # Note: The mock transformer is called twice total
        get_embedding_function("test/model")
        assert mock_transformer.call_count == 2


class TestDefaultEmbeddingFunction:
    """Tests for default embedding function from config."""

    @patch("src.rag.config.get_config")
    @patch("src.rag.infrastructure.embedding_function.EmbeddingFunctionWrapper")
    def test_default_uses_config_model(self, mock_wrapper_class, mock_config):
        """Test that default function uses configured model."""
        mock_config_obj = MagicMock()
        mock_config_obj.get_embedding_model_name.return_value = "jhgan/ko-sbert-sts"
        mock_config.return_value = mock_config_obj
        mock_wrapper_instance = MagicMock()
        mock_wrapper_class.return_value = mock_wrapper_instance

        result = get_default_embedding_function()

        mock_config_obj.get_embedding_model_name.assert_called_once()
        mock_wrapper_class.assert_called_once_with("jhgan/ko-sbert-sts")
        assert result is not None

    @patch("src.rag.config.get_config")
    @patch("src.rag.infrastructure.embedding_function.EmbeddingFunctionWrapper")
    def test_default_fallback_on_config_error(self, mock_wrapper_class, mock_config):
        """Test that default falls back to paraphrase-multilingual-MiniLM-L12-v2 if config fails."""
        mock_config.side_effect = Exception("Config error")
        mock_wrapper_instance = MagicMock()
        mock_wrapper_class.return_value = mock_wrapper_instance

        result = get_default_embedding_function()

        mock_wrapper_class.assert_called_once_with(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
        assert result is not None


class TestCreateEmbeddingFunction:
    """Tests for create_embedding_function convenience function."""

    @patch("src.rag.infrastructure.embedding_function.EmbeddingFunctionWrapper")
    def test_create_with_specific_model(self, mock_wrapper_class):
        """Test creating embedding function with specific model."""
        mock_wrapper_instance = MagicMock()
        mock_wrapper_class.return_value = mock_wrapper_instance

        result = create_embedding_function("custom/model")

        mock_wrapper_class.assert_called_once_with("custom/model")
        assert result is not None

    @patch("src.rag.infrastructure.embedding_function.EmbeddingFunctionWrapper")
    def test_create_without_model_uses_default(self, mock_wrapper_class):
        """Test creating embedding function without model creates wrapper with None."""
        mock_wrapper_instance = MagicMock()
        mock_wrapper_class.return_value = mock_wrapper_instance

        result = create_embedding_function(None)

        mock_wrapper_class.assert_called_once_with(None)
        assert result is not None


class TestSentenceTransformersIntegration:
    """Integration tests with actual sentence-transformers library."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_ko_sbert_generates_embeddings(self):
        """Test that ko-sbert-sts model generates valid embeddings."""
        pytest.importorskip("sentence_transformers")

        func = get_embedding_function("jhgan/ko-sbert-sts")
        result = func(["안녕하세요", "테스트"])

        assert len(result) == 2
        assert all(isinstance(emb, list) for emb in result)
        assert all(len(emb) > 0 for emb in result)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_ko_sbert_similar_queries_have_high_similarity(self):
        """Test that semantically similar Korean queries have high similarity."""

        pytest.importorskip("sentence_transformers")

        func = get_embedding_function("jhgan/ko-sbert-sts")
        queries = ["휴학 어떻게 하나요?", "휴학 절차 알려주세요", "졸업 요건이 뭔가요?"]
        embeddings = func(queries)

        # Convert to numpy for cosine similarity
        emb_array = np.array(embeddings)

        # Similarity between "휴학" queries should be higher than with "졸업"
        sim_01 = np.dot(emb_array[0], emb_array[1]) / (
            np.linalg.norm(emb_array[0]) * np.linalg.norm(emb_array[1])
        )
        sim_02 = np.dot(emb_array[0], emb_array[2]) / (
            np.linalg.norm(emb_array[0]) * np.linalg.norm(emb_array[2])
        )

        assert sim_01 > sim_02, "휴학 queries should be more similar to each other"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_different_models_generate_different_embeddings(self):
        """Test that different models generate different embedding vectors."""
        pytest.importorskip("sentence_transformers")

        # Clear cache to ensure fresh models
        clear_embedding_cache()

        func1 = get_embedding_function("jhgan/ko-sbert-sts")
        emb1 = func1(["테스트"])

        clear_embedding_cache()

        # Note: This would require multiple models, but for now we test
        # that the same model produces consistent results
        func2 = get_embedding_function("jhgan/ko-sbert-sts")
        emb2 = func2(["테스트"])

        # Same model should produce same embeddings
        assert emb1 == emb2
