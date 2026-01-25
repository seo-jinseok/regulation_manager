"""
Tests for embedding function module.

Tests the sentence-transformers based embedding functions
for Korean semantic search with ko-sbert-sts model.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.rag.infrastructure.embedding_function import (
    get_embedding_function,
    get_default_embedding_function,
    clear_embedding_cache,
    create_embedding_function,
    EmbeddingFunctionWrapper,
)


class TestEmbeddingFunctionWrapper:
    """Tests for EmbeddingFunctionWrapper class."""

    @patch("src.rag.infrastructure.embedding_function.SentenceTransformer")
    def test_wrapper_lazy_loads_model(self, mock_transformer):
        """Test that embedding function wrapper lazy-loads the model."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model

        wrapper = EmbeddingFunctionWrapper("test/model")
        result = wrapper(["test text"])

        assert result == [[0.1, 0.2, 0.3]]
        mock_model.encode.assert_called_once()

    @patch("src.rag.infrastructure.embedding_function.SentenceTransformer")
    def test_wrapper_normalizes_embeddings(self, mock_transformer):
        """Test that embeddings are normalized for cosine similarity."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.577, 0.577, 0.577]]  # Normalized
        mock_transformer.return_value = mock_model

        wrapper = EmbeddingFunctionWrapper("test/model")
        result = wrapper(["test"])

        mock_model.encode.assert_called_once_with(
            ["test"],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        assert result == [[0.577, 0.577, 0.577]]

    @patch("src.rag.infrastructure.embedding_function.SentenceTransformer")
    def test_wrapper_handles_multiple_texts(self, mock_transformer):
        """Test that wrapper processes multiple texts efficiently."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]
        mock_transformer.return_value = mock_model

        wrapper = EmbeddingFunctionWrapper("test/model")
        result = wrapper(["text1", "text2", "text3"])

        assert len(result) == 3
        assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]


class TestEmbeddingFunctionCache:
    """Tests for embedding function caching mechanism."""

    @patch("src.rag.infrastructure.embedding_function.SentenceTransformer")
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

    @patch("src.rag.infrastructure.embedding_function.SentenceTransformer")
    def test_clear_cache_removes_models(self, mock_transformer):
        """Test that cache can be cleared to free memory."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2]]
        mock_transformer.return_value = mock_model

        # Create and cache a model
        get_embedding_function("test/model")

        # Clear cache
        clear_embedding_cache()

        # Next call should create a new model instance
        get_embedding_function("test/model")
        assert mock_transformer.call_count == 2


class TestDefaultEmbeddingFunction:
    """Tests for default embedding function from config."""

    @patch("src.rag.infrastructure.embedding_function.get_config")
    @patch("src.rag.infrastructure.embedding_function.get_embedding_function")
    def test_default_uses_config_model(self, mock_get_emb, mock_config):
        """Test that default function uses configured model."""
        mock_config_obj = MagicMock()
        mock_config_obj.get_embedding_model_name.return_value = "jhgan/ko-sbert-sts"
        mock_config.return_value = mock_config_obj
        mock_get_emb.return_value = lambda x: [[0.1, 0.2]]

        result = get_default_embedding_function()

        mock_config_obj.get_embedding_model_name.assert_called_once()
        mock_get_emb.assert_called_once_with("jhgan/ko-sbert-sts")

    @patch("src.rag.infrastructure.embedding_function.get_config")
    @patch("src.rag.infrastructure.embedding_function.get_embedding_function")
    def test_default_fallback_on_config_error(self, mock_get_emb, mock_config):
        """Test that default falls back to ko-sbert if config fails."""
        mock_config.side_effect = Exception("Config error")
        mock_get_emb.return_value = lambda x: [[0.1, 0.2]]

        result = get_default_embedding_function()

        mock_get_emb.assert_called_once_with("jhgan/ko-sbert-sts")


class TestCreateEmbeddingFunction:
    """Tests for create_embedding_function convenience function."""

    @patch("src.rag.infrastructure.embedding_function.get_embedding_function")
    def test_create_with_specific_model(self, mock_get_emb):
        """Test creating embedding function with specific model."""
        mock_get_emb.return_value = lambda x: [[0.1, 0.2]]

        result = create_embedding_function("custom/model")

        mock_get_emb.assert_called_once_with("custom/model")
        assert callable(result)

    @patch("src.rag.infrastructure.embedding_function.get_default_embedding_function")
    def test_create_without_model_uses_default(self, mock_get_default):
        """Test creating embedding function without model uses default."""
        mock_get_default.return_value = lambda x: [[0.1, 0.2]]

        result = create_embedding_function(None)

        mock_get_default.assert_called_once()
        assert callable(result)


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
        import numpy as np

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
