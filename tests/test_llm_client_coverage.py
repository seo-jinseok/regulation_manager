"""
Focused coverage tests for llm_client.py.

Targets low-coverage module to improve overall coverage from 67% to 85%.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.rag.exceptions import MissingAPIKeyError


class TestMockLLMClient:
    """Tests for MockLLMClient class."""

    def test_mock_llm_client_generate_returns_string(self):
        """
        SPEC: MockLLMClient.generate should return a string response.
        """
        from src.rag.infrastructure.llm_client import MockLLMClient

        client = MockLLMClient()
        result = client.generate("System prompt", "User message")
        assert isinstance(result, str)

    def test_mock_llm_client_generate_includes_system_prompt_length(self):
        """
        SPEC: MockLLMClient.generate should include system prompt length in response.
        """
        from src.rag.infrastructure.llm_client import MockLLMClient

        client = MockLLMClient()
        system_prompt = "Test system prompt"
        result = client.generate(system_prompt, "User message")
        assert str(len(system_prompt)) in result

    def test_mock_llm_client_generate_with_different_temperatures(self):
        """
        SPEC: MockLLMClient.generate should accept temperature parameter.
        """
        from src.rag.infrastructure.llm_client import MockLLMClient

        client = MockLLMClient()
        result1 = client.generate("System", "User", temperature=0.0)
        result2 = client.generate("System", "User", temperature=1.0)
        # Both should return strings (temperature ignored in mock)
        assert isinstance(result1, str)
        assert isinstance(result2, str)

    def test_mock_llm_client_get_embedding_returns_list(self):
        """
        SPEC: MockLLMClient.get_embedding should return a list of floats.
        """
        from src.rag.infrastructure.llm_client import MockLLMClient

        client = MockLLMClient()
        result = client.get_embedding("test text")
        assert isinstance(result, list)
        assert all(isinstance(x, (int, float)) for x in result)

    def test_mock_llm_client_get_embedding_dimension(self):
        """
        SPEC: MockLLMClient.get_embedding should return 384-dimensional vector.
        """
        from src.rag.infrastructure.llm_client import MockLLMClient

        client = MockLLMClient()
        result = client.get_embedding("test text")
        assert len(result) == 384  # text-embedding-3-small dimension

    def test_mock_llm_client_get_embedding_deterministic(self):
        """
        SPEC: MockLLMClient.get_embedding should return same values for same text.
        """
        from src.rag.infrastructure.llm_client import MockLLMClient

        client = MockLLMClient()
        text = "consistent text"
        result1 = client.get_embedding(text)
        result2 = client.get_embedding(text)
        assert result1 == result2

    def test_mock_llm_client_get_embedding_different_for_different_text(self):
        """
        SPEC: MockLLMClient.get_embedding should return different values for different text.
        """
        from src.rag.infrastructure.llm_client import MockLLMClient

        client = MockLLMClient()
        result1 = client.get_embedding("text one")
        result2 = client.get_embedding("text two")
        assert result1 != result2

    def test_mock_llm_client_get_embedding_with_empty_string(self):
        """
        SPEC: MockLLMClient.get_embedding should handle empty string.
        """
        from src.rag.infrastructure.llm_client import MockLLMClient

        client = MockLLMClient()
        result = client.get_embedding("")
        assert len(result) == 384

    def test_mock_llm_client_get_embedding_with_unicode(self):
        """
        SPEC: MockLLMClient.get_embedding should handle unicode text.
        """
        from src.rag.infrastructure.llm_client import MockLLMClient

        client = MockLLMClient()
        result = client.get_embedding("한글 테스트 Ñoño")
        assert len(result) == 384


class TestOpenAIClientInitialization:
    """Tests for OpenAIClient initialization."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-from-env"})
    def test_init_with_api_key_from_env(self):
        """
        SPEC: OpenAIClient should use OPENAI_API_KEY env variable when api_key is None.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
            with patch("src.rag.infrastructure.llm_client.OpenAI") as mock_openai:
                client = OpenAIClient()
                assert client.api_key == "test-key-from-env"
                mock_openai.assert_called_once()

    def test_init_with_explicit_api_key(self):
        """
        SPEC: OpenAIClient should use explicit api_key parameter when provided.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch.dict("os.environ", {}, clear=True):
            with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
                with patch("src.rag.infrastructure.llm_client.OpenAI") as mock_openai:
                    client = OpenAIClient(api_key="explicit-key")
                    assert client.api_key == "explicit-key"
                    mock_openai.assert_called_once_with(api_key="explicit-key")

    @patch.dict("os.environ", {}, clear=True)
    def test_init_without_api_key_raises_error(self):
        """
        SPEC: OpenAIClient should raise MissingAPIKeyError when no API key is available.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
            with pytest.raises(MissingAPIKeyError) as exc_info:
                OpenAIClient()
            assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_init_with_custom_model(self):
        """
        SPEC: OpenAIClient should accept custom model parameter.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
            with patch("src.rag.infrastructure.llm_client.OpenAI"):
                client = OpenAIClient(api_key="test", model="gpt-4")
                assert client.model == "gpt-4"

    def test_init_with_default_model(self):
        """
        SPEC: OpenAIClient should use gpt-4o-mini as default model.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
            with patch("src.rag.infrastructure.llm_client.OpenAI"):
                client = OpenAIClient(api_key="test")
                assert client.model == "gpt-4o-mini"

    def test_init_with_custom_embedding_model(self):
        """
        SPEC: OpenAIClient should accept custom embedding_model parameter.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
            with patch("src.rag.infrastructure.llm_client.OpenAI"):
                client = OpenAIClient(api_key="test", embedding_model="custom-embed")
                assert client.embedding_model == "custom-embed"

    def test_init_with_default_embedding_model(self):
        """
        SPEC: OpenAIClient should use text-embedding-3-small as default embedding model.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
            with patch("src.rag.infrastructure.llm_client.OpenAI"):
                client = OpenAIClient(api_key="test")
                assert client.embedding_model == "text-embedding-3-small"

    @patch.dict("os.environ", {}, clear=True)
    def test_init_without_openai_raises_import_error(self):
        """
        SPEC: OpenAIClient should raise ImportError when openai is not installed.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", False):
            with pytest.raises(ImportError) as exc_info:
                OpenAIClient(api_key="test")
            assert "openai is required" in str(exc_info.value)
            assert "uv add openai" in str(exc_info.value)


class TestOpenAIClientGenerate:
    """Tests for OpenAIClient.generate method."""

    def test_generate_returns_content(self):
        """
        SPEC: OpenAIClient.generate should return message content.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
            with patch("src.rag.infrastructure.llm_client.OpenAI") as mock_openai:
                # Setup mock response
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "Generated response"
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(api_key="test")
                result = client.generate("System", "User")

                assert result == "Generated response"

    def test_generate_with_empty_content(self):
        """
        SPEC: OpenAIClient.generate should return empty string when content is None.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
            with patch("src.rag.infrastructure.llm_client.OpenAI") as mock_openai:
                # Setup mock response with None content
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = None
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(api_key="test")
                result = client.generate("System", "User")

                assert result == ""

    def test_generate_calls_openai_correctly(self):
        """
        SPEC: OpenAIClient.generate should call OpenAI API with correct parameters.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
            with patch("src.rag.infrastructure.llm_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = MagicMock(
                    choices=[MagicMock(message=MagicMock(content="Response"))]
                )
                mock_openai.return_value = mock_client

                client = OpenAIClient(api_key="test", model="gpt-4")
                client.generate("Test system", "Test user", temperature=0.7)

                # Verify API call
                mock_client.chat.completions.create.assert_called_once()
                call_kwargs = mock_client.chat.completions.create.call_args[1]
                assert call_kwargs["model"] == "gpt-4"
                assert call_kwargs["temperature"] == 0.7
                assert len(call_kwargs["messages"]) == 2
                assert call_kwargs["messages"][0] == {
                    "role": "system",
                    "content": "Test system",
                }
                assert call_kwargs["messages"][1] == {
                    "role": "user",
                    "content": "Test user",
                }

    def test_generate_max_tokens(self):
        """
        SPEC: OpenAIClient.generate should use max_tokens=2048.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
            with patch("src.rag.infrastructure.llm_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = MagicMock(
                    choices=[MagicMock(message=MagicMock(content="Response"))]
                )
                mock_openai.return_value = mock_client

                client = OpenAIClient(api_key="test")
                client.generate("System", "User")

                call_kwargs = mock_client.chat.completions.create.call_args[1]
                assert call_kwargs["max_tokens"] == 2048


class TestOpenAIClientGetEmbedding:
    """Tests for OpenAIClient.get_embedding method."""

    def test_get_embedding_returns_vector(self):
        """
        SPEC: OpenAIClient.get_embedding should return embedding vector.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
            with patch("src.rag.infrastructure.llm_client.OpenAI") as mock_openai:
                # Setup mock response
                mock_response = MagicMock()
                mock_response.data = [MagicMock()]
                mock_response.data[0].embedding = [0.1, 0.2, 0.3]
                mock_client = MagicMock()
                mock_client.embeddings.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(api_key="test")
                result = client.get_embedding("test text")

                assert result == [0.1, 0.2, 0.3]

    def test_get_embedding_calls_openai_correctly(self):
        """
        SPEC: OpenAIClient.get_embedding should call OpenAI embeddings API correctly.
        """
        from src.rag.infrastructure.llm_client import OpenAIClient

        with patch("src.rag.infrastructure.llm_client.OPENAI_AVAILABLE", True):
            with patch("src.rag.infrastructure.llm_client.OpenAI") as mock_openai:
                mock_client = MagicMock()
                mock_client.embeddings.create.return_value = MagicMock(
                    data=[MagicMock(embedding=[])]
                )
                mock_openai.return_value = mock_client

                client = OpenAIClient(api_key="test", embedding_model="custom-model")
                client.get_embedding("test text")

                # Verify API call
                mock_client.embeddings.create.assert_called_once()
                call_kwargs = mock_client.embeddings.create.call_args[1]
                assert call_kwargs["model"] == "custom-model"
                assert call_kwargs["input"] == "test text"
