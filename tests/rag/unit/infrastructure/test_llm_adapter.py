"""
Comprehensive tests for LLMClientAdapter.

Covers error handling, edge cases, and missing lines.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.rag.infrastructure.llm_adapter import LLMClientAdapter


class TestLLMClientAdapterInit:
    """Test adapter initialization with various configurations."""

    @patch.dict(
        os.environ, {"LLM_PROVIDER": "ollama", "LLM_MODEL": "gemma2"}, clear=True
    )
    @patch("src.llm_client.LLMClient")
    def test_init_from_env_vars(self, mock_llm_client):
        adapter = LLMClientAdapter()
        assert adapter.provider == "ollama"
        assert adapter.model == "gemma2"
        mock_llm_client.assert_called_once()

    @patch("src.llm_client.LLMClient")
    def test_init_defaults_to_ollama(self, mock_llm_client, monkeypatch):
        # Ensure clean environment - remove LLM env vars that may be loaded from .env
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        adapter = LLMClientAdapter()
        assert adapter.provider == "ollama"
        mock_llm_client.assert_called_once()

    @patch("src.llm_client.LLMClient")
    def test_init_with_explicit_params(self, mock_llm_client):
        adapter = LLMClientAdapter(
            provider="openai",
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        assert adapter.provider == "openai"
        assert adapter.model == "gpt-4"
        assert adapter.base_url == "https://api.openai.com/v1"

    @patch("src.llm_client.LLMClient")
    def test_init_with_partial_params(self, mock_llm_client):
        adapter = LLMClientAdapter(provider="gemini")
        assert adapter.provider == "gemini"


class TestLLMClientAdapterGenerate:
    """Test the generate method with various scenarios."""

    @patch("src.llm_client.LLMClient")
    def test_generate_with_valid_inputs(self, mock_llm_client):
        mock_instance = MagicMock()
        mock_instance.complete.return_value = "Test response"
        mock_llm_client.return_value = mock_instance

        adapter = LLMClientAdapter()
        result = adapter.generate(
            system_prompt="You are a helpful assistant.",
            user_message="Hello!",
            temperature=0.5,
        )

        assert result == "Test response"
        mock_instance.complete.assert_called_once()
        call_args = mock_instance.complete.call_args[0][0]
        assert "<system>" in call_args
        assert "You are a helpful assistant." in call_args
        assert "Hello!" in call_args

    @patch("src.llm_client.LLMClient")
    def test_generate_with_empty_system_prompt(self, mock_llm_client):
        mock_instance = MagicMock()
        mock_instance.complete.return_value = "Response"
        mock_llm_client.return_value = mock_instance

        adapter = LLMClientAdapter()
        result = adapter.generate(system_prompt="", user_message="Hello")

        assert result == "Response"
        call_args = mock_instance.complete.call_args[0][0]
        assert "<system>" in call_args
        assert "<user>" in call_args

    @patch("src.llm_client.LLMClient")
    def test_generate_with_special_characters(self, mock_llm_client):
        mock_instance = MagicMock()
        mock_instance.complete.return_value = "Response"
        mock_llm_client.return_value = mock_instance

        adapter = LLMClientAdapter()
        result = adapter.generate(
            system_prompt='System <tag> & "quotes"',
            user_message="User's message\nwith newlines",
        )

        assert result == "Response"

    @patch("src.llm_client.LLMClient")
    def test_generate_with_zero_temperature(self, mock_llm_client):
        mock_instance = MagicMock()
        mock_instance.complete.return_value = "Deterministic"
        mock_llm_client.return_value = mock_instance

        adapter = LLMClientAdapter()
        result = adapter.generate(
            system_prompt="System", user_message="Message", temperature=0.0
        )

        assert result == "Deterministic"

    @patch("src.llm_client.LLMClient")
    def test_generate_with_high_temperature(self, mock_llm_client):
        mock_instance = MagicMock()
        mock_instance.complete.return_value = "Creative"
        mock_llm_client.return_value = mock_instance

        adapter = LLMClientAdapter()
        result = adapter.generate(
            system_prompt="System", user_message="Message", temperature=1.5
        )

        assert result == "Creative"


class TestLLMClientAdapterStreamGenerate:
    """Test the stream_generate method."""

    @patch("src.llm_client.LLMClient")
    def test_stream_generate_yields_tokens(self, mock_llm_client):
        mock_instance = MagicMock()
        mock_instance.stream_complete.return_value = iter(["Hello", " world", "!"])
        mock_llm_client.return_value = mock_instance

        adapter = LLMClientAdapter()
        tokens = list(
            adapter.stream_generate(system_prompt="System", user_message="Message")
        )

        assert tokens == ["Hello", " world", "!"]
        mock_instance.stream_complete.assert_called_once()
        call_args = mock_instance.stream_complete.call_args[0][0]
        assert "<system>" in call_args
        assert "Message" in call_args

    @patch("src.llm_client.LLMClient")
    def test_stream_generate_with_empty_response(self, mock_llm_client):
        mock_instance = MagicMock()
        mock_instance.stream_complete.return_value = iter([])
        mock_llm_client.return_value = mock_instance

        adapter = LLMClientAdapter()
        tokens = list(
            adapter.stream_generate(system_prompt="System", user_message="Message")
        )

        assert tokens == []

    @patch("src.llm_client.LLMClient")
    def test_stream_generate_with_long_text(self, mock_llm_client):
        mock_instance = MagicMock()
        tokens = ["token"] * 100
        mock_instance.stream_complete.return_value = iter(tokens)
        mock_llm_client.return_value = mock_instance

        adapter = LLMClientAdapter()
        result = list(
            adapter.stream_generate(system_prompt="System", user_message="Long message")
        )

        assert len(result) == 100


class TestLLMClientAdapterGetEmbedding:
    """Test the get_embedding method."""

    @patch("src.llm_client.LLMClient")
    def test_get_embedding_raises_not_implemented(self, mock_llm_client):
        adapter = LLMClientAdapter()

        with pytest.raises(NotImplementedError) as exc_info:
            adapter.get_embedding("test text")

        assert "Embedding is handled by ChromaDB" in str(exc_info.value)

    @patch("src.llm_client.LLMClient")
    def test_get_embedding_with_empty_string(self, mock_llm_client):
        adapter = LLMClientAdapter()

        with pytest.raises(NotImplementedError):
            adapter.get_embedding("")


class TestLLMClientAdapterEdgeCases:
    """Test edge cases and error handling."""

    @patch("src.llm_client.LLMClient")
    def test_generate_with_unicode_content(self, mock_llm_client):
        mock_instance = MagicMock()
        mock_instance.complete.return_value = "한글 응답"
        mock_llm_client.return_value = mock_instance

        adapter = LLMClientAdapter()
        result = adapter.generate(system_prompt="안녕하세요", user_message="반갑습니다")

        assert result == "한글 응답"

    @patch("src.llm_client.LLMClient")
    def test_generate_with_multiline_content(self, mock_llm_client):
        mock_instance = MagicMock()
        mock_instance.complete.return_value = "Response"
        mock_llm_client.return_value = mock_instance

        adapter = LLMClientAdapter()
        result = adapter.generate(
            system_prompt="Line1\nLine2\nLine3", user_message="Msg1\nMsg2"
        )

        assert result == "Response"
        call_args = mock_instance.complete.call_args[0][0]
        assert "Line1" in call_args
        assert "Msg1" in call_args

    @patch("src.llm_client.LLMClient")
    def test_stream_generate_with_special_chars(self, mock_llm_client):
        mock_instance = MagicMock()
        mock_instance.stream_complete.return_value = iter(["<tag>", "&quot;", ""])
        mock_llm_client.return_value = mock_instance

        adapter = LLMClientAdapter()
        tokens = list(adapter.stream_generate("System", "User"))

        assert tokens == ["<tag>", "&quot;", ""]

    @patch("src.llm_client.LLMClient")
    def test_multiple_generate_calls(self, mock_llm_client):
        mock_instance = MagicMock()
        mock_instance.complete.side_effect = ["Response1", "Response2", "Response3"]
        mock_llm_client.return_value = mock_instance

        adapter = LLMClientAdapter()
        r1 = adapter.generate("S1", "U1")
        r2 = adapter.generate("S2", "U2")
        r3 = adapter.generate("S3", "U3")

        assert r1 == "Response1"
        assert r2 == "Response2"
        assert r3 == "Response3"
        assert mock_instance.complete.call_count == 3
