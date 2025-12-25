import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.llm_client import LLMClient

class TestLLMClientExtended(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {
            "OPENAI_API_KEY": "test",
            "GEMINI_API_KEY": "test",
            "OPENROUTER_API_KEY": "test"
        })
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch('src.llm_client.Gemini')
    def test_provider_gemini(self, mock_gemini):
        client = LLMClient(provider="gemini")
        self.assertEqual(client.provider, "gemini")

    @patch('src.llm_client.OpenRouter')
    def test_provider_openrouter(self, mock_or):
        client = LLMClient(provider="openrouter")
        self.assertEqual(client.provider, "openrouter")

    @patch('src.llm_client.Ollama')
    def test_provider_ollama(self, mock_ollama):
        client = LLMClient(provider="ollama", model="gemma2")
        self.assertEqual(client.provider, "ollama")

    @patch('src.llm_client.OpenAILike')
    def test_provider_lmstudio(self, mock_oa):
        client = LLMClient(provider="lmstudio", base_url="http://localhost:1234")
        self.assertEqual(client.provider, "lmstudio")

    @patch('src.llm_client.OpenAILike')
    def test_provider_mlx(self, mock_oa):
        client = LLMClient(provider="mlx", base_url="http://localhost:8080")
        self.assertEqual(client.provider, "mlx")

    def test_api_key_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                LLMClient(provider="openai")

    def test_complete_success(self):
        client = LLMClient(provider="openai")
        client.llm = MagicMock()
        client.llm.complete.return_value = MagicMock(text="Response")
        self.assertEqual(client.complete("Hi"), "Response")

if __name__ == "__main__":
    unittest.main()
