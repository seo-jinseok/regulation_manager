"""
OpenAI LLM Client for Regulation RAG System.

Provides LLM integration for answer generation and embeddings.
"""

import os
from typing import List, Optional

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..domain.repositories import ILLMClient


class OpenAIClient(ILLMClient):
    """
    OpenAI API client for LLM operations.

    Supports:
    - Chat completion (GPT-4o-mini, GPT-4o)
    - Embeddings (text-embedding-3-small)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Chat model to use.
            embedding_model: Embedding model to use.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is required. Install with: uv add openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )

        self._client = OpenAI(api_key=self.api_key)
        self.model = model
        self.embedding_model = embedding_model

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a response using chat completion.

        Args:
            system_prompt: System instructions.
            user_message: User's question with context.
            temperature: Sampling temperature.

        Returns:
            Generated response text.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=2048,
        )

        return response.choices[0].message.content or ""

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        response = self._client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )

        return response.data[0].embedding


class MockLLMClient(ILLMClient):
    """
    Mock LLM client for testing without API calls.
    """

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
    ) -> str:
        """Return mock response."""
        return f"[Mock Response] 질문에 대한 답변입니다. (시스템 프롬프트 길이: {len(system_prompt)})"

    def get_embedding(self, text: str) -> List[float]:
        """Return mock embedding."""
        # Simple hash-based mock embedding
        import hashlib

        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Convert to 384-dim vector (text-embedding-3-small dimension)
        embedding = []
        for i in range(384):
            byte_idx = i % len(hash_bytes)
            embedding.append((hash_bytes[byte_idx] - 128) / 128.0)
        return embedding
