"""
LLM Client Adapter for Regulation RAG System.

Adapts the existing LLMClient to the ILLMClient interface.
Supports multiple providers: Ollama, LM Studio, MLX, OpenAI, Gemini, OpenRouter.
"""

import os
from typing import List, Optional

from ..domain.repositories import ILLMClient


class LLMClientAdapter(ILLMClient):
    """
    Adapter that wraps the existing LLMClient to implement ILLMClient.

    Supports:
    - Ollama (local)
    - LM Studio (local)
    - MLX (local, OpenAI-compatible server)
    - OpenAI (cloud)
    - Gemini (cloud)
    - OpenRouter (cloud)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize LLM client adapter.

        Args:
            provider: LLM provider (ollama, lmstudio, mlx, local, openai, gemini, openrouter).
                      If not provided, reads from LLM_PROVIDER env var.
            model: Model name (optional, uses provider default or LLM_MODEL env var)
            base_url: Base URL for local providers (optional, uses LLM_BASE_URL env var)
            api_key: API key for cloud providers
        """
        # Lazy import to avoid circular dependencies
        from ...llm_client import LLMClient

        # Read from environment variables if not provided
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        self.model = model or os.getenv("LLM_MODEL")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")

        self._client = LLMClient(
            provider=self.provider,
            model=self.model,
            api_key=api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System instructions.
            user_message: User's question with context.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            Generated response text.
        """
        # Combine system prompt and user message for the underlying client
        # Most LLM providers handle this via chat format internally
        full_prompt = f"""<system>
{system_prompt}
</system>

<user>
{user_message}
</user>

<assistant>"""

        return self._client.complete(full_prompt)

    def stream_generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
    ):
        """
        Stream a response from the LLM token by token.

        Args:
            system_prompt: System instructions.
            user_message: User's question with context.
            temperature: Sampling temperature (0.0 = deterministic).

        Yields:
            str: Each token/chunk as it becomes available.
        """
        full_prompt = f"""<system>
{system_prompt}
</system>

<user>
{user_message}
</user>

<assistant>"""

        for token in self._client.stream_complete(full_prompt):
            yield token

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text.

        Note: Not implemented as ChromaDB handles embeddings internally.
        """
        raise NotImplementedError(
            "Embedding is handled by ChromaDB's default embedding function."
        )
