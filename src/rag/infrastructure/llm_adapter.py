"""
LLM Client Adapter for Regulation RAG System.

Adapts the existing LLMClient to the ILLMClient interface.
Supports multiple providers: Ollama, LM Studio, OpenAI, Gemini.
"""

from typing import List, Optional

from ..domain.repositories import ILLMClient


class LLMClientAdapter(ILLMClient):
    """
    Adapter that wraps the existing LLMClient to implement ILLMClient.
    
    Supports:
    - Ollama (local)
    - LM Studio (local)
    - OpenAI (cloud)
    - Gemini (cloud)
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize LLM client adapter.

        Args:
            provider: LLM provider (ollama, lmstudio, openai, gemini)
            model: Model name (optional, uses provider default)
            base_url: Base URL for local providers
            api_key: API key for cloud providers
        """
        # Lazy import to avoid circular dependencies
        from ...llm_client import LLMClient
        
        self.provider = provider
        self.model = model
        self.base_url = base_url
        
        self._client = LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
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

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text.
        
        Note: Not implemented as ChromaDB handles embeddings internally.
        """
        raise NotImplementedError(
            "Embedding is handled by ChromaDB's default embedding function."
        )
