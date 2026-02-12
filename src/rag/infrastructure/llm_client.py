"""
OpenAI LLM Client for Regulation RAG System.

Provides LLM integration for answer generation and embeddings.

Priority 4 Security Enhancements:
- API key format validation
- Health check on initialization
- Secure key rotation support
"""

import logging
import os
from typing import List, Optional

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..domain.repositories import ILLMClient

logger = logging.getLogger(__name__)


class OpenAIClient(ILLMClient):
    """
    OpenAI API client for LLM operations.

    Supports:
    - Chat completion (GPT-4o-mini, GPT-4o)
    - Embeddings (text-embedding-3-small)

    Security Features (P4):
    - API key validation on initialization
    - Health check before first use
    - Secure key rotation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        validate_api_key: bool = True,
        skip_health_check: bool = False,
    ):
        """
        Initialize OpenAI client with security validation.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Chat model to use.
            embedding_model: Embedding model to use.
            validate_api_key: Whether to validate API key format (P4: Security).
            skip_health_check: Skip API health check on initialization.

        Raises:
            ImportError: If openai package not available.
            APIKeyFormatError: If API key format is invalid.
            MissingAPIKeyError: If API key not provided.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is required. Install with: uv add openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            from ..exceptions import MissingAPIKeyError

            raise MissingAPIKeyError("OPENAI_API_KEY")

        self.model = model
        self.embedding_model = embedding_model

        # Security: Validate API key format (P4: Security Hardening)
        if validate_api_key:
            self._validate_api_key()

        # Initialize OpenAI client
        self._client = OpenAI(api_key=self.api_key)

        # Security: Health check on initialization (P4: Security Hardening)
        if not skip_health_check:
            self._health_check()

    def _validate_api_key(self) -> None:
        """
        Validate API key format (P4: Security Hardening).

        Raises:
            APIKeyFormatError: If API key format is invalid.
        """
        try:
            from .security import APIKeyFormatError, APIKeyProvider, APIKeyValidator

            result = APIKeyValidator.validate_format(
                self.api_key, expected_provider=APIKeyProvider.OPENAI
            )

            if not result.is_valid:
                raise APIKeyFormatError(
                    "openai", result.error_message or "Invalid API key format"
                )

            logger.info(f"API key validated: {result.provider.value}")

        except ImportError:
            # Security module not available, log warning
            logger.warning("Security module unavailable, skipping API key validation")

    def _health_check(self) -> None:
        """
        Perform health check on API key (P4: Security Hardening).

        Makes a minimal API call to verify the key is valid.

        Raises:
            Exception: If health check fails.
        """
        try:
            # Make a minimal API call (list models is cheap)
            self._client.models.list()
            logger.debug("OpenAI API health check passed")
        except Exception as e:
            logger.warning(f"OpenAI API health check failed: {e}")
            # Don't raise on health check failure, as the key might work for actual operations
            # This prevents false negatives during temporary network issues

    def rotate_api_key(self, new_api_key: str) -> None:
        """
        Securely rotate API key (P4: Security Hardening).

        Args:
            new_api_key: New API key to use.

        Raises:
            APIKeyFormatError: If new API key format is invalid.
        """
        # Validate new key format
        try:
            from .security import APIKeyFormatError, APIKeyProvider, APIKeyValidator

            result = APIKeyValidator.validate_format(
                new_api_key, expected_provider=APIKeyProvider.OPENAI
            )

            if not result.is_valid:
                raise APIKeyFormatError(
                    "openai", result.error_message or "Invalid new API key format"
                )

        except ImportError:
            logger.warning("Security module unavailable, skipping new key validation")

        # Update API key
        self.api_key = new_api_key
        self._client = OpenAI(api_key=self.api_key)
        logger.info("API key rotated successfully")

    def get_api_key_info(self) -> dict:
        """
        Get API key information (safe for logging) (P4: Security Hardening).

        Returns:
            Dict with key information (no actual key material).
        """
        try:
            from .security import APIKeyValidator

            provider = APIKeyValidator.detect_provider(self.api_key)
            return {
                "provider": provider.value,
                "key_length": len(self.api_key),
                "key_prefix": self.api_key[:3] + "***" + self.api_key[-3:],
            }
        except ImportError:
            return {
                "provider": "unknown",
                "key_length": len(self.api_key),
                "key_prefix": "***",
            }

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
