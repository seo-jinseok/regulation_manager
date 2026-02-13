"""
LLM Client Adapter for Regulation RAG System with Multi-Provider Fallback.

Adapts the existing LLMClient to the ILLMClient interface.
Supports multiple providers: Ollama, LM Studio, MLX, OpenAI, Gemini, OpenRouter.

Features:
- Multi-provider fallback chain with graceful degradation
- Retry logic with exponential backoff
- Comprehensive logging for monitoring
- Failure caching to avoid repeated failed attempts
"""

import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Generator, List, Optional

from ..domain.repositories import ILLMClient

if TYPE_CHECKING:
    from llm_client import LLMClient

# Configure logger for fallback events
logger = logging.getLogger(__name__)


class LLMFallbackError(Exception):
    """Raised when all LLM providers in the fallback chain fail."""

    def __init__(self, message: str, attempts: List[Dict]):
        self.attempts = attempts
        super().__init__(message)


# API 오류 타입 상수 (SPEC-RAG-Q-001 Phase 1)
API_ERROR_INSUFFICIENT_BALANCE = "insufficient_balance"  # 402, 429
API_ERROR_RATE_LIMIT = "rate_limit"  # 429
API_ERROR_AUTHENTICATION = "authentication"  # 401, 403
API_ERROR_NETWORK = "network"  # Connection errors
API_ERROR_UNKNOWN = "unknown"


def classify_api_error(error: Exception, error_message: str) -> str:
    """API 오류를 분류하여 오류 타입을 반환합니다.

    Args:
        error: 발생한 예외 객체
        error_message: 오류 메시지 문자열

    Returns:
        오류 타입 문자열 (API_ERROR_* 상수 중 하나)
    """
    error_msg_lower = error_message.lower()
    error_type_name = type(error).__name__.lower()

    # 잔액 부족 오류 (402 Payment Required, 429 Too Many Requests)
    if "402" in error_msg_lower or "payment" in error_msg_lower:
        return API_ERROR_INSUFFICIENT_BALANCE
    if "429" in error_msg_lower or "insufficient balance" in error_msg_lower or "no resource package" in error_msg_lower:
        return API_ERROR_INSUFFICIENT_BALANCE

    # Rate limiting (429에서 잔액 부족이 아닌 경우)
    if "rate limit" in error_msg_lower and "balance" not in error_msg_lower:
        return API_ERROR_RATE_LIMIT

    # 인증 오류
    if "401" in error_msg_lower or "403" in error_msg_lower or "unauthorized" in error_msg_lower or "forbidden" in error_msg_lower:
        return API_ERROR_AUTHENTICATION

    # 네트워크 오류
    if "connection" in error_type_name or "timeout" in error_msg_lower or "network" in error_msg_lower:
        return API_ERROR_NETWORK

    return API_ERROR_UNKNOWN


class FailureCache:
    """Cache for recent provider failures to avoid repeated failed attempts."""

    def __init__(self, ttl_seconds: int = 300):
        """Initialize failure cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 5 minutes)
        """
        self._failures: Dict[str, datetime] = {}
        self._ttl_seconds = ttl_seconds

    def mark_failure(self, provider_key: str) -> None:
        """Mark a provider as failed at current time."""
        self._failures[provider_key] = datetime.now()

    def is_failed(self, provider_key: str) -> bool:
        """Check if a provider is marked as recently failed.

        Args:
            provider_key: Unique identifier for the provider (e.g., "openrouter:model")

        Returns:
            True if provider failed recently and is within TTL
        """
        if provider_key not in self._failures:
            return False

        failure_time = self._failures[provider_key]
        if datetime.now() - failure_time > timedelta(seconds=self._ttl_seconds):
            # Expired entry
            del self._failures[provider_key]
            return False

        return True

    def clear(self) -> None:
        """Clear all failure entries."""
        self._failures.clear()

    def cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        now = datetime.now()
        expired = [
            key
            for key, failure_time in self._failures.items()
            if now - failure_time > timedelta(seconds=self._ttl_seconds)
        ]
        for key in expired:
            del self._failures[key]


class LLMClientAdapter(ILLMClient):
    """
    Adapter that wraps the existing LLMClient to implement ILLMClient.

    Features:
    - Multi-provider fallback chain
    - Retry logic with exponential backoff
    - Failure caching
    - Comprehensive logging

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
        fallback_enabled: Optional[bool] = None,
        fallback_chain: Optional[List[Dict]] = None,
    ):
        """
        Initialize LLM client adapter.

        Args:
            provider: LLM provider (ollama, lmstudio, mlx, local, openai, gemini, openrouter).
                      If not provided, reads from LLM_PROVIDER env var.
            model: Model name (optional, uses provider default or LLM_MODEL env var)
            base_url: Base URL for local providers (optional, uses LLM_BASE_URL env var)
            api_key: API key for cloud providers
            fallback_enabled: Enable/disable fallback chain (overrides config)
            fallback_chain: Custom fallback chain (overrides config)
        """
        # Lazy import to avoid circular dependencies

        # Add project root to path
        project_root = str(Path(__file__).parent.parent.parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Add src to path if not already there
        src_path = str(Path(__file__).parent.parent.parent.parent)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            from llm_client import LLMClient
        except ImportError:
            # Fallback to absolute import
            from src.llm_client import LLMClient

        from ..config import get_config

        # Get configuration
        config = get_config()
        fallback_config = config.llm_fallback

        # Initialize settings
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        self.model = model or os.getenv("LLM_MODEL")
        self.base_url = base_url or os.getenv("LLM_BASE_URL")

        # Fallback configuration
        self._fallback_enabled = (
            fallback_enabled
            if fallback_enabled is not None
            else fallback_config.enabled
        )
        self._fallback_chain = fallback_chain or fallback_config.provider_chain
        self._max_retries = fallback_config.max_retries
        self._initial_backoff = fallback_config.initial_backoff_seconds
        self._max_backoff = fallback_config.max_backoff_seconds
        self._backoff_multiplier = fallback_config.backoff_multiplier
        self._allow_partial = fallback_config.allow_partial_results
        self._partial_message = fallback_config.partial_result_fallback_message

        # Sort fallback chain by priority
        self._fallback_chain_sorted = sorted(
            self._fallback_chain, key=lambda x: x.get("priority", 999)
        )

        # Initialize failure cache
        self._failure_cache = FailureCache(
            ttl_seconds=fallback_config.failure_cache_ttl_seconds
            if fallback_config.cache_failures
            else 0
        )

        # Create primary client
        self._client = LLMClient(
            provider=self.provider,
            model=self.model,
            api_key=api_key,
            base_url=self.base_url,
        )

        # Statistics for monitoring
        self._stats = defaultdict(int)
        self._stats["primary_success"] = 0
        self._stats["primary_failure"] = 0
        self._stats["fallback_success"] = 0
        self._stats["fallback_failure"] = 0
        self._stats["retries"] = 0

    def _create_client_for_config(self, config: Dict) -> "LLMClient":
        """Create an LLMClient instance for a specific provider configuration.

        Args:
            config: Provider configuration dictionary

        Returns:
            LLMClient instance
        """
        # Import with absolute path (project root already added to sys.path)
        from src.llm_client import LLMClient

        provider = config["provider"]
        model = config.get("model")
        base_url = config.get("base_url")
        api_key_env_var = config.get("api_key_env_var")

        # Get API key from environment if specified
        api_key = None
        if api_key_env_var:
            api_key = os.getenv(api_key_env_var)

        return LLMClient(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    def _get_provider_key(self, provider: str, model: Optional[str] = None) -> str:
        """Generate a unique key for a provider configuration.

        Args:
            provider: Provider name
            model: Model name (optional)

        Returns:
            Unique key string
        """
        if model:
            return f"{provider}:{model}"
        return provider

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = min(
            self._initial_backoff * (self._backoff_multiplier**attempt),
            self._max_backoff,
        )
        return delay

    def _try_generate(
        self,
        client: "LLMClient",
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        provider_name: str = "unknown",
    ) -> str:
        """Try to generate a response from a specific client.

        Args:
            client: LLMClient instance
            system_prompt: System instructions
            user_message: User's question with context
            temperature: Sampling temperature
            provider_name: Name of the provider (for logging)

        Returns:
            Generated response text

        Raises:
            Exception: If generation fails
        """
        full_prompt = f"""<system>
{system_prompt}
</system>

<user>
{user_message}
</user>

<assistant>"""

        try:
            response = client.complete(full_prompt)
            # 빈 응답 검증 - 빈 응답은 실패로 처리하여 재시도 유도
            if not response or not response.strip():
                logger.warning(
                    f"LLM returned empty response for provider '{provider_name}'"
                )
                raise ValueError("Empty response from LLM")
            return response
        except Exception as e:
            logger.warning(
                f"LLM generation failed for provider '{provider_name}': {type(e).__name__}: {e}"
            )
            raise

    def _execute_with_fallback(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
    ) -> tuple[str, List[Dict]]:
        """
        Execute LLM generation with fallback chain.

        Args:
            system_prompt: System instructions
            user_message: User's question with context
            temperature: Sampling temperature

        Returns:
            Tuple of (response_text, attempts_list)

        Raises:
            LLMFallbackError: If all providers fail
        """
        attempts = []

        # Try primary provider first
        try:
            response = self._try_generate(
                self._client,
                system_prompt,
                user_message,
                temperature,
                provider_name=self.provider,
            )
            self._stats["primary_success"] += 1
            attempts.append(
                {
                    "provider": self.provider,
                    "model": self.model,
                    "success": True,
                    "attempt": 1,
                }
            )
            return response, attempts
        except Exception as e:
            self._stats["primary_failure"] += 1
            primary_error = e
            error_msg = str(e)

            # SPEC-RAG-Q-001 Phase 1: 오류 타입 분류
            api_error_type = classify_api_error(e, error_msg)

            attempts.append(
                {
                    "provider": self.provider,
                    "model": self.model,
                    "success": False,
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "api_error_type": api_error_type,
                    "attempt": 1,
                }
            )

            # 잔액 부족 오류 시 특별 로깅
            if api_error_type == API_ERROR_INSUFFICIENT_BALANCE:
                logger.warning(
                    f"Primary provider '{self.provider}' failed due to insufficient balance/rate limit. "
                    f"Initiating fallback chain..."
                )
            else:
                logger.info(
                    f"Primary provider '{self.provider}' failed ({api_error_type}), trying fallback chain..."
                )

        # Try fallback chain if enabled
        if not self._fallback_enabled:
            if self._allow_partial:
                raise LLMFallbackError(
                    f"Primary provider failed and fallback is disabled: {primary_error}",
                    attempts,
                )
            raise primary_error

        # Clean up expired failures
        self._failure_cache.cleanup_expired()

        # Try each provider in the fallback chain
        for idx, provider_config in enumerate(self._fallback_chain_sorted):
            provider = provider_config["provider"]
            model = provider_config.get("model")
            provider_key = self._get_provider_key(provider, model)

            # Skip if recently failed (caching)
            if self._failure_cache.is_failed(provider_key):
                logger.info(
                    f"Skipping provider '{provider_key}' - marked as recently failed"
                )
                attempts.append(
                    {
                        "provider": provider,
                        "model": model,
                        "success": False,
                        "skipped": "cached_failure",
                        "attempt": idx + 2,
                    }
                )
                continue

            # Try with retry logic
            for retry in range(self._max_retries):
                try:
                    client = self._create_client_for_config(provider_config)
                    response = self._try_generate(
                        client,
                        system_prompt,
                        user_message,
                        temperature,
                        provider_name=provider,
                    )

                    self._stats["fallback_success"] += 1
                    if retry > 0:
                        self._stats["retries"] += retry

                    log_msg = (
                        f"Fallback to provider '{provider}' succeeded"
                        if retry == 0
                        else f"Fallback to provider '{provider}' succeeded after {retry} retries"
                    )
                    logger.info(log_msg)

                    attempts.append(
                        {
                            "provider": provider,
                            "model": model,
                            "success": True,
                            "retries": retry,
                            "attempt": idx + 2,
                        }
                    )

                    # Add fallback notice if configured
                    if self._allow_partial and self._partial_message:
                        response = f"{self._partial_message}\n\n{response}"

                    return response, attempts

                except Exception as e:
                    self._stats["retries"] += 1
                    error_msg = str(e)

                    # SPEC-RAG-Q-001 Phase 1: 폴백 제공자 오류 분류
                    api_error_type = classify_api_error(e, error_msg)

                    if retry < self._max_retries - 1:
                        backoff = self._calculate_backoff(retry)
                        logger.warning(
                            f"Provider '{provider}' failed ({api_error_type}, retry {retry + 1}/{self._max_retries}), "
                            f"retrying in {backoff:.1f}s: {e}"
                        )
                        time.sleep(backoff)
                    else:
                        # All retries exhausted, mark as failed
                        self._failure_cache.mark_failure(provider_key)
                        self._stats["fallback_failure"] += 1

                        # 잔액 부족 오류 시 특별 로깅
                        if api_error_type == API_ERROR_INSUFFICIENT_BALANCE:
                            logger.error(
                                f"Provider '{provider}' failed due to insufficient balance after {self._max_retries} retries. "
                                f"Consider recharging API balance or checking account status."
                            )
                        else:
                            logger.error(
                                f"Provider '{provider}' failed ({api_error_type}) after {self._max_retries} retries"
                            )

                        attempts.append(
                            {
                                "provider": provider,
                                "model": model,
                                "success": False,
                                "error": error_msg,
                                "error_type": type(e).__name__,
                                "api_error_type": api_error_type,
                                "retries": retry,
                                "attempt": idx + 2,
                            }
                        )

        # All providers failed
        error_msg = f"All LLM providers failed after {len(attempts)} attempts"
        if self._allow_partial:
            raise LLMFallbackError(error_msg, attempts)
        else:
            raise LLMFallbackError(error_msg, attempts)

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a response from the LLM with fallback support.

        Args:
            system_prompt: System instructions.
            user_message: User's question with context.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            Generated response text.
        """
        response, attempts = self._execute_with_fallback(
            system_prompt, user_message, temperature
        )

        # Log successful fallback for monitoring
        successful_attempts = [a for a in attempts if a.get("success")]
        if len(successful_attempts) > 1 or attempts[0].get("retries", 0) > 0:
            logger.info(
                f"LLM generation completed with fallback: "
                f"{len(successful_attempts)} successful attempts, "
                f"{len(attempts) - len(successful_attempts)} failed providers"
            )

        # SPEC-RAG-Q-001 Phase 1: 잔액 부족 오류가 발생한 경우 사용자 알림 메시지 추가
        insufficient_balance_errors = [
            a for a in attempts
            if a.get("api_error_type") == API_ERROR_INSUFFICIENT_BALANCE
        ]
        if insufficient_balance_errors:
            failed_providers = ", ".join(set(a.get("provider", "unknown") for a in insufficient_balance_errors))
            logger.warning(
                f"API INSUFFICIENT BALANCE DETECTED: The following providers reported balance/payment issues: {failed_providers}. "
                f"Fallback to alternative providers was used. Please check your API account balance."
            )

        return response

    def stream_generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
    ) -> Generator[str, None, None]:
        """
        Stream a response from the LLM token by token.

        Note: Fallback is not supported for streaming.
        If the primary provider fails, the exception is raised immediately.

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

        try:
            for token in self._client.stream_complete(full_prompt):
                yield token
        except Exception as e:
            logger.warning(
                f"LLM streaming failed for provider '{self.provider}': "
                f"{type(e).__name__}: {e}"
            )
            # For streaming, we don't implement fallback as it would break
            # the streaming contract. The caller should handle retries.
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text.

        Note: Not implemented as ChromaDB handles embeddings internally.
        """
        raise NotImplementedError(
            "Embedding is handled by ChromaDB's default embedding function."
        )

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about LLM usage and fallback behavior.

        Returns:
            Dictionary with statistics keys:
            - primary_success: Successful generations using primary provider
            - primary_failure: Failed generations using primary provider
            - fallback_success: Successful generations using fallback providers
            - fallback_failure: Failed generations using fallback providers
            - retries: Total retry attempts
        """
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats.clear()
        self._stats["primary_success"] = 0
        self._stats["primary_failure"] = 0
        self._stats["fallback_success"] = 0
        self._stats["fallback_failure"] = 0
        self._stats["retries"] = 0

    def clear_failure_cache(self) -> None:
        """Clear the failure cache, allowing all providers to be retried."""
        self._failure_cache.clear()
