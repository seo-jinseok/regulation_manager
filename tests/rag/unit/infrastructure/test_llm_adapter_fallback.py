"""
Tests for LLMClientAdapter multi-provider fallback functionality.

Tests fallback chain, retry logic, exponential backoff, failure caching,
and graceful degradation.
"""

import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

import pytest

from src.rag.infrastructure.llm_adapter import (
    LLMClientAdapter,
    LLMFallbackError,
    FailureCache,
)


class TestFailureCache:
    """Test the FailureCache class."""

    def test_mark_and_check_failure(self):
        """Test basic failure marking and checking."""
        cache = FailureCache(ttl_seconds=60)
        
        assert not cache.is_failed("provider:model")
        cache.mark_failure("provider:model")
        assert cache.is_failed("provider:model")

    def test_failure_expiration(self):
        """Test that failures expire after TTL."""
        cache = FailureCache(ttl_seconds=1)
        
        cache.mark_failure("provider:model")
        assert cache.is_failed("provider:model")
        
        # Wait for expiration
        time.sleep(1.1)
        assert not cache.is_failed("provider:model")

    def test_clear(self):
        """Test clearing the failure cache."""
        cache = FailureCache(ttl_seconds=60)
        
        cache.mark_failure("provider1:model1")
        cache.mark_failure("provider2:model2")
        
        assert cache.is_failed("provider1:model1")
        assert cache.is_failed("provider2:model2")
        
        cache.clear()
        assert not cache.is_failed("provider1:model1")
        assert not cache.is_failed("provider2:model2")

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = FailureCache(ttl_seconds=1)
        
        cache.mark_failure("provider1:model1")
        cache.mark_failure("provider2:model2")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Cleanup should remove expired entries
        cache.cleanup_expired()
        
        # Add a new non-expired entry
        cache.mark_failure("provider3:model3")
        assert cache.is_failed("provider3:model3")

    def test_multiple_provider_keys(self):
        """Test multiple provider keys are tracked independently."""
        cache = FailureCache(ttl_seconds=60)
        
        cache.mark_failure("openrouter:gemma")
        cache.mark_failure("lmstudio:default")
        
        assert cache.is_failed("openrouter:gemma")
        assert cache.is_failed("lmstudio:default")


class TestLLMFallbackChain:
    """Test fallback chain functionality."""

    @patch("src.llm_client.LLMClient")
    def test_fallback_to_second_provider(self, mock_llm_client_class):
        """Test fallback when primary provider fails."""
        # Setup: Primary client fails, fallback succeeds
        primary_client = MagicMock()
        primary_client.complete.side_effect = Exception("Primary failed")
        
        fallback_client = MagicMock()
        fallback_client.complete.return_value = "Fallback response"
        
        def create_client(provider, model, api_key, base_url):
            if provider == "ollama":
                return primary_client
            elif provider == "lmstudio":
                return fallback_client
            return MagicMock()
        
        mock_llm_client_class.side_effect = create_client
        
        # Create adapter with fallback chain
        fallback_chain = [
            {"provider": "lmstudio", "model": None, "base_url": "http://localhost:1234", "priority": 2},
        ]
        
        adapter = LLMClientAdapter(
            provider="ollama",
            fallback_enabled=True,
            fallback_chain=fallback_chain,
        )
        
        # Generate with fallback
        result = adapter.generate("System", "User message")
        
        # Result should contain the fallback response (with notice prepended)
        assert "Fallback response" in result
        assert "Note: Response generated using fallback provider" in result
        assert primary_client.complete.call_count == 1
        assert fallback_client.complete.call_count == 1

    @patch("src.llm_client.LLMClient")
    def test_all_providers_fail(self, mock_llm_client_class):
        """Test exception when all providers in chain fail."""
        # All clients fail
        def create_client(provider, model, api_key, base_url):
            client = MagicMock()
            client.complete.side_effect = Exception(f"{provider} failed")
            return client
        
        mock_llm_client_class.side_effect = create_client
        
        fallback_chain = [
            {"provider": "lmstudio", "model": None, "base_url": "http://localhost:1234", "priority": 2},
            {"provider": "openrouter", "model": "gemini", "priority": 3},
        ]
        
        adapter = LLMClientAdapter(
            provider="ollama",
            fallback_enabled=True,
            fallback_chain=fallback_chain,
        )
        
        # Should raise LLMFallbackError
        with pytest.raises(LLMFallbackError) as exc_info:
            adapter.generate("System", "User")
        
        assert "All LLM providers failed" in str(exc_info.value)
        assert len(exc_info.value.attempts) == 3  # primary + 2 fallbacks

    @patch("src.llm_client.LLMClient")
    def test_fallback_disabled(self, mock_llm_client_class):
        """Test behavior when fallback is disabled."""
        primary_client = MagicMock()
        primary_client.complete.side_effect = Exception("Primary failed")
        
        mock_llm_client_class.return_value = primary_client
        
        adapter = LLMClientAdapter(
            provider="ollama",
            fallback_enabled=False,
        )
        
        # Should raise the primary exception, not LLMFallbackError
        with pytest.raises(Exception) as exc_info:
            adapter.generate("System", "User")
        
        assert "Primary failed" in str(exc_info.value)

    @patch("src.llm_client.LLMClient")
    def test_primary_succeeds_no_fallback(self, mock_llm_client_class):
        """Test that fallback is not used when primary succeeds."""
        primary_client = MagicMock()
        primary_client.complete.return_value = "Primary response"
        
        fallback_client = MagicMock()
        fallback_client.complete.return_value = "Fallback response"
        
        def create_client(provider, model, api_key, base_url):
            if provider == "ollama":
                return primary_client
            return fallback_client
        
        mock_llm_client_class.side_effect = create_client
        
        fallback_chain = [
            {"provider": "lmstudio", "model": None, "base_url": "http://localhost:1234", "priority": 2},
        ]
        
        adapter = LLMClientAdapter(
            provider="ollama",
            fallback_enabled=True,
            fallback_chain=fallback_chain,
        )
        
        result = adapter.generate("System", "User")
        
        assert result == "Primary response"
        assert primary_client.complete.call_count == 1
        assert fallback_client.complete.call_count == 0  # Never called


class TestRetryWithBackoff:
    """Test retry logic with exponential backoff."""

    @patch("src.llm_client.LLMClient")
    @patch("time.sleep")  # Mock sleep to speed up tests
    def test_retry_on_failure(self, mock_sleep, mock_llm_client_class):
        """Test that failed requests are retried within fallback providers."""
        primary_call_count = [0]
        
        # Primary fails immediately
        def create_primary_client():
            client = MagicMock()
            client.complete.side_effect = Exception("Primary fails")
            return client
        
        # Fallback succeeds after 2 retries
        fallback_call_count = [0]
        def create_fallback_client():
            client = MagicMock()
            
            def complete_side_effect(prompt):
                fallback_call_count[0] += 1
                if fallback_call_count[0] < 3:  # First 2 attempts fail
                    raise Exception("Temporary failure")
                return "Success after retry"
            
            client.complete.side_effect = complete_side_effect
            return client
        
        def create_client(provider, model, api_key, base_url):
            if provider == "ollama":
                return create_primary_client()
            return create_fallback_client()
        
        mock_llm_client_class.side_effect = create_client
        
        fallback_chain = [
            {"provider": "lmstudio", "model": None, "base_url": "http://localhost:1234", "priority": 2},
        ]
        
        adapter = LLMClientAdapter(
            provider="ollama",
            fallback_enabled=True,
            fallback_chain=fallback_chain,
        )
        
        # Should succeed after retries on fallback provider
        result = adapter.generate("System", "User")
        assert "Success after retry" in result
        assert fallback_call_count[0] == 3  # 2 failures + 1 success

    @patch("src.llm_client.LLMClient")
    @patch("time.sleep")
    def test_exponential_backoff_timing(self, mock_sleep, mock_llm_client_class):
        """Test that backoff delays increase exponentially for fallback providers."""
        # Primary fails immediately, no retries
        def create_primary_client():
            client = MagicMock()
            client.complete.side_effect = Exception("Primary fails")
            return client
        
        # Fallback fails after retries, triggering backoff
        def create_fallback_client():
            client = MagicMock()
            client.complete.side_effect = Exception("Fallback fails")
            return client
        
        def create_client(provider, model, api_key, base_url):
            if provider == "ollama":
                return create_primary_client()
            return create_fallback_client()
        
        mock_llm_client_class.side_effect = create_client
        
        fallback_chain = [
            {"provider": "lmstudio", "model": None, "base_url": "http://localhost:1234", "priority": 2},
        ]
        
        adapter = LLMClientAdapter(
            provider="ollama",
            fallback_enabled=True,
            fallback_chain=fallback_chain,
        )
        
        with pytest.raises(LLMFallbackError):
            adapter.generate("System", "User")
        
        # Check that sleep was called with increasing delays (from fallback retries)
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        # Should have at least 2 retries on the fallback provider
        assert len(sleep_calls) >= 2
        # Each delay should be larger than the previous (exponential backoff)
        assert sleep_calls[1] > sleep_calls[0]

    @patch("src.llm_client.LLMClient")
    @patch("time.sleep")
    def test_max_backoff_limit(self, mock_sleep, mock_llm_client_class):
        """Test that backoff is capped at max_backoff_seconds."""
        def create_client(provider, model, api_key, base_url):
            client = MagicMock()
            client.complete.side_effect = Exception("Always fails")
            return client
        
        mock_llm_client_class.side_effect = create_client
        
        adapter = LLMClientAdapter(
            provider="ollama",
            fallback_enabled=False,
        )
        
        # Adapter uses max_backoff of 32 seconds
        # Check that no sleep call exceeds this
        with pytest.raises(Exception):
            adapter.generate("System", "User")
        
        for call in mock_sleep.call_args_list:
            delay = call[0][0]
            assert delay <= 32.0, f"Backoff delay {delay} exceeds max of 32.0"


class TestFailureCaching:
    """Test provider failure caching behavior."""

    @patch("src.llm_client.LLMClient")
    def test_failed_provider_is_skipped(self, mock_llm_client_class):
        """Test that recently failed providers are skipped."""
        # First call fails and gets cached
        call_counts = {"ollama": 0, "lmstudio": 0, "openrouter": 0}
        
        def create_client(provider, model, api_key, base_url):
            call_counts[provider] = call_counts.get(provider, 0) + 1
            client = MagicMock()
            client.complete.side_effect = Exception(f"{provider} failed")
            return client
        
        mock_llm_client_class.side_effect = create_client
        
        fallback_chain = [
            {"provider": "lmstudio", "model": None, "base_url": "http://localhost:1234", "priority": 2},
            {"provider": "openrouter", "model": "gemini", "priority": 3},
        ]
        
        adapter = LLMClientAdapter(
            provider="ollama",
            fallback_enabled=True,
            fallback_chain=fallback_chain,
        )
        
        # First attempt - all fail
        with pytest.raises(LLMFallbackError):
            adapter.generate("System", "User")
        
        # Second attempt - cached failures should be skipped immediately
        # The providers should not even be tried (no increase in call count)
        previous_counts = call_counts.copy()
        
        with pytest.raises(LLMFallbackError):
            adapter.generate("System", "User")
        
        # Call counts should not have increased for cached failures
        assert call_counts == previous_counts

    @patch("src.llm_client.LLMClient")
    def test_clear_failure_cache(self, mock_llm_client_class):
        """Test clearing the failure cache."""
        def create_client(provider, model, api_key, base_url):
            client = MagicMock()
            client.complete.side_effect = Exception(f"{provider} failed")
            return client
        
        mock_llm_client_class.side_effect = create_client
        
        adapter = LLMClientAdapter(provider="ollama", fallback_enabled=True)
        
        # First attempt fails
        with pytest.raises(LLMFallbackError):
            adapter.generate("System", "User")
        
        # Clear cache
        adapter.clear_failure_cache()
        
        # Now providers should be tried again (not skipped due to cache)
        # We can verify by checking that exceptions are raised again
        with pytest.raises(LLMFallbackError):
            adapter.generate("System", "User")


class TestGracefulDegradation:
    """Test graceful degradation behavior."""

    @patch("src.llm_client.LLMClient")
    def test_partial_result_with_fallback_notice(self, mock_llm_client_class):
        """Test that fallback responses include notice when configured."""
        primary_client = MagicMock()
        primary_client.complete.side_effect = Exception("Primary failed")
        
        fallback_client = MagicMock()
        fallback_client.complete.return_value = "Fallback response"
        
        def create_client(provider, model, api_key, base_url):
            if provider == "ollama":
                return primary_client
            return fallback_client
        
        mock_llm_client_class.side_effect = create_client
        
        fallback_chain = [
            {"provider": "lmstudio", "model": None, "base_url": "http://localhost:1234", "priority": 2},
        ]
        
        adapter = LLMClientAdapter(
            provider="ollama",
            fallback_enabled=True,
            fallback_chain=fallback_chain,
        )
        
        result = adapter.generate("System", "User")
        
        # Should include the fallback notice
        assert "Fallback response" in result
        assert "Note: Response generated using fallback provider" in result

    @patch("src.llm_client.LLMClient")
    def test_statistics_tracking(self, mock_llm_client_class):
        """Test that usage statistics are tracked correctly."""
        primary_client = MagicMock()
        primary_client.complete.side_effect = Exception("Primary failed")
        
        fallback_client = MagicMock()
        fallback_client.complete.return_value = "Fallback response"
        
        def create_client(provider, model, api_key, base_url):
            if provider == "ollama":
                return primary_client
            return fallback_client
        
        mock_llm_client_class.side_effect = create_client
        
        fallback_chain = [
            {"provider": "lmstudio", "model": None, "base_url": "http://localhost:1234", "priority": 2},
        ]
        
        adapter = LLMClientAdapter(
            provider="ollama",
            fallback_enabled=True,
            fallback_chain=fallback_chain,
        )
        
        # Reset stats
        adapter.reset_stats()
        
        # Generate with fallback
        adapter.generate("System", "User")
        
        # Check stats
        stats = adapter.get_stats()
        assert stats["primary_failure"] == 1
        assert stats["fallback_success"] == 1

    def test_stats_reset(self):
        """Test resetting statistics."""
        adapter = LLMClientAdapter(provider="ollama", fallback_enabled=False)
        
        # Stats should be initialized
        stats = adapter.get_stats()
        assert "primary_success" in stats
        
        # Reset and verify
        adapter.reset_stats()
        stats = adapter.get_stats()
        assert stats["primary_success"] == 0
        assert stats["primary_failure"] == 0


class TestStreamingFallback:
    """Test that streaming does not use fallback."""

    @patch("src.llm_client.LLMClient")
    def test_streaming_no_fallback_on_failure(self, mock_llm_client_class):
        """Test that streaming fails immediately without fallback."""
        primary_client = MagicMock()
        primary_client.stream_complete.side_effect = Exception("Stream failed")
        
        fallback_client = MagicMock()
        
        def create_client(provider, model, api_key, base_url):
            if provider == "ollama":
                return primary_client
            return fallback_client
        
        mock_llm_client_class.side_effect = create_client
        
        fallback_chain = [
            {"provider": "lmstudio", "model": None, "base_url": "http://localhost:1234", "priority": 2},
        ]
        
        adapter = LLMClientAdapter(
            provider="ollama",
            fallback_enabled=True,
            fallback_chain=fallback_chain,
        )
        
        # Should raise exception immediately, not fallback
        with pytest.raises(Exception) as exc_info:
            list(adapter.stream_generate("System", "User"))
        
        assert "Stream failed" in str(exc_info.value)
        assert fallback_client.stream_complete.call_count == 0

    @patch("src.llm_client.LLMClient")
    def test_streaming_success(self, mock_llm_client_class):
        """Test successful streaming."""
        primary_client = MagicMock()
        primary_client.stream_complete.return_value = iter(["Hello", " world"])
        
        mock_llm_client_class.return_value = primary_client
        
        adapter = LLMClientAdapter(provider="ollama")
        
        result = list(adapter.stream_generate("System", "User"))
        assert result == ["Hello", " world"]
