"""
Unit tests for Circuit Breaker pattern.

Tests characterize and verify circuit breaker behavior for LLM provider resilience.
"""

import time
from unittest.mock import MagicMock

import pytest

from src.rag.domain.llm import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
)


class TestCircuitBreakerInit:
    """Test circuit breaker initialization."""

    def test_init_with_defaults(self):
        """Characterize: Default configuration values."""
        breaker = CircuitBreaker("test_provider")

        assert breaker.provider_name == "test_provider"
        assert isinstance(breaker.config, CircuitBreakerConfig)
        assert breaker.config.failure_threshold == 3
        assert breaker.config.recovery_timeout == 60.0
        assert breaker.config.success_threshold == 2
        assert breaker.config.timeout == 30.0

    def test_init_with_custom_config(self):
        """Characterize: Custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=120.0,
            success_threshold=3,
            timeout=45.0,
        )
        breaker = CircuitBreaker("test_provider", config)

        assert breaker.config.failure_threshold == 5
        assert breaker.config.recovery_timeout == 120.0
        assert breaker.config.success_threshold == 3
        assert breaker.config.timeout == 45.0

    def test_init_metrics_state(self):
        """Characterize: Initial metrics state."""
        breaker = CircuitBreaker("test_provider")

        assert breaker.metrics.state == CircuitState.CLOSED
        assert breaker.metrics.failure_count == 0
        assert breaker.metrics.success_count == 0
        assert breaker.metrics.last_failure_time is None
        assert breaker.metrics.total_requests == 0
        assert breaker.metrics.total_failures == 0
        assert breaker.metrics.total_successes == 0


class TestCircuitBreakerClosedState:
    """Test circuit breaker behavior in CLOSED state."""

    def test_successful_call_in_closed_state(self):
        """Characterize: Success increments counters and maintains CLOSED state."""
        breaker = CircuitBreaker("test_provider")
        mock_func = MagicMock(return_value="result")

        result = breaker.call(mock_func)

        assert result == "result"
        mock_func.assert_called_once()
        assert breaker.metrics.state == CircuitState.CLOSED
        assert breaker.metrics.success_count == 1
        assert breaker.metrics.total_successes == 1
        assert breaker.metrics.total_requests == 1
        assert breaker.metrics.failure_count == 0

    def test_failure_in_closed_state_increments_counter(self):
        """Characterize: Single failure increments failure counter."""
        breaker = CircuitBreaker(
            "test_provider", CircuitBreakerConfig(failure_threshold=3)
        )
        mock_func = MagicMock(side_effect=Exception("Test error"))

        with pytest.raises(Exception, match="Test error"):
            breaker.call(mock_func)

        assert breaker.metrics.state == CircuitState.CLOSED
        assert breaker.metrics.failure_count == 1
        assert breaker.metrics.total_failures == 1
        assert breaker.metrics.last_failure_time is not None

    def test_failure_threshold_opens_circuit(self):
        """Characterize: Consecutive failures at threshold open circuit."""
        breaker = CircuitBreaker(
            "test_provider", CircuitBreakerConfig(failure_threshold=2)
        )
        mock_func = MagicMock(side_effect=Exception("Test error"))

        # First failure
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func)
        assert breaker.metrics.state == CircuitState.CLOSED

        # Second failure - should open circuit
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func)
        assert breaker.metrics.state == CircuitState.OPEN

    def test_success_resets_failure_counter(self):
        """Characterize: Success resets failure counter in CLOSED state."""
        breaker = CircuitBreaker(
            "test_provider", CircuitBreakerConfig(failure_threshold=3)
        )
        mock_func_fail = MagicMock(side_effect=Exception("Test error"))
        mock_func_success = MagicMock(return_value="success")

        # Two failures
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)
        assert breaker.metrics.failure_count == 2

        # Success resets counter
        breaker.call(mock_func_success)
        assert breaker.metrics.failure_count == 0
        assert breaker.metrics.state == CircuitState.CLOSED


class TestCircuitBreakerOpenState:
    """Test circuit breaker behavior in OPEN state."""

    def test_open_state_rejects_requests(self):
        """Characterize: OPEN state immediately rejects requests with error."""
        breaker = CircuitBreaker(
            "test_provider",
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout=10.0),
        )
        mock_func = MagicMock(side_effect=Exception("Test error"))

        # Trigger open circuit
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func)
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func)

        assert breaker.metrics.state == CircuitState.OPEN

        # Request should be rejected
        mock_func.reset_mock()
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(mock_func)

        # Original function should NOT be called
        mock_func.assert_not_called()

    def test_open_error_contains_provider_info(self):
        """Characterize: CircuitBreakerOpenError contains provider and timeout info."""
        breaker = CircuitBreaker(
            "test_provider",
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout=10.0),
        )
        mock_func = MagicMock(side_effect=Exception("Test error"))

        # Trigger open circuit
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func)
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func)

        # Check error message
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            breaker.call(mock_func)

        error = exc_info.value
        assert "test_provider" in str(error)
        assert error.provider_name == "test_provider"
        assert error.recovery_timeout > 0


class TestCircuitBreakerHalfOpenState:
    """Test circuit breaker behavior in HALF_OPEN state."""

    def test_recovery_timeout_transitions_to_half_open_then_back_to_open(self):
        """Characterize: HALF_OPEN failure immediately reopens circuit."""
        breaker = CircuitBreaker(
            "test_provider",
            CircuitBreakerConfig(
                failure_threshold=2, recovery_timeout=0.1, success_threshold=2
            ),
        )
        mock_func_fail = MagicMock(side_effect=Exception("Test error"))

        # Trigger open circuit
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)

        assert breaker.metrics.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Next request transitions to HALF_OPEN but fails, reopening circuit
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)

        # Should be back to OPEN since HALF_OPEN request failed
        assert breaker.metrics.state == CircuitState.OPEN

    def test_half_open_success_closes_circuit(self):
        """Characterize: Success threshold in HALF_OPEN closes circuit."""
        breaker = CircuitBreaker(
            "test_provider",
            CircuitBreakerConfig(
                failure_threshold=2, recovery_timeout=0.1, success_threshold=2
            ),
        )
        mock_func_fail = MagicMock(side_effect=Exception("Test error"))
        mock_func_success = MagicMock(return_value="success")

        # Open circuit
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)

        # Wait for recovery timeout
        time.sleep(0.15)

        # First success in HALF_OPEN
        result = breaker.call(mock_func_success)
        assert result == "success"
        assert breaker.metrics.state == CircuitState.HALF_OPEN

        # Second success closes circuit
        result = breaker.call(mock_func_success)
        assert result == "success"
        assert breaker.metrics.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self):
        """Characterize: Failure in HALF_OPEN reopens circuit."""
        breaker = CircuitBreaker(
            "test_provider",
            CircuitBreakerConfig(
                failure_threshold=2, recovery_timeout=0.1, success_threshold=2
            ),
        )
        mock_func_fail = MagicMock(side_effect=Exception("Test error"))
        mock_func_success = MagicMock(return_value="success")

        # Open circuit
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)

        # Wait for recovery timeout
        time.sleep(0.15)

        # One success in HALF_OPEN
        breaker.call(mock_func_success)
        assert breaker.metrics.state == CircuitState.HALF_OPEN

        # Failure reopens circuit
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)
        assert breaker.metrics.state == CircuitState.OPEN


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics tracking."""

    def test_metrics_tracking_over_time(self):
        """Characterize: Metrics accurately track circuit state over time."""
        breaker = CircuitBreaker(
            "test_provider",
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=0.1),
        )
        mock_func_fail = MagicMock(side_effect=Exception("Test error"))
        mock_func_success = MagicMock(return_value="success")

        # Initial state
        metrics = breaker.get_metrics()
        assert metrics.total_requests == 0
        assert metrics.total_failures == 0
        assert metrics.total_successes == 0

        # Some successes
        breaker.call(mock_func_success)
        breaker.call(mock_func_success)
        metrics = breaker.get_metrics()
        assert metrics.total_requests == 2
        assert metrics.total_successes == 2
        assert metrics.total_failures == 0

        # Some failures
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)
        metrics = breaker.get_metrics()
        assert metrics.total_requests == 4
        assert metrics.total_successes == 2
        assert metrics.total_failures == 2

    def test_get_state_returns_current_state(self):
        """Characterize: get_state() returns current circuit state."""
        breaker = CircuitBreaker("test_provider")
        mock_func = MagicMock(side_effect=Exception("Test error"))

        assert breaker.get_state() == CircuitState.CLOSED

        # Trigger failures to open circuit
        for _ in range(3):
            try:
                breaker.call(mock_func)
            except Exception:
                pass

        assert breaker.get_state() == CircuitState.OPEN


class TestCircuitBreakerControlMethods:
    """Test circuit breaker control methods."""

    def test_reset_resets_to_initial_state(self):
        """Characterize: reset() resets circuit to initial CLOSED state."""
        breaker = CircuitBreaker("test_provider")
        mock_func = MagicMock(side_effect=Exception("Test error"))

        # Trigger failures to open circuit
        for _ in range(3):
            try:
                breaker.call(mock_func)
            except Exception:
                pass

        assert breaker.metrics.state == CircuitState.OPEN
        assert breaker.metrics.failure_count == 3

        # Reset
        breaker.reset()

        assert breaker.metrics.state == CircuitState.CLOSED
        assert breaker.metrics.failure_count == 0
        assert breaker.metrics.total_requests == 0
        assert breaker.metrics.total_failures == 0

    def test_force_open_opens_circuit(self):
        """Characterize: force_open() immediately opens circuit."""
        breaker = CircuitBreaker("test_provider")

        assert breaker.metrics.state == CircuitState.CLOSED

        breaker.force_open()

        assert breaker.metrics.state == CircuitState.OPEN

        # Requests should be rejected
        mock_func = MagicMock(return_value="result")
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(mock_func)
        mock_func.assert_not_called()

    def test_force_close_closes_circuit(self):
        """Characterize: force_close() closes circuit and resets counters."""
        breaker = CircuitBreaker("test_provider")
        mock_func = MagicMock(side_effect=Exception("Test error"))

        # Open circuit
        for _ in range(3):
            try:
                breaker.call(mock_func)
            except Exception:
                pass

        assert breaker.metrics.state == CircuitState.OPEN
        assert breaker.metrics.failure_count == 3

        # Force close
        breaker.force_close()

        assert breaker.metrics.state == CircuitState.CLOSED
        assert breaker.metrics.failure_count == 0

        # Mock should still raise exceptions (circuit is closed but will track failures)
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func)

        # Now change mock to return success and verify it works
        mock_func.side_effect = None
        mock_func.return_value = "result"
        result = breaker.call(mock_func)
        assert result == "result"


class TestCircuitBreakerIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_fallback_provider_pattern(self):
        """Characterize: Fallback provider usage when circuit opens."""
        # Primary provider breaker
        primary_breaker = CircuitBreaker(
            "primary",
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1),
        )
        primary_func = MagicMock(side_effect=Exception("Primary error"))

        # Fallback provider
        fallback_func = MagicMock(return_value="fallback result")

        # Primary failures open circuit
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            primary_breaker.call(primary_func)
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            primary_breaker.call(primary_func)

        # Circuit should be open, use fallback
        with pytest.raises(CircuitBreakerOpenError):
            primary_breaker.call(primary_func)

        result = fallback_func()
        assert result == "fallback result"

    def test_automatic_recovery_after_timeout(self):
        """Characterize: Automatic recovery after provider comes back online."""
        breaker = CircuitBreaker(
            "test_provider",
            CircuitBreakerConfig(
                failure_threshold=2, recovery_timeout=0.1, success_threshold=1
            ),
        )
        mock_func_fail = MagicMock(side_effect=Exception("Test error"))
        mock_func_success = MagicMock(return_value="success")

        # Fail and open circuit
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)
        with pytest.raises(Exception):  # noqa: B017 - Intentional for characterization
            breaker.call(mock_func_fail)

        assert breaker.get_state() == CircuitState.OPEN

        # Wait for recovery
        time.sleep(0.15)

        # Provider recovers
        result = breaker.call(mock_func_success)
        assert result == "success"
        assert breaker.get_state() == CircuitState.CLOSED
