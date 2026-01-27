"""
Circuit Breaker Pattern for LLM Provider Resilience.

Implements circuit breaker pattern to prevent cascading failures and
provide automatic failover for LLM provider connections.

States:
- CLOSED (Normal): Requests pass through, failure counter resets on success
- OPEN (Failed): Requests immediately rejected, timeout before half-open transition
- HALF_OPEN (Recovering): Single test request allowed, success closes circuit
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failed, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 3  # Consecutive failures to open circuit
    recovery_timeout: float = 60.0  # Seconds before half-open transition
    success_threshold: int = 2  # Consecutive successes to close circuit
    timeout: float = 30.0  # Request timeout in seconds


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreakerOpenError(Exception):
    """Raised when circuit is open and request is rejected."""

    def __init__(self, provider_name: str, recovery_timeout: float):
        self.provider_name = provider_name
        self.recovery_timeout = recovery_timeout
        super().__init__(
            f"Circuit is OPEN for provider '{provider_name}'. "
            f"Retry after {recovery_timeout:.1f} seconds."
        )


class CircuitBreaker:
    """
    Circuit breaker for LLM provider resilience.

    Prevents cascading failures by:
    1. Tracking consecutive failures
    2. Opening circuit after threshold
    3. Allowing recovery testing
    4. Closing circuit on successful recovery

    Usage:
        breaker = CircuitBreaker("openai", config)

        try:
            result = breaker.call(llm_client.generate, prompt, system_msg)
        except CircuitBreakerOpenError:
            # Use fallback provider
            result = fallback_client.generate(prompt, system_msg)
    """

    def __init__(
        self,
        provider_name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            provider_name: Name of the LLM provider (e.g., "openai", "ollama")
            config: Circuit breaker configuration
        """
        self.provider_name = provider_name
        self.config = config or CircuitBreakerConfig()
        self.metrics = CircuitBreakerMetrics()
        self._half_open_success_count = 0

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to call (e.g., llm_client.generate)
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: If function call fails (and records failure)
        """
        self.metrics.total_requests += 1

        # Check if circuit should transition to HALF_OPEN
        self._check_recovery_timeout()

        # Reject if circuit is OPEN
        if self.metrics.state == CircuitState.OPEN:
            logger.warning(
                f"Circuit OPEN for {self.provider_name}, rejecting request. "
                f"Recovery in {self._get_recovery_time():.1f}s"
            )
            raise CircuitBreakerOpenError(
                self.provider_name,
                self.config.recovery_timeout
                - (time.time() - self.metrics.last_failure_time),
            )

        # Execute request
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _check_recovery_timeout(self) -> None:
        """Transition to HALF_OPEN if recovery timeout has passed."""
        if (
            self.metrics.state == CircuitState.OPEN
            and self.metrics.last_failure_time is not None
        ):
            elapsed = time.time() - self.metrics.last_failure_time
            if elapsed >= self.config.recovery_timeout:
                logger.info(
                    f"Recovery timeout reached for {self.provider_name}, "
                    f"transitioning to HALF_OPEN"
                )
                self._set_state(CircuitState.HALF_OPEN)
                self._half_open_success_count = 0

    def _on_success(self) -> None:
        """Handle successful request."""
        self.metrics.total_successes += 1

        if self.metrics.state == CircuitState.HALF_OPEN:
            self._half_open_success_count += 1
            logger.info(
                f"Success in HALF_OPEN for {self.provider_name} "
                f"({self._half_open_success_count}/{self.config.success_threshold})"
            )

            if self._half_open_success_count >= self.config.success_threshold:
                logger.info(
                    f"Success threshold reached for {self.provider_name}, "
                    f"closing circuit"
                )
                self._set_state(CircuitState.CLOSED)
                self.metrics.failure_count = 0
                self._half_open_success_count = 0
        elif self.metrics.state == CircuitState.CLOSED:
            self.metrics.failure_count = 0
            self.metrics.success_count += 1

    def _on_failure(self) -> None:
        """Handle failed request."""
        self.metrics.total_failures += 1
        self.metrics.failure_count += 1
        self.metrics.last_failure_time = time.time()

        logger.warning(
            f"Failure for {self.provider_name} "
            f"(count: {self.metrics.failure_count}/{self.config.failure_threshold})"
        )

        if self.metrics.state == CircuitState.HALF_OPEN:
            # Failed in HALF_OPEN, reopen circuit
            logger.warning(
                f"Failure in HALF_OPEN for {self.provider_name}, reopening circuit"
            )
            self._set_state(CircuitState.OPEN)
            self._half_open_success_count = 0
        elif self.metrics.state == CircuitState.CLOSED:
            # Check if failure threshold reached
            if self.metrics.failure_count >= self.config.failure_threshold:
                logger.error(
                    f"Failure threshold reached for {self.provider_name}, "
                    f"opening circuit"
                )
                self._set_state(CircuitState.OPEN)

    def _set_state(self, new_state: CircuitState) -> None:
        """Update circuit state with logging."""
        old_state = self.metrics.state
        self.metrics.state = new_state
        self.metrics.last_state_change = time.time()

        logger.info(
            f"Circuit breaker state transition for {self.provider_name}: "
            f"{old_state.value} -> {new_state.value}"
        )

    def _get_recovery_time(self) -> float:
        """Get time remaining until recovery."""
        if (
            self.metrics.state != CircuitState.OPEN
            or self.metrics.last_failure_time is None
        ):
            return 0.0

        elapsed = time.time() - self.metrics.last_failure_time
        remaining = self.config.recovery_timeout - elapsed
        return max(0.0, remaining)

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self.metrics.state

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self.metrics

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        logger.info(f"Resetting circuit breaker for {self.provider_name}")
        self.metrics = CircuitBreakerMetrics()
        self._half_open_success_count = 0

    def force_open(self) -> None:
        """Force circuit open (for testing)."""
        logger.warning(f"Forcing circuit OPEN for {self.provider_name}")
        self._set_state(CircuitState.OPEN)
        # Set last_failure_time to enable recovery timeout calculation
        if self.metrics.last_failure_time is None:
            self.metrics.last_failure_time = time.time()

    def force_close(self) -> None:
        """Force circuit closed (for testing)."""
        logger.info(f"Forcing circuit CLOSED for {self.provider_name}")
        self._set_state(CircuitState.CLOSED)
        self.metrics.failure_count = 0
