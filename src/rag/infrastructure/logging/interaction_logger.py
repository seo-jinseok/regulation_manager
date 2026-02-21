"""
SPEC-RAG-MONITOR-001: RAG Interaction Logger

Central logging for RAG pipeline events with correlation ID tracking
and latency measurement.

This module provides the main interface for logging RAG interactions
with structured output and event emission.
"""

import json
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .events import (
    EventType,
    EventEmitter,
    QueryReceivedEvent,
    QueryRewrittenEvent,
    SearchCompletedEvent,
    RerankingCompletedEvent,
    LLMGenerationStartedEvent,
    TokenGeneratedEvent,
    AnswerGeneratedEvent,
    event_to_dict,
)


class LatencyTimer:
    """Timer for measuring latency.

    Used as a context manager to measure execution time.

    Example:
        >>> with LatencyTimer() as timer:
        ...     time.sleep(0.1)
        >>> print(f"Elapsed: {timer.elapsed_ms}ms")
    """

    def __init__(self, callback: Optional[Callable[[float], None]] = None):
        """Initialize timer.

        Args:
            callback: Optional callback to receive latency in ms on exit.
        """
        self._start_time: Optional[float] = None
        self._elapsed_ms: float = 0.0
        self._callback = callback

    def __enter__(self) -> "LatencyTimer":
        """Start timing."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        """Stop timing and call callback if provided."""
        if self._start_time is not None:
            self._elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        if self._callback is not None:
            self._callback(self._elapsed_ms)

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self._start_time is None:
            return self._elapsed_ms
        return (time.perf_counter() - self._start_time) * 1000


class RAGInteractionLogger:
    """Singleton logger for RAG pipeline interactions.

    Provides methods to log events at each stage of the RAG pipeline
    with automatic correlation ID generation and event emission.

    Example:
        >>> logger = RAGInteractionLogger()
        >>> correlation_id = logger.log_query_received("휴학 신청 방법")
        >>> logger.log_search_completed(results, scores, 45, correlation_id)
    """

    _instance: Optional["RAGInteractionLogger"] = None

    def __new__(cls) -> "RAGInteractionLogger":
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._emitter = EventEmitter()
            cls._instance._logger = logging.getLogger("rag.interaction")
        return cls._instance

    def generate_correlation_id(self) -> str:
        """Generate a unique correlation ID.

        Returns:
            UUID string for tracking requests through the pipeline.
        """
        return str(uuid.uuid4())

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format.

        Returns:
            ISO format timestamp string.
        """
        return datetime.now().isoformat()

    def _emit_event(self, event_type: EventType, event: Any) -> None:
        """Emit an event to subscribers and log it.

        Args:
            event_type: Type of event being emitted.
            event: Event data to emit.
        """
        # Log the event as JSON
        event_dict = event_to_dict(event)
        event_dict["event_type"] = event_type.value
        self._logger.info(json.dumps(event_dict, ensure_ascii=False))

        # Emit to subscribers
        self._emitter.emit(event_type, event)

    def log_query_received(self, query_text: str) -> str:
        """Log that a query was received.

        Args:
            query_text: The query text from the user.

        Returns:
            Correlation ID for tracking this request.
        """
        correlation_id = self.generate_correlation_id()
        event = QueryReceivedEvent(
            timestamp=self._get_timestamp(),
            query_text=query_text,
            correlation_id=correlation_id,
        )
        self._emit_event(EventType.QUERY_RECEIVED, event)
        return correlation_id

    def log_query_rewritten(
        self,
        original_query: str,
        rewritten_query: str,
        strategy: str,
        correlation_id: str,
    ) -> None:
        """Log that query rewriting completed.

        Args:
            original_query: The original query text.
            rewritten_query: The rewritten query text.
            strategy: The rewriting strategy used.
            correlation_id: Correlation ID for this request.
        """
        event = QueryRewrittenEvent(
            timestamp=self._get_timestamp(),
            original_query=original_query,
            rewritten_query=rewritten_query,
            strategy=strategy,
        )
        self._emit_event(EventType.QUERY_REWRITTEN, event)

    def log_search_completed(
        self,
        results: List[str],
        scores: List[float],
        latency_ms: int,
        correlation_id: str,
    ) -> None:
        """Log that vector search completed.

        Args:
            results: List of search result identifiers.
            scores: List of similarity scores.
            latency_ms: Search latency in milliseconds.
            correlation_id: Correlation ID for this request.
        """
        event = SearchCompletedEvent(
            timestamp=self._get_timestamp(),
            results=results,
            scores=scores,
            latency_ms=latency_ms,
        )
        self._emit_event(EventType.SEARCH_COMPLETED, event)

    def log_reranking_completed(
        self,
        reordered_results: List[str],
        score_changes: Dict[str, float],
        latency_ms: int,
        correlation_id: str,
    ) -> None:
        """Log that reranking completed.

        Args:
            reordered_results: Results in their new order.
            score_changes: Dict of result ID to score change.
            latency_ms: Reranking latency in milliseconds.
            correlation_id: Correlation ID for this request.
        """
        event = RerankingCompletedEvent(
            timestamp=self._get_timestamp(),
            reordered_results=reordered_results,
            score_changes=score_changes,
            latency_ms=latency_ms,
        )
        self._emit_event(EventType.RERANKING_COMPLETED, event)

    def log_llm_generation_started(
        self,
        prompt_preview: str,
        model_name: str,
        max_tokens: int,
        correlation_id: str,
    ) -> None:
        """Log that LLM generation started.

        Args:
            prompt_preview: Preview of the prompt sent to LLM.
            model_name: Name of the LLM model.
            max_tokens: Maximum tokens for generation.
            correlation_id: Correlation ID for this request.
        """
        event = LLMGenerationStartedEvent(
            timestamp=self._get_timestamp(),
            prompt_preview=prompt_preview,
            model_name=model_name,
            max_tokens=max_tokens,
        )
        self._emit_event(EventType.LLM_GENERATION_STARTED, event)

    def log_token_generated(
        self,
        token: str,
        accumulated_length: int,
        correlation_id: str,
    ) -> None:
        """Log that a token was generated during streaming.

        Args:
            token: The generated token text.
            accumulated_length: Total tokens generated so far.
            correlation_id: Correlation ID for this request.
        """
        event = TokenGeneratedEvent(
            timestamp=self._get_timestamp(),
            token=token,
            accumulated_length=accumulated_length,
        )
        self._emit_event(EventType.TOKEN_GENERATED, event)

    def log_answer_generated(
        self,
        full_response: str,
        token_count: int,
        latency_ms: int,
        correlation_id: str,
    ) -> None:
        """Log that answer generation completed.

        Args:
            full_response: The complete generated response.
            token_count: Total tokens in the response.
            latency_ms: Total generation latency in milliseconds.
            correlation_id: Correlation ID for this request.
        """
        event = AnswerGeneratedEvent(
            timestamp=self._get_timestamp(),
            full_response=full_response,
            token_count=token_count,
            latency_ms=latency_ms,
        )
        self._emit_event(EventType.ANSWER_GENERATED, event)

    @contextmanager
    def measure_latency(self, callback: Optional[Callable[[float], None]] = None):
        """Context manager for measuring latency.

        Args:
            callback: Optional callback to receive latency in ms on exit.

        Yields:
            LatencyTimer instance for accessing elapsed time.

        Example:
            >>> with logger.measure_latency() as timer:
            ...     # Do work
            ...     pass
            >>> print(f"Took {timer.elapsed_ms}ms")
        """
        timer = LatencyTimer(callback)
        with timer:
            yield timer
