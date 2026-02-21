"""
SPEC-RAG-MONITOR-001: Event System

Core event types and EventEmitter for RAG monitoring.

This module provides the event infrastructure for real-time monitoring
of the RAG pipeline execution.
"""

import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Module logger for error reporting
_logger = logging.getLogger("rag.monitoring.events")


class EventType(Enum):
    """Event types for RAG pipeline monitoring.

    Each event type corresponds to a specific stage in the RAG pipeline.
    """

    QUERY_RECEIVED = "query_received"
    QUERY_REWRITTEN = "query_rewritten"
    SEARCH_COMPLETED = "search_completed"
    RERANKING_COMPLETED = "reranking_completed"
    LLM_GENERATION_STARTED = "llm_generation_started"
    TOKEN_GENERATED = "token_generated"
    ANSWER_GENERATED = "answer_generated"


@dataclass
class QueryReceivedEvent:
    """Event emitted when a query is received.

    Attributes:
        timestamp: ISO format timestamp when query was received.
        query_text: The original query text from the user.
        correlation_id: Unique identifier to trace this request through the pipeline.
    """

    timestamp: str
    query_text: str
    correlation_id: str


@dataclass
class QueryRewrittenEvent:
    """Event emitted when query rewriting completes.

    Attributes:
        timestamp: ISO format timestamp when rewriting completed.
        original_query: The original query text.
        rewritten_query: The rewritten/expanded query text.
        strategy: The rewriting strategy used (e.g., "hyde_expansion", "synonym_expansion").
    """

    timestamp: str
    original_query: str
    rewritten_query: str
    strategy: str


@dataclass
class SearchCompletedEvent:
    """Event emitted when vector search completes.

    Attributes:
        timestamp: ISO format timestamp when search completed.
        results: List of search result identifiers or previews.
        scores: List of similarity scores corresponding to results.
        latency_ms: Search latency in milliseconds.
    """

    timestamp: str
    results: List[str]
    scores: List[float]
    latency_ms: int


@dataclass
class RerankingCompletedEvent:
    """Event emitted when reranking completes.

    Attributes:
        timestamp: ISO format timestamp when reranking completed.
        reordered_results: List of result identifiers in new order.
        score_changes: Dict mapping result IDs to their score changes.
        latency_ms: Reranking latency in milliseconds.
    """

    timestamp: str
    reordered_results: List[str]
    score_changes: Dict[str, float]
    latency_ms: int


@dataclass
class LLMGenerationStartedEvent:
    """Event emitted when LLM generation starts.

    Attributes:
        timestamp: ISO format timestamp when generation started.
        prompt_preview: Preview of the prompt sent to LLM (may be truncated).
        model_name: Name of the LLM model being used.
        max_tokens: Maximum tokens requested for generation.
    """

    timestamp: str
    prompt_preview: str
    model_name: str
    max_tokens: int


@dataclass
class TokenGeneratedEvent:
    """Event emitted for each token during streaming generation.

    Attributes:
        timestamp: ISO format timestamp when token was generated.
        token: The generated token text.
        accumulated_length: Total number of tokens generated so far.
    """

    timestamp: str
    token: str
    accumulated_length: int


@dataclass
class AnswerGeneratedEvent:
    """Event emitted when answer generation completes.

    Attributes:
        timestamp: ISO format timestamp when generation completed.
        full_response: The complete generated response.
        token_count: Total number of tokens in the response.
        latency_ms: Total generation latency in milliseconds.
    """

    timestamp: str
    full_response: str
    token_count: int
    latency_ms: int


# Union type for all events
RAGEvent = (
    QueryReceivedEvent
    | QueryRewrittenEvent
    | SearchCompletedEvent
    | RerankingCompletedEvent
    | LLMGenerationStartedEvent
    | TokenGeneratedEvent
    | AnswerGeneratedEvent
)


class EventEmitter:
    """Singleton event emitter for RAG pipeline events.

    Implements the observer pattern to allow multiple subscribers
    to receive events when they are emitted.

    Example:
        >>> emitter = EventEmitter()
        >>> def handler(event):
        ...     print(f"Received: {event}")
        >>> emitter.subscribe(EventType.QUERY_RECEIVED, handler)
        >>> emitter.emit(EventType.QUERY_RECEIVED, event)
    """

    _instance: Optional["EventEmitter"] = None

    def __new__(cls) -> "EventEmitter":
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers: Dict[EventType, List[Callable]] = {
                event_type: [] for event_type in EventType
            }
        return cls._instance

    def subscribe(self, event_type: EventType, callback: Callable[[RAGEvent], None]) -> None:
        """Subscribe to an event type.

        Args:
            event_type: The type of event to subscribe to.
            callback: Function to call when event is emitted.
        """
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable[[RAGEvent], None]) -> None:
        """Unsubscribe from an event type.

        Args:
            event_type: The type of event to unsubscribe from.
            callback: The callback to remove.
        """
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)

    def emit(self, event_type: EventType, event: RAGEvent) -> None:
        """Emit an event to all subscribers.

        Args:
            event_type: The type of event being emitted.
            event: The event data to send to subscribers.
        """
        for callback in self._subscribers[event_type]:
            try:
                callback(event)
            except Exception as e:
                # Log callback errors but continue to other subscribers
                _logger.warning(
                    f"Event callback failed for {event_type.value}: {e}",
                    exc_info=True,
                )

    def clear(self) -> None:
        """Clear all subscribers."""
        for event_type in EventType:
            self._subscribers[event_type] = []


def event_to_dict(event: RAGEvent) -> Dict[str, Any]:
    """Convert an event to a dictionary for serialization.

    Args:
        event: The event to convert.

    Returns:
        Dictionary representation of the event.
    """
    return asdict(event)
