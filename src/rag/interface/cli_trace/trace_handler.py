"""
SPEC-RAG-MONITOR-001: Trace Output Handler

Real-time CLI trace output for RAG pipeline events.

This module provides the TraceOutputHandler class that subscribes to
RAG events and formats them for display in the CLI.
"""

import logging
from typing import Optional

from ...infrastructure.logging.events import (
    EventType,
    EventEmitter,
    QueryReceivedEvent,
    QueryRewrittenEvent,
    SearchCompletedEvent,
    RerankingCompletedEvent,
    LLMGenerationStartedEvent,
    TokenGeneratedEvent,
    AnswerGeneratedEvent,
    RAGEvent,
)

# Rich for pretty output (optional)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

logger = logging.getLogger(__name__)


class TraceOutputHandler:
    """Handler for displaying RAG pipeline events in real-time.

    Subscribes to all RAG event types and formats them for CLI display.
    Supports both Rich-formatted output (when available) and plain text.

    Attributes:
        show_tokens: Whether to show individual token events (default: False).

    Example:
        >>> handler = TraceOutputHandler(show_tokens=False)
        >>> handler.subscribe()
        >>> # Events are now displayed in real-time
        >>> handler.unsubscribe()
    """

    def __init__(self, show_tokens: bool = False):
        """Initialize trace output handler.

        Args:
            show_tokens: Whether to show individual token events.
                        Set to False for cleaner output.
        """
        self.show_tokens = show_tokens
        self._emitter = EventEmitter()
        self._callbacks = {}  # Store callbacks for unsubscription

    def subscribe(self) -> None:
        """Subscribe to all RAG event types."""
        # Create callbacks for each event type
        for event_type in EventType:
            callback = lambda e, et=event_type: self.handle_event(et, e)
            self._callbacks[event_type] = callback
            self._emitter.subscribe(event_type, callback)

    def unsubscribe(self) -> None:
        """Unsubscribe from all RAG event types."""
        for event_type, callback in self._callbacks.items():
            self._emitter.unsubscribe(event_type, callback)
        self._callbacks.clear()

    def handle_event(self, event_type: EventType, event: RAGEvent) -> None:
        """Handle an emitted event.

        Args:
            event_type: Type of the event.
            event: Event data.
        """
        formatted = self.format_event(event_type, event)
        if formatted:
            self._print(formatted)

    def format_event(self, event_type: EventType, event: RAGEvent) -> Optional[str]:
        """Format an event for display.

        Args:
            event_type: Type of the event.
            event: Event data.

        Returns:
            Formatted string for display, or None/empty to skip.
        """
        # Skip token events if disabled
        if event_type == EventType.TOKEN_GENERATED and not self.show_tokens:
            return ""

        # Format based on event type
        if event_type == EventType.QUERY_RECEIVED:
            return self._format_query_received(event)
        elif event_type == EventType.QUERY_REWRITTEN:
            return self._format_query_rewritten(event)
        elif event_type == EventType.SEARCH_COMPLETED:
            return self._format_search_completed(event)
        elif event_type == EventType.RERANKING_COMPLETED:
            return self._format_reranking_completed(event)
        elif event_type == EventType.LLM_GENERATION_STARTED:
            return self._format_llm_generation_started(event)
        elif event_type == EventType.TOKEN_GENERATED:
            return self._format_token_generated(event)
        elif event_type == EventType.ANSWER_GENERATED:
            return self._format_answer_generated(event)
        else:
            return str(event)

    def _format_query_received(self, event: QueryReceivedEvent) -> str:
        """Format QueryReceived event."""
        if RICH_AVAILABLE:
            return f"[bold cyan]📝 Query Received:[/bold cyan] {event.query_text}"
        return f"[Query] {event.query_text}"

    def _format_query_rewritten(self, event: QueryRewrittenEvent) -> str:
        """Format QueryRewritten event."""
        if RICH_AVAILABLE:
            return (
                f"[yellow]🔄 Query Rewrite ({event.strategy}):[/yellow]\n"
                f"  Original: {event.original_query}\n"
                f"  Expanded: {event.rewritten_query}"
            )
        return f"[Rewrite] {event.original_query} -> {event.rewritten_query}"

    def _format_search_completed(self, event: SearchCompletedEvent) -> str:
        """Format SearchCompleted event."""
        if RICH_AVAILABLE:
            return (
                f"[green]🔍 Search Completed:[/green] "
                f"{len(event.results)} results in {event.latency_ms}ms"
            )
        return f"[Search] {len(event.results)} results ({event.latency_ms}ms)"

    def _format_reranking_completed(self, event: RerankingCompletedEvent) -> str:
        """Format RerankingCompleted event."""
        if RICH_AVAILABLE:
            return (
                f"[magenta]📊 Reranking Completed:[/magenta] "
                f"{len(event.reordered_results)} results reordered in {event.latency_ms}ms"
            )
        return f"[Rerank] {len(event.reordered_results)} results ({event.latency_ms}ms)"

    def _format_llm_generation_started(self, event: LLMGenerationStartedEvent) -> str:
        """Format LLMGenerationStarted event."""
        if RICH_AVAILABLE:
            return (
                f"[blue]🤖 LLM Generation Started:[/blue] "
                f"{event.model_name} (max {event.max_tokens} tokens)"
            )
        return f"[LLM] {event.model_name} starting generation"

    def _format_token_generated(self, event: TokenGeneratedEvent) -> str:
        """Format TokenGenerated event."""
        # Token events are typically suppressed for cleaner output
        if RICH_AVAILABLE:
            return f"[dim]{event.token}[/dim]"
        return event.token

    def _format_answer_generated(self, event: AnswerGeneratedEvent) -> str:
        """Format AnswerGenerated event."""
        if RICH_AVAILABLE:
            return (
                f"[bold green]✅ Answer Generated:[/bold green] "
                f"{event.token_count} tokens in {event.latency_ms}ms"
            )
        return f"[Answer] {event.token_count} tokens ({event.latency_ms}ms)"

    def _print(self, message: str) -> None:
        """Print formatted message to console.

        Args:
            message: Formatted message to print.
        """
        if RICH_AVAILABLE:
            console.print(message)
        else:
            # Strip Rich formatting for plain output
            import re
            clean = re.sub(r'\[/?[^\]]+\]', '', message)
            print(clean)
