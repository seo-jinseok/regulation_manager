"""
SPEC-RAG-MONITOR-001 Phase 4: Live Monitor Dashboard Backend.

Provides real-time event monitoring and interactive query testing
for the Gradio dashboard.

Key Features:
- Event buffer with max 100 events (FIFO)
- Subscription to EventEmitter singleton
- Event filtering by type
- Thread-safe event buffer management
- Interactive query submission
- Gradio integration support
"""

import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from ...infrastructure.logging.events import (
    EventType,
    EventEmitter,
    RAGEvent,
    event_to_dict,
)

logger = logging.getLogger(__name__)

# Try to import Gradio
try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None


class LiveMonitor:
    """Real-time event monitor for RAG pipeline dashboard.

    Subscribes to RAG events via EventEmitter and maintains a buffer
    for display in the Gradio dashboard. Supports event filtering,
    interactive query submission, and thread-safe operations.

    Attributes:
        max_events: Maximum number of events to buffer (default: 100).
        show_tokens: Whether to capture TOKEN_GENERATED events (default: False).

    Example:
        >>> monitor = LiveMonitor()
        >>> # Events are now being captured
        >>> events = monitor.get_events()
        >>> monitor.cleanup()  # Unsubscribe when done
    """

    def __init__(self, max_events: int = 100, show_tokens: bool = False):
        """Initialize LiveMonitor.

        Args:
            max_events: Maximum number of events to keep in buffer.
            show_tokens: Whether to capture individual token events.
        """
        self.max_events = max_events
        self.show_tokens = show_tokens
        self._buffer: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._callbacks: Dict[EventType, Callable] = {}

        # Subscribe to all event types
        self._subscribe_to_events()

        logger.info(f"LiveMonitor initialized (max_events={max_events}, show_tokens={show_tokens})")

    def _subscribe_to_events(self) -> None:
        """Subscribe to all RAG event types."""
        emitter = EventEmitter()

        for event_type in EventType:
            # Skip token events if not enabled
            if event_type == EventType.TOKEN_GENERATED and not self.show_tokens:
                continue

            # Create callback for this event type
            callback = lambda e, et=event_type: self._on_event(et, e)
            self._callbacks[event_type] = callback
            emitter.subscribe(event_type, callback)

    def _on_event(self, event_type: EventType, event: RAGEvent) -> None:
        """Handle incoming event.

        Args:
            event_type: Type of the event.
            event: Event data.
        """
        try:
            # Convert event to dict
            event_dict = event_to_dict(event)
            event_dict["_event_type"] = event_type.value
            event_dict["_timestamp_iso"] = datetime.now().isoformat()

            # Add to buffer (thread-safe)
            with self._lock:
                self._buffer.append(event_dict)

        except Exception as e:
            logger.error(f"Error capturing event: {e}", exc_info=True)

    def get_events(
        self, event_type: Optional[EventType] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get events from buffer.

        Args:
            event_type: Filter by event type (optional).
            limit: Maximum number of events to return (optional).

        Returns:
            List of event dictionaries (copy of internal buffer).
        """
        with self._lock:
            events = list(self._buffer)

        # Filter by event type if specified
        if event_type:
            events = [e for e in events if e.get("_event_type") == event_type.value]

        # Apply limit if specified
        if limit:
            events = events[-limit:]

        return events

    def get_event_count(self) -> int:
        """Get current number of events in buffer.

        Returns:
            Number of events in buffer.
        """
        with self._lock:
            return len(self._buffer)

    def clear_events(self) -> None:
        """Clear the event buffer."""
        with self._lock:
            self._buffer.clear()
        logger.info("Event buffer cleared")

    def get_events_for_gradio(self) -> List[List[Any]]:
        """Get events formatted for Gradio Dataframe display.

        Returns:
            List of rows for Gradio Dataframe.
        """
        events = self.get_events()

        rows = []
        for event in events:
            event_type = event.get("_event_type", "unknown")
            timestamp = event.get("_timestamp_iso", event.get("timestamp", ""))

            # Format summary based on event type
            if event_type == EventType.QUERY_RECEIVED.value:
                summary = f"📝 Query: {event.get('query_text', '')}"
            elif event_type == EventType.QUERY_REWRITTEN.value:
                summary = f"🔄 Rewrite: {event.get('strategy', '')}"
            elif event_type == EventType.SEARCH_COMPLETED.value:
                summary = f"🔍 Search: {len(event.get('results', []))} results ({event.get('latency_ms', 0)}ms)"
            elif event_type == EventType.RERANKING_COMPLETED.value:
                summary = f"📊 Rerank: {len(event.get('reordered_results', []))} results"
            elif event_type == EventType.LLM_GENERATION_STARTED.value:
                summary = f"🤖 LLM: {event.get('model_name', '')}"
            elif event_type == EventType.ANSWER_GENERATED.value:
                summary = f"✅ Answer: {event.get('token_count', 0)} tokens"
            else:
                summary = str(event)[:50]

            rows.append([timestamp, event_type, summary])

        return rows

    def get_event_stream(self) -> Generator[Dict[str, Any], None, None]:
        """Get streaming generator for real-time event updates.

        Yields:
            Event dictionaries as they arrive.
        """
        last_count = 0

        while True:
            current_events = self.get_events()
            current_count = len(current_events)

            # Yield new events
            if current_count > last_count:
                for event in current_events[last_count:]:
                    yield event
                last_count = current_count

            # Sleep to prevent busy-waiting
            time.sleep(0.1)

    def submit_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Submit a query for interactive testing.

        This method triggers the RAG pipeline and captures all events.

        Args:
            query: Query text to submit.
            **kwargs: Additional query options (top_k, etc.).

        Returns:
            Dictionary containing query result and captured events.
        """
        from ...interface.query_handler import QueryHandler, QueryContext, QueryOptions
        from ...infrastructure.chroma_store import ChromaVectorStore
        from ...infrastructure.llm_adapter import LLMClientAdapter

        logger.info(f"Submitting query: {query}")

        # Clear previous events for clean capture
        self.clear_events()

        try:
            # Initialize components
            store = ChromaVectorStore(persist_directory="data/chroma_db")

            # Try to initialize LLM
            llm_client = None
            try:
                llm_client = LLMClientAdapter()
            except Exception:
                logger.warning("LLM not available, using search only")

            handler = QueryHandler(
                store=store,
                llm_client=llm_client,
                use_reranker=True,
            )

            context = QueryContext(
                state={},
                interactive=True,
            )

            options = QueryOptions(
                top_k=kwargs.get("top_k", 5),
                use_rerank=True,
                **kwargs
            )

            # Process query
            result = handler.process_query(query, context, options)

            # Collect captured events
            events = self.get_events()

            return {
                "query": query,
                "result": result.content,
                "result_type": result.type.value if hasattr(result.type, "value") else str(result.type),
                "success": result.success,
                "events": events,
                "event_count": len(events),
            }

        except Exception as e:
            logger.error(f"Query submission failed: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "success": False,
                "events": self.get_events(),
            }

    def cleanup(self) -> None:
        """Unsubscribe from events and cleanup resources."""
        emitter = EventEmitter()

        for event_type, callback in self._callbacks.items():
            try:
                emitter.unsubscribe(event_type, callback)
            except Exception as e:
                logger.warning(f"Error unsubscribing from {event_type}: {e}")

        self._callbacks.clear()
        self._buffer.clear()
        logger.info("LiveMonitor cleanup complete")


def create_monitor_tab(
    db_path: str = "data/chroma_db",
) -> Tuple[LiveMonitor, Dict[str, Any]]:
    """Create Live Monitor tab for Gradio dashboard.

    Args:
        db_path: Path to ChromaDB storage.

    Returns:
        Tuple of (LiveMonitor instance, dict of Gradio components).
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is required. Install with: uv add gradio")

    # Initialize monitor
    monitor = LiveMonitor()

    components = {}

    with gr.TabItem("📡 실시간 모니터"):
        gr.Markdown("### RAG 파이프라인 실시간 모니터링")

        with gr.Row():
            # Left column: Event timeline
            with gr.Column(scale=2):
                gr.Markdown("#### 📊 이벤트 타임라인")

                # Auto-refresh checkbox
                auto_refresh = gr.Checkbox(
                    label="자동 새로고침 (5초)",
                    value=True,
                )

                # Event type filter
                event_filter = gr.Dropdown(
                    choices=["전체"] + [et.value for et in EventType],
                    value="전체",
                    label="이벤트 유형 필터",
                )

                # Event display (Dataframe)
                event_display = gr.Dataframe(
                    headers=["시간", "유형", "요약"],
                    datatype=["str", "str", "str"],
                    value=[],
                    label="이벤트",
                    interactive=False,
                    wrap=True,
                )

                # Manual refresh button
                refresh_btn = gr.Button("🔄 새로고침", variant="secondary")

            # Right column: Query testing
            with gr.Column(scale=1):
                gr.Markdown("#### 🧪 쿼리 테스트")

                query_input = gr.Textbox(
                    label="테스트 쿼리",
                    placeholder="질문을 입력하세요...",
                    lines=2,
                )

                query_top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="결과 수",
                )

                submit_btn = gr.Button("▶ 실행", variant="primary")

                # Result display
                result_display = gr.Markdown(
                    value="쿼리를 실행하면 결과가 여기에 표시됩니다.",
                    label="결과",
                )

        components["event_display"] = event_display
        components["query_input"] = query_input
        components["result_display"] = result_display
        components["event_filter"] = event_filter
        components["auto_refresh"] = auto_refresh
        components["refresh_btn"] = refresh_btn
        components["submit_btn"] = submit_btn
        components["query_top_k"] = query_top_k

    # Event handlers
    def refresh_events(filter_type: str):
        """Refresh event display."""
        if filter_type == "전체":
            events = monitor.get_events_for_gradio()
        else:
            event_type = None
            for et in EventType:
                if et.value == filter_type:
                    event_type = et
                    break
            events = monitor.get_events_for_gradio()

        return events

    def run_query(query: str, top_k: int):
        """Run test query and return results."""
        if not query.strip():
            return "쿼리를 입력해주세요."

        result = monitor.submit_query(query, top_k=top_k)

        if not result.get("success"):
            return f"❌ 오류: {result.get('error', 'Unknown error')}"

        # Format result
        output = f"### 결과\n\n"
        output += f"**쿼리**: {result['query']}\n\n"
        output += f"**응답 유형**: {result.get('result_type', 'unknown')}\n\n"
        output += f"**캡처된 이벤트**: {result.get('event_count', 0)}개\n\n"

        if result.get("result"):
            output += f"---\n\n{result['result'][:500]}"

        return output

    # Wire up event handlers
    refresh_btn.click(
        fn=refresh_events,
        inputs=[event_filter],
        outputs=[event_display],
    )

    submit_btn.click(
        fn=run_query,
        inputs=[query_input, query_top_k],
        outputs=[result_display],
    )

    return monitor, components
