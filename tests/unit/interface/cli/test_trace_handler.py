"""
SPEC-RAG-MONITOR-001: TraceOutputHandler Tests

TDD RED Phase: Tests for CLI trace output handler.
These tests are written BEFORE implementation to drive the design.

Purpose: Verify trace handler correctly subscribes to and formats events.
"""

import time
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from src.rag.infrastructure.logging.events import (
    EventType,
    EventEmitter,
    QueryReceivedEvent,
    QueryRewrittenEvent,
    SearchCompletedEvent,
    RerankingCompletedEvent,
    LLMGenerationStartedEvent,
    TokenGeneratedEvent,
    AnswerGeneratedEvent,
)


class TestTraceOutputHandler:
    """Tests for TraceOutputHandler class."""

    def test_handler_initialization(self):
        """Test handler initializes correctly."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        handler = TraceOutputHandler()
        assert handler is not None

    def test_handler_subscribes_to_all_events(self):
        """Test handler subscribes to all RAG event types."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        emitter = EventEmitter()
        emitter.clear()  # Start fresh

        handler = TraceOutputHandler()
        handler.subscribe()

        # Check all event types have at least one subscriber
        for event_type in EventType:
            assert len(emitter._subscribers[event_type]) >= 1, \
                f"No subscriber for {event_type}"

        # Cleanup
        handler.unsubscribe()

    def test_handler_formats_query_received_event(self):
        """Test handler formats QueryReceived event correctly."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        handler = TraceOutputHandler()
        event = QueryReceivedEvent(
            timestamp="2024-01-01T12:00:00",
            query_text="휴학 신청 방법",
            correlation_id="test-123",
        )

        output = handler.format_event(EventType.QUERY_RECEIVED, event)

        assert "휴학 신청 방법" in output
        # Check for query-related keywords (case-insensitive)
        output_lower = output.lower()
        assert "query" in output_lower or "received" in output_lower or "쿼리" in output or "질문" in output

    def test_handler_formats_query_rewritten_event(self):
        """Test handler formats QueryRewritten event correctly."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        handler = TraceOutputHandler()
        event = QueryRewrittenEvent(
            timestamp="2024-01-01T12:00:01",
            original_query="휴학",
            rewritten_query="대학 휴학 신청 방법 절차",
            strategy="hyde_expansion",
        )

        output = handler.format_event(EventType.QUERY_REWRITTEN, event)

        assert "휴학" in output
        assert "대학 휴학 신청 방법 절차" in output
        assert "rewrite" in output.lower() or "확장" in output or "변환" in output

    def test_handler_formats_search_completed_event(self):
        """Test handler formats SearchCompleted event correctly."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        handler = TraceOutputHandler()
        event = SearchCompletedEvent(
            timestamp="2024-01-01T12:00:02",
            results=["result1", "result2", "result3"],
            scores=[0.95, 0.87, 0.72],
            latency_ms=45,
        )

        output = handler.format_event(EventType.SEARCH_COMPLETED, event)

        assert "3" in output  # Number of results
        assert "45" in output or "0.045" in output  # Latency
        assert "search" in output.lower() or "검색" in output

    def test_handler_formats_reranking_completed_event(self):
        """Test handler formats RerankingCompleted event correctly."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        handler = TraceOutputHandler()
        event = RerankingCompletedEvent(
            timestamp="2024-01-01T12:00:03",
            reordered_results=["result2", "result1"],
            score_changes={"result1": -0.05, "result2": 0.08},
            latency_ms=12,
        )

        output = handler.format_event(EventType.RERANKING_COMPLETED, event)

        assert "12" in output or "0.012" in output  # Latency
        assert "rerank" in output.lower() or "재정렬" in output or "순위" in output

    def test_handler_formats_llm_generation_started_event(self):
        """Test handler formats LLMGenerationStarted event correctly."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        handler = TraceOutputHandler()
        event = LLMGenerationStartedEvent(
            timestamp="2024-01-01T12:00:04",
            prompt_preview="Context: ...",
            model_name="gpt-4o-mini",
            max_tokens=1000,
        )

        output = handler.format_event(EventType.LLM_GENERATION_STARTED, event)

        assert "gpt-4o-mini" in output or "1000" in output
        assert "llm" in output.lower() or "generation" in output or "생성" in output

    def test_handler_formats_token_generated_event(self):
        """Test handler formats TokenGenerated event correctly."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        handler = TraceOutputHandler()
        event = TokenGeneratedEvent(
            timestamp="2024-01-01T12:00:05",
            token="안녕",
            accumulated_length=5,
        )

        output = handler.format_event(EventType.TOKEN_GENERATED, event)

        # Token events might be suppressed or shown minimally
        # Just verify it doesn't crash
        assert output is not None

    def test_handler_formats_answer_generated_event(self):
        """Test handler formats AnswerGenerated event correctly."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        handler = TraceOutputHandler()
        event = AnswerGeneratedEvent(
            timestamp="2024-01-01T12:00:06",
            full_response="휴학 신청은 학기 시작 전까지 가능합니다.",
            token_count=25,
            latency_ms=850,
        )

        output = handler.format_event(EventType.ANSWER_GENERATED, event)

        assert "25" in output or "850" in output  # Token count or latency
        assert "answer" in output.lower() or "답변" in output or "완료" in output

    def test_handler_unsubscribes_correctly(self):
        """Test handler unsubscribes from all events."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        emitter = EventEmitter()
        emitter.clear()

        handler = TraceOutputHandler()
        handler.subscribe()

        # Verify subscribed
        initial_counts = {et: len(emitter._subscribers[et]) for et in EventType}
        assert all(c >= 1 for c in initial_counts.values())

        # Unsubscribe
        handler.unsubscribe()

        # Verify unsubscribed (counts should decrease)
        for event_type in EventType:
            assert len(emitter._subscribers[event_type]) < initial_counts[event_type], \
                f"Failed to unsubscribe from {event_type}"

    def test_handler_uses_rich_for_formatting(self):
        """Test handler uses Rich library when available."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        handler = TraceOutputHandler()
        event = QueryReceivedEvent(
            timestamp="2024-01-01T12:00:00",
            query_text="test query",
            correlation_id="test-123",
        )

        # Should use Rich if available, plain text otherwise
        output = handler.format_event(EventType.QUERY_RECEIVED, event)
        assert isinstance(output, str)

    def test_handler_prints_to_console(self):
        """Test handler can print formatted output to console."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        handler = TraceOutputHandler()
        event = QueryReceivedEvent(
            timestamp="2024-01-01T12:00:00",
            query_text="test query",
            correlation_id="test-123",
        )

        # Should not raise exception
        with patch('sys.stdout', new_callable=StringIO):
            handler.handle_event(EventType.QUERY_RECEIVED, event)

    def test_handler_ignores_token_events_when_disabled(self):
        """Test handler can suppress token-level events for cleaner output."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        handler = TraceOutputHandler(show_tokens=False)
        event = TokenGeneratedEvent(
            timestamp="2024-01-01T12:00:05",
            token="안녕",
            accumulated_length=5,
        )

        output = handler.format_event(EventType.TOKEN_GENERATED, event)

        # Should return empty string or None when tokens are suppressed
        assert output == "" or output is None


class TestTraceOutputHandlerIntegration:
    """Integration tests for TraceOutputHandler with EventEmitter."""

    def test_handler_receives_emitted_events(self):
        """Test handler receives events when emitted."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        emitter = EventEmitter()
        emitter.clear()

        handler = TraceOutputHandler()
        handler.subscribe()

        # Track if handler was called
        received_events = []

        def track_call(event_type, event):
            received_events.append((event_type, event))

        # Monkey patch handle_event to track calls
        original_handle = handler.handle_event
        handler.handle_event = lambda et, e: (track_call(et, e), original_handle(et, e))

        # Emit event
        event = QueryReceivedEvent(
            timestamp="2024-01-01T12:00:00",
            query_text="test",
            correlation_id="123",
        )
        emitter.emit(EventType.QUERY_RECEIVED, event)

        # Should have received the event
        assert len(received_events) == 1
        assert received_events[0][0] == EventType.QUERY_RECEIVED

        # Cleanup
        handler.unsubscribe()

    def test_multiple_handlers_can_subscribe(self):
        """Test multiple handlers can subscribe to same events."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        emitter = EventEmitter()
        emitter.clear()

        handler1 = TraceOutputHandler()
        handler2 = TraceOutputHandler()

        handler1.subscribe()
        handler2.subscribe()

        # Both should be subscribed
        for event_type in EventType:
            assert len(emitter._subscribers[event_type]) >= 2

        # Cleanup
        handler1.unsubscribe()
        handler2.unsubscribe()


class TestTraceOutputHandlerPerformance:
    """Performance tests for TraceOutputHandler."""

    def test_format_event_is_fast(self):
        """Test event formatting is fast (< 5ms per event)."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler

        handler = TraceOutputHandler()
        event = QueryReceivedEvent(
            timestamp="2024-01-01T12:00:00",
            query_text="test query",
            correlation_id="test-123",
        )

        # Measure formatting time
        start = time.perf_counter()
        for _ in range(100):
            handler.format_event(EventType.QUERY_RECEIVED, event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Average time per format should be < 5ms
        avg_ms = elapsed_ms / 100
        assert avg_ms < 5, f"Format took {avg_ms}ms, expected < 5ms"


if __name__ == "__main__":
    # Run tests in verbose mode
    pytest.main([__file__, "-v", "--tb=short"])
