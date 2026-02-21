"""
Tests for LiveMonitor class (SPEC-RAG-MONITOR-001 Phase 4).

TDD approach: Tests define expected behavior before implementation.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.rag.infrastructure.logging.events import (
    EventType,
    EventEmitter,
    QueryReceivedEvent,
    SearchCompletedEvent,
    AnswerGeneratedEvent,
    event_to_dict,
)


class TestLiveMonitor:
    """Test suite for LiveMonitor class."""

    def test_live_monitor_initialization(self):
        """LiveMonitor should initialize with empty event buffer."""
        from src.rag.interface.web.live_monitor import LiveMonitor

        monitor = LiveMonitor()
        assert monitor.get_events() == []
        assert monitor.get_event_count() == 0

    def test_live_monitor_subscribes_to_event_emitter(self):
        """LiveMonitor should subscribe to EventEmitter on initialization."""
        from src.rag.interface.web.live_monitor import LiveMonitor

        emitter = EventEmitter()
        initial_subscriber_count = len(emitter._subscribers[EventType.QUERY_RECEIVED])

        monitor = LiveMonitor()

        # Should have added a subscriber
        assert len(emitter._subscribers[EventType.QUERY_RECEIVED]) > initial_subscriber_count

    def test_live_monitor_captures_events(self):
        """LiveMonitor should capture events when emitted."""
        from src.rag.interface.web.live_monitor import LiveMonitor

        monitor = LiveMonitor()
        emitter = EventEmitter()

        # Emit an event
        event = QueryReceivedEvent(
            timestamp=datetime.now().isoformat(),
            query_text="휴학 신청 방법",
            correlation_id="test-123"
        )
        emitter.emit(EventType.QUERY_RECEIVED, event)

        # Monitor should have captured the event
        events = monitor.get_events()
        assert len(events) == 1
        assert events[0]["query_text"] == "휴학 신청 방법"

    def test_live_monitor_event_buffer_limit(self):
        """LiveMonitor should limit event buffer to 100 events."""
        from src.rag.interface.web.live_monitor import LiveMonitor

        monitor = LiveMonitor()
        emitter = EventEmitter()

        # Emit 150 events
        for i in range(150):
            event = QueryReceivedEvent(
                timestamp=datetime.now().isoformat(),
                query_text=f"Query {i}",
                correlation_id=f"test-{i}"
            )
            emitter.emit(EventType.QUERY_RECEIVED, event)

        # Should only keep last 100 events
        events = monitor.get_events()
        assert len(events) == 100
        assert events[0]["query_text"] == "Query 50"  # First 50 should be dropped

    def test_live_monitor_get_events_returns_copy(self):
        """LiveMonitor.get_events should return a copy, not the internal buffer."""
        from src.rag.interface.web.live_monitor import LiveMonitor

        monitor = LiveMonitor()
        events1 = monitor.get_events()
        events2 = monitor.get_events()

        # Modifying returned list should not affect internal buffer
        events1.append({"test": "data"})
        assert len(monitor.get_events()) == 0

    def test_live_monitor_clear_events(self):
        """LiveMonitor should support clearing the event buffer."""
        from src.rag.interface.web.live_monitor import LiveMonitor

        monitor = LiveMonitor()
        emitter = EventEmitter()

        # Emit an event
        event = QueryReceivedEvent(
            timestamp=datetime.now().isoformat(),
            query_text="Test query",
            correlation_id="test-123"
        )
        emitter.emit(EventType.QUERY_RECEIVED, event)

        assert monitor.get_event_count() == 1

        # Clear events
        monitor.clear_events()
        assert monitor.get_event_count() == 0

    def test_live_monitor_filter_by_event_type(self):
        """LiveMonitor should support filtering events by type."""
        from src.rag.interface.web.live_monitor import LiveMonitor

        monitor = LiveMonitor()
        emitter = EventEmitter()

        # Emit different event types
        query_event = QueryReceivedEvent(
            timestamp=datetime.now().isoformat(),
            query_text="Test query",
            correlation_id="test-123"
        )
        emitter.emit(EventType.QUERY_RECEIVED, query_event)

        search_event = SearchCompletedEvent(
            timestamp=datetime.now().isoformat(),
            results=["result1", "result2"],
            scores=[0.9, 0.8],
            latency_ms=100
        )
        emitter.emit(EventType.SEARCH_COMPLETED, search_event)

        # Filter by QUERY_RECEIVED
        query_events = monitor.get_events(event_type=EventType.QUERY_RECEIVED)
        assert len(query_events) == 1
        assert query_events[0]["query_text"] == "Test query"

        # Filter by SEARCH_COMPLETED
        search_events = monitor.get_events(event_type=EventType.SEARCH_COMPLETED)
        assert len(search_events) == 1
        assert search_events[0]["latency_ms"] == 100

    def test_live_monitor_unsubscribe_cleanup(self):
        """LiveMonitor should properly unsubscribe when cleanup() is called."""
        from src.rag.interface.web.live_monitor import LiveMonitor

        emitter = EventEmitter()
        initial_count = len(emitter._subscribers[EventType.QUERY_RECEIVED])

        monitor = LiveMonitor()
        after_init_count = len(emitter._subscribers[EventType.QUERY_RECEIVED])
        assert after_init_count > initial_count

        monitor.cleanup()
        after_cleanup_count = len(emitter._subscribers[EventType.QUERY_RECEIVED])
        assert after_cleanup_count == initial_count

    def test_live_monitor_submit_query_returns_result(self):
        """LiveMonitor.submit_query should return a query result."""
        from src.rag.interface.web.live_monitor import LiveMonitor

        monitor = LiveMonitor()

        result = monitor.submit_query("휴학 신청 방법")

        assert result is not None
        assert "query" in result
        assert result["query"] == "휴학 신청 방법"

    def test_live_monitor_submit_query_triggers_events(self):
        """LiveMonitor.submit_query should trigger event emission."""
        from src.rag.interface.web.live_monitor import LiveMonitor

        monitor = LiveMonitor()

        # Submit a query
        monitor.submit_query("휴학 신청 방법")

        # Should have captured events
        events = monitor.get_events()
        assert len(events) > 0

        # Should include QUERY_RECEIVED event
        query_events = [e for e in events if "query_text" in e]
        assert len(query_events) > 0


class TestLiveMonitorIntegration:
    """Integration tests for LiveMonitor with Gradio."""

    def test_live_monitor_gradio_tab_creation(self):
        """LiveMonitor should create a Gradio tab."""
        from src.rag.interface.web.live_monitor import create_monitor_tab

        # Mock Gradio
        with patch("src.rag.interface.web.live_monitor.gr") as mock_gr:
            mock_gr.Tab.return_value.__enter__ = Mock(return_value=Mock())
            mock_gr.Tab.return_value.__exit__ = Mock(return_value=False)

            monitor, components = create_monitor_tab()

            assert monitor is not None
            assert "event_display" in components
            assert "query_input" in components

    def test_live_monitor_gradio_event_streaming(self):
        """LiveMonitor should support Gradio streaming for events."""
        from src.rag.interface.web.live_monitor import LiveMonitor

        monitor = LiveMonitor()

        # Get streaming generator
        stream = monitor.get_event_stream()

        # Should be a generator
        assert hasattr(stream, '__iter__')

    def test_live_monitor_gradio_refresh(self):
        """LiveMonitor should support Gradio refresh mechanism."""
        from src.rag.interface.web.live_monitor import LiveMonitor

        monitor = LiveMonitor()

        # Initial state
        initial_events = monitor.get_events_for_gradio()

        # After event
        emitter = EventEmitter()
        event = QueryReceivedEvent(
            timestamp=datetime.now().isoformat(),
            query_text="Test",
            correlation_id="test-123"
        )
        emitter.emit(EventType.QUERY_RECEIVED, event)

        # Should have updated state
        updated_events = monitor.get_events_for_gradio()
        assert len(updated_events) > len(initial_events)


class TestLiveMonitorThreadSafety:
    """Thread safety tests for LiveMonitor."""

    def test_concurrent_event_emission(self):
        """LiveMonitor should handle concurrent event emissions safely."""
        from src.rag.interface.web.live_monitor import LiveMonitor
        import threading

        monitor = LiveMonitor()
        emitter = EventEmitter()

        def emit_events(count):
            for i in range(count):
                event = QueryReceivedEvent(
                    timestamp=datetime.now().isoformat(),
                    query_text=f"Query-{threading.current_thread().name}-{i}",
                    correlation_id=f"test-{i}"
                )
                emitter.emit(EventType.QUERY_RECEIVED, event)

        # Create multiple threads
        threads = [
            threading.Thread(target=emit_events, args=(20,), name=f"Thread-{i}")
            for i in range(5)
        ]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have captured events (at least some)
        events = monitor.get_events()
        assert len(events) > 0
