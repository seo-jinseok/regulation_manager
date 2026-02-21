"""
SPEC-RAG-MONITOR-001: Event System Tests

TDD RED Phase: Tests for event types and EventEmitter.
These tests are written BEFORE implementation to drive the design.

Purpose: Verify event emission and subscription patterns.
"""

import time
from dataclasses import asdict
from datetime import datetime
from typing import List
from unittest.mock import MagicMock

import pytest


class TestEventType:
    """Tests for EventType enum."""

    def test_event_type_values(self):
        """Test that all required event types are defined."""
        # RED: This test verifies EventType enum has all required values
        from src.rag.infrastructure.logging.events import EventType

        required_types = [
            "QUERY_RECEIVED",
            "QUERY_REWRITTEN",
            "SEARCH_COMPLETED",
            "RERANKING_COMPLETED",
            "LLM_GENERATION_STARTED",
            "TOKEN_GENERATED",
            "ANSWER_GENERATED",
        ]

        for event_type in required_types:
            assert hasattr(EventType, event_type), f"Missing EventType: {event_type}"

    def test_event_type_count(self):
        """Test that exactly 7 event types are defined."""
        # RED: This test verifies event type count matches SPEC
        from src.rag.infrastructure.logging.events import EventType

        assert len(EventType) == 7, f"Expected 7 event types, got {len(EventType)}"


class TestEventDataclasses:
    """Tests for event dataclasses."""

    def test_query_received_event_structure(self):
        """Test QueryReceived event dataclass structure."""
        # RED: This test verifies QueryReceived event structure
        from src.rag.infrastructure.logging.events import QueryReceivedEvent

        event = QueryReceivedEvent(
            timestamp="2024-01-01T12:00:00",
            query_text="휴학 신청 방법",
            correlation_id="test-correlation-123",
        )

        assert event.timestamp == "2024-01-01T12:00:00"
        assert event.query_text == "휴학 신청 방법"
        assert event.correlation_id == "test-correlation-123"

    def test_query_rewritten_event_structure(self):
        """Test QueryRewritten event dataclass structure."""
        # RED: This test verifies QueryRewritten event structure
        from src.rag.infrastructure.logging.events import QueryRewrittenEvent

        event = QueryRewrittenEvent(
            timestamp="2024-01-01T12:00:01",
            original_query="휴학",
            rewritten_query="대학 휴학 신청 방법 절차",
            strategy="hyde_expansion",
        )

        assert event.original_query == "휴학"
        assert event.rewritten_query == "대학 휴학 신청 방법 절차"
        assert event.strategy == "hyde_expansion"

    def test_search_completed_event_structure(self):
        """Test SearchCompleted event dataclass structure."""
        # RED: This test verifies SearchCompleted event structure
        from src.rag.infrastructure.logging.events import SearchCompletedEvent

        event = SearchCompletedEvent(
            timestamp="2024-01-01T12:00:02",
            results=["result1", "result2"],
            scores=[0.95, 0.87],
            latency_ms=45,
        )

        assert len(event.results) == 2
        assert len(event.scores) == 2
        assert event.latency_ms == 45

    def test_reranking_completed_event_structure(self):
        """Test RerankingCompleted event dataclass structure."""
        # RED: This test verifies RerankingCompleted event structure
        from src.rag.infrastructure.logging.events import RerankingCompletedEvent

        event = RerankingCompletedEvent(
            timestamp="2024-01-01T12:00:03",
            reordered_results=["result2", "result1"],
            score_changes={"result1": -0.05, "result2": 0.08},
            latency_ms=12,
        )

        assert event.reordered_results == ["result2", "result1"]
        assert event.score_changes["result1"] == -0.05
        assert event.latency_ms == 12

    def test_llm_generation_started_event_structure(self):
        """Test LLMGenerationStarted event dataclass structure."""
        # RED: This test verifies LLMGenerationStarted event structure
        from src.rag.infrastructure.logging.events import LLMGenerationStartedEvent

        event = LLMGenerationStartedEvent(
            timestamp="2024-01-01T12:00:04",
            prompt_preview="Context: ...",
            model_name="gpt-4o-mini",
            max_tokens=1000,
        )

        assert event.prompt_preview == "Context: ..."
        assert event.model_name == "gpt-4o-mini"
        assert event.max_tokens == 1000

    def test_token_generated_event_structure(self):
        """Test TokenGenerated event dataclass structure."""
        # RED: This test verifies TokenGenerated event structure
        from src.rag.infrastructure.logging.events import TokenGeneratedEvent

        event = TokenGeneratedEvent(
            timestamp="2024-01-01T12:00:05",
            token="안녕",
            accumulated_length=5,
        )

        assert event.token == "안녕"
        assert event.accumulated_length == 5

    def test_answer_generated_event_structure(self):
        """Test AnswerGenerated event dataclass structure."""
        # RED: This test verifies AnswerGenerated event structure
        from src.rag.infrastructure.logging.events import AnswerGeneratedEvent

        event = AnswerGeneratedEvent(
            timestamp="2024-01-01T12:00:06",
            full_response="휴학 신청은 학기 시작 전까지 가능합니다.",
            token_count=25,
            latency_ms=850,
        )

        assert event.full_response == "휴학 신청은 학기 시작 전까지 가능합니다."
        assert event.token_count == 25
        assert event.latency_ms == 850

    def test_event_can_be_serialized_to_dict(self):
        """Test that events can be serialized to dictionary."""
        # RED: This test verifies events are serializable
        from src.rag.infrastructure.logging.events import QueryReceivedEvent

        event = QueryReceivedEvent(
            timestamp="2024-01-01T12:00:00",
            query_text="test",
            correlation_id="123",
        )

        event_dict = asdict(event)
        assert isinstance(event_dict, dict)
        assert "timestamp" in event_dict
        assert "query_text" in event_dict
        assert "correlation_id" in event_dict


class TestEventEmitter:
    """Tests for EventEmitter singleton class."""

    def test_event_emitter_is_singleton(self):
        """Test that EventEmitter is a singleton."""
        # RED: This test verifies singleton pattern
        from src.rag.infrastructure.logging.events import EventEmitter

        emitter1 = EventEmitter()
        emitter2 = EventEmitter()

        assert emitter1 is emitter2, "EventEmitter should be singleton"

    def test_subscribe_to_event(self):
        """Test subscribing to an event type."""
        # RED: This test verifies subscribe pattern
        from src.rag.infrastructure.logging.events import EventEmitter, EventType

        emitter = EventEmitter()
        callback = MagicMock()

        emitter.subscribe(EventType.QUERY_RECEIVED, callback)
        assert callback in emitter._subscribers[EventType.QUERY_RECEIVED]

    def test_unsubscribe_from_event(self):
        """Test unsubscribing from an event type."""
        # RED: This test verifies unsubscribe pattern
        from src.rag.infrastructure.logging.events import EventEmitter, EventType

        emitter = EventEmitter()
        callback = MagicMock()

        emitter.subscribe(EventType.QUERY_RECEIVED, callback)
        emitter.unsubscribe(EventType.QUERY_RECEIVED, callback)

        assert callback not in emitter._subscribers[EventType.QUERY_RECEIVED]

    def test_emit_event_calls_subscribers(self):
        """Test that emitting an event calls all subscribers."""
        # RED: This test verifies event emission
        from src.rag.infrastructure.logging.events import (
            EventEmitter,
            EventType,
            QueryReceivedEvent,
        )

        emitter = EventEmitter()
        callback1 = MagicMock()
        callback2 = MagicMock()

        emitter.subscribe(EventType.QUERY_RECEIVED, callback1)
        emitter.subscribe(EventType.QUERY_RECEIVED, callback2)

        event = QueryReceivedEvent(
            timestamp="2024-01-01T12:00:00",
            query_text="test",
            correlation_id="123",
        )
        emitter.emit(EventType.QUERY_RECEIVED, event)

        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)

    def test_multiple_subscribers_for_same_event(self):
        """Test that multiple subscribers receive the same event."""
        # RED: This test verifies multiple subscriber support
        from src.rag.infrastructure.logging.events import (
            EventEmitter,
            EventType,
            QueryReceivedEvent,
        )

        emitter = EventEmitter()
        received_events: List = []

        def collector1(event):
            received_events.append(("collector1", event))

        def collector2(event):
            received_events.append(("collector2", event))

        emitter.subscribe(EventType.QUERY_RECEIVED, collector1)
        emitter.subscribe(EventType.QUERY_RECEIVED, collector2)

        event = QueryReceivedEvent(
            timestamp="2024-01-01T12:00:00",
            query_text="test",
            correlation_id="123",
        )
        emitter.emit(EventType.QUERY_RECEIVED, event)

        assert len(received_events) == 2
        assert received_events[0][0] == "collector1"
        assert received_events[1][0] == "collector2"
        assert received_events[0][1] is event
        assert received_events[1][1] is event

    def test_emit_with_no_subscribers_does_not_error(self):
        """Test that emitting with no subscribers is safe."""
        # RED: This test verifies safe emission without subscribers
        from src.rag.infrastructure.logging.events import (
            EventEmitter,
            EventType,
            QueryReceivedEvent,
        )

        emitter = EventEmitter()
        event = QueryReceivedEvent(
            timestamp="2024-01-01T12:00:00",
            query_text="test",
            correlation_id="123",
        )

        # Should not raise any exception
        emitter.emit(EventType.QUERY_RECEIVED, event)

    def test_clear_all_subscribers(self):
        """Test clearing all subscribers."""
        # RED: This test verifies clear functionality
        from src.rag.infrastructure.logging.events import EventEmitter, EventType

        emitter = EventEmitter()
        callback = MagicMock()

        emitter.subscribe(EventType.QUERY_RECEIVED, callback)
        emitter.subscribe(EventType.SEARCH_COMPLETED, callback)

        emitter.clear()

        assert len(emitter._subscribers[EventType.QUERY_RECEIVED]) == 0
        assert len(emitter._subscribers[EventType.SEARCH_COMPLETED]) == 0


class TestEventTimestamp:
    """Tests for event timestamp handling."""

    def test_timestamp_is_iso_format(self):
        """Test that timestamp follows ISO format."""
        # RED: This test verifies timestamp format
        from src.rag.infrastructure.logging.events import QueryReceivedEvent

        event = QueryReceivedEvent(
            timestamp="2024-01-15T14:30:00.123456",
            query_text="test",
            correlation_id="123",
        )

        # Should parse as valid ISO format
        parsed = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 15


class TestEventEmitterPerformance:
    """Tests for EventEmitter performance characteristics."""

    def test_emit_performance_with_many_subscribers(self):
        """Test that emit is fast even with many subscribers."""
        # RED: This test verifies performance requirement (< 50ms per event)
        from src.rag.infrastructure.logging.events import (
            EventEmitter,
            EventType,
            QueryReceivedEvent,
        )

        emitter = EventEmitter()

        # Add 10 subscribers
        for _ in range(10):
            emitter.subscribe(EventType.QUERY_RECEIVED, lambda e: None)

        event = QueryReceivedEvent(
            timestamp="2024-01-01T12:00:00",
            query_text="test",
            correlation_id="123",
        )

        # Measure time
        start = time.perf_counter()
        for _ in range(100):
            emitter.emit(EventType.QUERY_RECEIVED, event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Average time per emit should be < 50ms
        avg_ms = elapsed_ms / 100
        assert avg_ms < 50, f"Emit took {avg_ms}ms, expected < 50ms"

        # Cleanup
        emitter.clear()


if __name__ == "__main__":
    # Run tests in verbose mode
    pytest.main([__file__, "-v", "--tb=short"])
