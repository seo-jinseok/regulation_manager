"""
SPEC-RAG-MONITOR-001: Interaction Logger Tests

TDD RED Phase: Tests for RAGInteractionLogger.
These tests are written BEFORE implementation to drive the design.

Purpose: Verify correlation ID generation, logging format, and latency measurement.
"""

import json
import logging
import time
from io import StringIO
from unittest.mock import patch

import pytest


class TestRAGInteractionLogger:
    """Tests for RAGInteractionLogger class."""

    def test_logger_initialization(self):
        """Test that logger can be initialized."""
        # RED: This test verifies basic initialization
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )

        logger = RAGInteractionLogger()
        assert logger is not None

    def test_logger_is_singleton(self):
        """Test that logger is a singleton."""
        # RED: This test verifies singleton pattern
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )

        logger1 = RAGInteractionLogger()
        logger2 = RAGInteractionLogger()

        assert logger1 is logger2, "RAGInteractionLogger should be singleton"

    def test_generate_correlation_id(self):
        """Test that correlation ID is generated correctly."""
        # RED: This test verifies correlation ID generation
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )

        logger = RAGInteractionLogger()
        correlation_id = logger.generate_correlation_id()

        assert correlation_id is not None
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0

    def test_correlation_id_uniqueness(self):
        """Test that correlation IDs are unique."""
        # RED: This test verifies correlation ID uniqueness
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )

        logger = RAGInteractionLogger()
        ids = {logger.generate_correlation_id() for _ in range(100)}

        assert len(ids) == 100, "All correlation IDs should be unique"

    def test_correlation_id_format(self):
        """Test that correlation ID follows expected format."""
        # RED: This test verifies correlation ID format (UUID-like)
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )

        logger = RAGInteractionLogger()
        correlation_id = logger.generate_correlation_id()

        # Should be UUID format (8-4-4-4-12 hex digits)
        import re

        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        assert uuid_pattern.match(
            correlation_id
        ), f"Correlation ID {correlation_id} should be UUID format"


class TestInteractionLoggerEventEmission:
    """Tests for event emission through logger."""

    def test_log_query_received(self):
        """Test logging QueryReceived event."""
        # RED: This test verifies query received logging
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )
        from src.rag.infrastructure.logging.events import EventType

        logger = RAGInteractionLogger()

        with patch.object(logger, "_emit_event") as mock_emit:
            correlation_id = logger.log_query_received("휴학 신청 방법")

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == EventType.QUERY_RECEIVED
            assert correlation_id is not None

    def test_log_query_rewritten(self):
        """Test logging QueryRewritten event."""
        # RED: This test verifies query rewritten logging
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )
        from src.rag.infrastructure.logging.events import EventType

        logger = RAGInteractionLogger()

        with patch.object(logger, "_emit_event") as mock_emit:
            logger.log_query_rewritten(
                original_query="휴학",
                rewritten_query="대학 휴학 신청 방법",
                strategy="expansion",
                correlation_id="test-id-123",
            )

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == EventType.QUERY_REWRITTEN

    def test_log_search_completed(self):
        """Test logging SearchCompleted event."""
        # RED: This test verifies search completed logging
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )
        from src.rag.infrastructure.logging.events import EventType

        logger = RAGInteractionLogger()

        with patch.object(logger, "_emit_event") as mock_emit:
            logger.log_search_completed(
                results=["result1", "result2"],
                scores=[0.95, 0.87],
                latency_ms=45,
                correlation_id="test-id-123",
            )

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == EventType.SEARCH_COMPLETED

    def test_log_reranking_completed(self):
        """Test logging RerankingCompleted event."""
        # RED: This test verifies reranking completed logging
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )
        from src.rag.infrastructure.logging.events import EventType

        logger = RAGInteractionLogger()

        with patch.object(logger, "_emit_event") as mock_emit:
            logger.log_reranking_completed(
                reordered_results=["r2", "r1"],
                score_changes={"r1": -0.05, "r2": 0.08},
                latency_ms=12,
                correlation_id="test-id-123",
            )

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == EventType.RERANKING_COMPLETED

    def test_log_llm_generation_started(self):
        """Test logging LLMGenerationStarted event."""
        # RED: This test verifies LLM generation started logging
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )
        from src.rag.infrastructure.logging.events import EventType

        logger = RAGInteractionLogger()

        with patch.object(logger, "_emit_event") as mock_emit:
            logger.log_llm_generation_started(
                prompt_preview="Context: ...",
                model_name="gpt-4o-mini",
                max_tokens=1000,
                correlation_id="test-id-123",
            )

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == EventType.LLM_GENERATION_STARTED

    def test_log_token_generated(self):
        """Test logging TokenGenerated event."""
        # RED: This test verifies token generated logging
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )
        from src.rag.infrastructure.logging.events import EventType

        logger = RAGInteractionLogger()

        with patch.object(logger, "_emit_event") as mock_emit:
            logger.log_token_generated(
                token="안녕",
                accumulated_length=5,
                correlation_id="test-id-123",
            )

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == EventType.TOKEN_GENERATED

    def test_log_answer_generated(self):
        """Test logging AnswerGenerated event."""
        # RED: This test verifies answer generated logging
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )
        from src.rag.infrastructure.logging.events import EventType

        logger = RAGInteractionLogger()

        with patch.object(logger, "_emit_event") as mock_emit:
            logger.log_answer_generated(
                full_response="휴학 신청은 ...",
                token_count=25,
                latency_ms=850,
                correlation_id="test-id-123",
            )

            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == EventType.ANSWER_GENERATED


class TestLatencyMeasurement:
    """Tests for latency measurement helpers."""

    def test_measure_latency_context_manager(self):
        """Test latency measurement with context manager."""
        # RED: This test verifies latency measurement
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )

        logger = RAGInteractionLogger()

        with logger.measure_latency() as timer:
            time.sleep(0.01)  # 10ms

        latency_ms = timer.elapsed_ms
        assert latency_ms >= 10, f"Expected >= 10ms, got {latency_ms}ms"
        assert latency_ms < 100, f"Expected < 100ms, got {latency_ms}ms"

    def test_measure_latency_callback(self):
        """Test latency measurement with callback."""
        # RED: This test verifies latency callback
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )

        logger = RAGInteractionLogger()
        captured_latency = None

        def callback(latency: float):
            nonlocal captured_latency
            captured_latency = latency

        with logger.measure_latency(callback):
            time.sleep(0.01)

        assert captured_latency is not None
        assert captured_latency >= 10


class TestLogOutputFormat:
    """Tests for logging output format."""

    def test_log_format_includes_correlation_id(self):
        """Test that log output includes correlation ID."""
        # RED: This test verifies correlation ID in logs
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )

        logger = RAGInteractionLogger()

        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(
            logging.Formatter("%(message)s")
        )

        # Add handler to logger
        internal_logger = logging.getLogger("rag.interaction")
        internal_logger.addHandler(handler)
        internal_logger.setLevel(logging.INFO)

        try:
            correlation_id = logger.log_query_received("test query")

            log_output = log_stream.getvalue()
            assert correlation_id in log_output, "Correlation ID should be in log output"
        finally:
            internal_logger.removeHandler(handler)

    def test_log_format_is_structured(self):
        """Test that log output is structured (JSON-like)."""
        # RED: This test verifies structured logging
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )

        logger = RAGInteractionLogger()

        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)

        internal_logger = logging.getLogger("rag.interaction")
        internal_logger.addHandler(handler)
        internal_logger.setLevel(logging.INFO)

        try:
            logger.log_query_received("test query")
            log_output = log_stream.getvalue()

            # Should be parseable as JSON
            try:
                parsed = json.loads(log_output.strip())
                assert isinstance(parsed, dict), "Log should be JSON object"
            except json.JSONDecodeError:
                pytest.fail(f"Log output is not valid JSON: {log_output}")
        finally:
            internal_logger.removeHandler(handler)

    def test_log_includes_timestamp(self):
        """Test that log output includes timestamp."""
        # RED: This test verifies timestamp in logs
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )

        logger = RAGInteractionLogger()

        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)

        internal_logger = logging.getLogger("rag.interaction")
        internal_logger.addHandler(handler)
        internal_logger.setLevel(logging.INFO)

        try:
            logger.log_query_received("test query")
            log_output = log_stream.getvalue()

            try:
                parsed = json.loads(log_output.strip())
                assert "timestamp" in parsed, "Log should include timestamp"
            except json.JSONDecodeError:
                pytest.fail(f"Log output is not valid JSON: {log_output}")
        finally:
            internal_logger.removeHandler(handler)


class TestLoggerPerformance:
    """Tests for logger performance characteristics."""

    def test_logging_overhead(self):
        """Test that logging overhead is minimal (< 10ms per call)."""
        # RED: This test verifies performance requirement
        from src.rag.infrastructure.logging.interaction_logger import (
            RAGInteractionLogger,
        )

        logger = RAGInteractionLogger()

        # Measure time for 100 log operations
        start = time.perf_counter()
        for _ in range(100):
            logger.log_query_received("test query")
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / 100
        assert avg_ms < 10, f"Logging overhead {avg_ms}ms exceeds 10ms limit"


if __name__ == "__main__":
    # Run tests in verbose mode
    pytest.main([__file__, "-v", "--tb=short"])
