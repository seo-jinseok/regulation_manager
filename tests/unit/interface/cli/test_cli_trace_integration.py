"""
SPEC-RAG-MONITOR-001: CLI Integration Tests

Tests for CLI --trace flag integration with query processing.
"""

import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


class TestCLITraceIntegration:
    """Tests for CLI --trace flag integration."""

    def test_trace_handler_subscription_with_flag(self):
        """Test that trace handler subscribes when --trace is set."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler
        from src.rag.infrastructure.logging.events import EventEmitter, EventType

        # Clear any existing subscribers
        emitter = EventEmitter()
        emitter.clear()

        # Create handler and subscribe
        handler = TraceOutputHandler()
        handler.subscribe()

        # Verify all event types have subscribers
        for event_type in EventType:
            assert len(emitter._subscribers[event_type]) >= 1

        # Cleanup
        handler.unsubscribe()

    def test_trace_handler_cleanup(self):
        """Test that trace handler properly unsubscribes."""
        from src.rag.interface.cli_trace.trace_handler import TraceOutputHandler
        from src.rag.infrastructure.logging.events import EventEmitter, EventType

        emitter = EventEmitter()
        emitter.clear()

        handler = TraceOutputHandler()
        handler.subscribe()

        # Get initial counts
        initial_counts = {et: len(emitter._subscribers[et]) for et in EventType}

        # Unsubscribe
        handler.unsubscribe()

        # Verify all subscribers removed
        for event_type in EventType:
            assert len(emitter._subscribers[event_type]) < initial_counts[event_type]

    def test_cli_file_has_trace_flag(self):
        """Test that CLI file contains --trace flag."""
        import pathlib
        cli_file = pathlib.Path("/Users/truestone/Dropbox/repo/University/regulation_manager/src/rag/interface/cli.py")
        content = cli_file.read_text()

        # Verify --trace flag is defined
        assert '"--trace"' in content
        assert "상세 처리 과정 실시간 출력" in content

    def test_cli_file_has_monitor_flag(self):
        """Test that CLI file contains --monitor flag."""
        import pathlib
        cli_file = pathlib.Path("/Users/truestone/Dropbox/repo/University/regulation_manager/src/rag/interface/cli.py")
        content = cli_file.read_text()

        # Verify --monitor flag is defined
        assert '"--monitor"' in content
        assert "Gradio 웹 대시보드 실행" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
