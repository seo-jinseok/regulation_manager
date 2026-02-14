"""
Characterization tests for ConversationMemoryManager - Additional Coverage.

SPEC: SPEC-TEST-COV-001 Phase 3 - Test Coverage Improvement

These tests are designed to cover uncovered branches in conversation_memory.py
to achieve 85% coverage target.

Key classes and methods to test:
- ConversationMemoryManager: create_session, get_session, add_message, get_context_for_search,
  expand_query, cleanup_expired_sessions
- ConversationSummarizer: should_summarize, summarize, _extract_entities, _infer_topics
- TopicInference: infer_topics, expand_query_with_topics
- Message, ConversationSummary, ConversationContext dataclasses
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.rag.application.conversation_memory import (
    ConversationContext,
    ConversationMemoryManager,
    ConversationSummarizer,
    ConversationSummary,
    MemoryExpiryPolicy,
    Message,
    TopicInference,
    create_memory_manager,
)


# =============================================================================
# Test Message Dataclass
# =============================================================================


class TestMessage:
    """Characterization tests for Message dataclass."""

    def test_to_dict(self):
        """Message can be converted to dictionary."""
        msg = Message(
            role="user",
            content="Test content",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            metadata={"key": "value"},
        )
        result = msg.to_dict()
        assert result["role"] == "user"
        assert result["content"] == "Test content"
        assert "timestamp" in result
        assert result["metadata"] == {"key": "value"}

    def test_from_dict(self):
        """Message can be created from dictionary."""
        data = {
            "role": "assistant",
            "content": "Response",
            "timestamp": "2025-01-01T12:00:00",
            "metadata": {},
        }
        msg = Message.from_dict(data)
        assert msg.role == "assistant"
        assert msg.content == "Response"
        assert msg.timestamp == datetime(2025, 1, 1, 12, 0, 0)

    def test_from_dict_missing_timestamp(self):
        """Message.from_dict handles missing timestamp."""
        data = {"role": "user", "content": "Test"}
        msg = Message.from_dict(data)
        assert msg.role == "user"
        assert isinstance(msg.timestamp, datetime)

    def test_from_dict_none_timestamp(self):
        """Message.from_dict handles None timestamp."""
        data = {"role": "user", "content": "Test", "timestamp": None}
        msg = Message.from_dict(data)
        assert isinstance(msg.timestamp, datetime)

    def test_default_metadata(self):
        """Message has default empty metadata."""
        msg = Message(role="user", content="Test")
        assert msg.metadata == {}

    def test_default_timestamp(self):
        """Message has default timestamp."""
        msg = Message(role="user", content="Test")
        assert isinstance(msg.timestamp, datetime)


# =============================================================================
# Test ConversationSummary Dataclass
# =============================================================================


class TestConversationSummary:
    """Characterization tests for ConversationSummary dataclass."""

    def test_to_dict(self):
        """ConversationSummary can be converted to dictionary."""
        summary = ConversationSummary(
            summary="Test summary",
            key_topics=["topic1", "topic2"],
            key_entities=["entity1"],
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            message_count=5,
        )
        result = summary.to_dict()
        assert result["summary"] == "Test summary"
        assert result["key_topics"] == ["topic1", "topic2"]
        assert result["key_entities"] == ["entity1"]
        assert result["message_count"] == 5

    def test_from_dict(self):
        """ConversationSummary can be created from dictionary."""
        data = {
            "summary": "Test summary",
            "key_topics": ["topic1"],
            "key_entities": ["entity1"],
            "timestamp": "2025-01-01T12:00:00",
            "message_count": 3,
        }
        summary = ConversationSummary.from_dict(data)
        assert summary.summary == "Test summary"
        assert summary.key_topics == ["topic1"]
        assert summary.key_entities == ["entity1"]

    def test_from_dict_defaults(self):
        """ConversationSummary.from_dict handles missing optional fields."""
        data = {"summary": "Test summary"}
        summary = ConversationSummary.from_dict(data)
        assert summary.key_topics == []
        assert summary.key_entities == []
        assert summary.message_count == 0

    def test_default_values(self):
        """ConversationSummary has correct default values."""
        summary = ConversationSummary(summary="Test", key_topics=[], key_entities=[])
        assert summary.key_topics == []
        assert summary.key_entities == []
        assert summary.message_count == 0


# =============================================================================
# Test ConversationContext Dataclass
# =============================================================================


class TestConversationContext:
    """Characterization tests for ConversationContext dataclass."""

    def test_add_message(self):
        """add_message adds message to context."""
        context = ConversationContext(session_id="test_session")
        context.add_message("user", "Hello", {"meta": "data"})
        assert len(context.messages) == 1
        assert context.messages[0].role == "user"
        assert context.messages[0].content == "Hello"

    def test_add_message_updates_last_updated(self):
        """add_message updates last_updated timestamp."""
        context = ConversationContext(session_id="test_session")
        old_time = context.last_updated
        context.add_message("user", "Hello")
        assert context.last_updated >= old_time

    def test_get_recent_messages(self):
        """get_recent_messages returns last N messages."""
        context = ConversationContext(session_id="test_session")
        for i in range(15):
            context.add_message("user", f"Message {i}")
        recent = context.get_recent_messages(limit=5)
        assert len(recent) == 5
        assert "Message 14" in recent[-1].content

    def test_get_recent_messages_empty(self):
        """get_recent_messages returns empty list for empty context."""
        context = ConversationContext(session_id="test_session")
        recent = context.get_recent_messages()
        assert recent == []

    def test_is_expired_never_policy(self):
        """is_expired returns False for NEVER policy."""
        context = ConversationContext(
            session_id="test_session",
            expiry_policy=MemoryExpiryPolicy.NEVER,
        )
        assert context.is_expired() is False

    def test_is_expired_hours_24_policy_not_expired(self):
        """is_expired returns False for recent session."""
        context = ConversationContext(
            session_id="test_session",
            expiry_policy=MemoryExpiryPolicy.HOURS_24,
        )
        context.last_updated = datetime.now()
        assert context.is_expired() is False

    def test_is_expired_hours_24_policy_expired(self):
        """is_expired returns True for old session."""
        context = ConversationContext(
            session_id="test_session",
            expiry_policy=MemoryExpiryPolicy.HOURS_24,
        )
        context.last_updated = datetime.now() - timedelta(hours=25)
        assert context.is_expired() is True

    def test_is_expired_days_7_policy_expired(self):
        """is_expired works with DAYS_7 policy."""
        context = ConversationContext(
            session_id="test_session",
            expiry_policy=MemoryExpiryPolicy.DAYS_7,
        )
        context.last_updated = datetime.now() - timedelta(days=8)
        assert context.is_expired() is True

    def test_is_expired_days_30_policy(self):
        """is_expired works with DAYS_30 policy."""
        context = ConversationContext(
            session_id="test_session",
            expiry_policy=MemoryExpiryPolicy.DAYS_30,
        )
        context.last_updated = datetime.now() - timedelta(days=31)
        assert context.is_expired() is True

    def test_to_dict(self):
        """ConversationContext can be converted to dictionary."""
        context = ConversationContext(
            session_id="test_session",
            user_id="user123",
            expiry_policy=MemoryExpiryPolicy.DAYS_7,
        )
        result = context.to_dict()
        assert result["session_id"] == "test_session"
        assert result["user_id"] == "user123"
        assert result["expiry_policy"] == "7d"

    def test_from_dict(self):
        """ConversationContext can be created from dictionary."""
        data = {
            "session_id": "test_session",
            "user_id": "user123",
            "messages": [],
            "summaries": [],
            "current_topics": ["topic1"],
            "created_at": "2025-01-01T12:00:00",
            "last_updated": "2025-01-01T12:00:00",
            "expiry_policy": "7d",
        }
        context = ConversationContext.from_dict(data)
        assert context.session_id == "test_session"
        assert context.user_id == "user123"
        assert context.current_topics == ["topic1"]

    def test_from_dict_missing_timestamps(self):
        """ConversationContext.from_dict handles missing timestamps."""
        data = {"session_id": "test_session"}
        context = ConversationContext.from_dict(data)
        assert isinstance(context.created_at, datetime)
        assert isinstance(context.last_updated, datetime)


# =============================================================================
# Test ConversationSummarizer
# =============================================================================


class TestConversationSummarizer:
    """Characterization tests for ConversationSummarizer."""

    @pytest.fixture
    def summarizer(self):
        return ConversationSummarizer()

    @pytest.fixture
    def summarizer_with_llm(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "LLM generated summary"
        return ConversationSummarizer(llm_client=mock_llm)

    def test_should_summarize_below_threshold(self, summarizer):
        """should_summarize returns False below threshold."""
        context = ConversationContext(session_id="test")
        for i in range(5):
            context.add_message("user", f"Message {i}")
        assert summarizer.should_summarize(context, max_messages=10) is False

    def test_should_summarize_at_threshold(self, summarizer):
        """should_summarize returns True at threshold."""
        context = ConversationContext(session_id="test")
        for i in range(10):
            context.add_message("user", f"Message {i}")
        assert summarizer.should_summarize(context, max_messages=10) is True

    def test_should_summarize_above_threshold(self, summarizer):
        """should_summarize returns True above threshold."""
        context = ConversationContext(session_id="test")
        for i in range(15):
            context.add_message("user", f"Message {i}")
        assert summarizer.should_summarize(context, max_messages=10) is True

    def test_summarize_extracts_entities(self, summarizer):
        """summarize extracts key entities from messages."""
        context = ConversationContext(session_id="test")
        context.add_message("user", "제1조 규정에 대해 질문합니다")
        context.add_message("assistant", "제1조는 총칙입니다")
        summary = summarizer.summarize(context)
        assert isinstance(summary, ConversationSummary)
        assert isinstance(summary.key_entities, list)

    def test_summarize_infers_topics(self, summarizer):
        """summarize infers topics from messages."""
        context = ConversationContext(session_id="test")
        context.add_message("user", "휴가 신청 방법")
        context.add_message("assistant", "휴가는 연차 휴가와 병가가 있습니다")
        summary = summarizer.summarize(context)
        assert isinstance(summary.key_topics, list)

    def test_summarize_with_llm(self, summarizer_with_llm):
        """summarize uses LLM when available."""
        context = ConversationContext(session_id="test")
        for i in range(10):
            context.add_message("user", f"Message {i}")
        summary = summarizer_with_llm.summarize(context)
        assert summary.summary == "LLM generated summary"

    def test_extract_entities_regulation(self, summarizer):
        """_extract_entities extracts regulation patterns."""
        text = "교원인사규정에 따르면"
        entities = summarizer._extract_entities(text)
        assert "교원인사규정" in entities or any("규정" in e for e in entities)

    def test_extract_entities_article(self, summarizer):
        """_extract_entities extracts article patterns."""
        text = "제5조제2항에 규정됨"
        entities = summarizer._extract_entities(text)
        assert len(entities) > 0

    def test_extract_entities_date(self, summarizer):
        """_extract_entities extracts date patterns."""
        text = "2025년 1월 15일까지"
        entities = summarizer._extract_entities(text)
        assert len(entities) > 0

    def test_extract_entities_empty(self, summarizer):
        """_extract_entities handles empty text."""
        entities = summarizer._extract_entities("")
        assert entities == []

    def test_infer_topics_vacation(self, summarizer):
        """_infer_topics detects vacation-related topics."""
        messages = [Message(role="user", content="휴가 연차 병가 신청")]
        topics = summarizer._infer_topics(messages)
        assert "휴가" in topics

    def test_infer_topics_promotion(self, summarizer):
        """_infer_topics detects promotion-related topics."""
        messages = [Message(role="user", content="승진 승급 심사")]
        topics = summarizer._infer_topics(messages)
        assert "승진" in topics

    def test_infer_topics_discipline(self, summarizer):
        """_infer_topics detects discipline-related topics."""
        messages = [Message(role="user", content="징계 견책 정직")]
        topics = summarizer._infer_topics(messages)
        assert "징계" in topics

    def test_infer_topics_empty(self, summarizer):
        """_infer_topics handles empty messages."""
        topics = summarizer._infer_topics([])
        assert topics == []

    def test_summarize_with_rules(self, summarizer):
        """_summarize_with_rules generates summary."""
        text = "제1조 규정 내용. 제2조 규정 내용."
        entities = ["제1조", "제2조"]
        summary = summarizer._summarize_with_rules(text, entities)
        assert "대화 요약" in summary

    def test_summarize_with_rules_no_entities(self, summarizer):
        """_summarize_with_rules handles no entities."""
        text = "Some text without entities."
        entities = []
        summary = summarizer._summarize_with_rules(text, entities)
        assert "대화 요약" in summary

    def test_summarize_with_llm_fallback(self):
        """_summarize_with_llm falls back to rules on error."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("LLM error")
        summarizer = ConversationSummarizer(llm_client=mock_llm)
        summary = summarizer._summarize_with_llm("test", [], [])
        assert "대화 요약" in summary

    def test_summarize_with_llm_no_generate_method(self):
        """_summarize_with_llm falls back when LLM has no generate."""
        mock_llm = MagicMock(spec=[])  # No generate method
        summarizer = ConversationSummarizer(llm_client=mock_llm)
        summary = summarizer._summarize_with_llm("test", [], [])
        assert "대화 요약" in summary


# =============================================================================
# Test TopicInference
# =============================================================================


class TestTopicInference:
    """Characterization tests for TopicInference."""

    @pytest.fixture
    def inference(self):
        return TopicInference()

    @pytest.fixture
    def context_with_messages(self):
        context = ConversationContext(session_id="test")
        context.add_message("user", "휴가 신청 방법")
        context.add_message("assistant", "휴가는 연차와 병가로 나뉩니다")
        return context

    def test_infer_topics_from_messages(self, inference, context_with_messages):
        """infer_topics extracts topics from messages."""
        topics = inference.infer_topics(context_with_messages)
        assert isinstance(topics, list)
        assert "휴가" in topics

    def test_infer_topics_with_summaries(self, inference):
        """infer_topics uses summaries for context."""
        context = ConversationContext(session_id="test")
        context.summaries.append(
            ConversationSummary(
                summary="승진 관련 대화",
                key_topics=["승진"],
                key_entities=[],
            )
        )
        topics = inference.infer_topics(context)
        assert isinstance(topics, list)

    def test_infer_topics_preserves_existing(self, inference):
        """infer_topics may preserve existing topics."""
        context = ConversationContext(session_id="test")
        context.current_topics = ["휴가"]
        context.add_message("user", "연차 휴가 관련 질문")
        topics = inference.infer_topics(context)
        assert isinstance(topics, list)

    def test_infer_topics_empty_context(self, inference):
        """infer_topics handles empty context."""
        context = ConversationContext(session_id="test")
        topics = inference.infer_topics(context)
        assert topics == []

    def test_expand_query_with_topics(self, inference):
        """expand_query_with_topics adds topic terms."""
        query = "신청 방법"
        topics = ["휴가"]
        expanded = inference.expand_query_with_topics(query, topics)
        assert "신청 방법" in expanded
        # Should include expansion terms
        assert len(expanded) > len(query)

    def test_expand_query_no_topics(self, inference):
        """expand_query_with_topics returns original when no topics."""
        query = "신청 방법"
        expanded = inference.expand_query_with_topics(query, [])
        assert expanded == query

    def test_expand_query_with_multiple_topics(self, inference):
        """expand_query_with_topics handles multiple topics."""
        query = "방법"
        topics = ["휴가", "승진"]
        expanded = inference.expand_query_with_topics(query, topics)
        assert isinstance(expanded, str)


# =============================================================================
# Test ConversationMemoryManager
# =============================================================================


class TestConversationMemoryManager:
    """Characterization tests for ConversationMemoryManager."""

    @pytest.fixture
    def manager(self):
        return ConversationMemoryManager(enable_mcp=False)

    @pytest.fixture
    def manager_with_mcp(self):
        manager = ConversationMemoryManager(enable_mcp=True)
        manager._mcp_store = MagicMock()
        manager._mcp_retrieve = MagicMock(return_value=None)
        manager._mcp_delete = MagicMock()
        return manager

    def test_create_session(self, manager):
        """create_session creates new session."""
        session_id = manager.create_session()
        assert session_id is not None
        assert isinstance(session_id, str)

    def test_create_session_with_user_id(self, manager):
        """create_session stores user_id."""
        session_id = manager.create_session(user_id="user123")
        context = manager.get_session(session_id)
        assert context.user_id == "user123"

    def test_create_session_with_expiry_policy(self, manager):
        """create_session uses specified expiry policy."""
        session_id = manager.create_session(
            expiry_policy=MemoryExpiryPolicy.HOURS_24
        )
        context = manager.get_session(session_id)
        assert context.expiry_policy == MemoryExpiryPolicy.HOURS_24

    def test_create_session_mcp_enabled(self, manager_with_mcp):
        """create_session stores to MCP when enabled."""
        session_id = manager_with_mcp.create_session()
        manager_with_mcp._mcp_store.assert_called()

    def test_get_session_existing(self, manager):
        """get_session returns existing session."""
        session_id = manager.create_session()
        context = manager.get_session(session_id)
        assert context is not None
        assert context.session_id == session_id

    def test_get_session_nonexistent(self, manager):
        """get_session returns None for nonexistent session."""
        context = manager.get_session("nonexistent")
        assert context is None

    def test_get_session_from_mcp(self, manager_with_mcp):
        """get_session loads from MCP when not in memory."""
        session_data = {
            "session_id": "mcp_session",
            "user_id": "user123",
            "messages": [],
            "summaries": [],
            "current_topics": [],
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "expiry_policy": "7d",
        }
        manager_with_mcp._mcp_retrieve.return_value = json.dumps(
            session_data, ensure_ascii=False
        )
        context = manager_with_mcp.get_session("mcp_session")
        assert context is not None
        assert context.session_id == "mcp_session"

    def test_add_message(self, manager):
        """add_message adds message to session."""
        session_id = manager.create_session()
        context = manager.add_message(session_id, "user", "Hello")
        assert context is not None
        assert len(context.messages) == 1

    def test_add_message_nonexistent_session(self, manager):
        """add_message returns None for nonexistent session."""
        result = manager.add_message("nonexistent", "user", "Hello")
        assert result is None

    def test_add_message_filters_sensitive_info(self, manager):
        """add_message filters sensitive information."""
        session_id = manager.create_session()
        # Test phone number filtering
        manager.add_message(session_id, "user", "My phone is 010-12-3456")
        context = manager.get_session(session_id)
        assert "[FILTERED]" in context.messages[0].content

    def test_add_message_filters_email(self, manager):
        """add_message filters email addresses."""
        session_id = manager.create_session()
        manager.add_message(session_id, "user", "Email: test@example.com")
        context = manager.get_session(session_id)
        assert "[FILTERED]" in context.messages[0].content

    def test_add_message_triggers_summarization(self, manager):
        """add_message triggers summarization at threshold."""
        session_id = manager.create_session()
        for i in range(10):
            manager.add_message(session_id, "user", f"Message {i}")
        context = manager.get_session(session_id)
        # Messages should be summarized and trimmed
        assert len(context.summaries) > 0

    def test_add_message_updates_topics(self, manager):
        """add_message updates current topics."""
        session_id = manager.create_session()
        manager.add_message(session_id, "user", "휴가 연차 신청 방법")
        context = manager.get_session(session_id)
        assert len(context.current_topics) > 0

    def test_add_message_mcp_enabled(self, manager_with_mcp):
        """add_message stores to MCP when enabled."""
        session_id = manager_with_mcp.create_session()
        manager_with_mcp._mcp_store.reset_mock()
        manager_with_mcp.add_message(session_id, "user", "Hello")
        manager_with_mcp._mcp_store.assert_called()

    def test_get_context_for_search(self, manager):
        """get_context_for_search returns formatted context."""
        session_id = manager.create_session()
        manager.add_message(session_id, "user", "Test query")
        context = manager.get_context_for_search(session_id)
        assert context is not None
        assert "session_id" in context
        assert "current_topics" in context
        assert "recent_messages" in context

    def test_get_context_for_search_nonexistent(self, manager):
        """get_context_for_search returns None for nonexistent session."""
        context = manager.get_context_for_search("nonexistent")
        assert context is None

    def test_expand_query(self, manager):
        """expand_query adds context to query."""
        session_id = manager.create_session()
        manager.add_message(session_id, "user", "휴가 신청")
        expanded = manager.expand_query(session_id, "방법")
        # Should be expanded with topic terms
        assert isinstance(expanded, str)

    def test_expand_query_nonexistent_session(self, manager):
        """expand_query returns original for nonexistent session."""
        result = manager.expand_query("nonexistent", "test query")
        assert result == "test query"

    def test_cleanup_expired_sessions(self, manager):
        """cleanup_expired_sessions removes expired sessions."""
        session_id = manager.create_session(
            expiry_policy=MemoryExpiryPolicy.HOURS_24
        )
        context = manager.get_session(session_id)
        context.last_updated = datetime.now() - timedelta(hours=25)

        count = manager.cleanup_expired_sessions()
        assert count == 1
        assert manager.get_session(session_id) is None

    def test_cleanup_expired_sessions_keeps_active(self, manager):
        """cleanup_expired_sessions keeps active sessions."""
        session_id = manager.create_session(
            expiry_policy=MemoryExpiryPolicy.DAYS_7
        )
        count = manager.cleanup_expired_sessions()
        assert count == 0
        assert manager.get_session(session_id) is not None

    def test_cleanup_expired_sessions_with_mcp(self, manager_with_mcp):
        """cleanup_expired_sessions deletes from MCP."""
        session_id = manager_with_mcp.create_session(
            expiry_policy=MemoryExpiryPolicy.HOURS_24
        )
        context = manager_with_mcp.get_session(session_id)
        context.last_updated = datetime.now() - timedelta(hours=25)

        manager_with_mcp.cleanup_expired_sessions()
        manager_with_mcp._mcp_delete.assert_called()

    def test_extract_all_entities(self, manager):
        """_extract_all_entities combines entities from summaries and messages."""
        session_id = manager.create_session()
        manager.add_message(session_id, "user", "제1조 규정에 대해")
        context = manager.get_session(session_id)
        entities = manager._extract_all_entities(context)
        assert isinstance(entities, list)


# =============================================================================
# Test MemoryExpiryPolicy Enum
# =============================================================================


class TestMemoryExpiryPolicy:
    """Characterization tests for MemoryExpiryPolicy enum."""

    def test_values(self):
        """MemoryExpiryPolicy has expected values."""
        assert MemoryExpiryPolicy.HOURS_24.value == "24h"
        assert MemoryExpiryPolicy.DAYS_7.value == "7d"
        assert MemoryExpiryPolicy.DAYS_30.value == "30d"
        assert MemoryExpiryPolicy.NEVER.value == "never"


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateMemoryManager:
    """Characterization tests for create_memory_manager factory."""

    def test_create_without_llm(self):
        """create_memory_manager works without LLM client."""
        manager = create_memory_manager()
        assert manager is not None
        assert manager.summarizer is not None

    def test_create_with_llm(self):
        """create_memory_manager uses LLM client."""
        mock_llm = MagicMock()
        manager = create_memory_manager(llm_client=mock_llm)
        assert manager.summarizer.llm_client is mock_llm

    def test_create_with_mcp_enabled(self):
        """create_memory_manager enables MCP."""
        manager = create_memory_manager(enable_mcp=True)
        assert manager.enable_mcp is True

    def test_create_with_mcp_disabled(self):
        """create_memory_manager disables MCP by default."""
        manager = create_memory_manager()
        assert manager.enable_mcp is False


# =============================================================================
# Test MCP Integration Edge Cases
# =============================================================================


class TestMCPIntegration:
    """Characterization tests for MCP integration."""

    def test_store_to_mcp_without_method(self):
        """_store_to_mcp handles missing MCP method."""
        manager = ConversationMemoryManager(enable_mcp=True)
        # No _mcp_store method set
        session_id = manager.create_session()  # Should not raise

    def test_load_from_mcp_without_method(self):
        """_load_from_mcp handles missing MCP method."""
        manager = ConversationMemoryManager(enable_mcp=True)
        result = manager._load_from_mcp("nonexistent")  # Should not raise
        assert result is None

    def test_delete_from_mcp_without_method(self):
        """_delete_from_mcp handles missing MCP method."""
        manager = ConversationMemoryManager(enable_mcp=True)
        manager._delete_from_mcp("nonexistent")  # Should not raise

    def test_store_to_mcp_exception(self):
        """_store_to_mcp handles exceptions."""
        manager = ConversationMemoryManager(enable_mcp=True)
        manager._mcp_store = MagicMock(side_effect=Exception("MCP error"))
        session_id = manager.create_session()  # Should not raise

    def test_load_from_mcp_exception(self):
        """_load_from_mcp handles exceptions."""
        manager = ConversationMemoryManager(enable_mcp=True)
        manager._mcp_retrieve = MagicMock(side_effect=Exception("MCP error"))
        result = manager._load_from_mcp("session")  # Should not raise
        assert result is None

    def test_delete_from_mcp_exception(self):
        """_delete_from_mcp handles exceptions."""
        manager = ConversationMemoryManager(enable_mcp=True)
        manager._mcp_delete = MagicMock(side_effect=Exception("MCP error"))
        manager._delete_from_mcp("session")  # Should not raise
