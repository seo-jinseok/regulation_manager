"""
Unit tests for conversation service.
"""

import time

from src.rag.application.conversation_service import ConversationService
from src.rag.domain.conversation.session import (
    SessionStatus,
)


class TestConversationService:
    """Tests for ConversationService."""

    def test_create_session(self):
        """Test creating a new session (REQ-MUL-001, REQ-MUL-006)."""
        service = ConversationService()
        session = service.create_session()

        assert session.session_id is not None
        assert session.turn_count == 0
        assert session.status == SessionStatus.ACTIVE

    def test_create_session_with_user_id(self):
        """Test creating session with user ID."""
        service = ConversationService()
        session = service.create_session(user_id="user-123")

        assert session.user_id == "user-123"

    def test_create_session_custom_timeout(self):
        """Test creating session with custom timeout."""
        service = ConversationService()
        session = service.create_session(timeout_minutes=60)

        assert session.timeout_minutes == 60

    def test_get_existing_session(self):
        """Test retrieving an existing session (REQ-MUL-001)."""
        service = ConversationService()
        created = service.create_session()

        retrieved = service.get_session(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_nonexistent_session(self):
        """Test retrieving a nonexistent session."""
        service = ConversationService()

        retrieved = service.get_session("nonexistent-id")

        assert retrieved is None

    def test_get_expired_session(self):
        """Test that expired sessions return None (REQ-MUL-003)."""
        # Create service with very short timeout
        service = ConversationService(timeout_minutes=0.01)  # 0.6 seconds
        session = service.create_session()
        session_id = session.session_id

        # Wait for expiration
        time.sleep(1)

        # Session should be expired
        retrieved = service.get_session(session_id)
        assert retrieved is None

    def test_add_turn_to_session(self):
        """Test adding a turn to session (REQ-MUL-002)."""
        service = ConversationService()
        session = service.create_session()

        turn = service.add_turn(
            session_id=session.session_id,
            query="휴학 방법",
            response="휴학 신청서 제출",
        )

        assert turn is not None
        assert turn.query == "휴학 방법"
        assert turn.response == "휴학 신청서 제출"

    def test_add_turn_with_metadata(self):
        """Test adding turn with metadata."""
        service = ConversationService()
        session = service.create_session()

        metadata = {"regulation": "학칙", "article": "제12조"}
        turn = service.add_turn(
            session_id=session.session_id,
            query="학칙 알려줘",
            response="학칙은...",
            metadata=metadata,
        )

        assert turn.metadata == metadata

    def test_add_turn_to_nonexistent_session(self):
        """Test adding turn to nonexistent session."""
        service = ConversationService()

        turn = service.add_turn(
            session_id="nonexistent",
            query="test",
            response="test",
        )

        assert turn is None

    def test_get_conversation_context_empty_session(self):
        """Test getting context from empty session."""
        service = ConversationService()
        session = service.create_session()

        context = service.get_conversation_context(session.session_id)

        assert context == ""

    def test_get_conversation_context_with_turns(self):
        """Test getting conversation context (REQ-MUL-008)."""
        service = ConversationService()
        session = service.create_session()

        service.add_turn(
            session_id=session.session_id,
            query="휴학 방법",
            response="휴학 신청서를 제출하세요",
        )
        service.add_turn(
            session_id=session.session_id,
            query="서류는 뭐가 필요해?",
            response="신청서와 성적증명서",
        )

        context = service.get_conversation_context(session.session_id)

        assert "휴학 방법" in context
        assert "서류는 뭐가 필요해?" in context

    def test_get_conversation_context_respects_max_turns(self):
        """Test context respects max_turns parameter (REQ-MUL-005)."""
        service = ConversationService(context_window_size=3)
        session = service.create_session()

        # Add 5 turns
        for i in range(5):
            service.add_turn(
                session_id=session.session_id,
                query=f"질문 {i}",
                response=f"답변 {i}",
            )

        context = service.get_conversation_context(session.session_id, max_turns=2)

        # Should only have last 2 turns
        assert "질문 3" in context
        assert "질문 4" in context
        assert "질문 0" not in context

    def test_context_summary_created_when_window_exceeded(self):
        """Test context summary when window exceeded (REQ-MUL-005)."""
        service = ConversationService(context_window_size=3)
        session = service.create_session()

        # Add turns beyond context window
        for i in range(5):
            service.add_turn(
                session_id=session.session_id,
                query=f"질문 {i}",
                response=f"답변 {i}",
            )

        # Retrieve session to check summary
        retrieved = service.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.context_summary != ""

    def test_expand_query_with_context_no_session(self):
        """Test query expansion with no session."""
        service = ConversationService()

        expanded = service.expand_query_with_context("session-123", "그거 알려줘")

        assert expanded == "그거 알려줘"

    def test_expand_query_with_context_empty_session(self):
        """Test query expansion with empty session."""
        service = ConversationService()
        session = service.create_session()

        expanded = service.expand_query_with_context(session.session_id, "그거 알려줘")

        # No context to expand with
        assert expanded == "그거 알려줘"

    def test_expand_query_with_pronoun(self):
        """Test query expansion with pronoun reference (REQ-MUL-009)."""
        service = ConversationService()
        session = service.create_session()

        # Add a turn about specific regulation
        service.add_turn(
            session_id=session.session_id,
            query="휴학 규정",
            response="휴학 신청서 제출",
        )

        # Follow-up with pronoun
        expanded = service.expand_query_with_context(
            session.session_id,
            "그거 자세히 알려줘",
        )

        # Should expand with previous context
        assert "휴학 규정" in expanded

    def test_expand_query_without_pronoun(self):
        """Test query without pronoun is not expanded."""
        service = ConversationService()
        session = service.create_session()

        service.add_turn(
            session_id=session.session_id,
            query="휴학 규정",
            response="휴학 신청서 제출",
        )

        # Direct question without pronoun
        expanded = service.expand_query_with_context(
            session.session_id,
            "등록금은 어떻게 되나요?",
        )

        # Should not expand
        assert expanded == "등록금은 어떻게 되나요?"

    def test_detect_topic_change_no_session(self):
        """Test topic detection with no session."""
        service = ConversationService()

        changed = service.detect_topic_change("session-123", "등록금 알려줘")

        assert not changed

    def test_detect_topic_change_empty_session(self):
        """Test topic detection with empty session."""
        service = ConversationService()
        session = service.create_session()

        changed = service.detect_topic_change(session.session_id, "등록금 알려줘")

        assert not changed

    def test_detect_topic_change_same_topic(self):
        """Test topic detection with high word overlap (REQ-MUL-010)."""
        service = ConversationService()
        session = service.create_session()

        service.add_turn(
            session_id=session.session_id,
            query="휴학 방법 알려줘",
            response="휴학 신청서 제출",
        )

        # Query with higher word overlap
        changed = service.detect_topic_change(
            session.session_id,
            "휴학 방법과 절차 알려줘",
        )

        # Should not detect topic change (similarity > threshold)
        # Both have "휴학", "방법" - high overlap
        assert not changed

    def test_detect_topic_change_different_topic(self):
        """Test topic detection with different topic (REQ-MUL-010)."""
        service = ConversationService()
        session = service.create_session()

        service.add_turn(
            session_id=session.session_id,
            query="휴학 방법 알려줘",
            response="휴학 신청서 제출",
        )

        # Completely different topic
        changed = service.detect_topic_change(
            session.session_id,
            "등록금 납부 기간이 언제예요?",
        )

        # Should detect topic change (low similarity)
        assert changed

    def test_clear_session(self):
        """Test clearing a session (REQ-MUL-006)."""
        service = ConversationService()
        session = service.create_session()
        session_id = session.session_id

        cleared = service.clear_session(session_id)

        assert cleared

        # Session should no longer exist
        retrieved = service.get_session(session_id)
        assert retrieved is None

    def test_clear_nonexistent_session(self):
        """Test clearing nonexistent session."""
        service = ConversationService()

        cleared = service.clear_session("nonexistent")

        assert not cleared

    def test_get_session_stats(self):
        """Test getting session statistics."""
        service = ConversationService()
        session = service.create_session()
        service.add_turn(
            session_id=session.session_id,
            query="test",
            response="response",
        )

        stats = service.get_session_stats(session.session_id)

        assert stats is not None
        assert stats["session_id"] == session.session_id
        assert stats["turn_count"] == 1
        assert stats["status"] == SessionStatus.ACTIVE.value

    def test_get_session_stats_nonexistent(self):
        """Test getting stats for nonexistent session."""
        service = ConversationService()

        stats = service.get_session_stats("nonexistent")

        assert stats is None

    def test_default_configuration(self):
        """Test default configuration values."""
        service = ConversationService()

        assert service._timeout_minutes == 30  # REQ-MUL-003
        assert service._retention_hours == 24  # REQ-MUL-015
        assert service._context_window_size == 10  # REQ-MUL-005
        assert service._topic_change_threshold == 0.3  # REQ-MUL-010

    def test_custom_configuration(self):
        """Test custom configuration."""
        service = ConversationService(
            timeout_minutes=60,
            retention_hours=48,
            context_window_size=5,
            topic_change_threshold=0.5,
        )

        assert service._timeout_minutes == 60
        assert service._retention_hours == 48
        assert service._context_window_size == 5
        assert service._topic_change_threshold == 0.5

    def test_session_persistence_across_retrievals(self):
        """Test session persists across multiple retrievals."""
        service = ConversationService()
        session = service.create_session()

        # Add a turn
        service.add_turn(
            session_id=session.session_id,
            query="질문 1",
            response="답변 1",
        )

        # Retrieve again
        retrieved = service.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.turn_count == 1

        # Add another turn
        service.add_turn(
            session_id=session.session_id,
            query="질문 2",
            response="답변 2",
        )

        # Retrieve again
        retrieved = service.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.turn_count == 2

    def test_context_isolation_between_sessions(self):
        """Test sessions don't leak context (REQ-MUL-014)."""
        service = ConversationService()

        # Create two sessions
        session1 = service.create_session(user_id="user-1")
        session2 = service.create_session(user_id="user-2")

        # Add turns to session1
        service.add_turn(
            session_id=session1.session_id,
            query="휴학 방법",
            response="휴학 신청서 제출",
        )

        # Add turns to session2
        service.add_turn(
            session_id=session2.session_id,
            query="등록금 납부",
            response="등록금은 학기 시작 전",
        )

        # Get context for each
        context1 = service.get_conversation_context(session1.session_id)
        context2 = service.get_conversation_context(session2.session_id)

        # Contexts should be isolated
        assert "휴학 방법" in context1
        assert "등록금 납부" not in context1

        assert "등록금 납부" in context2
        assert "휴학 방법" not in context2

    def test_compute_similarity_same_text(self):
        """Test similarity computation with identical text."""
        service = ConversationService()

        similarity = service._compute_similarity("휴학 방법", "휴학 방법")

        assert similarity == 1.0

    def test_compute_similarity_different_text(self):
        """Test similarity computation with different text."""
        service = ConversationService()

        similarity = service._compute_similarity("휴학 방법", "등록금 납부")

        assert similarity == 0.0

    def test_compute_similarity_partial_overlap(self):
        """Test similarity computation with partial overlap."""
        service = ConversationService()

        similarity = service._compute_similarity("휴학 방법과 절차", "휴학 방법 알려줘")

        # Should have some overlap due to shared words
        assert similarity > 0.0
        assert similarity < 1.0
