"""
Unit tests for conversation session domain entity.
"""

import time

from src.rag.domain.conversation.session import (
    ConversationSession,
    ConversationTurn,
    SessionStatus,
)


class TestConversationTurn:
    """Tests for ConversationTurn entity."""

    def test_create_turn(self):
        """Test creating a conversation turn."""
        turn = ConversationTurn(
            turn_id="turn-1",
            timestamp=time.time(),
            query="휴학은 어떻게 하나요?",
            response="휴학 신청서를 제출해야 합니다.",
        )

        assert turn.turn_id == "turn-1"
        assert turn.query == "휴학은 어떻게 하나요?"
        assert turn.response == "휴학 신청서를 제출해야 합니다."

    def test_turn_with_metadata(self):
        """Test turn with metadata."""
        metadata = {"regulation": "학칙", "articles": ["제12조"]}
        turn = ConversationTurn(
            turn_id="turn-2",
            timestamp=time.time(),
            query="학칙 알려줘",
            response="학칙은...",
            metadata=metadata,
        )

        assert turn.metadata == metadata
        assert turn.metadata["regulation"] == "학칙"

    def test_turn_serialization(self):
        """Test turn to_dict and from_dict."""
        original = ConversationTurn(
            turn_id="turn-3",
            timestamp=1234567890.0,
            query="제7조가 뭐예요?",
            response="제7조는...",
            metadata={"article": "제7조"},
        )

        data = original.to_dict()
        restored = ConversationTurn.from_dict(data)

        assert restored.turn_id == original.turn_id
        assert restored.timestamp == original.timestamp
        assert restored.query == original.query
        assert restored.response == original.response
        assert restored.metadata == original.metadata


class TestConversationSession:
    """Tests for ConversationSession entity."""

    def test_create_session(self):
        """Test creating a new session (REQ-MUL-001)."""
        session = ConversationSession.create(user_id="user-123")

        assert session.session_id is not None
        assert session.user_id == "user-123"
        assert session.turn_count == 0
        assert session.status == SessionStatus.ACTIVE
        assert session.context_summary == ""

    def test_add_turn(self):
        """Test adding a turn to session (REQ-MUL-002)."""
        session = ConversationSession.create()
        initial_activity = session.last_activity

        # Add a turn
        turn = session.add_turn(
            query="휴학 방법",
            response="휴학 신청서 제출",
        )

        assert session.turn_count == 1
        assert turn.query == "휴학 방법"
        assert turn.response == "휴학 신청서 제출"
        assert session.last_activity >= initial_activity

    def test_add_multiple_turns(self):
        """Test adding multiple turns."""
        session = ConversationSession.create()

        session.add_turn("질문 1", "답변 1")
        session.add_turn("질문 2", "답변 2")
        session.add_turn("질문 3", "답변 3")

        assert session.turn_count == 3
        assert session.turns[0].query == "질문 1"
        assert session.turns[1].query == "질문 2"
        assert session.turns[2].query == "질문 3"

    def test_get_recent_turns_within_limit(self):
        """Test getting recent turns when within limit (REQ-MUL-005)."""
        session = ConversationSession.create()

        for i in range(5):
            session.add_turn(f"질문 {i}", f"답변 {i}")

        recent = session.get_recent_turns(max_turns=10)

        assert len(recent) == 5

    def test_get_recent_turns_exceeds_limit(self):
        """Test getting recent turns when exceeding limit (REQ-MUL-005)."""
        session = ConversationSession.create()

        # Add 15 turns
        for i in range(15):
            session.add_turn(f"질문 {i}", f"답변 {i}")

        # Get last 10
        recent = session.get_recent_turns(max_turns=10)

        assert len(recent) == 10
        assert recent[0].query == "질문 5"
        assert recent[-1].query == "질문 14"

    def test_context_window_same_as_recent(self):
        """Test context window management (REQ-MUL-005)."""
        session = ConversationSession.create()

        for i in range(5):
            session.add_turn(f"질문 {i}", f"답변 {i}")

        context_window = session.get_context_window(max_turns=10)

        assert len(context_window) == 5

    def test_context_window_exceeds_limit(self):
        """Test context window when exceeding limit (REQ-MUL-005)."""
        session = ConversationSession.create()

        # Add 15 turns - more than default 10 turn window
        for i in range(15):
            session.add_turn(f"질문 {i}", f"답변 {i}")

        context_window = session.get_context_window(max_turns=10)

        assert len(context_window) == 10
        # Should return last 10 turns
        assert context_window[0].query == "질문 5"
        assert context_window[-1].query == "질문 14"

    def test_session_not_expired_initially(self):
        """Test new session is not expired (REQ-MUL-003)."""
        session = ConversationSession.create()

        assert not session.is_expired
        assert session.status == SessionStatus.ACTIVE

    def test_session_expires_after_timeout(self):
        """Test session expires after timeout (REQ-MUL-003, REQ-MUL-007)."""
        # Create session with 1 second timeout for testing
        session = ConversationSession.create(timeout_minutes=0.01)  # 0.6 seconds

        # Should not be expired immediately
        assert not session.is_expired

        # Wait for expiration
        time.sleep(1)

        # Should now be expired
        assert session.is_expired

    def test_mark_session_expired(self):
        """Test marking session as expired (REQ-MUL-007)."""
        session = ConversationSession.create()

        session.mark_expired()

        assert session.status == SessionStatus.EXPIRED
        assert session.is_expired

    def test_archive_session(self):
        """Test archiving a session."""
        session = ConversationSession.create()

        session.archive()

        assert session.status == SessionStatus.ARCHIVED

    def test_session_serialization(self):
        """Test session to_dict and from_dict."""
        original = ConversationSession.create(user_id="user-456")
        original.add_turn("질문", "답변", {"key": "value"})
        original.context_summary = "휴학 규정에 대한 문의"

        data = original.to_dict()
        restored = ConversationSession.from_dict(data)

        assert restored.session_id == original.session_id
        assert restored.user_id == original.user_id
        assert restored.turn_count == original.turn_count
        assert restored.context_summary == original.context_summary
        assert restored.status == original.status

    def test_retention_expiration(self):
        """Test session retention expiration (REQ-MUL-015)."""
        # Create session with 1 hour retention for testing
        session = ConversationSession.create(retention_hours=0.01)  # 0.6 minutes

        # Create a turn to update timestamp
        session.add_turn("test", "test")

        # Should not be retention expired immediately
        assert not session.is_retention_expired

        # Modify created_at to simulate old session
        session.created_at = time.time() - 3700  # ~1 hour ago

        # Should now be retention expired
        assert session.is_retention_expired

    def test_default_timeout_30_minutes(self):
        """Test default timeout is 30 minutes (REQ-MUL-003)."""
        session = ConversationSession.create()

        assert session.timeout_minutes == 30

    def test_default_retention_24_hours(self):
        """Test default retention is 24 hours (REQ-MUL-015)."""
        session = ConversationSession.create()

        assert session.retention_hours == 24

    def test_custom_timeout(self):
        """Test custom timeout configuration."""
        session = ConversationSession.create(timeout_minutes=60)

        assert session.timeout_minutes == 60

    def test_custom_retention(self):
        """Test custom retention configuration."""
        session = ConversationSession.create(retention_hours=48)

        assert session.retention_hours == 48
