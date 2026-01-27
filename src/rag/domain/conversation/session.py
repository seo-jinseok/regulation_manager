"""
Conversation session domain entities.

Implements REQ-MUL-001 through REQ-MUL-015 for multi-turn conversation support.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SessionStatus(Enum):
    """Status of a conversation session (REQ-MUL-003)."""

    ACTIVE = "active"
    EXPIRED = "expired"
    ARCHIVED = "archived"


@dataclass
class ConversationTurn:
    """
    A single turn in a conversation.

    A turn consists of a user query and the system's response.
    (REQ-MUL-002)
    """

    turn_id: str
    timestamp: float
    query: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "response": self.response,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            turn_id=data["turn_id"],
            timestamp=data["timestamp"],
            query=data["query"],
            response=data["response"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationSession:
    """
    A conversation session tracking multiple turns.

    Implements:
    - REQ-MUL-001: Maintain conversation session state
    - REQ-MUL-002: Track conversation context
    - REQ-MUL-003: Session timeout (default 30 minutes)
    - REQ-MUL-007: Session persistence
    - REQ-MUL-014: No context leakage between sessions
    - REQ-MUL-015: Session retention period (default 24 hours)
    """

    session_id: str
    user_id: Optional[str]
    turns: List[ConversationTurn] = field(default_factory=list)
    context_summary: str = ""
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    status: SessionStatus = SessionStatus.ACTIVE
    timeout_minutes: int = 30  # REQ-MUL-003
    retention_hours: int = 24  # REQ-MUL-015

    @property
    def is_expired(self) -> bool:
        """Check if session has expired (REQ-MUL-003, REQ-MUL-007)."""
        if self.status != SessionStatus.ACTIVE:
            return True

        elapsed_seconds = time.time() - self.last_activity
        elapsed_minutes = elapsed_seconds / 60
        return elapsed_minutes >= self.timeout_minutes

    @property
    def is_retention_expired(self) -> bool:
        """Check if session should be deleted (REQ-MUL-015)."""
        elapsed_seconds = time.time() - self.created_at
        elapsed_hours = elapsed_seconds / 3600
        return elapsed_hours >= self.retention_hours

    @property
    def turn_count(self) -> int:
        """Get number of turns in session."""
        return len(self.turns)

    def add_turn(
        self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """
        Add a new turn to the session.

        Updates last_activity timestamp (REQ-MUL-003).
        """
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            timestamp=time.time(),
            query=query,
            response=response,
            metadata=metadata or {},
        )
        self.turns.append(turn)
        self.last_activity = time.time()
        return turn

    def get_recent_turns(self, max_turns: int = 10) -> List[ConversationTurn]:
        """
        Get most recent turns.

        Implements REQ-MUL-005: Context window management (10 turns).
        """
        return self.turns[-max_turns:]

    def get_context_window(self, max_turns: int = 10) -> List[ConversationTurn]:
        """
        Get turns within context window.

        For sessions with more than max_turns, returns recent turns and
        relies on context_summary for early context (REQ-MUL-005).
        """
        if self.turn_count <= max_turns:
            return self.turns

        # Return recent turns when over window
        # Context summary should include summarized early turns
        return self.turns[-max_turns:]

    def mark_expired(self) -> None:
        """Mark session as expired (REQ-MUL-007)."""
        self.status = SessionStatus.EXPIRED

    def archive(self) -> None:
        """Archive session for long-term storage."""
        self.status = SessionStatus.ARCHIVED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "turns": [turn.to_dict() for turn in self.turns],
            "context_summary": self.context_summary,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "status": self.status.value,
            "timeout_minutes": self.timeout_minutes,
            "retention_hours": self.retention_hours,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Create from dictionary."""
        turns = [ConversationTurn.from_dict(t) for t in data.get("turns", [])]

        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            turns=turns,
            context_summary=data.get("context_summary", ""),
            created_at=data["created_at"],
            last_activity=data["last_activity"],
            status=SessionStatus(data.get("status", SessionStatus.ACTIVE.value)),
            timeout_minutes=data.get("timeout_minutes", 30),
            retention_hours=data.get("retention_hours", 24),
        )

    @classmethod
    def create(
        cls,
        user_id: Optional[str] = None,
        timeout_minutes: int = 30,
        retention_hours: int = 24,
    ) -> "ConversationSession":
        """Create a new conversation session."""
        return cls(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            timeout_minutes=timeout_minutes,
            retention_hours=retention_hours,
        )
