"""
Conversation service for multi-turn conversation support.

Implements REQ-MUL-001 through REQ-MUL-015 including:
- Session management and persistence
- Context tracking across turns
- Topic change detection
- Reference resolution for pronouns
- Context-aware search methods
"""

import logging
from typing import Any, Dict, Optional

from ..domain.conversation.session import (
    ConversationSession,
    ConversationTurn,
)
from ..infrastructure.cache import CacheType, RAGQueryCache

logger = logging.getLogger(__name__)


class ConversationService:
    """
    Service for managing conversation sessions.

    Features:
    - Session creation and retrieval (REQ-MUL-001)
    - Context tracking (REQ-MUL-002)
    - Session timeout (REQ-MUL-003)
    - Follow-up query interpretation (REQ-MUL-004)
    - Context window management (REQ-MUL-005)
    - Session clearing (REQ-MUL-006)
    - Session persistence (REQ-MUL-007)
    - Conversation-aware search (REQ-MUL-008)
    - Reference resolution (REQ-MUL-009)
    - Topic change detection (REQ-MUL-010)
    - Feedback handling (REQ-MUL-011)
    - Context isolation (REQ-MUL-014)
    - Retention management (REQ-MUL-015)
    """

    def __init__(
        self,
        cache: Optional[RAGQueryCache] = None,
        timeout_minutes: int = 30,
        retention_hours: int = 24,
        context_window_size: int = 10,
        topic_change_threshold: float = 0.3,
    ):
        """
        Initialize conversation service.

        Args:
            cache: RAG query cache for session persistence.
            timeout_minutes: Session timeout in minutes (default 30, REQ-MUL-003).
            retention_hours: Session retention period in hours (default 24, REQ-MUL-015).
            context_window_size: Max turns in context window (default 10, REQ-MUL-005).
            topic_change_threshold: Semantic similarity threshold for topic detection (REQ-MUL-010).
        """
        self._cache = cache or RAGQueryCache(enabled=True, ttl_hours=1)
        self._timeout_minutes = timeout_minutes
        self._retention_hours = retention_hours
        self._context_window_size = context_window_size
        self._topic_change_threshold = topic_change_threshold

    def create_session(
        self,
        user_id: Optional[str] = None,
        timeout_minutes: Optional[int] = None,
    ) -> ConversationSession:
        """
        Create a new conversation session (REQ-MUL-001, REQ-MUL-006).

        Args:
            user_id: Optional user identifier for session tracking.
            timeout_minutes: Optional custom timeout (overrides default).

        Returns:
            New conversation session.
        """
        timeout = timeout_minutes or self._timeout_minutes

        session = ConversationSession.create(
            user_id=user_id,
            timeout_minutes=timeout,
            retention_hours=self._retention_hours,
        )

        self._save_session(session)
        logger.info(f"Created conversation session: {session.session_id}")

        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Retrieve a conversation session by ID (REQ-MUL-001).

        Args:
            session_id: Session identifier.

        Returns:
            Session if found and not expired, None otherwise.
        """
        cache_key = self._make_cache_key(session_id)
        entry = self._cache.get(
            cache_type=CacheType.LLM_RESPONSE,
            query=cache_key,
        )

        if not entry:
            return None

        session = ConversationSession.from_dict(entry["session"])

        # Check expiration
        if session.is_expired:
            session.mark_expired()
            self._save_session(session)
            logger.info(f"Session expired: {session_id}")
            return None

        return session

    def add_turn(
        self,
        session_id: str,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConversationTurn]:
        """
        Add a turn to the conversation (REQ-MUL-002).

        Args:
            session_id: Session identifier.
            query: User query.
            response: System response.
            metadata: Optional turn metadata.

        Returns:
            Created turn if session exists, None otherwise.
        """
        session = self.get_session(session_id)
        if not session:
            return None

        turn = session.add_turn(query, response, metadata)

        # Update context summary if needed (REQ-MUL-005)
        if session.turn_count > self._context_window_size:
            session.context_summary = self._summarize_early_turns(session)

        self._save_session(session)
        logger.debug(f"Added turn to session {session_id}: {turn.turn_id}")

        return turn

    def get_conversation_context(
        self,
        session_id: str,
        max_turns: Optional[int] = None,
    ) -> str:
        """
        Get conversation context for query expansion (REQ-MUL-008).

        Args:
            session_id: Session identifier.
            max_turns: Optional max turns to include (defaults to context_window_size).

        Returns:
            Formatted conversation context string.
        """
        session = self.get_session(session_id)
        if not session or session.turn_count == 0:
            return ""

        max_turns = max_turns or self._context_window_size
        turns = session.get_context_window(max_turns=max_turns)

        # Build context string
        context_parts = []

        # Add summary if available (REQ-MUL-005)
        if session.context_summary:
            context_parts.append(f"[이전 대화 요약: {session.context_summary}]")

        # Add recent turns
        for turn in turns:
            context_parts.append(f"Q: {turn.query}")
            context_parts.append(f"A: {turn.response[:100]}...")  # Truncate for brevity

        return "\n".join(context_parts)

    def expand_query_with_context(
        self,
        session_id: str,
        query: str,
    ) -> str:
        """
        Expand query using conversation context (REQ-MUL-004, REQ-MUL-009).

        Args:
            session_id: Session identifier.
            query: Current user query.

        Returns:
            Expanded query with context if applicable.
        """
        session = self.get_session(session_id)
        if not session or session.turn_count == 0:
            return query

        # Get last turn for context
        last_turn = session.turns[-1]

        # Simple reference resolution (REQ-MUL-009)
        # Check for pronouns and context references
        pronouns = ["그것", "그거", "그것", "이것", "이거", "저것", "저거", "해당"]
        has_pronoun = any(pronoun in query for pronoun in pronouns)

        if has_pronoun:
            # Extract last query context
            last_query = last_turn.query
            # Simple expansion: prepend last query's main topic
            # This is a basic implementation; advanced version would use NLP
            expanded_query = f"{last_query} {query}"
            logger.debug(f"Expanded query with context: {query} -> {expanded_query}")
            return expanded_query

        return query

    def detect_topic_change(
        self,
        session_id: str,
        current_query: str,
    ) -> bool:
        """
        Detect if current query indicates a topic change (REQ-MUL-010).

        Args:
            session_id: Session identifier.
            current_query: Current user query.

        Returns:
            True if topic change detected, False otherwise.
        """
        session = self.get_session(session_id)
        if not session or session.turn_count == 0:
            return False

        # Get last query
        last_query = session.turns[-1].query

        # Simple similarity detection using keyword overlap
        # Advanced implementation would use semantic similarity with embeddings
        similarity = self._compute_similarity(current_query, last_query)

        topic_changed = similarity < self._topic_change_threshold

        if topic_changed:
            logger.info(
                f"Topic change detected in session {session_id}: "
                f"similarity={similarity:.2f}"
            )

        return topic_changed

    def clear_session(self, session_id: str) -> bool:
        """
        Clear a conversation session (REQ-MUL-006).

        Args:
            session_id: Session identifier.

        Returns:
            True if session was cleared, False otherwise.
        """
        cache_key = self._make_cache_key(session_id)
        deleted = self._cache.invalidate(
            cache_type=CacheType.LLM_RESPONSE,
            query_pattern=cache_key,
        )

        if deleted:
            logger.info(f"Cleared conversation session: {session_id}")

        return deleted > 0

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired and retention-expired sessions.

        Returns:
            Number of sessions cleaned up.
        """
        # This is a placeholder - actual implementation would require
        # iterating through all session keys in cache
        # For now, sessions expire naturally through cache TTL
        logger.debug("Session cleanup called (cache handles expiration automatically)")
        return 0

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session statistics.

        Args:
            session_id: Session identifier.

        Returns:
            Session statistics or None if session not found.
        """
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "turn_count": session.turn_count,
            "status": session.status.value,
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "is_expired": session.is_expired,
            "context_summary": session.context_summary,
        }

    def _save_session(self, session: ConversationSession) -> None:
        """Save session to cache."""
        cache_key = self._make_cache_key(session.session_id)
        data = {"session": session.to_dict()}

        self._cache.set(
            cache_type=CacheType.LLM_RESPONSE,
            query=cache_key,
            data=data,
        )

    def _make_cache_key(self, session_id: str) -> str:
        """Create cache key for session."""
        return f"conversation_session:{session_id}"

    def _summarize_early_turns(self, session: ConversationSession) -> str:
        """
        Summarize early turns when context window is exceeded (REQ-MUL-005).

        Args:
            session: Session with turns to summarize.

        Returns:
            Summary string.
        """
        # Get early turns (those outside context window)
        early_turns = session.turns[: -self._context_window_size]

        # Simple summary: extract main topics
        topics = []
        for turn in early_turns:
            # Extract keywords from query (basic implementation)
            words = turn.query.split()
            topics.extend(words[:3])  # Take first 3 words

        # Create simple summary
        if topics:
            unique_topics = list(set(topics))[:5]
            return f"이전에 {', '.join(unique_topics)}에 대해 질문하셨습니다."

        return "이전 대화가 있습니다."

    def _compute_similarity(self, query1: str, query2: str) -> float:
        """
        Compute semantic similarity between two queries (REQ-MUL-010).

        Basic implementation using word overlap.
        Advanced implementation would use embeddings.

        Args:
            query1: First query.
            query2: Second query.

        Returns:
            Similarity score from 0.0 to 1.0.
        """
        # Simple word overlap similarity
        words1 = set(query1.split())
        words2 = set(query2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0
