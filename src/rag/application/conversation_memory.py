"""
Conversation Memory Management for RAG System.

Provides long-term memory capabilities including:
- Conversation summarization
- Topic inference
- Memory persistence with MCP Memory Server
- Privacy-aware context management
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class MemoryExpiryPolicy(Enum):
    """Memory expiry policies."""

    HOURS_24 = "24h"
    DAYS_7 = "7d"
    DAYS_30 = "30d"
    NEVER = "never"


@dataclass
class Message:
    """A single message in conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationSummary:
    """Summary of a conversation segment."""

    summary: str
    key_topics: List[str]
    key_entities: List[str]  # Regulation names, article numbers, dates
    timestamp: datetime = field(default_factory=datetime.now)
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "summary": self.summary,
            "key_topics": self.key_topics,
            "key_entities": self.key_entities,
            "timestamp": self.timestamp.isoformat(),
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSummary":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            summary=data["summary"],
            key_topics=data.get("key_topics", []),
            key_entities=data.get("key_entities", []),
            timestamp=timestamp,
            message_count=data.get("message_count", 0),
        )


@dataclass
class ConversationContext:
    """Complete context for a conversation session."""

    session_id: str
    user_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    summaries: List[ConversationSummary] = field(default_factory=list)
    current_topics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    expiry_policy: MemoryExpiryPolicy = MemoryExpiryPolicy.DAYS_7

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a message to the conversation."""
        self.messages.append(
            Message(role=role, content=content, metadata=metadata or {})
        )
        self.last_updated = datetime.now()

    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """Get recent messages."""
        return self.messages[-limit:] if self.messages else []

    def is_expired(self) -> bool:
        """Check if context has expired."""
        if self.expiry_policy == MemoryExpiryPolicy.NEVER:
            return False

        expiry_hours = {
            MemoryExpiryPolicy.HOURS_24: 24,
            MemoryExpiryPolicy.DAYS_7: 24 * 7,
            MemoryExpiryPolicy.DAYS_30: 24 * 30,
        }.get(self.expiry_policy, 24 * 7)

        return datetime.now() - self.last_updated > timedelta(hours=expiry_hours)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
            "summaries": [s.to_dict() for s in self.summaries],
            "current_topics": self.current_topics,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "expiry_policy": self.expiry_policy.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        else:
            last_updated = datetime.now()

        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            summaries=[
                ConversationSummary.from_dict(s) for s in data.get("summaries", [])
            ],
            current_topics=data.get("current_topics", []),
            created_at=created_at,
            last_updated=last_updated,
            expiry_policy=MemoryExpiryPolicy(data.get("expiry_policy", "7d")),
        )


class ConversationSummarizer:
    """
    Summarizes conversations to maintain long-term context.

    Extracts key information including:
    - Regulation names and article numbers
    - Important dates and deadlines
    - User questions and concerns
    - Action items and follow-ups
    """

    # Patterns for extracting key information
    REGULATION_PATTERN = re.compile(r"(\w+[규정칙])")
    ARTICLE_PATTERN = re.compile(r"(제\d+조[제\d항항?])")
    DATE_PATTERN = re.compile(
        r"(\d{4}[-년]\s*\d{1,2}[-월]\s*\d{1,2}일?|내일|모레|다음 주|이번 주)"
    )

    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize summarizer.

        Args:
            llm_client: Optional LLM client for enhanced summarization
        """
        self.llm_client = llm_client

    def should_summarize(
        self, context: ConversationContext, max_messages: int = 10
    ) -> bool:
        """
        Determine if conversation should be summarized.

        Args:
            context: Conversation context
            max_messages: Maximum messages before summarization

        Returns:
            True if summarization is needed
        """
        return len(context.messages) >= max_messages

    def summarize(self, context: ConversationContext) -> ConversationSummary:
        """
        Summarize conversation messages.

        Args:
            context: Conversation context to summarize

        Returns:
            ConversationSummary with extracted information
        """
        messages_text = "\n".join([f"{m.role}: {m.content}" for m in context.messages])

        # Extract key information using patterns
        key_entities = self._extract_entities(messages_text)
        key_topics = self._infer_topics(context.messages)

        # Generate summary
        if self.llm_client:
            summary = self._summarize_with_llm(messages_text, key_entities, key_topics)
        else:
            summary = self._summarize_with_rules(messages_text, key_entities)

        return ConversationSummary(
            summary=summary,
            key_topics=key_topics,
            key_entities=key_entities,
            message_count=len(context.messages),
        )

    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities (regulations, articles, dates)."""
        entities = set()

        # Extract regulation names
        for match in self.REGULATION_PATTERN.finditer(text):
            entities.add(match.group(1))

        # Extract article references
        for match in self.ARTICLE_PATTERN.finditer(text):
            entities.add(match.group(1))

        # Extract dates
        for match in self.DATE_PATTERN.finditer(text):
            entities.add(match.group(1))

        return sorted(list(entities))

    def _infer_topics(self, messages: List[Message]) -> List[str]:
        """
        Infer topics from conversation messages.

        Uses keyword analysis to identify main topics.
        """
        # Topic keywords with weights
        topic_keywords = {
            "휴가": ["휴가", "연차", "병가", "공가"],
            "승진": ["승진", "승급", "진용", "임용"],
            "복지": ["급여", "수당", "복지", "지원"],
            "징계": ["징계", "견책", "정직", "해임"],
            "연구": ["연구", "연구비", "학술", "논문"],
            "교육": ["교육", "수업", "강의", "출석"],
            "인사": ["인사", "임용", "채용", "전보"],
            "평가": ["평가", "성과", "업적", "심사"],
        }

        # Combine all message content
        all_text = " ".join([m.content for m in messages])

        # Score topics based on keyword matches
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            if score > 0:
                topic_scores[topic] = score

        # Return top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:3]]

    def _summarize_with_rules(self, text: str, entities: List[str]) -> str:
        """Generate summary using rule-based approach."""
        # Split into sentences
        sentences = re.split(r"[.!?]+\s+", text)

        # Select key sentences (those containing entities)
        key_sentences = [
            s for s in sentences if any(entity in s for entity in entities)
        ]

        if key_sentences:
            summary = " ".join(key_sentences[:3])
        else:
            # Fallback to first few sentences
            summary = " ".join(sentences[:3])

        return f"대화 요약: {summary}"

    def _summarize_with_llm(
        self, text: str, entities: List[str], topics: List[str]
    ) -> str:
        """Generate summary using LLM."""
        try:
            prompt = f"""다음 대화를 간결하게 요약해주세요.

주제: {", ".join(topics)}
관련 규정/조항: {", ".join(entities)}

대화 내용:
{text}

요약:"""

            if hasattr(self.llm_client, "generate"):
                response = self.llm_client.generate(prompt, max_tokens=200)
                return response.strip()
            elif hasattr(self.llm_client, "complete"):
                response = self.llm_client.complete(prompt, max_tokens=200)
                return response.strip()
            else:
                logger.warning("LLM client does not support generate/complete")
                return self._summarize_with_rules(text, entities)
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            return self._summarize_with_rules(text, entities)


class TopicInference:
    """
    Infers topics from conversation for context-aware retrieval.

    Topics improve retrieval by:
    - Providing additional query terms
    - Focusing search on relevant regulations
    - Understanding user intent
    """

    # Topic-specific search term expansions
    TOPIC_EXPANSIONS = {
        "휴가": ["연차", "휴가일", "휴가청구"],
        "승진": ["승급", "진용", "임용周期", "승진심사"],
        "복지": ["급여", "수당", "지원금", "복지급여"],
        "징계": ["징계처분", "견책", "정직", "해임", "감봉"],
        "연구": ["연구비", "연구지원", "학술연구", "논문심사"],
        "교육": ["수업", "강의", "교육과정", "출석인정"],
        "인사": ["임용", "채용", "전보", "인사발령"],
        "평가": ["성과평가", "업적평정", "심사", "연구실적"],
    }

    def infer_topics(self, context: ConversationContext) -> List[str]:
        """
        Infer current topics from conversation context.

        Args:
            context: Conversation context

        Returns:
            List of inferred topics
        """
        # Combine recent messages and summaries
        recent_messages = context.get_recent_messages(limit=5)
        text = " ".join([m.content for m in recent_messages])

        # Add summary context
        if context.summaries:
            summary_text = " ".join([s.summary for s in context.summaries[-2:]])
            text += " " + summary_text

        # Infer topics
        topics = []
        for topic, keywords in self.TOPIC_EXPANSIONS.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)

        # Add existing topics if still relevant
        for existing_topic in context.current_topics:
            if existing_topic not in topics:
                # Check if topic is still mentioned
                topic_keywords = self.TOPIC_EXPANSIONS.get(existing_topic, [])
                if any(keyword in text for keyword in topic_keywords[:3]):
                    topics.append(existing_topic)

        return topics

    def expand_query_with_topics(self, query: str, topics: List[str]) -> str:
        """
        Expand query with topic-related terms.

        Args:
            query: Original search query
            topics: Inferred topics

        Returns:
            Expanded query string
        """
        if not topics:
            return query

        # Add topic-specific terms
        expansion_terms = []
        for topic in topics:
            keywords = self.TOPIC_EXPANSIONS.get(topic, [])[:3]
            expansion_terms.extend(keywords)

        # Combine with original query
        if expansion_terms:
            expanded = f"{query} ({' '.join(expansion_terms)})"
            return expanded

        return query


class ConversationMemoryManager:
    """
    Manages conversation memory with persistence and summarization.

    Features:
    - Session-based memory management
    - Automatic summarization
    - Topic inference
    - Privacy-aware filtering
    - MCP Memory Server integration
    """

    def __init__(
        self,
        summarizer: Optional[ConversationSummarizer] = None,
        topic_inference: Optional[TopicInference] = None,
        enable_mcp: bool = False,
    ):
        """
        Initialize memory manager.

        Args:
            summarizer: Conversation summarizer
            topic_inference: Topic inference engine
            enable_mcp: Enable MCP Memory Server integration
        """
        self.summarizer = summarizer or ConversationSummarizer()
        self.topic_inference = topic_inference or TopicInference()
        self.enable_mcp = enable_mcp

        # In-memory session storage
        self._sessions: Dict[str, ConversationContext] = {}

        # Privacy filters (patterns to redact)
        self._privacy_patterns = [
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # Phone number
            re.compile(r"\b\d{2}\s*\d{2}\s*\d{2}\s*-\s*\d{6}\b"),  # RRN
            re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),  # Email
        ]

    def create_session(
        self,
        user_id: Optional[str] = None,
        expiry_policy: MemoryExpiryPolicy = MemoryExpiryPolicy.DAYS_7,
    ) -> str:
        """
        Create a new conversation session.

        Args:
            user_id: Optional user identifier
            expiry_policy: Memory expiry policy

        Returns:
            Session ID
        """
        session_id = str(uuid4())
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            expiry_policy=expiry_policy,
        )

        self._sessions[session_id] = context

        # Persist to MCP if enabled
        if self.enable_mcp:
            self._store_to_mcp(session_id, context)

        logger.info(f"Created session {session_id} for user {user_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """
        Get conversation context for session.

        Args:
            session_id: Session identifier

        Returns:
            ConversationContext or None if not found
        """
        # Check in-memory cache
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Try loading from MCP
        if self.enable_mcp:
            context = self._load_from_mcp(session_id)
            if context:
                self._sessions[session_id] = context
                return context

        return None

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConversationContext]:
        """
        Add a message to conversation session.

        Args:
            session_id: Session identifier
            role: Message role ("user" or "assistant")
            content: Message content
            metadata: Optional metadata

        Returns:
            Updated context or None if session not found
        """
        context = self.get_session(session_id)
        if not context:
            logger.warning(f"Session {session_id} not found")
            return None

        # Filter sensitive information
        filtered_content = self._filter_sensitive_info(content)

        # Add message
        context.add_message(role, filtered_content, metadata)

        # Check if summarization is needed
        if self.summarizer.should_summarize(context):
            summary = self.summarizer.summarize(context)
            context.summaries.append(summary)

            # Clear old messages (keep recent ones)
            context.messages = context.get_recent_messages(limit=5)

            logger.info(f"Summarized session {session_id}: {summary.summary[:100]}...")

        # Update topics
        context.current_topics = self.topic_inference.infer_topics(context)

        # Persist to MCP
        if self.enable_mcp:
            self._store_to_mcp(session_id, context)

        return context

    def get_context_for_search(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get formatted context for search query enhancement.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with context information or None
        """
        context = self.get_session(session_id)
        if not context:
            return None

        return {
            "session_id": session_id,
            "current_topics": context.current_topics,
            "recent_messages": [
                m.to_dict() for m in context.get_recent_messages(limit=3)
            ],
            "summaries": [s.to_dict() for s in context.summaries[-2:]],
            "key_entities": self._extract_all_entities(context),
        }

    def expand_query(self, session_id: str, query: str) -> str:
        """
        Expand query with context from conversation.

        Args:
            session_id: Session identifier
            query: Original search query

        Returns:
            Context-expanded query
        """
        context = self.get_context_for_search(session_id)
        if not context:
            return query

        topics = context.get("current_topics", [])
        expanded = self.topic_inference.expand_query_with_topics(query, topics)

        return expanded

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions from memory.

        Returns:
            Number of sessions removed
        """
        expired = [
            session_id
            for session_id, context in self._sessions.items()
            if context.is_expired()
        ]

        for session_id in expired:
            del self._sessions[session_id]
            if self.enable_mcp:
                self._delete_from_mcp(session_id)

        logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)

    def _filter_sensitive_info(self, text: str) -> str:
        """Filter out sensitive information from text."""
        filtered = text
        for pattern in self._privacy_patterns:
            filtered = pattern.sub("[FILTERED]", filtered)
        return filtered

    def _extract_all_entities(self, context: ConversationContext) -> List[str]:
        """Extract all entities from context."""
        entities = set()

        # From summaries
        for summary in context.summaries:
            entities.update(summary.key_entities)

        # From recent messages
        recent_text = " ".join(
            [m.content for m in context.get_recent_messages(limit=5)]
        )
        entities.update(self.summarizer._extract_entities(recent_text))

        return sorted(list(entities))

    def _store_to_mcp(self, session_id: str, context: ConversationContext) -> None:
        """Store context to MCP Memory Server."""
        try:
            # Try to import MCP memory tools
            try:
                # Check if MCP tools are available
                if hasattr(self, "_mcp_store"):
                    key = f"rag_conversation_{session_id}"
                    value = json.dumps(context.to_dict(), ensure_ascii=False)
                    self._mcp_store(key, value)
                    logger.debug(f"Stored session {session_id} to MCP")
            except ImportError:
                logger.debug("MCP memory tools not available, skipping storage")
        except Exception as e:
            logger.error(f"Failed to store to MCP: {e}")

    def _load_from_mcp(self, session_id: str) -> Optional[ConversationContext]:
        """Load context from MCP Memory Server."""
        try:
            if hasattr(self, "_mcp_retrieve"):
                key = f"rag_conversation_{session_id}"
                value = self._mcp_retrieve(key)
                if value:
                    data = json.loads(value)
                    return ConversationContext.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load from MCP: {e}")
        return None

    def _delete_from_mcp(self, session_id: str) -> None:
        """Delete context from MCP Memory Server."""
        try:
            if hasattr(self, "_mcp_delete"):
                key = f"rag_conversation_{session_id}"
                self._mcp_delete(key)
                logger.debug(f"Deleted session {session_id} from MCP")
        except Exception as e:
            logger.error(f"Failed to delete from MCP: {e}")


def create_memory_manager(
    llm_client: Optional[Any] = None,
    enable_mcp: bool = False,
) -> ConversationMemoryManager:
    """
    Factory function to create memory manager.

    Args:
        llm_client: Optional LLM client for summarization
        enable_mcp: Enable MCP Memory Server integration

    Returns:
        Configured ConversationMemoryManager
    """
    summarizer = ConversationSummarizer(llm_client=llm_client)
    topic_inference = TopicInference()

    return ConversationMemoryManager(
        summarizer=summarizer,
        topic_inference=topic_inference,
        enable_mcp=enable_mcp,
    )
