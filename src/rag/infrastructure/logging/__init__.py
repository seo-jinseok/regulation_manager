"""
Logging infrastructure for RAG system.

Provides event emission and interaction logging for monitoring
the RAG pipeline execution.
"""

from .events import (
    EventType,
    EventEmitter,
    QueryReceivedEvent,
    QueryRewrittenEvent,
    SearchCompletedEvent,
    RerankingCompletedEvent,
    LLMGenerationStartedEvent,
    TokenGeneratedEvent,
    AnswerGeneratedEvent,
    RAGEvent,
    event_to_dict,
)
from .interaction_logger import (
    RAGInteractionLogger,
    LatencyTimer,
)

__all__ = [
    # Event types
    "EventType",
    # Event dataclasses
    "QueryReceivedEvent",
    "QueryRewrittenEvent",
    "SearchCompletedEvent",
    "RerankingCompletedEvent",
    "LLMGenerationStartedEvent",
    "TokenGeneratedEvent",
    "AnswerGeneratedEvent",
    "RAGEvent",
    # Event emitter
    "EventEmitter",
    "event_to_dict",
    # Logger
    "RAGInteractionLogger",
    "LatencyTimer",
]
