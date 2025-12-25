# Domain Layer
"""
Core domain entities and repository interfaces.
This layer has no external dependencies.
"""

from .entities import Chunk, ChunkLevel, Regulation, SearchResult
from .value_objects import Query, SearchFilter, SyncResult
from .repositories import IVectorStore, IDocumentLoader, ILLMClient

__all__ = [
    "Chunk",
    "ChunkLevel",
    "Regulation",
    "SearchResult",
    "Query",
    "SearchFilter",
    "SyncResult",
    "IVectorStore",
    "IDocumentLoader",
    "ILLMClient",
]
