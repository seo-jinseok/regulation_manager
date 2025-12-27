"""
RAG-specific exceptions.

Re-exports from the main exceptions module for convenience,
plus any RAG-specific additions.
"""

from ..exceptions import (
    RAGError,
    VectorStoreError,
    SyncError,
    SearchError,
    LLMError,
    ConfigurationError,
    MissingAPIKeyError,
)

__all__ = [
    "RAGError",
    "VectorStoreError",
    "SyncError",
    "SearchError",
    "LLMError",
    "ConfigurationError",
    "MissingAPIKeyError",
    "RerankerError",
    "HybridSearchError",
]


class RerankerError(RAGError):
    """Error occurred during reranking."""
    
    def __init__(self, message: str, model: str = "unknown"):
        self.model = model
        super().__init__(f"[Reranker:{model}] {message}")


class HybridSearchError(RAGError):
    """Error occurred during hybrid search."""
    
    pass


class DocumentNotFoundError(RAGError):
    """Requested document/chunk was not found."""
    
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        super().__init__(f"Document not found: {doc_id}")
