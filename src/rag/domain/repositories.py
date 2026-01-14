"""
Repository Interfaces for Regulation RAG System.

These are abstract base classes defining the contracts that infrastructure
implementations must fulfill. The domain layer depends only on these
interfaces, not on concrete implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set

from .entities import Chunk, RegulationOverview, SearchResult
from .value_objects import Query, SearchFilter, SyncState


class IVectorStore(ABC):
    """
    Abstract interface for vector storage and search.

    Implementations may use ChromaDB, Qdrant, Pinecone, etc.
    """

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Add chunks to the vector store.

        Args:
            chunks: List of Chunk entities to add.

        Returns:
            Number of chunks successfully added.
        """
        pass

    @abstractmethod
    def delete_by_rule_codes(self, rule_codes: List[str]) -> int:
        """
        Delete all chunks belonging to the given rule codes.

        Args:
            rule_codes: List of rule codes to delete.

        Returns:
            Number of chunks deleted.
        """
        pass

    @abstractmethod
    def search(
        self,
        query: Query,
        filter: Optional[SearchFilter] = None,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Perform hybrid search (dense + sparse).

        Args:
            query: The search query.
            filter: Optional metadata filters.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult sorted by relevance.
        """
        pass

    @abstractmethod
    def get_all_rule_codes(self) -> Set[str]:
        """
        Get all unique rule codes in the store.

        Returns:
            Set of rule codes.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Get total number of chunks in the store.

        Returns:
            Total chunk count.
        """
        pass

    @abstractmethod
    def get_all_documents(self) -> list:
        """
        Get all documents for sparse index building.

        Returns:
            List of (doc_id, text, metadata) tuples.
        """
        pass

    @abstractmethod
    def clear_all(self) -> int:
        """
        Delete all chunks from the store.

        Returns:
            Number of chunks deleted.
        """
        pass


class IDocumentLoader(ABC):
    """
    Abstract interface for loading documents from JSON.

    Converts regulation JSON to domain entities.
    """

    @abstractmethod
    def load_all_chunks(self, json_path: str) -> List[Chunk]:
        """
        Load all searchable chunks from a JSON file.

        Args:
            json_path: Path to the regulation JSON file.

        Returns:
            List of Chunk entities.
        """
        pass

    @abstractmethod
    def load_chunks_by_rule_codes(
        self, json_path: str, rule_codes: Set[str]
    ) -> List[Chunk]:
        """
        Load chunks only for specific rule codes.

        Args:
            json_path: Path to the regulation JSON file.
            rule_codes: Set of rule codes to load.

        Returns:
            List of Chunk entities for the specified rules.
        """
        pass

    @abstractmethod
    def compute_state(self, json_path: str) -> SyncState:
        """
        Compute sync state (rule_code -> content_hash) for a JSON file.

        Used for incremental sync to detect changes.

        Args:
            json_path: Path to the regulation JSON file.

        Returns:
            SyncState with content hashes for each regulation.
        """
        pass

    @abstractmethod
    def get_regulation_titles(self, json_path: str) -> dict:
        """
        Get mapping of rule_code to regulation title.

        Args:
            json_path: Path to the regulation JSON file.

        Returns:
            Dict mapping rule_code to title.
        """
        pass

    @abstractmethod
    def get_all_regulations(self, json_path: str) -> List[tuple]:
        """
        Get all regulation metadata (rule_code, title).
        Handles cases where rule codes might be duplicated.

        Args:
            json_path: Path to the regulation JSON file.

        Returns:
            List of (rule_code, title) tuples.
        """
        pass

    @abstractmethod
    def get_regulation_doc(self, json_path: str, identifier: str) -> Optional[dict]:
        """
        Get a regulation document by rule_code or title.

        Args:
            json_path: Path to the regulation JSON file.
            identifier: rule_code or title.

        Returns:
            Regulation document dict or None.
        """
        pass

    @abstractmethod
    def get_regulation_overview(
        self, json_path: str, identifier: str
    ) -> Optional[RegulationOverview]:
        """
        Get regulation overview with table of contents.

        Args:
            json_path: Path to the regulation JSON file.
            identifier: rule_code or regulation title.

        Returns:
            RegulationOverview or None if not found.
        """
        pass


class ILLMClient(ABC):
    """
    Abstract interface for LLM interactions.

    Implementations may use OpenAI, Anthropic, local models, etc.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System instructions.
            user_message: User's question with context.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            Generated response text.
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        pass


class IReranker(ABC):
    """
    Abstract interface for reranking search results.

    Implementations may use BGE, Cohere, etc.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[tuple],
        top_k: int = 10,
    ) -> List[tuple]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: The search query.
            documents: List of (doc_id, content, metadata) tuples.
            top_k: Maximum number of results to return.

        Returns:
            List of (doc_id, content, score, metadata) tuples sorted by relevance.
        """
        pass


class IHybridSearcher(ABC):
    """
    Abstract interface for hybrid search (dense + sparse).

    Combines BM25 sparse search with dense vector search
    using Reciprocal Rank Fusion (RRF).
    """

    @abstractmethod
    def add_documents(self, documents: List[tuple]) -> None:
        """
        Add documents to the sparse index.

        Args:
            documents: List of (doc_id, text, metadata) tuples.
        """
        pass

    @abstractmethod
    def search_sparse(self, query: str, top_k: int = 10) -> List:
        """
        Perform sparse (BM25) search.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of ScoredDocument objects.
        """
        pass

    @abstractmethod
    def fuse_results(
        self,
        sparse_results: List,
        dense_results: List,
        top_k: int = 10,
        query_text: Optional[str] = None,
    ) -> List:
        """
        Fuse sparse and dense results using RRF.

        Args:
            sparse_results: Results from sparse search.
            dense_results: Results from dense search.
            top_k: Maximum number of results.
            query_text: Original query for context.

        Returns:
            Fused and sorted results.
        """
        pass

    @abstractmethod
    def set_llm_client(self, llm_client: "ILLMClient") -> None:
        """
        Set LLM client for query rewriting.

        Args:
            llm_client: LLM client implementation.
        """
        pass

    @abstractmethod
    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms.

        Args:
            query: Original query.

        Returns:
            Expanded query string.
        """
        pass
