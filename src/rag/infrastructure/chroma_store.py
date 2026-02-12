"""
ChromaDB Vector Store for Regulation RAG System.

Provides vector storage and hybrid search using ChromaDB.
Supports dense (embedding) and sparse (keyword) retrieval.
"""

import logging
import os
from typing import List, Optional, Set

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from ..domain.entities import Chunk, SearchResult
from ..domain.repositories import IVectorStore
from ..domain.value_objects import Query, SearchFilter

logger = logging.getLogger(__name__)


class ChromaVectorStore(IVectorStore):
    """
    ChromaDB-based vector store implementation.

    Supports:
    - Dense retrieval via embeddings
    - Metadata filtering
    - Persistence to disk

    Automatically uses ko-sbert-sts embedding model for Korean semantic search.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "regulations",
        embedding_function=None,
        auto_create_embedding: bool = True,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory for persistent storage.
            collection_name: Name of the ChromaDB collection.
            embedding_function: Optional custom embedding function.
                If None and auto_create_embedding is True, will use
                ko-sbert-sts model for Korean semantic search.
            auto_create_embedding: If True and no embedding_function provided,
                automatically creates sentence-transformers embedding function.
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required. Install with: uv add chromadb")

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._embedding_function = None  # Initialize with default

        # Handle embedding function
        if embedding_function is None and auto_create_embedding:
            try:
                from .embedding_function import get_default_embedding_function

                self._embedding_function = get_default_embedding_function()
                logger.info("Using default embedding function (ko-sbert-sts)")
            except Exception as e:
                logger.warning(f"Failed to create default embedding function: {e}")
                self._embedding_function = None
        else:
            self._embedding_function = embedding_function

        # Create directory if needed
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        # Log collection status
        logger.info(f"ChromaDB collection '{collection_name}' initialized")

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Add chunks to the vector store.

        Args:
            chunks: List of Chunk entities to add.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        # Check if embedding function is available
        if not hasattr(self, "_embedding_function") or self._embedding_function is None:
            logger.error(
                "Cannot add chunks: no embedding function available. "
                "Provide an embedding_function during initialization."
            )
            return 0

        # Deduplicate by ID (keep first occurrence)
        seen_ids = set()
        unique_chunks = []
        for c in chunks:
            if c.id not in seen_ids:
                seen_ids.add(c.id)
                unique_chunks.append(c)

        if not unique_chunks:
            return 0

        # ChromaDB has max batch size of ~5000
        BATCH_SIZE = 5000
        total_added = 0

        for i in range(0, len(unique_chunks), BATCH_SIZE):
            batch = unique_chunks[i : i + BATCH_SIZE]
            ids = [c.id for c in batch]
            documents = [c.embedding_text for c in batch]
            metadatas = [c.to_metadata() for c in batch]

            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            total_added += len(batch)

        logger.info(f"Added {total_added} chunks to ChromaDB collection")
        return total_added

    def delete_by_rule_codes(self, rule_codes: List[str]) -> int:
        """
        Delete all chunks belonging to the given rule codes.

        Args:
            rule_codes: List of rule codes to delete.

        Returns:
            Number of chunks deleted.
        """
        if not rule_codes:
            return 0

        # Get IDs of chunks with matching rule_codes
        results = self._collection.get(
            where={"rule_code": {"$in": rule_codes}},
        )

        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info(
                f"Deleted {len(ids_to_delete)} chunks for rule codes: {rule_codes}"
            )

        return len(ids_to_delete)

    def search(
        self,
        query: Query,
        filter: Optional[SearchFilter] = None,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Perform search using ChromaDB.

        Args:
            query: The search query.
            filter: Optional metadata filters.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult sorted by relevance.
        """
        # Check if embedding function is available
        if not hasattr(self, "_embedding_function") or self._embedding_function is None:
            logger.error(
                "Cannot search: no embedding function available. "
                "Provide an embedding_function during initialization."
            )
            return []

        where = self._build_where(query, filter)
        query_text = query.text
        if not isinstance(query_text, str):
            if isinstance(query_text, (list, tuple)):
                query_text = " ".join(str(part) for part in query_text)
            elif query_text is None:
                query_text = ""
            else:
                query_text = str(query_text)

        # Ensure query_text is a non-empty string
        query_text = query_text.strip()
        if not query_text:
            return []  # Return empty results for empty queries

        # Query ChromaDB
        results = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to SearchResult
        search_results: List[SearchResult] = []

        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, (id_, doc, meta, dist) in enumerate(
                zip(ids, documents, metadatas, distances, strict=True)
            ):
                # Convert distance to similarity score (cosine)
                # ChromaDB returns distance; clamp to [0, 1]
                score = self._distance_to_score(dist)

                chunk = self._metadata_to_chunk(id_, doc, meta)
                result = SearchResult(chunk=chunk, score=score, rank=i + 1)
                search_results.append(result)

        return search_results

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        """Convert distance to similarity score with clamping."""
        if distance is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - distance))

    @staticmethod
    def _build_where(query: Query, filter: Optional[SearchFilter]) -> Optional[dict]:
        where = None
        if filter:
            where_clauses = filter.to_metadata_filter()

            if not query.include_abolished and "status" not in where_clauses:
                where_clauses["status"] = "active"

            if where_clauses:
                if len(where_clauses) == 1:
                    where = where_clauses
                else:
                    where = {"$and": [{k: v} for k, v in where_clauses.items()]}
        elif not query.include_abolished:
            where = {"status": "active"}
        return where

    def get_all_rule_codes(self) -> Set[str]:
        """
        Get all unique rule codes in the store.

        Returns:
            Set of rule codes.
        """
        results = self._collection.get(include=["metadatas"])
        rule_codes = set()

        for meta in results.get("metadatas", []):
            if meta and meta.get("rule_code"):
                rule_codes.add(meta["rule_code"])

        return rule_codes

    def count(self) -> int:
        """
        Get total number of chunks in the store.

        Returns:
            Total chunk count.
        """
        return self._collection.count()

    def get_all_documents(self) -> list:
        """
        Get all documents for BM25 index building.

        Returns:
            List of (doc_id, text, metadata) tuples.
        """
        results = self._collection.get(include=["documents", "metadatas"])

        documents = []
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        for doc_id, text, meta in zip(ids, docs, metas, strict=True):
            if text:  # Only include non-empty documents
                documents.append((doc_id, text, meta or {}))

        return documents

    def clear_all(self) -> int:
        """
        Delete all chunks from the store.

        Returns:
            Number of chunks deleted.
        """
        count = self.count()

        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_function,
        )

        logger.info(f"Cleared {count} chunks from ChromaDB collection")
        return count

    def _metadata_to_chunk(self, id_: str, document: str, metadata: dict) -> Chunk:
        """Convert stored metadata back to Chunk entity.

        Delegates to Chunk.from_metadata() to avoid code duplication.
        """
        return Chunk.from_metadata(id_, document, metadata)

    def close(self):
        """
        Release ChromaDB client resources.

        Call this method when done using the vector store to properly
        release memory and file handles. This is especially important
        in test environments where many instances may be created.
        """
        try:
            if hasattr(self, "_client") and self._client is not None:
                # Reset collection reference
                self._collection = None
                # ChromaDB PersistentClient doesn't have explicit close in all versions
                # but we can release references
                self._client = None
        except Exception:
            # Silently fail during cleanup to avoid errors during shutdown
            pass

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False
