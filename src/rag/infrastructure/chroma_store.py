"""
ChromaDB Vector Store for Regulation RAG System.

Provides vector storage and hybrid search using ChromaDB.
Supports dense (embedding) and sparse (keyword) retrieval.
"""

import json
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


class ChromaVectorStore(IVectorStore):
    """
    ChromaDB-based vector store implementation.

    Supports:
    - Dense retrieval via embeddings
    - Metadata filtering
    - Persistence to disk
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "regulations",
        embedding_function=None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory for persistent storage.
            collection_name: Name of the ChromaDB collection.
            embedding_function: Optional custom embedding function.
                If None, will need to be set before adding documents.
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required. Install with: uv add chromadb")

        self.persist_directory = persist_directory
        self.collection_name = collection_name
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
                zip(ids, documents, metadatas, distances, strict=False)
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

        for doc_id, text, meta in zip(ids, docs, metas, strict=False):
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

        return count

    def _metadata_to_chunk(self, id_: str, document: str, metadata: dict) -> Chunk:
        """Convert stored metadata back to Chunk entity."""
        from ..domain.entities import ChunkLevel, Keyword, RegulationStatus

        # Parse parent_path from string
        parent_path_str = metadata.get("parent_path", "")
        parent_path = parent_path_str.split(" > ") if parent_path_str else []

        keywords = []
        raw_keywords = metadata.get("keywords")
        if raw_keywords:
            try:
                if isinstance(raw_keywords, str):
                    parsed = json.loads(raw_keywords)
                elif isinstance(raw_keywords, list):
                    parsed = raw_keywords
                else:
                    parsed = []
                for item in parsed:
                    if isinstance(item, dict) and "term" in item and "weight" in item:
                        keywords.append(Keyword.from_dict(item))
            except json.JSONDecodeError:
                pass

        return Chunk(
            id=id_,
            rule_code=metadata.get("rule_code", ""),
            level=ChunkLevel.from_string(metadata.get("level", "text")),
            title=metadata.get("title", ""),
            text=document,
            embedding_text=document,
            full_text="",  # Not stored
            parent_path=parent_path,
            token_count=metadata.get("token_count", 0),
            keywords=keywords,
            is_searchable=metadata.get("is_searchable", True),
            effective_date=metadata.get("effective_date") or None,
            status=RegulationStatus(metadata.get("status", "active")),
        )
