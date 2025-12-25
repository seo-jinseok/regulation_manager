"""
JSON Document Loader for Regulation RAG System.

Loads and parses regulation JSON files, converting them to domain entities.
Supports incremental sync by computing content hashes.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

from ..domain.entities import Chunk, ChunkLevel
from ..domain.repositories import IDocumentLoader
from ..domain.value_objects import SyncState


class JSONDocumentLoader(IDocumentLoader):
    """
    Loads regulation chunks from JSON files.
    
    Implements IDocumentLoader interface for the regulation JSON schema v2.0.
    """

    def load_all_chunks(self, json_path: str) -> List[Chunk]:
        """
        Load all searchable chunks from a JSON file.

        Args:
            json_path: Path to the regulation JSON file.

        Returns:
            List of Chunk entities.
        """
        data = self._load_json(json_path)
        chunks: List[Chunk] = []

        for doc in data.get("docs", []):
            # Skip index/toc documents
            if doc.get("is_index_duplicate", False):
                continue
            if doc.get("doc_type") in ("toc", "index_alpha", "index_dept", "index"):
                continue

            rule_code = self._extract_rule_code(doc)
            if not rule_code:
                continue

            # Extract chunks from content
            content = doc.get("content", [])
            chunks.extend(self._extract_chunks_recursive(content, rule_code))

            # Extract chunks from addenda
            addenda = doc.get("addenda", [])
            chunks.extend(self._extract_chunks_recursive(addenda, rule_code))

        return chunks

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
        data = self._load_json(json_path)
        chunks: List[Chunk] = []

        for doc in data.get("docs", []):
            rule_code = self._extract_rule_code(doc)
            if not rule_code or rule_code not in rule_codes:
                continue

            content = doc.get("content", [])
            chunks.extend(self._extract_chunks_recursive(content, rule_code))

            addenda = doc.get("addenda", [])
            chunks.extend(self._extract_chunks_recursive(addenda, rule_code))

        return chunks

    def compute_state(self, json_path: str) -> SyncState:
        """
        Compute sync state (rule_code -> content_hash) for a JSON file.

        Args:
            json_path: Path to the regulation JSON file.

        Returns:
            SyncState with content hashes for each regulation.
        """
        data = self._load_json(json_path)
        regulations: Dict[str, str] = {}

        for doc in data.get("docs", []):
            # Skip index documents
            if doc.get("is_index_duplicate", False):
                continue

            rule_code = self._extract_rule_code(doc)
            if not rule_code:
                continue

            # Compute hash of document content (exclude volatile metadata)
            normalized_doc = self._normalize_doc_for_hash(doc)
            content_str = json.dumps(normalized_doc, ensure_ascii=False, sort_keys=True)
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
            regulations[rule_code] = content_hash

        return SyncState(
            last_sync=datetime.now(timezone.utc).isoformat(),
            json_file=Path(json_path).name,
            regulations=regulations,
        )

    def _load_json(self, json_path: str) -> Dict[str, Any]:
        """Load JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_doc_for_hash(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Remove volatile fields that should not affect incremental sync."""
        normalized = json.loads(json.dumps(doc, ensure_ascii=False))
        metadata = normalized.get("metadata")
        if isinstance(metadata, dict):
            for key in ("scan_date", "file_name", "page_range"):
                metadata.pop(key, None)
        return normalized

    def _extract_rule_code(self, doc: Dict[str, Any]) -> str:
        """Extract rule_code from document."""
        # Try metadata first
        metadata = doc.get("metadata", {})
        if metadata.get("rule_code"):
            return metadata["rule_code"]
        
        # Try content nodes for rule_code
        for node in doc.get("content", []):
            node_meta = node.get("metadata", {})
            if node_meta.get("rule_code"):
                return node_meta["rule_code"]
        
        return ""

    def _extract_chunks_recursive(
        self, 
        nodes: List[Dict[str, Any]], 
        rule_code: str
    ) -> List[Chunk]:
        """
        Recursively extract Chunk entities from nested nodes.

        Only includes nodes with is_searchable=True and non-empty text.
        """
        chunks: List[Chunk] = []

        for node in nodes:
            # Check if searchable
            is_searchable = node.get("is_searchable", True)
            text = node.get("text", "")
            embedding_text = node.get("embedding_text", text)

            if is_searchable and embedding_text:
                chunk = Chunk.from_json_node(node, rule_code)
                chunks.append(chunk)

            # Recurse into children
            children = node.get("children", [])
            if children:
                chunks.extend(self._extract_chunks_recursive(children, rule_code))

        return chunks

    def get_regulation_titles(self, json_path: str) -> Dict[str, str]:
        """
        Get mapping of rule_code to regulation title.

        Useful for displaying search results with regulation names.

        Args:
            json_path: Path to the regulation JSON file.

        Returns:
            Dict mapping rule_code to title.
        """
        data = self._load_json(json_path)
        titles: Dict[str, str] = {}

        for doc in data.get("docs", []):
            rule_code = self._extract_rule_code(doc)
            if rule_code:
                titles[rule_code] = doc.get("title", "")

        return titles
