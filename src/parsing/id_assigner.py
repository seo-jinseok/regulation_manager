"""
Stable ID generation for regulation nodes.

Generates deterministic, content-based UUIDs for nodes to ensure
consistent IDs across runs even when content is reprocessed.
"""

import hashlib
import re
import uuid
from typing import Any, Dict, List, Optional

# Namespace UUID for stable ID generation
STABLE_ID_NAMESPACE = uuid.UUID("f24a86f2-2c2d-4a08-9cc4-6b51b0b4043a")


class StableIdAssigner:
    """
    Assigns stable, deterministic IDs to regulation nodes.

    Uses content hashing and path-based disambiguation to generate
    consistent UUIDs that remain stable across processing runs.
    """

    def assign_ids(
        self, docs: List[Dict[str, Any]], source_file_name: Optional[str] = None
    ) -> None:
        """
        Assign stable IDs to all nodes in documents.

        Args:
            docs: List of document dictionaries.
            source_file_name: Optional source file name for ID generation.
        """
        for doc in docs:
            doc_key = self._stable_doc_key(doc, source_file_name=source_file_name)
            for section in ("content", "addenda"):
                nodes = doc.get(section) or []
                self._assign_ids_recursive(nodes, parent_path=section, doc_key=doc_key)

    def _stable_doc_key(
        self, doc: Dict[str, Any], source_file_name: Optional[str] = None
    ) -> str:
        """Generate a stable key for document identification."""
        parts = []
        if source_file_name:
            parts.append(f"file:{source_file_name}")

        doc_type = (doc.get("doc_type") or "").strip()
        if doc_type:
            parts.append(f"type:{doc_type}")

        metadata = doc.get("metadata") or {}
        rule_code = (metadata.get("rule_code") or "").strip()
        if rule_code:
            parts.append(f"rule:{self._normalize_token(rule_code)}")
            return "|".join(parts)

        title = (doc.get("title") or "").strip()
        part = (doc.get("part") or "").strip()
        if part:
            parts.append(f"part:{self._normalize_token(part)}")
        if title:
            parts.append(f"title:{self._normalize_token(title)}")

        return "|".join(parts) or "doc:unknown"

    def _assign_ids_recursive(
        self, nodes: List[Dict[str, Any]], parent_path: str, doc_key: str
    ) -> None:
        """Recursively assign IDs to nodes."""
        if not nodes:
            return

        base_keys: List[str] = []
        refined_keys: List[str] = []

        for node in nodes:
            base = self._stable_node_base_key(node)
            base_keys.append(base)

        # First pass: add content hash only when necessary
        base_counts: Dict[str, int] = {}
        for base in base_keys:
            base_counts[base] = base_counts.get(base, 0) + 1

        for node, base in zip(nodes, base_keys, strict=False):
            if base_counts.get(base, 0) <= 1:
                refined_keys.append(base)
                continue
            refined_keys.append(f"{base}|h:{self._stable_node_content_hash(node)}")

        # Second pass: disambiguate exact duplicates after hashing
        refined_counts: Dict[str, int] = {}
        for key in refined_keys:
            refined_counts[key] = refined_counts.get(key, 0) + 1

        refined_seen: Dict[str, int] = {}
        for node, key in zip(nodes, refined_keys, strict=False):
            if refined_counts.get(key, 0) > 1:
                refined_seen[key] = refined_seen.get(key, 0) + 1
                key = f"{key}|i:{refined_seen[key]}"

            path = f"{parent_path}/{key}" if parent_path else key
            node["id"] = str(uuid.uuid5(STABLE_ID_NAMESPACE, f"{doc_key}::{path}"))
            self._assign_ids_recursive(
                node.get("children") or [], parent_path=path, doc_key=doc_key
            )

    def _stable_node_base_key(self, node: Dict[str, Any]) -> str:
        """Generate base key for a node."""
        node_type = (node.get("type") or "").strip()
        display_no = self._normalize_token(node.get("display_no") or "")
        sort_no = node.get("sort_no") or {}
        main = sort_no.get("main", 0)
        sub = sort_no.get("sub", 0)
        return f"{node_type}|{display_no}|{main}|{sub}"

    def _stable_node_content_hash(self, node: Dict[str, Any]) -> str:
        """Generate content hash for a node."""
        title = self._normalize_ws(node.get("title") or "")
        text = self._normalize_ws(node.get("text") or "")
        digest = hashlib.sha1(f"{title}|{text}".encode("utf-8")).hexdigest()
        return digest[:12]

    def _normalize_ws(self, value: str) -> str:
        """Normalize whitespace in a string."""
        return re.sub(r"\s+", " ", str(value)).strip()

    def _normalize_token(self, value: str) -> str:
        """Normalize token by removing all whitespace."""
        return re.sub(r"\s+", "", str(value)).strip()
