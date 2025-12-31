"""
JSON Document Loader for Regulation RAG System.

Loads and parses regulation JSON files, converting them to domain entities.
Supports incremental sync by computing content hashes.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..domain.entities import ChapterInfo, Chunk, RegulationOverview, RegulationStatus
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
        self, nodes: List[Dict[str, Any]], rule_code: str
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
            if doc.get("doc_type") != "regulation":
                continue
            rule_code = self._extract_rule_code(doc)
            if rule_code:
                titles[rule_code] = doc.get("title", "")

        return titles

    def get_all_regulations(self, json_path: str) -> List[Tuple[str, str]]:
        """
        Get all regulation metadata (rule_code, title).
        Handles cases where rule codes might be duplicated.

        Args:
            json_path: Path to the regulation JSON file.

        Returns:
            List of (rule_code, title) tuples.
        """
        data = self._load_json(json_path)
        regulations: List[Tuple[str, str]] = []

        for doc in data.get("docs", []):
            if doc.get("doc_type") != "regulation":
                continue
            rule_code = self._extract_rule_code(doc)
            title = doc.get("title", "")
            if rule_code and title:
                regulations.append((rule_code, title))
        
        return regulations

    def get_regulation_doc(
        self, json_path: str, identifier: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a regulation document by rule_code or title.

        Args:
            json_path: Path to the regulation JSON file.
            identifier: rule_code or title.

        Returns:
            Regulation document dict or None.
        """
        data = self._load_json(json_path)
        target = self._normalize_title(identifier)

        for doc in data.get("docs", []):
            if doc.get("doc_type") != "regulation":
                continue

            rule_code = self._extract_rule_code(doc)
            title = doc.get("title", "")
            if rule_code and rule_code == identifier:
                return doc
            if target and self._normalize_title(title) == target:
                return doc

        return None

    @staticmethod
    def _normalize_title(title: str) -> str:
        return "".join(str(title).split())

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
        doc = self.get_regulation_doc(json_path, identifier)
        if not doc:
            return None

        rule_code = self._extract_rule_code(doc)
        title = doc.get("title", "")
        status = (
            RegulationStatus.ABOLISHED
            if doc.get("status") == "abolished" or "폐지" in title
            else RegulationStatus.ACTIVE
        )

        content = doc.get("content", [])
        addenda = doc.get("addenda", [])

        # Extract chapters and count articles
        chapters = self._extract_chapters(content)
        article_count = self._count_articles(content)
        has_addenda = len(addenda) > 0

        return RegulationOverview(
            rule_code=rule_code,
            title=title,
            status=status,
            article_count=article_count,
            chapters=chapters,
            has_addenda=has_addenda,
        )

    def _extract_chapters(self, nodes: List[Dict[str, Any]]) -> List[ChapterInfo]:
        """Extract chapter information from content nodes."""
        chapters: List[ChapterInfo] = []

        for node in nodes:
            node_type = node.get("type", "")
            if node_type == "chapter":
                display_no = node.get("display_no", "")
                title = node.get("title", "")

                # Get article range from children
                children = node.get("children", [])
                article_range = self._get_article_range(children)

                chapters.append(
                    ChapterInfo(
                        display_no=display_no,
                        title=title,
                        article_range=article_range,
                    )
                )

        return chapters

    def _get_article_range(self, nodes: List[Dict[str, Any]]) -> str:
        """Get article range string (e.g., '제1조~제5조') from nodes."""
        articles: List[str] = []

        def collect_articles(children: List[Dict[str, Any]]) -> None:
            for node in children:
                if node.get("type") == "article":
                    display_no = node.get("display_no", "")
                    if display_no:
                        articles.append(display_no)
                # Check children recursively (for nested structures)
                child_nodes = node.get("children", [])
                if child_nodes:
                    collect_articles(child_nodes)

        collect_articles(nodes)

        if not articles:
            return ""
        if len(articles) == 1:
            return articles[0]
        return f"{articles[0]}~{articles[-1]}"

    def _count_articles(self, nodes: List[Dict[str, Any]]) -> int:
        """Count total number of articles in content."""
        count = 0

        def count_recursive(children: List[Dict[str, Any]]) -> None:
            nonlocal count
            for node in children:
                if node.get("type") == "article":
                    count += 1
                child_nodes = node.get("children", [])
                if child_nodes:
                    count_recursive(child_nodes)

        count_recursive(nodes)
        return count
