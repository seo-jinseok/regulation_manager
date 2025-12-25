"""
Domain Entities for Regulation RAG System.

Entities are the core business objects that encapsulate the most general
and high-level rules. They are the least likely to change when something
external changes.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ChunkLevel(Enum):
    """Hierarchical level of a regulation chunk."""

    CHAPTER = "chapter"
    SECTION = "section"
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    ITEM = "item"
    SUBITEM = "subitem"
    ADDENDUM = "addendum"
    ADDENDUM_ITEM = "addendum_item"
    PREAMBLE = "preamble"
    TEXT = "text"

    @classmethod
    def from_string(cls, value: str) -> "ChunkLevel":
        """Convert string to ChunkLevel, defaulting to TEXT."""
        try:
            return cls(value)
        except ValueError:
            return cls.TEXT


class RegulationStatus(Enum):
    """Status of a regulation."""

    ACTIVE = "active"
    ABOLISHED = "abolished"


@dataclass
class Keyword:
    """A keyword with its weight for sparse retrieval."""

    term: str
    weight: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Keyword":
        """Create Keyword from dict."""
        return cls(term=data["term"], weight=data["weight"])


@dataclass
class Chunk:
    """
    A searchable chunk of regulation content.

    This is the primary unit for vector search and retrieval.
    Each chunk corresponds to a node in the regulation JSON.
    """

    id: str
    rule_code: str
    level: ChunkLevel
    title: str
    text: str
    embedding_text: str
    full_text: str
    parent_path: List[str]
    token_count: int
    keywords: List[Keyword]
    is_searchable: bool
    effective_date: Optional[str] = None
    status: RegulationStatus = RegulationStatus.ACTIVE

    @classmethod
    def from_json_node(cls, node: Dict[str, Any], rule_code: str) -> "Chunk":
        """
        Create a Chunk from a JSON node.

        Args:
            node: A node dict from the regulation JSON.
            rule_code: The rule code of the parent regulation.

        Returns:
            A Chunk instance.
        """
        keywords = [
            Keyword.from_dict(kw) for kw in node.get("keywords", [])
        ]

        # Determine status from title
        title = node.get("title", "")
        status = (
            RegulationStatus.ABOLISHED
            if "폐지" in title or "【폐지】" in title
            else RegulationStatus.ACTIVE
        )

        return cls(
            id=node.get("id", ""),
            rule_code=rule_code,
            level=ChunkLevel.from_string(node.get("chunk_level", "text")),
            title=title,
            text=node.get("text", ""),
            embedding_text=node.get("embedding_text", node.get("text", "")),
            full_text=node.get("full_text", ""),
            parent_path=node.get("parent_path", []),
            token_count=node.get("token_count", 0),
            keywords=keywords,
            is_searchable=node.get("is_searchable", True),
            effective_date=node.get("effective_date"),
            status=status,
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to metadata dict for vector store."""
        payload = {
            "id": self.id,
            "rule_code": self.rule_code,
            "level": self.level.value,
            "title": self.title,
            "parent_path": " > ".join(self.parent_path),
            "token_count": self.token_count,
            "is_searchable": self.is_searchable,
            "effective_date": self.effective_date or "",
            "status": self.status.value,
        }
        if self.keywords:
            payload["keywords"] = json.dumps(
                [{"term": k.term, "weight": k.weight} for k in self.keywords],
                ensure_ascii=False,
            )
        return payload


@dataclass
class Regulation:
    """
    A regulation document containing multiple chunks.

    Represents a single regulation (e.g., "교원인사규정") with its metadata.
    """

    rule_code: str
    title: str
    status: RegulationStatus
    chunks: List[Chunk] = field(default_factory=list)
    content_hash: str = ""  # For incremental sync

    @classmethod
    def from_json_doc(cls, doc: Dict[str, Any]) -> "Regulation":
        """Create a Regulation from a JSON document."""
        rule_code = doc.get("metadata", {}).get("rule_code", "")
        title = doc.get("title", "")
        status = (
            RegulationStatus.ABOLISHED
            if doc.get("status") == "abolished"
            else RegulationStatus.ACTIVE
        )

        return cls(
            rule_code=rule_code,
            title=title,
            status=status,
            chunks=[],
            content_hash="",
        )


@dataclass
class SearchResult:
    """
    A search result containing a chunk and its relevance score.

    Returned by the vector store search operations.
    """

    chunk: Chunk
    score: float
    rank: int = 0

    def __lt__(self, other: "SearchResult") -> bool:
        """Enable sorting by score (descending)."""
        return self.score > other.score


@dataclass
class Answer:
    """
    An answer generated by the LLM based on search results.
    """

    text: str
    sources: List[SearchResult]
    confidence: float = 0.0
