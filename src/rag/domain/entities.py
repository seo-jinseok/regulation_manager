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
    doc_type: str = "regulation"  # Document type for filtering (regulation, note, etc.)
    effective_date: Optional[str] = None
    status: RegulationStatus = RegulationStatus.ACTIVE
    article_number: Optional[str] = None  # Enhanced citation support (Component 3)

    @classmethod
    def from_json_node(cls, node: Dict[str, Any], rule_code: str, doc_type: str = "regulation") -> "Chunk":
        """
        Create a Chunk from a JSON node.

        Args:
            node: A node dict from the regulation JSON.
            rule_code: The rule code of the parent regulation.
            doc_type: The document type (regulation, note, etc.).

        Returns:
            A Chunk instance.
        """
        keywords = [Keyword.from_dict(kw) for kw in node.get("keywords", [])]

        # Determine status from title
        title = node.get("title", "")
        status = (
            RegulationStatus.ABOLISHED
            if "폐지" in title or "【폐지】" in title
            else RegulationStatus.ACTIVE
        )

        # Extract article number for enhanced citation support (Component 3)
        article_number = None
        if title:
            try:
                from .citation.article_number_extractor import ArticleNumberExtractor

                extractor = ArticleNumberExtractor()
                result = extractor.extract(title)
                if result:
                    article_number = result.to_citation_format()
            except ImportError:
                # Gracefully handle if citation module not available
                pass

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
            doc_type=doc_type,
            effective_date=node.get("effective_date"),
            status=status,
            article_number=article_number,
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
            "doc_type": self.doc_type,
            "effective_date": self.effective_date or "",
            "status": self.status.value,
            "article_number": self.article_number
            or "",  # Enhanced citation (Component 3)
        }
        if self.keywords:
            payload["keywords"] = json.dumps(
                [{"term": k.term, "weight": k.weight} for k in self.keywords],
                ensure_ascii=False,
            )
        return payload

    @classmethod
    def from_metadata(cls, doc_id: str, text: str, metadata: Dict[str, Any]) -> "Chunk":
        """
        Reconstruct a Chunk from stored metadata.

        Used when rebuilding Chunk from BM25 search results that don't have
        the full Chunk object available.

        Args:
            doc_id: The chunk ID.
            text: The chunk text content.
            metadata: Metadata dict from vector store.

        Returns:
            A Chunk instance.
        """
        # Parse parent_path from string
        parent_path_str = metadata.get("parent_path", "")
        parent_path = parent_path_str.split(" > ") if parent_path_str else []

        # Parse keywords if present
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

        return cls(
            id=doc_id,
            rule_code=metadata.get("rule_code", ""),
            level=ChunkLevel.from_string(metadata.get("level", "text")),
            title=metadata.get("title", ""),
            text=text,
            embedding_text=text,
            full_text="",
            parent_path=parent_path,
            token_count=metadata.get("token_count", 0),
            keywords=keywords,
            is_searchable=metadata.get("is_searchable", True),
            doc_type=metadata.get("doc_type", "regulation"),
            effective_date=metadata.get("effective_date") or None,
            status=RegulationStatus(metadata.get("status", "active")),
            article_number=metadata.get("article_number")
            or None,  # Enhanced citation (Component 3)
        )


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


@dataclass
class ChapterInfo:
    """Information about a chapter (장) in a regulation."""

    display_no: str  # e.g., "제1장"
    title: str  # e.g., "총칙"
    article_range: str  # e.g., "제1조~제5조"


@dataclass
class RegulationOverview:
    """
    Overview information for a regulation.

    Used when displaying regulation summary instead of search results.
    """

    rule_code: str
    title: str
    status: RegulationStatus
    article_count: int
    chapters: List[ChapterInfo] = field(default_factory=list)
    has_addenda: bool = False


@dataclass
class RerankingMetrics:
    """
    Metrics tracking reranker performance and usage patterns (Cycle 3).

    Provides insights into:
    - How often reranker is applied vs skipped
    - Performance impact of reranking
    - Query type distribution for reranking decisions
    """

    total_queries: int = 0
    reranker_applied: int = 0
    reranker_skipped: int = 0

    # Query type breakdown for skips
    article_reference_skips: int = 0
    regulation_name_skips: int = 0
    short_simple_skips: int = 0
    no_intent_skips: int = 0

    # Query type breakdown for applies
    natural_question_applies: int = 0
    intent_applies: int = 0
    complex_applies: int = 0

    # Performance tracking
    total_reranker_time_ms: float = 0.0
    total_skip_saved_time_ms: float = 0.0  # Estimated time saved by skipping

    def record_skip(
        self,
        query_type: Optional[str] = None,
        reason: str = "unknown",
    ) -> None:
        """
        Record a reranker skip event.

        Args:
            query_type: The type of query that was skipped.
            reason: The reason for skipping (article_reference, regulation_name, etc.)
        """
        self.reranker_skipped += 1
        if reason == "article_reference":
            self.article_reference_skips += 1
        elif reason == "regulation_name":
            self.regulation_name_skips += 1
        elif reason == "short_simple":
            self.short_simple_skips += 1
        elif reason == "no_intent":
            self.no_intent_skips += 1

    def record_apply(
        self,
        query_type: Optional[str] = None,
        reranker_time_ms: float = 0.0,
    ) -> None:
        """
        Record a reranker apply event.

        Args:
            query_type: The type of query that was reranked.
            reranker_time_ms: Time taken for reranking in milliseconds.
        """
        self.reranker_applied += 1
        self.total_reranker_time_ms += reranker_time_ms

        if query_type == "NATURAL_QUESTION":
            self.natural_question_applies += 1
        elif query_type == "INTENT":
            self.intent_applies += 1
        elif query_type == "complex":
            self.complex_applies += 1

    def record_query(self) -> None:
        """Record a total query event."""
        self.total_queries += 1

    @property
    def skip_rate(self) -> float:
        """Calculate the rate of reranker skips (0.0 to 1.0)."""
        if self.total_queries == 0:
            return 0.0
        return self.reranker_skipped / self.total_queries

    @property
    def apply_rate(self) -> float:
        """Calculate the rate of reranker applies (0.0 to 1.0)."""
        if self.total_queries == 0:
            return 0.0
        return self.reranker_applied / self.total_queries

    @property
    def avg_reranker_time_ms(self) -> float:
        """Calculate average reranker time in milliseconds."""
        if self.reranker_applied == 0:
            return 0.0
        return self.total_reranker_time_ms / self.reranker_applied

    @property
    def estimated_time_saved_ms(self) -> float:
        """
        Estimate total time saved by skipping reranker.

        Assumes each skip would have taken avg_reranker_time_ms if not skipped.
        """
        if self.reranker_skipped == 0:
            return 0.0
        avg_time = self.avg_reranker_time_ms
        if avg_time == 0:
            return 0.0
        return self.reranker_skipped * avg_time

    def get_summary(self) -> str:
        """
        Generate a human-readable summary of reranking metrics.

        Returns:
            Formatted string with key metrics.
        """
        lines = [
            "📊 Reranking Metrics Summary:",
            f"   Total queries: {self.total_queries}",
            f"   Reranker applied: {self.reranker_applied} ({self.apply_rate:.1%})",
            f"   Reranker skipped: {self.reranker_skipped} ({self.skip_rate:.1%})",
            "",
            "🔍 Skip Reasons:",
            f"   Article reference: {self.article_reference_skips}",
            f"   Regulation name: {self.regulation_name_skips}",
            f"   Short simple: {self.short_simple_skips}",
            f"   No intent: {self.no_intent_skips}",
            "",
            "✅ Apply Types:",
            f"   Natural questions: {self.natural_question_applies}",
            f"   Intent queries: {self.intent_applies}",
            f"   Complex queries: {self.complex_applies}",
            "",
            "⏱️  Performance:",
            f"   Avg reranker time: {self.avg_reranker_time_ms:.2f}ms",
            f"   Est. time saved: {self.estimated_time_saved_ms:.2f}ms",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary for JSON serialization.

        Returns:
            Dictionary representation of metrics.
        """
        return {
            "total_queries": self.total_queries,
            "reranker_applied": self.reranker_applied,
            "reranker_skipped": self.reranker_skipped,
            "skip_rate": self.skip_rate,
            "apply_rate": self.apply_rate,
            "article_reference_skips": self.article_reference_skips,
            "regulation_name_skips": self.regulation_name_skips,
            "short_simple_skips": self.short_simple_skips,
            "no_intent_skips": self.no_intent_skips,
            "natural_question_applies": self.natural_question_applies,
            "intent_applies": self.intent_applies,
            "complex_applies": self.complex_applies,
            "total_reranker_time_ms": self.total_reranker_time_ms,
            "avg_reranker_time_ms": self.avg_reranker_time_ms,
            "estimated_time_saved_ms": self.estimated_time_saved_ms,
        }
