"""
Value Objects for Regulation RAG System.

Value Objects are immutable objects that represent a concept by its attributes.
Two value objects are equal if their attributes are equal.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .entities import RegulationStatus, ChunkLevel


@dataclass(frozen=True)
class Query:
    """
    A search query.

    Immutable value object representing a user's search intent.
    """

    text: str
    include_abolished: bool = False

    def __post_init__(self) -> None:
        """Validate query."""
        if not self.text or not self.text.strip():
            raise ValueError("Query text cannot be empty")


@dataclass(frozen=True)
class SearchFilter:
    """
    Filters to apply during search.

    All filters are optional and can be combined.
    """

    status: Optional[RegulationStatus] = None
    levels: Optional[List[ChunkLevel]] = None
    rule_codes: Optional[List[str]] = None
    effective_date_from: Optional[str] = None
    effective_date_to: Optional[str] = None

    def to_metadata_filter(self) -> Dict[str, Any]:
        """
        Convert to metadata filter dict for vector store.

        Returns:
            Dict with filter conditions for the vector store.
        """
        filters = {}

        if self.status:
            filters["status"] = self.status.value

        if self.levels:
            filters["level"] = {"$in": [level.value for level in self.levels]}

        if self.rule_codes:
            filters["rule_code"] = {"$in": self.rule_codes}

        return filters


@dataclass(frozen=True)
class SyncResult:
    """
    Result of a synchronization operation.

    Tracks what changed during sync for reporting and auditing.
    """

    added: int = 0
    modified: int = 0
    removed: int = 0
    unchanged: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        """Total number of regulations processed."""
        return self.added + self.modified + self.removed + self.unchanged

    @property
    def has_changes(self) -> bool:
        """Whether any changes were made."""
        return self.added > 0 or self.modified > 0 or self.removed > 0

    @property
    def has_errors(self) -> bool:
        """Whether any errors occurred."""
        return len(self.errors) > 0

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"SyncResult: {self.added} added, {self.modified} modified, "
            f"{self.removed} removed, {self.unchanged} unchanged"
        )


@dataclass(frozen=True)
class SyncState:
    """
    Persistent state for incremental synchronization.

    Tracks the content hash of each regulation for change detection.
    """

    last_sync: str  # ISO format datetime
    json_file: str
    regulations: Dict[str, str]  # rule_code -> content_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "last_sync": self.last_sync,
            "json_file": self.json_file,
            "regulations": self.regulations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncState":
        """Create from dict."""
        return cls(
            last_sync=data.get("last_sync", ""),
            json_file=data.get("json_file", ""),
            regulations=data.get("regulations", {}),
        )

    @classmethod
    def empty(cls) -> "SyncState":
        """Create an empty state."""
        return cls(last_sync="", json_file="", regulations={})
