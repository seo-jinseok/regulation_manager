"""
Disambiguation Dialog domain entities.

Implements user feedback loop for ambiguous queries.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DialogStatus(Enum):
    """Status of disambiguation dialog."""

    PENDING = "pending"  # Waiting for user response
    RESOLVED = "resolved"  # User provided clarification
    CANCELLED = "cancelled"  # User cancelled or timeout


@dataclass
class DisambiguationOption:
    """
    A single option for disambiguation.

    Represents one possible interpretation of an ambiguous query.
    """

    option_id: str
    label: str  # Display label (e.g., "졸업 요건")
    description: str  # Brief description
    keywords: List[str]  # Search keywords for this option
    confidence: float = 0.0  # Confidence score (0.0 ~ 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "option_id": self.option_id,
            "label": self.label,
            "description": self.description,
            "keywords": self.keywords,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisambiguationOption":
        """Create from dictionary."""
        return cls(
            option_id=data["option_id"],
            label=data["label"],
            description=data["description"],
            keywords=data.get("keywords", []),
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class DisambiguationDialog:
    """
    A disambiguation dialog for ambiguous queries.

    When a query is ambiguous (e.g., "졸업" - graduation requirements vs application vs deferral),
    this dialog manages the user feedback loop to clarify intent.

    Lifecycle:
    1. Created when ambiguous query detected
    2. Present options to user
    3. User selects option or cancels
    4. Resolved with selected keywords
    """

    dialog_id: str
    query: str
    options: List[DisambiguationOption]
    status: DialogStatus = DialogStatus.PENDING
    selected_option_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_pending(self) -> bool:
        """Check if dialog is waiting for user response."""
        return self.status == DialogStatus.PENDING

    @property
    def is_resolved(self) -> bool:
        """Check if dialog has been resolved."""
        return self.status == DialogStatus.RESOLVED

    @property
    def is_cancelled(self) -> bool:
        """Check if dialog was cancelled."""
        return self.status == DialogStatus.CANCELLED

    @property
    def resolved_keywords(self) -> Optional[List[str]]:
        """Get keywords from resolved option, or None if not resolved."""
        if not self.is_resolved or not self.selected_option_id:
            return None

        for option in self.options:
            if option.option_id == self.selected_option_id:
                return option.keywords
        return None

    def select_option(self, option_id: str) -> bool:
        """
        Select a disambiguation option.

        Args:
            option_id: ID of the selected option.

        Returns:
            True if option was found and selected, False otherwise.
        """
        for option in self.options:
            if option.option_id == option_id:
                self.selected_option_id = option_id
                self.status = DialogStatus.RESOLVED
                self.resolved_at = time.time()
                return True
        return False

    def cancel(self) -> None:
        """Cancel the dialog."""
        self.status = DialogStatus.CANCELLED
        self.resolved_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dialog_id": self.dialog_id,
            "query": self.query,
            "options": [opt.to_dict() for opt in self.options],
            "status": self.status.value,
            "selected_option_id": self.selected_option_id,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisambiguationDialog":
        """Create from dictionary."""
        options = [
            DisambiguationOption.from_dict(opt) for opt in data.get("options", [])
        ]

        return cls(
            dialog_id=data["dialog_id"],
            query=data["query"],
            options=options,
            status=DialogStatus(data.get("status", DialogStatus.PENDING.value)),
            selected_option_id=data.get("selected_option_id"),
            created_at=data.get("created_at", time.time()),
            resolved_at=data.get("resolved_at"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def create(
        cls,
        query: str,
        options: List[DisambiguationOption],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DisambiguationDialog":
        """Create a new disambiguation dialog."""
        return cls(
            dialog_id=str(uuid.uuid4()),
            query=query,
            options=options,
            metadata=metadata or {},
        )

    def get_prompt_message(self) -> str:
        """
        Generate user-facing prompt message.

        Returns:
            A message asking the user to clarify their intent.
        """
        options_text = "\n".join(
            [
                f"{i + 1}. {opt.label}: {opt.description}"
                for i, opt in enumerate(self.options)
            ]
        )
        return f"""'{self.query}'에 대한 정확한 정보를 제공하기 위해 선택해주세요:

{options_text}

번호를 선택해주세요 (1-{len(self.options)}):"""
