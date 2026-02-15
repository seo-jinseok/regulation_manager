"""
Checkpoint Manager for RAG Evaluation Sessions.

Persists evaluation progress to enable resume from interruptions.
Part of SPEC-RAG-EVAL-001 Milestone 3: Automation Pipeline.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PersonaProgress:
    """Progress tracking for a single persona."""

    persona_name: str
    total_queries: int
    completed_queries: int
    failed_queries: int
    completed_query_ids: List[str] = field(default_factory=list)
    failed_query_ids: List[str] = field(default_factory=list)
    last_query: Optional[str] = None
    last_timestamp: Optional[str] = None

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate as percentage."""
        if self.total_queries == 0:
            return 0.0
        return (self.completed_queries / self.total_queries) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonaProgress":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EvaluationProgress:
    """Complete evaluation session progress."""

    session_id: str
    started_at: str
    updated_at: str
    status: str  # "running", "paused", "completed", "failed"
    total_queries: int
    completed_queries: int
    failed_queries: int
    personas: Dict[str, PersonaProgress] = field(default_factory=dict)
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def completion_rate(self) -> float:
        """Calculate overall completion rate."""
        if self.total_queries == 0:
            return 0.0
        return (self.completed_queries / self.total_queries) * 100

    @property
    def is_complete(self) -> bool:
        """Check if evaluation is complete."""
        return self.status == "completed"

    @property
    def can_resume(self) -> bool:
        """Check if evaluation can be resumed."""
        return self.status in ("paused", "failed") and self.completed_queries < self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "total_queries": self.total_queries,
            "completed_queries": self.completed_queries,
            "failed_queries": self.failed_queries,
            "personas": {k: v.to_dict() for k, v in self.personas.items()},
            "results": self.results,
            "errors": self.errors,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationProgress":
        """Create from dictionary."""
        personas_data = data.pop("personas", {})
        personas = {
            k: PersonaProgress.from_dict(v) for k, v in personas_data.items()
        }
        return cls(personas=personas, **data)


class CheckpointManager:
    """
    Manages evaluation progress checkpoints.

    Provides persistence for evaluation sessions, enabling:
    - Save progress at any point
    - Resume from last checkpoint after interruption
    - Track per-persona progress
    - Handle corrupted checkpoints gracefully
    """

    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        logger.info(f"CheckpointManager initialized with directory: {checkpoint_dir}")

    def _get_checkpoint_path(self, session_id: str) -> Path:
        """Get the file path for a session's checkpoint."""
        safe_session_id = session_id.replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"checkpoint_{safe_session_id}.json"

    def create_session(
        self,
        session_id: str,
        total_queries: int,
        personas: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationProgress:
        """
        Create a new evaluation session.

        Args:
            session_id: Unique session identifier
            total_queries: Total number of queries to evaluate
            personas: List of persona names (optional)
            metadata: Additional session metadata

        Returns:
            EvaluationProgress for the new session
        """
        now = datetime.now().isoformat()

        persona_progresses = {}
        if personas:
            queries_per_persona = total_queries // len(personas)
            for persona in personas:
                persona_progresses[persona] = PersonaProgress(
                    persona_name=persona,
                    total_queries=queries_per_persona,
                    completed_queries=0,
                    failed_queries=0,
                )

        progress = EvaluationProgress(
            session_id=session_id,
            started_at=now,
            updated_at=now,
            status="running",
            total_queries=total_queries,
            completed_queries=0,
            failed_queries=0,
            personas=persona_progresses,
            metadata=metadata or {},
        )

        self.save_checkpoint(session_id, progress)
        logger.info(f"Created new session: {session_id} with {total_queries} queries")

        return progress

    def save_checkpoint(self, session_id: str, progress: EvaluationProgress) -> None:
        """
        Save evaluation progress to checkpoint file.

        Thread-safe operation that atomically writes the checkpoint.

        Args:
            session_id: Session identifier
            progress: EvaluationProgress to save
        """
        with self._lock:
            # Update timestamp
            progress.updated_at = datetime.now().isoformat()

            checkpoint_path = self._get_checkpoint_path(session_id)

            # Write to temp file first, then rename for atomicity
            temp_path = checkpoint_path.with_suffix(".tmp")

            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(progress.to_dict(), f, indent=2, ensure_ascii=False)

                # Atomic rename
                temp_path.rename(checkpoint_path)
                logger.debug(f"Saved checkpoint for session: {session_id}")

            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                # Clean up temp file if exists
                if temp_path.exists():
                    temp_path.unlink()
                raise

    def load_checkpoint(self, session_id: str) -> Optional[EvaluationProgress]:
        """
        Load evaluation progress from checkpoint file.

        Args:
            session_id: Session identifier

        Returns:
            EvaluationProgress or None if checkpoint doesn't exist or is corrupted
        """
        with self._lock:
            checkpoint_path = self._get_checkpoint_path(session_id)

            if not checkpoint_path.exists():
                logger.debug(f"No checkpoint found for session: {session_id}")
                return None

            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                progress = EvaluationProgress.from_dict(data)
                logger.info(f"Loaded checkpoint for session: {session_id}")
                return progress

            except json.JSONDecodeError as e:
                logger.error(f"Corrupted checkpoint file: {e}")
                return None

            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return None

    def update_progress(
        self,
        session_id: str,
        persona: Optional[str] = None,
        query_id: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[EvaluationProgress]:
        """
        Update evaluation progress.

        Args:
            session_id: Session identifier
            persona: Persona name (optional)
            query_id: Query identifier (optional)
            result: Evaluation result to store (optional)
            error: Error message if failed (optional)

        Returns:
            Updated EvaluationProgress or None if session not found
        """
        progress = self.load_checkpoint(session_id)
        if progress is None:
            logger.warning(f"Cannot update: session {session_id} not found")
            return None

        # Update persona progress if specified
        if persona and persona in progress.personas:
            persona_progress = progress.personas[persona]
            if error:
                persona_progress.failed_queries += 1
                if query_id:
                    persona_progress.failed_query_ids.append(query_id)
            else:
                persona_progress.completed_queries += 1
                if query_id:
                    persona_progress.completed_query_ids.append(query_id)
            persona_progress.last_query = query_id
            persona_progress.last_timestamp = datetime.now().isoformat()

        # Update overall progress
        if error:
            progress.failed_queries += 1
            progress.errors.append(error)
        else:
            progress.completed_queries += 1
            if result:
                progress.results.append(result)

        # Check if complete
        if progress.completed_queries + progress.failed_queries >= progress.total_queries:
            progress.status = "completed"

        self.save_checkpoint(session_id, progress)
        return progress

    def pause_session(self, session_id: str) -> Optional[EvaluationProgress]:
        """
        Mark a session as paused.

        Args:
            session_id: Session identifier

        Returns:
            Updated EvaluationProgress or None if not found
        """
        progress = self.load_checkpoint(session_id)
        if progress is None:
            return None

        progress.status = "paused"
        self.save_checkpoint(session_id, progress)
        logger.info(f"Paused session: {session_id}")
        return progress

    def resume_session(self, session_id: str) -> Optional[EvaluationProgress]:
        """
        Mark a session as resumed.

        Args:
            session_id: Session identifier

        Returns:
            Updated EvaluationProgress or None if not found
        """
        progress = self.load_checkpoint(session_id)
        if progress is None:
            return None

        if not progress.can_resume:
            logger.warning(f"Session {session_id} cannot be resumed")
            return None

        progress.status = "running"
        self.save_checkpoint(session_id, progress)
        logger.info(f"Resumed session: {session_id}")
        return progress

    def clear_checkpoint(self, session_id: str) -> bool:
        """
        Delete a checkpoint file.

        Args:
            session_id: Session identifier

        Returns:
            True if checkpoint was deleted, False otherwise
        """
        with self._lock:
            checkpoint_path = self._get_checkpoint_path(session_id)

            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                    logger.info(f"Cleared checkpoint for session: {session_id}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to clear checkpoint: {e}")
                    return False

            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoint sessions.

        Returns:
            List of session info dictionaries
        """
        sessions = []

        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract summary info
                sessions.append({
                    "session_id": data.get("session_id"),
                    "status": data.get("status"),
                    "started_at": data.get("started_at"),
                    "updated_at": data.get("updated_at"),
                    "completion_rate": (
                        data.get("completed_queries", 0) / data.get("total_queries", 1) * 100
                        if data.get("total_queries", 0) > 0
                        else 0
                    ),
                    "can_resume": data.get("status") in ("paused", "failed"),
                })

            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")

        # Sort by updated_at descending
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions

    def get_pending_queries(
        self,
        session_id: str,
        persona: Optional[str] = None,
    ) -> List[str]:
        """
        Get list of query IDs that haven't been processed yet.

        Args:
            session_id: Session identifier
            persona: Optional persona to filter by

        Returns:
            List of pending query IDs
        """
        progress = self.load_checkpoint(session_id)
        if progress is None:
            return []

        pending = []

        if persona and persona in progress.personas:
            persona_progress = progress.personas[persona]
            # Return all query IDs minus completed and failed
            # Note: This assumes query IDs are known; in practice,
            # you might need to pass the full query list
            completed = set(persona_progress.completed_query_ids)
            failed = set(persona_progress.failed_query_ids)
            # This is a placeholder - actual implementation depends on how queries are tracked
            pending_count = persona_progress.total_queries - len(completed) - len(failed)
            pending = [f"pending_{i}" for i in range(pending_count)]

        return pending
