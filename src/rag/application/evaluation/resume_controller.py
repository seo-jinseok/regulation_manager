"""
Resume Controller for RAG Evaluation Sessions.

Handles interrupted session detection and resumption logic.
Part of SPEC-RAG-EVAL-001 Milestone 3: Automation Pipeline.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .checkpoint_manager import CheckpointManager, PersonaProgress

logger = logging.getLogger(__name__)


@dataclass
class ResumeContext:
    """Context for resuming an evaluation session."""

    session_id: str
    can_resume: bool
    reason: str
    completed_count: int
    failed_count: int
    total_count: int
    remaining_personas: List[str]
    persona_progress: Dict[str, PersonaProgress]
    partial_results: List[Dict[str, Any]]
    errors: List[str]

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate."""
        if self.total_count == 0:
            return 0.0
        return (self.completed_count / self.total_count) * 100

    @property
    def needs_rerun_queries(self) -> bool:
        """Check if any queries need to be re-run."""
        return self.failed_count > 0 or self.completed_count < self.total_count


@dataclass
class MergedResults:
    """Results after merging old and new evaluation results."""

    results: List[Dict[str, Any]]
    total_count: int
    successful_count: int
    failed_count: int
    duplicates_merged: int
    conflicts_resolved: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_count": self.total_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "duplicates_merged": self.duplicates_merged,
            "conflicts_resolved": self.conflicts_resolved,
        }


class ResumeController:
    """
    Controls evaluation session resumption.

    Provides:
    - Detection of interrupted sessions
    - Resume from last checkpoint
    - Handling of partial results
    - Merging of resumed results with new results
    """

    def __init__(self, checkpoint_manager: CheckpointManager):
        """
        Initialize the resume controller.

        Args:
            checkpoint_manager: CheckpointManager instance for persistence
        """
        self.checkpoint_manager = checkpoint_manager
        logger.info("ResumeController initialized")

    def can_resume(self, session_id: str) -> Tuple[bool, str]:
        """
        Check if a session can be resumed.

        Args:
            session_id: Session identifier to check

        Returns:
            Tuple of (can_resume, reason)
        """
        progress = self.checkpoint_manager.load_checkpoint(session_id)

        if progress is None:
            return False, "Session not found"

        if progress.status == "completed":
            return False, "Session already completed"

        if progress.status not in ("paused", "failed", "running"):
            return False, f"Session status '{progress.status}' does not support resume"

        if progress.completed_queries >= progress.total_queries:
            return False, "All queries already processed"

        return True, f"Session can resume from {progress.completed_queries}/{progress.total_queries}"

    def get_resume_context(self, session_id: str) -> Optional[ResumeContext]:
        """
        Get full context for resuming a session.

        Args:
            session_id: Session identifier

        Returns:
            ResumeContext or None if session cannot be resumed
        """
        can_resume, reason = self.can_resume(session_id)

        if not can_resume:
            logger.info(f"Cannot resume session {session_id}: {reason}")
            return None

        progress = self.checkpoint_manager.load_checkpoint(session_id)
        if progress is None:
            return None

        # Determine remaining personas
        remaining_personas = []
        for persona_name, persona_prog in progress.personas.items():
            if persona_prog.completed_queries < persona_prog.total_queries:
                remaining_personas.append(persona_name)

        context = ResumeContext(
            session_id=session_id,
            can_resume=True,
            reason=reason,
            completed_count=progress.completed_queries,
            failed_count=progress.failed_queries,
            total_count=progress.total_queries,
            remaining_personas=remaining_personas,
            persona_progress=progress.personas.copy(),
            partial_results=progress.results.copy(),
            errors=progress.errors.copy(),
        )

        logger.info(
            f"Resume context created: {context.completed_count}/{context.total_count} "
            f"completed, {len(context.remaining_personas)} personas remaining"
        )

        return context

    def resume(
        self,
        session_id: str,
        evaluator: Callable,
        query_generator: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ) -> MergedResults:
        """
        Resume an interrupted evaluation session.

        Args:
            session_id: Session identifier
            evaluator: Async callable that evaluates a query
            query_generator: Optional callable to generate remaining queries
            progress_callback: Optional callback for progress updates

        Returns:
            MergedResults with all results

        Raises:
            ValueError: If session cannot be resumed
        """

        context = self.get_resume_context(session_id)
        if context is None:
            raise ValueError(f"Cannot resume session {session_id}")

        logger.info(f"Resuming session {session_id}...")

        # Mark session as running
        self.checkpoint_manager.resume_session(session_id)

        new_results = []

        # Process remaining queries for each persona
        for persona_name in context.remaining_personas:
            persona_prog = context.persona_progress[persona_name]

            # Determine which queries to run
            # In a real implementation, you'd use query_generator here
            # to generate only the missing queries

            remaining_count = (
                persona_prog.total_queries
                - persona_prog.completed_queries
                - persona_prog.failed_queries
            )

            logger.info(
                f"Processing {remaining_count} remaining queries for {persona_name}"
            )

            # This is a simplified loop - actual implementation would
            # use the query generator and evaluator properly
            for i in range(remaining_count):
                try:
                    # Placeholder for actual evaluation
                    # result = await evaluator(query, persona_name)
                    # new_results.append(result)

                    # Update checkpoint
                    self.checkpoint_manager.update_progress(
                        session_id=session_id,
                        persona=persona_name,
                        query_id=f"query_{i}",
                        result={"placeholder": True},
                    )

                    if progress_callback:
                        progress_callback(i + 1, remaining_count)

                except Exception as e:
                    logger.error(f"Error evaluating query: {e}")
                    self.checkpoint_manager.update_progress(
                        session_id=session_id,
                        persona=persona_name,
                        query_id=f"query_{i}",
                        error=str(e),
                    )

        # Merge results
        final_progress = self.checkpoint_manager.load_checkpoint(session_id)
        if final_progress:
            merged = self.merge_results(
                context.partial_results,
                new_results,
            )

            logger.info(
                f"Session {session_id} complete: "
                f"{merged.successful_count}/{merged.total_count} successful"
            )

            return merged

        # Fallback if checkpoint lost
        return MergedResults(
            results=context.partial_results + new_results,
            total_count=context.total_count,
            successful_count=len(context.partial_results) + len(new_results),
            failed_count=0,
            duplicates_merged=0,
            conflicts_resolved=0,
        )

    def merge_results(
        self,
        old_results: List[Dict[str, Any]],
        new_results: List[Dict[str, Any]],
        conflict_strategy: str = "keep_new",
    ) -> MergedResults:
        """
        Merge old and new evaluation results.

        Args:
            old_results: Results from previous session
            new_results: Results from current session
            conflict_strategy: How to handle conflicts ("keep_new", "keep_old", "keep_best")

        Returns:
            MergedResults with deduplicated and merged results
        """
        # Use query_id as the key for deduplication
        results_by_id: Dict[str, Dict[str, Any]] = {}
        duplicates = 0
        conflicts = 0

        # Add old results
        for result in old_results:
            query_id = result.get("query_id") or result.get("query", str(hash(str(result))))
            if query_id in results_by_id:
                duplicates += 1
            results_by_id[query_id] = result

        # Merge new results
        for result in new_results:
            query_id = result.get("query_id") or result.get("query", str(hash(str(result))))

            if query_id in results_by_id:
                conflicts += 1

                if conflict_strategy == "keep_new":
                    results_by_id[query_id] = result
                elif conflict_strategy == "keep_best":
                    # Keep the one with higher score
                    old_score = results_by_id[query_id].get("overall_score", 0)
                    new_score = result.get("overall_score", 0)
                    if new_score > old_score:
                        results_by_id[query_id] = result
                # "keep_old" - do nothing

            else:
                results_by_id[query_id] = result

        final_results = list(results_by_id.values())
        successful = sum(1 for r in final_results if r.get("passed", True))
        failed = len(final_results) - successful

        return MergedResults(
            results=final_results,
            total_count=len(final_results),
            successful_count=successful,
            failed_count=failed,
            duplicates_merged=duplicates,
            conflicts_resolved=conflicts,
        )

    def find_interrupted_sessions(self) -> List[Dict[str, Any]]:
        """
        Find all sessions that can be resumed.

        Returns:
            List of session info dictionaries
        """
        sessions = self.checkpoint_manager.list_sessions()
        interrupted = []

        for session in sessions:
            if session.get("can_resume"):
                interrupted.append(session)

        logger.info(f"Found {len(interrupted)} interrupted sessions")
        return interrupted

    def get_resume_recommendation(self) -> Optional[str]:
        """
        Get a recommendation for which session to resume.

        Returns:
            Session ID of the best candidate, or None
        """
        sessions = self.find_interrupted_sessions()

        if not sessions:
            return None

        # Prefer the most recent session with highest completion rate
        # that was interrupted (not completed)
        best = max(
            sessions,
            key=lambda s: (s.get("completion_rate", 0), s.get("updated_at", "")),
        )

        return best.get("session_id")

    def cleanup_completed_sessions(self, keep_days: int = 7) -> int:
        """
        Remove completed session checkpoints older than specified days.

        Args:
            keep_days: Number of days to keep completed sessions

        Returns:
            Number of sessions cleaned up
        """
        from datetime import datetime, timedelta

        sessions = self.checkpoint_manager.list_sessions()
        cleaned = 0
        cutoff = datetime.now() - timedelta(days=keep_days)

        for session in sessions:
            if session.get("status") != "completed":
                continue

            updated_at_str = session.get("updated_at")
            if not updated_at_str:
                continue

            try:
                updated_at = datetime.fromisoformat(updated_at_str)
                if updated_at < cutoff:
                    session_id = session.get("session_id")
                    if session_id:
                        self.checkpoint_manager.clear_checkpoint(session_id)
                        cleaned += 1
                        logger.info(f"Cleaned up completed session: {session_id}")

            except ValueError:
                continue

        logger.info(f"Cleaned up {cleaned} completed sessions")
        return cleaned
