"""
Progress Reporter for RAG Evaluation Sessions.

Provides real-time progress updates with ETA calculation.
Part of SPEC-RAG-EVAL-001 Milestone 3: Automation Pipeline.
"""

import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProgressInfo:
    """Current progress information."""

    total_queries: int
    completed_queries: int
    failed_queries: int
    current_persona: Optional[str]
    current_query: Optional[str]
    start_time: float
    elapsed_seconds: float
    queries_per_second: float
    estimated_remaining_seconds: float
    completion_percentage: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_queries": self.total_queries,
            "completed_queries": self.completed_queries,
            "failed_queries": self.failed_queries,
            "current_persona": self.current_persona,
            "current_query": self.current_query,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "queries_per_second": round(self.queries_per_second, 4),
            "estimated_remaining_seconds": round(self.estimated_remaining_seconds, 2),
            "completion_percentage": round(self.completion_percentage, 2),
        }

    def format_elapsed(self) -> str:
        """Format elapsed time as human-readable string."""
        return self._format_duration(self.elapsed_seconds)

    def format_eta(self) -> str:
        """Format ETA as human-readable string."""
        return self._format_duration(self.estimated_remaining_seconds)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        if seconds < 0:
            return "--:--:--"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class PersonaProgressInfo:
    """Progress information for a single persona."""

    persona_name: str
    total: int
    completed: int
    failed: int
    avg_score: float = 0.0

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "persona_name": self.persona_name,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "avg_score": round(self.avg_score, 3),
            "completion_percentage": round(self.completion_percentage, 1),
        }


class ProgressReporter:
    """
    Reports real-time progress for evaluation sessions.

    Features:
    - Real-time progress updates
    - ETA calculation based on current speed
    - Per-persona progress tracking
    - Support for both CLI and callback-based reporting
    """

    def __init__(
        self,
        total_queries: int,
        callback: Optional[Callable[[ProgressInfo], None]] = None,
        report_interval: float = 1.0,
        persona_counts: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the progress reporter.

        Args:
            total_queries: Total number of queries to evaluate
            callback: Optional callback function for progress updates
            report_interval: Minimum interval between reports in seconds
            persona_counts: Optional dict mapping persona names to query counts
        """
        self.total_queries = total_queries
        self.callback = callback
        self.report_interval = report_interval
        self.persona_counts = persona_counts or {}

        # State tracking
        self._start_time = time.time()
        self._last_report_time = 0.0
        self._completed_queries = 0
        self._failed_queries = 0
        self._current_persona: Optional[str] = None
        self._current_query: Optional[str] = None

        # Per-persona tracking
        self._persona_progress: Dict[str, PersonaProgressInfo] = {}
        for persona, count in self.persona_counts.items():
            self._persona_progress[persona] = PersonaProgressInfo(
                persona_name=persona,
                total=count,
                completed=0,
                failed=0,
            )

        # Speed calculation (moving average)
        self._completion_times: List[float] = []
        self._max_time_samples = 100

        logger.info(
            f"ProgressReporter initialized: {total_queries} queries, "
            f"{len(self.persona_counts)} personas"
        )

    def update(
        self,
        completed: int,
        persona: Optional[str] = None,
        query: Optional[str] = None,
        failed: bool = False,
        score: Optional[float] = None,
    ) -> ProgressInfo:
        """
        Update progress with completion of a query.

        Args:
            completed: Number of queries completed (usually 1)
            persona: Persona name for the query
            query: Query text or identifier
            failed: Whether the query failed
            score: Optional evaluation score for the query

        Returns:
            Current ProgressInfo
        """
        self._current_persona = persona
        self._current_query = query

        if failed:
            self._failed_queries += completed
        else:
            self._completed_queries += completed
            # Track completion time for speed calculation
            self._completion_times.append(time.time())
            if len(self._completion_times) > self._max_time_samples:
                self._completion_times.pop(0)

        # Update persona progress
        if persona and persona in self._persona_progress:
            persona_info = self._persona_progress[persona]
            if failed:
                persona_info.failed += completed
            else:
                persona_info.completed += completed
            if score is not None:
                # Update running average
                total_completed = persona_info.completed + persona_info.failed
                if total_completed > 0:
                    persona_info.avg_score = (
                        persona_info.avg_score * (total_completed - 1) + score
                    ) / total_completed

        progress_info = self.get_progress()

        # Check if we should report
        current_time = time.time()
        if current_time - self._last_report_time >= self.report_interval:
            self._report(progress_info)
            self._last_report_time = current_time

        return progress_info

    def get_progress(self) -> ProgressInfo:
        """
        Get current progress information.

        Returns:
            ProgressInfo with current state
        """
        elapsed = time.time() - self._start_time
        total_processed = self._completed_queries + self._failed_queries

        # Calculate queries per second
        if elapsed > 0:
            qps = total_processed / elapsed
        else:
            qps = 0.0

        # Calculate ETA
        if qps > 0:
            remaining_queries = self.total_queries - total_processed
            estimated_remaining = remaining_queries / qps
        else:
            estimated_remaining = 0.0

        # Calculate completion percentage
        if self.total_queries > 0:
            completion_pct = (total_processed / self.total_queries) * 100
        else:
            completion_pct = 0.0

        return ProgressInfo(
            total_queries=self.total_queries,
            completed_queries=self._completed_queries,
            failed_queries=self._failed_queries,
            current_persona=self._current_persona,
            current_query=self._current_query,
            start_time=self._start_time,
            elapsed_seconds=elapsed,
            queries_per_second=qps,
            estimated_remaining_seconds=estimated_remaining,
            completion_percentage=completion_pct,
        )

    def get_eta(self) -> timedelta:
        """
        Get estimated time remaining.

        Returns:
            timedelta representing estimated remaining time
        """
        progress = self.get_progress()
        return timedelta(seconds=int(progress.estimated_remaining_seconds))

    def get_persona_progress(self) -> Dict[str, PersonaProgressInfo]:
        """
        Get progress for all personas.

        Returns:
            Dictionary mapping persona names to their progress
        """
        return self._persona_progress.copy()

    def _report(self, progress: ProgressInfo) -> None:
        """
        Internal method to report progress.

        Args:
            progress: ProgressInfo to report
        """
        # Log progress
        logger.info(
            f"Progress: {progress.completed_queries}/{progress.total_queries} "
            f"({progress.completion_percentage:.1f}%) | "
            f"Failed: {progress.failed_queries} | "
            f"Speed: {progress.queries_per_second:.2f} q/s | "
            f"ETA: {progress.format_eta()}"
        )

        # Call callback if provided
        if self.callback:
            try:
                self.callback(progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def complete(self) -> ProgressInfo:
        """
        Mark evaluation as complete and return final progress.

        Returns:
            Final ProgressInfo
        """
        progress = self.get_progress()

        logger.info(
            f"Evaluation complete: {progress.completed_queries} passed, "
            f"{progress.failed_queries} failed in {progress.format_elapsed()}"
        )

        # Always report final state
        self._report(progress)

        return progress

    def format_cli_progress(self, progress: Optional[ProgressInfo] = None) -> str:
        """
        Format progress as a CLI-friendly string.

        Args:
            progress: Optional ProgressInfo (uses current if not provided)

        Returns:
            Formatted progress string
        """
        if progress is None:
            progress = self.get_progress()

        bar_width = 30
        filled = int(bar_width * progress.completion_percentage / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        return (
            f"[{bar}] {progress.completion_percentage:.1f}% | "
            f"{progress.completed_queries}/{progress.total_queries} | "
            f"ETA: {progress.format_eta()}"
        )

    def format_persona_summary(self) -> str:
        """
        Format a summary of per-persona progress.

        Returns:
            Multi-line string with persona progress
        """
        lines = ["Persona Progress:"]

        for persona_info in self._persona_progress.values():
            status = "✓" if persona_info.completed == persona_info.total else "○"
            lines.append(
                f"  {status} {persona_info.persona_name}: "
                f"{persona_info.completed}/{persona_info.total} "
                f"({persona_info.completion_percentage:.0f}%) "
                f"avg={persona_info.avg_score:.2f}"
            )

        return "\n".join(lines)


def create_progress_bar(
    current: int,
    total: int,
    width: int = 30,
    fill_char: str = "█",
    empty_char: str = "░",
) -> str:
    """
    Create a simple progress bar string.

    Args:
        current: Current progress value
        total: Total value
        width: Width of the bar in characters
        fill_char: Character for filled portion
        empty_char: Character for empty portion

    Returns:
        Progress bar string
    """
    if total == 0:
        percentage = 0
    else:
        percentage = (current / total) * 100

    filled = int(width * percentage / 100)
    bar = fill_char * filled + empty_char * (width - filled)

    return f"[{bar}] {percentage:.1f}%"
