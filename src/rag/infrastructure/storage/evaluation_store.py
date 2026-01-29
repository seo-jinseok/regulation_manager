"""
Evaluation Store - Persistent storage for RAG evaluation results.

Provides JSON file-based storage with thread-safe operations,
historical data retrieval, and aggregate statistics.
"""

import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from ...domain.evaluation.models import EvaluationResult

logger = logging.getLogger(__name__)

# Platform-specific file locking (Unix-like systems only)
USE_FCNTL = platform.system() != "Windows"

if USE_FCNTL:
    import fcntl


@dataclass
class EvaluationStatistics:
    """
    Aggregate statistics for evaluation results.

    Provides summary metrics across multiple evaluations.
    """

    total_evaluations: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_contextual_precision: float
    avg_contextual_recall: float
    avg_overall_score: float
    pass_rate: float
    min_score: float
    max_score: float
    std_deviation: float
    timestamp_range: tuple[datetime, datetime]
    trend: str = "stable"  # "improving", "declining", "stable"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_evaluations": self.total_evaluations,
            "avg_faithfulness": self.avg_faithfulness,
            "avg_answer_relevancy": self.avg_answer_relevancy,
            "avg_contextual_precision": self.avg_contextual_precision,
            "avg_contextual_recall": self.avg_contextual_recall,
            "avg_overall_score": self.avg_overall_score,
            "pass_rate": self.pass_rate,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "std_deviation": self.std_deviation,
            "timestamp_range": (
                self.timestamp_range[0].isoformat(),
                self.timestamp_range[1].isoformat(),
            ),
            "trend": self.trend,
        }


class EvaluationStore:
    """
    Thread-safe JSON file-based storage for evaluation results.

    Provides persistent storage with automatic file naming,
    historical data retrieval, and statistical analysis.
    """

    def __init__(self, storage_dir: str = "data/evaluations"):
        """
        Initialize the evaluation store.

        Args:
            storage_dir: Directory path for storing evaluation files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        logger.info(f"Initialized EvaluationStore with directory: {self.storage_dir}")

    def save_evaluation(self, result: EvaluationResult) -> None:
        """
        Save an evaluation result with timestamp-based filename.

        Thread-safe operation that creates a new JSON file for each evaluation.

        Args:
            result: EvaluationResult to save
        """
        with self._lock:
            filename = self._generate_filename(result.timestamp)
            filepath = self.storage_dir / filename

            data = result.to_dict()

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    # Use file locking for write safety (Unix-like systems)
                    if USE_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        try:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    else:
                        json.dump(data, f, indent=2, ensure_ascii=False)

                logger.info(f"Saved evaluation result to {filename}")
            except Exception as e:
                logger.error(f"Failed to save evaluation to {filepath}: {e}")
                raise

    def get_latest_evaluation(self) -> Optional[EvaluationResult]:
        """
        Retrieve the most recent evaluation result.

        Returns:
            Latest EvaluationResult or None if no evaluations exist
        """
        with self._lock:
            files = self._list_evaluation_files()
            if not files:
                return None

            latest_file = files[-1]  # Files are sorted by timestamp
            return self._load_evaluation(latest_file)

    def get_history_since(self, start_date: datetime) -> List[EvaluationResult]:
        """
        Retrieve all evaluation results since a specific date.

        Args:
            start_date: Start date for history retrieval (inclusive)

        Returns:
            List of EvaluationResult sorted by timestamp (newest first)
        """
        with self._lock:
            files = self._list_evaluation_files()
            results = []

            for filepath in files:
                # Extract timestamp from filename
                try:
                    file_timestamp = self._extract_timestamp_from_filename(
                        filepath.name
                    )
                    if file_timestamp >= start_date:
                        result = self._load_evaluation(filepath)
                        if result:
                            results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
                    continue

            return results

    def get_baseline(self) -> Optional[EvaluationResult]:
        """
        Get baseline evaluation for comparison.

        Returns the first stored evaluation as baseline,
        typically representing the initial performance.

        Returns:
            Baseline EvaluationResult or None if no evaluations exist
        """
        with self._lock:
            files = self._list_evaluation_files()
            if not files:
                return None

            baseline_file = files[0]  # First file = oldest = baseline
            return self._load_evaluation(baseline_file)

    def get_statistics(
        self, days: Optional[int] = None, start_date: Optional[datetime] = None
    ) -> EvaluationStatistics:
        """
        Calculate aggregate statistics for evaluation results.

        Args:
            days: Number of recent days to include (None = all time)
            start_date: Alternative to days, specific start date

        Returns:
            EvaluationStatistics with aggregate metrics
        """
        with self._lock:
            # Load results directly without calling get_history_since to avoid deadlock
            files = self._list_evaluation_files()
            results = []
            cutoff_date = None

            if start_date:
                cutoff_date = start_date
            elif days:
                cutoff_date = datetime.now() - timedelta(days=days)

            for filepath in files:
                try:
                    # Extract timestamp from filename for filtering
                    file_timestamp = self._extract_timestamp_from_filename(
                        filepath.name
                    )

                    # Filter by date if specified
                    if cutoff_date and file_timestamp < cutoff_date:
                        continue

                    result = self._load_evaluation(filepath)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
                    continue

            if not results:
                # Return empty statistics
                return EvaluationStatistics(
                    total_evaluations=0,
                    avg_faithfulness=0.0,
                    avg_answer_relevancy=0.0,
                    avg_contextual_precision=0.0,
                    avg_contextual_recall=0.0,
                    avg_overall_score=0.0,
                    pass_rate=0.0,
                    min_score=0.0,
                    max_score=0.0,
                    std_deviation=0.0,
                    timestamp_range=(datetime.now(), datetime.now()),
                    trend="stable",
                )

            # Calculate statistics
            total = len(results)
            faithfulness_scores = [r.faithfulness for r in results]
            relevancy_scores = [r.answer_relevancy for r in results]
            precision_scores = [r.contextual_precision for r in results]
            recall_scores = [r.contextual_recall for r in results]
            overall_scores = [r.overall_score for r in results]

            avg_faithfulness = sum(faithfulness_scores) / total
            avg_relevancy = sum(relevancy_scores) / total
            avg_precision = sum(precision_scores) / total
            avg_recall = sum(recall_scores) / total
            avg_overall = sum(overall_scores) / total

            passed_count = sum(1 for r in results if r.passed)
            pass_rate = passed_count / total

            min_score = min(overall_scores)
            max_score = max(overall_scores)

            # Calculate standard deviation
            variance = sum((s - avg_overall) ** 2 for s in overall_scores) / total
            std_deviation = variance**0.5

            # Determine timestamp range
            timestamps = [r.timestamp for r in results]
            timestamp_range = (min(timestamps), max(timestamps))

            # Determine trend (compare first half vs second half)
            trend = self._calculate_trend(results)

            return EvaluationStatistics(
                total_evaluations=total,
                avg_faithfulness=round(avg_faithfulness, 4),
                avg_answer_relevancy=round(avg_relevancy, 4),
                avg_contextual_precision=round(avg_precision, 4),
                avg_contextual_recall=round(avg_recall, 4),
                avg_overall_score=round(avg_overall, 4),
                pass_rate=round(pass_rate, 4),
                min_score=round(min_score, 4),
                max_score=round(max_score, 4),
                std_deviation=round(std_deviation, 4),
                timestamp_range=timestamp_range,
                trend=trend,
            )

    def compare_to_baseline(self, current_result: EvaluationResult) -> Dict[str, Any]:
        """
        Compare current evaluation result to baseline.

        Args:
            current_result: Current evaluation result

        Returns:
            Dictionary with comparison metrics and improvements
        """
        baseline = self.get_baseline()
        if not baseline:
            return {
                "has_baseline": False,
                "message": "No baseline available for comparison",
            }

        return {
            "has_baseline": True,
            "baseline_overall": baseline.overall_score,
            "current_overall": current_result.overall_score,
            "overall_delta": current_result.overall_score - baseline.overall_score,
            "improvements": {
                "faithfulness": current_result.faithfulness - baseline.faithfulness,
                "answer_relevancy": (
                    current_result.answer_relevancy - baseline.answer_relevancy
                ),
                "contextual_precision": (
                    current_result.contextual_precision - baseline.contextual_precision
                ),
                "contextual_recall": (
                    current_result.contextual_recall - baseline.contextual_recall
                ),
            },
            "is_improved": current_result.overall_score > baseline.overall_score,
        }

    def _generate_filename(self, timestamp: datetime) -> str:
        """
        Generate filename based on timestamp.

        Format: evaluation_YYYYMMDD_HHMMSS.json

        Args:
            timestamp: Timestamp for filename

        Returns:
            Filename string
        """
        return f"evaluation_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

    def _extract_timestamp_from_filename(self, filename: str) -> datetime:
        """
        Extract timestamp from evaluation filename.

        Args:
            filename: Evaluation filename

        Returns:
            Datetime object

        Raises:
            ValueError: If filename format is invalid
        """
        # Remove .json extension
        basename = filename.replace(".json", "")

        # Extract timestamp part
        try:
            timestamp_str = basename.split("_", 1)[1]
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid evaluation filename format: {filename}") from e

    def _list_evaluation_files(self) -> List[Path]:
        """
        List all evaluation files sorted by timestamp.

        Returns:
            List of file paths sorted oldest to newest
        """
        pattern = "evaluation_*.json"
        files = sorted(self.storage_dir.glob(pattern))
        return files

    def _load_evaluation(self, filepath: Path) -> Optional[EvaluationResult]:
        """
        Load evaluation result from file.

        Args:
            filepath: Path to evaluation file

        Returns:
            EvaluationResult or None if loading fails
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                # Use file locking for read safety (Unix-like systems)
                if USE_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                else:
                    data = json.load(f)

            return EvaluationResult.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load evaluation from {filepath}: {e}")
            return None

    def _calculate_trend(self, results: List[EvaluationResult]) -> str:
        """
        Calculate performance trend from results.

        Compares first half vs second half of results.

        Args:
            results: List of evaluation results

        Returns:
            Trend string: "improving", "declining", or "stable"
        """
        if len(results) < 2:
            return "stable"

        # Split results into two halves
        mid = len(results) // 2
        first_half = results[:mid]
        second_half = results[mid:]

        # Calculate average overall scores
        first_avg = sum(r.overall_score for r in first_half) / len(first_half)
        second_avg = sum(r.overall_score for r in second_half) / len(second_half)

        # Determine trend with 2% threshold for stability
        delta = second_avg - first_avg
        threshold = 0.02  # 2% threshold

        if delta > threshold:
            return "improving"
        elif delta < -threshold:
            return "declining"
        else:
            return "stable"
