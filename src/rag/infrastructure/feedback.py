"""
Feedback Collection for RAG System.

Collects and stores user feedback on search results
to enable continuous improvement of the system.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FeedbackEntry:
    """A single feedback entry."""

    timestamp: str
    query: str
    result_id: str  # Chunk ID
    rule_code: str
    rating: str  # "positive", "negative", "neutral"
    comment: str = ""
    matched_intents: List[str] = None
    rewritten_query: str = ""

    def __post_init__(self):
        if self.matched_intents is None:
            self.matched_intents = []


@dataclass
class FeedbackStats:
    """Statistics from feedback data."""

    total_entries: int
    positive_count: int
    negative_count: int
    neutral_count: int
    positive_rate: float
    top_negative_queries: List[Dict[str, Any]]
    top_negative_rule_codes: List[Dict[str, Any]]


class FeedbackCollector:
    """
    Collects and manages user feedback on search results.

    Stores feedback in a JSONL file for analysis and
    continuous improvement of the RAG system.
    """

    def __init__(self, feedback_path: Optional[str] = None):
        """
        Initialize feedback collector.

        Args:
            feedback_path: Path to feedback log file.
        """
        self._feedback_path = feedback_path or self._default_path()
        self._ensure_file_exists()

    def _default_path(self) -> str:
        """Get default feedback log path."""
        from ..config import get_config

        return str(get_config().feedback_log_path_resolved)

    def _ensure_file_exists(self):
        """Create feedback file if it doesn't exist."""
        path = Path(self._feedback_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()

    def record_feedback(
        self,
        query: str,
        result_id: str,
        rule_code: str,
        rating: str,
        comment: str = "",
        matched_intents: Optional[List[str]] = None,
        rewritten_query: str = "",
    ) -> FeedbackEntry:
        """
        Record a feedback entry.

        Args:
            query: Original search query.
            result_id: ID of the result chunk.
            rule_code: Rule code of the result.
            rating: "positive", "negative", or "neutral".
            comment: Optional user comment.
            matched_intents: List of matched intent labels.
            rewritten_query: LLM-rewritten query.

        Returns:
            The recorded FeedbackEntry.
        """
        entry = FeedbackEntry(
            timestamp=datetime.now().isoformat(),
            query=query,
            result_id=result_id,
            rule_code=rule_code,
            rating=rating,
            comment=comment,
            matched_intents=matched_intents or [],
            rewritten_query=rewritten_query,
        )

        # Append to JSONL file
        with open(self._feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

        return entry

    def get_all_feedback(self) -> List[FeedbackEntry]:
        """
        Load all feedback entries.

        Returns:
            List of FeedbackEntry objects.
        """
        entries = []
        path = Path(self._feedback_path)

        if not path.exists() or path.stat().st_size == 0:
            return entries

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(FeedbackEntry(**data))
                except (json.JSONDecodeError, TypeError):
                    continue

        return entries

    def get_statistics(self) -> FeedbackStats:
        """
        Calculate statistics from feedback data.

        Returns:
            FeedbackStats with aggregated metrics.
        """
        entries = self.get_all_feedback()

        if not entries:
            return FeedbackStats(
                total_entries=0,
                positive_count=0,
                negative_count=0,
                neutral_count=0,
                positive_rate=0.0,
                top_negative_queries=[],
                top_negative_rule_codes=[],
            )

        positive = [e for e in entries if e.rating == "positive"]
        negative = [e for e in entries if e.rating == "negative"]
        neutral = [e for e in entries if e.rating == "neutral"]

        # Top negative queries
        query_counts: Dict[str, int] = {}
        rule_code_counts: Dict[str, int] = {}
        for e in negative:
            query_counts[e.query] = query_counts.get(e.query, 0) + 1
            rule_code_counts[e.rule_code] = rule_code_counts.get(e.rule_code, 0) + 1

        top_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_rule_codes = sorted(
            rule_code_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return FeedbackStats(
            total_entries=len(entries),
            positive_count=len(positive),
            negative_count=len(negative),
            neutral_count=len(neutral),
            positive_rate=len(positive) / len(entries) if entries else 0.0,
            top_negative_queries=[{"query": q, "count": c} for q, c in top_queries],
            top_negative_rule_codes=[
                {"rule_code": r, "count": c} for r, c in top_rule_codes
            ],
        )

    def get_negative_feedback(self) -> List[FeedbackEntry]:
        """Get only negative feedback entries."""
        return [e for e in self.get_all_feedback() if e.rating == "negative"]

    def clear_feedback(self):
        """Clear all feedback data."""
        Path(self._feedback_path).write_text("")

    def format_statistics(self, stats: FeedbackStats) -> str:
        """Format statistics as readable string."""
        lines = [
            "=" * 60,
            "피드백 통계",
            "=" * 60,
            f"총 피드백: {stats.total_entries}개",
            f"긍정: {stats.positive_count} ({stats.positive_rate:.1%})",
            f"부정: {stats.negative_count}",
            f"중립: {stats.neutral_count}",
            "-" * 60,
        ]

        if stats.top_negative_queries:
            lines.append("\n🔴 부정 피드백이 많은 쿼리:")
            for item in stats.top_negative_queries:
                lines.append(f'  - "{item["query"]}" ({item["count"]}회)')

        if stats.top_negative_rule_codes:
            lines.append("\n🔴 부정 피드백이 많은 규정:")
            for item in stats.top_negative_rule_codes:
                lines.append(f"  - {item['rule_code']} ({item['count']}회)")

        lines.append("=" * 60)
        return "\n".join(lines)
