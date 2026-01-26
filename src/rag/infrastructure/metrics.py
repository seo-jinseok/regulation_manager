"""
Reranking Metrics Storage and Reporting System (Cycle 3).

Provides persistent storage, time-based aggregation, and reporting
for RerankingMetrics collected from SearchUseCase.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..domain.entities import RerankingMetrics

logger = logging.getLogger(__name__)


class MetricsRepository:
    """
    Repository for storing and retrieving reranking metrics.

    Supports:
    - JSON file persistence
    - Time-based aggregation (hourly, daily, weekly)
    - Historical metrics comparison
    """

    def __init__(self, storage_dir: str = ".metrics/reranking"):
        """
        Initialize metrics repository.

        Args:
            storage_dir: Directory to store metrics JSON files.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics(
        self,
        metrics: RerankingMetrics,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Save metrics to JSON file.

        Args:
            metrics: RerankingMetrics to save.
            session_id: Optional session identifier.

        Returns:
            Path to saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_suffix = f"_{session_id}" if session_id else ""
        filename = f"reranking_metrics_{timestamp}{session_suffix}.json"
        filepath = self.storage_dir / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "metrics": metrics.to_dict(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved reranking metrics to {filepath}")
        return str(filepath)

    def load_metrics(self, filepath: str) -> Dict[str, Any]:
        """
        Load metrics from JSON file.

        Args:
            filepath: Path to metrics JSON file.

        Returns:
            Dictionary with timestamp, session_id, and metrics.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def load_recent_metrics(
        self, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Load metrics from the last N hours.

        Args:
            hours: Number of hours to look back.

        Returns:
            List of metrics data dictionaries.
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = []

        for filepath in sorted(self.storage_dir.glob("reranking_metrics_*.json")):
            try:
                data = self.load_metrics(str(filepath))
                timestamp = datetime.fromisoformat(data["timestamp"])

                if timestamp >= cutoff_time:
                    recent_metrics.append(data)
            except Exception as e:
                logger.warning(f"Failed to load metrics from {filepath}: {e}")

        return recent_metrics

    def aggregate_metrics(
        self, metrics_list: List[Dict[str, Any]]
    ) -> RerankingMetrics:
        """
        Aggregate multiple RerankingMetrics into one.

        Args:
            metrics_list: List of metrics data dictionaries.

        Returns:
            Aggregated RerankingMetrics.
        """
        if not metrics_list:
            return RerankingMetrics()

        aggregated = RerankingMetrics()

        for data in metrics_list:
            m = data["metrics"]
            aggregated.total_queries += m.get("total_queries", 0)
            aggregated.reranker_applied += m.get("reranker_applied", 0)
            aggregated.reranker_skipped += m.get("reranker_skipped", 0)
            aggregated.article_reference_skips += m.get("article_reference_skips", 0)
            aggregated.regulation_name_skips += m.get("regulation_name_skips", 0)
            aggregated.short_simple_skips += m.get("short_simple_skips", 0)
            aggregated.no_intent_skips += m.get("no_intent_skips", 0)
            aggregated.natural_question_applies += m.get("natural_question_applies", 0)
            aggregated.intent_applies += m.get("intent_applies", 0)
            aggregated.complex_applies += m.get("complex_applies", 0)
            aggregated.total_reranker_time_ms += m.get("total_reranker_time_ms", 0.0)
            aggregated.total_skip_saved_time_ms += m.get("total_skip_saved_time_ms", 0.0)

        return aggregated

    def get_daily_summary(self, days: int = 1) -> RerankingMetrics:
        """
        Get aggregated metrics for the last N days.

        Args:
            days: Number of days to aggregate.

        Returns:
            Aggregated RerankingMetrics.
        """
        hours = days * 24
        recent_metrics = self.load_recent_metrics(hours=hours)
        return self.aggregate_metrics(recent_metrics)

    def clear_old_metrics(self, days: int = 7) -> int:
        """
        Delete metrics files older than N days.

        Args:
            days: Age threshold in days.

        Returns:
            Number of files deleted.
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0

        for filepath in self.storage_dir.glob("reranking_metrics_*.json"):
            try:
                # Parse timestamp from filename
                parts = filepath.stem.split("_")[2:4]
                if len(parts) == 2:
                    date_str, time_str = parts
                    file_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

                    if file_time < cutoff_time:
                        filepath.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted old metrics file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to process {filepath}: {e}")

        return deleted_count


class MetricsReporter:
    """
    Generate reports from reranking metrics.

    Supports:
    - Console summary output
    - HTML report generation
    - CSV export for analysis
    """

    def __init__(self, repository: Optional[MetricsRepository] = None):
        """
        Initialize metrics reporter.

        Args:
            repository: Optional MetricsRepository for loading historical data.
        """
        self.repository = repository or MetricsRepository()

    def print_summary(self, metrics: RerankingMetrics) -> None:
        """
        Print metrics summary to console.

        Args:
            metrics: RerankingMetrics to report.
        """
        print(metrics.get_summary())

    def generate_html_report(
        self,
        metrics: RerankingMetrics,
        output_path: str = ".metrics/reranking_report.html",
    ) -> str:
        """
        Generate HTML report for metrics.

        Args:
            metrics: RerankingMetrics to report.
            output_path: Path to output HTML file.

        Returns:
            Path to generated HTML file.
        """
        html_content = self._create_html_report(metrics)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated HTML report: {output_file}")
        return str(output_file)

    def _create_html_report(self, metrics: RerankingMetrics) -> str:
        """Create HTML content for metrics report."""
        m = metrics.to_dict()

        # Calculate percentages for progress bars
        skip_rate = m["skip_rate"] * 100
        apply_rate = m["apply_rate"] * 100

        # Color coding
        skip_color = "#ef4444" if skip_rate > 50 else "#f59e0b" if skip_rate > 30 else "#10b981"
        apply_color = "#10b981" if apply_rate > 50 else "#f59e0b" if apply_rate > 30 else "#ef4444"

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reranking Metrics Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 24px;
        }}
        h1 {{
            color: #1f2937;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 12px;
        }}
        h2 {{
            color: #374151;
            margin-top: 24px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
            margin: 16px 0;
        }}
        .metric-card {{
            background: #f9fafb;
            padding: 16px;
            border-radius: 6px;
            border-left: 4px solid #3b82f6;
        }}
        .metric-label {{
            font-size: 14px;
            color: #6b7280;
            margin-bottom: 4px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #1f2937;
        }}
        .progress-bar {{
            height: 24px;
            background: #e5e7eb;
            border-radius: 4px;
            overflow: hidden;
            margin: 8px 0;
        }}
        .progress-fill {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.3s ease;
        }}
        .breakdown {{
            margin: 16px 0;
        }}
        .breakdown-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e5e7eb;
        }}
        .timestamp {{
            color: #6b7280;
            font-size: 12px;
            margin-top: 24px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Reranking Metrics Report</h1>

        <h2>Overview</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Queries</div>
                <div class="metric-value">{m["total_queries"]:,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Reranker Time</div>
                <div class="metric-value">{m["avg_reranker_time_ms"]:.2f}ms</div>
            </div>
        </div>

        <h2>Reranker Usage</h2>
        <div class="metric-card">
            <div class="metric-label">Applied: {m["reranker_applied"]:,} ({apply_rate:.1f}%)</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {apply_rate}%; background: {apply_color};">
                    {apply_rate:.1f}%
                </div>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Skipped: {m["reranker_skipped"]:,} ({skip_rate:.1f}%)</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {skip_rate}%; background: {skip_color};">
                    {skip_rate:.1f}%
                </div>
            </div>
        </div>

        <h2>Skip Reasons Breakdown</h2>
        <div class="breakdown">
            <div class="breakdown-item">
                <span>Article Reference</span>
                <strong>{m["article_reference_skips"]:,}</strong>
            </div>
            <div class="breakdown-item">
                <span>Regulation Name</span>
                <strong>{m["regulation_name_skips"]:,}</strong>
            </div>
            <div class="breakdown-item">
                <span>Short Simple</span>
                <strong>{m["short_simple_skips"]:,}</strong>
            </div>
            <div class="breakdown-item">
                <span>No Intent</span>
                <strong>{m["no_intent_skips"]:,}</strong>
            </div>
        </div>

        <h2>Apply Types Breakdown</h2>
        <div class="breakdown">
            <div class="breakdown-item">
                <span>Natural Questions</span>
                <strong>{m["natural_question_applies"]:,}</strong>
            </div>
            <div class="breakdown-item">
                <span>Intent Queries</span>
                <strong>{m["intent_applies"]:,}</strong>
            </div>
            <div class="breakdown-item">
                <span>Complex Queries</span>
                <strong>{m["complex_applies"]:,}</strong>
            </div>
        </div>

        <h2>Performance Impact</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Reranker Time</div>
                <div class="metric-value">{m["total_reranker_time_ms"]:.2f}ms</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Est. Time Saved</div>
                <div class="metric-value">{m["estimated_time_saved_ms"]:.2f}ms</div>
            </div>
        </div>

        <div class="timestamp">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>"""
        return html

    def export_to_csv(
        self,
        metrics: RerankingMetrics,
        output_path: str = ".metrics/reranking_metrics.csv",
    ) -> str:
        """
        Export metrics to CSV file.

        Args:
            metrics: RerankingMetrics to export.
            output_path: Path to output CSV file.

        Returns:
            Path to generated CSV file.
        """
        import csv

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        m = metrics.to_dict()

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])

            for key, value in m.items():
                if isinstance(value, float):
                    writer.writerow([key, f"{value:.4f}"])
                else:
                    writer.writerow([key, value])

        logger.info(f"Exported metrics to CSV: {output_file}")
        return str(output_file)

    def compare_sessions(
        self, session_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare metrics across multiple sessions.

        Args:
            session_metrics: List of metrics data dictionaries.

        Returns:
            Comparison summary with best/worst performing sessions.
        """
        if not session_metrics:
            return {}

        # Extract key metrics for comparison
        comparisons = []
        for data in session_metrics:
            m = data["metrics"]
            comparisons.append(
                {
                    "session_id": data.get("session_id", "unknown"),
                    "timestamp": data.get("timestamp"),
                    "skip_rate": m.get("skip_rate", 0),
                    "apply_rate": m.get("apply_rate", 0),
                    "avg_time": m.get("avg_reranker_time_ms", 0),
                    "total_queries": m.get("total_queries", 0),
                }
            )

        # Find best and worst
        if comparisons:
            best_skip = max(comparisons, key=lambda x: x["skip_rate"])
            worst_skip = min(comparisons, key=lambda x: x["skip_rate"])
            fastest = min(comparisons, key=lambda x: x["avg_time"])
            most_active = max(comparisons, key=lambda x: x["total_queries"])

            return {
                "total_sessions": len(comparisons),
                "best_skip_rate": best_skip,
                "worst_skip_rate": worst_skip,
                "fastest_avg_time": fastest,
                "most_active_session": most_active,
                "all_sessions": comparisons,
            }

        return {}
