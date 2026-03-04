"""
Improvement Radar for evaluation.

SPEC: SPEC-RAG-EVAL-002
EARS: EARS-U-014 (Failure Clustering), EARS-U-015 (Roadmap), EARS-U-016 (Trend)
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FailureCategory:
    """Root cause categories for failure clustering."""
    RETRIEVAL_MISS = "retrieval_miss"
    CITATION_HALLUCINATION = "citation_hallucination"
    INCOMPLETE_ANSWER = "incomplete_answer"
    WRONG_REGULATION = "wrong_regulation"
    OUTDATED_INFO = "outdated_info"
    CONTEXT_OVERFLOW = "context_overflow"
    AMBIGUITY_HANDLING = "ambiguity_handling"
    FORMAT_ERROR = "format_error"
    UNKNOWN = "unknown"


@dataclass
class FailureCluster:
    """A group of failures with the same root cause."""
    category: str
    count: int = 0
    examples: List[Dict[str, Any]] = field(default_factory=list)
    avg_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "count": self.count,
            "avg_score": round(self.avg_score, 3),
            "examples": self.examples[:5],  # Limit examples
        }


class FailureClusterer:
    """Groups failures by root cause.

    EARS-U-014: Classify failures into actionable categories.
    """

    # Keyword patterns for classification
    CLASSIFICATION_RULES = [
        (FailureCategory.CITATION_HALLUCINATION, [
            r"인용.*불일치", r"citation.*hallucin", r"잘못된.*출처",
            r"없는.*조항", r"존재하지.*않는.*규정",
        ]),
        (FailureCategory.RETRIEVAL_MISS, [
            r"검색.*실패", r"관련.*문서.*없", r"retrieval.*miss",
            r"찾을.*수.*없", r"문맥.*부족",
        ]),
        (FailureCategory.INCOMPLETE_ANSWER, [
            r"불완전", r"incomplete", r"누락", r"빠진.*내용",
            r"일부.*만", r"부분적",
        ]),
        (FailureCategory.WRONG_REGULATION, [
            r"잘못된.*규정", r"다른.*규정", r"wrong.*regulation",
            r"혼동", r"착각",
        ]),
        (FailureCategory.FORMAT_ERROR, [
            r"형식.*오류", r"format", r"구조.*잘못",
        ]),
        (FailureCategory.AMBIGUITY_HANDLING, [
            r"모호", r"ambig", r"명확하지.*않",
        ]),
    ]

    def cluster(self, results: List[Dict[str, Any]]) -> List[FailureCluster]:
        """Cluster failed results by root cause.

        Args:
            results: List of evaluation result dicts. Each should have:
                - query: str
                - passed: bool
                - overall_score: float
                - reasoning: str (optional)
                - issues: list[str] (optional)

        Returns:
            Sorted list of failure clusters (by count desc).
        """
        clusters: Dict[str, FailureCluster] = {}

        failed = [r for r in results if not r.get("passed", True)]

        for result in failed:
            category = self._classify(result)
            if category not in clusters:
                clusters[category] = FailureCluster(category=category)

            cluster = clusters[category]
            cluster.count += 1
            score = result.get("overall_score", 0.0)
            # Running average
            cluster.avg_score = (
                (cluster.avg_score * (cluster.count - 1) + score) / cluster.count
            )
            if len(cluster.examples) < 5:
                cluster.examples.append({
                    "query": result.get("query", "")[:100],
                    "score": round(score, 3),
                })

        # Sort by count desc
        return sorted(clusters.values(), key=lambda c: c.count, reverse=True)

    def _classify(self, result: Dict[str, Any]) -> str:
        """Classify a failure into a root cause category."""
        # Combine reasoning and issues into searchable text
        text_parts = []
        if result.get("reasoning"):
            text_parts.append(str(result["reasoning"]))
        for issue in result.get("issues", []):
            text_parts.append(str(issue))
        text = " ".join(text_parts).lower()

        # Score-based heuristics
        scores = {
            "accuracy": result.get("accuracy", 1.0),
            "completeness": result.get("completeness", 1.0),
            "citations": result.get("citations", 1.0),
            "context_relevance": result.get("context_relevance", 1.0),
        }

        # Check keyword patterns
        for category, patterns in self.CLASSIFICATION_RULES:
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return category

        # Score-based fallback classification
        if scores["citations"] < 0.5:
            return FailureCategory.CITATION_HALLUCINATION
        if scores["context_relevance"] < 0.5:
            return FailureCategory.RETRIEVAL_MISS
        if scores["completeness"] < 0.5:
            return FailureCategory.INCOMPLETE_ANSWER
        if scores["accuracy"] < 0.5:
            return FailureCategory.WRONG_REGULATION

        return FailureCategory.UNKNOWN


@dataclass
class RoadmapItem:
    """A prioritized improvement item."""
    title: str
    category: str
    priority_score: float  # frequency × impact × fixability
    frequency: int
    impact: float  # 0-1, based on avg score delta
    fixability: float  # 0-1, estimated ease of fix
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "category": self.category,
            "priority_score": round(self.priority_score, 2),
            "frequency": self.frequency,
            "impact": round(self.impact, 2),
            "fixability": round(self.fixability, 2),
            "description": self.description,
        }


# Fixability estimates by category (higher = easier to fix)
FIXABILITY_MAP = {
    FailureCategory.FORMAT_ERROR: 0.9,
    FailureCategory.CITATION_HALLUCINATION: 0.7,
    FailureCategory.RETRIEVAL_MISS: 0.6,
    FailureCategory.INCOMPLETE_ANSWER: 0.5,
    FailureCategory.WRONG_REGULATION: 0.5,
    FailureCategory.AMBIGUITY_HANDLING: 0.4,
    FailureCategory.OUTDATED_INFO: 0.3,
    FailureCategory.CONTEXT_OVERFLOW: 0.3,
    FailureCategory.UNKNOWN: 0.2,
}

# Human-readable titles by category
IMPROVEMENT_TITLES = {
    FailureCategory.RETRIEVAL_MISS: "Improve document retrieval recall",
    FailureCategory.CITATION_HALLUCINATION: "Fix citation accuracy",
    FailureCategory.INCOMPLETE_ANSWER: "Improve answer completeness",
    FailureCategory.WRONG_REGULATION: "Fix regulation identification",
    FailureCategory.OUTDATED_INFO: "Update regulation content",
    FailureCategory.CONTEXT_OVERFLOW: "Optimize context window usage",
    FailureCategory.AMBIGUITY_HANDLING: "Improve ambiguity resolution",
    FailureCategory.FORMAT_ERROR: "Fix response formatting",
    FailureCategory.UNKNOWN: "Investigate misc. failures",
}


class RoadmapGenerator:
    """Generates prioritized improvement roadmap.

    EARS-U-015: Priority = frequency × impact × fixability
    """

    def generate(
        self, clusters: List[FailureCluster], total_queries: int = 1,
    ) -> List[RoadmapItem]:
        """Generate prioritized improvement roadmap from failure clusters.

        Args:
            clusters: Failure clusters from FailureClusterer.
            total_queries: Total number of queries evaluated.

        Returns:
            Sorted list of roadmap items (by priority desc).
        """
        items = []

        for cluster in clusters:
            if cluster.count == 0:
                continue

            frequency = cluster.count
            # Impact: how far from passing (1 - avg_score)
            impact = max(0.0, min(1.0, 1.0 - cluster.avg_score))
            fixability = FIXABILITY_MAP.get(cluster.category, 0.2)

            priority = frequency * impact * fixability

            items.append(RoadmapItem(
                title=IMPROVEMENT_TITLES.get(cluster.category, cluster.category),
                category=cluster.category,
                priority_score=priority,
                frequency=frequency,
                impact=impact,
                fixability=fixability,
                description=(
                    f"{frequency} failures ({frequency*100//max(total_queries,1)}% of queries). "
                    f"Avg score: {cluster.avg_score:.2f}."
                ),
            ))

        return sorted(items, key=lambda x: x.priority_score, reverse=True)


@dataclass
class TrendPoint:
    """A single data point in trend analysis."""
    timestamp: str
    avg_score: float
    pass_rate: float
    total_queries: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrendAnalysis:
    """Trend analysis of evaluation metrics over time."""
    data_points: List[TrendPoint] = field(default_factory=list)
    avg_score_trend: str = "stable"  # improving, degrading, stable
    pass_rate_trend: str = "stable"
    moving_avg_score: float = 0.0
    moving_avg_pass_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_points": [dp.to_dict() for dp in self.data_points[-10:]],
            "avg_score_trend": self.avg_score_trend,
            "pass_rate_trend": self.pass_rate_trend,
            "moving_avg_score": round(self.moving_avg_score, 3),
            "moving_avg_pass_rate": round(self.moving_avg_pass_rate, 3),
        }


class TrendAnalyzer:
    """Analyzes evaluation trends over time.

    EARS-U-016: Moving average over 3+ historical evaluation JSONs.
    """

    WINDOW_SIZE = 3  # Minimum window for trend detection
    IMPROVEMENT_THRESHOLD = 0.02  # 2% improvement to count as "improving"
    DEGRADATION_THRESHOLD = -0.02  # 2% drop to count as "degrading"

    def __init__(self, evaluations_dir: str = "data/evaluations"):
        self.evaluations_dir = Path(evaluations_dir)

    def analyze(self) -> TrendAnalysis:
        """Analyze historical evaluation data for trends."""
        analysis = TrendAnalysis()

        # Load historical data
        data_points = self._load_historical_data()
        if not data_points:
            return analysis

        analysis.data_points = data_points

        # Calculate moving averages
        if len(data_points) >= self.WINDOW_SIZE:
            recent = data_points[-self.WINDOW_SIZE:]
            analysis.moving_avg_score = sum(
                dp.avg_score for dp in recent
            ) / len(recent)
            analysis.moving_avg_pass_rate = sum(
                dp.pass_rate for dp in recent
            ) / len(recent)

            # Determine trends
            if len(data_points) >= self.WINDOW_SIZE * 2:
                older = data_points[-(self.WINDOW_SIZE * 2):-self.WINDOW_SIZE]
                older_avg_score = sum(dp.avg_score for dp in older) / len(older)
                older_avg_pass_rate = sum(dp.pass_rate for dp in older) / len(older)

                score_delta = analysis.moving_avg_score - older_avg_score
                rate_delta = analysis.moving_avg_pass_rate - older_avg_pass_rate

                analysis.avg_score_trend = self._classify_trend(score_delta)
                analysis.pass_rate_trend = self._classify_trend(rate_delta)
            else:
                # With exactly WINDOW_SIZE points, compare first/last
                first = data_points[0]
                last = data_points[-1]
                score_delta = last.avg_score - first.avg_score
                rate_delta = last.pass_rate - first.pass_rate
                analysis.avg_score_trend = self._classify_trend(score_delta)
                analysis.pass_rate_trend = self._classify_trend(rate_delta)
        elif data_points:
            # Fewer than WINDOW_SIZE: just use average
            analysis.moving_avg_score = sum(
                dp.avg_score for dp in data_points
            ) / len(data_points)
            analysis.moving_avg_pass_rate = sum(
                dp.pass_rate for dp in data_points
            ) / len(data_points)

        return analysis

    def _classify_trend(self, delta: float) -> str:
        """Classify a numeric delta into a trend direction."""
        if delta >= self.IMPROVEMENT_THRESHOLD:
            return "improving"
        elif delta <= self.DEGRADATION_THRESHOLD:
            return "degrading"
        return "stable"

    def _load_historical_data(self) -> List[TrendPoint]:
        """Load historical evaluation data sorted by timestamp."""
        if not self.evaluations_dir.exists():
            return []

        points = []
        for json_file in sorted(self.evaluations_dir.glob("eval_*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                # Extract summary metrics
                summary = data.get("summary", {})
                if not summary:
                    continue

                timestamp = data.get("timestamp", json_file.stem)
                avg_score = summary.get("avg_overall_score", 0.0)
                pass_rate = summary.get("pass_rate", 0.0)
                total_queries = summary.get("total_queries", 0)

                points.append(TrendPoint(
                    timestamp=str(timestamp),
                    avg_score=avg_score,
                    pass_rate=pass_rate,
                    total_queries=total_queries,
                ))
            except (json.JSONDecodeError, OSError, KeyError):
                continue

        return points


@dataclass
class ImprovementRadarReport:
    """Aggregated improvement radar output."""
    failure_clusters: List[FailureCluster] = field(default_factory=list)
    roadmap: List[RoadmapItem] = field(default_factory=list)
    trend: Optional[TrendAnalysis] = None
    never_nothing_to_improve: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "failure_clusters": [c.to_dict() for c in self.failure_clusters],
            "roadmap": [r.to_dict() for r in self.roadmap],
        }
        if self.trend:
            result["trend"] = self.trend.to_dict()
        if self.never_nothing_to_improve:
            result["improvement_note"] = self.never_nothing_to_improve
        return result


def run_improvement_radar(
    results: List[Dict[str, Any]],
    evaluations_dir: str = "data/evaluations",
) -> ImprovementRadarReport:
    """Run the improvement radar pipeline.

    EARS-AC-NF-4: Never reports "nothing to improve."
    """
    report = ImprovementRadarReport()

    # Cluster failures
    clusterer = FailureClusterer()
    report.failure_clusters = clusterer.cluster(results)

    # Generate roadmap
    generator = RoadmapGenerator()
    report.roadmap = generator.generate(
        report.failure_clusters, total_queries=len(results),
    )

    # Trend analysis
    analyzer = TrendAnalyzer(evaluations_dir=evaluations_dir)
    report.trend = analyzer.analyze()

    # EARS-AC-NF-4: Never "nothing to improve"
    if not report.roadmap:
        report.never_nothing_to_improve = (
            "All queries passed! Suggested next steps: "
            "escalate difficulty tier, add adversarial queries, "
            "test cross-regulation scenarios, or review latency/consistency metrics."
        )

    return report
