"""Unit tests for Improvement Radar."""

import json
from pathlib import Path

import pytest

from src.rag.domain.evaluation.improvement_radar import (
    FailureCategory,
    FailureCluster,
    FailureClusterer,
    ImprovementRadarReport,
    RoadmapGenerator,
    RoadmapItem,
    TrendAnalysis,
    TrendAnalyzer,
    TrendPoint,
    run_improvement_radar,
)


class TestFailureClusterer:
    """Test FailureClusterer class."""

    @pytest.fixture
    def sample_failures(self):
        return [
            {
                "query": "제3조 내용",
                "passed": False,
                "overall_score": 0.3,
                "accuracy": 0.4,
                "completeness": 0.3,
                "citations": 0.2,
                "context_relevance": 0.3,
                "reasoning": "인용 불일치 발생",
                "issues": ["잘못된 출처 인용"],
            },
            {
                "query": "휴학 절차",
                "passed": False,
                "overall_score": 0.4,
                "accuracy": 0.5,
                "completeness": 0.3,
                "citations": 0.8,
                "context_relevance": 0.4,
                "reasoning": "불완전한 답변",
                "issues": ["일부만 답변됨"],
            },
            {
                "query": "장학금 조건",
                "passed": False,
                "overall_score": 0.2,
                "accuracy": 0.3,
                "completeness": 0.2,
                "citations": 0.1,
                "context_relevance": 0.2,
                "reasoning": "인용 불일치, 없는 조항 인용",
                "issues": [],
            },
            {
                "query": "졸업 요건",
                "passed": True,
                "overall_score": 0.9,
            },
        ]

    def test_cluster_failures(self, sample_failures):
        clusterer = FailureClusterer()
        clusters = clusterer.cluster(sample_failures)

        # Should only cluster failed results (3 of 4)
        total_count = sum(c.count for c in clusters)
        assert total_count == 3

    def test_cluster_empty(self):
        clusterer = FailureClusterer()
        clusters = clusterer.cluster([])
        assert clusters == []

    def test_cluster_all_passing(self):
        clusterer = FailureClusterer()
        results = [{"passed": True, "overall_score": 0.9}]
        clusters = clusterer.cluster(results)
        assert clusters == []

    def test_classify_citation_hallucination(self):
        clusterer = FailureClusterer()
        result = {
            "passed": False,
            "overall_score": 0.3,
            "reasoning": "인용 불일치가 발생했습니다",
            "issues": [],
        }
        category = clusterer._classify(result)
        assert category == FailureCategory.CITATION_HALLUCINATION

    def test_classify_by_low_score(self):
        clusterer = FailureClusterer()
        # Low citation score should map to citation_hallucination
        result = {
            "passed": False,
            "overall_score": 0.3,
            "citations": 0.2,
            "accuracy": 0.8,
            "completeness": 0.8,
            "context_relevance": 0.8,
            "reasoning": "",
            "issues": [],
        }
        category = clusterer._classify(result)
        assert category == FailureCategory.CITATION_HALLUCINATION

    def test_classify_retrieval_miss(self):
        clusterer = FailureClusterer()
        result = {
            "passed": False,
            "overall_score": 0.3,
            "citations": 0.8,
            "accuracy": 0.8,
            "completeness": 0.8,
            "context_relevance": 0.2,
            "reasoning": "",
            "issues": [],
        }
        category = clusterer._classify(result)
        assert category == FailureCategory.RETRIEVAL_MISS

    def test_cluster_sorted_by_count(self, sample_failures):
        clusterer = FailureClusterer()
        clusters = clusterer.cluster(sample_failures)
        counts = [c.count for c in clusters]
        assert counts == sorted(counts, reverse=True)

    def test_cluster_to_dict(self):
        cluster = FailureCluster(category="test", count=3, avg_score=0.5)
        d = cluster.to_dict()
        assert d["category"] == "test"
        assert d["count"] == 3


class TestRoadmapGenerator:
    """Test RoadmapGenerator class."""

    def test_generate_from_clusters(self):
        clusters = [
            FailureCluster(
                category=FailureCategory.RETRIEVAL_MISS,
                count=10,
                avg_score=0.3,
            ),
            FailureCluster(
                category=FailureCategory.FORMAT_ERROR,
                count=5,
                avg_score=0.6,
            ),
        ]
        generator = RoadmapGenerator()
        roadmap = generator.generate(clusters, total_queries=50)

        assert len(roadmap) == 2
        # Should be sorted by priority desc
        assert roadmap[0].priority_score >= roadmap[1].priority_score

    def test_generate_empty(self):
        generator = RoadmapGenerator()
        roadmap = generator.generate([], total_queries=10)
        assert roadmap == []

    def test_roadmap_item_fields(self):
        clusters = [
            FailureCluster(
                category=FailureCategory.CITATION_HALLUCINATION,
                count=5,
                avg_score=0.4,
            ),
        ]
        generator = RoadmapGenerator()
        roadmap = generator.generate(clusters)

        item = roadmap[0]
        assert item.frequency == 5
        assert item.impact > 0
        assert item.fixability > 0
        assert item.priority_score == pytest.approx(
            item.frequency * item.impact * item.fixability
        )

    def test_roadmap_item_to_dict(self):
        item = RoadmapItem(
            title="Fix citations",
            category="test",
            priority_score=5.0,
            frequency=10,
            impact=0.5,
            fixability=0.7,
        )
        d = item.to_dict()
        assert d["title"] == "Fix citations"
        assert d["priority_score"] == 5.0


class TestTrendAnalyzer:
    """Test TrendAnalyzer class."""

    @pytest.fixture
    def eval_dir(self, tmp_path):
        """Create sample evaluation history files."""
        evals = tmp_path / "evaluations"
        evals.mkdir()

        for i, (score, rate) in enumerate([
            (0.7, 0.6),
            (0.72, 0.65),
            (0.75, 0.7),
            (0.78, 0.75),
            (0.8, 0.8),
            (0.82, 0.85),
        ]):
            data = {
                "timestamp": f"2025-01-0{i+1}",
                "summary": {
                    "avg_overall_score": score,
                    "pass_rate": rate,
                    "total_queries": 30,
                },
            }
            (evals / f"eval_2025010{i+1}.json").write_text(
                json.dumps(data), encoding="utf-8"
            )
        return evals

    def test_analyze_improving_trend(self, eval_dir):
        analyzer = TrendAnalyzer(evaluations_dir=str(eval_dir))
        analysis = analyzer.analyze()

        assert len(analysis.data_points) == 6
        assert analysis.avg_score_trend == "improving"
        assert analysis.pass_rate_trend == "improving"

    def test_analyze_empty_dir(self, tmp_path):
        analyzer = TrendAnalyzer(evaluations_dir=str(tmp_path / "empty"))
        analysis = analyzer.analyze()
        assert len(analysis.data_points) == 0
        assert analysis.avg_score_trend == "stable"

    def test_analyze_single_point(self, tmp_path):
        evals = tmp_path / "evaluations"
        evals.mkdir()
        data = {
            "timestamp": "2025-01-01",
            "summary": {"avg_overall_score": 0.8, "pass_rate": 0.7, "total_queries": 10},
        }
        (evals / "eval_20250101.json").write_text(json.dumps(data), encoding="utf-8")

        analyzer = TrendAnalyzer(evaluations_dir=str(evals))
        analysis = analyzer.analyze()
        assert len(analysis.data_points) == 1
        assert analysis.moving_avg_score == pytest.approx(0.8)

    def test_classify_trend(self):
        analyzer = TrendAnalyzer()
        assert analyzer._classify_trend(0.05) == "improving"
        assert analyzer._classify_trend(-0.05) == "degrading"
        assert analyzer._classify_trend(0.01) == "stable"

    def test_trend_to_dict(self):
        analysis = TrendAnalysis(
            avg_score_trend="improving",
            pass_rate_trend="stable",
            moving_avg_score=0.8,
        )
        d = analysis.to_dict()
        assert d["avg_score_trend"] == "improving"

    def test_trend_point_to_dict(self):
        point = TrendPoint(
            timestamp="2025-01-01",
            avg_score=0.8,
            pass_rate=0.7,
            total_queries=30,
        )
        d = point.to_dict()
        assert d["timestamp"] == "2025-01-01"


class TestRunImprovementRadar:
    """Test run_improvement_radar integration."""

    def test_with_failures(self, tmp_path):
        results = [
            {
                "query": "test",
                "passed": False,
                "overall_score": 0.3,
                "accuracy": 0.3,
                "completeness": 0.3,
                "citations": 0.2,
                "context_relevance": 0.3,
                "reasoning": "",
                "issues": [],
            },
            {
                "query": "test2",
                "passed": True,
                "overall_score": 0.9,
            },
        ]
        report = run_improvement_radar(results, evaluations_dir=str(tmp_path))
        assert len(report.failure_clusters) > 0
        assert len(report.roadmap) > 0

    def test_all_passing_never_nothing_to_improve(self, tmp_path):
        """EARS-AC-NF-4: Never reports 'nothing to improve'."""
        results = [
            {"query": "test", "passed": True, "overall_score": 0.9},
        ]
        report = run_improvement_radar(results, evaluations_dir=str(tmp_path))
        assert report.never_nothing_to_improve != ""
        assert "escalate" in report.never_nothing_to_improve.lower()

    def test_report_to_dict(self, tmp_path):
        results = [
            {
                "query": "test",
                "passed": False,
                "overall_score": 0.3,
                "citations": 0.2,
                "accuracy": 0.8,
                "completeness": 0.8,
                "context_relevance": 0.8,
                "reasoning": "",
                "issues": [],
            },
        ]
        report = run_improvement_radar(results, evaluations_dir=str(tmp_path))
        d = report.to_dict()
        assert "failure_clusters" in d
        assert "roadmap" in d
