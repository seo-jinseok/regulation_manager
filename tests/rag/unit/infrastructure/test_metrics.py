"""
Unit tests for Reranking Metrics Storage and Reporting System (Cycle 3).

Tests cover:
- MetricsRepository save/load operations
- Time-based aggregation
- Old metrics cleanup
- MetricsReporter HTML generation
- MetricsReporter CSV export
- Session comparison
"""

import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

from src.rag.domain.entities import RerankingMetrics
from src.rag.infrastructure.metrics import MetricsRepository, MetricsReporter


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> str:
    """Create temporary storage directory for tests."""
    storage_dir = tmp_path / ".metrics" / "reranking"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return str(storage_dir)


@pytest.fixture
def sample_metrics() -> RerankingMetrics:
    """Create sample RerankingMetrics for testing."""
    metrics = RerankingMetrics()
    metrics.total_queries = 100
    metrics.reranker_applied = 60
    metrics.reranker_skipped = 40
    metrics.article_reference_skips = 15
    metrics.regulation_name_skips = 10
    metrics.short_simple_skips = 10
    metrics.no_intent_skips = 5
    metrics.natural_question_applies = 30
    metrics.intent_applies = 20
    metrics.complex_applies = 10
    metrics.total_reranker_time_ms = 6000.0
    metrics.total_skip_saved_time_ms = 4000.0
    return metrics


class TestMetricsRepository:
    """Test MetricsRepository functionality."""

    def test_init_creates_storage_dir(self, temp_storage_dir: str):
        """저장소 초기화 시 디렉터리가 생성되는지 확인"""
        repo = MetricsRepository(storage_dir=temp_storage_dir)
        assert repo.storage_dir.exists()
        assert repo.storage_dir.is_dir()

    def test_save_metrics_creates_json_file(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """메트릭 저장 시 JSON 파일이 생성되는지 확인"""
        repo = MetricsRepository(storage_dir=temp_storage_dir)
        filepath = repo.save_metrics(sample_metrics, session_id="test_session")

        assert Path(filepath).exists()
        assert filepath.endswith(".json")

    def test_save_metrics_includes_session_id(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """저장된 메트릭에 session_id가 포함되는지 확인"""
        repo = MetricsRepository(storage_dir=temp_storage_dir)
        filepath = repo.save_metrics(sample_metrics, session_id="test_123")

        data = repo.load_metrics(filepath)
        assert data["session_id"] == "test_123"

    def test_load_metrics_restores_data(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """저장된 메트릭을 로드하면 데이터가 복원되는지 확인"""
        repo = MetricsRepository(storage_dir=temp_storage_dir)
        filepath = repo.save_metrics(sample_metrics)

        loaded_data = repo.load_metrics(filepath)
        assert loaded_data["metrics"]["total_queries"] == 100
        assert loaded_data["metrics"]["reranker_applied"] == 60
        assert loaded_data["metrics"]["reranker_skipped"] == 40

    def test_load_recent_metrics_filters_by_time(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """시간 기반으로 최근 메트릭만 필터링하는지 확인"""
        repo = MetricsRepository(storage_dir=temp_storage_dir)

        # Save multiple metrics files
        for i in range(3):
            repo.save_metrics(sample_metrics, session_id=f"session_{i}")

        # Load recent metrics (should return all 3)
        recent = repo.load_recent_metrics(hours=1)
        assert len(recent) == 3

    def test_aggregate_metrics_combines_multiple(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """여러 메트릭을 하나로 집계하는지 확인"""
        repo = MetricsRepository(storage_dir=temp_storage_dir)

        # Create two metrics sessions
        metrics_list = [
            {
                "timestamp": datetime.now().isoformat(),
                "session_id": "session_1",
                "metrics": sample_metrics.to_dict(),
            },
            {
                "timestamp": datetime.now().isoformat(),
                "session_id": "session_2",
                "metrics": sample_metrics.to_dict(),
            },
        ]

        aggregated = repo.aggregate_metrics(metrics_list)

        # Values should be doubled
        assert aggregated.total_queries == 200
        assert aggregated.reranker_applied == 120
        assert aggregated.reranker_skipped == 80

    def test_aggregate_empty_metrics_returns_empty(
        self, temp_storage_dir: str
    ):
        """빈 메트릭 리스트를 집계하면 빈 결과를 반환하는지 확인"""
        repo = MetricsRepository(storage_dir=temp_storage_dir)
        aggregated = repo.aggregate_metrics([])
        assert aggregated.total_queries == 0

    def test_get_daily_summary_aggregates_recent(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """일일 요약이 최근 메트릭을 집계하는지 확인"""
        repo = MetricsRepository(storage_dir=temp_storage_dir)

        # Save multiple sessions
        for i in range(5):
            repo.save_metrics(sample_metrics, session_id=f"day_{i}")

        # Get daily summary
        daily = repo.get_daily_summary(days=1)
        assert daily.total_queries == 500  # 100 * 5

    def test_clear_old_metrics_deletes_old_files(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """오래된 메트릭 파일을 삭제하는지 확인"""
        repo = MetricsRepository(storage_dir=temp_storage_dir)

        # Create a mock old file (mocking old timestamp via filename)
        old_date = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d_%H%M%S")
        old_filename = f"reranking_metrics_{old_date}.json"
        old_filepath = repo.storage_dir / old_filename

        # Create old file with valid content
        with open(old_filepath, "w") as f:
            json.dump({
                "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
                "session_id": "old_session",
                "metrics": sample_metrics.to_dict(),
            }, f)

        # Create a recent file
        repo.save_metrics(sample_metrics, session_id="recent_session")

        # Clear old metrics (older than 7 days)
        deleted_count = repo.clear_old_metrics(days=7)

        # Should delete only the old file
        assert deleted_count == 1
        assert not old_filepath.exists()

    def test_clear_old_metrics_preserves_recent(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """최근 메트릭 파일은 보존되는지 확인"""
        repo = MetricsRepository(storage_dir=temp_storage_dir)

        # Save recent metrics
        repo.save_metrics(sample_metrics, session_id="recent")

        # Clear old metrics (older than 7 days)
        repo.clear_old_metrics(days=7)

        # Recent file should still exist
        recent_files = list(repo.storage_dir.glob("reranking_metrics_*.json"))
        assert len(recent_files) == 1


class TestMetricsReporter:
    """Test MetricsReporter functionality."""

    def test_init_creates_repository(self):
        """MetricsReporter 초기화 시 repository가 생성되는지 확인"""
        reporter = MetricsReporter()
        assert reporter.repository is not None

    def test_print_summary_outputs_to_console(
        self, sample_metrics: RerankingMetrics, capsys
    ):
        """콘솔에 요약이 출력되는지 확인"""
        reporter = MetricsReporter()
        reporter.print_summary(sample_metrics)

        captured = capsys.readouterr()
        assert "Reranking Metrics Summary" in captured.out
        assert "Total queries: 100" in captured.out

    def test_generate_html_report_creates_file(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """HTML 리포트 파일이 생성되는지 확인"""
        reporter = MetricsReporter()
        output_path = f"{temp_storage_dir}/test_report.html"

        result_path = reporter.generate_html_report(sample_metrics, output_path)

        assert Path(result_path).exists()
        assert result_path == output_path

    def test_html_report_contains_metrics_data(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """HTML 리포트에 메트릭 데이터가 포함되는지 확인"""
        reporter = MetricsReporter()
        output_path = f"{temp_storage_dir}/test_report.html"

        reporter.generate_html_report(sample_metrics, output_path)

        with open(output_path, "r") as f:
            html_content = f.read()

        assert "100" in html_content  # Total queries
        assert "60" in html_content  # Applied
        assert "40" in html_content  # Skipped

    def test_html_report_has_valid_html_structure(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """HTML 리포트가 유효한 HTML 구조를 갖는지 확인"""
        reporter = MetricsReporter()
        output_path = f"{temp_storage_dir}/test_report.html"

        reporter.generate_html_report(sample_metrics, output_path)

        with open(output_path, "r") as f:
            html_content = f.read()

        assert "<!DOCTYPE html>" in html_content
        assert "<html" in html_content
        assert "</html>" in html_content
        assert "<title>Reranking Metrics Report</title>" in html_content

    def test_export_to_csv_creates_file(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """CSV export 파일이 생성되는지 확인"""
        reporter = MetricsReporter()
        output_path = f"{temp_storage_dir}/test_metrics.csv"

        result_path = reporter.export_to_csv(sample_metrics, output_path)

        assert Path(result_path).exists()
        assert result_path == output_path

    def test_csv_export_contains_metrics(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """CSV export에 메트릭 데이터가 포함되는지 확인"""
        reporter = MetricsReporter()
        output_path = f"{temp_storage_dir}/test_metrics.csv"

        reporter.export_to_csv(sample_metrics, output_path)

        with open(output_path, "r") as f:
            csv_content = f.read()

        assert "Metric,Value" in csv_content
        assert "total_queries" in csv_content

    def test_compare_sessions_finds_best_worst(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """세션 비교 시 최고/최저 성능을 찾는지 확인"""
        reporter = MetricsReporter()

        # Create multiple session metrics with different skip rates
        session_metrics = [
            {
                "timestamp": datetime.now().isoformat(),
                "session_id": "high_skip",
                "metrics": {**sample_metrics.to_dict(), "skip_rate": 0.8, "avg_reranker_time_ms": 100},
            },
            {
                "timestamp": datetime.now().isoformat(),
                "session_id": "low_skip",
                "metrics": {**sample_metrics.to_dict(), "skip_rate": 0.2, "avg_reranker_time_ms": 200},
            },
            {
                "timestamp": datetime.now().isoformat(),
                "session_id": "most_active",
                "metrics": {**sample_metrics.to_dict(), "skip_rate": 0.5, "avg_reranker_time_ms": 150, "total_queries": 500},
            },
        ]

        comparison = reporter.compare_sessions(session_metrics)

        assert comparison["total_sessions"] == 3
        assert comparison["best_skip_rate"]["session_id"] == "high_skip"
        assert comparison["worst_skip_rate"]["session_id"] == "low_skip"
        assert comparison["fastest_avg_time"]["session_id"] == "high_skip"
        assert comparison["most_active_session"]["session_id"] == "most_active"

    def test_compare_sessions_empty_returns_empty(self):
        """빈 세션 리스트 비교 시 빈 결과를 반환하는지 확인"""
        reporter = MetricsReporter()
        comparison = reporter.compare_sessions([])
        assert comparison == {}


class TestRerankingMetricsIntegration:
    """Integration tests for metrics system."""

    def test_full_metrics_workflow(
        self, temp_storage_dir: str, sample_metrics: RerankingMetrics
    ):
        """전체 메트릭 워크플로우 테스트 (save -> load -> report)"""
        # Setup
        repo = MetricsRepository(storage_dir=temp_storage_dir)
        reporter = MetricsReporter(repository=repo)

        # Save metrics
        filepath = repo.save_metrics(sample_metrics, session_id="integration_test")
        assert Path(filepath).exists()

        # Load metrics
        loaded_data = repo.load_metrics(filepath)
        assert loaded_data["session_id"] == "integration_test"

        # Generate report
        html_path = f"{temp_storage_dir}/integration_report.html"
        reporter.generate_html_report(sample_metrics, html_path)
        assert Path(html_path).exists()

        # Export CSV
        csv_path = f"{temp_storage_dir}/integration_metrics.csv"
        reporter.export_to_csv(sample_metrics, csv_path)
        assert Path(csv_path).exists()

    def test_metrics_property_calculations(self):
        """메트릭 속성 계산이 올바른지 확인"""
        metrics = RerankingMetrics()
        metrics.total_queries = 100
        metrics.reranker_skipped = 40
        metrics.reranker_applied = 60
        metrics.total_reranker_time_ms = 6000.0

        # Test skip rate
        assert metrics.skip_rate == 0.4

        # Test apply rate
        assert metrics.apply_rate == 0.6

        # Test avg reranker time
        assert metrics.avg_reranker_time_ms == 100.0

        # Test estimated time saved
        assert metrics.estimated_time_saved_ms == 4000.0

    def test_metrics_to_dict_serialization(self, sample_metrics: RerankingMetrics):
        """메트릭이 올바르게 dict로 직렬화되는지 확인"""
        data = sample_metrics.to_dict()

        assert isinstance(data, dict)
        assert "total_queries" in data
        assert "skip_rate" in data
        assert "apply_rate" in data
        assert "avg_reranker_time_ms" in data
        assert data["total_queries"] == 100

    def test_metrics_summary_format(self, sample_metrics: RerankingMetrics, capsys):
        """메트릭 요약 형식이 올바른지 확인"""
        summary = sample_metrics.get_summary()

        assert "Reranking Metrics Summary" in summary
        assert "Total queries: 100" in summary
        assert "Reranker applied: 60" in summary
        assert "Reranker skipped: 40" in summary
        assert "Article reference: 15" in summary
        assert "Regulation name: 10" in summary
