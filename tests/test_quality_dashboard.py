"""
Tests for Quality Dashboard components.

Tests the Gradio dashboard and FastAPI routes for quality metrics.
"""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_metrics_file():
    """Create a temporary metrics file for testing."""
    metrics = {
        "timestamp": "2025-01-29T12:00:00",
        "faithfulness": {"score": 0.85, "trend": "improve", "change": 0.05},
        "answer_relevancy": {"score": 0.78, "trend": "maintain", "change": 0.01},
        "precision": {"score": 0.82, "trend": "improve", "change": 0.03},
        "recall": {"score": 0.75, "trend": "decrease", "change": -0.02},
        "personas": {
            "faculty": {"score": 0.88, "count": 45},
            "student": {"score": 0.76, "count": 120},
            "staff": {"score": 0.82, "count": 35},
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(metrics, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_history_file():
    """Create a temporary history file for testing."""
    from datetime import datetime, timedelta

    history = []
    base_date = datetime.now() - timedelta(days=30)

    for i in range(30):
        date = base_date + timedelta(days=i)
        history.append(
            {
                "timestamp": date.isoformat(),
                "faithfulness": 0.80 + (i * 0.001),
                "answer_relevancy": 0.75 + (i * 0.0015),
                "precision": 0.78 + (i * 0.0012),
                "recall": 0.72 + (i * 0.0008),
            }
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for record in history:
            f.write(json.dumps(record) + "\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestQualityDashboard:
    """Tests for the quality dashboard Gradio interface."""

    def test_load_metrics(self, temp_metrics_file):
        """Test loading metrics from file."""
        from src.rag.interface.web.quality_dashboard import _load_metrics

        metrics = _load_metrics(temp_metrics_file)

        assert "faithfulness" in metrics
        assert metrics["faithfulness"]["score"] == 0.85
        assert metrics["faithfulness"]["trend"] == "improve"

    def test_load_history(self, temp_history_file):
        """Test loading history from file."""
        from src.rag.interface.web.quality_dashboard import _load_history

        history = _load_history(temp_history_file, days=30)

        assert len(history) == 30
        assert "faithfulness" in history[0]
        assert "timestamp" in history[0]

    def test_create_metric_card(self):
        """Test metric card HTML generation."""
        from src.rag.interface.web.quality_dashboard import _create_metric_card

        card_html = _create_metric_card("Test", 0.85, "improve", 0.05)

        assert "Test" in card_html
        assert "85%" in card_html
        assert "📈" in card_html

    def test_format_trend_indicator(self):
        """Test trend indicator formatting."""
        from src.rag.interface.web.quality_dashboard import _format_trend_indicator

        assert "📈" in _format_trend_indicator("improve", 0.05)
        assert "📉" in _format_trend_indicator("decrease", -0.02)
        assert "➡️" in _format_trend_indicator("maintain", 0.0)

    def test_create_persona_table(self):
        """Test persona table HTML generation."""
        from src.rag.interface.web.quality_dashboard import _create_persona_table

        metrics = {
            "personas": {
                "faculty": {"score": 0.88, "count": 45},
                "student": {"score": 0.76, "count": 120},
            }
        }

        table_html = _create_persona_table(metrics)

        assert "교수" in table_html
        assert "학생" in table_html
        assert "88%" in table_html
        assert "76%" in table_html


class TestQualityRoutes:
    """Tests for the FastAPI quality routes."""

    @pytest.fixture
    def client(self, temp_metrics_file, temp_history_file):
        """Create a test FastAPI client."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        # Import router
        from src.rag.interface.web.routes.quality import router

        app = FastAPI()
        app.include_router(router)

        return TestClient(app)

    def test_get_latest_metrics(self, client, temp_metrics_file):
        """Test GET /quality/metrics/latest endpoint."""
        response = client.get(
            f"/quality/metrics/latest?metrics_path={temp_metrics_file}"
        )

        assert response.status_code == 200
        data = response.json()
        assert "faithfulness" in data
        assert data["faithfulness"]["score"] == 0.85

    def test_get_metrics_history(self, client, temp_history_file):
        """Test GET /quality/metrics/history endpoint."""
        response = client.get(
            f"/quality/metrics/history?history_path={temp_history_file}&days=30"
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 30

    def test_get_persona_metrics(self, client, temp_metrics_file):
        """Test GET /quality/metrics/personas endpoint."""
        response = client.get(
            f"/quality/metrics/personas?metrics_path={temp_metrics_file}"
        )

        assert response.status_code == 200
        data = response.json()
        assert "personas" in data
        assert "faculty" in data["personas"]
        assert data["personas"]["faculty"]["score"] == 0.88

    def test_health_check(self, client):
        """Test GET /quality/health endpoint."""
        response = client.get("/quality/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data

    def test_record_metrics(self, client, temp_metrics_file, temp_history_file):
        """Test POST /quality/metrics/record endpoint."""
        response = client.post(
            f"/quality/metrics/record?metrics_path={temp_metrics_file}&history_path={temp_history_file}",
            params={
                "faithfulness": 0.90,
                "answer_relevancy": 0.85,
                "precision": 0.88,
                "recall": 0.80,
                "persona": "faculty",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "metrics" in data


class TestPDFReportGeneration:
    """Tests for PDF report generation."""

    @pytest.fixture
    def reportlab_available(self):
        """Check if reportlab is available."""
        try:
            import reportlab  # noqa: F401

            return True
        except ImportError:
            return False

    def test_generate_pdf_report(
        self, temp_metrics_file, temp_history_file, reportlab_available
    ):
        """Test PDF report generation."""
        from src.rag.interface.web.quality_dashboard import generate_pdf_report

        if not reportlab_available:
            pytest.skip("reportlab not installed")

        result = generate_pdf_report(temp_metrics_file, temp_history_file)

        assert "✅" in result or "❌" in result
        if "✅" in result:
            assert "quality_report_" in result


@pytest.mark.integration
class TestDashboardIntegration:
    """Integration tests for the full dashboard."""

    def test_dashboard_creation(self):
        """Test that the dashboard can be created."""
        from src.rag.interface.web.quality_dashboard import create_dashboard

        dashboard = create_dashboard()

        assert dashboard is not None

    def test_dashboard_refresh(self, temp_metrics_file, temp_history_file):
        """Test dashboard refresh functionality."""
        from src.rag.interface.web.quality_dashboard import refresh_dashboard

        result = refresh_dashboard(temp_metrics_file, temp_history_file, 30)

        assert len(result) == 7  # 4 cards, chart, table, status
