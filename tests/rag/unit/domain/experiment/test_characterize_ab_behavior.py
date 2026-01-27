"""
Characterization tests for existing A/B testing behavior.

These tests document WHAT the current system does, not what it SHOULD do.
Created during PRESERVE phase to ensure behavior preservation during refactoring.
"""

import json
from datetime import datetime
from pathlib import Path

from src.rag.infrastructure.ab_test_framework import (
    ABTestManager,
    ABTestMetrics,
    ABTestRepository,
    ABTestSession,
    RerankerModelType,
    create_ab_manager,
)


class TestCharacterizeABTestMetrics:
    """Characterize ABTestMetrics behavior."""

    def test_characterize_initial_metrics_state(self):
        """Document: Metrics initialize with zero values."""
        metrics = ABTestMetrics(
            model_name="test_model",
            model_type=RerankerModelType.MULTILINGUAL,
        )

        # Characterize actual initial state
        assert metrics.total_queries == 0
        assert metrics.successful_queries == 0
        assert metrics.failed_queries == 0
        assert metrics.total_latency_ms == 0.0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.avg_relevance_score == 0.0
        assert metrics.ndcg_score == 0.0
        assert metrics.first_query_time is None
        assert metrics.last_query_time is None

    def test_characterize_to_dict_format(self):
        """Document: Serialization format includes all fields."""
        metrics = ABTestMetrics(
            model_name="test_model",
            model_type=RerankerModelType.KOREAN,
        )
        metrics.first_query_time = datetime.now()

        data = metrics.to_dict()

        # Characterize actual dictionary structure
        assert "model_name" in data
        assert "model_type" in data
        assert "total_queries" in data
        assert "successful_queries" in data
        assert "failed_queries" in data
        assert "total_latency_ms" in data
        assert "avg_latency_ms" in data
        assert "avg_relevance_score" in data
        assert "ndcg_score" in data
        assert "first_query_time" in data
        assert "last_query_time" in data


class TestCharacterizeABTestSession:
    """Characterize ABTestSession behavior."""

    def test_characterize_session_initialization(self):
        """Document: Session initializes with current time."""
        before = datetime.now()
        session = ABTestSession(session_id="test_session")
        after = datetime.now()

        # Characterize actual initialization behavior
        assert session.session_id == "test_session"
        assert before <= session.start_time <= after
        assert session.end_time is None
        assert session.test_ratio == 0.5
        assert session.model_metrics == {}

    def test_characterize_auto_model_type_detection(self):
        """Document: Automatic model type detection based on name."""
        session = ABTestSession(session_id="detection_test")

        # Test Korean model detection
        korean_metrics = session.get_metrics("Dongjin-kr/kr-reranker")
        assert korean_metrics.model_type == RerankerModelType.KOREAN

        # Test multilingual model detection
        multi_metrics = session.get_metrics("BAAI/bge-reranker-v2-m3")
        assert multi_metrics.model_type == RerankerModelType.MULTILINGUAL

    def test_characterize_query_recording_updates_metrics(self):
        """Document: Recording query updates all related metrics."""
        session = ABTestSession(session_id="recording_test")

        # Record first query
        session.record_query(
            "model_a", latency_ms=50.0, success=True, relevance_score=0.8
        )

        metrics = session.get_metrics("model_a")
        assert metrics.total_queries == 1
        assert metrics.successful_queries == 1
        assert metrics.failed_queries == 0
        assert metrics.total_latency_ms == 50.0
        assert metrics.avg_latency_ms == 50.0
        assert metrics.first_query_time is not None
        assert metrics.last_query_time is not None

        # Record second query
        session.record_query(
            "model_a", latency_ms=70.0, success=True, relevance_score=0.6
        )

        assert metrics.total_queries == 2
        assert metrics.avg_latency_ms == 60.0  # (50 + 70) / 2
        assert metrics.avg_relevance_score == 0.7  # (0.8 + 0.6) / 2

    def test_characterize_failed_query_recording(self):
        """Document: Failed queries update failure count but not relevance."""
        session = ABTestSession(session_id="failure_test")

        session.record_query("model_a", latency_ms=50.0, success=False)

        metrics = session.get_metrics("model_a")
        assert metrics.total_queries == 1
        assert metrics.successful_queries == 0
        assert metrics.failed_queries == 1
        # Relevance score not updated for failed queries
        assert metrics.avg_relevance_score == 0.0


class TestCharacterizeABTestRepository:
    """Characterize ABTestRepository behavior."""

    def test_characterize_save_creates_json_file(self, tmp_path):
        """Document: Save creates JSON file with specific naming."""
        repo = ABTestRepository(storage_dir=str(tmp_path))
        session = ABTestSession(session_id="test_save")

        saved_path = repo.save_session(session)

        # Characterize actual file creation behavior
        assert Path(saved_path).exists()
        assert saved_path.endswith("ab_test_test_save.json")

    def test_characterize_serialization_format(self, tmp_path):
        """Document: JSON serialization includes all session data."""
        repo = ABTestRepository(storage_dir=str(tmp_path))
        session = ABTestSession(session_id="format_test")
        session.record_query("model_a", latency_ms=50.0, success=True)

        repo.save_session(session)

        # Load and verify format
        with open(Path(tmp_path) / "ab_test_format_test.json", "r") as f:
            data = json.load(f)

        # Characterize actual JSON structure
        assert "session_id" in data
        assert "start_time" in data
        assert "end_time" in data
        assert "test_ratio" in data
        assert "model_metrics" in data
        assert isinstance(data["model_metrics"], dict)

    def test_characterize_load_restores_session(self, tmp_path):
        """Document: Load restores session from file."""
        repo = ABTestRepository(storage_dir=str(tmp_path))
        original = ABTestSession(session_id="load_test")
        original.record_query("model_a", latency_ms=50.0, success=True)

        repo.save_session(original)
        loaded = repo.load_session("load_test")

        # Characterize actual restoration behavior
        assert loaded is not None
        assert loaded.session_id == "load_test"
        assert "model_a" in loaded.model_metrics
        assert loaded.get_metrics("model_a").total_queries == 1

    def test_characterize_load_missing_returns_none(self, tmp_path):
        """Document: Loading non-existent session returns None."""
        repo = ABTestRepository(storage_dir=str(tmp_path))

        # Characterize actual behavior for missing files
        result = repo.load_session("nonexistent")
        assert result is None


class TestCharacterizeABTestManager:
    """Characterize ABTestManager behavior."""

    def test_characterize_initialization_creates_session(self):
        """Document: Manager creates session with timestamp ID."""
        manager = ABTestManager(
            control_model="control",
            test_models=["test_a", "test_b"],
            test_ratio=0.5,
        )

        # Characterize actual initialization
        assert manager.control_model == "control"
        assert manager.test_models == ["test_a", "test_b"]
        assert manager.test_ratio == 0.5
        assert manager.session_id is not None
        assert len(manager.session_id.split("_")) == 2  # YYYYMMDD_HHMMSS format

    def test_characterize_model_selection_ratio(self):
        """Document: Model selection follows configured ratio."""
        manager = ABTestManager(
            control_model="control",
            test_models=["test"],
            test_ratio=0.5,
        )

        # Characterize actual selection behavior with 100 iterations
        selections = {"control": 0, "test": 0}
        for _ in range(100):
            selected = manager.select_model()
            selections[selected] += 1

        # With 0.5 ratio and 100 iterations, expect approximately 50/50 split
        # Allow some variance due to randomness
        assert 40 <= selections["test"] <= 60
        assert 40 <= selections["control"] <= 60

    def test_characterize_record_result_updates_session(self):
        """Document: Recording result updates session metrics."""
        manager = ABTestManager(
            control_model="control",
            test_models=["test"],
            test_ratio=0.5,
        )

        manager.record_result("control", latency_ms=50.0, success=True)

        # Characterize actual update behavior
        metrics = manager.session.get_metrics("control")
        assert metrics.total_queries == 1
        assert metrics.avg_latency_ms == 50.0

    def test_characterize_summary_structure(self):
        """Document: Summary contains specific structure."""
        manager = ABTestManager(
            control_model="control",
            test_models=["test"],
            test_ratio=0.5,
        )
        manager.record_result(
            "control", latency_ms=50.0, success=True, relevance_score=0.7
        )
        manager.record_result(
            "test", latency_ms=40.0, success=True, relevance_score=0.8
        )

        summary = manager.get_summary()

        # Characterize actual summary structure
        assert "session_id" in summary
        assert "test_ratio" in summary
        assert "models" in summary
        assert "control" in summary["models"]
        assert "test" in summary["models"]
        # Comparison data
        assert "test_vs_control" in summary
        assert "latency_improvement_percent" in summary["test_vs_control"]
        assert "relevance_improvement_percent" in summary["test_vs_control"]
        assert "recommendation" in summary["test_vs_control"]

    def test_characterize_recommendation_thresholds(self):
        """Document: Recommendation logic thresholds."""
        manager = ABTestManager(
            control_model="control",
            test_models=["test"],
            test_ratio=0.5,
        )

        # Test ADOPT threshold: relevance > 10% improvement, latency not much worse
        manager.session = ABTestSession(session_id="adopt_test")
        manager.record_result(
            "control", latency_ms=100.0, success=True, relevance_score=0.7
        )
        manager.record_result(
            "test", latency_ms=90.0, success=True, relevance_score=0.85
        )

        summary = manager.get_summary()
        assert "ADOPT" in summary["test_vs_control"]["recommendation"]

    def test_characterize_ratio_clamping(self):
        """Document: Test ratio is clamped to [0.0, 1.0]."""
        # Test upper bound
        manager_high = ABTestManager(
            control_model="control",
            test_models=["test"],
            test_ratio=1.5,
        )
        assert manager_high.test_ratio == 1.0

        # Test lower bound
        manager_low = ABTestManager(
            control_model="control",
            test_models=["test"],
            test_ratio=-0.5,
        )
        assert manager_low.test_ratio == 0.0


class TestCharacterizeCreateABManager:
    """Characterize create_ab_manager factory function."""

    def test_characterize_default_models(self):
        """Document: Factory uses default Korean models if none specified."""
        manager = create_ab_manager()

        # Characterize actual default behavior
        assert manager.control_model == "BAAI/bge-reranker-v2-m3"
        assert manager.test_models == ["Dongjin-kr/kr-reranker", "NLPai/ko-reranker"]
        assert manager.test_ratio == 0.5

    def test_characterize_custom_parameters(self):
        """Document: Factory accepts custom parameters."""
        manager = create_ab_manager(
            control_model="custom_control",
            test_models=["custom_test"],
            test_ratio=0.3,
        )

        # Characterize actual parameter application
        assert manager.control_model == "custom_control"
        assert manager.test_models == ["custom_test"]
        assert manager.test_ratio == 0.3
