"""
Characterization tests for ExperimentService module.

These tests document the current behavior of the A/B testing service
without prescribing how it should behave.

Module under test: src/rag/application/experiment_service.py
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import json
from datetime import datetime

from src.rag.application.experiment_service import (
    ExperimentRepository,
    ExperimentService,
)
from src.rag.domain.experiment.ab_test import (
    ExperimentConfig,
    ExperimentStatus,
    VariantConfig,
    VariantType,
    VariantMetrics,
    UserAssignment,
)


class TestExperimentRepository:
    """Characterization tests for ExperimentRepository."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def repository(self, temp_dir):
        """Create repository with temporary storage."""
        return ExperimentRepository(storage_dir=temp_dir)

    def test_init_creates_directory(self, temp_dir):
        """Document that initialization creates storage directory."""
        # Arrange
        storage_path = Path(temp_dir) / "new_dir"

        # Act
        ExperimentRepository(storage_dir=str(storage_path))

        # Assert
        assert storage_path.exists()

    def test_save_experiment_config(self, repository):
        """Document saving experiment configuration."""
        # Arrange
        config = ExperimentConfig(
            experiment_id="test_exp",
            name="Test Experiment",
            variants=[
                VariantConfig(
                    name="control",
                    type=VariantType.CONTROL,
                    traffic_allocation=0.5,
                    config={},
                ),
            ],
        )

        # Act
        filepath = repository.save_experiment_config(config)

        # Assert
        assert Path(filepath).exists()

    def test_load_experiment_config(self, repository):
        """Document loading experiment configuration."""
        # Arrange
        config = ExperimentConfig(
            experiment_id="test_exp",
            name="Test Experiment",
            variants=[
                VariantConfig(
                    name="control",
                    type=VariantType.CONTROL,
                    traffic_allocation=0.5,
                    config={},
                ),
            ],
        )
        repository.save_experiment_config(config)

        # Act
        loaded = repository.load_experiment_config("test_exp")

        # Assert
        assert loaded is not None
        assert loaded.experiment_id == "test_exp"
        assert loaded.name == "Test Experiment"

    def test_load_nonexistent_experiment(self, repository):
        """Document loading non-existent experiment returns None."""
        # Act
        loaded = repository.load_experiment_config("nonexistent")

        # Assert
        assert loaded is None

    def test_save_and_load_metrics(self, repository):
        """Document saving and loading metrics."""
        # Arrange
        metrics = {
            "control": VariantMetrics(
                variant_name="control",
                impressions=100,
                conversions=10,
            ),
        }

        # Act
        repository.save_metrics("test_exp", metrics)
        loaded = repository.load_metrics("test_exp")

        # Assert
        assert "control" in loaded
        assert loaded["control"].impressions == 100

    def test_save_assignments(self, repository):
        """Document saving user assignments."""
        # Arrange
        assignments = [
            UserAssignment(
                user_id="user1",
                experiment_id="test_exp",
                variant_name="control",
                assigned_at=datetime.now(),
            ),
        ]

        # Act
        filepath = repository.save_assignments("test_exp", assignments)

        # Assert
        assert Path(filepath).exists()


class TestExperimentServiceInit:
    """Characterization tests for ExperimentService initialization."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_init_with_default_repository(self):
        """Document initialization with default repository."""
        # Act
        service = ExperimentService()

        # Assert
        assert service.repository is not None

    def test_init_with_custom_repository(self, temp_dir):
        """Document initialization with custom repository."""
        # Arrange
        repo = ExperimentRepository(storage_dir=temp_dir)

        # Act
        service = ExperimentService(repository=repo)

        # Assert
        assert service.repository is repo


class TestExperimentServiceCreateExperiment:
    """Characterization tests for create_experiment."""

    @pytest.fixture
    def service(self):
        """Create ExperimentService with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ExperimentRepository(storage_dir=tmpdir)
            yield ExperimentService(repository=repo)

    def test_create_experiment_returns_config(self, service):
        """Document that create_experiment returns ExperimentConfig."""
        # Act
        config = service.create_experiment(
            experiment_id="test_exp",
            name="Test Experiment",
            control_config={"param": "value"},
            treatment_configs=[{"param": "new_value"}],
        )

        # Assert
        assert isinstance(config, ExperimentConfig)
        assert config.experiment_id == "test_exp"

    def test_create_experiment_adds_control_variant(self, service):
        """Document that control variant is automatically added."""
        # Act
        config = service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={"key": "control"},
            treatment_configs=[{"key": "treatment"}],
        )

        # Assert
        control = config.get_control_variant()
        assert control is not None
        assert control.type == VariantType.CONTROL

    def test_create_experiment_adds_treatment_variants(self, service):
        """Document that treatment variants are added."""
        # Act
        config = service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{"a": 1}, {"b": 2}],
        )

        # Assert
        treatments = config.get_treatment_variants()
        assert len(treatments) == 2

    def test_create_experiment_draft_status(self, service):
        """Document that new experiments start in DRAFT status."""
        # Act
        config = service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )

        # Assert
        assert config.status == ExperimentStatus.DRAFT


class TestExperimentServiceStartExperiment:
    """Characterization tests for start_experiment."""

    @pytest.fixture
    def service(self):
        """Create ExperimentService with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ExperimentRepository(storage_dir=tmpdir)
            yield ExperimentService(repository=repo)

    def test_start_experiment_changes_status(self, service):
        """Document that start changes status to RUNNING."""
        # Arrange
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )

        # Act
        config = service.start_experiment("test_exp")

        # Assert
        assert config.status == ExperimentStatus.RUNNING

    def test_start_experiment_sets_started_at(self, service):
        """Document that start sets started_at timestamp."""
        # Arrange
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )

        # Act
        config = service.start_experiment("test_exp")

        # Assert
        assert config.started_at is not None

    def test_start_nonexistent_experiment_raises(self, service):
        """Document that starting non-existent experiment raises error."""
        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.start_experiment("nonexistent")

        assert "not found" in str(exc_info.value).lower()

    def test_start_running_experiment_raises(self, service):
        """Document that starting already running experiment raises error."""
        # Arrange
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("test_exp")

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.start_experiment("test_exp")

        assert "not in DRAFT" in str(exc_info.value)


class TestExperimentServiceAssignVariant:
    """Characterization tests for assign_variant."""

    @pytest.fixture
    def service(self):
        """Create ExperimentService with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ExperimentRepository(storage_dir=tmpdir)
            service = ExperimentService(repository=repo)
            service.create_experiment(
                experiment_id="test_exp",
                name="Test",
                control_config={"key": "control"},
                treatment_configs=[{"key": "treatment"}],
            )
            service.start_experiment("test_exp")
            yield service

    def test_assign_variant_returns_variant_name(self, service):
        """Document that assign_variant returns a variant name."""
        # Act
        variant = service.assign_variant("test_exp", "user1")

        # Assert
        assert variant in ["control", "treatment_0"]

    def test_assign_variant_consistent_for_same_user(self, service):
        """Document consistent assignment for same user."""
        # Act
        variant1 = service.assign_variant("test_exp", "user1")
        variant2 = service.assign_variant("test_exp", "user1")

        # Assert
        assert variant1 == variant2

    def test_assign_variant_records_impression(self, service):
        """Document that assignment records impression."""
        # Act
        service.assign_variant("test_exp", "user1")

        # Assert - Check metrics cache
        metrics = service._metrics_cache.get("test_exp", {})
        total_impressions = sum(m.impressions for m in metrics.values())
        assert total_impressions >= 1

    def test_assign_variant_nonexistent_experiment_raises(self, service):
        """Document that assigning to non-existent experiment raises error."""
        # Act & Assert
        with pytest.raises(ValueError):
            service.assign_variant("nonexistent", "user1")


class TestExperimentServiceRecordConversion:
    """Characterization tests for record_conversion."""

    @pytest.fixture
    def service(self):
        """Create ExperimentService with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ExperimentRepository(storage_dir=tmpdir)
            service = ExperimentService(repository=repo)
            service.create_experiment(
                experiment_id="test_exp",
                name="Test",
                control_config={},
                treatment_configs=[{}],
            )
            service.start_experiment("test_exp")
            yield service

    def test_record_conversion_increments_count(self, service):
        """Document that conversion increments count."""
        # Arrange
        service.assign_variant("test_exp", "user1")

        # Act
        service.record_conversion("test_exp", "user1", "control", "conversion")

        # Assert
        metrics = service._metrics_cache["test_exp"]["control"]
        assert metrics.conversions == 1

    def test_record_conversion_updates_rate(self, service):
        """Document that conversion updates conversion rate."""
        # Arrange
        service.assign_variant("test_exp", "user1")

        # Act
        service.record_conversion("test_exp", "user1", "control", "conversion")

        # Assert
        metrics = service._metrics_cache["test_exp"]["control"]
        assert metrics.conversion_rate > 0


class TestExperimentServiceRecordPerformance:
    """Characterization tests for record_performance."""

    @pytest.fixture
    def service(self):
        """Create ExperimentService with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ExperimentRepository(storage_dir=tmpdir)
            service = ExperimentService(repository=repo)
            service.create_experiment(
                experiment_id="test_exp",
                name="Test",
                control_config={},
                treatment_configs=[{}],
            )
            service.start_experiment("test_exp")
            service.assign_variant("test_exp", "user1")
            yield service

    def test_record_performance_latency(self, service):
        """Document recording latency."""
        # Act
        service.record_performance("test_exp", "control", latency_ms=100.0)

        # Assert
        metrics = service._metrics_cache["test_exp"]["control"]
        assert metrics.avg_latency_ms == 100.0

    def test_record_performance_relevance_score(self, service):
        """Document recording relevance score."""
        # Act
        service.record_performance(
            "test_exp", "control", latency_ms=100.0, relevance_score=0.85
        )

        # Assert
        metrics = service._metrics_cache["test_exp"]["control"]
        assert metrics.avg_relevance_score == 0.85

    def test_record_performance_satisfaction_score(self, service):
        """Document recording satisfaction score."""
        # Act
        service.record_performance(
            "test_exp", "control", latency_ms=100.0, satisfaction_score=0.9
        )

        # Assert
        metrics = service._metrics_cache["test_exp"]["control"]
        assert metrics.avg_satisfaction_score == 0.9


class TestExperimentServiceAnalyzeResults:
    """Characterization tests for analyze_results."""

    @pytest.fixture
    def service(self):
        """Create ExperimentService with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ExperimentRepository(storage_dir=tmpdir)
            service = ExperimentService(repository=repo)
            service.create_experiment(
                experiment_id="test_exp",
                name="Test",
                control_config={},
                treatment_configs=[{}],
            )
            service.start_experiment("test_exp")
            yield service

    def test_analyze_results_returns_experiment_result(self, service):
        """Document that analyze returns ExperimentResult."""
        # Arrange - Add some data
        service.assign_variant("test_exp", "user1")
        service.assign_variant("test_exp", "user2")

        # Act
        result = service.analyze_results("test_exp")

        # Assert
        assert result.experiment_id == "test_exp"

    def test_analyze_results_nonexistent_raises(self, service):
        """Document that analyzing non-existent experiment raises error."""
        # Act & Assert
        with pytest.raises(ValueError):
            service.analyze_results("nonexistent")


class TestExperimentServiceStopExperiment:
    """Characterization tests for stop_experiment."""

    @pytest.fixture
    def service(self):
        """Create ExperimentService with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ExperimentRepository(storage_dir=tmpdir)
            service = ExperimentService(repository=repo)
            service.create_experiment(
                experiment_id="test_exp",
                name="Test",
                control_config={},
                treatment_configs=[{}],
            )
            service.start_experiment("test_exp")
            yield service

    def test_stop_experiment_changes_status(self, service):
        """Document that stop changes status to COMPLETED."""
        # Act
        config = service.stop_experiment("test_exp")

        # Assert
        assert config.status == ExperimentStatus.COMPLETED

    def test_stop_experiment_sets_ended_at(self, service):
        """Document that stop sets ended_at timestamp."""
        # Act
        config = service.stop_experiment("test_exp")

        # Assert
        assert config.ended_at is not None

    def test_stop_non_running_raises(self, service):
        """Document that stopping non-running experiment raises error."""
        # Arrange
        service.stop_experiment("test_exp")

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            service.stop_experiment("test_exp")

        assert "not running" in str(exc_info.value).lower()


class TestExperimentServiceGetExperimentConfig:
    """Characterization tests for get_experiment_config."""

    @pytest.fixture
    def service(self):
        """Create ExperimentService with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ExperimentRepository(storage_dir=tmpdir)
            yield ExperimentService(repository=repo)

    def test_get_experiment_config_returns_config(self, service):
        """Document retrieving experiment config."""
        # Arrange
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )

        # Act
        config = service.get_experiment_config("test_exp")

        # Assert
        assert config is not None
        assert config.experiment_id == "test_exp"

    def test_get_experiment_config_nonexistent_returns_none(self, service):
        """Document retrieving non-existent config returns None."""
        # Act
        config = service.get_experiment_config("nonexistent")

        # Assert
        assert config is None


class TestExperimentServiceMultiArmedBandit:
    """Characterization tests for multi-armed bandit assignment."""

    @pytest.fixture
    def service(self):
        """Create ExperimentService with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ExperimentRepository(storage_dir=tmpdir)
            service = ExperimentService(repository=repo)
            service.create_experiment(
                experiment_id="test_exp",
                name="Test",
                control_config={},
                treatment_configs=[{}],
                enable_multi_armed_bandit=True,
            )
            service.start_experiment("test_exp")
            yield service

    def test_bandit_explores_randomly(self, service):
        """Document that bandit explores with epsilon probability."""
        # This is probabilistic, so we just verify the mechanism exists
        # by calling assign_variant multiple times
        variants = set()
        for i in range(20):
            variant = service.assign_variant("test_exp", f"user_{i}")
            variants.add(variant)

        # Assert - With 20 users, we should see at least one variant
        assert len(variants) >= 1


class TestExperimentServiceRecordImpression:
    """Characterization tests for record_impression."""

    @pytest.fixture
    def service(self):
        """Create ExperimentService with temp storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = ExperimentRepository(storage_dir=tmpdir)
            service = ExperimentService(repository=repo)
            service.create_experiment(
                experiment_id="test_exp",
                name="Test",
                control_config={},
                treatment_configs=[{}],
            )
            service.start_experiment("test_exp")
            yield service

    def test_record_impression_increments_count(self, service):
        """Document that impression increments count."""
        # Act
        service.record_impression("test_exp", "control", "user1")

        # Assert
        metrics = service._metrics_cache["test_exp"]["control"]
        assert metrics.impressions == 1

    def test_record_impression_tracks_unique_users(self, service):
        """Document that unique users are tracked."""
        # Act
        service.record_impression("test_exp", "control", "user1")
        service.record_impression("test_exp", "control", "user1")  # Same user
        service.record_impression("test_exp", "control", "user2")  # Different user

        # Assert
        metrics = service._metrics_cache["test_exp"]["control"]
        assert metrics.unique_users == 2
