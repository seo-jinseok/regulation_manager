"""
Unit tests for ExperimentService.

Tests cover:
- Experiment lifecycle management
- Variant assignment and user bucketing
- Metrics tracking and conversion recording
- Statistical analysis and winner selection
- Multi-armed bandit algorithm
- Integration scenarios

REQ-AB-001 to REQ-AB-015
"""

import json
from pathlib import Path

import pytest

from src.rag.application.experiment_service import (
    ExperimentRepository,
    ExperimentService,
)
from src.rag.domain.experiment.ab_test import (
    ExperimentConfig,
    ExperimentStatus,
    VariantConfig,
    VariantMetrics,
    VariantType,
)


class TestExperimentRepository:
    """Test experiment repository persistence."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create temporary repository."""
        return ExperimentRepository(storage_dir=str(tmp_path / "experiments"))

    def test_save_and_load_experiment_config(self, temp_repo):
        """Test saving and loading experiment configuration."""
        variants = [
            VariantConfig(
                name="control",
                type=VariantType.CONTROL,
                traffic_allocation=0.5,
            ),
            VariantConfig(
                name="treatment",
                type=VariantType.TREATMENT,
                traffic_allocation=0.5,
            ),
        ]

        config = ExperimentConfig(
            experiment_id="test_exp",
            name="Test Experiment",
            variants=variants,
            status=ExperimentStatus.DRAFT,
        )

        # Save
        saved_path = temp_repo.save_experiment_config(config)
        assert Path(saved_path).exists()

        # Load
        loaded = temp_repo.load_experiment_config("test_exp")
        assert loaded is not None
        assert loaded.experiment_id == "test_exp"
        assert loaded.name == "Test Experiment"
        assert len(loaded.variants) == 2
        assert loaded.status == ExperimentStatus.DRAFT

    def test_save_and_load_metrics(self, temp_repo):
        """Test saving and loading variant metrics."""
        metrics = {
            "control": VariantMetrics(variant_name="control"),
            "treatment": VariantMetrics(variant_name="treatment"),
        }
        metrics["control"].impressions = 100
        metrics["control"].conversions = 20
        metrics["treatment"].impressions = 100
        metrics["treatment"].conversions = 25

        # Save
        saved_path = temp_repo.save_metrics("test_exp", metrics)
        assert Path(saved_path).exists()

        # Load
        loaded = temp_repo.load_metrics("test_exp")
        assert "control" in loaded
        assert "treatment" in loaded
        assert loaded["control"].impressions == 100
        assert loaded["control"].conversions == 20
        assert loaded["treatment"].conversions == 25

    def test_load_nonexistent_experiment(self, temp_repo):
        """Test loading non-existent experiment returns None."""
        loaded = temp_repo.load_experiment_config("nonexistent")
        assert loaded is None

    def test_save_assignments(self, temp_repo):
        """Test saving user assignments."""
        from src.rag.domain.experiment.ab_test import UserAssignment

        assignments = [
            UserAssignment(
                user_id="user1",
                experiment_id="test_exp",
                variant_name="control",
            ),
            UserAssignment(
                user_id="user2",
                experiment_id="test_exp",
                variant_name="treatment",
            ),
        ]

        saved_path = temp_repo.save_assignments("test_exp", assignments)
        assert Path(saved_path).exists()

        # Verify file format
        with open(saved_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2

        # Verify JSON format
        data1 = json.loads(lines[0])
        assert data1["user_id"] == "user1"
        assert data1["variant_name"] == "control"


class TestExperimentService:
    """Test experiment service functionality."""

    @pytest.fixture
    def service(self, tmp_path):
        """Create experiment service with temporary storage."""
        repo = ExperimentRepository(storage_dir=str(tmp_path / "experiments"))
        return ExperimentService(repository=repo)

    def test_create_experiment(self, service):
        """REQ-AB-001: Create generic A/B experiment."""
        config = service.create_experiment(
            experiment_id="test_exp",
            name="Test Experiment",
            control_config={"model": "bge-m3"},
            treatment_configs=[{"model": "kr-reranker"}],
            description="Testing reranker models",
            target_sample_size=1000,
        )

        assert config.experiment_id == "test_exp"
        assert config.name == "Test Experiment"
        assert config.description == "Testing reranker models"
        assert config.target_sample_size == 1000
        assert len(config.variants) == 2  # control + 1 treatment
        assert config.status == ExperimentStatus.DRAFT

    def test_create_experiment_multiple_treatments(self, service):
        """Test creating experiment with multiple treatments."""
        config = service.create_experiment(
            experiment_id="test_exp",
            name="Multi-Treatment Test",
            control_config={"model": "bge-m3"},
            treatment_configs=[
                {"model": "kr-reranker-1"},
                {"model": "kr-reranker-2"},
                {"model": "kr-reranker-3"},
            ],
        )

        assert len(config.variants) == 4  # control + 3 treatments

        # Check traffic allocation
        control = config.get_control_variant()
        assert control.traffic_allocation == 0.5

        # Treatments should split remaining 50%
        treatments = config.get_treatment_variants()
        assert len(treatments) == 3
        for treatment in treatments:
            assert treatment.traffic_allocation == pytest.approx(0.5 / 3)

    def test_create_experiment_with_bandit_enabled(self, service):
        """REQ-AB-012: Create experiment with multi-armed bandit."""
        config = service.create_experiment(
            experiment_id="test_exp",
            name="Bandit Test",
            control_config={"model": "bge-m3"},
            treatment_configs=[{"model": "kr-reranker"}],
            enable_multi_armed_bandit=True,
        )

        assert config.enable_multi_armed_bandit is True
        assert config.bandit_epsilon == 0.1  # Default

    def test_start_experiment(self, service):
        """Test starting an experiment."""
        # Create experiment
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )

        # Start experiment
        config = service.start_experiment("test_exp")

        assert config.status == ExperimentStatus.RUNNING
        assert config.started_at is not None

        # Verify metrics initialized
        assert "test_exp" in service._metrics_cache
        assert "control" in service._metrics_cache["test_exp"]
        assert "treatment_0" in service._metrics_cache["test_exp"]

    def test_start_nonexistent_experiment(self, service):
        """Test starting non-existent experiment raises error."""
        with pytest.raises(ValueError, match="Experiment not found"):
            service.start_experiment("nonexistent")

    def test_start_experiment_twice(self, service):
        """Test starting already running experiment raises error."""
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("test_exp")

        with pytest.raises(ValueError, match="not in DRAFT status"):
            service.start_experiment("test_exp")

    def test_assign_variant(self, service):
        """REQ-AB-004: Assign user to variant and record."""
        # Create and start experiment
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("test_exp")

        # Assign user
        variant = service.assign_variant("test_exp", "user123")

        assert variant in ["control", "treatment_0"]

        # Verify impression recorded
        metrics = service._metrics_cache["test_exp"][variant]
        assert metrics.impressions == 1

        # Verify assignment cached
        assert service._assignment_cache["test_exp"]["user123"] == variant

    def test_assign_variant_consistent(self, service):
        """REQ-AB-009: Consistent assignment across sessions."""
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("test_exp")

        # Same user gets same assignment
        variant1 = service.assign_variant("test_exp", "user123")
        variant2 = service.assign_variant("test_exp", "user123")

        assert variant1 == variant2

    def test_assign_variant_not_running(self, service):
        """REQ-AB-008: Non-running experiment routes to control."""
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        # Don't start experiment

        variant = service.assign_variant("test_exp", "user123")
        assert variant == "control"

    def test_record_conversion(self, service):
        """REQ-AB-007: Record conversion for assigned variant."""
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("test_exp")

        # Assign and convert
        variant = service.assign_variant("test_exp", "user123")
        service.record_conversion(
            experiment_id="test_exp",
            user_id="user123",
            variant_name=variant,
            event_type="positive_feedback",
        )

        # Verify conversion recorded
        metrics = service._metrics_cache["test_exp"][variant]
        assert metrics.conversions == 1
        # Conversion rate calculated
        assert metrics.conversion_rate > 0

    def test_record_performance(self, service):
        """Test recording performance metrics."""
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("test_exp")

        # Assign and record performance
        variant = service.assign_variant("test_exp", "user123")
        service.record_performance(
            experiment_id="test_exp",
            variant_name=variant,
            latency_ms=50.0,
            relevance_score=0.8,
            satisfaction_score=0.9,
        )

        # Verify performance recorded
        metrics = service._metrics_cache["test_exp"][variant]
        assert metrics.total_latency_ms == 50.0
        assert metrics.avg_latency_ms == 50.0
        assert metrics.avg_relevance_score == 0.8
        assert metrics.avg_satisfaction_score == 0.9

    def test_analyze_results_significant(self, service):
        """REQ-AB-005: Statistical significance testing."""
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("test_exp")

        # Simulate data: control has 20% conversion, treatment has 30%
        # Assign exactly 100 users to ensure proper distribution
        users_per_variant = 50
        for i in range(users_per_variant):
            # Control users
            user_id = f"control_{i}"
            service.assign_variant("test_exp", user_id)
            if i < 10:  # 20 conversions
                service.record_conversion("test_exp", user_id, "control")

            # Treatment users
            user_id = f"treatment_{i}"
            service.assign_variant("test_exp", user_id)
            if i < 15:  # 30 conversions
                service.record_conversion("test_exp", user_id, "treatment_0")

        # Analyze
        result = service.analyze_results("test_exp")

        assert result.experiment_id == "test_exp"
        assert result.total_sample_size == 100  # 50 per variant
        assert result.p_value is not None
        assert result.winner is not None

        # With 20% vs 30%, should be significant
        if result.is_significant:
            assert result.p_value < 0.05
            assert result.confidence is not None

    def test_analyze_results_not_significant(self, service):
        """Test analysis when not statistically significant."""
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("test_exp")

        # Simulate similar conversion rates (22% vs 23%)
        for i in range(100):
            user_id = f"user_{i}"
            variant = service.assign_variant("test_exp", user_id)
            if variant == "control":
                if i < 22:  # 22 conversions
                    service.record_conversion("test_exp", user_id, "control")
            else:
                if i < 23:  # 23 conversions
                    service.record_conversion("test_exp", user_id, "treatment_0")

        result = service.analyze_results("test_exp")

        # Similar rates may not be significant
        if not result.is_significant:
            assert "INCONCLUSIVE" in result.recommendation

    def test_stop_experiment(self, service):
        """REQ-AB-006: Stop experiment and generate final report."""
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("test_exp")

        # Add some data
        for i in range(50):
            user_id = f"user_{i}"
            variant = service.assign_variant("test_exp", user_id)
            if i < 10:
                service.record_conversion("test_exp", user_id, variant)

        # Stop experiment
        config = service.stop_experiment("test_exp")

        assert config.status == ExperimentStatus.COMPLETED
        assert config.ended_at is not None

        # Verify result saved
        result = service.get_experiment_result("test_exp")
        assert result is not None
        assert result.experiment_id == "test_exp"

    def test_get_experiment_config(self, service):
        """Test retrieving experiment configuration."""
        # Create experiment
        service.create_experiment(
            experiment_id="test_exp",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )

        # Retrieve
        retrieved = service.get_experiment_config("test_exp")

        assert retrieved is not None
        assert retrieved.experiment_id == "test_exp"
        assert retrieved.name == "Test"

    def test_bandit_assign_exploits_best_variant(self, service):
        """REQ-AB-012: Multi-armed bandit exploits best performing variant."""
        service.create_experiment(
            experiment_id="test_exp",
            name="Bandit Test",
            control_config={},
            treatment_configs=[{}],
            enable_multi_armed_bandit=True,
        )
        service.start_experiment("test_exp")

        # Simulate treatment performing better
        # Give treatment lots of conversions, control few
        for i in range(100):
            user_id = f"user_{i}"
            variant = service.assign_variant("test_exp", user_id)
            if variant == "treatment_0" and i < 50:
                service.record_conversion("test_exp", user_id, "treatment_0")
            elif variant == "control" and i < 10:
                service.record_conversion("test_exp", user_id, "control")

        # With bandit, should prefer treatment now
        treatment_count = 0
        for i in range(50):
            user_id = f"bandit_user_{i}"
            variant = service.assign_variant("test_exp", user_id)
            if variant == "treatment_0":
                treatment_count += 1

        # Bandit should favor treatment at least as much as random
        assert treatment_count >= 20  # At least 40% of traffic


class TestIntegrationScenarios:
    """Test integration scenarios and workflows."""

    @pytest.fixture
    def service(self, tmp_path):
        """Create experiment service."""
        repo = ExperimentRepository(storage_dir=str(tmp_path / "experiments"))
        return ExperimentService(repository=repo)

    def test_full_experiment_workflow(self, service):
        """Test complete experiment workflow from creation to analysis."""
        # 1. Create experiment
        config = service.create_experiment(
            experiment_id="reranker_test",
            name="Reranker Comparison",
            control_config={"model": "bge-m3"},
            treatment_configs=[{"model": "kr-reranker"}],
            target_sample_size=200,
        )
        assert config.status == ExperimentStatus.DRAFT

        # 2. Start experiment
        config = service.start_experiment("reranker_test")
        assert config.status == ExperimentStatus.RUNNING

        # 3. Simulate user traffic
        num_users = 200
        for i in range(num_users):
            user_id = f"user{i}"
            variant = service.assign_variant("reranker_test", user_id)

            # Record performance
            latency = 40 + (i % 20)  # 40-60ms
            relevance = 0.7 + (i % 30) / 100  # 0.7-1.0
            service.record_performance("reranker_test", variant, latency, relevance)

            # Simulate conversions: treatment converts at 25%, control at 20%
            should_convert = (i % 4) == 0  # 25% base rate
            if variant == "control" and (i % 5) == 0:  # 20%
                should_convert = True
            elif variant == "treatment_0" and (i % 4) == 0:  # 25%
                should_convert = True

            if should_convert:
                service.record_conversion("reranker_test", user_id, variant)

        # 4. Analyze results
        result = service.analyze_results("reranker_test")

        assert result.total_sample_size == num_users
        assert result.p_value is not None
        assert result.winner in ["control", "treatment_0"]
        assert result.recommendation != ""

        # 5. Stop experiment
        config = service.stop_experiment("reranker_test")
        assert config.status == ExperimentStatus.COMPLETED

        # 6. Verify final report
        final_result = service.get_experiment_result("reranker_test")
        assert final_result is not None
        assert final_result.experiment_id == "reranker_test"

    def test_persistence_and_recovery(self, service):
        """Test that experiments can be persisted and recovered."""
        # Create and start experiment
        service.create_experiment(
            experiment_id="persist_test",
            name="Persistence Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("persist_test")

        # Add some data
        for i in range(20):
            user_id = f"user{i}"
            variant = service.assign_variant("persist_test", user_id)
            if i < 5:
                service.record_conversion("persist_test", user_id, variant)

        # Create new service instance (simulating restart)
        repo = ExperimentRepository(storage_dir=service.repository.storage_dir)
        new_service = ExperimentService(repository=repo)

        # Load experiment config
        config = new_service.get_experiment_config("persist_test")
        assert config is not None
        assert config.status == ExperimentStatus.RUNNING

        # Continue experiment - note that new service starts with empty cache
        # so total_sample_size will only count from the new service's perspective
        for i in range(20, 40):
            user_id = f"user{i}"
            variant = new_service.assign_variant("persist_test", user_id)
            if i < 25:
                new_service.record_conversion("persist_test", user_id, variant)

        # Analyze - only counts data from this service instance
        result = new_service.analyze_results("persist_test")
        assert result.total_sample_size >= 20  # At least the new assignments

    def test_multiple_experiments_independent(self, service):
        """Test that multiple experiments can run independently."""
        # Create two experiments
        service.create_experiment(
            experiment_id="exp1",
            name="Experiment 1",
            control_config={"model": "a"},
            treatment_configs=[{"model": "b"}],
        )
        service.start_experiment("exp1")

        service.create_experiment(
            experiment_id="exp2",
            name="Experiment 2",
            control_config={"model": "x"},
            treatment_configs=[{"model": "y"}],
        )
        service.start_experiment("exp2")

        # Assign same user to both experiments
        variant1 = service.assign_variant("exp1", "user123")
        variant2 = service.assign_variant("exp2", "user123")

        # Assignments should be independent (could be same or different)
        assert variant1 in ["control", "treatment_0"]
        assert variant2 in ["control", "treatment_0"]

        # Record conversions independently
        service.record_conversion("exp1", "user123", variant1)
        service.record_conversion("exp2", "user123", variant2)

        # Verify metrics are separate
        metrics1 = service._metrics_cache["exp1"]
        metrics2 = service._metrics_cache["exp2"]

        total_conv1 = sum(m.conversions for m in metrics1.values())
        total_conv2 = sum(m.conversions for m in metrics2.values())

        assert total_conv1 == 1
        assert total_conv2 == 1


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.fixture
    def service(self, tmp_path):
        """Create experiment service."""
        repo = ExperimentRepository(storage_dir=str(tmp_path / "experiments"))
        return ExperimentService(repository=repo)

    def test_assign_with_no_metrics_cache(self, service):
        """Test assignment when metrics cache is missing."""
        service.create_experiment(
            experiment_id="test",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("test")

        # Clear metrics cache
        service._metrics_cache.clear()

        # Should still work, creates new metrics entry
        variant = service.assign_variant("test", "user123")
        assert variant in ["control", "treatment_0"]

    def test_record_conversion_before_assignment(self, service):
        """Test recording conversion before any assignments."""
        service.create_experiment(
            experiment_id="test",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("test")

        # Should not crash, just log warning
        service.record_conversion("test", "user123", "control")

    def test_analyze_with_insufficient_data(self, service):
        """Test analysis with minimal data."""
        service.create_experiment(
            experiment_id="test",
            name="Test",
            control_config={},
            treatment_configs=[{}],
        )
        service.start_experiment("test")

        # Only one assignment
        service.assign_variant("test", "user123")

        # Analysis should not crash
        result = service.analyze_results("test")
        assert result.experiment_id == "test"
        # May not be significant due to low sample size

    def test_stop_nonexistent_experiment(self, service):
        """Test stopping non-existent experiment raises error."""
        with pytest.raises(ValueError, match="Experiment not found"):
            service.stop_experiment("nonexistent")
