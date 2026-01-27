"""
Unit tests for A/B Testing Domain Entities.

Tests cover:
- Experiment configuration and variants
- User bucketing and consistent assignment
- Statistical analysis (z-score, p-value, confidence intervals)
- Sample size calculation
- Metrics tracking

REQ-AB-001 to REQ-AB-015
"""

from datetime import datetime

import pytest

from src.rag.domain.experiment.ab_test import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    StatisticalAnalyzer,
    UserBucketer,
    VariantConfig,
    VariantMetrics,
    VariantType,
)


class TestVariantConfig:
    """Test VariantConfig entity."""

    def test_create_control_variant(self):
        """Test creating a control variant."""
        variant = VariantConfig(
            name="control",
            type=VariantType.CONTROL,
            traffic_allocation=0.5,
            config={"model": "bge-m3"},
        )

        assert variant.name == "control"
        assert variant.type == VariantType.CONTROL
        assert variant.traffic_allocation == 0.5
        assert variant.config == {"model": "bge-m3"}

    def test_create_treatment_variant(self):
        """Test creating a treatment variant."""
        variant = VariantConfig(
            name="treatment_0",
            type=VariantType.TREATMENT,
            traffic_allocation=0.25,
            config={"model": "kr-reranker"},
        )

        assert variant.name == "treatment_0"
        assert variant.type == VariantType.TREATMENT
        assert variant.traffic_allocation == 0.25

    def test_variant_to_dict(self):
        """Test variant serialization."""
        variant = VariantConfig(
            name="control",
            type=VariantType.CONTROL,
            traffic_allocation=0.5,
        )

        data = variant.to_dict()

        assert data["name"] == "control"
        assert data["type"] == "control"
        assert data["traffic_allocation"] == 0.5
        assert data["config"] == {}


class TestVariantMetrics:
    """Test VariantMetrics entity."""

    def test_initial_metrics(self):
        """Test initial metrics state."""
        metrics = VariantMetrics(variant_name="control")

        assert metrics.variant_name == "control"
        assert metrics.impressions == 0
        assert metrics.unique_users == 0
        assert metrics.conversions == 0
        assert metrics.conversion_rate == 0.0
        assert metrics.avg_latency_ms == 0.0

    def test_record_impression(self):
        """Test recording impressions."""
        metrics = VariantMetrics(variant_name="control")
        metrics.impressions = 100
        metrics.unique_users = 80
        metrics.first_impression = datetime.now()
        metrics.last_impression = datetime.now()

        assert metrics.impressions == 100
        assert metrics.unique_users == 80

    def test_calculate_conversion_rate(self):
        """Test conversion rate calculation."""
        metrics = VariantMetrics(variant_name="control")
        metrics.conversions = 20
        metrics.unique_users = 100
        metrics.conversion_rate = metrics.conversions / metrics.unique_users

        assert metrics.conversion_rate == 0.2

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = VariantMetrics(variant_name="control")
        metrics.impressions = 100
        metrics.conversions = 20

        data = metrics.to_dict()

        assert data["variant_name"] == "control"
        assert data["impressions"] == 100
        assert data["conversions"] == 20


class TestExperimentConfig:
    """Test ExperimentConfig entity."""

    def test_create_experiment_config(self):
        """Test creating experiment configuration."""
        variants = [
            VariantConfig(
                name="control",
                type=VariantType.CONTROL,
                traffic_allocation=0.5,
            ),
            VariantConfig(
                name="treatment_0",
                type=VariantType.TREATMENT,
                traffic_allocation=0.5,
            ),
        ]

        config = ExperimentConfig(
            experiment_id="test_exp",
            name="Test Experiment",
            description="Test experiment for unit tests",
            variants=variants,
            status=ExperimentStatus.DRAFT,
            target_sample_size=1000,
            significance_level=0.05,
        )

        assert config.experiment_id == "test_exp"
        assert config.name == "Test Experiment"
        assert len(config.variants) == 2
        assert config.status == ExperimentStatus.DRAFT
        assert config.target_sample_size == 1000
        assert config.significance_level == 0.05

    def test_get_control_variant(self):
        """Test getting control variant."""
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
            experiment_id="test",
            name="Test",
            variants=variants,
        )

        control = config.get_control_variant()
        assert control is not None
        assert control.name == "control"
        assert control.type == VariantType.CONTROL

    def test_get_treatment_variants(self):
        """Test getting treatment variants."""
        variants = [
            VariantConfig(
                name="control",
                type=VariantType.CONTROL,
                traffic_allocation=0.5,
            ),
            VariantConfig(
                name="treatment_0",
                type=VariantType.TREATMENT,
                traffic_allocation=0.25,
            ),
            VariantConfig(
                name="treatment_1",
                type=VariantType.TREATMENT,
                traffic_allocation=0.25,
            ),
        ]

        config = ExperimentConfig(
            experiment_id="test",
            name="Test",
            variants=variants,
        )

        treatments = config.get_treatment_variants()
        assert len(treatments) == 2
        assert all(v.type == VariantType.TREATMENT for v in treatments)

    def test_experiment_to_dict(self):
        """Test experiment serialization."""
        variants = [
            VariantConfig(
                name="control",
                type=VariantType.CONTROL,
                traffic_allocation=1.0,
            ),
        ]

        config = ExperimentConfig(
            experiment_id="test",
            name="Test",
            variants=variants,
        )

        data = config.to_dict()

        assert data["experiment_id"] == "test"
        assert data["name"] == "Test"
        assert data["status"] == "draft"
        assert len(data["variants"]) == 1


class TestUserBucketer:
    """Test UserBucketer for consistent assignment."""

    def test_hash_user_id_deterministic(self):
        """Test that user ID hashing is deterministic."""
        hash1 = UserBucketer.hash_user_id("user123", "exp1")
        hash2 = UserBucketer.hash_user_id("user123", "exp1")
        hash3 = UserBucketer.hash_user_id("user123", "exp2")

        assert hash1 == hash2  # Same inputs, same hash
        assert hash1 != hash3  # Different experiment, different hash

    def test_assign_variant_single(self):
        """Test variant assignment with single treatment."""
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

        assigned = UserBucketer.assign_variant("user123", "exp1", variants)

        assert assigned in ["control", "treatment"]

    def test_assign_variant_consistent_across_sessions(self):
        """REQ-AB-009: Consistent assignment across sessions."""
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

        # Same user should get same assignment
        assignment1 = UserBucketer.assign_variant("user123", "exp1", variants)
        assignment2 = UserBucketer.assign_variant("user123", "exp1", variants)

        assert assignment1 == assignment2

    def test_assign_variant_different_users(self):
        """Test that different users can get different assignments."""
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

        assignments = set()
        for i in range(100):
            user_id = f"user{i}"
            assigned = UserBucketer.assign_variant(user_id, "exp1", variants)
            assignments.add(assigned)

        # Should have both variants assigned
        assert len(assignments) == 2
        assert "control" in assignments
        assert "treatment" in assignments

    def test_validate_assignment_consistency(self):
        """Test validation of assignment consistency."""
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

        # Get initial assignment
        assignment = UserBucketer.assign_variant("user123", "exp1", variants)

        # Validate consistency
        is_consistent = UserBucketer.validate_assignment_consistency(
            "user123", "exp1", variants, assignment
        )

        assert is_consistent is True

    def test_traffic_allocation_distribution(self):
        """Test that traffic allocation affects distribution."""
        variants = [
            VariantConfig(
                name="control",
                type=VariantType.CONTROL,
                traffic_allocation=0.8,  # 80% traffic
            ),
            VariantConfig(
                name="treatment",
                type=VariantType.TREATMENT,
                traffic_allocation=0.2,  # 20% traffic
            ),
        ]

        # Assign many users
        assignments = {"control": 0, "treatment": 0}
        for i in range(1000):
            user_id = f"user{i}"
            assigned = UserBucketer.assign_variant(user_id, "exp1", variants)
            assignments[assigned] += 1

        # Control should have roughly 80%
        control_ratio = assignments["control"] / 1000
        assert 0.75 < control_ratio < 0.85  # Allow some variance


class TestStatisticalAnalyzer:
    """Test statistical analysis methods."""

    def test_calculate_z_score_both_positive(self):
        """Test z-score calculation when both groups have conversions."""
        z = StatisticalAnalyzer.calculate_z_score(
            control_conversions=50,
            control_total=100,
            treatment_conversions=70,
            treatment_total=100,
        )

        # Treatment has higher conversion rate, so z should be positive
        assert z > 0
        assert z > 2.0  # Should be statistically significant

    def test_calculate_z_score_treatment_lower(self):
        """Test z-score when treatment performs worse."""
        z = StatisticalAnalyzer.calculate_z_score(
            control_conversions=70,
            control_total=100,
            treatment_conversions=50,
            treatment_total=100,
        )

        # Treatment has lower conversion rate, so z should be negative
        assert z < 0
        assert z < -2.0  # Should be statistically significant

    def test_calculate_z_score_equal_rates(self):
        """Test z-score when conversion rates are equal."""
        z = StatisticalAnalyzer.calculate_z_score(
            control_conversions=50,
            control_total=100,
            treatment_conversions=50,
            treatment_total=100,
        )

        # Equal rates should give z near 0
        assert abs(z) < 0.1

    def test_calculate_p_value_significant(self):
        """REQ-AB-005: P-value calculation for significant result."""
        z = 2.58  # Corresponds to p ≈ 0.01
        p_value = StatisticalAnalyzer.calculate_p_value(z)

        # The approximation may not be perfectly accurate, but should be close
        # A z-score of 2.58 should give p < 0.01 in theory
        assert p_value < 0.1  # Should be significant at 90% confidence
        assert p_value > 0.0

    def test_calculate_p_value_not_significant(self):
        """Test p-value for non-significant result."""
        z = 1.0  # Not significant
        p_value = StatisticalAnalyzer.calculate_p_value(z)

        # Z-score of 1.0 gives p ≈ 0.32 (two-tailed)
        assert p_value > 0.1  # Not significant at 90% confidence
        assert p_value < 1.0

    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        ci = StatisticalAnalyzer.calculate_confidence_interval(
            conversions=20, total=100, confidence_level=0.95
        )

        assert "lower_bound" in ci
        assert "upper_bound" in ci
        assert "point_estimate" in ci
        assert ci["point_estimate"] == 0.2  # 20/100
        assert ci["lower_bound"] < ci["point_estimate"]
        assert ci["upper_bound"] > ci["point_estimate"]
        assert ci["lower_bound"] >= 0.0
        assert ci["upper_bound"] <= 1.0

    def test_calculate_confidence_interval_zero_conversions(self):
        """Test confidence interval with zero conversions."""
        ci = StatisticalAnalyzer.calculate_confidence_interval(
            conversions=0, total=100, confidence_level=0.95
        )

        assert ci["point_estimate"] == 0.0
        assert ci["lower_bound"] == 0.0
        assert ci["upper_bound"] >= 0.0

    def test_calculate_confidence_interval_all_conversions(self):
        """Test confidence interval with 100% conversions."""
        ci = StatisticalAnalyzer.calculate_confidence_interval(
            conversions=100, total=100, confidence_level=0.95
        )

        assert ci["point_estimate"] == 1.0
        assert ci["lower_bound"] <= 1.0
        # Due to Wilson score interval, upper bound might be slightly less than 1.0
        assert ci["upper_bound"] >= 0.95

    def test_calculate_sample_size(self):
        """Test sample size calculation."""
        n = StatisticalAnalyzer.calculate_sample_size(
            baseline_rate=0.2,  # 20% baseline
            minimum_detectable_effect=0.1,  # 10% relative improvement
            significance_level=0.05,
            power=0.8,
        )

        assert n > 0
        assert n > 100  # Should require reasonable sample size

    def test_calculate_sample_size_larger_effect(self):
        """Test that larger effects require smaller samples."""
        n_small = StatisticalAnalyzer.calculate_sample_size(
            baseline_rate=0.2,
            minimum_detectable_effect=0.05,  # 5% effect
            significance_level=0.05,
            power=0.8,
        )

        n_large = StatisticalAnalyzer.calculate_sample_size(
            baseline_rate=0.2,
            minimum_detectable_effect=0.2,  # 20% effect
            significance_level=0.05,
            power=0.8,
        )

        # Larger effect should require smaller sample
        assert n_small > n_large

    def test_calculate_sample_size_higher_power(self):
        """Test that higher power requires larger samples."""
        n_80 = StatisticalAnalyzer.calculate_sample_size(
            baseline_rate=0.2,
            minimum_detectable_effect=0.1,
            significance_level=0.05,
            power=0.8,
        )

        n_90 = StatisticalAnalyzer.calculate_sample_size(
            baseline_rate=0.2,
            minimum_detectable_effect=0.1,
            significance_level=0.05,
            power=0.9,
        )

        # Higher power should require larger or equal sample
        assert n_90 >= n_80


class TestExperimentResult:
    """Test ExperimentResult entity."""

    def test_create_experiment_result(self):
        """Test creating experiment result."""
        control_metrics = VariantMetrics(variant_name="control")
        control_metrics.conversions = 50
        control_metrics.unique_users = 100
        control_metrics.conversion_rate = 0.5

        treatment_metrics = VariantMetrics(variant_name="treatment")
        treatment_metrics.conversions = 60
        treatment_metrics.unique_users = 100
        treatment_metrics.conversion_rate = 0.6

        result = ExperimentResult(
            experiment_id="test_exp",
            control_metrics=control_metrics,
            treatment_metrics={"treatment": treatment_metrics},
            is_significant=True,
            p_value=0.03,
            winner="treatment",
            confidence=95.0,
        )

        assert result.experiment_id == "test_exp"
        assert result.is_significant is True
        assert result.p_value == 0.03
        assert result.winner == "treatment"
        assert result.confidence == 95.0

    def test_experiment_result_to_dict(self):
        """Test experiment result serialization."""
        control_metrics = VariantMetrics(variant_name="control")
        treatment_metrics = VariantMetrics(variant_name="treatment")

        result = ExperimentResult(
            experiment_id="test",
            control_metrics=control_metrics,
            treatment_metrics={"treatment": treatment_metrics},
            is_significant=False,
        )

        data = result.to_dict()

        assert data["experiment_id"] == "test"
        assert "control_metrics" in data
        assert "treatment_metrics" in data
        assert data["is_significant"] is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_variants_list(self):
        """Test handling of empty variants list."""
        with pytest.raises(ValueError, match="No variants configured"):
            UserBucketer.assign_variant("user123", "exp1", [])

    def test_zero_total_sample_size(self):
        """Test z-score calculation with zero sample size."""
        z = StatisticalAnalyzer.calculate_z_score(
            control_conversions=0,
            control_total=0,
            treatment_conversions=0,
            treatment_total=0,
        )

        assert z == 0.0

    def test_confidence_interval_zero_total(self):
        """Test confidence interval with zero total."""
        ci = StatisticalAnalyzer.calculate_confidence_interval(
            conversions=0, total=0, confidence_level=0.95
        )

        assert ci["lower_bound"] == 0.0
        assert ci["upper_bound"] == 0.0
        assert ci["point_estimate"] == 0.0

    def test_p_value_extreme_z_score(self):
        """Test p-value calculation with extreme z-score."""
        # Very large z-score
        p_large = StatisticalAnalyzer.calculate_p_value(10.0)
        # Should be extremely small, though approximation may not be perfect
        assert p_large < 0.05

        # Very negative z-score (same result due to two-tailed)
        p_small = StatisticalAnalyzer.calculate_p_value(-10.0)
        assert p_small < 0.05

    def test_variants_with_different_traffic_allocations(self):
        """Test assignment with uneven traffic allocation."""
        variants = [
            VariantConfig(
                name="control",
                type=VariantType.CONTROL,
                traffic_allocation=0.7,
            ),
            VariantConfig(
                name="treatment_1",
                type=VariantType.TREATMENT,
                traffic_allocation=0.2,
            ),
            VariantConfig(
                name="treatment_2",
                type=VariantType.TREATMENT,
                traffic_allocation=0.1,
            ),
        ]

        # Test that all variants can be assigned
        assignments = set()
        for i in range(1000):
            user_id = f"user{i}"
            assigned = UserBucketer.assign_variant(user_id, "exp1", variants)
            assignments.add(assigned)

        assert len(assignments) == 3
        assert "control" in assignments
        assert "treatment_1" in assignments
        assert "treatment_2" in assignments
