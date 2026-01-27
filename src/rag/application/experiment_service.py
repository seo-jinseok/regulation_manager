"""
A/B Testing Experiment Service.

Provides high-level experiment management:
- Experiment lifecycle management
- Variant assignment with user bucketing
- Metrics tracking and conversion recording
- Statistical analysis and winner selection
- Integration with RAG search pipeline

REQ-AB-001 to REQ-AB-015
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..domain.experiment.ab_test import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    StatisticalAnalyzer,
    UserAssignment,
    UserBucketer,
    VariantConfig,
    VariantMetrics,
    VariantType,
)

logger = logging.getLogger(__name__)


class ExperimentRepository:
    """Repository for experiment persistence."""

    def __init__(self, storage_dir: str = ".data/experiments"):
        """
        Initialize repository.

        Args:
            storage_dir: Directory to store experiment data.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_experiment_config(self, config: ExperimentConfig) -> str:
        """Save experiment configuration."""
        filepath = self.storage_dir / f"experiment_{config.experiment_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved experiment config: {config.experiment_id}")
        return str(filepath)

    def load_experiment_config(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Load experiment configuration."""
        filepath = self.storage_dir / f"experiment_{experiment_id}.json"
        if not filepath.exists():
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            config = ExperimentConfig(
                experiment_id=data["experiment_id"],
                name=data["name"],
                description=data.get("description", ""),
                variants=[
                    VariantConfig(
                        name=v["name"],
                        type=VariantType(v["type"]),
                        traffic_allocation=v["traffic_allocation"],
                        config=v.get("config", {}),
                    )
                    for v in data.get("variants", [])
                ],
                status=ExperimentStatus(data.get("status", "draft")),
                created_at=datetime.fromisoformat(data["created_at"]),
                started_at=datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None,
                ended_at=datetime.fromisoformat(data["ended_at"])
                if data.get("ended_at")
                else None,
                target_sample_size=data.get("target_sample_size"),
                significance_level=data.get("significance_level", 0.05),
                min_confidence_interval=data.get("min_confidence_interval", 0.95),
                enable_multi_armed_bandit=data.get("enable_multi_armed_bandit", False),
                bandit_epsilon=data.get("bandit_epsilon", 0.1),
                user_bucketing_enabled=data.get("user_bucketing_enabled", True),
                auto_stop_on_significance=data.get("auto_stop_on_significance", True),
            )
            return config
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to load experiment {experiment_id}: {e}")
            return None

    def save_metrics(
        self, experiment_id: str, metrics: Dict[str, VariantMetrics]
    ) -> str:
        """Save variant metrics."""
        filepath = self.storage_dir / f"metrics_{experiment_id}.json"
        data = {name: m.to_dict() for name, m in metrics.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(filepath)

    def load_metrics(self, experiment_id: str) -> Dict[str, VariantMetrics]:
        """Load variant metrics."""
        filepath = self.storage_dir / f"metrics_{experiment_id}.json"
        if not filepath.exists():
            return {}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            metrics = {}
            for name, m_data in data.items():
                metrics[name] = VariantMetrics(
                    variant_name=m_data["variant_name"],
                    impressions=m_data.get("impressions", 0),
                    unique_users=m_data.get("unique_users", 0),
                    conversions=m_data.get("conversions", 0),
                    conversion_rate=m_data.get("conversion_rate", 0.0),
                    total_latency_ms=m_data.get("total_latency_ms", 0.0),
                    avg_latency_ms=m_data.get("avg_latency_ms", 0.0),
                    avg_relevance_score=m_data.get("avg_relevance_score", 0.0),
                    avg_satisfaction_score=m_data.get("avg_satisfaction_score", 0.0),
                    first_impression=datetime.fromisoformat(m_data["first_impression"])
                    if m_data.get("first_impression")
                    else None,
                    last_impression=datetime.fromisoformat(m_data["last_impression"])
                    if m_data.get("last_impression")
                    else None,
                )
            return metrics
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to load metrics for {experiment_id}: {e}")
            return {}

    def save_assignments(
        self, experiment_id: str, assignments: List[UserAssignment]
    ) -> str:
        """Save user assignments."""
        filepath = self.storage_dir / f"assignments_{experiment_id}.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            for assignment in assignments:
                f.write(json.dumps(assignment.to_dict(), ensure_ascii=False) + "\n")
        return str(filepath)

    def save_result(self, result: ExperimentResult) -> str:
        """Save experiment result."""
        filepath = self.storage_dir / f"result_{result.experiment_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved experiment result: {result.experiment_id}")
        return str(filepath)


class ExperimentService:
    """Service for managing A/B experiments."""

    def __init__(self, repository: Optional[ExperimentRepository] = None):
        """
        Initialize experiment service.

        Args:
            repository: Optional repository for persistence.
        """
        self.repository = repository or ExperimentRepository()
        self._metrics_cache: Dict[str, Dict[str, VariantMetrics]] = {}
        self._assignment_cache: Dict[
            str, Dict[str, str]
        ] = {}  # {experiment_id: {user_id: variant}}

    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        control_config: Dict[str, Any],
        treatment_configs: List[Dict[str, Any]],
        description: str = "",
        target_sample_size: Optional[int] = None,
        enable_multi_armed_bandit: bool = False,
    ) -> ExperimentConfig:
        """
        Create a new experiment.

        REQ-AB-001: Generic A/B testing service.

        Args:
            experiment_id: Unique experiment identifier
            name: Human-readable experiment name
            control_config: Control variant configuration
            treatment_configs: List of treatment variant configurations
            description: Experiment description
            target_sample_size: Target sample size per variant
            enable_multi_armed_bandit: Enable multi-armed bandit (REQ-AB-012)

        Returns:
            Created experiment configuration
        """
        # Create variants
        variants = [
            VariantConfig(
                name="control",
                type=VariantType.CONTROL,
                traffic_allocation=0.5,
                config=control_config,
            )
        ]

        # Distribute traffic among treatments
        treatment_allocation = 0.5 / len(treatment_configs)
        for i, treatment in enumerate(treatment_configs):
            variants.append(
                VariantConfig(
                    name=f"treatment_{i}",
                    type=VariantType.TREATMENT,
                    traffic_allocation=treatment_allocation,
                    config=treatment,
                )
            )

        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            status=ExperimentStatus.DRAFT,
            target_sample_size=target_sample_size,
            enable_multi_armed_bandit=enable_multi_armed_bandit,
        )

        self.repository.save_experiment_config(config)
        return config

    def start_experiment(self, experiment_id: str) -> ExperimentConfig:
        """
        Start an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Updated experiment configuration
        """
        config = self.repository.load_experiment_config(experiment_id)
        if config is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if config.status != ExperimentStatus.DRAFT:
            raise ValueError(
                f"Experiment is not in DRAFT status: {config.status.value}"
            )

        config.status = ExperimentStatus.RUNNING
        config.started_at = datetime.now()

        self.repository.save_experiment_config(config)

        # Initialize metrics
        self._metrics_cache[experiment_id] = {
            variant.name: VariantMetrics(variant_name=variant.name)
            for variant in config.variants
        }

        logger.info(f"Started experiment: {experiment_id}")
        return config

    def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
    ) -> str:
        """
        Assign user to experiment variant.

        REQ-AB-004: Record assignment in metrics store.
        REQ-AB-009: Maintain consistent assignment across session.

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier

        Returns:
            Assigned variant name
        """
        config = self.repository.load_experiment_config(experiment_id)
        if config is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # REQ-AB-008: If experiment not running, route to default
        if config.status != ExperimentStatus.RUNNING:
            control = config.get_control_variant()
            return control.name if control else config.variants[0].name

        # Check cache for existing assignment (REQ-AB-009)
        if (
            experiment_id in self._assignment_cache
            and user_id in self._assignment_cache[experiment_id]
        ):
            variant_name = self._assignment_cache[experiment_id][user_id]
            # Validate consistency
            if config.user_bucketing_enabled:
                if UserBucketer.validate_assignment_consistency(
                    user_id, experiment_id, config.variants, variant_name
                ):
                    return variant_name

        # Multi-armed bandit or random assignment
        if config.enable_multi_armed_bandit:
            variant_name = self._bandit_assign(experiment_id, config)
        else:
            variant_name = UserBucketer.assign_variant(
                user_id, experiment_id, config.variants
            )

        # Cache assignment
        if experiment_id not in self._assignment_cache:
            self._assignment_cache[experiment_id] = {}
        self._assignment_cache[experiment_id][user_id] = variant_name

        # Record impression
        self._record_impression(experiment_id, variant_name, user_id)

        return variant_name

    def _bandit_assign(self, experiment_id: str, config: ExperimentConfig) -> str:
        """
        Assign variant using epsilon-greedy multi-armed bandit.

        REQ-AB-012: Multi-armed bandit algorithm for automatic optimization.
        """
        metrics = self._metrics_cache.get(experiment_id, {})

        # Explore: choose random variant
        if random.random() < config.bandit_epsilon:
            return random.choice(config.variants).name

        # Exploit: choose best performing variant
        best_variant = None
        best_score = -1.0

        for variant in config.variants:
            variant_metrics = metrics.get(variant.name)
            if not variant_metrics or variant_metrics.impressions == 0:
                continue

            # Calculate score: conversion rate with exploration bonus
            if variant_metrics.conversions > 0:
                score = variant_metrics.conversion_rate
            else:
                score = 0.0

            # Add small exploration bonus for variants with fewer impressions
            exploration_bonus = 1.0 / (variant_metrics.impressions + 1)
            score += exploration_bonus * 0.1

            if score > best_score:
                best_score = score
                best_variant = variant.name

        return best_variant if best_variant else config.variants[0].name

    def record_impression(
        self,
        experiment_id: str,
        variant_name: str,
        user_id: str,
    ) -> None:
        """
        Record variant impression.

        Args:
            experiment_id: Experiment identifier
            variant_name: Variant name
            user_id: User identifier
        """
        self._record_impression(experiment_id, variant_name, user_id)

    def _record_impression(
        self,
        experiment_id: str,
        variant_name: str,
        user_id: str,
    ) -> None:
        """Internal impression recording."""
        if experiment_id not in self._metrics_cache:
            self._metrics_cache[experiment_id] = {}

        if variant_name not in self._metrics_cache[experiment_id]:
            self._metrics_cache[experiment_id][variant_name] = VariantMetrics(
                variant_name=variant_name
            )

        metrics = self._metrics_cache[experiment_id][variant_name]
        metrics.impressions += 1
        metrics.last_impression = datetime.now()

        if metrics.first_impression is None:
            metrics.first_impression = metrics.last_impression

        # Track unique users
        if user_id not in getattr(metrics, "_tracked_users", set()):
            if not hasattr(metrics, "_tracked_users"):
                metrics._tracked_users = set()
            metrics._tracked_users.add(user_id)
            metrics.unique_users += 1

    def record_conversion(
        self,
        experiment_id: str,
        user_id: str,
        variant_name: str,
        event_type: str = "conversion",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record conversion event.

        REQ-AB-007: Record conversion for assigned variant.

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            variant_name: Variant name
            event_type: Type of conversion event
            metadata: Optional event metadata
        """
        if experiment_id not in self._metrics_cache:
            logger.warning(f"No metrics found for experiment: {experiment_id}")
            return

        if variant_name not in self._metrics_cache[experiment_id]:
            logger.warning(f"Variant not found in metrics: {variant_name}")
            return

        metrics = self._metrics_cache[experiment_id][variant_name]
        metrics.conversions += 1
        metrics.conversion_rate = (
            metrics.conversions / metrics.unique_users
            if metrics.unique_users > 0
            else 0.0
        )

        logger.info(
            f"Recorded conversion: experiment={experiment_id}, "
            f"variant={variant_name}, user={user_id}, "
            f"total_conversions={metrics.conversions}"
        )

    def record_performance(
        self,
        experiment_id: str,
        variant_name: str,
        latency_ms: float,
        relevance_score: Optional[float] = None,
        satisfaction_score: Optional[float] = None,
    ) -> None:
        """
        Record performance metrics.

        Args:
            experiment_id: Experiment identifier
            variant_name: Variant name
            latency_ms: Request latency in milliseconds
            relevance_score: Optional relevance score (0-1)
            satisfaction_score: Optional satisfaction score (0-1)
        """
        if experiment_id not in self._metrics_cache:
            return

        if variant_name not in self._metrics_cache[experiment_id]:
            return

        metrics = self._metrics_cache[experiment_id][variant_name]
        metrics.total_latency_ms += latency_ms
        metrics.avg_latency_ms = metrics.total_latency_ms / metrics.impressions

        if relevance_score is not None:
            # Update running average
            n = metrics.impressions
            metrics.avg_relevance_score = (
                metrics.avg_relevance_score * (n - 1) + relevance_score
            ) / n

        if satisfaction_score is not None:
            n = metrics.impressions
            metrics.avg_satisfaction_score = (
                metrics.avg_satisfaction_score * (n - 1) + satisfaction_score
            ) / n

    def analyze_results(self, experiment_id: str) -> ExperimentResult:
        """
        Analyze experiment results with statistical testing.

        REQ-AB-005: Calculate statistical significance (p-value < 0.05).
        REQ-AB-006: Generate final report with winner recommendation.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Experiment result with statistical analysis
        """
        config = self.repository.load_experiment_config(experiment_id)
        if config is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        metrics = self._metrics_cache.get(experiment_id, {})

        # Get control and treatment metrics
        control_variant = config.get_control_variant()
        if not control_variant:
            raise ValueError(
                f"No control variant found for experiment: {experiment_id}"
            )

        control_metrics = metrics.get(control_variant.name)
        if not control_metrics:
            raise ValueError(
                f"No metrics found for control variant: {control_variant.name}"
            )

        treatment_metrics_dict = {}
        for variant in config.get_treatment_variants():
            if variant.name in metrics:
                treatment_metrics_dict[variant.name] = metrics[variant.name]

        # Statistical analysis
        result = ExperimentResult(
            experiment_id=experiment_id,
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics_dict,
        )

        # Calculate total sample size
        result.total_sample_size = control_metrics.impressions + sum(
            m.impressions for m in treatment_metrics_dict.values()
        )

        # Check if target reached
        if config.target_sample_size:
            result.reached_target = (
                result.total_sample_size >= config.target_sample_size
            )

        # Perform statistical tests for each treatment
        for treatment_name, treatment_metrics in treatment_metrics_dict.items():
            # Calculate z-score
            z_score = StatisticalAnalyzer.calculate_z_score(
                control_conversions=control_metrics.conversions,
                control_total=control_metrics.unique_users,
                treatment_conversions=treatment_metrics.conversions,
                treatment_total=treatment_metrics.unique_users,
            )

            # Calculate p-value
            p_value = StatisticalAnalyzer.calculate_p_value(z_score)

            # Track best result
            if result.p_value is None or p_value < result.p_value:
                result.p_value = p_value
                result.z_score = z_score

                # Determine winner
                if treatment_metrics.conversion_rate > control_metrics.conversion_rate:
                    result.winner = treatment_name
                    improvement = (
                        (
                            treatment_metrics.conversion_rate
                            - control_metrics.conversion_rate
                        )
                        / control_metrics.conversion_rate
                        * 100
                        if control_metrics.conversion_rate > 0
                        else 0
                    )
                    result.confidence = (1 - p_value) * 100
                else:
                    result.winner = control_variant.name
                    result.confidence = (1 - p_value) * 100

            # Calculate confidence interval for winning variant
            if result.winner == treatment_name:
                result.confidence_interval = (
                    StatisticalAnalyzer.calculate_confidence_interval(
                        conversions=treatment_metrics.conversions,
                        total=treatment_metrics.unique_users,
                        confidence_level=config.min_confidence_interval,
                    )
                )
            elif (
                result.winner == control_variant.name and not result.confidence_interval
            ):
                result.confidence_interval = (
                    StatisticalAnalyzer.calculate_confidence_interval(
                        conversions=control_metrics.conversions,
                        total=control_metrics.unique_users,
                        confidence_level=config.min_confidence_interval,
                    )
                )

        # Check significance
        result.is_significant = (
            result.p_value is not None and result.p_value < config.significance_level
        )

        # Generate recommendation
        result.recommendation = self._generate_recommendation(result, config)

        # Save result
        self.repository.save_result(result)

        # Auto-stop if significant (REQ-AB-005)
        if result.is_significant and config.auto_stop_on_significance:
            logger.info(
                f"Experiment {experiment_id} reached statistical significance. "
                f"Winner: {result.winner} with confidence {result.confidence:.1f}%"
            )

        return result

    def _generate_recommendation(
        self, result: ExperimentResult, config: ExperimentConfig
    ) -> str:
        """Generate recommendation based on results."""
        if not result.is_significant:
            return f"INCONCLUSIVE: No statistically significant difference (p={result.p_value:.4f})"

        if result.p_value is None:
            return "INSUFFICIENT_DATA: Not enough data to draw conclusion"

        if result.winner == config.get_control_variant().name:
            return "CONTROL: Control variant performs best. No statistically significant improvement from treatments."

        # Treatment is winner
        improvement = (
            (
                result.treatment_metrics[result.winner].conversion_rate
                - result.control_metrics.conversion_rate
            )
            / result.control_metrics.conversion_rate
            * 100
            if result.control_metrics.conversion_rate > 0
            else 0
        )

        if improvement > 10:
            return f"ADOPT: Treatment '{result.winner}' shows {improvement:.1f}% improvement with {result.confidence:.1f}% confidence"
        elif improvement > 5:
            return f"CONSIDER: Treatment '{result.winner}' shows moderate improvement ({improvement:.1f}%) with {result.confidence:.1f}% confidence"
        else:
            return f"NEUTRAL: Treatment '{result.winner}' shows slight improvement ({improvement:.1f}%) but may not be worth implementing"

    def stop_experiment(self, experiment_id: str) -> ExperimentConfig:
        """
        Stop an experiment.

        REQ-AB-006: Generate final report when stopped.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Updated experiment configuration
        """
        config = self.repository.load_experiment_config(experiment_id)
        if config is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if config.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Experiment is not running: {config.status.value}")

        # Analyze final results
        result = self.analyze_results(experiment_id)

        # Update status
        config.status = ExperimentStatus.COMPLETED
        config.ended_at = datetime.now()

        # Save metrics
        self.repository.save_metrics(experiment_id, self._metrics_cache[experiment_id])

        self.repository.save_experiment_config(config)

        logger.info(
            f"Stopped experiment: {experiment_id}. "
            f"Winner: {result.winner}, Recommendation: {result.recommendation}"
        )

        return config

    def get_experiment_config(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration."""
        return self.repository.load_experiment_config(experiment_id)

    def get_experiment_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment result if available."""
        result_path = Path(self.repository.storage_dir) / f"result_{experiment_id}.json"
        if not result_path.exists():
            return None

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct metrics
            control_metrics = VariantMetrics(**data["control_metrics"])
            treatment_metrics = {
                name: VariantMetrics(**m_data)
                for name, m_data in data["treatment_metrics"].items()
            }

            result = ExperimentResult(
                experiment_id=data["experiment_id"],
                control_metrics=control_metrics,
                treatment_metrics=treatment_metrics,
                is_significant=data["is_significant"],
                p_value=data.get("p_value"),
                confidence_interval=data.get("confidence_interval"),
                z_score=data.get("z_score"),
                winner=data.get("winner"),
                confidence=data.get("confidence"),
                recommendation=data.get("recommendation", ""),
                total_sample_size=data.get("total_sample_size", 0),
                reached_target=data.get("reached_target", False),
                calculated_at=datetime.fromisoformat(data["calculated_at"]),
            )

            return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to load result for {experiment_id}: {e}")
            return None
