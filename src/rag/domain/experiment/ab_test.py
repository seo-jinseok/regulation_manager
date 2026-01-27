"""
General A/B Testing Framework Domain Entities.

Provides generic experiment management for RAG components:
- Experiment configuration with variant definitions
- User bucketing for consistent assignment
- Metrics tracking and statistical analysis
- Winner selection with confidence intervals

REQ-AB-001 to REQ-AB-015
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment lifecycle status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class VariantType(Enum):
    """Types of experiment variants."""

    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class VariantConfig:
    """Configuration for a single experiment variant."""

    name: str
    type: VariantType
    traffic_allocation: float = 0.5  # 0.0 to 1.0
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "traffic_allocation": self.traffic_allocation,
            "config": self.config,
        }


@dataclass
class ConversionEvent:
    """A conversion event recorded for a variant."""

    user_id: str
    variant_name: str
    timestamp: datetime
    event_type: str  # e.g., "positive_feedback", "successful_citation"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "variant_name": self.variant_name,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "metadata": self.metadata,
        }


@dataclass
class VariantMetrics:
    """Metrics for a single variant."""

    variant_name: str

    # Exposure metrics
    impressions: int = 0
    unique_users: int = 0

    # Conversion metrics
    conversions: int = 0
    conversion_rate: float = 0.0

    # Performance metrics
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0

    # Quality metrics
    avg_relevance_score: float = 0.0
    avg_satisfaction_score: float = 0.0

    # Timestamps
    first_impression: Optional[datetime] = None
    last_impression: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_name": self.variant_name,
            "impressions": self.impressions,
            "unique_users": self.unique_users,
            "conversions": self.conversions,
            "conversion_rate": self.conversion_rate,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_relevance_score": self.avg_relevance_score,
            "avg_satisfaction_score": self.avg_satisfaction_score,
            "first_impression": self.first_impression.isoformat()
            if self.first_impression
            else None,
            "last_impression": self.last_impression.isoformat()
            if self.last_impression
            else None,
        }


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""

    experiment_id: str
    name: str
    description: str = ""

    # Variants
    variants: List[VariantConfig] = field(default_factory=list)

    # Status and lifecycle
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Configuration
    target_sample_size: Optional[int] = None
    significance_level: float = 0.05  # p-value threshold (REQ-AB-005)
    min_confidence_interval: float = 0.95  # 95% confidence

    # Statistical settings
    enable_multi_armed_bandit: bool = False  # REQ-AB-012
    bandit_epsilon: float = 0.1  # Exploration rate for epsilon-greedy

    # Settings
    user_bucketing_enabled: bool = True  # REQ-AB-009
    auto_stop_on_significance: bool = True  # REQ-AB-005

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "target_sample_size": self.target_sample_size,
            "significance_level": self.significance_level,
            "min_confidence_interval": self.min_confidence_interval,
            "enable_multi_armed_bandit": self.enable_multi_armed_bandit,
            "bandit_epsilon": self.bandit_epsilon,
            "user_bucketing_enabled": self.user_bucketing_enabled,
            "auto_stop_on_significance": self.auto_stop_on_significance,
        }

    def get_control_variant(self) -> Optional[VariantConfig]:
        """Get the control variant."""
        for variant in self.variants:
            if variant.type == VariantType.CONTROL:
                return variant
        return None

    def get_treatment_variants(self) -> List[VariantConfig]:
        """Get all treatment variants."""
        return [v for v in self.variants if v.type == VariantType.TREATMENT]


@dataclass
class ExperimentResult:
    """Results of an A/B experiment with statistical analysis."""

    experiment_id: str
    control_metrics: VariantMetrics
    treatment_metrics: Dict[str, VariantMetrics]

    # Statistical analysis
    is_significant: bool = False
    p_value: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None
    z_score: Optional[float] = None

    # Winner selection (REQ-AB-006)
    winner: Optional[str] = None
    confidence: Optional[float] = None
    recommendation: str = ""

    # Sample size
    total_sample_size: int = 0
    reached_target: bool = False

    # Timestamp
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "control_metrics": self.control_metrics.to_dict(),
            "treatment_metrics": {
                name: m.to_dict() for name, m in self.treatment_metrics.items()
            },
            "is_significant": self.is_significant,
            "p_value": self.p_value,
            "confidence_interval": self.confidence_interval,
            "z_score": self.z_score,
            "winner": self.winner,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "total_sample_size": self.total_sample_size,
            "reached_target": self.reached_target,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class UserAssignment:
    """User assignment to experiment variant."""

    user_id: str
    experiment_id: str
    variant_name: str
    assigned_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "experiment_id": self.experiment_id,
            "variant_name": self.variant_name,
            "assigned_at": self.assigned_at.isoformat(),
            "metadata": self.metadata,
        }


class StatisticalAnalyzer:
    """Statistical analysis for A/B testing."""

    @staticmethod
    def calculate_z_score(
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int,
    ) -> float:
        """
        Calculate z-score for proportion test.

        Uses two-proportion z-test to determine if difference in conversion
        rates is statistically significant.

        Args:
            control_conversions: Number of conversions in control group
            control_total: Total sample size in control group
            treatment_conversions: Number of conversions in treatment group
            treatment_total: Total sample size in treatment group

        Returns:
            Z-score for the difference
        """
        if control_total == 0 or treatment_total == 0:
            return 0.0

        # Conversion rates
        p1 = control_conversions / control_total
        p2 = treatment_conversions / treatment_total

        # Pooled proportion
        p_pooled = (control_conversions + treatment_conversions) / (
            control_total + treatment_total
        )

        # Standard error
        se = (
            p_pooled * (1 - p_pooled) * (1 / control_total + 1 / treatment_total)
        ) ** 0.5

        if se == 0:
            return 0.0

        # Z-score
        z_score = (p2 - p1) / se
        return z_score

    @staticmethod
    def calculate_p_value(z_score: float) -> float:
        """
        Calculate two-tailed p-value from z-score.

        Args:
            z_score: Calculated z-score

        Returns:
            Two-tailed p-value
        """
        import math

        # Use complementary error function approximation
        abs_z = abs(z_score)

        # Abramowitz and Stegun approximation for erfc
        t = 1.0 / (1.0 + 0.2316419 * abs_z)
        d = 0.3989423 * math.exp(-(abs_z * abs_z) / 2.0)
        poly = 0.319381530 + t * (
            -0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))
        )

        # One-tailed p-value (area in the tail)
        one_tailed = d * t * poly

        # Two-tailed test
        p_value = 2.0 * one_tailed

        return min(p_value, 1.0)

    @staticmethod
    def calculate_confidence_interval(
        conversions: int, total: int, confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate confidence interval for conversion rate.

        Uses Wilson score interval for better accuracy with small samples.

        Args:
            conversions: Number of conversions
            total: Total sample size
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Dictionary with lower_bound, upper_bound, and point_estimate
        """
        import math

        if total == 0:
            return {"lower_bound": 0.0, "upper_bound": 0.0, "point_estimate": 0.0}

        p = conversions / total
        z = 1.96  # For 95% confidence (approximately)

        denominator = 1 + z * z / total
        center = (p + z * z / (2 * total)) / denominator
        margin = (
            z
            * math.sqrt((p * (1 - p) / total) + (z * z / (4 * total * total)))
            / denominator
        )

        return {
            "lower_bound": max(0.0, center - margin),
            "upper_bound": min(1.0, center + margin),
            "point_estimate": p,
        }

    @staticmethod
    def calculate_sample_size(
        baseline_rate: float,
        minimum_detectable_effect: float,
        significance_level: float = 0.05,
        power: float = 0.8,
    ) -> int:
        """
        Calculate required sample size for A/B test.

        Args:
            baseline_rate: Expected conversion rate for control (0.0 to 1.0)
            minimum_detectable_effect: Minimum relative change to detect (e.g., 0.1 for 10%)
            significance_level: Type I error rate (alpha)
            power: Statistical power (1 - beta)

        Returns:
            Required sample size per variant
        """
        import math

        # Z-scores
        z_alpha = 1.96  # For significance_level = 0.05
        z_beta = 0.84  # For power = 0.8

        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        p_pooled = (p1 + p2) / 2

        n = (
            z_alpha * math.sqrt(2 * p_pooled * (1 - p_pooled))
            + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2 / (p2 - p1) ** 2

        return math.ceil(n)


class UserBucketer:
    """Consistent user assignment to variants (REQ-AB-009)."""

    @staticmethod
    def hash_user_id(user_id: str, experiment_id: str) -> str:
        """
        Create deterministic hash for user assignment.

        Args:
            user_id: User identifier
            experiment_id: Experiment identifier

        Returns:
            Hexadecimal hash string
        """
        combined = f"{user_id}:{experiment_id}"
        return hashlib.sha256(combined.encode()).hexdigest()

    @staticmethod
    def assign_variant(
        user_id: str,
        experiment_id: str,
        variants: List[VariantConfig],
    ) -> str:
        """
        Assign user to variant based on hash and traffic allocation.

        Ensures consistent assignment across sessions (REQ-AB-009).

        Args:
            user_id: User identifier
            experiment_id: Experiment identifier
            variants: List of variant configurations

        Returns:
            Assigned variant name
        """
        if not variants:
            raise ValueError("No variants configured")

        # Create deterministic hash
        hash_str = UserBucketer.hash_user_id(user_id, experiment_id)
        hash_int = int(hash_str[:8], 16)  # Use first 8 hex characters
        bucket_value = hash_int / (2**32)  # Normalize to [0, 1]

        # Assign based on traffic allocation
        cumulative = 0.0
        for variant in variants:
            cumulative += variant.traffic_allocation
            if bucket_value <= cumulative:
                return variant.name

        # Fallback to last variant if rounding issues
        return variants[-1].name

    @staticmethod
    def validate_assignment_consistency(
        user_id: str,
        experiment_id: str,
        variants: List[VariantConfig],
        previous_assignment: str,
    ) -> bool:
        """
        Validate that user gets same assignment across sessions.

        Args:
            user_id: User identifier
            experiment_id: Experiment identifier
            variants: List of variant configurations
            previous_assignment: Previously assigned variant name

        Returns:
            True if assignment is consistent
        """
        current_assignment = UserBucketer.assign_variant(
            user_id, experiment_id, variants
        )
        return current_assignment == previous_assignment
