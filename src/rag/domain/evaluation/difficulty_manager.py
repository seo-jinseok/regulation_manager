"""
Adaptive difficulty manager for evaluation.

SPEC: SPEC-RAG-EVAL-002
EARS: EARS-U-011 (Auto-Escalation), EARS-U-012 (Mastery State), EARS-U-013 (Tier Display)
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DifficultyTier(IntEnum):
    """5-level difficulty tier system."""
    L1 = 1  # Basic factual queries
    L2 = 2  # Multi-condition queries
    L3 = 3  # Cross-regulation inference
    L4 = 4  # Ambiguous/adversarial queries
    L5 = 5  # Frontier edge-cases


# Tier descriptions for display
TIER_DESCRIPTIONS = {
    DifficultyTier.L1: "Basic factual retrieval",
    DifficultyTier.L2: "Multi-condition lookup",
    DifficultyTier.L3: "Cross-regulation inference",
    DifficultyTier.L4: "Ambiguous / adversarial",
    DifficultyTier.L5: "Frontier edge-cases",
}


@dataclass
class TierState:
    """State of a single difficulty tier."""
    tier: int
    total_queries: int = 0
    passed_queries: int = 0
    pass_rate: float = 0.0
    mastered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def update_rate(self):
        if self.total_queries > 0:
            self.pass_rate = self.passed_queries / self.total_queries
        else:
            self.pass_rate = 0.0


@dataclass
class DifficultyState:
    """Persistent state for adaptive difficulty system."""
    current_tier: int = 1
    tiers: Dict[int, TierState] = field(default_factory=dict)
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # Initialize all tiers if not present
        for tier_val in DifficultyTier:
            if tier_val.value not in self.tiers:
                self.tiers[tier_val.value] = TierState(tier=tier_val.value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_tier": self.current_tier,
            "tiers": {str(k): v.to_dict() for k, v in self.tiers.items()},
            "escalation_history": self.escalation_history[-20:],  # Keep last 20
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DifficultyState":
        state = cls(
            current_tier=data.get("current_tier", 1),
            escalation_history=data.get("escalation_history", []),
        )
        tiers_data = data.get("tiers", {})
        for tier_key, tier_data in tiers_data.items():
            tier_int = int(tier_key)
            state.tiers[tier_int] = TierState(**tier_data)
        return state


class DifficultyManager:
    """Manages adaptive difficulty with tier escalation.

    EARS-U-011: When pass_rate >= 95% for current tier, escalate to next tier.
    EARS-U-012: Persist mastery state across evaluation runs.
    EARS-U-013: Display format "L1 ✓ | L2 ✓ | L3 78% | L4 -- | L5 --"
    """

    ESCALATION_THRESHOLD = 0.95  # 95% pass rate to escalate
    MIN_QUERIES_FOR_MASTERY = 5  # Minimum queries before tier can be mastered

    def __init__(self, state_path: str = "data/evaluations/difficulty_state.json"):
        self.state_path = Path(state_path)
        self.state = self._load_state()

    def _load_state(self) -> DifficultyState:
        """Load state from file or return default."""
        if not self.state_path.exists():
            return DifficultyState()
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)
            return DifficultyState.from_dict(data)
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning("Failed to load difficulty state: %s", e)
            return DifficultyState()

    def save_state(self):
        """Save current state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, ensure_ascii=False, indent=2)

    def record_result(self, tier: int, passed: bool):
        """Record a query result for the given tier."""
        if tier not in self.state.tiers:
            self.state.tiers[tier] = TierState(tier=tier)

        ts = self.state.tiers[tier]
        ts.total_queries += 1
        if passed:
            ts.passed_queries += 1
        ts.update_rate()

    def check_escalation(self) -> Optional[int]:
        """Check if escalation is warranted. Returns new tier or None.

        EARS-U-011: Escalate when current tier pass_rate >= 95%.
        """
        current = self.state.current_tier
        tier_state = self.state.tiers.get(current)

        if not tier_state:
            return None

        if (
            tier_state.total_queries >= self.MIN_QUERIES_FOR_MASTERY
            and tier_state.pass_rate >= self.ESCALATION_THRESHOLD
        ):
            # Mark as mastered
            tier_state.mastered = True

            # Escalate if not at max
            if current < DifficultyTier.L5.value:
                new_tier = current + 1
                self.state.current_tier = new_tier
                self.state.escalation_history.append({
                    "from_tier": current,
                    "to_tier": new_tier,
                    "pass_rate": round(tier_state.pass_rate, 3),
                    "queries": tier_state.total_queries,
                })
                logger.info(
                    "Difficulty escalated: L%d → L%d (pass_rate=%.1f%%)",
                    current, new_tier, tier_state.pass_rate * 100,
                )
                return new_tier

        return None

    def get_current_tier(self) -> int:
        """Return current difficulty tier."""
        return self.state.current_tier

    def get_tier_for_query(self, specified_tier: Optional[int] = None) -> int:
        """Get the tier to use for next query.

        If specified_tier is given, use it. Otherwise use current adaptive tier.
        """
        if specified_tier is not None:
            return max(1, min(5, specified_tier))
        return self.state.current_tier

    def format_tier_display(self) -> str:
        """Format tier display string.

        EARS-U-013: "L1 ✓ | L2 ✓ | L3 78% | L4 -- | L5 --"
        """
        parts = []
        for tier_val in DifficultyTier:
            ts = self.state.tiers.get(tier_val.value)
            if ts is None or ts.total_queries == 0:
                parts.append(f"L{tier_val.value} --")
            elif ts.mastered:
                parts.append(f"L{tier_val.value} \u2713")
            else:
                pct = int(ts.pass_rate * 100)
                parts.append(f"L{tier_val.value} {pct}%")
        return " | ".join(parts)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current difficulty state."""
        return {
            "current_tier": self.state.current_tier,
            "tier_display": self.format_tier_display(),
            "tiers": {
                tier_val.value: {
                    "description": TIER_DESCRIPTIONS[tier_val],
                    "pass_rate": round(
                        self.state.tiers.get(tier_val.value, TierState(tier=tier_val.value)).pass_rate, 3
                    ),
                    "total_queries": self.state.tiers.get(
                        tier_val.value, TierState(tier=tier_val.value)
                    ).total_queries,
                    "mastered": self.state.tiers.get(
                        tier_val.value, TierState(tier=tier_val.value)
                    ).mastered,
                }
                for tier_val in DifficultyTier
            },
            "escalation_history": self.state.escalation_history[-5:],
        }

    def reset(self):
        """Reset difficulty state."""
        self.state = DifficultyState()
