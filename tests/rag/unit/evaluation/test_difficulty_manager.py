"""Unit tests for Difficulty Manager."""

import json
from pathlib import Path

import pytest

from src.rag.domain.evaluation.difficulty_manager import (
    DifficultyManager,
    DifficultyState,
    DifficultyTier,
    TIER_DESCRIPTIONS,
    TierState,
)


class TestTierState:
    """Test TierState dataclass."""

    def test_update_rate(self):
        ts = TierState(tier=1, total_queries=10, passed_queries=8)
        ts.update_rate()
        assert ts.pass_rate == pytest.approx(0.8)

    def test_update_rate_zero(self):
        ts = TierState(tier=1)
        ts.update_rate()
        assert ts.pass_rate == 0.0

    def test_to_dict(self):
        ts = TierState(tier=2, total_queries=5, passed_queries=4, pass_rate=0.8)
        d = ts.to_dict()
        assert d["tier"] == 2
        assert d["pass_rate"] == 0.8


class TestDifficultyState:
    """Test DifficultyState persistence."""

    def test_init_creates_all_tiers(self):
        state = DifficultyState()
        assert len(state.tiers) == 5
        for tier in DifficultyTier:
            assert tier.value in state.tiers

    def test_to_dict_and_from_dict(self):
        state = DifficultyState(current_tier=3)
        state.tiers[1].total_queries = 10
        state.tiers[1].passed_queries = 10
        state.tiers[1].pass_rate = 1.0
        state.tiers[1].mastered = True

        d = state.to_dict()
        restored = DifficultyState.from_dict(d)

        assert restored.current_tier == 3
        assert restored.tiers[1].mastered is True
        assert restored.tiers[1].total_queries == 10

    def test_from_dict_empty(self):
        state = DifficultyState.from_dict({})
        assert state.current_tier == 1


class TestDifficultyManager:
    """Test DifficultyManager class."""

    @pytest.fixture
    def manager(self, tmp_path):
        state_path = tmp_path / "difficulty_state.json"
        return DifficultyManager(state_path=str(state_path))

    def test_initial_state(self, manager):
        assert manager.get_current_tier() == 1

    def test_record_result(self, manager):
        manager.record_result(tier=1, passed=True)
        assert manager.state.tiers[1].total_queries == 1
        assert manager.state.tiers[1].passed_queries == 1

    def test_no_escalation_below_threshold(self, manager):
        for _ in range(5):
            manager.record_result(tier=1, passed=True)
        for _ in range(1):
            manager.record_result(tier=1, passed=False)

        # 5/6 = 83% < 95%
        result = manager.check_escalation()
        assert result is None
        assert manager.get_current_tier() == 1

    def test_escalation_at_threshold(self, manager):
        # Record 20 passed, 1 failed = 95.2% > 95%
        for _ in range(20):
            manager.record_result(tier=1, passed=True)
        manager.record_result(tier=1, passed=False)

        result = manager.check_escalation()
        assert result == 2
        assert manager.get_current_tier() == 2
        assert manager.state.tiers[1].mastered is True

    def test_no_escalation_below_min_queries(self, manager):
        # Only 3 queries (< MIN_QUERIES_FOR_MASTERY=5)
        for _ in range(3):
            manager.record_result(tier=1, passed=True)

        result = manager.check_escalation()
        assert result is None

    def test_no_escalation_at_max_tier(self, manager):
        manager.state.current_tier = 5
        for _ in range(20):
            manager.record_result(tier=5, passed=True)

        result = manager.check_escalation()
        assert result is None  # Can't go beyond L5
        assert manager.state.tiers[5].mastered is True

    def test_save_and_load_state(self, tmp_path):
        state_path = tmp_path / "state.json"
        mgr1 = DifficultyManager(state_path=str(state_path))

        for _ in range(10):
            mgr1.record_result(tier=1, passed=True)
        mgr1.save_state()

        # Load in new manager
        mgr2 = DifficultyManager(state_path=str(state_path))
        assert mgr2.state.tiers[1].total_queries == 10

    def test_format_tier_display_initial(self, manager):
        display = manager.format_tier_display()
        assert "L1 --" in display
        assert "L5 --" in display

    def test_format_tier_display_with_data(self, manager):
        for _ in range(10):
            manager.record_result(tier=1, passed=True)
        manager.check_escalation()
        manager.state.tiers[1].mastered = True

        display = manager.format_tier_display()
        assert "L1 \u2713" in display  # ✓

    def test_format_tier_display_percentage(self, manager):
        manager.state.tiers[2].total_queries = 10
        manager.state.tiers[2].passed_queries = 7
        manager.state.tiers[2].update_rate()

        display = manager.format_tier_display()
        assert "L2 70%" in display

    def test_get_tier_for_query_specified(self, manager):
        assert manager.get_tier_for_query(specified_tier=3) == 3
        assert manager.get_tier_for_query(specified_tier=0) == 1  # Clamped
        assert manager.get_tier_for_query(specified_tier=10) == 5  # Clamped

    def test_get_tier_for_query_adaptive(self, manager):
        manager.state.current_tier = 3
        assert manager.get_tier_for_query() == 3

    def test_get_summary(self, manager):
        summary = manager.get_summary()
        assert "current_tier" in summary
        assert "tier_display" in summary
        assert "tiers" in summary

    def test_reset(self, manager):
        for _ in range(10):
            manager.record_result(tier=1, passed=True)
        manager.reset()
        assert manager.state.current_tier == 1
        assert manager.state.tiers[1].total_queries == 0

    def test_tier_descriptions_exist(self):
        for tier in DifficultyTier:
            assert tier in TIER_DESCRIPTIONS

    def test_escalation_history_recorded(self, manager):
        for _ in range(20):
            manager.record_result(tier=1, passed=True)
        manager.check_escalation()

        assert len(manager.state.escalation_history) == 1
        assert manager.state.escalation_history[0]["from_tier"] == 1
        assert manager.state.escalation_history[0]["to_tier"] == 2
