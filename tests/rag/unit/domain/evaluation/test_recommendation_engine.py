"""
Unit tests for RecommendationEngine.

Tests for SPEC-RAG-EVAL-001 Milestone 4: Report Enhancement.
"""

from typing import Dict

import pytest

from src.rag.domain.evaluation.failure_classifier import FailureType
from src.rag.domain.evaluation.recommendation_engine import (
    Priority,
    Recommendation,
    RecommendationEngine,
)


@pytest.fixture
def engine():
    """Create a RecommendationEngine instance."""
    return RecommendationEngine()


@pytest.fixture
def sample_failures():
    """Create sample failure counts."""
    return {
        FailureType.HALLUCINATION: 10,
        FailureType.MISSING_INFO: 8,
        FailureType.CITATION_ERROR: 5,
        FailureType.RETRIEVAL_FAILURE: 3,
        FailureType.AMBIGUITY: 2,  # Below default threshold
    }


class TestPriority:
    """Tests for Priority enum."""

    def test_all_priorities_exist(self):
        """Test that all expected priorities exist."""
        expected = ["critical", "high", "medium", "low"]

        for priority_name in expected:
            assert any(p.value == priority_name for p in Priority)


class TestRecommendation:
    """Tests for Recommendation."""

    def test_creation(self):
        """Test creating a Recommendation."""
        rec = Recommendation(
            id="REC-001",
            title="Test Recommendation",
            description="Test description",
            priority=Priority.HIGH,
            failure_types=[FailureType.HALLUCINATION],
            actions=["Action 1", "Action 2"],
            spec_mapping="SPEC-TEST-001",
            impact_estimate="High",
            effort_estimate="Medium",
        )

        assert rec.id == "REC-001"
        assert rec.title == "Test Recommendation"
        assert rec.priority == Priority.HIGH
        assert len(rec.actions) == 2
        assert rec.spec_mapping == "SPEC-TEST-001"

    def test_to_dict(self):
        """Test serialization."""
        rec = Recommendation(
            id="REC-002",
            title="Test",
            description="Description",
            priority=Priority.CRITICAL,
            failure_types=[FailureType.HALLUCINATION, FailureType.CITATION_ERROR],
            actions=["Action"],
            spec_mapping="SPEC-001",
        )

        data = rec.to_dict()

        assert data["id"] == "REC-002"
        assert data["priority"] == "critical"
        assert data["failure_types"] == ["hallucination", "citation_error"]
        assert data["actions"] == ["Action"]


class TestRecommendationEngine:
    """Tests for RecommendationEngine."""

    def test_init(self):
        """Test initialization."""
        engine = RecommendationEngine()

        assert len(engine.recommendations) > 0
        assert FailureType.HALLUCINATION in engine.recommendations

    def test_generate_recommendations_with_failures(self, engine, sample_failures):
        """Test generating recommendations from failure counts."""
        recommendations = engine.generate_recommendations(sample_failures, threshold=3)

        # Should generate recommendations for failures >= 3
        assert len(recommendations) >= 3  # HALLUCINATION, MISSING_INFO, CITATION_ERROR, RETRIEVAL_FAILURE

        # Check that hallucination recommendation is included
        hallucination_recs = [
            r for r in recommendations
            if FailureType.HALLUCINATION in r.failure_types
        ]
        assert len(hallucination_recs) == 1

    def test_generate_recommendations_below_threshold(self, engine, sample_failures):
        """Test that failures below threshold don't generate recommendations."""
        recommendations = engine.generate_recommendations(sample_failures, threshold=5)

        # Only HALLUCINATION (10) and MISSING_INFO (8) should generate
        assert len(recommendations) >= 2

        # AMBIGUITY (2) should not generate
        ambiguity_recs = [
            r for r in recommendations
            if FailureType.AMBIGUITY in r.failure_types
        ]
        assert len(ambiguity_recs) == 0

    def test_generate_recommendations_empty_failures(self, engine):
        """Test with empty failure dict."""
        recommendations = engine.generate_recommendations({})

        assert recommendations == []

    def test_generate_recommendations_no_matching_template(self, engine):
        """Test with failure type that has no template."""
        failures = {FailureType.UNKNOWN: 10}

        recommendations = engine.generate_recommendations(failures)

        # UNKNOWN type has no template
        assert recommendations == []

    def test_map_to_spec(self, engine):
        """Test mapping failure type to SPEC."""
        spec = engine.map_to_spec(FailureType.HALLUCINATION)

        assert spec is not None
        assert "SPEC" in spec

    def test_map_to_spec_no_mapping(self, engine):
        """Test mapping failure type with no SPEC mapping."""
        # LOW_QUALITY has no spec_mapping in templates
        spec = engine.map_to_spec(FailureType.LOW_QUALITY)

        # LOW_QUALITY has spec_mapping=None in the template
        assert spec is None

    def test_prioritize(self, engine):
        """Test prioritization of recommendations."""
        recommendations = [
            engine.recommendations[FailureType.AMBIGUITY],  # MEDIUM
            engine.recommendations[FailureType.HALLUCINATION],  # CRITICAL
            engine.recommendations[FailureType.MISSING_INFO],  # HIGH
        ]

        prioritized = engine.prioritize(recommendations)

        # Should be sorted by priority
        assert prioritized[0].priority == Priority.CRITICAL
        assert prioritized[1].priority == Priority.HIGH
        assert prioritized[2].priority == Priority.MEDIUM

    def test_prioritize_empty_list(self, engine):
        """Test prioritizing empty list."""
        prioritized = engine.prioritize([])

        assert prioritized == []

    def test_get_action_plan(self, engine, sample_failures):
        """Test generating action plan."""
        recommendations = engine.generate_recommendations(sample_failures, threshold=5)

        plan = engine.get_action_plan(recommendations)

        assert "immediate_actions" in plan
        assert "short_term_actions" in plan
        assert "long_term_actions" in plan
        assert "total_actions" in plan
        assert "action_list" in plan

        # Should have some immediate actions (CRITICAL or HIGH)
        assert len(plan["immediate_actions"]) >= 1

    def test_get_action_plan_empty(self, engine):
        """Test action plan with no recommendations."""
        plan = engine.get_action_plan([])

        assert plan["immediate_actions"] == []
        assert plan["short_term_actions"] == []
        assert plan["long_term_actions"] == []
        assert plan["total_actions"] == 0

    def test_add_custom_recommendation(self, engine):
        """Test adding custom recommendation."""
        custom_rec = Recommendation(
            id="CUSTOM-001",
            title="Custom Recommendation",
            description="Custom description",
            priority=Priority.HIGH,
            failure_types=[FailureType.UNKNOWN],
            actions=["Custom action"],
        )

        engine.add_custom_recommendation(FailureType.UNKNOWN, custom_rec)

        # Should now be able to generate recommendation for UNKNOWN
        failures = {FailureType.UNKNOWN: 5}
        recommendations = engine.generate_recommendations(failures)

        assert len(recommendations) == 1
        assert recommendations[0].id == "CUSTOM-001"

    def test_get_recommendation_summary(self, engine, sample_failures):
        """Test generating recommendation summary."""
        recommendations = engine.generate_recommendations(sample_failures, threshold=5)

        summary = engine.get_recommendation_summary(recommendations)

        assert "Recommendations" in summary
        assert len(summary.split("\n")) > 1  # Should have multiple lines

    def test_get_recommendation_summary_empty(self, engine):
        """Test summary with no recommendations."""
        summary = engine.get_recommendation_summary([])

        assert "No recommendations" in summary


class TestRecommendationTemplates:
    """Tests for built-in recommendation templates."""

    def test_hallucination_template(self, engine):
        """Test hallucination recommendation template."""
        rec = engine.recommendations[FailureType.HALLUCINATION]

        assert rec.priority == Priority.CRITICAL
        assert "hallucination" in rec.title.lower()
        assert len(rec.actions) >= 2
        assert rec.spec_mapping is not None

    def test_missing_info_template(self, engine):
        """Test missing info recommendation template."""
        rec = engine.recommendations[FailureType.MISSING_INFO]

        assert rec.priority == Priority.HIGH
        assert "complete" in rec.title.lower() or "missing" in rec.title.lower()
        assert len(rec.actions) >= 2

    def test_citation_error_template(self, engine):
        """Test citation error recommendation template."""
        rec = engine.recommendations[FailureType.CITATION_ERROR]

        assert rec.priority == Priority.MEDIUM
        assert "citation" in rec.title.lower()
        assert len(rec.actions) >= 2

    def test_retrieval_failure_template(self, engine):
        """Test retrieval failure recommendation template."""
        rec = engine.recommendations[FailureType.RETRIEVAL_FAILURE]

        assert rec.priority == Priority.HIGH
        assert "retrieval" in rec.title.lower() or "relevance" in rec.title.lower()
        assert len(rec.actions) >= 2

    def test_ambiguity_template(self, engine):
        """Test ambiguity recommendation template."""
        rec = engine.recommendations[FailureType.AMBIGUITY]

        assert rec.priority == Priority.MEDIUM
        assert "ambig" in rec.title.lower()
        assert len(rec.actions) >= 2

    def test_irrelevance_template(self, engine):
        """Test irrelevance recommendation template."""
        rec = engine.recommendations[FailureType.IRRELEVANCE]

        assert rec.priority == Priority.HIGH
        assert "relevance" in rec.title.lower() or "alignment" in rec.title.lower()
        assert len(rec.actions) >= 2


class TestRecommendationEngineEdgeCases:
    """Edge case tests for RecommendationEngine."""

    def test_all_failure_types_have_templates_or_handled(self, engine):
        """Test that all failure types are handled."""
        # Check that main failure types have templates
        main_types = [
            FailureType.HALLUCINATION,
            FailureType.MISSING_INFO,
            FailureType.CITATION_ERROR,
            FailureType.RETRIEVAL_FAILURE,
            FailureType.AMBIGUITY,
            FailureType.IRRELEVANCE,
            FailureType.LOW_QUALITY,
        ]

        for failure_type in main_types:
            # Should either have a template or return empty list
            failures = {failure_type: 10}
            recommendations = engine.generate_recommendations(failures)
            # We just verify no exception is raised

    def test_high_threshold_filters_all(self, engine, sample_failures):
        """Test that high threshold filters all failures."""
        recommendations = engine.generate_recommendations(sample_failures, threshold=100)

        assert recommendations == []

    def test_zero_threshold_includes_all(self, engine, sample_failures):
        """Test that zero threshold includes all failures with templates."""
        recommendations = engine.generate_recommendations(sample_failures, threshold=0)

        # Should include all failure types that have templates
        assert len(recommendations) >= 4  # At least the main types

    def test_large_failure_counts(self, engine):
        """Test with very large failure counts."""
        failures = {
            FailureType.HALLUCINATION: 10000,
            FailureType.MISSING_INFO: 5000,
        }

        recommendations = engine.generate_recommendations(failures)

        assert len(recommendations) >= 2
        # Should handle large numbers without issue
