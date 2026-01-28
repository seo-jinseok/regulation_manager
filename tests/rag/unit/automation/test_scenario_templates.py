"""
Test Scenario Templates for RAG Testing.

Tests for ambiguous query templates, multi-turn scenario templates,
and edge case templates defined in the infrastructure layer.
"""

import pytest

from src.rag.automation.domain.entities import (
    DifficultyLevel,
    FollowUpType,
    PersonaType,
)
from src.rag.automation.domain.extended_entities import AmbiguityType, EdgeCaseCategory
from src.rag.automation.infrastructure.test_scenario_templates import (
    AmbiguousQueryTemplates,
    EdgeCaseTemplates,
    MultiTurnScenarioTemplates,
)


class TestAmbiguousQueryTemplates:
    """Tests for AmbiguousQueryTemplates class."""

    def test_missing_context_templates_count(self):
        """Test that there are 6 missing context templates."""
        templates = AmbiguousQueryTemplates.MISSING_CONTEXT
        assert len(templates) == 6

    def test_missing_context_templates_structure(self):
        """Test that missing context templates have correct structure."""
        templates = AmbiguousQueryTemplates.MISSING_CONTEXT
        for template in templates:
            assert "query" in template
            assert "ambiguity_type" in template
            assert "difficulty" in template
            assert "expected_interpretations" in template
            assert "expected_clarifications" in template
            assert "context_hints" in template
            assert template["ambiguity_type"] == AmbiguityType.MISSING_CONTEXT

    def test_multiple_interpretations_templates_count(self):
        """Test that there are 6 multiple interpretations templates."""
        templates = AmbiguousQueryTemplates.MULTIPLE_INTERPRETATIONS
        assert len(templates) == 6

    def test_unclear_intent_templates_count(self):
        """Test that there are 6 unclear intent templates."""
        templates = AmbiguousQueryTemplates.UNCLEAR_INTENT
        assert len(templates) == 6

    def test_vague_terminology_templates_count(self):
        """Test that there are 6 vague terminology templates."""
        templates = AmbiguousQueryTemplates.VAGUE_TERMINOLOGY
        assert len(templates) == 6

    def test_incomplete_thought_templates_count(self):
        """Test that there are 6 incomplete thought templates."""
        templates = AmbiguousQueryTemplates.INCOMPLETE_THOUGHT
        assert len(templates) == 6

    def test_get_all_templates_count(self):
        """Test that get_all_templates returns 30 templates."""
        templates = AmbiguousQueryTemplates.get_all_templates()
        assert len(templates) == 30

    def test_get_templates_by_type(self):
        """Test getting templates by ambiguity type."""
        templates = AmbiguousQueryTemplates.get_templates_by_type(
            AmbiguityType.MISSING_CONTEXT
        )
        assert len(templates) == 6
        assert all(
            t["ambiguity_type"] == AmbiguityType.MISSING_CONTEXT for t in templates
        )

    def test_all_ambiguity_types_covered(self):
        """Test that all 5 ambiguity types have templates."""
        all_templates = AmbiguousQueryTemplates.get_all_templates()
        ambiguity_types = {t["ambiguity_type"] for t in all_templates}
        expected_types = {
            AmbiguityType.MISSING_CONTEXT,
            AmbiguityType.MULTIPLE_INTERPRETATIONS,
            AmbiguityType.UNCLEAR_INTENT,
            AmbiguityType.VAGUE_TERMINOLOGY,
            AmbiguityType.INCOMPLETE_THOUGHT,
        }
        assert ambiguity_types == expected_types

    @pytest.mark.parametrize(
        "template_index,expected_query",
        [
            (0, "그거 마감 언제야?"),
            (1, "신청하는데 뭐 필요해?"),
            (2, "어디서 해야 하나요?"),
            (3, "거기 누구 있어?"),
            (4, "방금 말한 거 다시 알려줘"),
            (5, "그 시험 언제 있지?"),
        ],
    )
    def test_missing_context_queries(self, template_index, expected_query):
        """Test specific missing context queries."""
        templates = AmbiguousQueryTemplates.MISSING_CONTEXT
        assert templates[template_index]["query"] == expected_query

    @pytest.mark.parametrize(
        "template_index,expected_query",
        [
            (0, "바뀌었나요?"),
            (1, "이거 되나요?"),
            (2, "신청 가능해?"),
            (3, "어떻게 돼?"),
            (4, "그거 받아?"),
            (5, "여기서도 돼?"),
        ],
    )
    def test_multiple_interpretation_queries(self, template_index, expected_query):
        """Test specific multiple interpretation queries."""
        templates = AmbiguousQueryTemplates.MULTIPLE_INTERPRETATIONS
        assert templates[template_index]["query"] == expected_query


class TestMultiTurnScenarioTemplates:
    """Tests for MultiTurnScenarioTemplates class."""

    def test_freshman_dormitory_scenario(self):
        """Test freshman dormitory scenario."""
        scenario = MultiTurnScenarioTemplates.FRESHMAN_DORMITORY
        assert scenario["scenario_id"] == "mt-freshman-dorm"
        assert scenario["persona_type"] == PersonaType.FRESHMAN
        assert scenario["difficulty"] == DifficultyLevel.EASY
        assert len(scenario["turns"]) == 4

    def test_junior_graduation_scenario(self):
        """Test junior graduation requirements scenario."""
        scenario = MultiTurnScenarioTemplates.JUNIOR_GRADUATION
        assert scenario["scenario_id"] == "mt-junior-grad"
        assert scenario["persona_type"] == PersonaType.JUNIOR
        assert scenario["difficulty"] == DifficultyLevel.MEDIUM
        assert len(scenario["turns"]) == 5

    def test_graduate_thesis_scenario(self):
        """Test graduate thesis review scenario."""
        scenario = MultiTurnScenarioTemplates.GRADUATE_THESIS
        assert scenario["scenario_id"] == "mt-grad-thesis"
        assert scenario["persona_type"] == PersonaType.GRADUATE
        assert scenario["difficulty"] == DifficultyLevel.HARD

    def test_distressed_leave_scenario(self):
        """Test distressed student leave of absence scenario."""
        scenario = MultiTurnScenarioTemplates.DISTRESSED_LEAVE
        assert scenario["scenario_id"] == "mt-distressed-leave"
        assert scenario["persona_type"] == PersonaType.DISTRESSED_STUDENT
        assert scenario["difficulty"] == DifficultyLevel.HARD
        assert len(scenario["turns"]) == 5

    def test_parent_tuition_scenario(self):
        """Test parent tuition inquiry scenario."""
        scenario = MultiTurnScenarioTemplates.PARENT_TUITION
        assert scenario["scenario_id"] == "mt-parent-tuition"
        assert scenario["persona_type"] == PersonaType.PARENT
        assert scenario["difficulty"] == DifficultyLevel.EASY
        assert len(scenario["turns"]) == 3

    def test_professor_sabbatical_scenario(self):
        """Test professor sabbatical leave scenario."""
        scenario = MultiTurnScenarioTemplates.PROFESSOR_SABBATICAL
        assert scenario["scenario_id"] == "mt-prof-sabbatical"
        assert scenario["persona_type"] == PersonaType.PROFESSOR
        assert scenario["difficulty"] == DifficultyLevel.HARD

    def test_international_visa_scenario(self):
        """Test international student visa scenario."""
        scenario = MultiTurnScenarioTemplates.INTERNATIONAL_VISA
        assert scenario["scenario_id"] == "mt-international-visa"
        assert scenario["initial_query"] == "학생 비자 연장 어떻게 하나요?"

    def test_staff_leave_scenario(self):
        """Test staff annual leave scenario."""
        scenario = MultiTurnScenarioTemplates.STAFF_LEAVE
        assert scenario["scenario_id"] == "mt-staff-leave"
        assert scenario["persona_type"] == PersonaType.NEW_STAFF

    def test_dissatisfied_complaint_scenario(self):
        """Test dissatisfied member complaint scenario."""
        scenario = MultiTurnScenarioTemplates.DISSATISFIED_COMPLAINT
        assert scenario["scenario_id"] == "mt-dissatisfied-complaint"
        assert scenario["persona_type"] == PersonaType.DISSATISFIED_MEMBER
        assert len(scenario["turns"]) == 5

    def test_get_all_scenarios_count(self):
        """Test that get_all_scenarios returns 15 scenarios."""
        scenarios = MultiTurnScenarioTemplates.get_all_scenarios()
        assert len(scenarios) == 15

    def test_all_scenarios_have_required_fields(self):
        """Test that all scenarios have required fields."""
        scenarios = MultiTurnScenarioTemplates.get_all_scenarios()
        for scenario in scenarios:
            assert "scenario_id" in scenario
            assert "name" in scenario
            assert "description" in scenario
            assert "persona_type" in scenario
            assert "difficulty" in scenario
            assert "initial_query" in scenario
            assert "initial_expected_intent" in scenario
            assert "turns" in scenario
            assert "expected_context_preservation_rate" in scenario

    def test_all_scenarios_have_unique_ids(self):
        """Test that all scenarios have unique IDs."""
        scenarios = MultiTurnScenarioTemplates.get_all_scenarios()
        ids = [s["scenario_id"] for s in scenarios]
        assert len(ids) == len(set(ids))

    def test_scenario_turns_structure(self):
        """Test that scenario turns have correct structure."""
        scenarios = MultiTurnScenarioTemplates.get_all_scenarios()
        for scenario in scenarios:
            for turn in scenario["turns"]:
                assert "turn_number" in turn
                assert "follow_up_type" in turn
                assert "query" in turn
                assert "expected_intent" in turn

    def test_follow_up_types_distribution(self):
        """Test that various follow-up types are used."""
        scenarios = MultiTurnScenarioTemplates.get_all_scenarios()
        follow_up_types = set()
        for scenario in scenarios:
            for turn in scenario["turns"]:
                follow_up_types.add(turn["follow_up_type"])

        expected_types = {
            FollowUpType.CLARIFICATION,
            FollowUpType.PROCEDURAL_DEEPENING,
            FollowUpType.EXCEPTION_CHECK,
            FollowUpType.RELATED_EXPANSION,
            FollowUpType.CONDITION_CHANGE,
            FollowUpType.CONFIRMATION,
            FollowUpType.COMPARISON,
        }
        assert expected_types.issubset(follow_up_types)

    @pytest.mark.parametrize(
        "scenario_key,expected_turn_count",
        [
            ("FRESHMAN_DORMITORY", 4),
            ("JUNIOR_GRADUATION", 5),
            ("GRADUATE_THESIS", 4),
            ("DISTRESSED_LEAVE", 5),
            ("PARENT_TUITION", 3),
        ],
    )
    def test_scenario_turn_counts(self, scenario_key, expected_turn_count):
        """Test that scenarios have expected number of turns."""
        scenario = getattr(MultiTurnScenarioTemplates, scenario_key)
        assert len(scenario["turns"]) == expected_turn_count


class TestEdgeCaseTemplates:
    """Tests for EdgeCaseTemplates class."""

    def test_emotional_templates_count(self):
        """Test that there are 4 emotional edge case templates."""
        templates = EdgeCaseTemplates.EMOTIONAL
        assert len(templates) == 4

    def test_emotional_templates_structure(self):
        """Test that emotional templates have correct structure."""
        templates = EdgeCaseTemplates.EMOTIONAL
        for template in templates:
            assert "scenario_id" in template
            assert "name" in template
            assert "category" in template
            assert "query" in template
            assert "is_emotional" in template
            assert "is_urgent" in template
            assert "is_confused" in template
            assert "is_frustrated" in template
            assert template["category"] == EdgeCaseCategory.EMOTIONAL
            assert template["is_emotional"] is True

    def test_emotional_template_frustrated_student(self):
        """Test frustrated student edge case."""
        template = EdgeCaseTemplates.EMOTIONAL[0]
        assert template["scenario_id"] == "edge-emotional-001"
        assert template["name"] == "좌절한 학생"
        assert "힘들어" in template["query"]
        assert template["is_frustrated"] is True
        assert template["should_show_empathy"] is True
        assert template["should_provide_contact"] is True

    def test_emotional_template_urgent_situation(self):
        """Test urgent situation edge case."""
        template = EdgeCaseTemplates.EMOTIONAL[1]
        assert template["scenario_id"] == "edge-emotional-002"
        assert template["name"] == "긴급한 상황"
        assert template["is_urgent"] is True
        assert template["expected_response_speed"] == "immediate"
        assert template["should_escalate"] is True

    def test_deadline_critical_templates_count(self):
        """Test that there are 3 deadline-critical templates."""
        templates = EdgeCaseTemplates.DEADLINE_CRITICAL
        assert len(templates) == 3

    def test_deadline_critical_templates_structure(self):
        """Test that deadline-critical templates have correct structure."""
        templates = EdgeCaseTemplates.DEADLINE_CRITICAL
        for template in templates:
            assert template["category"] == EdgeCaseCategory.DEADLINE_CRITICAL
            assert template["is_urgent"] is True

    def test_deadline_critical_template_one_hour_left(self):
        """Test one hour left deadline scenario."""
        template = EdgeCaseTemplates.DEADLINE_CRITICAL[0]
        assert template["scenario_id"] == "edge-deadline-001"
        assert template["name"] == "마감 1시간 전"
        assert "1시간" in template["query"]
        assert template["expected_response_speed"] == "immediate"

    def test_complex_synthesis_templates_count(self):
        """Test that there are 3 complex synthesis templates."""
        templates = EdgeCaseTemplates.COMPLEX_SYNTHESIS
        assert len(templates) == 3

    def test_complex_synthesis_templates_structure(self):
        """Test that complex synthesis templates have correct structure."""
        templates = EdgeCaseTemplates.COMPLEX_SYNTHESIS
        for template in templates:
            assert template["category"] == EdgeCaseCategory.COMPLEX_SYNTHESIS
            assert template["is_confused"] is True

    def test_cross_referenced_templates_count(self):
        """Test that there are 3 cross-referenced templates."""
        templates = EdgeCaseTemplates.CROSS_REFERENCED
        assert len(templates) == 3

    def test_cross_referenced_templates_structure(self):
        """Test that cross-referenced templates have correct structure."""
        templates = EdgeCaseTemplates.CROSS_REFERENCED
        for template in templates:
            assert template["category"] == EdgeCaseCategory.CROSS_REFERENCED

    def test_contradictory_templates_count(self):
        """Test that there are 2 contradictory templates."""
        templates = EdgeCaseTemplates.CONTRADICTORY
        assert len(templates) == 2

    def test_contradictory_templates_structure(self):
        """Test that contradictory templates have correct structure."""
        templates = EdgeCaseTemplates.CONTRADICTORY
        for template in templates:
            assert template["category"] == EdgeCaseCategory.CONTRADICTORY
            assert template["is_confused"] is True

    def test_get_all_templates_count(self):
        """Test that get_all_templates returns 15 edge case templates."""
        templates = EdgeCaseTemplates.get_all_templates()
        assert len(templates) == 15

    def test_get_by_category(self):
        """Test getting templates by category."""
        templates = EdgeCaseTemplates.get_by_category(EdgeCaseCategory.EMOTIONAL)
        assert len(templates) == 4
        assert all(t["category"] == EdgeCaseCategory.EMOTIONAL for t in templates)

    def test_all_categories_covered(self):
        """Test that all edge case categories have templates."""
        all_templates = EdgeCaseTemplates.get_all_templates()
        categories = {t["category"] for t in all_templates}
        expected_categories = {
            EdgeCaseCategory.EMOTIONAL,
            EdgeCaseCategory.DEADLINE_CRITICAL,
            EdgeCaseCategory.COMPLEX_SYNTHESIS,
            EdgeCaseCategory.CROSS_REFERENCED,
            EdgeCaseCategory.CONTRADICTORY,
        }
        assert categories == expected_categories

    def test_all_edge_cases_have_expected_regulations(self):
        """Test that all edge cases specify expected regulations."""
        templates = EdgeCaseTemplates.get_all_templates()
        for template in templates:
            assert "expected_regulations" in template
            assert isinstance(template["expected_regulations"], list)

    def test_all_edge_cases_have_expected_actions(self):
        """Test that all edge cases specify expected actions."""
        templates = EdgeCaseTemplates.get_all_templates()
        for template in templates:
            assert "expected_actions" in template
            assert isinstance(template["expected_actions"], list)

    @pytest.mark.parametrize(
        "category,expected_count",
        [
            (EdgeCaseCategory.EMOTIONAL, 4),
            (EdgeCaseCategory.DEADLINE_CRITICAL, 3),
            (EdgeCaseCategory.COMPLEX_SYNTHESIS, 3),
            (EdgeCaseCategory.CROSS_REFERENCED, 3),
            (EdgeCaseCategory.CONTRADICTORY, 2),
        ],
    )
    def test_category_template_counts(self, category, expected_count):
        """Test that each category has expected number of templates."""
        templates = EdgeCaseTemplates.get_by_category(category)
        assert len(templates) == expected_count


class TestScenarioTemplateIntegration:
    """Integration tests for scenario template collections."""

    def test_total_scenario_count(self):
        """Test total number of scenarios across all types."""
        ambiguous_count = len(AmbiguousQueryTemplates.get_all_templates())
        multi_turn_count = len(MultiTurnScenarioTemplates.get_all_scenarios())
        edge_case_count = len(EdgeCaseTemplates.get_all_templates())

        # Expected: 30 ambiguous + 15 multi-turn + 15 edge case = 60 total
        assert ambiguous_count == 30
        assert multi_turn_count == 15
        assert edge_case_count == 15

    def test_all_persona_types_covered(self):
        """Test that all persona types are covered in scenarios."""
        # Check multi-turn scenarios
        multi_turn_scenarios = MultiTurnScenarioTemplates.get_all_scenarios()
        multi_turn_personas = {s["persona_type"] for s in multi_turn_scenarios}

        # Check edge cases
        edge_cases = EdgeCaseTemplates.get_all_templates()
        edge_case_personas = {e["persona_type"] for e in edge_cases}

        # Should cover most persona types
        all_personas = multi_turn_personas | edge_case_personas
        assert len(all_personas) >= 5  # At least 5 different persona types

    def test_all_difficulty_levels_covered(self):
        """Test that all difficulty levels are represented."""
        multi_turn_scenarios = MultiTurnScenarioTemplates.get_all_scenarios()
        difficulties = {s["difficulty"] for s in multi_turn_scenarios}

        assert DifficultyLevel.EASY in difficulties
        assert DifficultyLevel.MEDIUM in difficulties
        assert DifficultyLevel.HARD in difficulties
