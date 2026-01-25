"""
Unit Tests for MultiTurnSimulator.

Tests for the multi-turn conversation simulator that generates context-aware
follow-up questions and executes multi-turn scenarios.

Clean Architecture: Infrastructure layer tests with mocked dependencies.
"""

from unittest.mock import MagicMock, Mock

import pytest

from src.rag.automation.domain.entities import (
    ContextHistory,
    DifficultyLevel,
    FollowUpType,
    MultiTurnScenario,
    Persona,
    PersonaType,
    Turn,
)
from src.rag.automation.infrastructure.multi_turn_simulator import MultiTurnSimulator

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def freshman_persona():
    """Create a freshman persona for testing."""
    return Persona(
        persona_type=PersonaType.FRESHMAN,
        name="신입생",
        description="학교 시스템에 익숙하지 않음",
        characteristics=["학교 생활 미숙", "궁금한 것 많음"],
        query_styles=["간단한 질문", "구체적인 도움 요청"],
        context_hints=["입학 첫 해", "규정을 잘 모름"],
    )


@pytest.fixture
def professor_persona():
    """Create a professor persona for testing."""
    return Persona(
        persona_type=PersonaType.PROFESSOR,
        name="정교수",
        description="세부 규정 확인 필요",
        characteristics=["연구 중심", "행정 절차 관심"],
        query_styles=["공식적인 어조", "구체적인 질문"],
        context_hints=["교원 평가", "연구비"],
    )


@pytest.fixture
def mock_execute_usecase():
    """Create mock ExecuteTestUseCase."""
    mock = MagicMock()
    mock.execute_query.return_value = MagicMock(
        answer="휴학 신청은 학사팀에 방문하시면 됩니다.",
        sources=["학칙 제12조"],
        confidence=0.85,
    )
    return mock


@pytest.fixture
def mock_context_tracker():
    """Create mock ContextTracker."""
    mock = MagicMock()

    # Setup create_initial_context to return proper ContextHistory
    initial_turn = Turn(
        turn_number=1,
        query="test query",
        answer="test answer",
        sources=["source1"],
        confidence=0.8,
    )
    mock.create_initial_context.return_value = ContextHistory(
        scenario_id="test_scenario",
        conversation_history=[initial_turn],
        implicit_entities={},
        topic_transitions=[],
        intent_history=["test query"],
    )

    # Setup update_context
    mock.update_context.return_value = Mock()
    mock.context_window_size = 3

    # Setup detect_context_preservation
    mock.detect_context_preservation.return_value = True

    # Setup _track_intent_evolution (private method, but will be called)
    mock._track_intent_evolution.return_value = "Intent evolved"

    # Setup _extract_implicit_info (private method)
    mock._extract_implicit_info.return_value = {}

    return mock


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client for testing."""
    mock = MagicMock()
    mock.generate.return_value = "추가로 궁금한 점이 있으신가요?"
    return mock


@pytest.fixture
def simulator(mock_execute_usecase, mock_context_tracker):
    """Create MultiTurnSimulator with mocked dependencies."""
    return MultiTurnSimulator(
        execute_usecase=mock_execute_usecase,
        context_tracker=mock_context_tracker,
        llm_client=None,  # Template-based generation
    )


@pytest.fixture
def sample_context():
    """Create sample conversation context."""
    turn1 = Turn(
        turn_number=1,
        query="휴학 신청은 어떻게 하나요?",
        answer="휴학 신청은 학사팀에 방문하시면 됩니다.",
        sources=["학칙 제12조"],
        confidence=0.85,
    )
    turn2 = Turn(
        turn_number=2,
        query="구체적으로 어떻게 되나요?",
        answer="신청서를 작성하고 제출하시면 됩니다.",
        sources=["학칙 제13조"],
        confidence=0.80,
    )

    return ContextHistory(
        scenario_id="test_scenario",
        conversation_history=[turn1, turn2],
        implicit_entities={},
        topic_transitions=[],
        intent_history=["휴학 신청은 어떻게 하나요?", "구체적으로 어떻게 되나요?"],
    )


@pytest.fixture
def sample_turn():
    """Create sample turn for testing."""
    return Turn(
        turn_number=1,
        query="휴학 신청은 어떻게 하나요?",
        answer="휴학 신청은 학사팀에 방문하시면 됩니다. 더 이상 질문이 없으신가요?",
        sources=["학칙 제12조"],
        confidence=0.85,
    )


# =============================================================================
# Test Suite: Initialization
# =============================================================================


class TestSimulatorInitialization:
    """Test suite for simulator initialization."""

    def test_simulator_initialization_with_all_dependencies(
        self, mock_execute_usecase, mock_context_tracker, mock_llm_client
    ):
        """WHEN simulator created with all dependencies, THEN should initialize correctly."""
        simulator = MultiTurnSimulator(
            execute_usecase=mock_execute_usecase,
            context_tracker=mock_context_tracker,
            llm_client=mock_llm_client,
        )

        assert simulator.execute is mock_execute_usecase
        assert simulator.context_tracker is mock_context_tracker
        assert simulator.llm_client is mock_llm_client

    def test_simulator_initialization_without_llm(
        self, mock_execute_usecase, mock_context_tracker
    ):
        """WHEN simulator created without LLM client, THEN should use template-based generation."""
        simulator = MultiTurnSimulator(
            execute_usecase=mock_execute_usecase,
            context_tracker=mock_context_tracker,
            llm_client=None,
        )

        assert simulator.llm_client is None


# =============================================================================
# Test Suite: Follow-Up Type Selection
# =============================================================================


class TestFollowUpTypeSelection:
    """Test suite for follow-up type selection logic."""

    def test_select_follow_up_type_turn_1(self, simulator):
        """WHEN selecting follow-up for turn 1, THEN should return CLARIFICATION."""
        follow_up_type = simulator._select_follow_up_type(turn_number=1, min_turns=3)

        assert follow_up_type == FollowUpType.CLARIFICATION

    def test_select_follow_up_type_turn_2(self, simulator):
        """WHEN selecting follow-up for turn 2, THEN should return CLARIFICATION."""
        follow_up_type = simulator._select_follow_up_type(turn_number=2, min_turns=3)

        assert follow_up_type == FollowUpType.CLARIFICATION

    def test_select_follow_up_type_turn_3(self, simulator):
        """WHEN selecting follow-up for turn 3, THEN should return RELATED_EXPANSION."""
        follow_up_type = simulator._select_follow_up_type(turn_number=3, min_turns=3)

        assert follow_up_type == FollowUpType.RELATED_EXPANSION

    def test_select_follow_up_type_turn_4(self, simulator):
        """WHEN selecting follow-up for turn 4, THEN should return EXCEPTION_CHECK."""
        follow_up_type = simulator._select_follow_up_type(turn_number=4, min_turns=3)

        assert follow_up_type == FollowUpType.EXCEPTION_CHECK

    def test_select_follow_up_type_turn_5(self, simulator):
        """WHEN selecting follow-up for turn 5, THEN should return PROCEDURAL_DEEPENING."""
        follow_up_type = simulator._select_follow_up_type(turn_number=5, min_turns=3)

        assert follow_up_type == FollowUpType.PROCEDURAL_DEEPENING

    def test_select_follow_up_type_after_min_turns(self, simulator):
        """WHEN selecting follow-up after min_turns, THEN may return various types or None."""
        # This test uses random, so we just verify it returns valid types or None
        valid_types = {
            None,
            FollowUpType.CONDITION_CHANGE,
            FollowUpType.CONFIRMATION,
            FollowUpType.COMPARISON,
        }

        # Run multiple times to check randomness
        results = set()
        for _ in range(20):
            result = simulator._select_follow_up_type(turn_number=6, min_turns=3)
            results.add(result)

        # All results should be valid
        assert results.issubset(valid_types)


# =============================================================================
# Test Suite: Follow-Up Question Generation
# =============================================================================


class TestFollowUpQuestionGeneration:
    """Test suite for follow-up question generation."""

    def test_generate_clarification_follow_up(
        self, simulator, sample_context, freshman_persona
    ):
        """WHEN generating CLARIFICATION follow-up, THEN should return clarification question."""
        follow_up = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.CLARIFICATION,
            persona=freshman_persona,
        )

        assert follow_up is not None
        assert len(follow_up) > 0
        # Check it's one of the expected templates
        assert any(
            template in follow_up
            for template in [
                "구체적으로",
                "자세히",
                "이해가 안",
            ]
        )

    def test_generate_related_expansion_follow_up(
        self, simulator, sample_context, freshman_persona
    ):
        """WHEN generating RELATED_EXPANSION follow-up, THEN should return expansion question."""
        follow_up = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.RELATED_EXPANSION,
            persona=freshman_persona,
        )

        assert follow_up is not None
        assert len(follow_up) > 0
        assert "관련" in follow_up or "비슷" in follow_up

    def test_generate_exception_check_follow_up(
        self, simulator, sample_context, freshman_persona
    ):
        """WHEN generating EXCEPTION_CHECK follow-up, THEN should return exception question."""
        follow_up = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.EXCEPTION_CHECK,
            persona=freshman_persona,
        )

        assert follow_up is not None
        assert len(follow_up) > 0
        assert "예외" in follow_up or "특별한" in follow_up

    def test_generate_procedural_deepening_follow_up(
        self, simulator, sample_context, freshman_persona
    ):
        """WHEN generating PROCEDURAL_DEEPENING follow-up, THEN should return procedural question."""
        follow_up = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.PROCEDURAL_DEEPENING,
            persona=freshman_persona,
        )

        assert follow_up is not None
        assert len(follow_up) > 0
        assert any(
            keyword in follow_up
            for keyword in ["구체적으로", "절차", "어디서", "어떻게"]
        )

    def test_generate_follow_up_with_empty_context(self, simulator, freshman_persona):
        """WHEN context is empty, THEN should return None."""
        empty_context = ContextHistory(
            scenario_id="empty",
            conversation_history=[],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        follow_up = simulator._generate_follow_up_question(
            context=empty_context,
            follow_up_type=FollowUpType.CLARIFICATION,
            persona=freshman_persona,
        )

        assert follow_up is None

    def test_generate_follow_up_with_persona_styles(
        self, simulator, sample_context, freshman_persona
    ):
        """WHEN persona has query_styles, THEN should include persona-specific prefix."""
        follow_up = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.CLARIFICATION,
            persona=freshman_persona,
        )

        # Freshman persona has query_styles: ["간단한 질문", "구체적인 도움 요청"]
        # The follow_up should potentially include one of these
        assert follow_up is not None
        assert len(follow_up) > 0


# =============================================================================
# Test Suite: Template Customization
# =============================================================================


class TestTemplateCustomization:
    """Test suite for template customization."""

    def test_customize_template_with_persona_styles(self, simulator, sample_turn):
        """WHEN persona has query_styles, THEN should add persona-specific prefix."""
        persona = Persona(
            persona_type=PersonaType.FRESHMAN,
            name="신입생",
            description="test",
            query_styles=["초보자라서", "잘 몰라서"],
            context_hints=[],
        )

        template = "구체적으로 어떻게 되나요?"
        customized = simulator._customize_template(template, sample_turn, persona)

        assert "구체적으로 어떻게 되나요?" in customized
        assert any(style in customized for style in ["초보자라서", "잘 몰라서"])

    def test_customize_template_without_persona_styles(self, simulator, sample_turn):
        """WHEN persona has no query_styles, THEN should return original template."""
        persona = Persona(
            persona_type=PersonaType.FRESHMAN,
            name="신입생",
            description="test",
            query_styles=[],
            context_hints=[],
        )

        template = "구체적으로 어떻게 되나요?"
        customized = simulator._customize_template(template, sample_turn, persona)

        assert customized == template


# =============================================================================
# Test Suite: Conversation Completion Detection
# =============================================================================


class TestConversationCompletion:
    """Test suite for conversation completion detection."""

    def test_conversation_complete_with_completion_indicator(
        self, simulator, sample_context
    ):
        """WHEN answer contains completion indicator, THEN should return True."""
        # Modify last turn to include completion indicator
        completed_turn = Turn(
            turn_number=3,
            query="더 질문 없나요?",
            answer="더 이상 드릴 정보가 없습니다. 완료되었습니다.",
            sources=["학칙 제15조"],
            confidence=0.90,
        )
        sample_context.conversation_history.append(completed_turn)

        is_complete = simulator._is_conversation_complete(sample_context)

        assert is_complete is True

    def test_conversation_complete_without_indicator(self, simulator, sample_context):
        """WHEN answer has no completion indicator, THEN should return False."""
        is_complete = simulator._is_conversation_complete(sample_context)

        assert is_complete is False

    def test_conversation_complete_empty_context(self, simulator):
        """WHEN context is empty, THEN should return False."""
        empty_context = ContextHistory(
            scenario_id="empty",
            conversation_history=[],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        is_complete = simulator._is_conversation_complete(empty_context)

        assert is_complete is False

    def test_conversation_complete_many_turns_similar_confidence(
        self, simulator, sample_context
    ):
        """WHEN many turns with similar confidence, THEN should return True."""
        # Add 4 more turns with similar confidence
        base_confidence = 0.80
        for i in range(3, 8):
            turn = Turn(
                turn_number=i,
                query=f"질문 {i}",
                answer=f"답변 {i}",
                sources=[f"source{i}"],
                confidence=base_confidence + (i % 3) * 0.02,  # Small variations
            )
            sample_context.conversation_history.append(turn)

        is_complete = simulator._is_conversation_complete(sample_context)

        assert is_complete is True


# =============================================================================
# Test Suite: Difficulty Assessment
# =============================================================================


class TestDifficultyAssessment:
    """Test suite for scenario difficulty assessment."""

    def test_assess_difficulty_empty_turns(self, simulator):
        """WHEN no turns, THEN should return EASY."""
        difficulty = simulator._assess_scenario_difficulty([])

        assert difficulty == DifficultyLevel.EASY

    def test_assess_difficulty_single_turn_high_confidence(self, simulator):
        """WHEN single turn with high confidence, THEN should return EASY."""
        turns = [
            Turn(
                turn_number=1,
                query="test query",
                answer="test answer",
                sources=["source1"],
                confidence=0.9,
                context_preserved=True,
            )
        ]

        difficulty = simulator._assess_scenario_difficulty(turns)

        assert difficulty == DifficultyLevel.EASY

    def test_assess_difficulty_many_turns(self, simulator):
        """WHEN 4 or more turns, THEN should increase difficulty."""
        turns = [
            Turn(
                turn_number=i,
                query=f"query {i}",
                answer=f"answer {i}",
                sources=[f"source{i}"],
                confidence=0.85,
                context_preserved=True,
            )
            for i in range(1, 5)
        ]

        difficulty = simulator._assess_scenario_difficulty(turns)

        # At least MEDIUM due to turn count
        assert difficulty in [DifficultyLevel.MEDIUM, DifficultyLevel.HARD]

    def test_assess_difficulty_context_failures(self, simulator):
        """WHEN context preservation failures, THEN should increase difficulty."""
        turns = [
            Turn(
                turn_number=1,
                query="query 1",
                answer="answer 1",
                sources=["source1"],
                confidence=0.85,
                context_preserved=False,  # Context not preserved
            ),
            Turn(
                turn_number=2,
                query="query 2",
                answer="answer 2",
                sources=["source2"],
                confidence=0.85,
                context_preserved=True,
            ),
        ]

        difficulty = simulator._assess_scenario_difficulty(turns)

        # At least MEDIUM due to context failure
        assert difficulty in [DifficultyLevel.MEDIUM, DifficultyLevel.HARD]

    def test_assess_difficulty_low_confidence(self, simulator):
        """WHEN average confidence is low, THEN should increase difficulty."""
        turns = [
            Turn(
                turn_number=1,
                query="query 1",
                answer="answer 1",
                sources=["source1"],
                confidence=0.5,  # Low confidence
                context_preserved=True,
            ),
            Turn(
                turn_number=2,
                query="query 2",
                answer="answer 2",
                sources=["source2"],
                confidence=0.6,  # Low confidence
                context_preserved=True,
            ),
        ]

        difficulty = simulator._assess_scenario_difficulty(turns)

        # At least MEDIUM due to low confidence
        assert difficulty in [DifficultyLevel.MEDIUM, DifficultyLevel.HARD]

    def test_assess_difficulty_multiple_factors(self, simulator):
        """WHEN multiple difficulty factors, THEN should return HARD."""
        turns = [
            Turn(
                turn_number=i,
                query=f"query {i}",
                answer=f"answer {i}",
                sources=[f"source{i}"],
                confidence=0.6,  # Low confidence
                context_preserved=False,  # Context not preserved
            )
            for i in range(1, 5)  # 4 turns
        ]

        difficulty = simulator._assess_scenario_difficulty(turns)

        assert difficulty == DifficultyLevel.HARD


# =============================================================================
# Test Suite: Scenario Generation
# =============================================================================


class TestScenarioGeneration:
    """Test suite for multi-turn scenario generation."""

    def test_generate_scenario_min_turns(
        self, simulator, freshman_persona, mock_execute_usecase, mock_context_tracker
    ):
        """WHEN generating scenario with min_turns=2, THEN should create scenario with at least 2 turns."""
        # Mock update_context to return proper context
        mock_context_tracker.update_context.return_value = ContextHistory(
            scenario_id="test_scenario",
            conversation_history=[],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        scenario = simulator.generate_scenario(
            scenario_id="test_001",
            persona=freshman_persona,
            initial_query="휴학 신청은 어떻게 하나요?",
            min_turns=2,
            max_turns=5,
        )

        assert isinstance(scenario, MultiTurnScenario)
        assert scenario.scenario_id == "test_001"
        assert scenario.persona_type == PersonaType.FRESHMAN
        assert scenario.initial_query == "휴학 신청은 어떻게 하나요?"
        assert len(scenario.turns) >= 2

    def test_generate_scenario_max_turns_limit(
        self, simulator, freshman_persona, mock_execute_usecase, mock_context_tracker
    ):
        """WHEN max_turns=3, THEN should not exceed 3 turns."""
        # Mock update_context
        mock_context_tracker.update_context.return_value = ContextHistory(
            scenario_id="test_scenario",
            conversation_history=[],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        scenario = simulator.generate_scenario(
            scenario_id="test_002",
            persona=freshman_persona,
            initial_query="휴학 신청은 어떻게 하나요?",
            min_turns=1,
            max_turns=3,
        )

        assert len(scenario.turns) <= 3

    def test_generate_scenario_context_preservation_rate(
        self, simulator, freshman_persona, mock_execute_usecase, mock_context_tracker
    ):
        """WHEN generating scenario, THEN should calculate context preservation rate."""
        # Mock update_context
        mock_context_tracker.update_context.return_value = ContextHistory(
            scenario_id="test_scenario",
            conversation_history=[],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        scenario = simulator.generate_scenario(
            scenario_id="test_003",
            persona=freshman_persona,
            initial_query="휴학 신청은 어떻게 하나요?",
            min_turns=1,
            max_turns=3,
        )

        # Context preservation rate should be between 0 and 1
        assert 0.0 <= scenario.context_preservation_rate <= 1.0

    def test_generate_scenario_metadata(
        self, simulator, freshman_persona, mock_execute_usecase, mock_context_tracker
    ):
        """WHEN generating scenario, THEN should include metadata."""
        # Mock update_context
        mock_context_tracker.update_context.return_value = ContextHistory(
            scenario_id="test_scenario",
            conversation_history=[],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        scenario = simulator.generate_scenario(
            scenario_id="test_004",
            persona=freshman_persona,
            initial_query="휴학 신청은 어떻게 하나요?",
            min_turns=1,
            max_turns=3,
        )

        assert "persona_name" in scenario.metadata
        assert "total_turns" in scenario.metadata
        assert "context_preservation_rate" in scenario.metadata
        assert scenario.metadata["persona_name"] == "신입생"
        assert scenario.metadata["total_turns"] == len(scenario.turns)


# =============================================================================
# Test Suite: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_empty_answer_handling(self, simulator, sample_context, freshman_persona):
        """WHEN answer is empty, THEN should still generate follow-up."""
        empty_answer_turn = Turn(
            turn_number=1,
            query="test",
            answer="",
            sources=[],
            confidence=0.0,
        )
        empty_context = ContextHistory(
            scenario_id="test",
            conversation_history=[empty_answer_turn],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        follow_up = simulator._generate_follow_up_question(
            context=empty_context,
            follow_up_type=FollowUpType.CLARIFICATION,
            persona=freshman_persona,
        )

        # Should still generate from template
        assert follow_up is not None

    def test_long_answer_handling(self, simulator, sample_context, freshman_persona):
        """WHEN answer is very long, THEN should still process correctly."""
        long_answer = "답변 내용 " * 1000  # Very long answer
        long_turn = Turn(
            turn_number=1,
            query="test",
            answer=long_answer,
            sources=["source1"],
            confidence=0.8,
        )
        long_context = ContextHistory(
            scenario_id="test",
            conversation_history=[long_turn],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[long_answer],
        )

        follow_up = simulator._generate_follow_up_question(
            context=long_context,
            follow_up_type=FollowUpType.CLARIFICATION,
            persona=freshman_persona,
        )

        # Should still generate
        assert follow_up is not None

    def test_none_follow_up_type(self, simulator, sample_context, freshman_persona):
        """WHEN follow_up_type is None, THEN should return None."""
        follow_up = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=None,
            persona=freshman_persona,
        )

        assert follow_up is None

    def test_special_characters_in_query(
        self, simulator, freshman_persona, mock_execute_usecase, mock_context_tracker
    ):
        """WHEN query has special characters, THEN should handle gracefully."""
        # Mock update_context
        mock_context_tracker.update_context.return_value = ContextHistory(
            scenario_id="test",
            conversation_history=[],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        scenario = simulator.generate_scenario(
            scenario_id="test_special",
            persona=freshman_persona,
            initial_query="휴학?! (신청) #방법",
            min_turns=1,
            max_turns=2,
        )

        assert scenario is not None
        assert len(scenario.turns) >= 1


# =============================================================================
# Test Suite: Turn Structure
# =============================================================================


class TestTurnStructure:
    """Test suite for turn structure and properties."""

    def test_turn_creation(self):
        """WHEN creating a turn, THEN should initialize correctly."""
        turn = Turn(
            turn_number=1,
            query="휴학 신청은 어떻게 하나요?",
            answer="휴학 신청은 학사팀에 방문하시면 됩니다.",
            sources=["학칙 제12조"],
            confidence=0.85,
            follow_up_type=FollowUpType.CLARIFICATION,
            intent_evolution="Clarifying: 휴학 신청은 어떻게 하나요?...",
            implicit_info_extracted=["implicit_그거"],
            context_preserved=True,
        )

        assert turn.turn_number == 1
        assert turn.follow_up_type == FollowUpType.CLARIFICATION
        assert turn.context_preserved is True
        assert len(turn.implicit_info_extracted) > 0

    def test_scenario_properties(self):
        """WHEN creating scenario, THEN should calculate properties correctly."""
        turns = [
            Turn(
                turn_number=1,
                query="query1",
                answer="answer1",
                sources=["source1"],
                confidence=0.9,
                context_preserved=True,
            ),
            Turn(
                turn_number=2,
                query="query2",
                answer="answer2",
                sources=["source2"],
                confidence=0.8,
                follow_up_type=FollowUpType.CLARIFICATION,
                context_preserved=False,
            ),
            Turn(
                turn_number=3,
                query="query3",
                answer="answer3",
                sources=["source3"],
                confidence=0.85,
                follow_up_type=FollowUpType.RELATED_EXPANSION,
                context_preserved=True,
            ),
        ]

        scenario = MultiTurnScenario(
            scenario_id="test_properties",
            persona_type=PersonaType.FRESHMAN,
            initial_query="query1",
            turns=turns,
            difficulty=DifficultyLevel.MEDIUM,
            context_window_size=3,
        )

        assert scenario.total_turns == 3
        assert scenario.context_preservation_rate == 2 / 3
        assert scenario.follow_up_distribution[FollowUpType.CLARIFICATION] == 1
        assert scenario.follow_up_distribution[FollowUpType.RELATED_EXPANSION] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
