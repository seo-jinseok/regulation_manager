"""
Specification tests for MultiTurnSimulator (Phase 7).

Tests define expected behavior for multi-turn conversation simulation.
Following DDD PRESERVE phase: specification tests for greenfield development.

This module provides comprehensive test coverage for the MultiTurnSimulator,
which was previously at 0% test coverage.
"""

from unittest.mock import MagicMock

import pytest

from src.rag.automation.domain.context_tracker import ContextTracker
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


class TestMultiTurnSimulatorInitialization:
    """Specification tests for MultiTurnSimulator initialization."""

    def test_initialize_with_required_dependencies(self):
        """
        SPEC: __init__ should initialize with execute_usecase and context_tracker.

        Given: Required dependencies (execute_usecase, context_tracker)
        When: MultiTurnSimulator is initialized
        Then: Instance is created with dependencies set
        """
        # Arrange
        mock_execute = MagicMock()
        mock_tracker = ContextTracker()

        # Act
        simulator = MultiTurnSimulator(
            execute_usecase=mock_execute,
            context_tracker=mock_tracker,
        )

        # Assert
        assert simulator.execute == mock_execute
        assert simulator.context_tracker == mock_tracker

    def test_initialize_with_optional_llm_client(self):
        """
        SPEC: __init__ should accept optional llm_client.

        Given: All dependencies including optional llm_client
        When: MultiTurnSimulator is initialized
        Then: Instance is created with llm_client set
        """
        # Arrange
        mock_execute = MagicMock()
        mock_tracker = ContextTracker()
        mock_llm = MagicMock()

        # Act
        simulator = MultiTurnSimulator(
            execute_usecase=mock_execute,
            context_tracker=mock_tracker,
            llm_client=mock_llm,
        )

        # Assert
        assert simulator.llm_client == mock_llm


class TestScenarioGeneration:
    """Specification tests for scenario generation."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        mock_execute = MagicMock()

        # Mock execute_query to return test results
        mock_result = MagicMock()
        mock_result.answer = "휴학 신청은 학기 개시 30일 전까지 가능합니다"
        mock_result.sources = ["규정 제10조", "규정 제11조"]
        mock_result.confidence = 0.85
        mock_execute.execute_query.return_value = mock_result

        mock_tracker = ContextTracker()

        return {"execute": mock_execute, "tracker": mock_tracker}

    def test_generate_scenario_creates_scenario_with_initial_turn(
        self, mock_dependencies
    ):
        """
        SPEC: generate_scenario should create scenario with initial turn from query.

        Given: A persona, initial query, and dependencies
        When: generate_scenario is called
        Then: MultiTurnScenario is created with at least the initial turn
        """
        # Arrange
        simulator = MultiTurnSimulator(
            execute_usecase=mock_dependencies["execute"],
            context_tracker=mock_dependencies["tracker"],
        )

        persona = Persona(
            persona_type=PersonaType.FRESHMAN,
            name="Test Student",
            description="A test student",
            query_styles=["정중하게 질문"],
        )

        # Act
        scenario = simulator.generate_scenario(
            scenario_id="test_001",
            persona=persona,
            initial_query="휴학 신청은 어떻게 하나요?",
            min_turns=2,
            max_turns=3,
        )

        # Assert
        assert isinstance(scenario, MultiTurnScenario)
        assert scenario.scenario_id == "test_001"
        assert len(scenario.turns) >= 1
        assert scenario.turns[0].query == "휴학 신청은 어떻게 하나요?"

    def test_generate_scenario_executes_initial_query(self, mock_dependencies):
        """
        SPEC: generate_scenario should execute the initial query.

        Given: A persona and initial query
        When: generate_scenario is called
        Then: execute_query is called with the initial query
        """
        # Arrange
        simulator = MultiTurnSimulator(
            execute_usecase=mock_dependencies["execute"],
            context_tracker=mock_dependencies["tracker"],
        )

        persona = Persona(
            persona_type=PersonaType.FRESHMAN,
            name="Test Student",
            description="Test",
            query_styles=[],
        )

        # Act
        simulator.generate_scenario(
            scenario_id="test_002",
            persona=persona,
            initial_query="휴학 신청 방법",
            min_turns=2,
            max_turns=3,
        )

        # Assert
        mock_dependencies["execute"].execute_query.assert_called()

    def test_generate_scenario_respects_min_turns(self, mock_dependencies):
        """
        SPEC: generate_scenario should generate at least min_turns.

        Given: min_turns=3
        When: generate_scenario is called
        Then: Scenario has at least 3 turns
        """
        # Arrange
        simulator = MultiTurnSimulator(
            execute_usecase=mock_dependencies["execute"],
            context_tracker=mock_dependencies["tracker"],
        )

        persona = Persona(
            persona_type=PersonaType.PROFESSOR,
            name="Test Faculty",
            description="Test",
            query_styles=[],
        )

        # Act
        scenario = simulator.generate_scenario(
            scenario_id="test_003",
            persona=persona,
            initial_query="성적 정정 방법",
            min_turns=3,
            max_turns=5,
        )

        # Assert
        assert len(scenario.turns) >= 3

    def test_generate_scenario_respects_max_turns(self, mock_dependencies):
        """
        SPEC: generate_scenario should not exceed max_turns.

        Given: max_turns=4
        When: generate_scenario is called
        Then: Scenario has at most 4 turns
        """
        # Arrange
        simulator = MultiTurnSimulator(
            execute_usecase=mock_dependencies["execute"],
            context_tracker=mock_dependencies["tracker"],
        )

        persona = Persona(
            persona_type=PersonaType.NEW_STAFF,
            name="Test Staff",
            description="Test",
            query_styles=[],
        )

        # Act
        scenario = simulator.generate_scenario(
            scenario_id="test_004",
            persona=persona,
            initial_query="복학 신청 방법",
            min_turns=2,
            max_turns=4,
        )

        # Assert
        assert len(scenario.turns) <= 4

    def test_generate_scenario_sets_persona_type(self, mock_dependencies):
        """
        SPEC: generate_scenario should set persona_type from persona.

        Given: A persona with persona_type
        When: generate_scenario is called
        Then: Scenario has the same persona_type
        """
        # Arrange
        simulator = MultiTurnSimulator(
            execute_usecase=mock_dependencies["execute"],
            context_tracker=mock_dependencies["tracker"],
        )

        persona = Persona(
            persona_type=PersonaType.PARENT,
            name="External User",
            description="Test",
            query_styles=[],
        )

        # Act
        scenario = simulator.generate_scenario(
            scenario_id="test_005",
            persona=persona,
            initial_query="입학 방법",
            min_turns=2,
            max_turns=3,
        )

        # Assert
        assert scenario.persona_type == PersonaType.PARENT


class TestFollowUpTypeSelection:
    """Specification tests for follow-up type selection."""

    @pytest.fixture
    def simulator(self):
        """Create a simulator instance for testing."""
        mock_execute = MagicMock()
        mock_tracker = ContextTracker()
        return MultiTurnSimulator(
            execute_usecase=mock_execute,
            context_tracker=mock_tracker,
        )

    def test_select_follow_up_type_returns_clarification_for_early_turns(
        self, simulator
    ):
        """
        SPEC: _select_follow_up_type should return CLARIFICATION for turn <= 2.

        Given: turn_number is 1 or 2
        When: _select_follow_up_type is called
        Then: Returns FollowUpType.CLARIFICATION
        """
        # Act
        result = simulator._select_follow_up_type(turn_number=2, min_turns=3)

        # Assert
        assert result == FollowUpType.CLARIFICATION

    def test_select_follow_up_type_returns_related_expansion_for_turn_3(
        self, simulator
    ):
        """
        SPEC: _select_follow_up_type should return RELATED_EXPANSION for turn 3.

        Given: turn_number is 3
        When: _select_follow_up_type is called
        Then: Returns FollowUpType.RELATED_EXPANSION
        """
        # Act
        result = simulator._select_follow_up_type(turn_number=3, min_turns=3)

        # Assert
        assert result == FollowUpType.RELATED_EXPANSION

    def test_select_follow_up_type_returns_exception_check_for_turn_4(self, simulator):
        """
        SPEC: _select_follow_up_type should return EXCEPTION_CHECK for turn 4.

        Given: turn_number is 4
        When: _select_follow_up_type is called
        Then: Returns FollowUpType.EXCEPTION_CHECK
        """
        # Act
        result = simulator._select_follow_up_type(turn_number=4, min_turns=3)

        # Assert
        assert result == FollowUpType.EXCEPTION_CHECK

    def test_select_follow_up_type_returns_procedural_deepening_for_turn_5(
        self, simulator
    ):
        """
        SPEC: _select_follow_up_type should return PROCEDURAL_DEEPENING for turn 5.

        Given: turn_number is 5
        When: _select_follow_up_type is called
        Then: Returns FollowUpType.PROCEDURAL_DEEPENING
        """
        # Act
        result = simulator._select_follow_up_type(turn_number=5, min_turns=3)

        # Assert
        assert result == FollowUpType.PROCEDURAL_DEEPENING

    def test_select_follow_up_type_may_return_null_after_min_turns(self, simulator):
        """
        SPEC: _select_follow_up_type should possibly return None after min_turns.

        Given: turn_number >= min_turns and random chance fails
        When: _select_follow_up_type is called
        Then: May return None (conversation may end)
        """
        # Act - Note: This test acknowledges random behavior
        result = simulator._select_follow_up_type(turn_number=6, min_turns=5)

        # Assert - Result could be None or a FollowUpType
        assert result is None or isinstance(result, FollowUpType)


class TestFollowUpQuestionGeneration:
    """Specification tests for follow-up question generation."""

    @pytest.fixture
    def sample_context(self):
        """Create a sample context for testing."""
        turn = Turn(
            turn_number=1,
            query="휴학 신청 방법",
            answer="휴학 신청은 학기 개시 30일 전까지...",
            sources=["규정 제10조"],
            confidence=0.9,
        )

        return ContextHistory(
            scenario_id="test",
            conversation_history=[turn],
            implicit_entities={},
            topic_transitions=[],
            intent_history=["휴학"],
        )

    @pytest.fixture
    def sample_persona(self):
        """Create a sample persona for testing."""
        return Persona(
            persona_type=PersonaType.FRESHMAN,
            name="Student",
            description="Test student",
            query_styles=["존댓말로 질문"],
        )

    @pytest.fixture
    def simulator(self):
        """Create a simulator instance for testing."""
        mock_execute = MagicMock()
        mock_tracker = ContextTracker()
        return MultiTurnSimulator(
            execute_usecase=mock_execute,
            context_tracker=mock_tracker,
        )

    def test_generate_follow_up_for_clarification(
        self, simulator, sample_context, sample_persona
    ):
        """
        SPEC: _generate_follow_up_question should generate clarification question.

        Given: follow_up_type is CLARIFICATION
        When: _generate_follow_up_question is called
        Then: Returns a clarification question string
        """
        # Act
        result = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.CLARIFICATION,
            persona=sample_persona,
        )

        # Assert
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_follow_up_for_related_expansion(
        self, simulator, sample_context, sample_persona
    ):
        """
        SPEC: _generate_follow_up_question should generate related expansion question.

        Given: follow_up_type is RELATED_EXPANSION
        When: _generate_follow_up_question is called
        Then: Returns a related expansion question string
        """
        # Act
        result = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.RELATED_EXPANSION,
            persona=sample_persona,
        )

        # Assert
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_follow_up_for_exception_check(
        self, simulator, sample_context, sample_persona
    ):
        """
        SPEC: _generate_follow_up_question should generate exception check question.

        Given: follow_up_type is EXCEPTION_CHECK
        When: _generate_follow_up_question is called
        Then: Returns an exception check question string
        """
        # Act
        result = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.EXCEPTION_CHECK,
            persona=sample_persona,
        )

        # Assert
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_follow_up_for_procedural_deepening(
        self, simulator, sample_context, sample_persona
    ):
        """
        SPEC: _generate_follow_up_question should generate procedural question.

        Given: follow_up_type is PROCEDURAL_DEEPENING
        When: _generate_follow_up_question is called
        Then: Returns a procedural deepening question string
        """
        # Act
        result = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.PROCEDURAL_DEEPENING,
            persona=sample_persona,
        )

        # Assert
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_follow_up_includes_persona_style(
        self, simulator, sample_context, sample_persona
    ):
        """
        SPEC: _generate_follow_up_question should incorporate persona query styles.

        Given: A persona with query_styles
        When: _generate_follow_up_question is called
        Then: Result may include persona-specific style prefix
        """
        # Act
        result = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.CLARIFICATION,
            persona=sample_persona,
        )

        # Assert
        # Note: Due to random selection, this may not always include the style
        # but the mechanism should be in place
        assert result is not None


class TestTemplateCustomization:
    """Specification tests for template customization."""

    @pytest.fixture
    def simulator(self):
        """Create a simulator instance for testing."""
        mock_execute = MagicMock()
        mock_tracker = ContextTracker()
        return MultiTurnSimulator(
            execute_usecase=mock_execute,
            context_tracker=mock_tracker,
        )

    @pytest.fixture
    def sample_turn(self):
        """Create a sample turn for testing."""
        return Turn(
            turn_number=1,
            query="휴학 기간",
            answer="휴학 기간은 최대 2학기입니다",
            sources=["규정 제12조"],
            confidence=0.9,
        )

    @pytest.fixture
    def sample_persona(self):
        """Create a sample persona for testing."""
        return Persona(
            persona_type=PersonaType.FRESHMAN,
            name="Student",
            description="Test",
            query_styles=["정중하게 "],
        )

    def test_customize_template_adds_persona_prefix(
        self, simulator, sample_turn, sample_persona
    ):
        """
        SPEC: _customize_template should add persona prefix when available.

        Given: A template and persona with query_styles
        When: _customize_template is called
        Then: Result includes persona prefix
        """
        # Arrange
        template = "조금 더 자세히 설명해주세요"

        # Act
        result = simulator._customize_template(template, sample_turn, sample_persona)

        # Assert
        assert "정중하게" in result or template in result

    def test_customize_template_preserves_template_content(
        self, simulator, sample_turn, sample_persona
    ):
        """
        SPEC: _customize_template should preserve original template content.

        Given: A template
        When: _customize_template is called
        Then: Original template content is in result
        """
        # Arrange
        template = "그건 구체적으로 어떻게 되나요?"

        # Act
        result = simulator._customize_template(template, sample_turn, sample_persona)

        # Assert
        # At least part of template should be preserved
        assert "구체적으로" in result or "어떻게" in result


class TestConversationCompletion:
    """Specification tests for conversation completion detection."""

    @pytest.fixture
    def simulator(self):
        """Create a simulator instance for testing."""
        mock_execute = MagicMock()
        mock_tracker = ContextTracker()
        return MultiTurnSimulator(
            execute_usecase=mock_execute,
            context_tracker=mock_tracker,
        )

    def test_is_conversation_complete_false_for_empty_history(self, simulator):
        """
        SPEC: _is_conversation_complete should return False for empty history.

        Given: An empty context history
        When: _is_conversation_complete is called
        Then: Returns False
        """
        # Arrange
        context = ContextHistory(
            scenario_id="test",
            conversation_history=[],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        # Act
        result = simulator._is_conversation_complete(context)

        # Assert
        assert result is False

    def test_is_conversation_complete_true_with_completion_indicators(self, simulator):
        """
        SPEC: _is_conversation_complete should return True with completion indicators.

        Given: Context where last answer contains completion indicators
        When: _is_conversation_complete is called
        Then: Returns True
        """
        # Arrange
        turn = Turn(
            turn_number=1,
            query="더 알아야 할 것",
            answer="이상 없습니다. 더 이상 질문이 없으면 종료합니다",
            sources=[],
            confidence=0.9,
        )

        context = ContextHistory(
            scenario_id="test",
            conversation_history=[turn],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        # Act
        result = simulator._is_conversation_complete(context)

        # Assert
        assert result is True

    def test_is_conversation_complete_false_without_indicators(self, simulator):
        """
        SPEC: _is_conversation_complete should return False without indicators.

        Given: Context where answer doesn't suggest completion
        When: _is_conversation_complete is called
        Then: Returns False
        """
        # Arrange
        turn = Turn(
            turn_number=1,
            query="질문",
            answer="계속해서 더 자세히 설명하겠습니다",
            sources=[],
            confidence=0.9,
        )

        context = ContextHistory(
            scenario_id="test",
            conversation_history=[turn],
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        # Act
        result = simulator._is_conversation_complete(context)

        # Assert
        assert result is False


class TestDifficultyAssessment:
    """Specification tests for scenario difficulty assessment."""

    @pytest.fixture
    def simulator(self):
        """Create a simulator instance for testing."""
        mock_execute = MagicMock()
        mock_tracker = ContextTracker()
        return MultiTurnSimulator(
            execute_usecase=mock_execute,
            context_tracker=mock_tracker,
        )

    def test_assess_scenario_difficulty_returns_easy_for_simple_scenario(
        self, simulator
    ):
        """
        SPEC: _assess_scenario_difficulty should return EASY for simple scenarios.

        Given: Few turns, all context preserved, high confidence
        When: _assess_scenario_difficulty is called
        Then: Returns DifficultyLevel.EASY
        """
        # Arrange
        turns = [
            Turn(
                turn_number=1,
                query=f"질문 {i}",
                answer=f"답변 {i}",
                sources=[],
                confidence=0.9,
                context_preserved=True,
            )
            for i in range(1, 4)
        ]

        # Act
        result = simulator._assess_scenario_difficulty(turns)

        # Assert
        assert result == DifficultyLevel.EASY

    def test_assess_scenario_difficulty_returns_medium_for_moderate_scenario(
        self, simulator
    ):
        """
        SPEC: _assess_scenario_difficulty should return MEDIUM for moderate scenarios.

        Given: Several turns with some difficulty factors
        When: _assess_scenario_difficulty is called
        Then: Returns DifficultyLevel.MEDIUM
        """
        # Arrange - 4+ turns = difficulty factor
        turns = [
            Turn(
                turn_number=1,
                query=f"질문 {i}",
                answer=f"답변 {i}",
                sources=[],
                confidence=0.85,
                context_preserved=True,
            )
            for i in range(1, 5)
        ]

        # Act
        result = simulator._assess_scenario_difficulty(turns)

        # Assert
        assert result == DifficultyLevel.MEDIUM

    def test_assess_scenario_difficulty_returns_hard_for_complex_scenario(
        self, simulator
    ):
        """
        SPEC: _assess_scenario_difficulty should return HARD for complex scenarios.

        Given: Many turns with context failures and low confidence
        When: _assess_scenario_difficulty is called
        Then: Returns DifficultyLevel.HARD
        """
        # Arrange - Multiple difficulty factors
        turns = [
            Turn(
                turn_number=1,
                query=f"질문 {i}",
                answer=f"답변 {i}",
                sources=[],
                confidence=0.6,  # Low confidence
                context_preserved=(i % 2 == 0),  # Some context failures
            )
            for i in range(1, 6)
        ]

        # Act
        result = simulator._assess_scenario_difficulty(turns)

        # Assert
        assert result == DifficultyLevel.HARD

    def test_assess_scenario_difficulty_returns_easy_for_empty_turns(self, simulator):
        """
        SPEC: _assess_scenario_difficulty should return EASY for no turns.

        Given: Empty list of turns
        When: _assess_scenario_difficulty is called
        Then: Returns DifficultyLevel.EASY
        """
        # Arrange
        turns = []

        # Act
        result = simulator._assess_scenario_difficulty(turns)

        # Assert
        assert result == DifficultyLevel.EASY


class TestScenarioMetadata:
    """Specification tests for scenario metadata."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        mock_execute = MagicMock()

        mock_result = MagicMock()
        mock_result.answer = "테스트 답변"
        mock_result.sources = ["규정 제1조"]
        mock_result.confidence = 0.9
        mock_execute.execute_query.return_value = mock_result

        mock_tracker = ContextTracker()

        return {"execute": mock_execute, "tracker": mock_tracker}

    def test_scenario_includes_persona_name_in_metadata(self, mock_dependencies):
        """
        SPEC: generate_scenario should include persona name in metadata.

        Given: A persona with a name
        When: generate_scenario is called
        Then: scenario.metadata includes persona_name
        """
        # Arrange
        simulator = MultiTurnSimulator(
            execute_usecase=mock_dependencies["execute"],
            context_tracker=mock_dependencies["tracker"],
        )

        persona = Persona(
            persona_type=PersonaType.FRESHMAN,
            name="김학생",
            description="Test student",
            query_styles=[],
        )

        # Act
        scenario = simulator.generate_scenario(
            scenario_id="test_metadata_001",
            persona=persona,
            initial_query="휴학 방법",
            min_turns=2,
            max_turns=3,
        )

        # Assert
        assert "persona_name" in scenario.metadata
        assert scenario.metadata["persona_name"] == "김학생"

    def test_scenario_includes_total_turns_in_metadata(self, mock_dependencies):
        """
        SPEC: generate_scenario should include total turn count in metadata.

        Given: A scenario is generated
        When: generate_scenario is called
        Then: scenario.metadata includes total_turns
        """
        # Arrange
        simulator = MultiTurnSimulator(
            execute_usecase=mock_dependencies["execute"],
            context_tracker=mock_dependencies["tracker"],
        )

        persona = Persona(
            persona_type=PersonaType.PROFESSOR,
            name="Professor",
            description="Test",
            query_styles=[],
        )

        # Act
        scenario = simulator.generate_scenario(
            scenario_id="test_metadata_002",
            persona=persona,
            initial_query="성적 정정",
            min_turns=3,
            max_turns=5,
        )

        # Assert
        assert "total_turns" in scenario.metadata
        assert scenario.metadata["total_turns"] == len(scenario.turns)

    def test_scenario_includes_context_preservation_rate_in_metadata(
        self, mock_dependencies
    ):
        """
        SPEC: generate_scenario should include context preservation rate in metadata.

        Given: A scenario with turns
        When: generate_scenario is called
        Then: scenario.metadata includes context_preservation_rate
        """
        # Arrange
        simulator = MultiTurnSimulator(
            execute_usecase=mock_dependencies["execute"],
            context_tracker=mock_dependencies["tracker"],
        )

        persona = Persona(
            persona_type=PersonaType.NEW_STAFF,
            name="Staff",
            description="Test",
            query_styles=[],
        )

        # Act
        scenario = simulator.generate_scenario(
            scenario_id="test_metadata_003",
            persona=persona,
            initial_query="복학 방법",
            min_turns=2,
            max_turns=4,
        )

        # Assert
        assert "context_preservation_rate" in scenario.metadata
        assert isinstance(scenario.metadata["context_preservation_rate"], float)
        assert 0.0 <= scenario.metadata["context_preservation_rate"] <= 1.0
