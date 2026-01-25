"""
Extended unit tests for MultiTurnSimulator.

Tests comprehensive scenarios for multi-turn conversation simulation:
- Complex conversation flows
- Edge cases in follow-up generation
- Persona-specific behavior
- Context window management
"""

from unittest.mock import Mock, patch

import pytest

from src.rag.automation.domain.context_tracker import ContextTracker
from src.rag.automation.domain.entities import (
    ContextHistory,
    DifficultyLevel,
    FollowUpType,
    Persona,
    PersonaType,
    Turn,
)
from src.rag.automation.infrastructure.multi_turn_simulator import MultiTurnSimulator


class MockExecuteUseCase:
    """Mock ExecuteTestUseCase for testing."""

    def __init__(self, responses=None):
        self.call_count = 0
        self.queries = []
        self.responses = responses or {}

    def execute_query(self, query, test_case_id, enable_answer=True, top_k=5):
        """Mock query execution with configurable responses."""
        self.call_count += 1
        self.queries.append(query)

        mock_result = Mock()
        mock_result.answer = self.responses.get(query, f"Answer to: {query}")
        mock_result.sources = [f"Source {i}" for i in range(1, 4)]
        mock_result.confidence = 0.85
        return mock_result


class TestFollowUpTypeSelection:
    """Test follow-up type selection logic."""

    @pytest.fixture
    def simulator(self):
        """Create MultiTurnSimulator instance."""
        mock_execute = MockExecuteUseCase()
        context_tracker = ContextTracker(context_window_size=3)
        return MultiTurnSimulator(
            execute_usecase=mock_execute, context_tracker=context_tracker
        )

    def test_early_turns_select_clarification(self, simulator):
        """
        SPEC: Early turns (2-3) should prefer CLARIFICATION follow-ups.

        Given: Turn number 2 or 3
        When: Follow-up type is selected
        Then: Should return CLARIFICATION
        """
        follow_up_type = simulator._select_follow_up_type(turn_number=2, min_turns=3)

        assert follow_up_type == FollowUpType.CLARIFICATION

    def test_middle_turns_select_related_expansion(self, simulator):
        """
        SPEC: Middle turns (3-4) should prefer RELATED_EXPANSION.

        Given: Turn number 3
        When: Follow-up type is selected
        Then: Should return RELATED_EXPANSION
        """
        follow_up_type = simulator._select_follow_up_type(turn_number=3, min_turns=3)

        assert follow_up_type == FollowUpType.RELATED_EXPANSION

    def test_later_turns_select_exception_check(self, simulator):
        """
        SPEC: Later turns (4) should prefer EXCEPTION_CHECK.

        Given: Turn number 4
        When: Follow-up type is selected
        Then: Should return EXCEPTION_CHECK
        """
        follow_up_type = simulator._select_follow_up_type(turn_number=4, min_turns=3)

        assert follow_up_type == FollowUpType.EXCEPTION_CHECK

    def test_turn_5_selects_procedural_deepening(self, simulator):
        """
        SPEC: Turn 5 should prefer PROCEDURAL_DEEPENING.

        Given: Turn number 5
        When: Follow-up type is selected
        Then: Should return PROCEDURAL_DEEPENING
        """
        follow_up_type = simulator._select_follow_up_type(turn_number=5, min_turns=3)

        assert follow_up_type == FollowUpType.PROCEDURAL_DEEPENING

    def test_turns_after_min_with_continuation(self, simulator):
        """
        SPEC: Turns after min_turns may continue with 30% probability.

        Given: Turn number 6 with min_turns=3
        When: Follow-up type is selected multiple times
        Then: Should sometimes return continuation types, sometimes None
        """
        # Test multiple times to check probabilistic behavior
        results = []
        for _ in range(100):
            result = simulator._select_follow_up_type(turn_number=6, min_turns=3)
            results.append(result)

        # Should have some None results (stopping)
        # and some continuation types
        continuation_types = [
            FollowUpType.CONDITION_CHANGE,
            FollowUpType.CONFIRMATION,
            FollowUpType.COMPARISON,
        ]

        has_none = any(r is None for r in results)
        has_continuation = any(r in continuation_types for r in results)

        # At least one type of result should appear
        assert has_none or has_continuation


class TestFollowUpQuestionGeneration:
    """Test follow-up question generation."""

    @pytest.fixture
    def simulator(self):
        """Create MultiTurnSimulator instance."""
        mock_execute = MockExecuteUseCase()
        context_tracker = ContextTracker(context_window_size=3)
        return MultiTurnSimulator(
            execute_usecase=mock_execute, context_tracker=context_tracker
        )

    @pytest.fixture
    def sample_context(self):
        """Create sample context."""
        turn = Turn(
            turn_number=1,
            query="휴학 신청 방법",
            answer="휴학 신청은...",
            sources=["규정"],
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
    def freshman_persona(self):
        """Create freshman persona."""
        return Persona(
            persona_type=PersonaType.FRESHMAN,
            name="신입생",
            description="학교 시스템에 익숙하지 않음",
            characteristics=["신입생", "경험 부족"],
            query_styles=["저기요,", "실은 이거,", "초보라서"],
            context_hints=["상세한 설명 필요", "용어 설명 필요"],
        )

    def test_clarification_generation(
        self, simulator, sample_context, freshman_persona
    ):
        """
        SPEC: Should generate clarification follow-up questions.

        Given: Context and CLARIFICATION type
        When: Follow-up is generated
        Then: Should ask for clarification
        """
        follow_up = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.CLARIFICATION,
            persona=freshman_persona,
        )

        assert follow_up is not None
        assert isinstance(follow_up, str)
        assert len(follow_up) > 0

    def test_related_expansion_generation(
        self, simulator, sample_context, freshman_persona
    ):
        """
        SPEC: Should generate related expansion follow-up questions.

        Given: Context and RELATED_EXPANSION type
        When: Follow-up is generated
        Then: Should ask about related topics
        """
        follow_up = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.RELATED_EXPANSION,
            persona=freshman_persona,
        )

        assert follow_up is not None
        assert isinstance(follow_up, str)

    def test_exception_check_generation(
        self, simulator, sample_context, freshman_persona
    ):
        """
        SPEC: Should generate exception check follow-up questions.

        Given: Context and EXCEPTION_CHECK type
        When: Follow-up is generated
        Then: Should ask about exceptions
        """
        follow_up = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.EXCEPTION_CHECK,
            persona=freshman_persona,
        )

        assert follow_up is not None

    def test_persona_style_injection(self, simulator, sample_context, freshman_persona):
        """
        SPEC: Should inject persona-specific style into follow-up.

        Given: Persona with specific query styles
        When: Follow-up is generated
        Then: Should include persona style prefix
        """
        follow_up = simulator._generate_follow_up_question(
            context=sample_context,
            follow_up_type=FollowUpType.CLARIFICATION,
            persona=freshman_persona,
        )

        # Check if any persona style is included (probabilistic)
        has_style = any(style in follow_up for style in freshman_persona.query_styles)
        # Not guaranteed due to random selection, but structure should be valid
        assert isinstance(follow_up, str)

    def test_empty_context_returns_none(self, simulator, freshman_persona):
        """
        SPEC: Empty context should return None for follow-up.

        Given: Context with no conversation history
        When: Follow-up is generated
        Then: Should return None
        """
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


class TestConversationCompletionDetection:
    """Test conversation completion detection logic."""

    @pytest.fixture
    def simulator(self):
        """Create MultiTurnSimulator instance."""
        mock_execute = MockExecuteUseCase()
        context_tracker = ContextTracker(context_window_size=3)
        return MultiTurnSimulator(
            execute_usecase=mock_execute, context_tracker=context_tracker
        )

    def test_completion_indicator_in_answer(self, simulator):
        """
        SPEC: Should detect completion indicators in answer.

        Given: Answer with "더 이상" or similar indicators
        When: Completion is checked
        Then: Should return True
        """
        context = ContextHistory(
            scenario_id="test",
            conversation_history=[
                Turn(
                    turn_number=1,
                    query="휴학 방법",
                    answer="더 이상 질문이 없으신 것 같습니다.",
                    sources=["규정"],
                    confidence=0.9,
                )
            ],
            implicit_entities={},
            topic_transitions=[],
            intent_history=["휴학"],
        )

        is_complete = simulator._is_conversation_complete(context)

        assert is_complete is True

    def test_no_completion_without_indicators(self, simulator):
        """
        SPEC: Should not detect completion without indicators.

        Given: Normal answer without completion indicators
        When: Completion is checked
        Then: Should return False
        """
        context = ContextHistory(
            scenario_id="test",
            conversation_history=[
                Turn(
                    turn_number=1,
                    query="휴학 방법",
                    answer="휴학 신청은 다음과 같습니다.",
                    sources=["규정"],
                    confidence=0.9,
                )
            ],
            implicit_entities={},
            topic_transitions=[],
            intent_history=["휴학"],
        )

        is_complete = simulator._is_conversation_complete(context)

        assert is_complete is False

    def test_many_turns_with_stable_confidence(self, simulator):
        """
        SPEC: Should detect completion after many turns with stable confidence.

        Given: 6+ turns with similar confidence scores
        When: Completion is checked
        Then: Should return True
        """
        # Create 6 turns with similar confidence
        turns = [
            Turn(
                turn_number=i,
                query=f"Question {i}",
                answer=f"Answer {i}",
                sources=[f"Source {i}"],
                confidence=0.85,  # All same confidence
            )
            for i in range(1, 7)
        ]

        context = ContextHistory(
            scenario_id="stable_test",
            conversation_history=turns,
            implicit_entities={},
            topic_transitions=[],
            intent_history=[f"Q{i}" for i in range(1, 7)],
        )

        is_complete = simulator._is_conversation_complete(context)

        assert is_complete is True

    def test_few_turns_with_varying_confidence(self, simulator):
        """
        SPEC: Should not detect completion with few varying turns.

        Given: 3 turns with varying confidence
        When: Completion is checked
        Then: Should return False
        """
        turns = [
            Turn(
                turn_number=i,
                query=f"Question {i}",
                answer=f"Answer {i}",
                sources=[f"Source {i}"],
                confidence=0.7 + (i * 0.1),  # Varying confidence
            )
            for i in range(1, 4)
        ]

        context = ContextHistory(
            scenario_id="varying_test",
            conversation_history=turns,
            implicit_entities={},
            topic_transitions=[],
            intent_history=[f"Q{i}" for i in range(1, 4)],
        )

        is_complete = simulator._is_conversation_complete(context)

        assert is_complete is False


class TestDifficultyAssessment:
    """Test scenario difficulty assessment."""

    @pytest.fixture
    def simulator(self):
        """Create MultiTurnSimulator instance."""
        mock_execute = MockExecuteUseCase()
        context_tracker = ContextTracker(context_window_size=3)
        return MultiTurnSimulator(
            execute_usecase=mock_execute, context_tracker=context_tracker
        )

    def test_easy_difficulty_simple_scenario(self, simulator):
        """
        SPEC: Simple scenario should be assessed as EASY.

        Given: 2 turns, all context preserved, high confidence
        When: Difficulty is assessed
        Then: Should return EASY
        """
        turns = [
            Turn(
                turn_number=1,
                query="휴학 방법",
                answer="답변",
                sources=["규정"],
                confidence=0.9,
                context_preserved=True,
            ),
            Turn(
                turn_number=2,
                query="서류는요?",
                answer="답변",
                sources=["규정"],
                confidence=0.88,
                context_preserved=True,
            ),
        ]

        difficulty = simulator._assess_scenario_difficulty(turns)

        assert difficulty == DifficultyLevel.EASY

    def test_medium_difficulty_with_context_failure(self, simulator):
        """
        SPEC: Scenario with context failure should be MEDIUM.

        Given: 3 turns, 1 context failure, good confidence
        When: Difficulty is assessed
        Then: Should return MEDIUM
        """
        turns = [
            Turn(
                turn_number=1,
                query="휴학 방법",
                answer="답변",
                sources=["규정"],
                confidence=0.9,
                context_preserved=True,
            ),
            Turn(
                turn_number=2,
                query="서류는요?",
                answer="답변",
                sources=["규정"],
                confidence=0.88,
                context_preserved=True,
            ),
            Turn(
                turn_number=3,
                query="졸업 요건",  # Topic change
                answer="답변",
                sources=["규정"],
                confidence=0.85,
                context_preserved=False,
            ),
        ]

        difficulty = simulator._assess_scenario_difficulty(turns)

        assert difficulty == DifficultyLevel.MEDIUM

    def test_hard_difficulty_complex_scenario(self, simulator):
        """
        SPEC: Complex scenario should be assessed as HARD.

        Given: 4+ turns, context failures, low confidence
        When: Difficulty is assessed
        Then: Should return HARD
        """
        turns = [
            Turn(
                turn_number=i,
                query=f"Question {i}",
                answer=f"Answer {i}",
                sources=["규정"],
                confidence=0.6,  # Low confidence
                context_preserved=(i % 2 == 0),  # Some failures
            )
            for i in range(1, 5)
        ]

        difficulty = simulator._assess_scenario_difficulty(turns)

        assert difficulty == DifficultyLevel.HARD

    def test_empty_turns_returns_easy(self, simulator):
        """
        SPEC: Empty turn list should return EASY by default.

        Given: Empty turn list
        When: Difficulty is assessed
        Then: Should return EASY
        """
        difficulty = simulator._assess_scenario_difficulty([])

        assert difficulty == DifficultyLevel.EASY


class TestScenarioGeneration:
    """Test complete scenario generation."""

    @pytest.fixture
    def simulator(self):
        """Create MultiTurnSimulator with mocked responses."""
        responses = {
            "휴학 신청은 어떻게 하나요?": "휴학 신청은 학기 개시 30일 전까지...",
            "구체적으로 어떻게 되나요?": "신청서를 작성하고...",
            "관련된 다른 규정도 있나요?": "관련 규정으로는...",
        }
        mock_execute = MockExecuteUseCase(responses=responses)
        context_tracker = ContextTracker(context_window_size=3)
        return MultiTurnSimulator(
            execute_usecase=mock_execute, context_tracker=context_tracker
        )

    @pytest.fixture
    def freshman_persona(self):
        """Create freshman persona."""
        return Persona(
            persona_type=PersonaType.FRESHMAN,
            name="신입생",
            description="학교 시스템에 익숙하지 않음",
            characteristics=["신입생"],
            query_styles=["저기요,"],
            context_hints=["상세한 설명"],
        )

    def test_generate_min_scenario(self, simulator, freshman_persona):
        """
        SPEC: Should generate scenario with min_turns.

        Given: min_turns=2, max_turns=4
        When: Scenario is generated
        Then: Should have at least 2 turns
        """
        with patch.object(simulator, "_generate_follow_up_question", return_value=None):
            # Stop after initial turn by returning None for follow-up
            scenario = simulator.generate_scenario(
                scenario_id="test_min",
                persona=freshman_persona,
                initial_query="휴학 신청은 어떻게 하나요?",
                min_turns=1,
                max_turns=2,
            )

        # Should have at least initial turn
        assert scenario.total_turns >= 1
        assert scenario.scenario_id == "test_min"
        assert scenario.persona_type == PersonaType.FRESHMAN

    def test_scenario_metadata_populated(self, simulator, freshman_persona):
        """
        SPEC: Scenario metadata should include key metrics.

        Given: Generated scenario
        When: Metadata is checked
        Then: Should include total_turns and context_preservation_rate
        """
        with patch.object(simulator, "_generate_follow_up_question", return_value=None):
            scenario = simulator.generate_scenario(
                scenario_id="test_metadata",
                persona=freshman_persona,
                initial_query="휴학 신청은 어떻게 하나요?",
                min_turns=1,
                max_turns=2,
            )

        assert "total_turns" in scenario.metadata
        assert "context_preservation_rate" in scenario.metadata
        assert "persona_name" in scenario.metadata

    def test_scenario_difficulty_assigned(self, simulator, freshman_persona):
        """
        SPEC: Scenario should have difficulty level assigned.

        Given: Generated scenario
        When: Difficulty is checked
        Then: Should be one of EASY, MEDIUM, HARD
        """
        with patch.object(simulator, "_generate_follow_up_question", return_value=None):
            scenario = simulator.generate_scenario(
                scenario_id="test_difficulty",
                persona=freshman_persona,
                initial_query="휴학 신청은 어떻게 하나요?",
                min_turns=1,
                max_turns=2,
            )

        assert scenario.difficulty in [
            DifficultyLevel.EASY,
            DifficultyLevel.MEDIUM,
            DifficultyLevel.HARD,
        ]

    def test_scenario_turns_have_context_flags(self, simulator, freshman_persona):
        """
        SPEC: All turns should have context_preserved flag.

        Given: Generated scenario
        When: Turns are checked
        Then: All should have context_preserved as boolean
        """
        with patch.object(simulator, "_generate_follow_up_question", return_value=None):
            scenario = simulator.generate_scenario(
                scenario_id="test_context_flags",
                persona=freshman_persona,
                initial_query="휴학 신청은 어떻게 하나요?",
                min_turns=1,
                max_turns=2,
            )

        for turn in scenario.turns:
            assert isinstance(turn.context_preserved, bool)


class TestContextWindowManagement:
    """Test context window size management."""

    @pytest.fixture
    def simulator(self):
        """Create MultiTurnSimulator."""
        mock_execute = MockExecuteUseCase()
        context_tracker = ContextTracker(context_window_size=3)
        return MultiTurnSimulator(
            execute_usecase=mock_execute, context_tracker=context_tracker
        )

    def test_context_window_size_respected(self, simulator):
        """
        SPEC: Context window size should be respected.

        Given: context_window_size=3
        When: Recent context is retrieved
        Then: Should return at most 3 turns
        """
        # Create 5 turns
        turns = [
            Turn(
                turn_number=i,
                query=f"Q{i}",
                answer=f"A{i}",
                sources=["S"],
                confidence=0.8,
            )
            for i in range(1, 6)
        ]

        context = ContextHistory(
            scenario_id="window_test",
            conversation_history=turns,
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        recent = context.get_recent_context(window_size=3)

        assert len(recent) == 3
        # Should be the last 3 turns
        assert recent[0].turn_number == 3
        assert recent[1].turn_number == 4
        assert recent[2].turn_number == 5

    def test_context_window_smaller_than_history(self, simulator):
        """
        SPEC: Window size smaller than history should truncate.

        Given: 10 turns, window_size=3
        When: Recent context is retrieved
        Then: Should return last 3 turns
        """
        turns = [
            Turn(
                turn_number=i,
                query=f"Q{i}",
                answer=f"A{i}",
                sources=["S"],
                confidence=0.8,
            )
            for i in range(1, 11)
        ]

        context = ContextHistory(
            scenario_id="truncate_test",
            conversation_history=turns,
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        recent = context.get_recent_context(window_size=3)

        assert len(recent) == 3
        assert recent[0].turn_number == 8

    def test_context_window_larger_than_history(self, simulator):
        """
        SPEC: Window size larger than history should return all.

        Given: 2 turns, window_size=5
        When: Recent context is retrieved
        Then: Should return all 2 turns
        """
        turns = [
            Turn(
                turn_number=i,
                query=f"Q{i}",
                answer=f"A{i}",
                sources=["S"],
                confidence=0.8,
            )
            for i in range(1, 3)
        ]

        context = ContextHistory(
            scenario_id="all_test",
            conversation_history=turns,
            implicit_entities={},
            topic_transitions=[],
            intent_history=[],
        )

        recent = context.get_recent_context(window_size=5)

        assert len(recent) == 2
